"""SQLite storage for the index.

One database file holds file metadata, frames, transcript segments (with an
FTS5 full-text index), scenes and transcript chunks. Embeddings are stored
as float16 BLOBs next to their rows.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
STAGES = ("frames", "visual", "speech", "speech_emb", "scenes")

# files.frames_rev / scenes_rev / chunks_rev: revision of the last change to the file's vectors,
# for the search cache (vectors.py). No SQL comments inside CREATE TABLE: ALTER TABLE rewrites it.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS files(
    id INTEGER PRIMARY KEY,
    relpath TEXT UNIQUE NOT NULL,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    fingerprint TEXT,
    duration REAL,
    width INTEGER,
    height INTEGER,
    fps REAL,
    vcodec TEXT,
    has_audio INTEGER,
    language TEXT,
    speech_seconds REAL,
    frames_done INTEGER NOT NULL DEFAULT 0,
    visual_done INTEGER NOT NULL DEFAULT 0,
    speech_done INTEGER NOT NULL DEFAULT 0,
    speech_emb_done INTEGER NOT NULL DEFAULT 0,
    scenes_done INTEGER NOT NULL DEFAULT 0,
    missing INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    updated_at REAL,
    frames_rev INTEGER NOT NULL DEFAULT 0,
    scenes_rev INTEGER NOT NULL DEFAULT 0,
    chunks_rev INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS frames(
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL,
    idx INTEGER NOT NULL,
    t REAL NOT NULL,
    image TEXT NOT NULL,
    emb BLOB
);
CREATE INDEX IF NOT EXISTS frames_by_file ON frames(file_id, t);

CREATE TABLE IF NOT EXISTS segments(
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL,
    start REAL NOT NULL,
    end REAL NOT NULL,
    text TEXT NOT NULL,
    words TEXT
);
CREATE INDEX IF NOT EXISTS segments_by_file ON segments(file_id, start);

CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts USING fts5(
    text, content='segments', content_rowid='id', tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS segments_ai AFTER INSERT ON segments BEGIN
    INSERT INTO segments_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS segments_ad AFTER DELETE ON segments BEGIN
    INSERT INTO segments_fts(segments_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

-- Word stems and folded text of each segment (rowid = segments.id), filled by Store.add_segments.
CREATE VIRTUAL TABLE IF NOT EXISTS segments_stem USING fts5(text, tokenize='unicode61 remove_diacritics 2');

CREATE TABLE IF NOT EXISTS speech_chunks(
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL,
    start REAL NOT NULL,
    end REAL NOT NULL,
    text TEXT NOT NULL,
    emb BLOB
);
CREATE INDEX IF NOT EXISTS chunks_by_file ON speech_chunks(file_id, start);

CREATE TABLE IF NOT EXISTS scenes(
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL,
    start REAL NOT NULL,
    end REAL NOT NULL,
    frame_ids TEXT NOT NULL,
    emb BLOB
);
CREATE INDEX IF NOT EXISTS scenes_by_file ON scenes(file_id, start);
"""


def to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float16).tobytes()


def from_blobs(blobs: Sequence[bytes], dim: int) -> np.ndarray:
    if not blobs:
        return np.zeros((0, dim), dtype=np.float16)
    return np.frombuffer(b"".join(blobs), dtype=np.float16).reshape(len(blobs), dim)


@dataclass
class FileRow:
    id: int
    relpath: str
    size: int
    mtime: float
    fingerprint: str | None
    duration: float | None
    width: int | None
    height: int | None
    fps: float | None
    vcodec: str | None
    has_audio: int | None
    language: str | None
    speech_seconds: float | None
    frames_done: int
    visual_done: int
    speech_done: int
    speech_emb_done: int
    scenes_done: int
    missing: int
    error: str | None
    updated_at: float | None

    def done(self, stage: str) -> bool:
        return bool(getattr(self, f"{stage}_done"))


_FILE_COLUMNS = [f for f in FileRow.__dataclass_fields__]
# Vector tables and the files column that stamps their last change.
_REV_COLUMNS = {"frames": "frames_rev", "scenes": "scenes_rev", "speech_chunks": "chunks_rev"}


class Store:
    """Thin wrapper around one SQLite connection. Safe to share between threads."""

    def __init__(self, db_path: Path, readonly: bool = False):
        self.path = db_path
        self.readonly = readonly
        if readonly:
            if not db_path.exists():
                raise FileNotFoundError(f"No index found at {db_path}")
            # as_uri() percent-encodes '#', '?', '%' and spaces in folder names.
            uri = db_path.resolve().as_uri() + "?mode=ro"
            self.conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=30)
        else:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self._touched: dict[str, set[int]] = {t: set() for t in _REV_COLUMNS}
        if not readonly:
            with self.lock:
                self.conn.execute("PRAGMA journal_mode=WAL")
                self.conn.execute("PRAGMA synchronous=NORMAL")
                self.conn.executescript(_SCHEMA)
                cols = {r[1] for r in self.conn.execute("PRAGMA table_info(files)")}
                for col in _REV_COLUMNS.values():  # indexes from before the search cache
                    if col not in cols:
                        self.conn.execute(f"ALTER TABLE files ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0")
                if self.get_meta("schema_version") is None:
                    self.set_meta("schema_version", str(SCHEMA_VERSION))
                if self.get_meta("index_id") is None:
                    self.set_meta("index_id", uuid.uuid4().hex)
                self._extra_fts()
                self.conn.commit()
        version = self.get_meta("schema_version")
        if version is not None and int(version) != SCHEMA_VERSION:
            raise RuntimeError(f"Index schema version {version} is not supported (need {SCHEMA_VERSION}).")

    # --- extra transcript indexes (stems, trigrams) ------------------------------------
    def _extra_fts(self) -> None:
        """Create the trigram index if SQLite supports it, and fill both extra indexes once."""
        c = self.conn
        try:
            c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS segments_tri USING fts5(text, tokenize='trigram')")
            self.has_trigram = True
        except sqlite3.OperationalError:  # SQLite older than 3.34
            self.has_trigram = False
        if self.get_meta("fts_extra") == "1":
            return
        c.execute("DELETE FROM segments_stem")
        if self.has_trigram:
            c.execute("DELETE FROM segments_tri")
        rows = c.execute("SELECT s.id, s.text, f.language FROM segments s JOIN files f ON f.id = s.file_id").fetchall()
        self._index_text([(r[0], r[1], r[2]) for r in rows])
        self.set_meta("fts_extra", "1")

    def _index_text(self, items: Sequence[tuple[int, str, str | None]]) -> None:
        from . import fts

        self.conn.executemany("INSERT INTO segments_stem(rowid, text) VALUES(?, ?)",
                              [(sid, fts.stem_text(text, lang)) for sid, text, lang in items])
        if getattr(self, "has_trigram", self._table_exists("segments_tri")):
            self.conn.executemany("INSERT INTO segments_tri(rowid, text) VALUES(?, ?)",
                                  [(sid, fts.trigram_text(text)) for sid, text, _ in items])

    def _table_exists(self, name: str) -> bool:
        return self.conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (name,)).fetchone() is not None

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    # --- meta -----------------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        with self.lock:
            row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def revision(self) -> int:
        value = self.get_meta("revision")
        return int(value) if value else 0

    def bump_revision(self) -> None:
        """Count a change. Files whose vectors changed get the new revision as their stamp."""
        with self.lock:
            rev = self.revision() + 1
            self.set_meta("revision", str(rev))
            for table, ids in self._touched.items():
                if ids:
                    ids_list = sorted(ids)
                    for i in range(0, len(ids_list), 500):
                        part = ids_list[i:i + 500]
                        self.conn.execute(
                            f"UPDATE files SET {_REV_COLUMNS[table]}=? WHERE id IN ({','.join('?' * len(part))})",
                            (rev, *part))
                    ids.clear()

    def _touch(self, table: str, file_id: int) -> None:
        self._touched[table].add(int(file_id))

    def commit(self) -> None:
        with self.lock:
            self.conn.commit()

    def rollback(self) -> None:
        with self.lock:
            self.conn.rollback()
            for ids in self._touched.values():
                ids.clear()

    # --- files ----------------------------------------------------------------
    def _file(self, row: sqlite3.Row | None) -> FileRow | None:
        if row is None:
            return None
        return FileRow(**{k: row[k] for k in _FILE_COLUMNS})

    def files(self, include_missing: bool = True) -> list[FileRow]:
        sql = "SELECT * FROM files"
        if not include_missing:
            sql += " WHERE missing=0"
        with self.lock:
            rows = self.conn.execute(sql + " ORDER BY relpath").fetchall()
        return [self._file(r) for r in rows]

    def file_by_id(self, file_id: int) -> FileRow | None:
        with self.lock:
            return self._file(self.conn.execute("SELECT * FROM files WHERE id=?", (file_id,)).fetchone())

    def file_by_path(self, relpath: str) -> FileRow | None:
        with self.lock:
            return self._file(self.conn.execute("SELECT * FROM files WHERE relpath=?", (relpath,)).fetchone())

    def add_file(self, relpath: str, size: int, mtime: float, fingerprint: str | None) -> int:
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO files(relpath, size, mtime, fingerprint, updated_at) VALUES(?,?,?,?,?)",
                (relpath, size, mtime, fingerprint, time.time()))
            return int(cur.lastrowid)

    def update_file(self, file_id: int, **values) -> None:
        if not values:
            return
        cols = ", ".join(f"{k}=?" for k in values)
        with self.lock:
            self.conn.execute(
                f"UPDATE files SET {cols}, updated_at=? WHERE id=?",
                (*values.values(), time.time(), file_id))

    def delete_file(self, file_id: int) -> None:
        with self.lock:
            for stage in STAGES:
                self._clear(file_id, stage)
            self.conn.execute("DELETE FROM files WHERE id=?", (file_id,))

    # --- stage data -------------------------------------------------------------
    def _clear(self, file_id: int, stage: str) -> None:
        c = self.conn
        if stage == "frames":
            c.execute("DELETE FROM frames WHERE file_id=?", (file_id,))
            self._touch("frames", file_id)
        elif stage == "visual":
            c.execute("UPDATE frames SET emb=NULL WHERE file_id=?", (file_id,))
            self._touch("frames", file_id)
        elif stage == "speech":
            ids = "(SELECT id FROM segments WHERE file_id=?)"
            c.execute(f"DELETE FROM segments_stem WHERE rowid IN {ids}", (file_id,))
            if self._table_exists("segments_tri"):
                c.execute(f"DELETE FROM segments_tri WHERE rowid IN {ids}", (file_id,))
            c.execute("DELETE FROM segments WHERE file_id=?", (file_id,))
            c.execute("DELETE FROM speech_chunks WHERE file_id=?", (file_id,))
            self._touch("speech_chunks", file_id)
        elif stage == "speech_emb":
            c.execute("UPDATE speech_chunks SET emb=NULL WHERE file_id=?", (file_id,))
            self._touch("speech_chunks", file_id)
        elif stage == "scenes":
            c.execute("DELETE FROM scenes WHERE file_id=?", (file_id,))
            self._touch("scenes", file_id)
        else:
            raise ValueError(stage)

    def reset_stages(self, file_id: int, stages: Iterable[str]) -> None:
        """Delete data for the given stages and every stage that depends on them."""
        depends = {
            "frames": {"frames", "visual", "scenes"},
            "visual": {"visual", "scenes"},
            "speech": {"speech", "speech_emb"},
            "speech_emb": {"speech_emb"},
            "scenes": {"scenes"},
        }
        todo: set[str] = set()
        for stage in stages:
            todo |= depends[stage]
        with self.lock:
            for stage in STAGES:  # keep a stable order
                if stage in todo:
                    self._clear(file_id, stage)
            self.update_file(file_id, **{f"{s}_done": 0 for s in todo})

    def add_frames(self, file_id: int, frames: Sequence[tuple[int, float, str]]) -> None:
        with self.lock:
            self.conn.executemany(
                "INSERT INTO frames(file_id, idx, t, image) VALUES(?,?,?,?)",
                [(file_id, idx, t, image) for idx, t, image in frames])

    def frames_for_file(self, file_id: int) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(
                "SELECT id, idx, t, image, emb FROM frames WHERE file_id=? ORDER BY t, idx",
                (file_id,)).fetchall()

    def set_frame_embeddings(self, items: Sequence[tuple[int, np.ndarray]]) -> None:
        with self.lock:
            self.conn.executemany(
                "UPDATE frames SET emb=? WHERE id=?", [(to_blob(v), fid) for fid, v in items])
            self._touch_rows("frames", [fid for fid, _ in items])

    def _touch_rows(self, table: str, row_ids: Sequence[int]) -> None:
        for i in range(0, len(row_ids), 500):
            part = list(row_ids[i:i + 500])
            for r in self.conn.execute(
                    f"SELECT DISTINCT file_id FROM {table} WHERE id IN ({','.join('?' * len(part))})", part):
                self._touch(table, r[0])

    def add_segments(self, file_id: int, segments: Sequence[dict], language: str | None = None) -> None:
        with self.lock:
            items = []
            for s in segments:
                cur = self.conn.execute(
                    "INSERT INTO segments(file_id, start, end, text, words) VALUES(?,?,?,?,?)",
                    (file_id, s["start"], s["end"], s["text"], json.dumps(s.get("words") or [])))
                items.append((int(cur.lastrowid), s["text"], language))
            self._index_text(items)

    def add_chunks(self, file_id: int, chunks: Sequence[tuple[float, float, str]]) -> None:
        with self.lock:
            self.conn.executemany(
                "INSERT INTO speech_chunks(file_id, start, end, text) VALUES(?,?,?,?)",
                [(file_id, s, e, t) for s, e, t in chunks])

    def chunks_for_file(self, file_id: int) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(
                "SELECT id, start, end, text, emb FROM speech_chunks WHERE file_id=? ORDER BY start",
                (file_id,)).fetchall()

    def set_chunk_embeddings(self, items: Sequence[tuple[int, np.ndarray]]) -> None:
        with self.lock:
            self.conn.executemany(
                "UPDATE speech_chunks SET emb=? WHERE id=?", [(to_blob(v), cid) for cid, v in items])
            self._touch_rows("speech_chunks", [cid for cid, _ in items])

    def add_scene(self, file_id: int, start: float, end: float, frame_ids: Sequence[int],
                  emb: np.ndarray) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT INTO scenes(file_id, start, end, frame_ids, emb) VALUES(?,?,?,?,?)",
                (file_id, start, end, json.dumps(list(frame_ids)), to_blob(emb)))
            self._touch("scenes", file_id)

    # --- read helpers for search ---------------------------------------------------
    def query(self, sql: str, params: Sequence = ()) -> list[sqlite3.Row]:
        with self.lock:
            return self.conn.execute(sql, params).fetchall()

    def stats(self) -> dict:
        with self.lock:
            c = self.conn
            files = c.execute(
                "SELECT COUNT(*), COALESCE(SUM(duration),0), "
                "SUM(missing), SUM(error IS NOT NULL AND missing=0), "
                "SUM(frames_done), SUM(visual_done), SUM(speech_done), "
                "SUM(speech_emb_done), SUM(scenes_done), COALESCE(SUM(speech_seconds),0) "
                "FROM files").fetchone()
            counts = {
                "frames": c.execute("SELECT COUNT(*) FROM frames").fetchone()[0],
                "segments": c.execute("SELECT COUNT(*) FROM segments").fetchone()[0],
                "speech_chunks": c.execute("SELECT COUNT(*) FROM speech_chunks").fetchone()[0],
                "scenes": c.execute("SELECT COUNT(*) FROM scenes").fetchone()[0],
            }
        keys = ["files", "duration_seconds", "missing", "failed", "frames_done", "visual_done",
                "speech_done", "speech_emb_done", "scenes_done", "speech_seconds"]
        out = {k: (v or 0) for k, v in zip(keys, files, strict=True)}
        out.update(counts)
        return out
