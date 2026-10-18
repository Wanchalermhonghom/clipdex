"""Memory-mapped vector cache for the search engine.

SQLite stays the source of truth. For each kind of vector ("frames", "scenes", "chunks") the cache
keeps two files in <index_dir>/vectors/:

  <kind>.f16   float16 matrix, one row per vector
  <kind>.rows  one record per vector: row id, file id and two times (t, or start and end)

The search engine maps these files instead of loading every vector into RAM. macOS and Linux keep
the pages that searches use in the file cache and drop them when other programs need memory, so the
memory of the search process stays flat while the index grows.

Updates are incremental. The indexer stamps every file whose vectors change with the revision of
that change (files.frames_rev, scenes_rev, chunks_rev). On sync the cache marks the rows of changed
or deleted files as dead (file id -1) and appends their current rows. It rebuilds from scratch after
a model change, for an index without the stamps, and when more than a quarter of the rows are dead.

When the index folder is read-only, the same data is kept in RAM instead.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .store import Store

log = logging.getLogger(__name__)

FORMAT = 1
ROW = np.dtype([("id", "<i8"), ("file", "<i8"), ("a", "<f8"), ("b", "<f8")])

# kind: (table, revision column in files, time columns)
KINDS = {
    "frames": ("frames", "frames_rev", ("t", "t")),
    "scenes": ("scenes", "scenes_rev", ("start", "end")),
    "chunks": ("speech_chunks", "chunks_rev", ("start", "end")),
}
_MODEL_KEYS = ("model.siglip", "model.qwen")
_BATCH = 20_000


class _Rebuild(Exception):
    """The incremental update cannot continue; rebuild from scratch."""


@dataclass
class VectorTable:
    """Vectors of one kind with their rows. Dead rows have file id -1."""

    rows: np.ndarray  # dtype ROW
    emb: np.ndarray  # float16, shape (n, dim)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def ids(self) -> np.ndarray:
        return self.rows["id"]

    @property
    def file_ids(self) -> np.ndarray:
        return self.rows["file"]

    @property
    def a(self) -> np.ndarray:  # frame time, or scene / chunk start
        return self.rows["a"]

    @property
    def b(self) -> np.ndarray:  # frame time, or scene / chunk end
        return self.rows["b"]


def _empty(dim: int = 0) -> VectorTable:
    return VectorTable(np.zeros(0, ROW), np.zeros((0, dim), np.float16))


class _FileLock:
    def __init__(self, path: Path):
        self.path = path
        self.fh = None

    def __enter__(self):
        self.fh = open(self.path, "a+")
        if sys.platform == "win32":
            import msvcrt

            self.fh.seek(0)
            msvcrt.locking(self.fh.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(self.fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if self.fh:
            self.fh.close()


class VectorCache:
    def __init__(self, index_dir: Path, store: Store):
        self.dir = index_dir / "vectors"
        self.store = store
        self.tables: dict[str, VectorTable] = {k: _empty() for k in KINDS}
        self.in_memory = not self._writable()
        self._mem_manifest: dict = {}
        self._opened: tuple | None = None
        self.last_sync: str = ""  # "none", "update" or "rebuild", for logs and tests

    # --- public ------------------------------------------------------------------------------
    def sync(self) -> None:
        """Bring the cache up to date with the database and map the current files."""
        with self.store.lock:
            if self.in_memory:
                self._sync_in_snapshot()
                return
            with _FileLock(self.dir / ".lock"):
                self._sync_in_snapshot()
        self._open()

    # --- helpers -----------------------------------------------------------------------------
    def _writable(self) -> bool:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            with open(self.dir / ".lock", "a"):
                pass
            return True
        except OSError:
            log.warning("The index folder is read-only: search keeps all vectors in memory.")
            return False

    def _sync_in_snapshot(self) -> None:
        conn = self.store.conn
        if conn.in_transaction:  # the indexer's own connection, inside its transaction
            self._sync()
            return
        conn.execute("BEGIN")  # one consistent snapshot of the database (WAL)
        try:
            self._sync()
        finally:
            conn.execute("COMMIT")

    def _manifest(self) -> dict:
        if self.in_memory:
            return self._mem_manifest
        try:
            return json.loads((self.dir / "manifest.json").read_text())
        except (OSError, ValueError):
            return {}

    def _write_manifest(self, man: dict) -> None:
        if self.in_memory:
            self._mem_manifest = man
            return
        tmp = self.dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(man, indent=1))
        os.replace(tmp, self.dir / "manifest.json")

    def _db_state(self) -> dict:
        c = self.store.conn
        cols = {r[1] for r in c.execute("PRAGMA table_info(files)")}
        return {
            "revision": int((c.execute("SELECT value FROM meta WHERE key='revision'").fetchone() or [0])[0]),
            "index_id": (c.execute("SELECT value FROM meta WHERE key='index_id'").fetchone() or [None])[0],
            "models": {k: (c.execute("SELECT value FROM meta WHERE key=?", (k,)).fetchone() or [None])[0]
                       for k in _MODEL_KEYS},
            "stamped": all(col in cols for _, col, _ in KINDS.values()),
        }

    def _sync(self) -> None:
        db = self._db_state()
        man = self._manifest()
        same_index = (man.get("format") == FORMAT and man.get("index_id") == db["index_id"]
                      and man.get("models") == db["models"])
        if same_index and man.get("revision") == db["revision"]:
            self.last_sync = "none"
            return
        rebuild = (not same_index or not db["stamped"] or man.get("revision", -1) > db["revision"]
                   or any(t["dead"] > max(10_000, t["rows"] // 4) for t in man.get("tables", {}).values()))
        if not rebuild:
            try:
                self._update(man, db)
                self.last_sync = "update"
                return
            except _Rebuild:
                pass
        self._rebuild(man, db)
        self.last_sync = "rebuild"

    def _path(self, kind: str, ext: str, tmp: bool = False) -> Path:
        return self.dir / f"{kind}.{ext}{'.tmp' if tmp else ''}"

    @staticmethod
    def _select(table: str, ca: str, cb: str, where: str = "") -> str:
        return (f"SELECT id, file_id, {ca}, {cb}, emb FROM {table} WHERE emb IS NOT NULL {where} "
                f"ORDER BY file_id, id")

    def _stream(self, sql: str, params=()):
        cur = self.store.conn.execute(sql, params)
        while True:
            batch = cur.fetchmany(_BATCH)
            if not batch:
                return
            yield batch

    @staticmethod
    def _decode(batch, dim: int) -> tuple[np.ndarray, np.ndarray]:
        rows = np.empty(len(batch), ROW)
        rows["id"] = [r[0] for r in batch]
        rows["file"] = [r[1] for r in batch]
        rows["a"] = [r[2] for r in batch]
        rows["b"] = [r[3] for r in batch]
        blobs = [r[4] for r in batch]
        if any(len(b) != 2 * dim for b in blobs):
            raise _Rebuild("vector size changed")
        emb = np.frombuffer(b"".join(blobs), dtype=np.float16).reshape(len(batch), dim)
        return rows, emb

    def _rebuild(self, man: dict, db: dict) -> None:
        """Write all vectors from the database into new files, then swap them in."""
        c = self.store.conn
        tables: dict[str, dict] = {}
        built: dict[str, VectorTable] = {}
        for kind, (table, _, (ca, cb)) in KINDS.items():
            first = c.execute(f"SELECT emb FROM {table} WHERE emb IS NOT NULL LIMIT 1").fetchone()
            dim = len(first[0]) // 2 if first else 0
            n = c.execute(f"SELECT COUNT(*) FROM {table} WHERE emb IS NOT NULL").fetchone()[0] if first else 0
            if self.in_memory:
                rows, emb = np.empty(n, ROW), np.empty((n, dim), np.float16)
            else:
                rows = self._create(self._path(kind, "rows", tmp=True), ROW, (n,))
                emb = self._create(self._path(kind, "f16", tmp=True), np.float16, (n, dim))
            k = 0
            for batch in self._stream(self._select(table, ca, cb)):
                r, e = self._decode(batch, dim)
                m = min(len(r), n - k)  # rows added after COUNT cannot appear inside the snapshot
                rows[k:k + m], emb[k:k + m] = r[:m], e[:m]
                k += m
            tables[kind] = {"rows": k, "dim": dim, "dead": 0}
            if self.in_memory:
                built[kind] = VectorTable(rows[:k], emb[:k])
            else:
                for arr in (rows, emb):
                    if isinstance(arr, np.memmap):
                        arr.flush()
                del rows, emb  # unmap before the files are renamed
        if self.in_memory:
            self.tables = built
        else:
            for kind in KINDS:
                for ext in ("rows", "f16"):
                    tmp = self._path(kind, ext, tmp=True)
                    if tmp.exists():
                        os.replace(tmp, self._path(kind, ext))
                    else:
                        self._path(kind, ext).unlink(missing_ok=True)
        self._write_manifest({"format": FORMAT, "index_id": db["index_id"], "models": db["models"],
                              "revision": db["revision"], "generation": uuid.uuid4().hex, "tables": tables})

    @staticmethod
    def _create(path: Path, dtype, shape) -> np.ndarray:
        if int(np.prod(shape)) == 0:
            path.write_bytes(b"")
            return np.zeros(shape, dtype)
        return np.memmap(path, dtype=dtype, mode="w+", shape=shape)

    def _update(self, man: dict, db: dict) -> None:
        """Drop the rows of changed and deleted files, then append their current rows."""
        c = self.store.conn
        last = int(man["revision"])
        existing = {r[0] for r in c.execute("SELECT id FROM files")}
        for kind, (table, col, (ca, cb)) in KINDS.items():
            info = man["tables"][kind]
            changed = [r[0] for r in c.execute(f"SELECT id FROM files WHERE {col} > ?", (last,))]
            rows_t, emb_t = self._load_for_update(kind, info)
            cached = np.unique(rows_t["file"]) if len(rows_t) else np.zeros(0, np.int64)
            deleted = [int(f) for f in cached if f >= 0 and int(f) not in existing]
            drop = np.asarray(sorted(set(changed) | set(deleted)), dtype=np.int64)
            if len(drop) and len(rows_t):
                hit = np.isin(rows_t["file"], drop)
                count = int(hit.sum())
                if count:
                    rows_t["file"][hit] = -1
                    info["dead"] += count
                    if isinstance(rows_t, np.memmap):
                        rows_t.flush()
            new_rows, new_emb = [], []
            for start in range(0, len(changed), 500):
                part = changed[start:start + 500]
                marks = ",".join("?" * len(part))
                for batch in self._stream(self._select(table, ca, cb, f"AND file_id IN ({marks})"), part):
                    dim = info["dim"] or len(batch[0][4]) // 2
                    r, e = self._decode(batch, dim)
                    info["dim"] = dim
                    new_rows.append(r)
                    new_emb.append(e)
            if new_rows:
                r = np.concatenate(new_rows)
                e = np.concatenate(new_emb)
                if self.in_memory:
                    self.tables[kind] = VectorTable(np.concatenate([rows_t, r]),
                                                    np.concatenate([emb_t.reshape(-1, info["dim"]), e]))
                else:
                    with open(self._path(kind, "rows"), "ab") as fh:
                        fh.write(r.tobytes())
                    with open(self._path(kind, "f16"), "ab") as fh:
                        fh.write(e.tobytes())
                info["rows"] += len(r)
            del rows_t, emb_t
        man["revision"] = db["revision"]
        self._write_manifest(man)

    def _load_for_update(self, kind: str, info: dict) -> tuple[np.ndarray, np.ndarray | None]:
        """The rows to mark (writable), and in memory mode the vectors to extend."""
        if self.in_memory:
            t = self.tables[kind]
            return t.rows, t.emb
        n, dim = info["rows"], info["dim"]
        for ext, size in (("rows", ROW.itemsize), ("f16", 2 * dim)):
            path = self._path(kind, ext)
            want = n * size
            have = path.stat().st_size if path.exists() else -1
            if have < want:
                raise _Rebuild("cache file is shorter than its manifest")
            if have > want:  # an append that stopped before the manifest was written
                os.truncate(path, want)
        if n == 0:
            return np.zeros(0, ROW), None
        return np.memmap(self._path(kind, "rows"), dtype=ROW, mode="r+", shape=(n,)), None

    def _open(self) -> None:
        """Map the files that the manifest describes (again, if they changed)."""
        man = self._manifest()
        key = (man.get("generation"), tuple(t["rows"] for t in man.get("tables", {}).values()))
        if key == self._opened:
            return
        tables = {}
        for kind in KINDS:
            info = man.get("tables", {}).get(kind, {"rows": 0, "dim": 0})
            n, dim = info["rows"], info["dim"]
            if n == 0:
                tables[kind] = _empty(dim)
                continue
            tables[kind] = VectorTable(
                np.memmap(self._path(kind, "rows"), dtype=ROW, mode="r", shape=(n,)),
                np.memmap(self._path(kind, "f16"), dtype=np.float16, mode="r", shape=(n, dim)))
        self.tables = tables
        self._opened = key
