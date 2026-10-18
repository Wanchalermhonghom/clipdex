"""Search the index: four channels fused into ranked moments.

Channels
  visual      SigLIP 2 text -> frame similarity (what is visible in a frame)
  scenes      Qwen3-VL-Embedding text -> short scene video (actions, what happens)
  speech      SQLite FTS5 over the Whisper transcript (exact words)
  speech_sem  Qwen3-VL-Embedding text -> transcript chunks (what is talked about)

Each channel filters out weak matches first, because nearest-neighbour search
always returns something. The rest is merged with reciprocal rank fusion into
moments: a file plus a time range.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import fts
from . import scenes as scene_split
from .config import Config
from .library import norm, volume_name
from .store import Store
from .timefmt import tc
from .vectors import VectorCache, VectorTable

# How exact the time of a hit is: a spoken word, a frame, a scene, a 20 s transcript chunk.
_PRECISION = {"speech": 3, "visual": 2, "scenes": 1, "speech_sem": 0}

MODES = {
    "auto": {"visual": 1.0, "scenes": 1.0, "speech": 1.0, "speech_sem": 0.8},
    "visual": {"visual": 1.0, "scenes": 0.7},
    "action": {"scenes": 1.0, "visual": 0.4},
    "speech": {"speech": 1.0, "speech_sem": 1.0},
}


@dataclass
class Hit:
    channel: str
    file_id: int
    start: float
    end: float
    peak: float
    score: float
    strength: float = 0.0  # comparable within a channel: logit, cosine or -bm25
    rank: int = 0
    ref: int = 0
    text: str | None = None


@dataclass
class Moment:
    file_id: int
    start: float
    end: float
    hits: list[Hit] = field(default_factory=list)
    score: float = 0.0
    best: float = 0.0


def _cosine(matrix: np.ndarray, query: np.ndarray, block: int = 32768) -> np.ndarray:
    """Dot products of all rows with the query, in blocks (the matrix can be a memory-mapped file)."""
    if len(matrix) == 0:
        return np.zeros(0, np.float32)
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    out = np.empty(len(matrix), np.float32)
    buf = np.empty((min(block, len(matrix)), matrix.shape[1]), np.float32)  # reused: no new memory per block
    for i in range(0, len(matrix), block):
        part = buf[: min(block, len(matrix) - i)]
        np.copyto(part, matrix[i:i + len(part)])
        np.matmul(part, q, out=out[i:i + len(part)])
    return out


class SearchEngine:
    def __init__(self, index_dir: Path, cfg: Config, root: Path | None = None, models=None,
                 use_qwen: bool = True):
        self.index_dir = index_dir.resolve()
        self.cfg = cfg
        self.store = Store(self.index_dir / "index.db", readonly=True)
        stored_root = self.store.get_meta("root")
        self.root = (root or (Path(stored_root) if stored_root else None))
        stored_volume = self.store.get_meta("root_volume")
        if stored_volume is None and self.root is not None and self.root.is_dir():
            stored_volume = volume_name(self.root) or ""
        self.volume = stored_volume or None  # drive name, e.g. "T7"; None = system disk or unknown
        self.use_qwen = use_qwen and cfg.qwen_enabled
        self.models = models
        self._lock = threading.RLock()
        self._revision = -1
        self._siglip = None
        self._qwen = None
        self._available = (0.0, False)
        self.cache = VectorCache(self.index_dir, self.store)

    # --- models ---------------------------------------------------------------------
    def _siglip_model(self):
        if self.models is not None:
            return self.models.siglip()
        if self._siglip is None:
            from .embed_siglip import SiglipEncoder

            name = self.store.get_meta("model.siglip") or self.cfg.siglip_model
            self._siglip = SiglipEncoder(name, self.cfg.device)
        return self._siglip

    def _qwen_model(self):
        if not self.use_qwen:
            return None
        if self.models is not None:
            return self.models.qwen()
        if self._qwen is None:
            from .embed_qwen import QwenEmbedder

            name = self.store.get_meta("model.qwen") or self.cfg.qwen_model
            self._qwen = QwenEmbedder(name, self.cfg.device, self.cfg.qwen_dtype,
                                      max_frames=self.cfg.scene_max_frames)
        return self._qwen

    # --- cache ----------------------------------------------------------------------------
    def _refresh(self) -> None:
        rev = self.store.revision()
        if rev == self._revision:
            return
        # After 'index --reembed' the stored vectors come from another model: reload ours too.
        if self._siglip is not None and self._siglip.model_name != self.store.get_meta("model.siglip"):
            self._siglip = None
        if self._qwen is not None and self._qwen.model_name != self.store.get_meta("model.qwen"):
            self._qwen = None
        self.files = {r.id: r for r in self.store.files()}
        # Vectors stay in memory-mapped files; only the changed files are read again.
        self.cache.sync()
        tables = self.cache.tables
        self.frames, self.scene_table, self.chunks = tables["frames"], tables["scenes"], tables["chunks"]
        self._revision = rev

    # --- helpers ------------------------------------------------------------------------------
    def _file_mask(self, table: VectorTable, allowed: set[int] | None) -> np.ndarray:
        if allowed is None:
            return np.ones(len(table), dtype=bool)
        return np.isin(table.file_ids, np.fromiter(allowed, dtype=np.int64, count=len(allowed)))

    def _allowed_files(self, path_contains: str | None) -> set[int]:
        needle = norm(path_contains) if path_contains else None
        return {fid for fid, r in self.files.items()
                if not r.missing and (needle is None or needle in norm(r.relpath))}

    def nearest_frame(self, file_id: int, t: float) -> int | None:
        rows = self.store.query("SELECT id FROM frames WHERE file_id=? ORDER BY abs(t - ?), t LIMIT 1",
                                (int(file_id), float(t)))
        return int(rows[0]["id"]) if rows else None

    def root_available(self) -> bool:
        """True when the library folder can be read now. False when its drive is not mounted."""
        checked, value = self._available
        now = time.monotonic()
        if now - checked < 2.0:
            return value
        ok = self.root is not None and self.root.is_dir()
        mount = self.store.get_meta("root_mount")
        if ok and mount and Path(mount) != Path(Path(mount).anchor) and self.root.is_relative_to(mount):
            # An empty folder can stay at /Volumes/T7 after the drive is gone.
            ok = os.path.ismount(mount)
        self._available = (now, ok)
        return ok

    def abs_path(self, relpath: str) -> str | None:
        return str(self.root / relpath) if self.root else None

    def location(self, relpath: str) -> dict:
        """Where a file is: drive name, whether the drive is mounted, and the absolute path if it is."""
        available = self.root_available()
        return {"volume": self.volume, "available": available,
                "path": self.abs_path(relpath) if available else None}

    # --- channels -------------------------------------------------------------------------------
    def _visual(self, query: str, allowed: set[int], diag: dict) -> list[Hit]:
        if len(self.frames) == 0:
            return []
        enc = self._siglip_model()
        q = enc.encode_texts([query])[0]
        cos = _cosine(self.frames.emb, q)
        logit = enc.logits(cos)
        mask = self._file_mask(self.frames, allowed)
        if not mask.any():
            return []
        logit = np.where(mask, logit, -np.inf)
        top = float(logit.max())
        min_logit = math.log(self.cfg.visual_min_prob / (1 - self.cfg.visual_min_prob))
        keep = (logit >= top - self.cfg.visual_rel_logit) & (logit >= min_logit)
        idx = np.flatnonzero(keep)
        order = [int(i) for i in idx[np.argsort(-logit[idx], kind="stable")][: self.cfg.channel_top_k]]
        diag["visual"] = {"best_prob": round(enc.prob(top), 4), "kept": len(order)}
        # A frame stands for the time until the next sampled frame.
        span = self.cfg.frame_interval
        hits = []
        for k in order:
            t = float(self.frames.a[k])
            hits.append(Hit("visual", int(self.frames.file_ids[k]), t, t + span, t,
                            enc.prob(float(logit[k])), float(logit[k]), ref=int(self.frames.ids[k])))
        return hits

    def _cosine_channel(self, name: str, table: VectorTable, q: np.ndarray, allowed: set[int], floor: float,
                        margin: float, diag: dict) -> tuple[list[int], np.ndarray]:
        cos = _cosine(table.emb, q)
        mask = self._file_mask(table, allowed)
        if not mask.any():
            return [], cos
        valid = cos[mask]
        threshold = floor
        baseline = None
        if len(valid) >= 20:
            baseline = float(np.percentile(valid, self.cfg.baseline_percentile))
            threshold = max(floor, baseline + margin)
        cos = np.where(mask, cos, -np.inf)
        idx = np.flatnonzero(cos >= threshold)
        order = [int(i) for i in idx[np.argsort(-cos[idx], kind="stable")][: self.cfg.channel_top_k]]
        diag[name] = {"best": round(float(cos.max()), 4), "baseline": None if baseline is None else round(baseline, 4),
                      "threshold": round(threshold, 4), "kept": len(order)}
        return order, cos

    def _scenes(self, query: str, allowed: set[int], diag: dict) -> list[Hit]:
        qwen = self._qwen_model()
        if qwen is None or len(self.scene_table) == 0:
            return []
        q = qwen.embed_texts([query], instruction=self.cfg.qwen_visual_instruction)[0]
        order, cos = self._cosine_channel("scenes", self.scene_table, q, allowed, self.cfg.scene_floor,
                                          self.cfg.scene_margin, diag)
        hits = []
        for k in order:
            s, e = float(self.scene_table.a[k]), float(self.scene_table.b[k])
            hits.append(Hit("scenes", int(self.scene_table.file_ids[k]), s, e, (s + e) / 2, float(cos[k]),
                            float(cos[k]), ref=int(self.scene_table.ids[k])))
        return hits

    def _speech_sem(self, query: str, allowed: set[int], diag: dict) -> list[Hit]:
        qwen = self._qwen_model()
        if qwen is None or len(self.chunks) == 0:
            return []
        q = qwen.embed_texts([query], instruction=self.cfg.qwen_speech_instruction)[0]
        order, cos = self._cosine_channel("speech_sem", self.chunks, q, allowed,
                                          self.cfg.speech_sem_floor, self.cfg.speech_sem_margin, diag)
        texts = self._chunk_texts([int(self.chunks.ids[k]) for k in order])
        hits = []
        for k in order:
            s, e = float(self.chunks.a[k]), float(self.chunks.b[k])
            cid = int(self.chunks.ids[k])
            hits.append(Hit("speech_sem", int(self.chunks.file_ids[k]), s, e, s, float(cos[k]),
                            float(cos[k]), ref=cid, text=texts.get(cid, "")))
        return hits

    def _chunk_texts(self, chunk_ids: list[int]) -> dict[int, str]:
        out: dict[int, str] = {}
        for i in range(0, len(chunk_ids), 500):
            part = chunk_ids[i:i + 500]
            for r in self.store.query(
                    f"SELECT id, text FROM speech_chunks WHERE id IN ({','.join('?' * len(part))})", part):
                out[int(r["id"])] = r["text"]
        return out

    def _speech(self, query: str, allowed: set[int], diag: dict) -> list[Hit]:
        terms = fts.match_terms(query)
        for expr in fts.build(query):
            rows = self.store.query(
                "SELECT s.id, s.file_id, s.start, s.end, s.text, s.words, bm25(segments_fts) AS score "
                "FROM segments_fts JOIN segments s ON s.id = segments_fts.rowid "
                "WHERE segments_fts MATCH ? AND s.file_id IN (SELECT value FROM json_each(?)) "
                "ORDER BY score LIMIT ?",
                (expr, json.dumps(sorted(allowed)), self.cfg.channel_top_k))
            if rows:
                diag["speech"] = {"query": expr, "kept": len(rows)}
                hits = []
                for r in rows:
                    words = json.loads(r["words"] or "[]")
                    peak = fts.first_word_time(words, terms)
                    start = float(r["start"])
                    hits.append(Hit("speech", int(r["file_id"]), start, float(r["end"]),
                                    peak if peak is not None else start, -float(r["score"]),
                                    -float(r["score"]), ref=int(r["id"]), text=r["text"]))
                return hits
        diag["speech"] = {"query": " | ".join(fts.build(query)), "kept": 0}
        return []

    # --- fusion -------------------------------------------------------------------------------------
    def _fuse(self, hits: list[Hit], weights: dict[str, float]) -> list[Moment]:
        by_file: dict[int, list[Hit]] = defaultdict(list)
        for h in hits:
            by_file[h.file_id].append(h)
        moments: list[Moment] = []
        for file_id, items in by_file.items():
            items.sort(key=lambda h: (h.start, h.end))
            cur: Moment | None = None
            for h in items:
                inside = cur is not None and h.start >= cur.start and h.end <= cur.end
                if cur is not None and (inside or (
                        h.start <= cur.end + self.cfg.merge_gap
                        and max(cur.end, h.end) - cur.start <= self.cfg.moment_max_seconds)):
                    cur.end = max(cur.end, h.end)
                    cur.hits.append(h)
                else:
                    cur = Moment(file_id, h.start, h.end, [h])
                    moments.append(cur)
        k = self.cfg.rrf_k
        for m in moments:
            best_rank: dict[str, Hit] = {}
            for h in m.hits:
                if h.channel not in best_rank or h.rank < best_rank[h.channel].rank:
                    best_rank[h.channel] = h
            m.score = sum(weights[c] / (k + h.rank) for c, h in best_rank.items())
            # On equal weight, prefer the channel with the most exact time: a spoken word beats a frame,
            # a frame beats a scene, a scene beats a 20 s transcript chunk.
            top = max(best_rank.values(), key=lambda h: (weights[h.channel] / (k + h.rank), _PRECISION[h.channel]))
            m.best = top.peak
            # Span only the hits that are nearly as strong as the best hit of their channel,
            # so a weak neighbouring scene does not stretch the moment.
            tolerance = {"visual": self.cfg.span_tolerance_logit, "scenes": self.cfg.span_tolerance_cos,
                         "speech_sem": self.cfg.span_tolerance_cos, "speech": math.inf}
            best_strength = {c: max(h.strength for h in m.hits if h.channel == c) for c in best_rank}
            strong = [h for h in m.hits if h.strength >= best_strength[h.channel] - tolerance[h.channel]]
            m.start = min(h.start for h in strong)
            m.end = max(h.end for h in strong)
            duration = self.files[m.file_id].duration
            if duration:
                m.end = min(m.end, duration)
            m.start = max(0.0, min(m.start, m.best))
        moments.sort(key=lambda m: -m.score)
        return moments

    def _transcript_excerpt(self, file_id: int, start: float, end: float, limit: int = 400) -> str:
        rows = self.store.query(
            "SELECT text FROM segments WHERE file_id=? AND end>=? AND start<=? ORDER BY start",
            (file_id, start - 0.5, end + 0.5))
        text = " ".join(r["text"] for r in rows).strip()
        return text if len(text) <= limit else text[: limit - 1] + "…"

    # --- public API ------------------------------------------------------------------------------------
    def search(self, query: str, mode: str = "auto", limit: int = 10, path_contains: str | None = None,
               max_per_file: int | None = None, explain: bool = False) -> dict:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        query = query.strip()
        if not query:
            raise ValueError("query is empty")
        with self._lock:
            self._refresh()
            allowed = self._allowed_files(path_contains)
            weights = dict(MODES[mode])
            if not self.use_qwen:
                weights.pop("scenes", None)
                weights.pop("speech_sem", None)
                if not weights:  # action mode without Qwen: fall back to frames
                    weights = {"visual": 1.0}
            diag: dict = {}
            channels = {"visual": self._visual, "scenes": self._scenes, "speech": self._speech,
                        "speech_sem": self._speech_sem}
            hits: list[Hit] = []
            for name in weights:
                found = channels[name](query, allowed, diag) if allowed else []
                for rank, h in enumerate(found, start=1):
                    h.rank = rank
                hits += found
            moments = self._fuse(hits, weights)
            per_file = max_per_file or self.cfg.max_per_file
            counts: dict[int, int] = defaultdict(int)
            results = []
            for m in moments:
                if counts[m.file_id] >= per_file:
                    continue
                counts[m.file_id] += 1
                results.append(self._moment_dict(m, len(results) + 1, explain, weights))
                if len(results) >= limit:
                    break
            out = {"query": query, "mode": mode, "results": results,
                   "total_moments": len(moments), "searched_files": len(allowed)}
            if explain:
                out["channels"] = diag
            return out

    def _moment_dict(self, m: Moment, rank: int, explain: bool, weights: dict[str, float]) -> dict:
        f = self.files[m.file_id]
        chans: dict[str, dict] = {}
        for h in sorted(m.hits, key=lambda h: h.rank):
            if h.channel in chans:
                continue
            entry: dict = {"rank": h.rank, "score": round(h.score, 4), "at": round(h.peak, 2)}
            if h.text and h.channel in ("speech", "speech_sem"):
                entry["text"] = h.text if len(h.text) <= 240 else h.text[:239] + "…"
            chans[h.channel] = entry
        frame_id = None
        for h in sorted(m.hits, key=lambda h: h.rank):
            if h.channel == "visual":
                frame_id = h.ref
                break
        if frame_id is None:
            frame_id = self.nearest_frame(m.file_id, m.best)
        out = {
            "rank": rank,
            "file": f.relpath,
            **self.location(f.relpath),
            "start": round(m.start, 2),
            "end": round(m.end, 2),
            "best": round(m.best, 2),
            "start_tc": tc(m.start),
            "end_tc": tc(m.end),
            "best_tc": tc(m.best),
            "file_duration": round(f.duration or 0.0, 2),
            # strongest contribution to the fused score first
            "matched": sorted(chans, key=lambda c: (-weights[c] / (self.cfg.rrf_k + chans[c]["rank"]),
                                                    -_PRECISION[c])),
            "channels": chans,
            "frame_id": frame_id,
            "transcript": self._transcript_excerpt(m.file_id, m.start, m.end),
        }
        if explain:
            out["fused_score"] = round(m.score, 5)
        return out

    def similar_frames(self, frame_id: int, limit: int = 10, path_contains: str | None = None) -> dict:
        with self._lock:
            self._refresh()
            src = self.store.query("SELECT file_id, t, emb FROM frames WHERE id=? AND emb IS NOT NULL",
                                   (int(frame_id),))
            if not src:
                raise ValueError(f"Unknown frame_id {frame_id} (or it has no embedding yet)")
            q = np.frombuffer(src[0]["emb"], dtype=np.float16).astype(np.float32)
            cos = _cosine(self.frames.emb, q)
            allowed = self._allowed_files(path_contains)
            mask = self._file_mask(self.frames, allowed)
            src_file, src_t = int(src[0]["file_id"]), float(src[0]["t"])
            same = (self.frames.file_ids == src_file) & (np.abs(self.frames.a - src_t) < 10.0)
            cos = np.where(mask & ~same, cos, -np.inf)
            results = []
            used: dict[int, list[float]] = defaultdict(list)
            for i in np.argsort(-cos):
                if not np.isfinite(cos[i]) or len(results) >= limit:
                    break
                fid, t = int(self.frames.file_ids[i]), float(self.frames.a[i])
                if any(abs(t - u) < 10.0 for u in used[fid]):
                    continue  # one result per 10 s window
                used[fid].append(t)
                f = self.files[fid]
                results.append({"rank": len(results) + 1, "file": f.relpath, **self.location(f.relpath),
                                "at": round(t, 2), "at_tc": tc(t), "similarity": round(float(cos[i]), 4),
                                "frame_id": int(self.frames.ids[i])})
            return {"frame_id": int(frame_id), "results": results}

    def frame_info(self, frame_ids: list[int]) -> list[dict]:
        out = []
        for fid in frame_ids:
            rows = self.store.query("SELECT id, file_id, t, image FROM frames WHERE id=?", (int(fid),))
            if not rows:
                out.append({"frame_id": int(fid), "error": "unknown frame_id"})
                continue
            r = rows[0]
            f = self.store.file_by_id(r["file_id"])
            out.append({"frame_id": int(fid), "file": f.relpath if f else None, "at": round(r["t"], 2),
                        "at_tc": tc(r["t"]), "image": str(self.index_dir / r["image"])})
        return out

    def _resolve_file(self, file: str):
        """Find a file by relative path, absolute path or a unique end of the path."""
        f = self.store.file_by_path(file)
        if f is not None:
            return f
        needle = norm(file)
        if self.root is not None:
            root = norm(str(self.root)).rstrip("/") + "/"
            if needle.startswith(root):
                needle = needle[len(root):]
        rows = self.store.files()
        candidates = [r for r in rows if norm(r.relpath) == needle] or \
                     [r for r in rows if norm(r.relpath).endswith("/" + needle)] or \
                     [r for r in rows if needle in norm(r.relpath).rsplit("/", 1)[-1]]  # part of the name
        if len(candidates) > 1:
            raise ValueError(f"'{file}' matches {len(candidates)} files; give more of the path")
        if not candidates:
            raise ValueError(f"File not in the index: {file}")
        return candidates[0]

    def transcript(self, file: str, start: float | None = None, end: float | None = None,
                   words: bool = False) -> dict:
        f = self._resolve_file(file)
        rows = self.store.query(
            "SELECT start, end, text, words FROM segments WHERE file_id=? AND end>=? AND start<=? ORDER BY start",
            (f.id, start if start is not None else -1.0, end if end is not None else 1e12))
        lines = []
        for r in rows:
            line = {"start": round(r["start"], 2), "end": round(r["end"], 2), "start_tc": tc(r["start"]),
                    "text": r["text"]}
            if words:
                line["words"] = json.loads(r["words"] or "[]")
            lines.append(line)
        return {"file": f.relpath, **self.location(f.relpath), "language": f.language,
                "duration": f.duration, "speech_done": bool(f.speech_done), "segments": lines}

    def scenes(self, file: str, changes_only: bool = False) -> dict:
        """Scene boundaries of one file, computed from the stored frame vectors.

        cut = "start" (first scene), "picture change" (the picture changed a lot) or
        "length limit" (the scene reached scene_max_seconds; no visible change).
        changes_only=True ignores the length limit, so every boundary is a real change.
        """
        f = self._resolve_file(file)
        rows = [r for r in self.store.frames_for_file(f.id) if r["emb"] is not None]
        out = {"file": f.relpath, **self.location(f.relpath), "duration": f.duration,
               "visual_done": bool(f.visual_done), "scenes": []}
        if not rows:
            return out
        times = [float(r["t"]) for r in rows]
        embs = np.stack([np.frombuffer(r["emb"], dtype=np.float16).astype(np.float32) for r in rows])
        max_seconds = math.inf if changes_only else self.cfg.scene_max_seconds
        found = scene_split.segment(times, embs, f.duration or 0.0, max_seconds, self.cfg.scene_cut_similarity,
                                    self.cfg.scene_step_similarity)
        for k, sc in enumerate(found):
            first = sc.frames[0]
            if k == 0:
                cut, similarity = "start", None
            else:
                similarity = float(embs[first] @ embs[first - 1])  # vs. the frame before the cut
                drift = float(embs[first] @ embs[found[k - 1].frames[0]])
                changed = similarity < self.cfg.scene_step_similarity or drift < self.cfg.scene_cut_similarity
                cut = "picture change" if changed else "length limit"
            out["scenes"].append({
                "index": k + 1, "start": round(sc.start, 2), "end": round(sc.end, 2),
                "start_tc": tc(sc.start), "end_tc": tc(sc.end), "cut": cut,
                "similarity_to_previous": None if similarity is None else round(similarity, 3),
                "frame_id": int(rows[first]["id"]),
                "frame_ids": [int(rows[i]["id"]) for i in sc.frames],
            })
        return out

    def list_files(self, path_contains: str | None = None, limit: int = 200) -> dict:
        needle = norm(path_contains) if path_contains else None
        rows = [r for r in self.store.files() if needle is None or needle in norm(r.relpath)]
        items = [{"file": r.relpath, "duration": round(r.duration or 0, 2), "size_mb": round(r.size / 1e6, 1),
                  "resolution": f"{r.width}x{r.height}" if r.width else None, "language": r.language,
                  "speech_seconds": r.speech_seconds, "missing": bool(r.missing), "error": r.error,
                  "indexed": all(r.done(s) for s in ("frames", "visual"))} for r in rows[:limit]]
        return {"count": len(rows), "files": items}

    def status(self) -> dict:
        s = self.store.stats()
        s["root"] = str(self.root) if self.root else None
        s["root_available"] = self.root_available()
        s["volume"] = self.volume
        s["index_dir"] = str(self.index_dir)
        s["models"] = {"siglip": self.store.get_meta("model.siglip"), "qwen": self.store.get_meta("model.qwen")}
        s["duration_hours"] = round(s.pop("duration_seconds") / 3600, 2)
        s["speech_hours"] = round(s.pop("speech_seconds") / 3600, 2)
        failed = self.store.query("SELECT relpath, error FROM files WHERE error IS NOT NULL AND missing=0 LIMIT 20")
        s["failed_files"] = [{"file": r["relpath"], "error": r["error"]} for r in failed]
        return s
