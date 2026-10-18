"""Build and update the index for a library root."""

from __future__ import annotations

import concurrent.futures as cf
import errno
import os
import re
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import framestore, media, scenes, speech
from .config import Config
from .library import ScannedFile, default_index_dir, fingerprint, mount_point, norm, scan_with_errors, volume_name
from .store import STAGES, FileRow, Store
from .timefmt import short

DEPENDS = {
    "frames": set(),
    "visual": {"frames"},
    "scenes": {"frames", "visual"},
    "speech": set(),
    "speech_emb": {"speech"},
}

# The indexer runs one pass per model over all files: (name, stages, model).
PASSES = (
    ("visual", ("frames", "visual"), "siglip"),
    ("speech", ("speech",), "asr"),
    ("qwen", ("speech_emb", "scenes"), "qwen"),
)


class IndexError_(RuntimeError):
    """Configuration problem that stops the run (named to avoid shadowing IndexError)."""


def log_stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _pause(seconds: float) -> None:
    time.sleep(seconds)


_FRAME_FILE = re.compile(r"^\d{6}\.jpg$")  # the names ffmpeg writes for sampled frames


def _remove_frame_folder(folder: Path) -> None:
    """Delete the frame JPEGs that clipdex wrote, then the folder if it is empty. Nothing else."""
    if not folder.is_dir():
        return
    for f in folder.iterdir():
        if f.is_file() and _FRAME_FILE.match(f.name):
            f.unlink(missing_ok=True)
    try:
        folder.rmdir()
    except OSError:  # other files in it: not ours, keep them
        pass


def _count(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def free_memory() -> None:
    """Give memory of unloaded models back: Python objects, then the PyTorch and MLX caches."""
    import gc

    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - only a cache
            pass
    mx = sys.modules.get("mlx.core")
    if mx is not None:
        clear = getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", None), "clear_cache", None)
        if callable(clear):
            try:
                clear()
            except Exception:  # noqa: BLE001 - only a cache
                pass


class Models:
    """Creates the AI models on first use. Tests replace this with fakes."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._siglip = None
        self._qwen = None
        self._asr = None
        self._asr_loaded = False

    def siglip(self):
        if self._siglip is None:
            from .embed_siglip import SiglipEncoder

            self._siglip = SiglipEncoder(self.cfg.siglip_model, self.cfg.device)
        return self._siglip

    def qwen(self):
        if not self.cfg.qwen_enabled:
            return None
        if self._qwen is None:
            from .embed_qwen import QwenEmbedder

            self._qwen = QwenEmbedder(self.cfg.qwen_model, self.cfg.device, self.cfg.qwen_dtype,
                                      max_frames=self.cfg.scene_max_frames)
        return self._qwen

    def asr_enabled(self) -> bool:
        return self.cfg.asr_backend != "none"

    def asr(self):
        if not self._asr_loaded:
            self._asr = speech.make_backend(self.cfg.asr_backend, self.cfg.asr_model, self.cfg.device)
            self._asr_loaded = True
        return self._asr

    def release(self, kind: str) -> None:
        """Unload one model ("siglip", "asr" or "qwen"). The next use loads it again."""
        if kind == "siglip":
            self._siglip = None
        elif kind == "qwen":
            self._qwen = None
        elif kind == "asr":
            close = getattr(self._asr, "close", None)
            if callable(close):
                close()
            self._asr, self._asr_loaded = None, False
        free_memory()

    def transcribe(self, audio: np.ndarray, duration: float, relpath: str = "") -> speech.Transcript:
        backend = self.asr()
        if backend is None:
            return speech.Transcript(language=None, speech_seconds=0.0, segments=[])
        return speech.transcribe(audio, backend, duration, self.cfg.language,
                                 self.cfg.vad_threshold, self.cfg.min_speech_seconds, self.cfg.languages)


@dataclass
class SyncReport:
    added: int = 0
    changed: int = 0
    moved: int = 0
    missing: int = 0
    pruned: int = 0
    total: int = 0
    prune_refused: bool = False
    unreadable: list[str] = field(default_factory=list)


@dataclass
class RunReport:
    sync: SyncReport = field(default_factory=SyncReport)
    processed: int = 0
    failed: int = 0
    skipped_failed: int = 0
    seconds: float = 0.0
    media_seconds: float = 0.0


@dataclass
class Prepared:
    row: FileRow
    info: media.MediaInfo | None = None
    frames: list[media.Frame] | None = None
    tmp_dir: Path | None = None
    audio: np.ndarray | None = None
    audio_error: str | None = None
    error: Exception | None = None
    seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)


class _Eta:
    """Time left in a pass, from the footage seconds still to do and the speed so far.

    The first file is left out of the speed (it includes loading the model). Files that were
    never probed get a duration from their size and the bit rate of the files done so far.
    """

    def __init__(self, rows: list[FileRow]):
        self.sizes = [r.size for r in rows]
        self.known = [r.duration or 0.0 for r in rows]
        self.bytes_done = self.seconds_done = 0.0  # for the bit rate
        self.media_after_first = 0.0
        self.first_done: float | None = None

    def skip(self, i: int) -> None:
        self.known[i] = 0.0
        self.sizes[i] = 0
        if self.first_done is None:
            self.first_done = time.time()

    def done(self, i: int, duration: float) -> None:
        self.known[i] = duration
        if duration:
            self.bytes_done += self.sizes[i]
            self.seconds_done += duration
        if self.first_done is None:
            self.first_done = time.time()
        else:
            self.media_after_first += duration

    def text(self, i: int) -> str:
        if i + 1 >= len(self.sizes) or self.first_done is None or self.media_after_first <= 0:
            return ""
        rate = self.media_after_first / max(time.time() - self.first_done, 1e-6)  # footage s per s
        bitrate = self.bytes_done / self.seconds_done if self.seconds_done else 0.0
        left = sum(d or (s / bitrate if bitrate else 0.0)
                   for d, s in zip(self.known[i + 1:], self.sizes[i + 1:], strict=True))
        return f"  ETA {short(left / rate)}" if rate > 0 else ""


class _Lock:
    def __init__(self, path: Path):
        self.path = path
        self.fh = None

    def __enter__(self):
        self.fh = open(self.path, "a+")
        try:
            if sys.platform == "win32":
                import msvcrt

                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                self.fh.close()
                self.fh = None
                raise IndexError_("Another 'clipdex index' run uses this index right now.") from exc
            # File systems without lock support: continue without a lock.
        return self

    def __exit__(self, *exc):
        if self.fh:
            self.fh.close()


class Indexer:
    def __init__(self, root: Path, cfg: Config, index_dir: Path | None = None,
                 models: Models | None = None, log: Callable[[str], None] = log_stderr):
        self.root = root.resolve()
        if not self.root.is_dir():
            raise IndexError_(f"Library folder not found: {self.root}")
        self.index_dir = (index_dir or default_index_dir(self.root)).resolve()
        if self.index_dir == self.root or self.root.is_relative_to(self.index_dir):
            raise IndexError_(f"The index folder ({self.index_dir}) must not be the library folder or contain it. "
                              f"Use {default_index_dir(self.root)} (the default) or a folder outside the library.")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.index_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)
        self.cfg = cfg
        self.models = models or Models(cfg)
        self.log = log
        self.store = Store(self.index_dir / "index.db")
        # Remember the drive, so search can name it when it is not mounted.
        location = {"root": str(self.root), "root_mount": str(mount_point(self.root)),
                    "root_volume": volume_name(self.root) or ""}
        if any(self.store.get_meta(k) != v for k, v in location.items()):
            for k, v in location.items():
                self.store.set_meta(k, v)
            self.store.commit()

    # --- sync -------------------------------------------------------------------------------
    def _frame_dir(self, file_id: int) -> Path:
        return self.frames_dir / str(file_id)

    def _drop_frame_dir(self, file_id: int) -> None:
        _remove_frame_folder(self._frame_dir(file_id))
        (self.frames_dir / f"{file_id}.pack").unlink(missing_ok=True)

    def _pack_loose_frames(self) -> None:
        """frame_store = "packed": move frames from one-JPEG-per-frame folders into one pack per video."""
        if self.cfg.frame_store != "packed":
            return
        ids = [r[0] for r in self.store.query("SELECT DISTINCT file_id FROM frames WHERE image NOT LIKE '%.pack#%'")]
        if ids:
            self.log(f"Packing the frames of {len(ids)} file(s) into one file per video.")
        for n, file_id in enumerate(ids, start=1):
            rows = self.store.query("SELECT id, image FROM frames WHERE file_id=? ORDER BY idx, t", (file_id,))
            rel = f"frames/{file_id}.pack"
            try:
                spans = framestore.write_pack([self.index_dir / r["image"] for r in rows], self.index_dir / rel)
            except OSError as exc:
                self.log(f"    warning: frames of file {file_id} not packed ({exc})")
                continue
            self.store.conn.executemany("UPDATE frames SET image=? WHERE id=?",
                                        [(framestore.pack_ref(rel, o, size), r["id"])
                                         for r, (o, size) in zip(rows, spans, strict=True)])
            self.store.commit()
            _remove_frame_folder(self._frame_dir(file_id))
            if n % 200 == 0:
                self.log(f"    packed {n}/{len(ids)}")
        # folders left over from a stopped run, for files whose frames are packed already
        for (file_id,) in self.store.query(
                "SELECT DISTINCT file_id FROM frames WHERE image LIKE '%.pack#%' AND file_id NOT IN "
                "(SELECT file_id FROM frames WHERE image NOT LIKE '%.pack#%')"):
            _remove_frame_folder(self._frame_dir(file_id))

    def sync(self, prune: bool = False, force: bool = False) -> SyncReport:
        report = SyncReport()
        result = scan_with_errors(self.root, self.cfg, self.index_dir)
        scanned = result.files
        report.total = len(scanned)
        report.unreadable = list(result.unreadable)
        rows = {r.relpath: r for r in self.store.files()}

        def protected(relpath: str) -> bool:
            # Files in folders we could not read are not "gone". Never mark or prune them.
            return any(relpath == u or (u.endswith("/") and relpath.startswith(u)) for u in result.unreadable)

        def fp_of(relpath: str) -> str | None:
            try:
                return fingerprint(self.root / relpath)
            except OSError:
                return None

        seen: set[str] = set()
        new: list[ScannedFile] = []
        for s in scanned:
            seen.add(s.relpath)
            row = rows.get(s.relpath)
            if row is None:
                new.append(s)
                continue
            if row.size == s.size and abs(row.mtime - s.mtime) <= 2.0:
                if row.missing:
                    self.store.update_file(row.id, missing=0)
                continue
            fp = fp_of(s.relpath)
            if row.size == s.size and fp is not None and fp == row.fingerprint:
                # Only the time stamp moved (time zone change on exFAT, copy without times).
                self.store.update_file(row.id, mtime=s.mtime, missing=0)
                continue
            self.store.reset_stages(row.id, STAGES)
            self.store.update_file(row.id, size=s.size, mtime=s.mtime, fingerprint=fp, duration=None,
                                   error=None, missing=0, language=None, speech_seconds=None)
            self._drop_frame_dir(row.id)
            report.changed += 1

        gone = [r for path, r in rows.items() if path not in seen and not protected(path)]
        by_fp = {(r.size, r.fingerprint): r for r in gone if r.fingerprint}
        for s in new:
            fp = fp_of(s.relpath)
            match = by_fp.pop((s.size, fp), None) if fp else None
            if match is not None:
                self.store.update_file(match.id, relpath=s.relpath, mtime=s.mtime, missing=0)
                gone.remove(match)
                report.moved += 1
            else:
                self.store.add_file(s.relpath, s.size, s.mtime, fp)
                report.added += 1

        if prune and gone and not force and (not scanned or (len(gone) > 10 and len(gone) > len(rows) / 2)):
            # Probably the wrong folder, an empty mount point or a drive error: do not delete.
            report.prune_refused = True
            prune = False
        for r in gone:
            if prune:
                self.store.delete_file(r.id)
                self._drop_frame_dir(r.id)
                report.pruned += 1
            else:
                if not r.missing:
                    self.store.update_file(r.id, missing=1)
                report.missing += 1
        self.store.bump_revision()
        self.store.commit()
        return report

    # --- models bookkeeping ---------------------------------------------------------------------
    def _check_models(self, wanted: set[str], reembed: Iterable[str]) -> None:
        reembed = set(reembed)
        checks = []
        if "visual" in wanted:
            checks.append(("model.siglip", self.cfg.siglip_model, "visual", ("visual",)))
        if wanted & {"scenes", "speech_emb"} and self.cfg.qwen_enabled:
            checks.append(("model.qwen", self.cfg.qwen_model, "qwen", ("scenes", "speech_emb")))
        for key, model, flag, stages in checks:
            stored = self.store.get_meta(key)
            if stored is None or stored == model:
                self.store.set_meta(key, model)
                continue
            if flag not in reembed:
                raise IndexError_(
                    f"The index was built with {stored}, but the config now says {model}. "
                    f"Run again with --reembed {flag} to rebuild these embeddings.")
            for row in self.store.files():
                self.store.reset_stages(row.id, stages)
            self.store.set_meta(key, model)
            self.log(f"Rebuilding {', '.join(stages)} embeddings with {model}.")
        self.store.commit()

    def _wanted_stages(self, stages: Iterable[str] | None) -> set[str]:
        if stages:
            unknown = set(stages) - set(STAGES)
            if unknown:
                raise IndexError_(f"Unknown stage(s): {', '.join(sorted(unknown))}. Use: {', '.join(STAGES)}")
            wanted = set(stages)
        else:
            wanted = {"frames", "visual"}
            if self.models.asr_enabled():
                wanted.add("speech")
            if self.cfg.qwen_enabled:
                wanted.add("scenes")
                if self.models.asr_enabled():
                    wanted.add("speech_emb")
        if not self.cfg.qwen_enabled:
            wanted -= {"scenes", "speech_emb"}
        if not self.models.asr_enabled():
            wanted -= {"speech", "speech_emb"}
        closure = set(wanted)
        for s in wanted:
            closure |= DEPENDS[s]
        return closure

    # --- run ------------------------------------------------------------------------------------
    def run(self, only: str | None = None, stages: Iterable[str] | None = None, limit: int | None = None,
            retry_failed: bool = False, prune: bool = False, reembed: Iterable[str] = (),
            force: bool = False, redo: Iterable[str] = ()) -> RunReport:
        with _Lock(self.index_dir / ".lock"):
            media.STOP.clear()
            try:
                return self._run(only, stages, limit, retry_failed, prune, reembed, force, list(redo))
            finally:
                media.STOP.clear()

    def _run(self, only, stages, limit, retry_failed, prune, reembed, force, redo) -> RunReport:
        media.ensure_tools()
        started = time.time()
        for tmp in self.frames_dir.glob("*.tmp"):  # from a stopped run: "<id>.tmp" folders, "<id>.pack.tmp"
            if tmp.is_dir() and re.fullmatch(r"\d+\.tmp", tmp.name):
                _remove_frame_folder(tmp)
            elif tmp.is_file() and re.fullmatch(r"\d+\.pack\.tmp", tmp.name):
                tmp.unlink(missing_ok=True)
        report = RunReport()
        report.sync = self.sync(prune=prune, force=force)
        s = report.sync
        self.log(f"Library {self.root}: {s.total} video files "
                 f"({s.added} new, {s.changed} changed, {s.moved} moved, {s.missing} missing"
                 + (f", {s.pruned} removed" if prune else "") + ").")
        if s.unreadable:
            self.log(f"Warning: {len(s.unreadable)} path(s) could not be read and were left unchanged: "
                     + ", ".join(s.unreadable[:5]) + (" ..." if len(s.unreadable) > 5 else ""))
        if s.prune_refused:
            self.log("Warning: --prune would remove more than half of the index (or the folder looks empty). "
                     "Nothing was removed. Check the drive, then use --prune --force if this is correct.")
        self._pack_loose_frames()
        wanted = self._wanted_stages(list(stages or []) + redo if stages else None)
        self._check_models(wanted, reembed)
        if redo:
            # Run these stages again (and the stages that depend on them), e.g. after a setting change.
            unknown = set(redo) - set(STAGES)
            if unknown:
                raise IndexError_(f"Unknown stage(s) for --redo: {', '.join(sorted(unknown))}")
            needle_redo = norm(only) if only else None
            count = 0
            for row in self.store.files(include_missing=False):
                if needle_redo is None or needle_redo in norm(row.relpath):
                    self.store.reset_stages(row.id, redo)
                    if "frames" in redo:
                        self._drop_frame_dir(row.id)
                    count += 1
            self._commit()
            self.log(f"Redo {', '.join(redo)} for {count} file(s).")

        todo: list[FileRow] = []
        needle = norm(only) if only else None
        for row in self.store.files(include_missing=False):
            if needle and needle not in norm(row.relpath):
                continue
            if not any(not row.done(st) for st in wanted):
                continue
            if row.error and not retry_failed:
                report.skipped_failed += 1
                continue
            todo.append(row)
        if limit:
            todo = todo[:limit]
        if not todo:
            self._update_search_cache()  # moved, pruned or reset files can still change it
            self.log("Nothing to do. The index is up to date.")
            report.seconds = time.time() - started
            return report
        self.log(f"Indexing {len(todo)} file(s). Stages: {', '.join(st for st in STAGES if st in wanted)}.")
        if "frames" in wanted:
            note = framestore.usage_note(self.cfg.frame_store, self.frames_dir)
            if note:
                self.log("Note: " + note)
        off = []
        if not self.cfg.qwen_enabled:
            off.append("action search and speech meaning (qwen_enabled = false)")
        if not self.models.asr_enabled():
            off.append("speech search (asr_backend = none)")
        if off:
            self.log("Off: " + "; ".join(off) + ". Visual search and exact speech search still work."
                     if self.models.asr_enabled() else "Off: " + "; ".join(off) + ".")

        # One pass per model: all files get SigLIP, then Whisper, then Qwen. Only one model is in
        # memory at a time, so the peak is the largest model, not the sum of all three.
        passes = [(name, [s for s in stages_ if s in wanted], model) for name, stages_, model in PASSES]
        passes = [p for p in passes if p[1]]
        failed: set[int] = set()
        durations: dict[int, float] = {}
        for k, (name, pass_stages, model) in enumerate(passes, start=1):
            rows = [r for r in (self.store.file_by_id(t.id) for t in todo if t.id not in failed)
                    if r is not None and any(not r.done(st) for st in pass_stages)]
            if not rows:
                continue
            self.log(f"Pass {k}/{len(passes)}: {name} ({', '.join(pass_stages)}) for {len(rows)} file(s).")
            try:
                self._run_pass(name, set(pass_stages), rows, failed, durations)
            finally:
                self.models.release(model)

        self._update_search_cache()
        report.failed = len(failed)
        done = {fid: d for fid, d in durations.items() if fid not in failed}
        report.processed = len(done)
        report.media_seconds = sum(done.values())
        report.seconds = time.time() - started
        from .hardware import memory_report

        speed = f" ({report.media_seconds / report.seconds:.1f}x realtime)" if report.seconds and done else ""
        self.log(f"Done: {report.processed} file(s), {report.failed} failed, "
                 f"{short(report.media_seconds)} of footage in {short(report.seconds)}{speed}. "
                 f"Memory: {memory_report()}.")
        return report

    def _update_search_cache(self) -> None:
        """Bring the search cache up to date now, so the next search starts fast."""
        from .vectors import VectorCache

        try:
            cache = VectorCache(self.index_dir, self.store)
            cache.sync()
            count = sum(len(t) for t in cache.tables.values())
            self.log(f"Search cache: {count:,} vectors ({'rebuilt' if cache.last_sync == 'rebuild' else 'updated'}).")
        except Exception as exc:  # noqa: BLE001 - search builds the cache itself when needed
            self.log(f"Warning: the search cache was not updated ({exc}). The next search updates it.")

    def _run_pass(self, name: str, stages: set[str], rows: list[FileRow], failed: set[int],
                  durations: dict[int, float]) -> None:
        """Run some stages for many files. A worker thread prepares the next file (ffmpeg only)."""
        uses_media = bool(stages & {"frames", "visual", "speech"})
        pool = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch") if uses_media else None

        def prepare(row: FileRow):
            return pool.submit(self._prepare, row, stages) if pool else None

        eta = _Eta(rows)
        future = prepare(rows[0])
        try:
            for i, row in enumerate(rows):
                prepared = future.result() if future is not None else Prepared(row=row)
                if pool is not None and i + 1 < len(rows):
                    future = prepare(rows[i + 1])
                t0 = time.time()
                label = f"[{name} {i + 1}/{len(rows)}]"
                try:
                    summary = self._process(prepared, stages)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
                    self.store.rollback()
                    message = f"{type(exc).__name__}: {exc}"[:800]
                    self.store.update_file(row.id, error=message)
                    self.store.bump_revision()
                    self.store.commit()
                    failed.add(row.id)
                    eta.skip(i)
                    self.log(f"{label} {row.relpath}  FAILED  {message}")
                    continue
                finally:
                    if prepared.tmp_dir is not None:
                        _remove_frame_folder(prepared.tmp_dir)
                for warning in prepared.warnings:
                    self.log(f"    warning: {warning}")
                elapsed = time.time() - t0 + prepared.seconds
                fresh = self.store.file_by_id(row.id)
                duration = (fresh.duration if fresh else None) or 0.0
                durations[row.id] = duration
                eta.done(i, duration)
                speed = f"  {duration / elapsed:.1f}x realtime" if elapsed > 0 and duration else ""
                left = eta.text(i)
                self.log(f"{label} {row.relpath} ({short(duration)})  {summary}  {elapsed:.1f}s{speed}{left}")
                if self.cfg.pause_seconds > 0 and i + 1 < len(rows):
                    _pause(self.cfg.pause_seconds)  # less heat on fanless Macs; Ctrl-C still works
        except KeyboardInterrupt:
            media.STOP.set()  # the prefetch thread starts no new ffmpeg process
            self.store.rollback()
            if future is not None:
                future.cancel()
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
            self.log("Stopped. Finished files are saved. Run the same command again to continue.")
            raise
        if pool is not None:
            pool.shutdown(wait=True)

    def _prepare(self, row: FileRow, wanted: set[str]) -> Prepared:
        """Runs in a worker thread: probe, sample frames, decode audio. No database writes."""
        t0 = time.time()
        prepared = Prepared(row=row)
        src = self.root / row.relpath
        try:
            if media.STOP.is_set():
                raise KeyboardInterrupt
            info = media.probe(src)
            prepared.info = info
            if "frames" in wanted and not row.frames_done and info.has_video:
                tmp = self.frames_dir / f"{row.id}.tmp"
                _remove_frame_folder(tmp)
                prepared.tmp_dir = tmp
                prepared.frames = media.extract_frames(
                    src, tmp, info.duration, self.cfg.frame_interval, self.cfg.frame_max_side,
                    self.cfg.frame_jpeg_quality, self.cfg.keyframes_first)
        except Exception as exc:  # noqa: BLE001
            prepared.error = exc
        if prepared.error is None and prepared.info is not None and "speech" in wanted \
                and not row.speech_done and prepared.info.has_audio and not media.STOP.is_set():
            try:
                prepared.audio = media.extract_audio(src, prepared.info.duration)
            except Exception as exc:  # noqa: BLE001 - a broken audio track must not cost the frames
                prepared.audio_error = f"audio track could not be decoded ({exc})"
        prepared.seconds = time.time() - t0
        return prepared

    def _commit(self) -> None:
        self.store.bump_revision()
        self.store.commit()

    def _process(self, p: Prepared, wanted: set[str]) -> str:
        if p.error is not None:
            raise p.error
        row, info = p.row, p.info
        notes: list[str] = []
        if info is not None:
            self.store.update_file(row.id, duration=info.duration, width=info.width, height=info.height,
                                   fps=info.fps, vcodec=info.vcodec, has_audio=int(info.has_audio), error=None)
            self._commit()
        duration = info.duration if info is not None else (row.duration or 0.0)

        # frames
        if "frames" in wanted and not row.frames_done:
            final = self._frame_dir(row.id)
            self._drop_frame_dir(row.id)
            items = []
            if p.frames and self.cfg.frame_store == "packed":
                rel = f"frames/{row.id}.pack"
                spans = framestore.write_pack([fr.path for fr in p.frames], self.index_dir / rel)
                items = [(fr.index, fr.t, framestore.pack_ref(rel, o, size))
                         for fr, (o, size) in zip(p.frames, spans, strict=True)]
            elif p.frames:
                assert p.tmp_dir is not None
                os.replace(p.tmp_dir, final)
                p.tmp_dir = None
                for fr in p.frames:
                    rel = f"frames/{row.id}/{fr.path.name}"
                    items.append((fr.index, fr.t, rel))
            self.store.add_frames(row.id, items)
            self.store.update_file(row.id, frames_done=1)
            self._commit()
            notes.append(_count(len(items), "frame"))

        # visual (SigLIP per frame)
        if "visual" in wanted and not row.visual_done:
            frame_rows = [r for r in self.store.frames_for_file(row.id) if r["emb"] is None]
            if frame_rows:
                enc = self.models.siglip()
                vecs = enc.encode_images([framestore.source(self.index_dir, r["image"]) for r in frame_rows],
                                         batch=self.cfg.siglip_batch)
                self.store.set_frame_embeddings([(r["id"], v) for r, v in zip(frame_rows, vecs, strict=True)])
            self.store.update_file(row.id, visual_done=1)
            self._commit()
            notes.append("visual")

        # speech (VAD + Whisper)
        if "speech" in wanted and not row.speech_done:
            if p.audio is not None and p.audio.size:
                tr = self.models.transcribe(p.audio, duration, row.relpath)
            else:
                tr = speech.Transcript(language=None, speech_seconds=0.0, segments=[])
                if p.audio_error:
                    p.warnings.append(p.audio_error + "; the file has no transcript")
            rows = [s.to_row() if isinstance(s, speech.Segment) else s for s in tr.segments]
            self.store.add_segments(row.id, rows, tr.language)
            self.store.add_chunks(row.id, speech.make_chunks(rows, self.cfg.speech_chunk_seconds))
            self.store.update_file(row.id, speech_done=1, language=tr.language,
                                   speech_seconds=round(tr.speech_seconds, 2))
            self._commit()
            notes.append(f"speech {tr.language or '-'} {short(tr.speech_seconds)}" if rows else "no speech")

        # transcript chunk embeddings (Qwen text)
        if "speech_emb" in wanted and not row.speech_emb_done:
            chunks = [c for c in self.store.chunks_for_file(row.id) if c["emb"] is None]
            if chunks:
                qwen = self.models.qwen()
                vecs = qwen.embed_texts([c["text"] for c in chunks])
                self.store.set_chunk_embeddings([(c["id"], v) for c, v in zip(chunks, vecs, strict=True)])
            self.store.update_file(row.id, speech_emb_done=1)
            self._commit()
            notes.append(_count(len(chunks), "transcript chunk"))

        # scenes (Qwen video embeddings over short frame sequences)
        if "scenes" in wanted and not row.scenes_done:
            frame_rows = [r for r in self.store.frames_for_file(row.id) if r["emb"] is not None]
            self.store.reset_stages(row.id, ["scenes"])
            count = 0
            if frame_rows:
                embs = np.stack([np.frombuffer(r["emb"], dtype=np.float16).astype(np.float32)
                                 for r in frame_rows])
                times = [float(r["t"]) for r in frame_rows]
                found = scenes.segment(times, embs, duration, self.cfg.scene_max_seconds,
                                       self.cfg.scene_cut_similarity, self.cfg.scene_step_similarity)
                qwen = self.models.qwen()
                refs = [[frame_rows[k]["image"] for k in sc.frames] for sc in found]
                vecs = []
                step = max(1, self.cfg.qwen_batch)
                for i in range(0, len(refs), step):  # load only the frames of one batch at a time
                    part = [[framestore.qwen_item(self.index_dir, ref) for ref in seq] for seq in refs[i:i + step]]
                    vecs.extend(qwen.embed_frame_sequences(part, batch=step))
                for sc, vec in zip(found, vecs, strict=True):
                    self.store.add_scene(row.id, sc.start, sc.end, [frame_rows[k]["id"] for k in sc.frames], vec)
                count = len(found)
            self.store.update_file(row.id, scenes_done=1)
            self._commit()
            notes.append(_count(count, "scene"))
        return " · ".join(notes) if notes else "up to date"
