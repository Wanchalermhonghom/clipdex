"""Find video files under a library root."""

from __future__ import annotations

import fnmatch
import hashlib
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .config import Config

INDEX_DIR_NAME = ".clipdex"
_FINGERPRINT_BYTES = 1 << 20  # 1 MiB from the start and from the end


@dataclass(frozen=True)
class ScannedFile:
    relpath: str  # POSIX style, relative to the library root
    size: int
    mtime: float


def default_index_dir(root: Path) -> Path:
    return root / INDEX_DIR_NAME


def mount_point(path: Path) -> Path:
    """The mount point that contains `path`, e.g. /Volumes/T7 for /Volumes/T7/Footage."""
    p = Path(os.path.abspath(path))
    while not os.path.ismount(p) and p != p.parent:
        p = p.parent
    return p


def volume_name(path: Path) -> str | None:
    """Name of the drive that holds `path` ("T7"), or None for the system disk."""
    mp = mount_point(path)
    return None if mp == Path(mp.anchor) else mp.name


def _is_hidden(name: str) -> bool:
    return name.startswith(".")


def norm(text: str) -> str:
    """Normal form for path matching. macOS stores "ü" decomposed (NFD); typed text is NFC."""
    return unicodedata.normalize("NFC", text).casefold()


@dataclass
class ScanResult:
    files: list[ScannedFile]
    unreadable: list[str]  # relative paths of folders ("x/") or files that could not be read


def scan(root: Path, cfg: Config, index_dir: Path | None = None) -> list[ScannedFile]:
    return scan_with_errors(root, cfg, index_dir).files


def scan_with_errors(root: Path, cfg: Config, index_dir: Path | None = None) -> ScanResult:
    """Walk the root and return all video files, sorted by path, plus what could not be read."""
    root = root.resolve()
    unreadable: list[str] = []

    def on_error(exc: OSError) -> None:
        try:
            rel = Path(exc.filename).resolve().relative_to(root).as_posix()
        except (TypeError, ValueError):
            rel = str(exc.filename)
        unreadable.append("" if rel == "." else rel.rstrip("/") + "/")

    extensions = {e.lower() if e.startswith(".") else "." + e.lower() for e in cfg.video_extensions}
    exclude_dirs = {d.lower() for d in cfg.exclude_dirs}
    index_real = index_dir.resolve() if index_dir is not None else None
    found: list[ScannedFile] = []

    for dirpath, dirnames, filenames in os.walk(root, onerror=on_error, followlinks=False):
        current = Path(dirpath)
        keep = []
        for d in dirnames:
            if _is_hidden(d) or d.lower() in exclude_dirs:
                continue
            full = current / d
            if index_real is not None and full.resolve() == index_real:
                continue
            rel = full.relative_to(root).as_posix() + "/"
            if any(fnmatch.fnmatch(rel, g) for g in cfg.exclude_globs):
                continue
            keep.append(d)
        dirnames[:] = sorted(keep)

        for name in filenames:
            if _is_hidden(name):  # also skips macOS "._" AppleDouble files
                continue
            if os.path.splitext(name)[1].lower() not in extensions:
                continue
            full = current / name
            rel = full.relative_to(root).as_posix()
            if any(fnmatch.fnmatch(rel, g) for g in cfg.exclude_globs):
                continue
            try:
                st = full.stat()
            except OSError:
                unreadable.append(rel)
                continue
            if st.st_size == 0:
                continue
            found.append(ScannedFile(rel, st.st_size, st.st_mtime))

    found.sort(key=lambda f: f.relpath)
    return ScanResult(found, unreadable)


def fingerprint(path: Path) -> str:
    """Cheap content fingerprint: size + first and last MiB. Used to detect moved files."""
    size = path.stat().st_size
    h = hashlib.sha1(str(size).encode())
    with open(path, "rb") as fh:
        h.update(fh.read(_FINGERPRINT_BYTES))
        if size > 2 * _FINGERPRINT_BYTES:
            fh.seek(size - _FINGERPRINT_BYTES)
            h.update(fh.read(_FINGERPRINT_BYTES))
    return h.hexdigest()
