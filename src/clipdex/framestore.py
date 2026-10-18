"""Where the stored frames (small JPEG images) live.

frame_store = "packed" (default): one file per video, frames/<file id>.pack, with the JPEGs one after
    the other. The database stores "frames/<id>.pack#<offset>+<length>" for each frame.
frame_store = "files": one JPEG per frame, frames/<file id>/<n>.jpg. That is about 1,800 files per
    hour of footage, and on drives with large blocks (exFAT) every small file takes a whole block.
"""

from __future__ import annotations

import io
import os
import re
from collections.abc import Sequence
from pathlib import Path

_PACK_RE = re.compile(r"^(?P<path>[^#]+\.pack)#(?P<offset>\d+)\+(?P<length>\d+)$")
FRAME_KB = 21  # measured: 512 px JPEG at quality 4 from iPhone footage
FRAMES_PER_HOUR = 1800  # one frame every 2 s


def is_packed(ref: str) -> bool:
    return _PACK_RE.match(ref) is not None


def pack_ref(rel_pack: str, offset: int, length: int) -> str:
    return f"{rel_pack}#{offset}+{length}"


def write_pack(jpegs: Sequence[Path], dest: Path) -> list[tuple[int, int]]:
    """Write the JPEG files into one pack file. Returns (offset, length) per frame."""
    spans = []
    tmp = dest.with_name(dest.name + ".tmp")
    with open(tmp, "wb") as out:
        for path in jpegs:
            data = Path(path).read_bytes()
            spans.append((out.tell(), len(data)))
            out.write(data)
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, dest)
    return spans


def read_bytes(index_dir: Path, ref: str) -> bytes:
    m = _PACK_RE.match(ref)
    if m is None:
        return (index_dir / ref).read_bytes()
    offset, length = int(m["offset"]), int(m["length"])
    with open(index_dir / m["path"], "rb") as fh:
        fh.seek(offset)
        data = fh.read(length)
    if len(data) != length:
        raise OSError(f"frame pack is too short: {m['path']}")
    return data


def source(index_dir: Path, ref: str):
    """Something PIL.Image.open accepts: the JPEG path, or the bytes from the pack."""
    return io.BytesIO(read_bytes(index_dir, ref)) if is_packed(ref) else index_dir / ref


def qwen_item(index_dir: Path, ref: str):
    """A frame for the Qwen embedder: a file path, or a loaded PIL image from the pack."""
    if not is_packed(ref):
        return str(index_dir / ref)
    from PIL import Image

    with Image.open(io.BytesIO(read_bytes(index_dir, ref))) as im:
        return im.convert("RGB")


def usage_note(frame_store: str, folder: Path) -> str | None:
    """A warning for one-file-per-frame storage, with the drive's block size when it is large."""
    if frame_store != "files":
        return None
    note = (f"frame_store = \"files\" keeps one JPEG per frame: about {FRAMES_PER_HOUR:,} files and "
            f"{FRAMES_PER_HOUR * FRAME_KB / 1000:.0f} MB per hour of footage.")
    try:
        block = os.statvfs(folder).f_frsize
    except (AttributeError, OSError):  # Windows, or the folder is gone
        block = 0
    if block >= 64 * 1024:
        note += (f" This drive uses {block // 1024} KB blocks, so each frame takes {block // 1024} KB: about "
                 f"{FRAMES_PER_HOUR * block / 1e6:.0f} MB per hour.")
    return note + ' frame_store = "packed" (the default) keeps one file per video.'
