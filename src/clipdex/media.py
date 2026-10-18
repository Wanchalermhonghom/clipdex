"""ffprobe / ffmpeg helpers: metadata, frame sampling and audio extraction."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

AUDIO_RATE = 16_000
# Set by the indexer on Ctrl-C so the prefetch thread starts no new ffmpeg process.
STOP = threading.Event()
_PTS_RE = re.compile(r"\bn:\s*(\d+)\b.*?\bpts_time:\s*(-?[0-9.]+|nan)")


class MediaError(RuntimeError):
    pass


@dataclass
class MediaInfo:
    duration: float
    width: int
    height: int
    fps: float
    vcodec: str
    has_video: bool
    has_audio: bool


@dataclass
class Frame:
    index: int
    t: float
    path: Path


def ensure_tools() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise MediaError(
            f"Missing {' and '.join(missing)}. Install it, for example with 'brew install ffmpeg'."
        )


def _rate(value: str | None) -> float:
    if not value or value in ("0/0", "N/A"):
        return 0.0
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            return float(num) / float(den) if float(den) else 0.0
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _rotation(stream: dict) -> int:
    for side in stream.get("side_data_list", []) or []:
        if "rotation" in side:
            try:
                return int(round(float(side["rotation"])))
            except (TypeError, ValueError):
                pass
    try:
        return int(stream.get("tags", {}).get("rotate", 0))
    except (TypeError, ValueError):
        return 0


def probe(path: Path, timeout: float = 120.0) -> MediaInfo:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True, timeout=timeout).stdout
    except subprocess.CalledProcessError as exc:
        raise MediaError(f"ffprobe failed: {exc.stderr.decode(errors='replace').strip()[-400:]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError("ffprobe timed out") from exc
    data = json.loads(out or b"{}")
    streams = data.get("streams", [])
    video = next(
        (s for s in streams
         if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")),
        None,
    )
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = 0.0
    for candidate in (data.get("format", {}).get("duration"), (video or {}).get("duration")):
        try:
            duration = float(candidate)
            if duration > 0:
                break
        except (TypeError, ValueError):
            continue

    width = int((video or {}).get("width") or 0)
    height = int((video or {}).get("height") or 0)
    if video is not None and abs(_rotation(video)) % 180 == 90:
        width, height = height, width
    fps = _rate((video or {}).get("avg_frame_rate")) or _rate((video or {}).get("r_frame_rate"))
    return MediaInfo(
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        vcodec=str((video or {}).get("codec_name") or ""),
        has_video=video is not None,
        has_audio=audio is not None,
    )


def _frame_cmd(src: Path, out_dir: Path, interval: float, max_side: int, quality: int,
               keyframes_only: bool) -> list[str]:
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "info", "-y"]
    if keyframes_only:
        cmd += ["-skip_frame", "nokey"]
    elif sys.platform == "darwin":
        cmd += ["-hwaccel", "videotoolbox"]
    # One frame per time bucket of `interval` seconds. Buckets do not drift, and the tiny
    # epsilon absorbs float rounding of timestamps (e.g. 4.023 - 2.023 < 2).
    bucket = f"floor((t+0.002)/{interval:g})"
    prev_bucket = f"floor((prev_selected_t+0.002)/{interval:g})"
    vf = (
        f"select='isnan(prev_selected_t)+gt({bucket},{prev_bucket})',"
        "showinfo,"
        f"scale=w={max_side}:h={max_side}:force_original_aspect_ratio=decrease:force_divisible_by=2"
    )
    cmd += [
        "-i", str(src),
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", vf,
        "-fps_mode", "vfr",
        "-q:v", str(quality),
        "-start_number", "0",
        str(out_dir / "%06d.jpg"),
    ]
    return cmd


def _run_frames(cmd: list[str], out_dir: Path, timeout: float) -> list[Frame]:
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise MediaError("ffmpeg timed out while sampling frames") from exc
    stderr = proc.stderr.decode(errors="replace")
    if proc.returncode != 0:
        if STOP.is_set() or _interrupted(proc.returncode, stderr):
            raise KeyboardInterrupt
        raise MediaError(f"ffmpeg failed while sampling frames: {stderr.strip()[-600:]}")
    times: list[float] = []
    for line in stderr.splitlines():
        if "Parsed_showinfo" not in line:
            continue
        match = _PTS_RE.search(line)
        if match:
            value = match.group(2)
            times.append(float("nan") if value == "nan" else float(value))
    files = sorted(out_dir.glob("*.jpg"))
    count = min(len(files), len(times))
    for extra in files[count:]:
        extra.unlink(missing_ok=True)
    frames: list[Frame] = []
    last_t = 0.0
    for i in range(count):
        t = times[i]
        if t != t:  # nan: keep order, reuse previous time
            t = last_t
        t = max(0.0, t)
        frames.append(Frame(index=i, t=t, path=files[i]))
        last_t = t
    return frames


def _interrupted(returncode: int, stderr: str) -> bool:
    """ffmpeg stopped by Ctrl-C (SIGINT reaches the whole process group)."""
    return returncode in (-2, 130) or "received signal 2" in stderr or "Exiting normally, received signal" in stderr


def max_gap(frames: list[Frame], duration: float) -> float:
    if not frames:
        return duration
    points = [0.0] + [f.t for f in frames] + [max(duration, frames[-1].t)]
    return max(b - a for a, b in zip(points, points[1:]))


def extract_frames(src: Path, out_dir: Path, duration: float, interval: float = 2.0,
                   max_side: int = 512, quality: int = 4, keyframes_first: bool = True) -> list[Frame]:
    """Sample about one frame per `interval` seconds into out_dir as JPEG.

    Keyframe-only decoding is much faster. If keyframes are too far apart, the
    whole stream is decoded instead.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    timeout = max(600.0, duration * 3.0)
    # Camera files have a keyframe every 0.5-2 s, so keyframe sampling stays close to the
    # interval. Screen recordings and stills can have 10 s+ between keyframes: decode fully.
    allowed_gap = 2.0 * interval + 0.25

    def clear() -> None:
        for f in out_dir.glob("*.jpg"):
            f.unlink(missing_ok=True)

    frames: list[Frame] = []
    if keyframes_first:
        clear()
        try:
            frames = _run_frames(
                _frame_cmd(src, out_dir, interval, max_side, quality, True), out_dir, timeout)
        except MediaError:
            frames = []
        if frames and max_gap(frames, duration) <= allowed_gap:
            return frames
    if STOP.is_set():
        raise KeyboardInterrupt
    clear()
    frames = _run_frames(_frame_cmd(src, out_dir, interval, max_side, quality, False), out_dir, timeout)
    if not frames:
        raise MediaError("ffmpeg produced no frames")
    return frames


def extract_audio(src: Path, duration: float) -> np.ndarray:
    """Decode the first audio stream as 16 kHz mono float32."""
    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
        "-i", str(src), "-map", "0:a:0", "-vn", "-sn", "-dn",
        "-ac", "1", "-ar", str(AUDIO_RATE), "-f", "s16le", "-acodec", "pcm_s16le", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=max(600.0, duration * 2.0))
    except subprocess.TimeoutExpired as exc:
        raise MediaError("ffmpeg timed out while decoding audio") from exc
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace")
        if STOP.is_set() or _interrupted(proc.returncode, stderr):
            raise KeyboardInterrupt
        raise MediaError(f"ffmpeg failed while decoding audio: {stderr[-400:]}")
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0
