"""Time formatting helpers."""

from __future__ import annotations


def tc(seconds: float | None) -> str:
    """Format seconds as HH:MM:SS.mmm (time from the start of the file)."""
    if seconds is None:
        return ""
    ms = int(round(max(0.0, float(seconds)) * 1000))
    h, rest = divmod(ms, 3_600_000)
    m, rest = divmod(rest, 60_000)
    s, ms = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def short(seconds: float | None) -> str:
    """Format seconds as M:SS or H:MM:SS for logs."""
    if seconds is None:
        return "?"
    total = int(round(max(0.0, float(seconds))))
    h, rest = divmod(total, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
