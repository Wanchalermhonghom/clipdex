"""Configuration with defaults, loaded from an optional TOML file.

Order of precedence (last wins):
  1. defaults in this file
  2. <index_dir>/config.toml (if it exists)
  3. a file passed with --config
  4. explicit command line flags
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_VIDEO_EXTENSIONS = [
    ".mp4", ".mov", ".m4v", ".mkv", ".avi", ".mts", ".m2ts", ".mxf", ".webm", ".3gp", ".insv",
]

# Folder names that are never scanned. Hidden folders (".name") are always skipped.
DEFAULT_EXCLUDE_DIRS = ["CacheClip", "ProxyMedia", "Proxy", "Optimized Media", "Render Cache"]


@dataclass
class Config:
    # --- scanning -----------------------------------------------------------
    video_extensions: list[str] = field(default_factory=lambda: list(DEFAULT_VIDEO_EXTENSIONS))
    exclude_dirs: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE_DIRS))
    # fnmatch patterns matched against the path relative to the library root, e.g. "Exports/*".
    # The default skips Sony proxy clips (M4ROOT/SUB).
    exclude_globs: list[str] = field(default_factory=lambda: ["*M4ROOT/SUB/*"])

    # --- frames -------------------------------------------------------------
    frame_interval: float = 2.0  # seconds between sampled frames
    frame_max_side: int = 512  # longest side of stored frames in pixels
    frame_jpeg_quality: int = 4  # ffmpeg -q:v (2 = best, 31 = worst)
    keyframes_first: bool = True  # decode only keyframes first; falls back to full decode on big gaps

    # --- speech -------------------------------------------------------------
    asr_backend: str = "auto"  # auto | mlx | faster-whisper | none
    asr_model: str = ""  # empty = backend default (whisper large-v3-turbo)
    # auto = detect once per file (one quick detection pass), or a Whisper code like "de" / "en".
    # A fixed code skips the detection pass, but speech in other languages comes out wrong.
    language: str = "auto"
    # With language = "auto": choose only from these codes, e.g. ["de", "en"]. Empty = any language.
    languages: list[str] = field(default_factory=list)
    vad_threshold: float = 0.5
    min_speech_seconds: float = 1.0  # less detected speech than this = no transcription
    speech_chunk_seconds: float = 20.0  # length of transcript chunks for semantic speech search

    # --- models -------------------------------------------------------------
    device: str = "auto"  # auto | mps | cuda | cpu
    siglip_model: str = "google/siglip2-so400m-patch14-384"
    siglip_batch: int = 32
    qwen_enabled: bool = True
    qwen_model: str = "Qwen/Qwen3-VL-Embedding-2B"
    qwen_dtype: str = "auto"  # auto | bfloat16 | float16 | float32
    qwen_batch: int = 4

    # --- scenes (input for the action / scene embeddings) -------------------
    scene_max_seconds: float = 10.0
    scene_cut_similarity: float = 0.80  # SigLIP cosine below this vs. scene start = new scene (slow drift)
    # SigLIP cosine below this vs. the previous frame = new scene (a jump). Calibrated on iPhone
    # footage: changes scored 0.70-0.88, continuous selfie and pan shots 0.89-0.99.
    scene_step_similarity: float = 0.885
    scene_max_frames: int = 8

    # --- search -------------------------------------------------------------
    # SigLIP gives a calibrated probability per frame. A hit needs both:
    visual_min_prob: float = 0.002
    visual_rel_logit: float = 4.0  # logit must be within this of the best frame's logit
    # Qwen cosine channels: score must beat the query's own baseline (a low percentile
    # of all scores) by a margin, and an absolute floor.
    # Defaults come from a small calibration run: matches scored 0.62-0.75, unrelated
    # scenes up to 0.35 and random text up to 0.39. Tune with 'clipdex search --explain'.
    baseline_percentile: float = 20.0
    scene_floor: float = 0.40
    scene_margin: float = 0.15
    speech_sem_floor: float = 0.42
    speech_sem_margin: float = 0.15
    channel_top_k: int = 200
    rrf_k: float = 60.0
    merge_gap: float = 2.0  # hits closer than this (seconds) merge into one moment
    # A merged moment only spans hits that score close to its best hit per channel.
    span_tolerance_logit: float = 2.0  # visual (SigLIP logit)
    span_tolerance_cos: float = 0.08  # scenes and speech_sem (cosine)
    moment_max_seconds: float = 30.0  # must be >= scene_max_seconds and speech_chunk_seconds
    max_per_file: int = 3
    qwen_visual_instruction: str = "Retrieve video clips that show what the user describes."
    qwen_speech_instruction: str = (
        "Retrieve transcript passages in which the speaker talks about what the user describes."
    )

    @classmethod
    def field_names(cls) -> set[str]:
        return {f.name for f in dataclasses.fields(cls)}

    def apply(self, values: dict[str, Any], source: str = "config") -> Config:
        """Return a copy with values applied. Unknown keys raise an error."""
        known = self.field_names()
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown setting(s) in {source}: {', '.join(unknown)}")
        updated = dataclasses.replace(self)
        for key, value in values.items():
            current = getattr(updated, key)
            if isinstance(current, bool) and not isinstance(value, bool):
                raise ValueError(f"{source}: '{key}' must be true or false")
            if isinstance(current, float) and isinstance(value, int) and not isinstance(value, bool):
                value = float(value)
            if isinstance(current, list) and not isinstance(value, list):
                raise ValueError(f"{source}: '{key}' must be a list")
            setattr(updated, key, value)
        longest = max(updated.scene_max_seconds, updated.speech_chunk_seconds)
        if updated.moment_max_seconds < longest:
            raise ValueError(f"{source}: moment_max_seconds ({updated.moment_max_seconds:g}) must be at least "
                             f"scene_max_seconds and speech_chunk_seconds ({longest:g}), or hits cannot merge")
        return updated

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    # Allow an optional [clipdex] table or flat keys.
    if "clipdex" in data and isinstance(data["clipdex"], dict):
        data = data["clipdex"]
    return data


def load_config(index_dir: Path | None, extra_file: Path | None = None,
                overrides: dict[str, Any] | None = None) -> Config:
    cfg = Config()
    if index_dir is not None:
        local = index_dir / "config.toml"
        if local.exists():
            cfg = cfg.apply(load_toml(local), source=str(local))
    if extra_file is not None:
        cfg = cfg.apply(load_toml(extra_file), source=str(extra_file))
    if overrides:
        cfg = cfg.apply({k: v for k, v in overrides.items() if v is not None}, source="command line")
    return cfg
