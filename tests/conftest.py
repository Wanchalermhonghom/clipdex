"""Shared fixtures: synthetic videos made with ffmpeg and deterministic fake models."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from clipdex import fts, speech
from clipdex.config import Config
from clipdex.indexer import Models

COLORS = {
    "red": ((220, 30, 30), ["red", "rot", "rote", "roten"]),
    "green": ((30, 200, 30), ["green", "grün", "grüne"]),
    "blue": ((30, 30, 220), ["blue", "blau", "blaue"]),
    "yellow": ((230, 220, 30), ["yellow", "gelb", "gelbe"]),
}
COLOR_NAMES = list(COLORS)
DIM = 64


def _color_of_image(path) -> int:
    with Image.open(path) as im:
        mean = np.asarray(im.convert("RGB"), dtype=np.float32).reshape(-1, 3).mean(axis=0)
    dists = [np.linalg.norm(mean - np.array(rgb)) for rgb, _ in COLORS.values()]
    return int(np.argmin(dists))


def _color_words(text: str) -> list[int]:
    words = re.findall(r"\w+", text.casefold())
    return [i for i, (_, names) in enumerate(COLORS.values()) if any(w in names for w in words)]


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


class FakeSiglip:
    """Image -> one-hot of its dominant color. Text -> one-hot of color words."""

    model_name = "fake-siglip"
    logit_scale = 10.0
    logit_bias = -8.0
    dim = DIM

    def encode_images(self, paths, batch=32):
        out = np.zeros((len(paths), DIM), np.float32)
        for i, p in enumerate(paths):
            out[i, _color_of_image(p)] = 1.0
        return out

    def encode_texts(self, texts):
        out = np.zeros((len(texts), DIM), np.float32)
        for i, t in enumerate(texts):
            idx = _color_words(t)
            if idx:
                for k in idx:
                    out[i, k] = 1.0
            else:
                out[i, DIM - 1] = 1.0  # matches nothing
            out[i] = _norm(out[i])
        return out

    def logits(self, cos):
        return cos * self.logit_scale + self.logit_bias

    @staticmethod
    def prob(logit):
        return float(1 / (1 + np.exp(-logit)))


class FakeQwen:
    """Frame sequences -> mean color. Text -> color words + hashed content words."""

    dim = DIM

    def __init__(self):
        self.calls = 0

    def embed_frame_sequences(self, sequences, batch=4):
        self.calls += 1
        out = np.zeros((len(sequences), DIM), np.float32)
        for i, seq in enumerate(sequences):
            for p in seq:
                out[i, _color_of_image(p)] += 1.0
            out[i] = _norm(out[i])
        return out

    def embed_texts(self, texts, instruction=None, batch=16):
        self.calls += 1
        out = np.zeros((len(texts), DIM), np.float32)
        for i, t in enumerate(texts):
            for k in _color_words(t):
                out[i, k] += 1.0
            _, terms = fts.parse(t)
            for term in terms:
                h = int(hashlib.md5(fts.fold(term).encode()).hexdigest(), 16)
                out[i, 4 + h % (DIM - 5)] += 1.0
            if not out[i].any():
                out[i, DIM - 1] = 1.0
            out[i] = _norm(out[i])
        return out


class FakeModels(Models):
    def __init__(self, cfg: Config, transcripts: dict[str, list[dict]] | None = None):
        super().__init__(cfg)
        self._siglip = FakeSiglip()
        self._qwen = FakeQwen()
        self.transcripts = transcripts or {}
        self.transcribed: list[str] = []

    def asr_enabled(self):
        return self.cfg.asr_backend != "none"

    def transcribe(self, audio, duration, relpath=""):
        self.transcribed.append(relpath)
        segs = []
        for s in self.transcripts.get(relpath, []):
            words = [speech.Word(w, a, b) for w, a, b in s.get("words", [])]
            segs.append(speech.Segment(s["start"], s["end"], s["text"], words))
        lang = "de" if segs else None
        return speech.Transcript(language=lang, speech_seconds=sum(s.end - s.start for s in segs), segments=segs)


def make_color_video(path: Path, parts: list[tuple[str, float]], audio: bool = True, size="320x180") -> None:
    """Concatenate solid color parts, e.g. [("red", 6), ("blue", 6)]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    inputs, labels = [], []
    for i, (color, seconds) in enumerate(parts):
        r, g, b = COLORS[color][0]
        rgb = f"0x{r:02x}{g:02x}{b:02x}"
        inputs += ["-f", "lavfi", "-i", f"color=c={rgb}:s={size}:r=25:d={seconds}"]
        labels.append(f"[{i}:v]")
    total = sum(s for _, s in parts)
    graph = "".join(labels) + f"concat=n={len(parts)}:v=1:a=0[v]"
    cmd = ["ffmpeg", "-v", "error", "-y", *inputs]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=330:duration={total}"]
    cmd += ["-filter_complex", graph, "-map", "[v]"]
    if audio:
        cmd += ["-map", f"{len(parts)}:a", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-g", "25", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)


TRANSCRIPTS = {
    "A/red_blue.mp4": [
        {"start": 1.0, "end": 4.0, "text": "Heute mache ich einen Kaffee in der Küche.",
         "words": [["Heute", 1.0, 1.3], ["mache", 1.3, 1.6], ["ich", 1.6, 1.8], ["einen", 1.8, 2.0],
                   ["Kaffee", 2.2, 2.6], ["in", 2.6, 2.7], ["der", 2.7, 2.8], ["Küche.", 2.8, 3.4]]},
        {"start": 7.0, "end": 10.0, "text": "Jetzt zeige ich euch Claude Code am Laptop.",
         "words": [["Jetzt", 7.0, 7.3], ["zeige", 7.3, 7.6], ["ich", 7.6, 7.8], ["euch", 7.8, 8.0],
                   ["Claude", 8.4, 8.8], ["Code", 8.8, 9.1], ["am", 9.1, 9.3], ["Laptop.", 9.3, 9.9]]},
    ],
    "B/sub/yellow_red.mkv": [
        {"start": 0.5, "end": 3.0, "text": "This is the morning routine in Bangkok.",
         "words": [["This", 0.5, 0.7], ["is", 0.7, 0.8], ["the", 0.8, 0.9], ["morning", 0.9, 1.4],
                   ["routine", 1.4, 1.9], ["in", 1.9, 2.0], ["Bangkok.", 2.0, 2.8]]},
    ],
}


@pytest.fixture(scope="session")
def video_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("videos")
    make_color_video(root / "A" / "red_blue.mp4", [("red", 6), ("blue", 6)])
    make_color_video(root / "B" / "green.mov", [("green", 8)], audio=False)
    make_color_video(root / "B" / "sub" / "yellow_red.mkv", [("yellow", 5), ("red", 5)])
    return root


@pytest.fixture()
def library(tmp_path, video_dir) -> Path:
    """A fresh copy of the video library plus files that must be ignored."""
    import shutil

    root = tmp_path / "lib"
    shutil.copytree(video_dir, root)
    (root / "._red_blue.mp4").write_bytes(b"appledouble")
    (root / ".hidden").mkdir()
    shutil.copy(root / "B" / "green.mov", root / ".hidden" / "green.mov")
    (root / "CacheClip").mkdir()
    shutil.copy(root / "B" / "green.mov", root / "CacheClip" / "cache.mov")
    (root / "notes.txt").write_text("not a video")
    return root


@pytest.fixture()
def cfg() -> Config:
    return Config(frame_interval=2.0, frame_max_side=128, scene_max_seconds=10.0)
