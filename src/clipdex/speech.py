"""Speech: voice activity detection, Whisper transcription and transcript chunking."""

from __future__ import annotations

import logging
import platform
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .media import AUDIO_RATE

log = logging.getLogger(__name__)

# Known Whisper hallucinations on silence/music (subtitle credits from its training data).
_HALLUCINATION_RE = re.compile(
    r"untertitel(ung)?\s+(im\s+auftrag\s+)?(des|der|von)\s+(zdf|wdr|ard|swr|br|ndr|mdr|funk)"
    r"|amara\.org|subtitles\s+by\s+the\s+amara|www\.zeoranger\.co\.uk",
    re.IGNORECASE,
)


@dataclass
class Word:
    word: str
    start: float
    end: float


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 1.0

    def to_row(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "words": [[w.word, round(w.start, 3), round(w.end, 3)] for w in self.words],
        }


@dataclass
class Transcript:
    language: str | None
    speech_seconds: float
    segments: list[Segment]


# --- voice activity detection --------------------------------------------------------

_vad_model = None


def detect_speech(audio: np.ndarray, threshold: float = 0.5) -> list[tuple[float, float]]:
    """Return speech regions in seconds using Silero VAD."""
    global _vad_model
    if audio.size < AUDIO_RATE // 4:
        return []
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    if _vad_model is None:
        _vad_model = load_silero_vad()
    with torch.no_grad():
        stamps = get_speech_timestamps(
            torch.from_numpy(np.ascontiguousarray(audio)), _vad_model,
            threshold=threshold, sampling_rate=AUDIO_RATE, return_seconds=True)
    return [(float(s["start"]), float(s["end"])) for s in stamps]


def merge_regions(regions: Sequence[tuple[float, float]], duration: float, gap: float = 2.0,
                  pad: float = 0.4) -> list[tuple[float, float]]:
    """Pad speech regions and merge regions that are closer than `gap` seconds."""
    merged: list[list[float]] = []
    for start, end in sorted(regions):
        start = max(0.0, start - pad)
        end = min(duration, end + pad) if duration > 0 else end + pad
        if merged and start - merged[-1][1] <= gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged if b > a]


def overlap(seg_start: float, seg_end: float, regions: Sequence[tuple[float, float]]) -> float:
    """Fraction of [seg_start, seg_end] that lies inside the regions."""
    length = seg_end - seg_start
    if length <= 0:
        return 0.0
    covered = sum(max(0.0, min(seg_end, b) - max(seg_start, a)) for a, b in regions)
    return min(1.0, covered / length)


# --- Whisper backends ---------------------------------------------------------------------

class WhisperBackend(Protocol):
    """A speech backend. It can also have `detect_language(audio) -> {code: probability}`.

    Without detect_language, clipdex detects the language with a transcription pass, which costs more.
    """

    name: str

    def transcribe(self, audio: np.ndarray, clips: list[tuple[float, float]],
                   language: str | None) -> tuple[str | None, list[Segment]]: ...


def _clip_list(clips: list[tuple[float, float]]) -> list[float]:
    flat: list[float] = []
    for a, b in clips:
        flat += [round(a, 3), round(b, 3)]
    return flat


class MLXWhisper:
    name = "mlx"
    default_model = "mlx-community/whisper-large-v3-turbo"

    def __init__(self, model: str = ""):
        import mlx_whisper  # noqa: F401  (fails early if not installed)

        self.model = model or self.default_model

    def transcribe(self, audio, clips, language):
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model,
            language=language,
            word_timestamps=True,
            condition_on_previous_text=False,
            clip_timestamps=_clip_list(clips) if clips else "0",
            hallucination_silence_threshold=2.0,
            verbose=None,
        )
        segments = []
        for s in result.get("segments", []):
            words = [Word(str(w["word"]), float(w["start"]), float(w["end"]))
                     for w in s.get("words", []) or []]
            segments.append(Segment(
                start=float(s["start"]), end=float(s["end"]), text=str(s["text"]).strip(),
                words=words, no_speech_prob=float(s.get("no_speech_prob", 0.0)),
                avg_logprob=float(s.get("avg_logprob", 0.0)),
                compression_ratio=float(s.get("compression_ratio", 1.0))))
        return result.get("language"), segments

    def detect_language(self, audio: np.ndarray) -> dict[str, float]:
        """Language probabilities for the first 30 s: one encoder pass, no transcription."""
        import mlx.core as mx
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.transcribe import ModelHolder

        # Same model cache and dtype as mlx_whisper.transcribe (fp16 by default).
        model = ModelHolder.get_model(self.model, mx.float16)
        mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels, padding=N_SAMPLES)
        segment = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
        _, probs = model.detect_language(segment)
        return {str(k): float(v) for k, v in probs.items()}


class FasterWhisper:
    name = "faster-whisper"
    default_model = "large-v3-turbo"

    def __init__(self, model: str = "", device: str = "auto"):
        from faster_whisper import WhisperModel

        from .devices import pick_device

        self.model_name = model or self.default_model
        # faster-whisper (CTranslate2) runs on CUDA or CPU only, not on Apple MPS.
        use_cuda = pick_device(device) == "cuda"
        self.model = WhisperModel(self.model_name, device="cuda" if use_cuda else "cpu",
                                  compute_type="float16" if use_cuda else "int8")

    def transcribe(self, audio, clips, language):
        seg_iter, info = self.model.transcribe(
            audio,
            language=language,
            word_timestamps=True,
            condition_on_previous_text=False,
            clip_timestamps=_clip_list(clips) if clips else "0",
            vad_filter=False,
        )
        segments = []
        for s in seg_iter:
            words = [Word(str(w.word), float(w.start), float(w.end)) for w in (s.words or [])]
            segments.append(Segment(
                start=float(s.start), end=float(s.end), text=str(s.text).strip(), words=words,
                no_speech_prob=float(s.no_speech_prob), avg_logprob=float(s.avg_logprob),
                compression_ratio=float(s.compression_ratio)))
        return info.language, segments

    def detect_language(self, audio: np.ndarray) -> dict[str, float]:
        """Language probabilities for the first 30 s: one encoder pass, no transcription."""
        _, _, all_probs = self.model.detect_language(audio=audio)
        return {str(code): float(p) for code, p in all_probs}


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def make_backend(name: str, model: str = "", device: str = "auto") -> WhisperBackend | None:
    if name == "none":
        return None
    if name == "auto":
        name = "mlx" if is_apple_silicon() else "faster-whisper"
    if name == "mlx":
        return MLXWhisper(model)
    if name == "faster-whisper":
        return FasterWhisper(model, device)
    raise ValueError(f"Unknown asr_backend '{name}' (use auto, mlx, faster-whisper or none)")


# --- transcription ------------------------------------------------------------------------

def _clean(segments: list[Segment], regions: list[tuple[float, float]]) -> list[Segment]:
    kept = []
    for s in segments:
        if not s.text or s.end <= s.start:
            continue
        if s.compression_ratio > 2.4:  # repetition loop
            continue
        if s.no_speech_prob > 0.6 and s.avg_logprob < -1.0:
            continue
        if regions and overlap(s.start, s.end, regions) < 0.3:
            continue
        if _HALLUCINATION_RE.search(s.text):
            continue
        kept.append(s)
    return kept


def detect_language_window(regions: list[tuple[float, float]], window: float = 30.0) -> tuple[float, float]:
    """Pick the 30 s window that starts at the first longer speech region."""
    for a, b in regions:
        if b - a >= 3.0:
            return a, a + window
    a = regions[0][0] if regions else 0.0
    return a, a + window


def choose_language(backend: WhisperBackend, audio: np.ndarray, allowed: Sequence[str] = ()) -> str | None:
    """Detect the language of `audio`. With `allowed`, pick only from these codes."""
    allowed = [a.strip().lower() for a in allowed if a.strip()]
    detect = getattr(backend, "detect_language", None)
    probs: dict[str, float] | None = None
    if callable(detect):
        try:
            probs = detect(audio)
        except Exception as exc:  # noqa: BLE001 - an API change must not stop the indexer
            log.warning("Language detection failed (%s). Using a transcription pass instead.", exc)
    if probs:
        candidates = {k: v for k, v in probs.items() if k in allowed} if allowed else probs
        if candidates:
            return max(candidates, key=candidates.__getitem__)
        return allowed[0]
    lang, _ = backend.transcribe(audio, [], None)
    if allowed and lang not in allowed:
        return allowed[0]  # the first allowed language is the fallback
    return lang


def transcribe(audio: np.ndarray, backend: WhisperBackend, duration: float, language: str = "auto",
               vad_threshold: float = 0.5, min_speech_seconds: float = 1.0,
               languages: Sequence[str] = ()) -> Transcript:
    raw_regions = detect_speech(audio, vad_threshold)
    speech_seconds = float(sum(b - a for a, b in raw_regions))
    if speech_seconds < min_speech_seconds:
        return Transcript(language=None, speech_seconds=speech_seconds, segments=[])
    clips = merge_regions(raw_regions, duration or len(audio) / AUDIO_RATE)

    lang: str | None = None if language in ("", "auto", None) else language
    if lang is None:
        # Whisper detects the language on the first 30 s of its input. Footage often starts
        # with silence or B-roll, so detect on a window that starts where speech starts.
        a, b = detect_language_window(clips)
        lang = choose_language(backend, audio[int(a * AUDIO_RATE): int(b * AUDIO_RATE)], languages)
    # Transcribe each speech clip on its own and shift the times back. Passing all clips at once
    # does not skip the silence between them in mlx-whisper 0.4 (it keeps decoding from the
    # end of one clip to the next).
    segments: list[Segment] = []
    detected = None
    for a, b in clips:
        piece = audio[int(a * AUDIO_RATE): int(b * AUDIO_RATE)]
        if piece.size < AUDIO_RATE // 10:
            continue
        found_lang, found = backend.transcribe(piece, [], lang)
        detected = detected or found_lang
        for seg in found:
            seg.start += a
            seg.end += a
            for w in seg.words:
                w.start += a
                w.end += a
            segments.append(seg)
    return Transcript(language=lang or detected, speech_seconds=speech_seconds,
                      segments=_clean(segments, raw_regions))


def make_chunks(segments: Sequence[Segment | dict], max_seconds: float = 30.0,
                max_gap: float = 8.0) -> list[tuple[float, float, str]]:
    """Group consecutive segments into chunks of up to `max_seconds` for semantic search."""
    chunks: list[tuple[float, float, str]] = []
    cur: list[tuple[float, float, str]] = []

    def flush() -> None:
        if cur:
            chunks.append((cur[0][0], cur[-1][1], " ".join(t for _, _, t in cur).strip()))
            cur.clear()

    for s in segments:
        start, end, text = (s["start"], s["end"], s["text"]) if isinstance(s, dict) else (s.start, s.end, s.text)
        if cur and (end - cur[0][0] > max_seconds or start - cur[-1][1] > max_gap):
            flush()
        cur.append((start, end, text))
    flush()
    return [c for c in chunks if c[2]]
