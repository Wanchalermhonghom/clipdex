"""Unit tests for the pure helpers."""

from __future__ import annotations

import numpy as np
import pytest

from clipdex import fts, media, scenes, speech
from clipdex.config import Config, load_config
from clipdex.library import scan
from clipdex.timefmt import short, tc


def test_tc_and_short():
    assert tc(0) == "00:00:00.000"
    assert tc(3725.5) == "01:02:05.500"
    assert tc(None) == ""
    assert short(65) == "1:05"
    assert short(3725) == "1:02:05"


def test_scan_skips_hidden_cache_and_non_video(library):
    found = [f.relpath for f in scan(library, Config())]
    assert found == ["A/red_blue.mp4", "B/green.mov", "B/sub/yellow_red.mkv"]


def test_scan_exclude_globs(library):
    cfg = Config(exclude_globs=["B/sub/*"])
    assert [f.relpath for f in scan(library, cfg)] == ["A/red_blue.mp4", "B/green.mov"]


def test_config_toml_and_errors(tmp_path):
    (tmp_path / "config.toml").write_text('frame_interval = 3\nlanguage = "de"\n')
    cfg = load_config(tmp_path)
    assert cfg.frame_interval == 3.0 and cfg.language == "de"
    (tmp_path / "config.toml").write_text("not_a_setting = 1\n")
    with pytest.raises(ValueError, match="not_a_setting"):
        load_config(tmp_path)
    (tmp_path / "config.toml").write_text("qwen_enabled = 1\n")
    with pytest.raises(ValueError, match="true or false"):
        load_config(tmp_path)


def test_fts_build_and_parse():
    assert fts.build("wo rede ich über Kaffee") == ['"Kaffee"*']
    assert fts.build('"claude code" Laptop') == ['"claude code" AND "Laptop"*']
    assert fts.build("morning routine Bangkok") == [
        '"morning"* AND "routine"* AND "Bangkok"*', '"morning"* OR "routine"* OR "Bangkok"*']
    assert fts.build("ich und du") == ['"ich"* AND "und"* AND "du"', '"ich"* OR "und"* OR "du"']
    assert fts.build("  ") == []


def test_fts_word_time():
    words = [["Heute", 1.0, 1.3], ["Kaffee,", 2.2, 2.6]]
    assert fts.first_word_time(words, fts.match_terms("kaffee")) == 2.2
    assert fts.first_word_time(words, fts.match_terms("küche")) is None
    assert fts.fold("Küche") == "kuche"


def test_scene_segmentation_cut_and_max_length():
    times = [0, 2, 4, 6, 8, 10, 12, 14, 16]
    a = np.array([1, 0], np.float32)
    b = np.array([0, 1], np.float32)
    embs = np.stack([a, a, a, b, b, b, b, b, b])
    out = scenes.segment(times, embs, duration=17.5, max_seconds=8.0, cut_similarity=0.8)
    assert [(s.start, s.end) for s in out] == [(0, 6), (6, 14), (14, 17.5)]
    assert out[0].frames == [0, 1, 2]
    assert scenes.segment([], np.zeros((0, 2)), 5.0) == []
    single = scenes.segment([0.0], np.array([a]), duration=0.0)
    assert single[0].start == 0.0 and single[0].end == 1.0


def test_merge_regions_and_overlap():
    merged = speech.merge_regions([(1.0, 2.0), (3.0, 4.0), (10.0, 11.0)], duration=10.5, gap=2.0, pad=0.4)
    assert merged == [(0.6, 4.4), (9.6, 10.5)]
    assert speech.overlap(0.0, 2.0, [(1.0, 3.0)]) == 0.5
    assert speech.overlap(5.0, 5.0, [(1.0, 3.0)]) == 0.0


def test_make_chunks():
    segs = [{"start": 0, "end": 10, "text": "a"}, {"start": 11, "end": 20, "text": "b"},
            {"start": 21, "end": 35, "text": "c"}, {"start": 60, "end": 62, "text": "d"}]
    assert speech.make_chunks(segs, max_seconds=30) == [(0, 20, "a b"), (21, 35, "c"), (60, 62, "d")]


def test_clean_drops_hallucinations_and_silence():
    regions = [(0.0, 5.0)]
    segs = [
        speech.Segment(0.5, 3.0, "Hallo zusammen"),
        speech.Segment(10.0, 14.0, "outside of any speech region"),
        speech.Segment(1.0, 2.0, "Untertitel im Auftrag des ZDF, 2017"),
        speech.Segment(1.0, 2.0, "la la la la", compression_ratio=3.0),
        speech.Segment(2.0, 2.0, "zero length"),
    ]
    assert [s.text for s in speech._clean(segs, regions)] == ["Hallo zusammen"]


class _Backend:
    name = "fake"

    def __init__(self):
        self.calls = []

    def transcribe(self, audio, clips, language):
        self.calls.append((len(audio), list(clips), language))
        return "de", [speech.Segment(0.5, 2.0, "Moin", [speech.Word("Moin", 0.6, 1.0)])]


def test_transcribe_detects_language_on_first_speech(monkeypatch):
    monkeypatch.setattr(speech, "detect_speech", lambda audio, threshold: [(12.0, 20.0), (30.0, 31.0)])
    backend = _Backend()
    audio = np.zeros(40 * media.AUDIO_RATE, np.float32)
    tr = speech.transcribe(audio, backend, duration=40.0)
    # first call: language window from the padded first speech region (11.6 s) to the end (40 s)
    assert backend.calls[0][2] is None
    assert backend.calls[0][0] == 40 * media.AUDIO_RATE - int(11.6 * media.AUDIO_RATE)
    # then one call per speech clip, with the detected language fixed
    assert [c[0] for c in backend.calls[1:]] == [int(20.4 * 16000) - int(11.6 * 16000),
                                                  int(31.4 * 16000) - int(29.6 * 16000)]
    assert all(c[2] == "de" and c[1] == [] for c in backend.calls[1:])
    # segment and word times are shifted back to file time (clips start at 11.6 s and 29.6 s)
    assert tr.language == "de" and tr.speech_seconds == 9.0
    assert [round(x.start, 2) for x in tr.segments] == [12.1, 30.1]
    assert tr.segments[1].words[0].start == pytest.approx(29.6 + 0.6)


def test_transcribe_skips_files_without_speech(monkeypatch):
    monkeypatch.setattr(speech, "detect_speech", lambda audio, threshold: [(1.0, 1.5)])
    backend = _Backend()
    tr = speech.transcribe(np.zeros(16000 * 5, np.float32), backend, duration=5.0, min_speech_seconds=1.0)
    assert tr.segments == [] and backend.calls == []


def test_forced_language_skips_detection(monkeypatch):
    monkeypatch.setattr(speech, "detect_speech", lambda audio, threshold: [(0.0, 5.0)])
    backend = _Backend()
    speech.transcribe(np.zeros(16000 * 6, np.float32), backend, duration=6.0, language="en")
    assert len(backend.calls) == 1 and backend.calls[0][2] == "en"


class _DetectBackend(_Backend):
    def __init__(self, probs=None, fail=False):
        super().__init__()
        self.probs = probs or {"nl": 0.5, "de": 0.3, "en": 0.2}
        self.fail = fail
        self.detected = []

    def detect_language(self, audio):
        self.detected.append(len(audio))
        if self.fail:
            raise RuntimeError("API changed")
        return self.probs


def test_language_detection_pass_does_not_transcribe(monkeypatch):
    monkeypatch.setattr(speech, "detect_speech", lambda audio, threshold: [(12.0, 20.0), (30.0, 31.0)])
    audio = np.zeros(40 * media.AUDIO_RATE, np.float32)
    backend = _DetectBackend()
    tr = speech.transcribe(audio, backend, duration=40.0)
    assert backend.detected == [40 * media.AUDIO_RATE - int(11.6 * media.AUDIO_RATE)]  # window at first speech
    assert len(backend.calls) == 2 and all(c[2] == "nl" for c in backend.calls)  # one call per clip, no extra
    assert tr.language == "nl"
    backend = _DetectBackend()
    tr = speech.transcribe(audio, backend, duration=40.0, languages=["de", "en"])
    assert tr.language == "de" and all(c[2] == "de" for c in backend.calls)


def test_choose_language_fallbacks():
    audio = np.zeros(16000, np.float32)
    # detection raises: fall back to a transcription pass ("de" from _Backend)
    backend = _DetectBackend(fail=True)
    assert speech.choose_language(backend, audio) == "de" and backend.calls[0][2] is None
    # the transcription pass finds a language that is not allowed: use the first allowed one
    assert speech.choose_language(_Backend(), audio, ["en", "fr"]) == "en"
    assert speech.choose_language(_DetectBackend({"ja": 0.9}), audio, [" EN "]) == "en"


def test_config_language_reaches_transcription(monkeypatch, tmp_path):
    from clipdex.config import load_config
    from clipdex.indexer import Models

    monkeypatch.setattr(speech, "detect_speech", lambda audio, threshold: [(0.0, 5.0)])
    (tmp_path / "config.toml").write_text('language = "de"\n')
    models = Models(load_config(tmp_path))
    backend = _DetectBackend()
    models._asr, models._asr_loaded = backend, True
    models.transcribe(np.zeros(16000 * 6, np.float32), 6.0)
    assert backend.detected == [] and [c[2] for c in backend.calls] == ["de"]
    (tmp_path / "config.toml").write_text('languages = ["de", "en"]\n')
    models = Models(load_config(tmp_path))
    models._asr, models._asr_loaded = _DetectBackend(), True
    assert models.transcribe(np.zeros(16000 * 6, np.float32), 6.0).language == "de"


def test_example_toml_lists_every_setting():
    import re
    from pathlib import Path

    from clipdex.config import Config

    text = (Path(__file__).parent.parent / "clipdex.example.toml").read_text()
    keys = set(re.findall(r"^# ([a-z_]+) = ", text, flags=re.M))
    assert keys == Config.field_names()


def test_extract_frames_short_gop(video_dir, tmp_path):
    src = video_dir / "A" / "red_blue.mp4"
    info = media.probe(src)
    assert info.has_audio and info.has_video and abs(info.duration - 12.0) < 0.2
    frames = media.extract_frames(src, tmp_path / "f", info.duration, interval=2.0, max_side=128)
    assert [round(f.t) for f in frames] == [0, 2, 4, 6, 8, 10]
    assert all(f.path.exists() for f in frames)


def test_extract_frames_long_gop_falls_back(tmp_path):
    import subprocess

    src = tmp_path / "long_gop.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25:duration=20",
                    "-c:v", "libx264", "-g", "500", "-keyint_min", "500", "-sc_threshold", "0",
                    "-pix_fmt", "yuv420p", str(src)], check=True)
    frames = media.extract_frames(src, tmp_path / "f", 20.0, interval=2.0, max_side=128)
    assert len(frames) == 10 and media.max_gap(frames, 20.0) <= 2.01


def test_extract_audio(video_dir):
    audio = media.extract_audio(video_dir / "A" / "red_blue.mp4", 12.0)
    assert audio.dtype == np.float32 and abs(len(audio) / media.AUDIO_RATE - 12.0) < 0.2
    assert 0.05 < float(np.abs(audio).max()) <= 1.0


def _unit(deg):
    import math

    r = math.radians(deg)
    return np.array([math.cos(r), math.sin(r)], np.float32)


def test_scene_cut_on_jump_and_on_slow_drift():
    # jump: 0 -> 0 -> 30 degrees (cos 0.866 < 0.885 step), anchor rule alone would not cut (0.866 > 0.80)
    embs = np.stack([_unit(0), _unit(0), _unit(30)])
    out = scenes.segment([0, 2, 4], embs, duration=6.0, max_seconds=100, cut_similarity=0.80,
                         step_similarity=0.885)
    assert [(s.start, s.end) for s in out] == [(0, 4), (4, 6.0)]
    # slow drift: 15 degrees per step (cos 0.966 > step), 45 degrees from the start (cos 0.707 < 0.80)
    embs = np.stack([_unit(0), _unit(15), _unit(30), _unit(45), _unit(60)])
    out = scenes.segment([0, 2, 4, 6, 8], embs, duration=10.0, max_seconds=100, cut_similarity=0.80,
                         step_similarity=0.885)
    assert [s.start for s in out] == [0, 6]
