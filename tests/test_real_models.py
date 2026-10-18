"""Integration test with the real models on real sample media (slow, downloads ~6 GB once).

Run with:  pytest -m models -s
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from clipdex.config import Config
from clipdex.indexer import Indexer
from clipdex.search import SearchEngine
from clipdex.speech import is_apple_silicon

pytestmark = pytest.mark.models

SAMPLES = [
    ("huggingface/documentation-images", "bee.jpg"),
    ("merve/coco", "val2017/000000039769.jpg"),  # two cats on a couch
    ("merve/coco", "val2017/000000000724.jpg"),  # stop sign
    ("merve/coco", "val2017/000000000776.jpg"),  # teddy bears
    ("merve/coco", "val2017/000000000632.jpg"),  # bedroom
    ("raushan-testing-hf/videos-test", "sample_demo_1.mp4"),  # child reading a book
    ("Narsil/asr_dummy", "mlk.flac"),  # "I have a dream ..."
    ("Narsil/asr_dummy", "i-know-kung-fu.mp3"),
]


def _ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def real_library(tmp_path_factory) -> Path:
    from huggingface_hub import hf_hub_download

    src = {Path(f).name: Path(hf_hub_download(repo, f, repo_type="dataset")) for repo, f in SAMPLES}
    root = tmp_path_factory.mktemp("real")
    (root / "talk").mkdir()
    (root / "broll").mkdir()
    still = ["-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25"]
    _ff("-loop", "1", "-t", "6", "-i", str(src["bee.jpg"]), *still, str(root / "broll" / "C0001.mp4"))
    _ff("-loop", "1", "-t", "5", "-i", str(src["000000000724.jpg"]), *still, str(root / "broll" / "C0002.mp4"))
    # cats with spoken audio
    _ff("-loop", "1", "-t", "6", "-i", str(src["000000039769.jpg"]), "-i", str(src["i-know-kung-fu.mp3"]),
        *still, "-c:a", "aac", "-shortest", str(root / "talk" / "C0003.mp4"))
    # teddy bears then bedroom, silent
    _ff("-loop", "1", "-t", "6", "-i", str(src["000000000776.jpg"]),
        "-loop", "1", "-t", "6", "-i", str(src["000000000632.jpg"]),
        "-filter_complex",
        "[0:v]scale=1280:720,setsar=1[a];[1:v]scale=1280:720,setsar=1[b];[a][b]concat=n=2:v=1[v]",
        "-map", "[v]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25", str(root / "broll" / "C0004.mp4"))
    # real video (child reading) with a speech track, 13 s
    _ff("-stream_loop", "1", "-i", str(src["sample_demo_1.mp4"]), "-i", str(src["mlk.flac"]),
        "-map", "0:v", "-map", "1:a", "-t", "13", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        str(root / "talk" / "C0005.mp4"))
    return root


@pytest.fixture(scope="module")
def real_cfg() -> Config:
    return Config(
        siglip_model="google/siglip2-base-patch16-256",
        asr_model="mlx-community/whisper-tiny" if is_apple_silicon() else "tiny",
        qwen_batch=2,
    )


@pytest.fixture(scope="module")
def real_engine(real_library, real_cfg):
    idx = Indexer(real_library, real_cfg, log=print)
    report = idx.run()
    assert report.failed == 0 and report.processed == 5
    return SearchEngine(idx.index_dir, real_cfg, root=real_library)


def _top(engine, query, mode):
    out = engine.search(query, mode=mode, explain=True)
    print(f"\n{mode:7s} {query!r}: ", [(r["file"], r["start"], r["end"], r["matched"]) for r in out["results"]],
          out["channels"])
    return out["results"]


@pytest.mark.parametrize("query,mode,expected", [
    ("a bee on a flower", "visual", "broll/C0001.mp4"),
    ("eine Biene auf einer Blüte", "visual", "broll/C0001.mp4"),
    ("two cats sleeping on a pink couch", "visual", "talk/C0003.mp4"),
    ("ein Stoppschild", "visual", "broll/C0002.mp4"),
    ("a child reading a book", "action", "talk/C0005.mp4"),
    ("teddy bears", "visual", "broll/C0004.mp4"),
    ('"I have a dream"', "speech", "talk/C0005.mp4"),
    ("a speech about freedom and the future of a nation", "speech", "talk/C0005.mp4"),
])
def test_queries_find_the_right_file(real_engine, query, mode, expected):
    results = _top(real_engine, query, mode)
    assert results, f"no results for {query!r}"
    assert results[0]["file"] == expected


def test_scene_split_inside_a_clip(real_engine):
    results = _top(real_engine, "a bed in a bedroom", "visual")
    assert results and results[0]["file"] == "broll/C0004.mp4" and results[0]["start"] >= 5.0


def test_short_speech_is_transcribed(real_engine):
    # whisper-tiny mishears "I know kung fu"; large-v3-turbo (the default) gets it right.
    tr = real_engine.transcript("C0003.mp4")
    assert tr["language"] == "en" and tr["segments"]


def test_transcript_language(real_engine):
    tr = real_engine.transcript("C0005.mp4")
    assert tr["language"] == "en"
    assert "dream" in " ".join(s["text"] for s in tr["segments"]).lower()


def test_gibberish_finds_little(real_engine):
    results = _top(real_engine, "qwxzz blorf vrrnk", "visual")
    assert len(results) <= 1
