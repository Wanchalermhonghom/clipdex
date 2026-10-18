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


# --- German speech --------------------------------------------------------------------------------
# Two sentences from VoxPopuli (CC0, European Parliament plenary sessions), in tests/data:
#   de_kinderministerium.flac  "Ich kenne kein Land, in dem es ein eigenes Kinderministerium gibt."
#   de_landwirtschaft.flac     "Es gibt eine Landwirtschaft, die sich an der Natur orientiert, und eine,
#                               die das nicht tut."

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def german_engine(tmp_path_factory):
    from huggingface_hub import hf_hub_download

    root = tmp_path_factory.mktemp("german")
    still = ["-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25", "-c:a", "aac", "-shortest"]
    for image, audio, name in (("val2017/000000039769.jpg", "de_kinderministerium.flac", "kinder.mp4"),
                               ("val2017/000000000632.jpg", "de_landwirtschaft.flac", "landwirtschaft.mp4")):
        img = hf_hub_download("merve/coco", image, repo_type="dataset")
        _ff("-loop", "1", "-i", img, "-i", str(DATA / audio), *still, str(root / name))
    # Whisper large-v3-turbo (the default): whisper-tiny is too weak for German words.
    cfg = Config(siglip_model="google/siglip2-base-patch16-256", qwen_batch=2, languages=["de", "en"])
    idx = Indexer(root, cfg, log=print)
    assert idx.run().failed == 0
    return SearchEngine(idx.index_dir, cfg, root=root)


def test_german_language_and_transcript(german_engine):
    for name, word in (("kinder.mp4", "kinderministerium"), ("landwirtschaft.mp4", "landwirtschaft")):
        tr = german_engine.transcript(name)
        text = " ".join(s["text"] for s in tr["segments"]).lower()
        print(f"\n{name}: {tr['language']} {text!r}")
        assert tr["language"] == "de" and word in text


@pytest.mark.parametrize("query,expected,index", [
    ("Kinderministerium", "kinder.mp4", "word"),
    ("Kinderministerien", "kinder.mp4", "stem"),  # plural
    ("Ministerium", "kinder.mp4", "trigram"),  # end of a compound word
    ("Wirtschaft", "landwirtschaft.mp4", "trigram"),
])
def test_german_word_forms(german_engine, query, expected, index):
    out = german_engine.search(query, mode="speech", explain=True)
    print(f"\n{query!r}: {[r['file'] for r in out['results']]} {out['channels'].get('speech')}")
    assert out["results"] and out["results"][0]["file"] == expected
    assert f"{index}:" in out["channels"]["speech"]["query"]


@pytest.mark.parametrize("query,expected", [
    ("Bauern, die mit der Umwelt arbeiten", "landwirtschaft.mp4"),
    ("eine Regierung, die sich um Kinder kümmert", "kinder.mp4"),
])
def test_german_speech_meaning(german_engine, query, expected):
    results = _top(german_engine, query, "speech")
    assert results and results[0]["file"] == expected


def test_calibrate_on_german_speech(german_engine):
    from clipdex import calibrate

    probes = [calibrate.parse_probe("speech: Landwirtschaft und Natur @ landwirtschaft", True),
              calibrate.parse_probe("speech: Fußball im Stadion", False)]
    for p in probes:
        calibrate.measure(german_engine, p)
    suggestions = calibrate.suggest(german_engine, probes)
    print("\n" + calibrate.report(probes, suggestions))
    assert probes[0].scores.get("speech_sem") is not None
    assert any(s.setting == "speech_sem_floor" for s in suggestions)
