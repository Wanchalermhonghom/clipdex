"""End-to-end tests: index the synthetic library with fake models, then search it."""

from __future__ import annotations

import asyncio
import os
import shutil

import pytest

from clipdex.indexer import Indexer, IndexError_
from clipdex.search import SearchEngine
from conftest import TRANSCRIPTS, FakeModels, make_color_video


def build(library, cfg, **kw):
    models = FakeModels(cfg, TRANSCRIPTS)
    idx = Indexer(library, cfg, models=models, log=lambda m: None)
    report = idx.run(**kw)
    return idx, models, report


def engine_for(idx, cfg, models=None):
    return SearchEngine(idx.index_dir, cfg, root=idx.root, models=models or FakeModels(cfg))


def test_index_builds_all_stages(library, cfg):
    idx, models, report = build(library, cfg)
    assert report.processed == 3 and report.failed == 0
    stats = idx.store.stats()
    assert stats["files"] == 3
    for stage in ("frames", "visual", "speech", "speech_emb", "scenes"):
        assert stats[f"{stage}_done"] == 3, stage
    assert stats["frames"] == 6 + 4 + 5  # 12 s, 8 s and 10 s at one frame per 2 s
    assert stats["segments"] == 3
    # green.mov has no audio track, so it is never transcribed
    assert sorted(models.transcribed) == ["A/red_blue.mp4", "B/sub/yellow_red.mkv"]
    green = idx.store.file_by_path("B/green.mov")
    assert green.has_audio == 0 and green.language is None
    red_blue = idx.store.file_by_path("A/red_blue.mp4")
    assert red_blue.language == "de" and (idx.index_dir / "frames" / str(red_blue.id)).is_dir()
    # scenes split at the color change
    rows = idx.store.query("SELECT start, end FROM scenes WHERE file_id=? ORDER BY start", (red_blue.id,))
    assert [(r["start"], round(r["end"])) for r in rows] == [(0.0, 6), (6.0, 12)]


def test_second_run_is_a_no_op_and_resumes(library, cfg):
    idx, models, _ = build(library, cfg)
    report = idx.run()
    assert report.processed == 0
    # simulate an interrupted run: one stage not done
    f = idx.store.file_by_path("B/green.mov")
    idx.store.reset_stages(f.id, ["scenes"])
    idx.store.commit()
    calls_before = idx.models.qwen().calls
    report = idx.run()
    assert report.processed == 1
    assert idx.models.qwen().calls == calls_before + 1
    assert idx.store.file_by_path("B/green.mov").scenes_done == 1


def test_changed_moved_missing_and_prune(library, cfg):
    idx, _, _ = build(library, cfg)
    old = idx.store.file_by_path("B/green.mov")
    # move: same content, new path -> keeps the same row and data
    os.makedirs(library / "C", exist_ok=True)
    shutil.move(library / "B" / "green.mov", library / "C" / "green.mov")
    report = idx.run()
    assert report.sync.moved == 1 and report.processed == 0
    moved = idx.store.file_by_path("C/green.mov")
    assert moved.id == old.id and moved.frames_done == 1
    # change: new content at the same path -> re-indexed
    make_color_video(library / "A" / "red_blue.mp4", [("blue", 4)])
    report = idx.run()
    assert report.sync.changed == 1 and report.processed == 1
    assert idx.store.file_by_path("A/red_blue.mp4").duration == pytest.approx(4.0, abs=0.2)
    # delete -> missing, then prune
    (library / "B" / "sub" / "yellow_red.mkv").unlink()
    report = idx.run()
    assert report.sync.missing == 1
    assert idx.store.file_by_path("B/sub/yellow_red.mkv").missing == 1
    report = idx.run(prune=True)
    assert report.sync.pruned == 1 and idx.store.file_by_path("B/sub/yellow_red.mkv") is None


def test_model_change_needs_reembed(library, cfg):
    idx, _, _ = build(library, cfg)
    cfg2 = cfg.apply({"siglip_model": "other/siglip"})
    idx2 = Indexer(library, cfg2, models=FakeModels(cfg2, TRANSCRIPTS), log=lambda m: None)
    with pytest.raises(IndexError_, match="--reembed visual"):
        idx2.run()
    report = idx2.run(reembed=["visual"])
    assert report.processed == 3
    assert idx2.store.get_meta("model.siglip") == "other/siglip"


def test_failed_file_is_recorded_and_skipped(library, cfg):
    (library / "broken.mp4").write_bytes(b"this is not a video file" * 100)
    idx, _, report = build(library, cfg)
    assert report.failed == 1 and report.processed == 3
    assert "ffprobe" in idx.store.file_by_path("broken.mp4").error
    report = idx.run()
    assert report.skipped_failed == 1 and report.processed == 0


def test_one_model_at_a_time_with_progress(library, cfg, monkeypatch):
    from clipdex import indexer as indexer_mod

    pauses = []
    monkeypatch.setattr(indexer_mod, "_pause", pauses.append)
    cfg = cfg.apply({"pause_seconds": 1.5})
    models = FakeModels(cfg, TRANSCRIPTS)
    logs: list[str] = []
    report = Indexer(library, cfg, models=models, log=logs.append).run()
    assert report.processed == 3 and report.failed == 0
    assert models.max_loaded == 1  # SigLIP, Whisper and Qwen were never in memory together
    assert models.released == ["siglip", "asr", "qwen"]
    assert [m.split(":")[0] for m in logs if m.startswith("Pass ")] == ["Pass 1/3", "Pass 2/3", "Pass 3/3"]
    visual = [m for m in logs if m.startswith("[visual ")]
    # no ETA after the first file (it includes the model load) and after the last one
    assert len(visual) == 3 and ["ETA" in m for m in visual] == [False, True, False]
    assert "x realtime" in logs[-1] and logs[-1].startswith("Done: 3 file(s)")
    assert pauses == [1.5] * 6  # after each file except the last one of each pass: 3 passes x 2


def test_eta_uses_footage_seconds(monkeypatch):
    from types import SimpleNamespace

    from clipdex import indexer as indexer_mod

    now = [100.0]
    monkeypatch.setattr(indexer_mod.time, "time", lambda: now[0])
    # four files: 60 s each when known; the last one was never probed (size only)
    rows = [SimpleNamespace(size=600, duration=60.0), SimpleNamespace(size=600, duration=None),
            SimpleNamespace(size=600, duration=60.0), SimpleNamespace(size=1200, duration=None)]
    eta = indexer_mod._Eta(rows)
    now[0] += 50.0  # first file: model load + work
    eta.done(0, 60.0)
    assert eta.text(0) == ""
    now[0] += 30.0  # second file: 60 s of footage in 30 s = 2x realtime
    eta.done(1, 60.0)
    # left: 60 s (known) + 1200 bytes at 10 bytes per second = 120 s -> 180 s at 2x = 90 s
    assert eta.text(1) == "  ETA 1:30"


def test_throttle_settings():
    from clipdex.config import Config

    cfg = Config(siglip_batch=32, qwen_batch=4, pause_seconds=10.0).throttled()
    assert (cfg.siglip_batch, cfg.qwen_batch, cfg.pause_seconds) == (8, 1, 10.0)
    assert Config(siglip_batch=4).throttled().siglip_batch == 4


def test_only_and_limit(library, cfg):
    idx, _, report = build(library, cfg, only="b/", limit=1)
    assert report.processed == 1
    assert idx.store.file_by_path("B/green.mov").frames_done == 1
    assert idx.store.file_by_path("A/red_blue.mp4").frames_done == 0


# --- search -----------------------------------------------------------------------------------

@pytest.fixture()
def engine(library, cfg):
    idx, _, _ = build(library, cfg)
    return engine_for(idx, cfg)


def _spans(results):
    return [(r["file"], round(r["start"]), round(r["end"])) for r in results]


def test_visual_search_finds_color_ranges(engine):
    out = engine.search("a red wall", mode="visual", limit=10)
    files = {r["file"] for r in out["results"]}
    assert files == {"A/red_blue.mp4", "B/sub/yellow_red.mkv"}
    for r in out["results"]:
        if r["file"] == "A/red_blue.mp4":
            assert r["start"] >= 0 and r["end"] <= 7.0
        else:
            assert r["start"] >= 4.0
        assert r["matched"][0] in ("visual", "scenes")
        assert r["path"].endswith(r["file"])


def test_german_query_and_action_mode(engine):
    out = engine.search("blaue Fläche", mode="action")
    assert _spans(out["results"]) == [("A/red_blue.mp4", 6, 12)]
    assert out["results"][0]["matched"] == ["scenes", "visual"]


def test_nothing_found_for_unrelated_query(engine):
    out = engine.search("purple elephant", mode="visual")
    assert out["results"] == []


def test_speech_exact_words_with_word_timing(engine):
    out = engine.search("wo rede ich über Kaffee", mode="speech")
    top = out["results"][0]
    assert top["file"] == "A/red_blue.mp4"
    assert top["best"] == pytest.approx(2.2)
    assert "speech" in top["matched"] and "Kaffee" in top["transcript"]


def test_speech_phrase(engine):
    out = engine.search('"claude code"', mode="speech")
    top = out["results"][0]
    assert top["file"] == "A/red_blue.mp4" and top["best"] == pytest.approx(8.4)


def test_auto_mode_and_path_filter(engine):
    out = engine.search("red", mode="auto", path_contains="sub/")
    assert {r["file"] for r in out["results"]} == {"B/sub/yellow_red.mkv"}
    out = engine.search("Bangkok morning", mode="auto", explain=True)
    assert out["results"][0]["file"] == "B/sub/yellow_red.mkv"
    assert "speech" in out["channels"]


def test_max_per_file(engine):
    out = engine.search("red", mode="visual", max_per_file=1)
    files = [r["file"] for r in out["results"]]
    assert len(files) == len(set(files))


def test_similar_transcript_and_files(engine):
    hit = engine.search("yellow", mode="visual")["results"][0]
    sim = engine.similar_frames(hit["frame_id"], limit=5)
    assert sim["results"] == [] or all(r["frame_id"] != hit["frame_id"] for r in sim["results"])
    red = engine.search("red", mode="visual")["results"][0]
    sim = engine.similar_frames(red["frame_id"], limit=5)
    assert sim["results"] and sim["results"][0]["similarity"] == pytest.approx(1.0)
    assert engine.transcript("red_bl")["file"] == "A/red_blue.mp4"  # part of the file name
    with pytest.raises(ValueError, match="matches 2 files"):
        engine.transcript("red")  # red_blue.mp4 and yellow_red.mkv
    tr = engine.transcript("red_blue.mp4", start=5.0)
    assert [s["text"][:5] for s in tr["segments"]] == ["Jetzt"]
    with pytest.raises(ValueError):
        engine.transcript("does_not_exist.mp4")
    files = engine.list_files()
    assert files["count"] == 3 and all(f["indexed"] for f in files["files"])
    status = engine.status()
    assert status["files"] == 3 and status["root_available"]


def test_engine_reloads_after_reindex(library, cfg):
    idx, _, _ = build(library, cfg)
    engine = engine_for(idx, cfg)
    assert engine.search("green", mode="visual")["results"]
    make_color_video(library / "D" / "more_green.mp4", [("green", 4)], audio=False)
    idx.run()
    files = {r["file"] for r in engine.search("green", mode="visual")["results"]}
    assert "D/more_green.mp4" in files


def test_search_without_qwen(library, cfg):
    idx, _, _ = build(library, cfg)
    engine = SearchEngine(idx.index_dir, cfg, root=idx.root, models=FakeModels(cfg), use_qwen=False)
    out = engine.search("blue", mode="action")
    assert out["results"] and out["results"][0]["matched"] == ["visual"]


def test_bad_arguments(engine):
    with pytest.raises(ValueError):
        engine.search("x", mode="nope")
    with pytest.raises(ValueError):
        engine.search("   ")


# --- MCP --------------------------------------------------------------------------------------

def test_mcp_tools(engine):
    from mcp import Client

    from clipdex.mcp_server import build_server

    server = build_server(engine)

    async def run():
        async with Client(server) as client:
            tools = {t.name for t in (await client.list_tools()).tools}
            assert tools == {"search_footage", "view_frames", "get_transcript", "get_scenes", "similar_shots",
                             "list_files", "index_status"}
            res = await client.call_tool("search_footage", {"query": "blue", "mode": "visual"})
            import json

            data = json.loads(res.content[0].text)
            assert data["results"][0]["file"] == "A/red_blue.mp4"
            frame_id = data["results"][0]["frame_id"]
            res = await client.call_tool("view_frames", {"frame_ids": [frame_id, 999999]})
            kinds = [c.type for c in res.content]
            assert kinds == ["text", "image", "text"]
            assert res.content[1].mime_type == "image/jpeg"
            res = await client.call_tool("get_transcript", {"file": "red_blue.mp4"})
            assert "Kaffee" in res.content[0].text
            # get_scenes after a search: the engine's scene table must not hide the scenes() method
            res = await client.call_tool("get_scenes", {"file": "red_blue.mp4"})
            assert not res.is_error and json.loads(res.content[0].text)["scenes"]
            res = await client.call_tool("search_footage", {"query": "x", "mode": "bad"})
            assert res.is_error

    asyncio.run(run())


def test_mcp_max_per_file_defaults_to_config(library, cfg):
    import json

    from mcp import Client

    from clipdex.mcp_server import build_server

    cfg = cfg.apply({"max_per_file": 1})
    idx, _, _ = build(library, cfg)
    # Without Qwen the transcript chunk does not join the two sentences: two moments in red_blue.mp4.
    server = build_server(SearchEngine(idx.index_dir, cfg, root=idx.root, models=FakeModels(cfg), use_qwen=False))
    query = {"query": "Kaffee Laptop", "mode": "speech"}

    async def files(args):
        async with Client(server) as client:
            res = await client.call_tool("search_footage", args)
            return [r["file"] for r in json.loads(res.content[0].text)["results"]]

    assert asyncio.run(files(query)).count("A/red_blue.mp4") == 1  # the config value
    assert asyncio.run(files({**query, "max_per_file": 2})).count("A/red_blue.mp4") == 2


def test_search_works_when_the_drive_is_not_mounted(tmp_path, library, cfg):
    from pathlib import Path

    from mcp import Client

    from clipdex.mcp_server import build_server

    index_dir = tmp_path / "internal-ssd" / "clipdex-index"  # like --index-dir
    idx = Indexer(library, cfg, index_dir=index_dir, models=FakeModels(cfg, TRANSCRIPTS), log=lambda m: None)
    idx.run()
    engine = SearchEngine(index_dir, cfg, models=FakeModels(cfg))  # root and drive name come from the index
    hit = engine.search("blue", mode="visual")["results"][0]
    assert hit["available"] is True and Path(hit["path"]).is_file()
    assert hit["volume"] == (idx.store.get_meta("root_volume") or None)

    library.rename(tmp_path / "elsewhere")  # the drive is unplugged
    engine._available = (0.0, False)  # do not wait for the 2 s cache
    hit = engine.search("blue", mode="visual")["results"][0]
    assert hit["file"] == "A/red_blue.mp4" and hit["available"] is False and hit["path"] is None
    tr = engine.transcript("red_blue.mp4")
    assert tr["available"] is False and tr["path"] is None and tr["segments"]
    assert engine.scenes("red_blue.mp4")["scenes"]
    assert engine.status()["root_available"] is False

    async def frames():
        async with Client(build_server(engine)) as client:
            res = await client.call_tool("view_frames", {"frame_ids": [hit["frame_id"]]})
            return [c.type for c in res.content]

    assert asyncio.run(frames()) == ["text", "image"]  # the frames live in the index

    # An empty folder can stay at the mount path after the drive is gone (/Volumes/T7).
    library.mkdir()
    idx.store.set_meta("root_mount", str(library))
    idx.store.commit()
    engine._available = (0.0, True)
    assert engine.search("blue", mode="visual")["results"][0]["available"] is False


def test_mount_point_and_volume_name(tmp_path):
    from clipdex.library import mount_point, volume_name

    assert mount_point(tmp_path / "does" / "not" / "exist").exists()
    assert os.path.ismount(mount_point(tmp_path))
    assert volume_name(tmp_path.anchor) is None  # the system disk has no drive name


def test_cli_search_and_status(library, cfg, capsys, monkeypatch):
    idx, _, _ = build(library, cfg)
    import clipdex.search as search_mod
    from clipdex import cli

    real = search_mod.SearchEngine

    def fake_engine(index_dir, cfg_, root=None, use_qwen=True):
        return real(index_dir, cfg_, root=root, models=FakeModels(cfg_), use_qwen=use_qwen)

    monkeypatch.setattr(search_mod, "SearchEngine", fake_engine)
    assert cli.main(["search", "blue", str(library), "--mode", "visual"]) == 0
    out = capsys.readouterr().out
    assert "A/red_blue.mp4" in out and "00:00:06" in out
    assert cli.main(["status", str(library)]) == 0
    assert '"files": 3' in capsys.readouterr().out
    assert cli.main(["transcript", "red_blue.mp4", str(library)]) == 0
    assert "Kaffee" in capsys.readouterr().out


def test_prune_also_cleans_full_text_index(library, cfg):
    idx, _, _ = build(library, cfg)
    engine = engine_for(idx, cfg)
    assert engine.search("Bangkok", mode="speech")["results"]
    (library / "B" / "sub" / "yellow_red.mkv").unlink()
    idx.run(prune=True)
    rows = idx.store.query("SELECT rowid FROM segments_fts WHERE segments_fts MATCH 'Bangkok'")
    assert rows == []
    assert engine.search("Bangkok", mode="speech")["results"] == []


def test_cosine_channel_uses_query_baseline(engine):
    import numpy as np

    from clipdex.search import _Table

    rng = np.random.default_rng(0)
    n = 40
    emb = rng.normal(size=(n, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    q = emb[0].copy()
    table = _Table(list(range(n)), [1] * n, emb.astype(np.float16), start=[0.0] * n, end=[1.0] * n)
    diag = {}
    order, cos = engine._cosine_channel("scenes", table, q, {1}, floor=0.0, margin=0.5, diag=diag)
    baseline = np.percentile(emb @ q, 20)
    assert diag["scenes"]["baseline"] == pytest.approx(baseline, abs=1e-3)
    assert order[0] == 0 and all(cos[i] >= baseline + 0.5 - 1e-6 for i in order)
    # a file filter that excludes everything returns nothing
    assert engine._cosine_channel("scenes", table, q, {2}, 0.0, 0.5, {})[0] == []


def test_umlaut_paths_match_in_both_unicode_forms(library, cfg):
    import unicodedata

    nfd = unicodedata.normalize("NFD", "Küche")  # how macOS stores names
    target = library / nfd / "kaffee.mp4"
    target.parent.mkdir()
    make_color_video(target, [("blue", 4)], audio=False)
    idx, _, _ = build(library, cfg, only=unicodedata.normalize("NFC", "küche"))
    assert idx.store.file_by_path(f"{nfd}/kaffee.mp4").frames_done == 1
    engine = engine_for(idx, cfg)
    nfc = unicodedata.normalize("NFC", "Küche")
    out = engine.search("blue", mode="visual", path_contains=nfc)
    assert [r["file"] for r in out["results"]] == [f"{nfd}/kaffee.mp4"]
    assert engine.transcript(f"{nfc}/kaffee.mp4")["file"] == f"{nfd}/kaffee.mp4"
    assert engine.transcript(str(library / nfc / "kaffee.mp4"))["file"] == f"{nfd}/kaffee.mp4"


def test_special_characters_in_index_path(tmp_path, video_dir, cfg):
    root = tmp_path / "Projekt #3 100% ü"
    shutil.copytree(video_dir, root)
    idx, _, _ = build(root, cfg)
    engine = engine_for(idx, cfg)
    assert engine.status()["files"] == 3


def test_unreadable_folder_is_not_pruned(library, cfg):
    if os.geteuid() == 0:
        pytest.skip("root can read every folder")
    idx, _, _ = build(library, cfg)
    sub = library / "B" / "sub"
    sub.chmod(0)
    try:
        report = idx.run(prune=True)
    finally:
        sub.chmod(0o755)
    assert report.sync.pruned == 0 and report.sync.unreadable == ["B/sub/"]
    assert idx.store.file_by_path("B/sub/yellow_red.mkv").missing == 0


def test_prune_refuses_to_empty_the_index(library, cfg, tmp_path):
    idx, _, _ = build(library, cfg)
    for f in library.rglob("*"):
        if f.is_file() and f.suffix in (".mp4", ".mov", ".mkv") and ".clipdex" not in f.parts:
            f.unlink()
    report = idx.run(prune=True)
    assert report.sync.prune_refused and report.sync.pruned == 0
    assert idx.store.stats()["missing"] == 3
    report = idx.run(prune=True, force=True)
    assert report.sync.pruned == 3


def test_touched_file_keeps_its_index(library, cfg):
    idx, _, _ = build(library, cfg)
    f = library / "A" / "red_blue.mp4"
    st = f.stat()
    os.utime(f, (st.st_atime, st.st_mtime + 3600))  # e.g. exFAT after a time zone change
    report = idx.run()
    assert report.sync.changed == 0 and report.processed == 0


def test_changed_file_gets_new_fingerprint_and_can_move_later(library, cfg):
    idx, _, _ = build(library, cfg)
    make_color_video(library / "A" / "red_blue.mp4", [("green", 4)])
    idx.run()
    row = idx.store.file_by_path("A/red_blue.mp4")
    assert row.fingerprint
    shutil.move(library / "A" / "red_blue.mp4", library / "A" / "renamed.mp4")
    report = idx.run()
    assert report.sync.moved == 1 and idx.store.file_by_path("A/renamed.mp4").id == row.id


def test_broken_audio_keeps_frames(library, cfg, monkeypatch):
    from clipdex import media

    def broken(src, duration):
        raise media.MediaError("bad audio packet")

    monkeypatch.setattr(media, "extract_audio", broken)
    idx, _, report = build(library, cfg)
    assert report.failed == 0
    row = idx.store.file_by_path("A/red_blue.mp4")
    assert row.frames_done == 1 and row.visual_done == 1 and row.speech_done == 1 and row.error is None


def test_speech_chunk_merges_with_word_hit(engine):
    out = engine.search("Claude Code Laptop", mode="speech")
    top = out["results"][0]
    assert top["file"] == "A/red_blue.mp4" and set(top["matched"]) == {"speech", "speech_sem"}
    assert out["total_moments"] == 1


def test_unreadable_folder_is_protected_without_permissions(library, cfg, monkeypatch):
    from clipdex import indexer as indexer_mod
    from clipdex.library import ScanResult, scan_with_errors

    idx, _, _ = build(library, cfg)
    real = scan_with_errors(library, cfg, idx.index_dir)
    fake = ScanResult([f for f in real.files if not f.relpath.startswith("B/sub/")], ["B/sub/"])
    monkeypatch.setattr(indexer_mod, "scan_with_errors", lambda *a, **k: fake)
    report = idx.run(prune=True)
    assert report.sync.pruned == 0 and report.sync.missing == 0
    assert idx.store.file_by_path("B/sub/yellow_red.mkv") is not None


def test_scenes_with_cut_reasons(library, cfg):
    cfg = cfg.apply({"scene_max_seconds": 4.0})
    idx, _, _ = build(library, cfg)
    engine = engine_for(idx, cfg)
    data = engine.scenes("red_blue.mp4")
    spans = [(s["start"], s["end"], s["cut"]) for s in data["scenes"]]
    # red 0-6, blue 6-12, frames every 2 s, scenes at most 4 s long
    assert spans == [(0.0, 4.0, "start"), (4.0, 6.0, "length limit"),
                     (6.0, 10.0, "picture change"), (10.0, 12.0, "length limit")]
    only = engine.scenes("red_blue.mp4", changes_only=True)["scenes"]
    assert [(s["start"], s["end"], s["cut"]) for s in only] == [(0.0, 6.0, "start"), (6.0, 12.0, "picture change")]
    assert only[1]["frame_id"] in only[1]["frame_ids"]


def test_redo_runs_a_stage_again(library, cfg):
    idx, models, _ = build(library, cfg)
    calls = idx.models.qwen().calls
    report = idx.run(redo=["scenes"], only="red_blue")
    assert report.processed == 1 and idx.models.qwen().calls == calls + 1
    assert idx.store.file_by_path("A/red_blue.mp4").scenes_done == 1
    assert idx.store.file_by_path("B/green.mov").scenes_done == 1


def test_best_time_prefers_the_spoken_word(engine):
    # speech and speech_sem both rank 1 with weight 1.0: the word time (8.4 s) wins over the chunk start (1.0 s)
    out = engine.search("Claude Code Laptop", mode="speech")
    top = out["results"][0]
    assert top["matched"] == ["speech", "speech_sem"] and top["best"] == pytest.approx(8.4)
