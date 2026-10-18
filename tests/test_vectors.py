"""The memory-mapped vector cache: incremental updates, rebuilds and fallbacks."""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from clipdex import vectors
from clipdex.indexer import Indexer
from clipdex.search import SearchEngine
from conftest import TRANSCRIPTS, FakeModels, make_color_video


def _index(library, cfg):
    return Indexer(library, cfg, models=FakeModels(cfg, TRANSCRIPTS), log=lambda m: None)


def _manifest(idx) -> dict:
    return json.loads((idx.index_dir / "vectors" / "manifest.json").read_text())


def _live(idx, kind: str) -> set[tuple[int, int]]:
    """(row id, file id) of the live rows in the cache files."""
    info = _manifest(idx)["tables"][kind]
    if not info["rows"]:
        return set()
    rows = np.memmap(idx.index_dir / "vectors" / f"{kind}.rows", dtype=vectors.ROW, mode="r",
                     shape=(info["rows"],))
    return {(int(r["id"]), int(r["file"])) for r in rows if r["file"] >= 0}


def _in_db(idx, kind: str) -> set[tuple[int, int]]:
    table = vectors.KINDS[kind][0]
    return {(r[0], r[1]) for r in idx.store.query(f"SELECT id, file_id FROM {table} WHERE emb IS NOT NULL")}


def _engine(idx, cfg):
    return SearchEngine(idx.index_dir, cfg, root=idx.root, models=FakeModels(cfg))


def test_cache_updates_only_changed_files(library, cfg):
    idx = _index(library, cfg)
    idx.run()
    first = _manifest(idx)
    assert first["tables"]["frames"]["rows"] == 15 and first["tables"]["frames"]["dead"] == 0
    for kind in vectors.KINDS:
        assert _live(idx, kind) == _in_db(idx, kind)
    engine = _engine(idx, cfg)
    assert engine.search("green", mode="visual")["results"]

    # a new file: its rows are appended, nothing is rebuilt
    make_color_video(library / "D" / "more_green.mp4", [("green", 4)], audio=False)
    idx.run()
    second = _manifest(idx)
    assert second["generation"] == first["generation"]
    assert second["tables"]["frames"]["rows"] == 17 and second["tables"]["frames"]["dead"] == 0
    assert "D/more_green.mp4" in {r["file"] for r in engine.search("green", mode="visual")["results"]}

    # redo one file: its old rows become dead, the new rows are appended
    idx.run(redo=["visual"], only="b/green")
    third = _manifest(idx)
    assert third["tables"]["frames"]["dead"] == 4 and third["tables"]["frames"]["rows"] == 21

    # a deleted file: its rows become dead and it never shows up in results
    (library / "D" / "more_green.mp4").unlink()
    idx.run(prune=True)
    assert _manifest(idx)["tables"]["frames"]["dead"] == 6
    for kind in vectors.KINDS:
        assert _live(idx, kind) == _in_db(idx, kind)
    assert "D/more_green.mp4" not in {r["file"] for r in engine.search("green", mode="visual")["results"]}


def test_cache_rebuilds_after_a_model_change(library, cfg):
    idx = _index(library, cfg)
    idx.run()
    generation = _manifest(idx)["generation"]
    cfg2 = cfg.apply({"siglip_model": "another/siglip"})
    Indexer(library, cfg2, models=FakeModels(cfg2, TRANSCRIPTS), log=lambda m: None).run(reembed=["visual"])
    man = _manifest(idx)
    assert man["generation"] != generation and man["models"]["model.siglip"] == "another/siglip"
    assert man["tables"]["frames"]["dead"] == 0 and _live(idx, "frames") == _in_db(idx, "frames")


def test_cache_drops_an_append_that_stopped_halfway(library, cfg):
    idx = _index(library, cfg)
    idx.run()
    folder = idx.index_dir / "vectors"
    for name in ("frames.rows", "frames.f16"):
        with open(folder / name, "ab") as fh:
            fh.write(b"\x01" * 100)  # bytes after the manifest's row count
    make_color_video(library / "D" / "more_green.mp4", [("green", 4)], audio=False)
    idx.run()
    assert _live(idx, "frames") == _in_db(idx, "frames")
    rows = _manifest(idx)["tables"]["frames"]["rows"]
    assert (folder / "frames.rows").stat().st_size == rows * vectors.ROW.itemsize


def test_read_only_index_folder_keeps_vectors_in_memory(library, cfg, monkeypatch):
    idx = _index(library, cfg)
    idx.run()
    monkeypatch.setattr(vectors.VectorCache, "_writable", lambda self: False)
    engine = _engine(idx, cfg)
    assert engine.search("green", mode="visual")["results"] and engine.cache.in_memory
    make_color_video(library / "D" / "more_green.mp4", [("green", 4)], audio=False)
    idx.run()
    assert "D/more_green.mp4" in {r["file"] for r in engine.search("green", mode="visual")["results"]}
    assert engine.cache.last_sync == "update"


def test_index_from_before_the_stamps_is_rebuilt(library, cfg):
    idx = _index(library, cfg)
    idx.run()
    idx.store.close()
    conn = sqlite3.connect(idx.index_dir / "index.db")
    for col in ("frames_rev", "scenes_rev", "chunks_rev"):
        conn.execute(f"ALTER TABLE files DROP COLUMN {col}")
    conn.execute("UPDATE meta SET value = value + 1 WHERE key = 'revision'")
    conn.commit()
    conn.close()
    engine = _engine(idx, cfg)
    assert engine.search("green", mode="visual")["results"]
    assert engine.cache.last_sync == "rebuild"


@pytest.mark.parametrize("kind", list(vectors.KINDS))
def test_empty_tables_open(tmp_path, kind):
    from clipdex.store import Store

    store = Store(tmp_path / "index.db")
    cache = vectors.VectorCache(tmp_path, store)
    cache.sync()
    assert len(cache.tables[kind]) == 0


def test_reused_file_id_does_not_keep_old_vectors(library, cfg):
    idx = _index(library, cfg)
    idx.run()
    top = max(idx.store.files(), key=lambda r: r.id)
    (library / top.relpath).unlink()
    idx.sync(prune=True)  # a run that stopped after its sync: the cache still has the old rows
    (library / "Z").mkdir()
    (library / "Z" / "broken.mp4").write_bytes(b"\x00not a video" * 1000)
    idx.run()
    new = idx.store.file_by_path("Z/broken.mp4")
    assert new.id == top.id and new.error  # SQLite reused the id; the new file cannot be read
    for kind in vectors.KINDS:
        assert _live(idx, kind) == _in_db(idx, kind)
    engine = _engine(idx, cfg)
    assert "Z/broken.mp4" not in {r["file"] for r in engine.search("yellow", mode="visual")["results"]}
