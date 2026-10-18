"""'clipdex calibrate' with fake models: probes, suggestions and the config writer."""

from __future__ import annotations

import pytest

from clipdex import calibrate
from clipdex.config import load_config
from clipdex.indexer import Indexer
from clipdex.search import SearchEngine
from conftest import TRANSCRIPTS, FakeModels


def _engine(library, cfg):
    idx = Indexer(library, cfg, models=FakeModels(cfg, TRANSCRIPTS), log=lambda m: None)
    idx.run()
    return idx, SearchEngine(idx.index_dir, cfg, root=idx.root, models=FakeModels(cfg))


def _run(engine, present, absent):
    probes = [calibrate.parse_probe(t, True) for t in present] + [calibrate.parse_probe(t, False) for t in absent]
    for p in probes:
        calibrate.measure(engine, p)
    return probes, {s.setting: s for s in calibrate.suggest(engine, probes)}


def test_parse_probe():
    p = calibrate.parse_probe("action: jemand trinkt aus einer Flasche @ IMG_2011", True)
    assert (p.mode, p.query, p.file) == ("action", "jemand trinkt aus einer Flasche", "IMG_2011")
    p = calibrate.parse_probe("Strand mit Palmen", False)
    assert (p.mode, p.query, p.file) == ("auto", "Strand mit Palmen", None)
    with pytest.raises(ValueError):
        calibrate.parse_probe("visual: Strand @ IMG_1", False)  # not in the footage: no file


def test_lowers_a_floor_that_hides_a_known_thing(library, cfg):
    cfg = cfg.apply({"visual_min_prob": 0.95})  # the red frames score 0.88 with the fake model
    _, engine = _engine(library, cfg)
    probes, s = _run(engine, ["visual: red @ red_blue"], ["visual: a beach with palm trees"])
    assert not probes[0].kept.get("visual")  # the visual channel drops it (the scene channel finds it)
    assert s["visual_min_prob"].value is not None and s["visual_min_prob"].value < 0.88


def test_raises_a_floor_that_lets_a_missing_thing_through(library, cfg):
    cfg = cfg.apply({"visual_min_prob": 1e-6})
    _, engine = _engine(library, cfg)
    probes, s = _run(engine, ["visual: red @ red_blue"], ["visual: a beach with palm trees"])
    assert probes[0].found and probes[1].found  # the missing thing has results: a leak
    value = s["visual_min_prob"].value
    assert 0.000335 < value < 0.88  # between the leak (logit -8) and the known thing (logit 2)
    engine.cfg = engine.cfg.apply({"visual_min_prob": value})
    assert engine.search("a beach with palm trees", mode="visual")["results"] == []
    assert engine.search("red", mode="visual", path_contains="red_blue")["results"]


def test_conflict_keeps_the_floor(library, cfg):
    _, engine = _engine(library, cfg)
    _, s = _run(engine, ["visual: blue @ red_blue"], ["visual: red"])  # red is there too
    assert s["visual_min_prob"].value is None and "conflict" in s["visual_min_prob"].reason


def test_write_config_keeps_other_lines(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('# my notes\nscene_floor = 0.40\nlanguages = ["de", "en"]\n')
    changes = [calibrate.Suggestion("scene_floor", 0.40, 0.36, "reason a"),
               calibrate.Suggestion("visual_min_prob", 0.002, 0.0015, "reason b"),
               calibrate.Suggestion("speech_sem_floor", 0.42, None, "keep")]
    backup = calibrate.write_config(path, changes)
    text = path.read_text()
    assert backup is not None and backup.read_text().startswith("# my notes")
    assert "# my notes" in text and 'languages = ["de", "en"]' in text and "0.36" in text
    assert text.count("# clipdex calibrate") == 2
    cfg = load_config(tmp_path)
    assert (cfg.scene_floor, cfg.visual_min_prob, cfg.speech_sem_floor) == (0.36, 0.0015, 0.42)


def test_cli_calibrate_writes_config(library, cfg, capsys, monkeypatch):
    import clipdex.search as search_mod
    from clipdex import cli

    idx, _ = _engine(library, cfg.apply({"visual_min_prob": 0.95}))
    (idx.index_dir / "config.toml").write_text("visual_min_prob = 0.95\n")
    real = search_mod.SearchEngine
    monkeypatch.setattr(search_mod, "SearchEngine", lambda index_dir, cfg_, root=None, use_qwen=True:
                        real(index_dir, cfg_, root=root, models=FakeModels(cfg_), use_qwen=use_qwen))
    assert cli.main(["calibrate", str(library), "--present", "visual: red @ red_blue",
                     "--absent", "visual: a beach", "--write"]) == 0
    out = capsys.readouterr().out
    assert "visual 0.8808," in out and "visual_min_prob = 0.95 -> 0.8176" in out
    assert load_config(idx.index_dir).visual_min_prob < 0.88
