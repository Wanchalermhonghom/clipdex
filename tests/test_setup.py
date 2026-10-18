"""Hardware profiles and the setup command."""

from __future__ import annotations

import pytest

from clipdex import hardware
from clipdex.config import load_config
from clipdex.setup_cmd import run_setup


def hw(accel, ram=16.0, vram=0.0):
    return hardware.Hardware(system="Darwin" if accel == "mps" else "Linux",
                             machine="arm64" if accel == "mps" else "x86_64",
                             cpu="test", ram_gb=ram, accelerator=accel, vram_gb=vram)


@pytest.mark.parametrize("machine,expected", [
    (hw("mps", 8), "apple-small"),
    (hw("mps", 15.9), "apple-medium"),  # a "16 GB" Mac
    (hw("mps", 18), "apple-medium"),
    (hw("mps", 36), "apple-medium"),
    (hw("mps", 48), "apple-large"),
    (hw("mps", 128), "apple-large"),
    (hw("cuda", 32, vram=8), "nvidia-small"),
    (hw("cuda", 32, vram=12), "nvidia-medium"),
    (hw("cuda", 64, vram=24), "nvidia-large"),
    (hw("cpu", 64), "cpu"),
])
def test_choose_profile(machine, expected):
    assert hardware.choose_profile(machine).name == expected


def test_every_profile_renders_a_valid_config(tmp_path):
    for profile in hardware.PROFILES.values():
        (tmp_path / "config.toml").write_text(hardware.render_config(profile, hw("mps", 24)))
        cfg = load_config(tmp_path)
        for key, value in profile.settings.items():
            assert getattr(cfg, key) == value


def test_detect_runs_here():
    found = hardware.detect()
    assert found.ram_gb > 0 and found.accelerator in ("mps", "cuda", "cpu")
    assert "process peak" in hardware.memory_report()


def test_setup_writes_config_and_keeps_a_backup(tmp_path, capsys):
    root = tmp_path / "lib"
    root.mkdir()
    index_dir = root / ".clipdex"
    assert run_setup(root, index_dir, "apple-medium", assume_yes=True, download=False, list_only=False) == 0
    cfg = load_config(index_dir)
    assert cfg.qwen_enabled and cfg.qwen_model == "Qwen/Qwen3-VL-Embedding-2B"
    assert run_setup(root, index_dir, "apple-small", assume_yes=True, download=False, list_only=False) == 0
    assert load_config(index_dir).qwen_enabled is False
    assert (index_dir / "config.toml.bak").exists()
    out = capsys.readouterr().out
    assert "claude mcp add --transport stdio --scope user clipdex --" in out
    assert run_setup(root, index_dir, "nope", True, False, False) == 2
