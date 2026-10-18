"""'clipdex setup': detect the machine, pick a profile, write the config, download models."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from . import hardware
from .config import load_config, load_toml
from .library import default_index_dir
from .media import MediaError, ensure_tools
from .speech import is_apple_silicon

APP = "clipdex"


def _ask(question: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print(f"{question} [skipped: no terminal; use --yes]")
        return False
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes", "j", "ja")


def _models_for(cfg) -> list[tuple[str, str]]:
    """(label, Hugging Face repo or faster-whisper name) for every model the config needs."""
    models = [("visual", cfg.siglip_model)]
    if cfg.qwen_enabled:
        models.append(("scenes + speech meaning", cfg.qwen_model))
    if cfg.asr_backend != "none":
        backend = cfg.asr_backend
        if backend == "auto":
            backend = "mlx" if is_apple_silicon() else "faster-whisper"
        if backend == "mlx":
            models.append(("speech (mlx-whisper)", cfg.asr_model or "mlx-community/whisper-large-v3-turbo"))
        else:
            models.append(("speech (faster-whisper)", "fw:" + (cfg.asr_model or "large-v3-turbo")))
    return models


def _repo_size_gb(repo: str) -> float | None:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, files_metadata=True)
        return sum((s.size or 0) for s in info.siblings or []) / hardware.GB
    except Exception:  # noqa: BLE001 - offline or unknown: size is only informational
        return None


def _download(repo: str) -> None:
    if repo.startswith("fw:"):
        from faster_whisper import download_model

        download_model(repo[3:])
        return
    from huggingface_hub import snapshot_download

    snapshot_download(repo)


def run_setup(root: Path, index_dir: Path, profile_name: str | None, assume_yes: bool,
              download: bool, list_only: bool) -> int:
    if list_only:
        for p in hardware.PROFILES.values():
            print(f"{p.name:14s} {p.description}  (indexer ~{p.ram_index_gb:.0f} GB, MCP ~{p.ram_mcp_gb:.0f} GB)")
        return 0

    print(f"{APP} setup\n")
    try:
        ensure_tools()
        print("OK  ffmpeg and ffprobe found")
    except MediaError as exc:
        print(f"NO  {exc}")
        return 1

    hw = hardware.detect()
    print(f"OK  {hw.summary()}")
    if profile_name:
        if profile_name not in hardware.PROFILES:
            print(f"Unknown profile '{profile_name}'. Use one of: {', '.join(hardware.PROFILES)}")
            return 2
        profile = hardware.PROFILES[profile_name]
    else:
        profile = hardware.choose_profile(hw)
    print(f"\nProfile: {profile.name} — {profile.description}")
    print(f"Estimated peak memory: indexer about {profile.ram_index_gb:.0f} GB, "
          f"MCP server about {profile.ram_mcp_gb:.0f} GB (estimate, not measured on your machine)")
    for note in profile.notes:
        print(f"Note: {note}")

    # --- config -------------------------------------------------------------------------
    index_dir.mkdir(parents=True, exist_ok=True)
    config_path = index_dir / "config.toml"
    text = hardware.render_config(profile, hw)
    write = True
    if config_path.exists():
        current = load_toml(config_path)
        changes = {k: v for k, v in profile.settings.items() if current.get(k) != v}
        if not changes:
            print(f"\n{config_path} already matches this profile.")
            write = False
        else:
            print(f"\n{config_path} exists. The profile would change: "
                  + ", ".join(f"{k}={v!r}" for k, v in changes.items()))
            write = _ask("Overwrite it?", assume_yes)
    if write:
        if config_path.exists():
            shutil.copy(config_path, config_path.with_suffix(".toml.bak"))
        config_path.write_text(text)
        print(f"Wrote {config_path}")
    cfg = load_config(index_dir)

    db = index_dir / "index.db"
    if db.exists():
        from .store import Store

        store = Store(db, readonly=True)
        old_siglip, old_qwen = store.get_meta("model.siglip"), store.get_meta("model.qwen")
        store.close()
        if old_siglip and old_siglip != cfg.siglip_model:
            print(f"Note: the index used {old_siglip}. Run 'index --reembed visual' once.")
        if cfg.qwen_enabled and old_qwen and old_qwen != cfg.qwen_model:
            print(f"Note: the index used {old_qwen}. Run 'index --reembed qwen' once.")

    # --- models -----------------------------------------------------------------------------
    models = _models_for(cfg)
    print("\nModels for this profile:")
    total = 0.0
    for label, repo in models:
        size = None if repo.startswith("fw:") else _repo_size_gb(repo)
        total += size or 0.0
        print(f"  {label:26s} {repo.removeprefix('fw:')}" + (f"  ({size:.1f} GB)" if size else ""))
    if download:
        if _ask(f"Download them now (about {total:.1f} GB, cached in ~/.cache/huggingface)?", assume_yes):
            for _label, repo in models:
                print(f"Downloading {repo.removeprefix('fw:')} ...", flush=True)
                _download(repo)
            print("Models are ready.")
        else:
            print("Skipped. The first 'index' run downloads them.")

    # --- next steps -----------------------------------------------------------------------------
    exe = shutil.which(APP) or str(Path(sys.argv[0]).resolve())
    mcp_cmd = [exe, "mcp", str(root)]
    if index_dir.resolve() != default_index_dir(root).resolve():
        mcp_cmd += ["--index-dir", str(index_dir)]
    print("\nNext steps:")
    print(f"  1. Test with 3 files:   {APP} index {root} --limit 3")
    print("     Check 'x realtime' and the memory line at the end.")
    print(f"  2. Index everything:    {APP} index {root}")
    print(f"  3. Search:              {APP} search \"laptop on a desk\" {root}")
    claude = shutil.which("claude")
    add_cmd = ["claude", "mcp", "add", "--transport", "stdio", "--scope", "user", APP, "--", *mcp_cmd]
    print("  4. Connect Claude Code: " + " ".join(f'"{a}"' if " " in a else a for a in add_cmd))
    if claude and _ask("Register the MCP server in Claude Code now?", assume_yes=False):
        result = subprocess.run(add_cmd, capture_output=True, text=True)
        print(result.stdout.strip() or result.stderr.strip())
    return 0
