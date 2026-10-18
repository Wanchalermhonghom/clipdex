"""Command line interface: clipdex setup | index | search | status | transcript | scenes | similar | mcp | doctor."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Must be set before torch is imported: run unsupported MPS ops on the CPU instead of failing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from .config import load_config  # noqa: E402
from .library import default_index_dir  # noqa: E402


def _index_dir(args) -> Path:
    if getattr(args, "index_dir", None):
        return Path(args.index_dir).expanduser().resolve()
    if getattr(args, "root", None):
        return default_index_dir(Path(args.root).expanduser().resolve())
    raise SystemExit("Give the library folder (ROOT) or --index-dir.")


def _root(args) -> Path | None:
    return Path(args.root).expanduser().resolve() if getattr(args, "root", None) else None


def _config(args, **overrides):
    extra = Path(args.config).expanduser() if getattr(args, "config", None) else None
    return load_config(_index_dir(args), extra, overrides)


def _engine(args):
    from .search import SearchEngine

    index_dir = _index_dir(args)
    if not (index_dir / "index.db").exists():
        raise SystemExit(f"No index at {index_dir}. Run 'clipdex index ROOT' first.")
    cfg = _config(args)
    return SearchEngine(index_dir, cfg, root=_root(args), use_qwen=not args.no_qwen)


def cmd_index(args) -> int:
    from .indexer import Indexer, IndexError_

    overrides = {
        "qwen_enabled": False if args.no_qwen else None,
        "asr_backend": "none" if args.no_speech else None,
        "language": args.language,
        "device": args.device,
    }
    cfg = _config(args, **overrides)
    stages = [s.strip() for s in args.stages.split(",")] if args.stages else None
    try:
        indexer = Indexer(Path(args.root).expanduser(), cfg, _index_dir(args))
        report = indexer.run(only=args.only, stages=stages, limit=args.limit,
                             retry_failed=args.retry_failed, prune=args.prune,
                             reembed=args.reembed or [], force=args.force, redo=args.redo or [])
    except IndexError_ as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    if report.skipped_failed:
        print(f"{report.skipped_failed} file(s) failed earlier and were skipped. "
              "Use --retry-failed to try them again. 'clipdex status' shows the errors.", file=sys.stderr)
    return 1 if report.failed else 0


def _print_results(data: dict) -> None:
    results = data["results"]
    if not results:
        print("No matches.")
        return
    for r in results:
        print(f"{r['rank']:>2}. {r['file']}  {r['start_tc']} – {r['end_tc']}  "
              f"(best {r['best_tc']})  [{', '.join(r['matched'])}]  frame {r['frame_id']}")
        for name, ch in r["channels"].items():
            if "text" in ch:
                print(f"      {name}: “{ch['text']}”")
    if not results[0].get("available", True):
        drive = results[0].get("volume") or "the library"
        print(f"\nNote: the drive ({drive}) is not mounted. Connect it to open the files.")
    if "channels" in data:
        print("\nChannels:", json.dumps(data["channels"], ensure_ascii=False))


def cmd_search(args) -> int:
    engine = _engine(args)
    data = engine.search(args.query, mode=args.mode, limit=args.limit, path_contains=args.path_contains,
                         explain=args.explain)
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=1))
    else:
        _print_results(data)
    return 0


def cmd_status(args) -> int:
    engine = _engine(args)
    print(json.dumps(engine.status(), ensure_ascii=False, indent=1))
    return 0


def cmd_transcript(args) -> int:
    engine = _engine(args)
    data = engine.transcript(args.file, args.start, args.end)
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=1))
    else:
        print(f"{data['file']}  language={data['language']}")
        for s in data["segments"]:
            print(f"[{s['start_tc']}] {s['text']}")
    return 0


def cmd_scenes(args) -> int:
    engine = _engine(args)
    data = engine.scenes(args.file, changes_only=args.changes_only)
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=1))
        return 0
    print(f"{data['file']}  ({len(data['scenes'])} scenes)")
    for sc in data["scenes"]:
        sim = "" if sc["similarity_to_previous"] is None else f"  similarity {sc['similarity_to_previous']:.2f}"
        print(f"  {sc['index']:>3}. {sc['start_tc']} – {sc['end_tc']}  {sc['cut']}{sim}  frame {sc['frame_id']}")
    return 0


def cmd_similar(args) -> int:
    engine = _engine(args)
    print(json.dumps(engine.similar_frames(args.frame_id, args.limit), ensure_ascii=False, indent=1))
    return 0


def cmd_mcp(args) -> int:
    from .mcp_server import build_server

    build_server(_engine(args)).run()
    return 0


def cmd_setup(args) -> int:
    from .setup_cmd import run_setup

    root = Path(args.root).expanduser().resolve() if args.root else None
    if root is None and not args.list:
        raise SystemExit("Give the library folder, e.g. 'clipdex setup /Volumes/T7'.")
    index_dir = _index_dir(args) if root else Path(".")
    return run_setup(root, index_dir, args.profile, args.yes, not args.no_download, args.list)


def cmd_doctor(args) -> int:
    import platform
    import shutil
    import subprocess

    ok = True

    def line(name, value, good=True):
        nonlocal ok
        ok = ok and good
        print(f"{'OK ' if good else 'NO '} {name}: {value}")

    line("python", sys.version.split()[0])
    line("platform", f"{platform.system()} {platform.machine()}")
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool)
        version = ""
        if path:
            version = subprocess.run([tool, "-version"], capture_output=True, text=True).stdout.split("\n")[0]
        line(tool, version or "not found (brew install ffmpeg)", bool(path))
    try:
        import torch

        from .devices import _bf16_works, pick_device

        device = pick_device("auto")
        line("torch", f"{torch.__version__}, device {device}, bfloat16 {'yes' if _bf16_works(device) else 'no'}")
    except Exception as exc:  # noqa: BLE001
        line("torch", f"import failed: {exc}", False)
    try:
        import transformers

        line("transformers", transformers.__version__)
    except Exception as exc:  # noqa: BLE001
        line("transformers", f"import failed: {exc}", False)
    from .speech import is_apple_silicon

    try:
        if is_apple_silicon():
            import mlx_whisper  # noqa: F401

            line("speech", "mlx-whisper")
        else:
            import faster_whisper  # noqa: F401

            line("speech", "faster-whisper")
    except Exception as exc:  # noqa: BLE001
        line("speech", f"import failed: {exc}", False)
    try:
        import silero_vad  # noqa: F401

        line("silero-vad", "ok")
    except Exception as exc:  # noqa: BLE001
        line("silero-vad", f"import failed: {exc}", False)
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="clipdex", description="Local AI search for video footage.")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, root_required=False):
        if root_required:
            sp.add_argument("root", help="library folder, e.g. /Volumes/T7")
        else:
            sp.add_argument("root", nargs="?", help="library folder, e.g. /Volumes/T7")
        sp.add_argument("--index-dir", help="index folder (default: ROOT/.clipdex)")
        sp.add_argument("--config", help="extra TOML config file")

    sp = sub.add_parser("setup", help="detect this machine, write a fitting config, download models")
    sp.add_argument("root", nargs="?", help="library folder, e.g. /Volumes/T7")
    sp.add_argument("--index-dir", help="index folder (default: ROOT/.clipdex)")
    sp.add_argument("--profile", help="use this profile instead of the detected one (see --list)")
    sp.add_argument("--list", action="store_true", help="list the profiles")
    sp.add_argument("--yes", action="store_true", help="answer yes to all questions")
    sp.add_argument("--no-download", action="store_true", help="do not download models now")
    sp.set_defaults(func=cmd_setup)

    sp = sub.add_parser("index", help="build or update the index")
    common(sp, root_required=True)
    sp.add_argument("--only", help="only files whose path contains this text")
    sp.add_argument("--stages", help="comma list: frames,visual,speech,speech_emb,scenes")
    sp.add_argument("--limit", type=int, help="process at most N files")
    sp.add_argument("--retry-failed", action="store_true", help="try failed files again")
    sp.add_argument("--prune", action="store_true", help="remove deleted files from the index")
    sp.add_argument("--force", action="store_true",
                    help="with --prune: also remove when more than half of the index would go")
    sp.add_argument("--reembed", action="append", choices=["visual", "qwen"],
                    help="rebuild embeddings after a model change")
    sp.add_argument("--redo", action="append", choices=["frames", "visual", "speech", "speech_emb", "scenes"],
                    help="run a stage again for the selected files, e.g. after a setting change")
    sp.add_argument("--no-qwen", action="store_true", help="skip scene and transcript embeddings")
    sp.add_argument("--no-speech", action="store_true", help="skip transcription")
    sp.add_argument("--language", help="force a speech language, e.g. de or en (default: detect per file)")
    sp.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"])
    sp.set_defaults(func=cmd_index)

    sp = sub.add_parser("search", help="search the index")
    sp.add_argument("query")
    common(sp)
    sp.add_argument("--mode", default="auto", choices=["auto", "visual", "action", "speech"])
    sp.add_argument("--limit", type=int, default=10)
    sp.add_argument("--in", dest="path_contains", help="only files whose path contains this text")
    sp.add_argument("--explain", action="store_true", help="show channel scores and thresholds")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--no-qwen", action="store_true", help="do not load the Qwen model")
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("status", help="show index statistics")
    common(sp)
    sp.add_argument("--no-qwen", action="store_true", help=argparse.SUPPRESS)
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("transcript", help="print the transcript of one file")
    sp.add_argument("file")
    common(sp)
    sp.add_argument("--start", type=float)
    sp.add_argument("--end", type=float)
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--no-qwen", action="store_true", help=argparse.SUPPRESS)
    sp.set_defaults(func=cmd_transcript)

    sp = sub.add_parser("scenes", help="list the scene boundaries of one file")
    sp.add_argument("file")
    common(sp)
    sp.add_argument("--changes-only", action="store_true", help="only boundaries where the picture changes")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--no-qwen", action="store_true", help=argparse.SUPPRESS)
    sp.set_defaults(func=cmd_scenes)

    sp = sub.add_parser("similar", help="find shots similar to a frame")
    sp.add_argument("frame_id", type=int)
    common(sp)
    sp.add_argument("--limit", type=int, default=10)
    sp.add_argument("--no-qwen", action="store_true", help=argparse.SUPPRESS)
    sp.set_defaults(func=cmd_similar)

    sp = sub.add_parser("mcp", help="run the MCP server (stdio) for Claude")
    common(sp)
    sp.add_argument("--no-qwen", action="store_true", help="only SigLIP and transcript search (less RAM)")
    sp.set_defaults(func=cmd_mcp)

    sp = sub.add_parser("doctor", help="check tools and libraries")
    sp.set_defaults(func=cmd_doctor)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
