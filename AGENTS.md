# AGENTS.md — guide for AI coding agents

clipdex indexes a local video library and makes it searchable by speech, by what is visible
and by what happens. A CLI builds the index. An MCP server lets agents search it.
Users run it on their own machine, so most changes are about fitting it to a machine.

## Commands

```bash
uv sync --extra dev                # install (Python 3.11-3.13)
uv run pytest                      # fast tests, fake models, ~30 s. Must pass before every commit.
uv run pytest -m models -s         # real models on sample media (downloads ~6 GB, slow on CPU)
uv run clipdex doctor              # check ffmpeg, torch device, speech backend
uv run clipdex setup LIB           # detect hardware, write LIB/.clipdex/config.toml, download models
uv run clipdex index LIB --limit 3 # test run; prints "x realtime" per file and peak memory at the end
uv run clipdex search "query" LIB --explain   # channel scores and thresholds
uv run clipdex mcp LIB             # MCP server on stdio (search only)
```

## Project map

| File | Job |
|---|---|
| `src/clipdex/cli.py` | argparse CLI. Sets `PYTORCH_ENABLE_MPS_FALLBACK=1` before torch loads. |
| `src/clipdex/setup_cmd.py`, `hardware.py` | `clipdex setup`: detection, profiles (`PROFILES`), config writing, model download, memory report |
| `src/clipdex/config.py` | Every setting with its default. Loaded from `.clipdex/config.toml`. |
| `src/clipdex/library.py` | Scan the library, skip proxies and hidden files, fingerprints, `norm()` for path matching |
| `src/clipdex/media.py` | ffprobe, frame sampling (keyframes first, full decode fallback), audio decode |
| `src/clipdex/speech.py` | Silero VAD, Whisper backends (mlx-whisper, faster-whisper), transcript chunks |
| `src/clipdex/embed_siglip.py` | SigLIP 2 frame and text vectors |
| `src/clipdex/embed_qwen.py` | Qwen3-VL-Embedding scene and text vectors (follows the official reference code) |
| `src/clipdex/scenes.py` | Split a file into short scenes from frame vectors |
| `src/clipdex/indexer.py` | Sync with the disk, run stages per file, resume, prefetch thread |
| `src/clipdex/store.py` | SQLite schema, FTS5 triggers, float16 vector BLOBs |
| `src/clipdex/search.py` | Four channels, weak-hit filters, reciprocal rank fusion into moments |
| `src/clipdex/fts.py` | Natural language to FTS5 query, German and English stop words |
| `src/clipdex/mcp_server.py` | MCP tools (mcp SDK 2.x, `MCPServer`) |
| `tests/conftest.py` | Synthetic color videos made with ffmpeg, fake SigLIP/Qwen/Whisper |

## Rules

1. Run `uv run pytest` after every change. Add a test for every bug you fix. Fast tests never load real models.
2. The MCP server talks JSON-RPC on stdout. Do not print to stdout in code that the server runs. Use stderr or `logging`.
3. Each indexing stage commits on its own, so a stopped run can continue. Only the main thread writes to the database. The prefetch thread only runs ffmpeg.
4. Stored paths are relative to the library root and keep the bytes from the file system (macOS often uses NFD). Compare user input with `library.norm()`.
5. All times are seconds from the start of the file.
6. A different embedding model makes old vectors useless. The index stores the model names in `meta`. Changing a model needs `clipdex index LIB --reembed visual` or `--reembed qwen`.
7. SigLIP text must be lower case and padded to 64 tokens (`padding="max_length"`). Other padding breaks the scores.
8. Keep `embed_qwen.py` aligned with https://github.com/QwenLM/Qwen3-VL-Embedding (right padding, last-token pooling, instruction as system message).
9. `moment_max_seconds` must be at least `scene_max_seconds` and `speech_chunk_seconds`. `Config.apply` checks this.
10. When you add or change a setting, update `clipdex.example.toml` and the README tables.

## Procedure: adapt clipdex to this machine

Use this when the user asks to set up, install, tune or fit clipdex to their computer.
Do one step at a time. Run the commands yourself. Show the key numbers after each step.
Ask before you download models (about 3-23 GB, depending on the profile) and before you overwrite a config.

1. **Tools.** Run `uv run clipdex doctor`. Fix every line that starts with `NO`.
   macOS: `brew install ffmpeg uv`. Debian/Ubuntu: `sudo apt install ffmpeg`, then install uv.
   On Apple Silicon the torch device must be `mps`.
2. **Profile.** Ask for the library folder (for example `/Volumes/T7`). Run `uv run clipdex setup LIB`.
   Tell the user the detected machine, the profile and the memory estimate. Use `--profile NAME`
   to choose another one (`uv run clipdex setup --list`).
3. **Test run.** Run `uv run clipdex index LIB --limit 3` and read the output:
   - `x realtime` per file. Below 1x on Apple Silicon: check the `mps` device, then try
     `frame_max_side = 384`, `frame_interval = 3.0` or `qwen_batch = 2`.
   - The last line `Memory: ...`. On macOS add the "process peak" and the "torch MPS" and "MLX" values.
     If the sum is above 75 % of the RAM, or the run crashed with an out-of-memory error, use a smaller
     profile, lower `qwen_batch` and `siglip_batch`, or set `qwen_enabled = false`.
   - For an exact peak on macOS: `/usr/bin/time -l uv run clipdex index LIB --limit 3`.
4. **Calibrate the search.** Ask the user for 3-5 things they know are in the footage (and in which file),
   and 2 things that are not. For each, run `uv run clipdex search "<thing>" LIB --explain --json`.
   - A known thing must appear in the results. If not, lower its channel floor in `LIB/.clipdex/config.toml`
     in steps of 0.02: `scene_floor` (action mode), `speech_sem_floor` (speech meaning), `visual_min_prob` (visual).
   - A missing thing should give no results. If it does, raise the floor in steps of 0.02.
   - Write a comment next to every changed value with the reason.
5. **Connect the agent.** Run the `claude mcp add ...` command that `setup` printed. For Claude Desktop,
   add the server to `~/Library/Application Support/Claude/claude_desktop_config.json` and restart the app.
   Test it: ask the agent to call `index_status`.
6. **Full index.** Tell the user to run `caffeinate -i clipdex index LIB` (macOS, keeps the Mac awake)
   in their own terminal. It can take hours. Ctrl-C is safe; the same command continues.
7. **Report.** Give the user: profile, measured speed, measured memory, changed settings.

## Extend clipdex

- New hardware profile: add it to `hardware.PROFILES`, extend `choose_profile`, add a case to `tests/test_setup.py`.
- New embedding model of the same kind: usually only a config value. Check the vector dimension and re-calibrate the floors.
- New speech backend: implement the `WhisperBackend` protocol in `speech.py` and add it to `make_backend`.
- New MCP tool: add it in `mcp_server.py` and extend `test_mcp_tools`.
