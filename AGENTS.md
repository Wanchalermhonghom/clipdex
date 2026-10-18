# AGENTS.md — guide for AI coding agents

clipdex indexes a local video library and makes it searchable by speech, by what is visible
and by what happens. A CLI builds the index. An MCP server lets agents search it.
Users run it on their own machine, so most changes are about fitting it to a machine.

## Commands

```bash
uv sync --extra dev                # install (Python 3.11-3.13)
uv run pytest                      # fast tests, fake models, ~30 s. Must pass before every commit.
uv run ruff check src tests        # lint. Must pass before every commit.
uv run pytest -m models -s         # real models on sample media, also German speech (downloads ~8 GB, slow on CPU)
uv run clipdex doctor              # check ffmpeg, torch device, speech backend
uv run clipdex setup LIB           # detect hardware, write LIB/.clipdex/config.toml, download models
uv run clipdex index LIB --limit 3 # test run; the last line shows "x realtime" and the peak memory
uv run clipdex search "query" LIB --explain   # channel scores and thresholds
uv run clipdex calibrate LIB --present "visual: ... @ FILE" --absent "..."  # suggest search floors
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
| `src/clipdex/framestore.py` | Stored frames: one pack file per video (default) or one JPEG per frame |
| `src/clipdex/indexer.py` | Sync with the disk, run stages per file, resume, prefetch thread |
| `src/clipdex/store.py` | SQLite schema, FTS5 triggers, float16 vector BLOBs |
| `src/clipdex/search.py` | Four channels, weak-hit filters, reciprocal rank fusion into moments |
| `src/clipdex/vectors.py` | Memory-mapped vector cache for search (`vectors/`), updated per changed file, rebuilt from SQLite |
| `src/clipdex/fts.py` | Natural language to FTS5 queries over three transcript indexes (words, Snowball stems, trigrams), German and English stop words |
| `src/clipdex/calibrate.py` | `clipdex calibrate`: probes (things in / not in the footage), floor suggestions, config writer |
| `src/clipdex/mcp_server.py` | MCP tools (mcp SDK 2.x, `MCPServer`): search, frames, transcript, scenes, similar shots |
| `tests/conftest.py` | Synthetic color videos made with ffmpeg, fake SigLIP/Qwen/Whisper |

## Rules

1. Run `uv run pytest` and `uv run ruff check src tests` after every change. Add a test for every bug you fix. Fast tests never load real models.
2. The MCP server talks JSON-RPC on stdout. Do not print to stdout in code that the server runs. Use stderr or `logging`.
3. Each indexing stage commits on its own, so a stopped run can continue. Only the main thread writes to the database. The prefetch thread only runs ffmpeg.
   The indexer runs one pass per model (SigLIP, Whisper, Qwen) and unloads each model after its pass. Keep only one model loaded at a time.
4. Stored paths are relative to the library root and keep the bytes from the file system (macOS often uses NFD). Compare user input with `library.norm()`.
5. All times are seconds from the start of the file.
6. A different embedding model makes old vectors useless. The index stores the model names in `meta`. Changing a model needs `clipdex index LIB --reembed visual` or `--reembed qwen`.
7. SigLIP text must be lower case and padded to 64 tokens (`padding="max_length"`). Other padding breaks the scores.
8. Keep `embed_qwen.py` aligned with https://github.com/QwenLM/Qwen3-VL-Embedding (right padding, last-token pooling, instruction as system message).
9. `moment_max_seconds` must be at least `scene_max_seconds` and `speech_chunk_seconds`. `Config.apply` checks this.
10. When you add or change a setting, update `clipdex.example.toml` and the README tables.
11. Stored results do not change when a setting changes. Tell the user to run `clipdex index LIB --redo STAGE` (frames, visual, speech, speech_emb or scenes).
12. SQLite is the source of truth for vectors. A `Store` method that adds, changes or deletes vectors must call `_touch`,
    so the next `bump_revision` stamps the file and the search cache reads it again.

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
   - `x realtime` on the last line (`Done: ...`) is the speed of the whole run. The indexer runs one pass
     per model (visual, speech, qwen), so a per-file line only shows the speed of one pass.
     Below 1x on Apple Silicon: check the `mps` device, then try
     `frame_max_side = 384`, `frame_interval = 3.0` or `qwen_batch = 2`.
   - The last line `Memory: ...`. On macOS add the "process peak" and the "torch MPS" and "MLX" values.
     If the sum is above 75 % of the RAM, or the run crashed with an out-of-memory error, use a smaller
     profile, lower `qwen_batch` and `siglip_batch`, or set `qwen_enabled = false`.
   - For an exact peak on macOS: `/usr/bin/time -l uv run clipdex index LIB --limit 3`.
4. **Calibrate the search.** Ask the user for 3-5 things they know are in the footage (with the mode and
   a part of the file name) and 2 things that are not. Then run:
   `uv run clipdex calibrate LIB --present "action: jemand trinkt aus einer Flasche @ IMG_2011" --present "visual: eine Katze @ IMG_2010" --absent "visual: Strand mit Palmen"`
   - It shows the best score per channel for every probe and suggests floors: `visual_min_prob` (visual),
     `scene_floor` (action), `speech_sem_floor` (speech meaning). A known thing must pass its own channel.
     A missing thing must not pass any channel.
   - Show the suggestions to the user. After the user agrees, run the same command with `--write`. It keeps
     `config.toml.bak` and writes a comment with the reason above every changed value.
   - "conflict" means that one floor cannot separate two probes. Ask the user to rephrase one of them.
   - Run the probes again to check. `clipdex calibrate LIB` without probes asks for them in the terminal.
5. **Connect the agent.** Run the `claude mcp add ...` command that `setup` printed. For Claude Desktop,
   add the server to `~/Library/Application Support/Claude/claude_desktop_config.json` and restart the app.
   Test it: ask the agent to call `index_status`.
6. **Full index.** Tell the user to run `caffeinate -i clipdex index LIB` (macOS, keeps the Mac awake)
   in their own terminal. It can take hours. Ctrl-C is safe; the same command continues.
   On a fanless MacBook Air, add `--throttle` (smaller batches, 5 s pause after each file).
7. **Report.** Give the user: profile, measured speed, measured memory, changed settings.

## Extend clipdex

- New hardware profile: add it to `hardware.PROFILES`, extend `choose_profile`, add a case to `tests/test_setup.py`.
- New embedding model of the same kind: usually only a config value. Check the vector dimension and re-calibrate the floors.
- New speech backend: implement the `WhisperBackend` protocol in `speech.py` and add it to `make_backend`.
- New MCP tool: add it in `mcp_server.py` and extend `test_mcp_tools`.
