# clipdex

Search your own video footage with local AI. Find a clip by what is said, what is visible or what happens.
clipdex runs on your computer. No video leaves your machine.
Claude and other AI agents can search the index through MCP.

Every result is a **moment**: a file plus a start and end time in seconds.

[Deutsche Version](README.de.md)

## What clipdex finds

| Channel | Model | Finds |
|---|---|---|
| `visual` | [SigLIP 2](https://huggingface.co/google/siglip2-so400m-patch14-384) | What one frame shows: objects, places, light |
| `scenes` | [Qwen3-VL-Embedding](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) | What happens in a few seconds: actions |
| `speech` | [Whisper large-v3-turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo) + SQLite FTS5 | The exact spoken words, with word timestamps |
| `speech_sem` | Qwen3-VL-Embedding | What the speaker talks about, also in other words |

Queries work in English, German and the other languages that the models know.

## Quick start (Apple Silicon Mac, macOS 14 or newer)

```bash
brew install ffmpeg uv
uv tool install git+https://github.com/Wanchalermhonghom/clipdex
clipdex setup /Volumes/MySSD          # detects your Mac, writes a config, downloads the models
clipdex index /Volumes/MySSD --limit 3   # test with 3 files
clipdex index /Volumes/MySSD             # index everything (stop with Ctrl-C, run again to continue)
clipdex search "someone pours coffee" /Volumes/MySSD --mode action
```

Linux with an NVIDIA GPU or only a CPU should also work: install ffmpeg and [uv](https://docs.astral.sh/uv/) first. The tests run on Linux and macOS. Windows is not tested yet.

## How the index is built

For every file, clipdex does these steps:

1. `ffprobe` reads the duration, the resolution and the audio track.
2. `ffmpeg` saves one frame every 2 seconds (512 px). It decodes only keyframes first, because that is fast. If the keyframes are too far apart, it decodes the full file.
3. SigLIP 2 makes one vector per frame.
4. [Silero VAD](https://github.com/snakers4/silero-vad) finds the speech. Whisper transcribes only these parts. Files without speech are skipped.
5. Whisper detects the language once per file, where the speech starts. This is a quick detection pass, not a full transcription.
   To skip it, set `language = "de"` in the config. For footage in German and English, keep `language = "auto"`
   and set `languages = ["de", "en"]`. Then clipdex picks only between these two.
6. clipdex splits each file into short scenes (10 s or less). A new scene starts when the picture changes a lot.
7. Qwen3-VL-Embedding makes one vector per scene and per transcript part (20 s).

Everything goes into one SQLite file and a folder of small JPEG frames: about 55 MB per hour of footage.

## Fit clipdex to your machine

`clipdex setup` detects the chip, the memory and the GPU. Then it picks a profile:

| Profile | Machine | Models | Peak memory (estimate) |
|---|---|---|---|
| `apple-small` | Apple Silicon, < 16 GB | SigLIP 2 base, no Qwen | indexer ~4 GB |
| `apple-medium` | Apple Silicon, 16–47 GB | SigLIP 2 so400m, Qwen 2B | indexer ~10 GB, MCP ~8 GB |
| `apple-large` | Apple Silicon, ≥ 48 GB | SigLIP 2 so400m, Qwen 8B | indexer ~24 GB, MCP ~21 GB |
| `nvidia-small/medium/large` | NVIDIA GPU by VRAM | as above | as above |
| `cpu` | no GPU | SigLIP 2 base, Whisper small, no Qwen | indexer ~3 GB |

The memory numbers come from the model sizes. They are not measurements on your machine.
The indexer prints its real peak memory at the end of each run.

**Let your AI coding agent do it.** Open the repository in Claude Code, Codex or Cursor and say
"set up clipdex for this machine". The agent follows the procedure in [AGENTS.md](AGENTS.md):
check the tools, pick the profile, test 3 files, measure speed and memory, tune the search, connect MCP.
In Claude Code you can also type `/clipdex-setup`.

## Search in the terminal

```bash
clipdex search "laptop on a wooden desk" /Volumes/MySSD
clipdex search '"claude code"' /Volumes/MySSD --mode speech        # exact words in double quotes
clipdex search "Sonnenuntergang am Strand" /Volumes/MySSD --in Thailand
clipdex transcript C0012.MP4 /Volumes/MySSD
clipdex scenes C0012.MP4 /Volumes/MySSD --changes-only               # every picture change
clipdex search "red car" /Volumes/MySSD --json                     # for scripts
```

Modes: `auto` (all channels, the default), `visual`, `action` and `speech`.

## Connect Claude (MCP)

`clipdex setup` prints the exact command for your machine. It looks like this:

```bash
claude mcp add --transport stdio --scope user clipdex -- clipdex mcp /Volumes/MySSD
```

For Claude Desktop, add this to `~/Library/Application Support/Claude/claude_desktop_config.json`.
Use the full path from `which clipdex`:

```json
{
  "mcpServers": {
    "clipdex": { "command": "/Users/YOU/.local/bin/clipdex", "args": ["mcp", "/Volumes/MySSD"] }
  }
}
```

The MCP server gives the agent these tools:

| Tool | What it does |
|---|---|
| `search_footage` | Find moments (file, start, end, best time) |
| `view_frames` | Show the stored frames, so the agent can check a hit |
| `get_transcript` | Transcript of one file with timestamps, also per word |
| `get_scenes` | Scene boundaries of one file, with the reason for each cut |
| `similar_shots` | Find shots that look like a given frame |
| `list_files` | Files with duration, resolution and language |
| `index_status` | Statistics and failed files |

The MCP server only reads the index. You can search while the indexer runs.
On a 16–24 GB Mac, start the server with `mcp ... --no-qwen` while you index. Then it uses less memory.

## Keep the index up to date

- Run `clipdex index /Volumes/MySSD` again. It only processes new and changed files.
- Moved and renamed files keep their index entry.
- Deleted files are marked `missing`. `--prune` removes them. clipdex refuses to remove more than half of the index unless you add `--force`.
- Files that fail are skipped. `clipdex status /Volumes/MySSD` shows the errors. `--retry-failed` tries them again.
- After you change a setting, `--redo STAGE` runs that stage again, e.g. `--redo scenes` after a scene setting.
- The index lives in `/Volumes/MySSD/.clipdex`. It travels with the drive. Use `--index-dir` to keep it somewhere else.

## Tune the search

Vector search always returns a nearest match, even for nonsense. So clipdex drops weak hits per channel.
Look at the scores with `--explain`:

```bash
clipdex search "red car" /Volumes/MySSD --explain
```

Then put new values into `/Volumes/MySSD/.clipdex/config.toml`, for example `scene_floor = 0.38`.
[clipdex.example.toml](clipdex.example.toml) lists every setting with its default.

## Limits

- The default search thresholds come from a small test with sample images and the 2B model. Check them on your footage.
- Whisper picks one language per file. A file that mixes languages gets the language that its speech starts with.
  A fixed `language` transcribes speech in other languages wrongly.
- Log and HDR footage looks flat in the stored frames. Visual search can be weaker.
- ffmpeg cannot read BRAW or R3D. These files show up as failed.
- A model change needs `clipdex index ... --reembed visual` or `--reembed qwen`.

## Development

```bash
git clone https://github.com/Wanchalermhonghom/clipdex && cd clipdex
uv sync --extra dev
uv run pytest                  # fast tests with fake models, about 30 s
uv run pytest -m models -s     # real models on sample media (downloads about 6 GB)
```

Read [AGENTS.md](AGENTS.md) before you change code. It has the project map and the rules.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). The models keep their own licenses:
SigLIP 2 and Qwen3-VL-Embedding (Apache-2.0), Whisper and Silero VAD (MIT).
