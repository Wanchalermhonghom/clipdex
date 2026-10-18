"""MCP server: lets Claude search the footage index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from mcp.server.mcpserver import Image, MCPServer

from .search import SearchEngine

INSTRUCTIONS = """\
Search a local video footage library that was indexed by clipdex.

Every result is a moment: a file plus a time range in seconds from the start of that file
(start, end, best).

Where the file is: "file" is the path inside the library folder, "volume" is the drive name
(e.g. "T7"; null for the system disk). If "available" is true, "path" is the absolute path.
If "available" is false, the drive is not mounted and "path" is null. Search, view_frames,
get_transcript and get_scenes still work, because they read the index. Ask the user to connect
the drive named in "volume" before anything opens or cuts the file.

How to search well:
- Describe what is visible or what happens in plain words, in English or German.
- mode="speech" finds what was said. Put exact spoken words in double quotes, e.g. "claude code".
- mode="visual" finds what is visible in a single frame (objects, places, light, framing).
- mode="action" finds what happens over a few seconds (someone walks in, pours coffee, waves).
- mode="auto" uses all channels. The "matched" list in each result shows which channels agreed.
- Use path_contains to stay inside one folder or project.
- Weak matches are filtered out, so an empty result means "probably not in the footage".
- Check important hits with view_frames before you cut. Use get_transcript to find exact
  sentence boundaries for speech cuts, and get_scenes to cut on picture changes.
"""


def _dump(data) -> str:
    return json.dumps(data, ensure_ascii=False, indent=1)


def build_server(engine: SearchEngine) -> MCPServer:
    server = MCPServer("clipdex", instructions=INSTRUCTIONS)

    @server.tool()
    def search_footage(
        query: str,
        mode: Literal["auto", "visual", "action", "speech"] = "auto",
        limit: int = 10,
        path_contains: str | None = None,
        max_per_file: int | None = None,
    ) -> str:
        """Search the footage library. Returns moments (file + start/end seconds) as JSON.

        Each result also has volume, available and path (null when the drive is not mounted).

        query: what you look for, e.g. "laptop on a wooden desk", "ich laufe durch den Markt",
               or "\\"claude code\\"" for exact spoken words.
        mode: auto | visual | action | speech.
        limit: number of moments to return (1-50).
        path_contains: only search files whose path contains this text (case-insensitive).
        max_per_file: at most this many moments from one file (default: the max_per_file setting).
        """
        limit = max(1, min(int(limit), 50))
        per_file = None if max_per_file is None else max(1, int(max_per_file))
        return _dump(engine.search(query, mode=mode, limit=limit, path_contains=path_contains,
                                   max_per_file=per_file))

    @server.tool()
    def view_frames(frame_ids: list[int]) -> list:
        """Show stored frames (JPEG thumbnails) so you can check a result visually.

        frame_ids: up to 8 frame_id values from search_footage or similar_shots results.
        """
        out: list = []
        for info in engine.frame_info([int(f) for f in frame_ids[:8]]):
            if "error" in info:
                out.append(f"frame {info['frame_id']}: {info['error']}")
                continue
            out.append(f"frame {info['frame_id']}: {info['file']} at {info['at_tc']} ({info['at']} s)")
            if Path(info["image"]).exists():
                out.append(Image(path=info["image"]))
            else:
                out.append("(image file is missing)")
        return out

    @server.tool()
    def get_transcript(file: str, start: float | None = None, end: float | None = None,
                       words: bool = False) -> str:
        """Return the transcript of one file with timestamps (seconds from file start).

        file: path as shown in results (a unique end of the path or part of the file name is enough).
        start, end: optional time range in seconds.
        words: also return word-level timestamps.
        """
        return _dump(engine.transcript(file, start, end, words))

    @server.tool()
    def get_scenes(file: str, changes_only: bool = False) -> str:
        """List the scenes of one file: start/end seconds, why each scene starts, and a frame_id.

        cut is "start", "picture change" or "length limit" (no visible change; the scene got long).
        changes_only=true returns only boundaries where the picture really changes. Use this for
        edits that cut on every scene change.
        """
        return _dump(engine.scenes(file, changes_only))

    @server.tool()
    def similar_shots(frame_id: int, limit: int = 10, path_contains: str | None = None) -> str:
        """Find shots that look like a given frame (visual similarity, other moments only)."""
        return _dump(engine.similar_frames(frame_id, max(1, min(int(limit), 50)), path_contains))

    @server.tool()
    def list_files(path_contains: str | None = None, limit: int = 200) -> str:
        """List indexed files with duration, resolution and detected speech language."""
        return _dump(engine.list_files(path_contains, max(1, min(int(limit), 2000))))

    @server.tool()
    def index_status() -> str:
        """Show index statistics: files, hours of footage, stages done, failed files, drive mounted or not."""
        return _dump(engine.status())

    return server
