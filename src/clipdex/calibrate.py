"""'clipdex calibrate': tune the search floors with things you know are, or are not, in your footage.

A probe is one line: an optional mode, the query, and for things that are in the footage a part of
the file name after "@":

    action: jemand trinkt aus einer Flasche @ IMG_2011
    visual: eine Katze auf einem Parkplatz @ E4736
    speech: wir reden über den Winter-Arc @ IMG_2010
    Strand mit Palmen                      (a thing that is not in the footage)

Each mode has one floor: visual -> visual_min_prob, action -> scene_floor, speech -> speech_sem_floor
(the meaning of what is said; exact words have no floor). A thing that is in the footage must
pass its floor. A thing that is not in the footage should not pass any floor.
"""

from __future__ import annotations

import datetime
import math
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .search import SearchEngine

# channel: (setting, mode that it belongs to, kind of score)
FLOORS = {
    "visual": ("visual_min_prob", "visual", "prob"),
    "scenes": ("scene_floor", "action", "cos"),
    "speech_sem": ("speech_sem_floor", "speech", "cos"),
}
_MODE_CHANNEL = {mode: channel for channel, (_, mode, _) in FLOORS.items()}
_PROBE_RE = re.compile(r"^\s*(?:(visual|action|speech|auto)\s*:\s*)?(.*?)\s*(?:@\s*(.+?))?\s*$", re.I)
STEP_COS = 0.02  # the step that AGENTS.md uses for cosine floors
STEP_LOGIT = 0.5  # the same idea for the visual probability, in logit space


@dataclass
class Probe:
    query: str
    present: bool
    mode: str = "auto"
    file: str | None = None  # part of the expected file path (present things only)
    scores: dict[str, float | None] = field(default_factory=dict)  # best score per floor channel
    kept: dict[str, int] = field(default_factory=dict)  # hits that passed each channel's filters
    found: bool = False  # in the results (in the expected file, for present things)
    by_margin: set[str] = field(default_factory=set)  # channels where the baseline margin decided

    @property
    def label(self) -> str:
        where = f" @ {self.file}" if self.file else ""
        return f"{self.mode}: {self.query}{where}"


def parse_probe(text: str, present: bool) -> Probe:
    m = _PROBE_RE.match(text)
    mode, query, file = (m.group(1) or "auto").lower(), m.group(2).strip(), m.group(3)
    if not query:
        raise ValueError(f"Empty query in {text!r}")
    if file and not present:
        raise ValueError(f"A thing that is not in the footage has no file: {text!r}")
    return Probe(query, present, mode, file.strip() if file else None)


def measure(engine: SearchEngine, probe: Probe) -> Probe:
    out = engine.search(probe.query, mode=probe.mode, limit=10, path_contains=probe.file, explain=True)
    diag = out.get("channels", {})
    for channel in FLOORS:
        d = diag.get(channel)
        if not d:
            continue
        probe.scores[channel] = d.get("best_prob") if channel == "visual" else d.get("best")
        probe.kept[channel] = int(d.get("kept", 0))
        if channel != "visual" and d.get("threshold") is not None and d["threshold"] > _floor(engine, channel) + 1e-9:
            probe.by_margin.add(channel)  # the query's baseline + margin was higher than the floor
    probe.found = bool(out["results"])
    return probe


def _floor(engine: SearchEngine, channel: str) -> float:
    return float(getattr(engine.cfg, FLOORS[channel][0]))


def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def _prob(x: float) -> float:
    return 1 / (1 + math.exp(-x))


@dataclass
class Suggestion:
    setting: str
    current: float
    value: float | None  # None = keep
    reason: str


def suggest(engine: SearchEngine, probes: list[Probe]) -> list[Suggestion]:
    out = []
    for channel, (setting, _mode, kind) in FLOORS.items():
        current = _floor(engine, channel)
        present = [p for p in probes if p.present and _MODE_CHANNEL.get(p.mode) == channel
                   and p.scores.get(channel) is not None]
        leaks = [p for p in probes if not p.present and p.kept.get(channel)]
        if not present and not leaks:
            continue
        to = (lambda v: _logit(v)) if kind == "prob" else (lambda v: v)
        back = _prob if kind == "prob" else (lambda v: v)
        step = STEP_LOGIT if kind == "prob" else STEP_COS
        low = min(present, key=lambda p: p.scores[channel]) if present else None
        high = max(leaks, key=lambda p: p.scores[channel]) if leaks else None
        # A known thing counts as missed when its own channel drops it, even if another channel finds it.
        missed = [p for p in present if not p.kept.get(channel) and p.scores[channel] < current]
        margin_only = [p for p in present if not p.kept.get(channel) and p.scores[channel] >= current
                       and channel in p.by_margin]
        cur = to(current)
        if low is not None and high is not None and to(high.scores[channel]) >= to(low.scores[channel]):
            out.append(Suggestion(setting, current, None,
                                  f"conflict: '{high.label}' (not in the footage) scores "
                                  f"{high.scores[channel]:.4g}, '{low.label}' (in the footage) "
                                  f"{low.scores[channel]:.4g}. One floor cannot separate them; rephrase a query."))
            continue
        target = None
        if missed or (high is not None and to(high.scores[channel]) >= cur):
            lo = to(high.scores[channel]) if high is not None else None
            hi = to(low.scores[channel]) if low is not None else None
            if lo is not None and hi is not None:
                target = (lo + hi) / 2
            elif hi is not None:
                target = hi - step
            else:
                target = lo + step
        if target is None:
            note = ""
            if margin_only:
                note = (f" '{margin_only[0].label}' is filtered by the baseline margin "
                        f"({setting.replace('floor', 'margin')}), not by the floor.")
            out.append(Suggestion(setting, current, None, "keep: the floor separates the probes." + note))
            continue
        value = back(target)
        value = round(value, 4) if kind == "prob" else math.floor(value * 100 + 1e-9) / 100
        parts = []
        if low is not None:
            parts.append(f"lowest thing in the footage: '{low.label}' {low.scores[channel]:.4g}")
        if high is not None:
            parts.append(f"highest thing not in the footage: '{high.label}' {high.scores[channel]:.4g}")
        out.append(Suggestion(setting, current, value, "; ".join(parts)))
    return out


def write_config(config_path: Path, changes: list[Suggestion]) -> Path | None:
    """Set the changed values in config.toml with a comment line each. Keeps every other line."""
    changes = [c for c in changes if c.value is not None]
    if not changes:
        return None
    backup = None
    lines = config_path.read_text().splitlines() if config_path.exists() else []
    if config_path.exists():
        backup = config_path.with_suffix(".toml.bak")
        shutil.copy2(config_path, backup)
    today = datetime.date.today().isoformat()
    for c in changes:
        comment = f"# clipdex calibrate {today}: {c.current:g} -> {c.value:g}. {c.reason}"
        line = f"{c.setting} = {c.value!r}"
        pattern = re.compile(rf"^\s*{re.escape(c.setting)}\s*=")
        for i, old in enumerate(lines):
            if pattern.match(old):
                lines[i] = line
                lines.insert(i, comment)
                break
        else:
            lines += [comment, line]
    config_path.write_text("\n".join(lines) + "\n")
    return backup


def report(probes: list[Probe], suggestions: list[Suggestion]) -> str:
    rows = ["Probes (best score per channel; * = passed the channel's filters):"]
    for p in probes:
        cells = []
        for channel in FLOORS:
            if channel in p.scores and p.scores[channel] is not None:
                cells.append(f"{channel} {p.scores[channel]:.4g}{'*' if p.kept.get(channel) else ''}")
        state = ("found" if p.found else "NOT FOUND") if p.present else ("no results" if not p.found else "HAS RESULTS")
        rows.append(f"  [{'in ' if p.present else 'not'}] {p.label}  ->  {state}  ({', '.join(cells) or 'no scores'})")
    rows.append("")
    rows.append("Suggestions:")
    for s in suggestions:
        if s.value is None:
            rows.append(f"  {s.setting} = {s.current:g}  ({s.reason})")
        else:
            rows.append(f"  {s.setting} = {s.current:g} -> {s.value:g}  ({s.reason})")
    if not suggestions:
        rows.append("  none: give things with a mode (visual:, action:, speech:) and things that are not there.")
    return "\n".join(rows)
