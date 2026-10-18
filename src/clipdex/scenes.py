"""Split a clip into short scenes from its frame embeddings.

Raw camera clips rarely contain cuts, so a scene ends when the picture changes:
- a jump: the frame differs a lot from the frame before (cosine < step_similarity),
- a slow drift: the frame differs a lot from the scene's first frame (cosine < cut_similarity),
- or the scene reaches max_seconds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class Scene:
    start: float
    end: float
    frames: list[int]  # indexes into the frame list


def segment(times: Sequence[float], embs: np.ndarray, duration: float, max_seconds: float = 10.0,
            cut_similarity: float = 0.80, step_similarity: float = 0.885) -> list[Scene]:
    n = len(times)
    if n == 0:
        return []
    embs = np.asarray(embs, dtype=np.float32)
    scenes: list[Scene] = []
    first = 0
    for j in range(1, n):
        too_long = times[j] - times[first] >= max_seconds
        changed = (float(embs[j] @ embs[j - 1]) < step_similarity
                   or float(embs[j] @ embs[first]) < cut_similarity)
        if too_long or changed:
            scenes.append(Scene(times[first], times[j], list(range(first, j))))
            first = j
    end = max(duration, times[-1]) if duration else times[-1]
    if end <= times[first]:
        end = times[first] + 1.0
    scenes.append(Scene(times[first], end, list(range(first, n))))
    return scenes
