"""Pick a torch device and dtype."""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=None)
def pick_device(preference: str = "auto") -> str:
    import torch

    if preference != "auto":
        return preference
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@functools.lru_cache(maxsize=None)
def _bf16_works(device: str) -> bool:
    import torch

    try:
        a = torch.ones((8, 8), dtype=torch.bfloat16, device=device)
        float((a @ a).sum().item())
        return True
    except Exception:  # noqa: BLE001 - any failure means "not usable"
        return False


def pick_dtype(device: str, preference: str = "auto", *, large_model: bool = False):
    """float16 for small models on GPU, bfloat16 for the large model, float32 on CPU."""
    import torch

    names = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if preference != "auto":
        return names[preference]
    if device == "cpu":
        # bfloat16 halves memory for the 2B/8B model; CPU matmul still works.
        return torch.bfloat16 if large_model else torch.float32
    if large_model:
        return torch.bfloat16 if _bf16_works(device) else torch.float32
    return torch.float16
