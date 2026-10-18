"""SigLIP 2 image/text embeddings for per-frame visual search."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from .devices import pick_device, pick_dtype


def _features(out):
    """transformers 4.x returns a tensor, 5.x a model output with pooler_output."""
    import torch

    if torch.is_tensor(out):
        return out
    pooled = getattr(out, "pooler_output", None)
    if pooled is None:
        raise RuntimeError("SigLIP returned no pooled features")
    return pooled


class SiglipEncoder:
    def __init__(self, model_name: str, device: str = "auto"):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)
        first = pick_device(device)
        attempts = [(first, pick_dtype(first))]
        if attempts[0][1] != torch.float32:
            attempts.append((first, torch.float32))  # float16 can overflow on some machines
        last: Exception | None = None
        for dev, dt in attempts:
            try:
                self.device, self.dtype = dev, dt
                self.model = AutoModel.from_pretrained(model_name, dtype=dt).to(dev).eval()
                self.logit_scale = float(self.model.logit_scale.detach().float().exp().item())
                self.logit_bias = float(self.model.logit_bias.detach().float().item())
                txt = self.encode_texts(["a test"])
                img = self._encode_pil([Image.new("RGB", (64, 64), (90, 120, 150))])
                if np.isfinite(txt).all() and np.isfinite(img).all():
                    self.dim = int(txt.shape[1])
                    return
                last = RuntimeError("NaN output")
            except Exception as exc:  # noqa: BLE001
                last = exc
        raise RuntimeError(f"SigLIP does not run on this machine: {last}") from last

    def _norm(self, t) -> np.ndarray:
        import torch

        t = torch.nn.functional.normalize(t.float(), dim=-1)
        return t.cpu().numpy().astype(np.float32)

    def _encode_pil(self, images) -> np.ndarray:
        import torch

        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device, dtype=self.dtype) if v.is_floating_point() else v.to(self.device)
                  for k, v in inputs.items()}
        with torch.no_grad():
            return self._norm(_features(self.model.get_image_features(**inputs)))

    def encode_images(self, paths: Sequence[Path | str], batch: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(paths), batch):
            images = []
            for p in paths[i:i + batch]:
                with Image.open(p) as im:
                    images.append(im.convert("RGB"))
            out.append(self._encode_pil(images))
        return np.concatenate(out) if out else np.zeros((0, getattr(self, "dim", 0)), np.float32)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        # SigLIP was trained on lower-case text padded to 64 tokens; other padding breaks scores.
        inputs = self.processor(text=[t.lower() for t in texts], padding="max_length", max_length=64,
                                truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            return self._norm(_features(self.model.get_text_features(**inputs)))

    def logits(self, cos: np.ndarray) -> np.ndarray:
        return cos * self.logit_scale + self.logit_bias

    @staticmethod
    def prob(logit: float) -> float:
        return 1.0 / (1.0 + math.exp(-logit))
