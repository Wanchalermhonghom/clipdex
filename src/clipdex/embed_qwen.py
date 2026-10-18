"""Qwen3-VL-Embedding for scene (action) and transcript embeddings.

The model wrapper and input formatting follow the official reference code at
https://github.com/QwenLM/Qwen3-VL-Embedding (Apache-2.0, Copyright Alibaba Qwen
team). Changes: Apple MPS device support, no silent fallback on preprocessing
errors, and a smaller interface.
"""

from __future__ import annotations

import gc
import logging
import os
import tempfile
import unicodedata
from collections.abc import Sequence

import numpy as np

from .devices import pick_device, pick_dtype

log = logging.getLogger(__name__)

IMAGE_FACTOR = 32  # patch size 16 x spatial merge 2
MAX_LENGTH = 8192
FRAME_MAX_PIXELS = 768 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_TOTAL_PIXELS = 10 * FRAME_MAX_PIXELS
DEFAULT_INSTRUCTION = "Represent the user's input."


def _model_class():
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLPreTrainedModel

    class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
        _checkpoint_conversion_mapping = {}
        accepts_loss_kwargs = False

        def __init__(self, config):
            super().__init__(config)
            self.model = Qwen3VLModel(config)
            self.post_init()

        def get_input_embeddings(self):
            return self.model.get_input_embeddings()

        def set_input_embeddings(self, value):
            self.model.set_input_embeddings(value)

        def forward(self, input_ids=None, attention_mask=None, pixel_values=None,
                    pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None, **kwargs):
            out = self.model(
                input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos, image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw, **kwargs)
            return out.last_hidden_state

    return Qwen3VLForEmbedding


class QwenEmbedder:
    def __init__(self, model_name: str, device: str = "auto", dtype: str = "auto",
                 max_frames: int = 8):
        import torch
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

        self.model_name = model_name
        self.max_frames = max_frames
        self.processor = Qwen3VLProcessor.from_pretrained(model_name, padding_side="right")
        first_device = pick_device(device)
        attempts = [(first_device, pick_dtype(first_device, dtype, large_model=True))]
        if dtype == "auto" and attempts[0][1] != torch.float32:
            attempts.append((first_device, torch.float32))
        if device == "auto" and first_device != "cpu":
            attempts.append(("cpu", torch.float32))
        self.model = None
        last: Exception | None = None
        for dev, dt in attempts:
            try:
                self._load(dev, dt)
                self._selftest()
                return
            except Exception as exc:  # noqa: BLE001 - try the next device / dtype
                last = exc
                log.warning("Qwen embedding on %s/%s failed (%s). Trying the next option.", dev, dt, exc)
                self.model = None
                gc.collect()
        raise RuntimeError(f"Qwen embedding model does not run on this machine: {last}") from last

    def _load(self, device: str, dtype) -> None:
        cls = _model_class()
        self.device, self.dtype = device, dtype
        self.model = cls.from_pretrained(self.model_name, dtype=dtype).to(device).eval()
        self.dim = int(self.model.config.text_config.hidden_size)

    def _selftest(self) -> None:
        """Embed a tiny video and a text once. Catches unsupported ops and NaN output early."""
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in range(2):
                path = os.path.join(tmp, f"{i}.jpg")
                Image.new("RGB", (64, 64), (60 * i, 90, 140)).save(path)
                paths.append(path)
            vec = self.embed_frame_sequences([paths])
        txt = self.embed_texts(["a short test"])
        if not (np.isfinite(vec).all() and np.isfinite(txt).all()):
            raise RuntimeError("the model returned NaN or Inf values")

    # --- input formatting (official reference behaviour) ----------------------------------
    @staticmethod
    def _instruction(text: str | None) -> str:
        text = (text or DEFAULT_INSTRUCTION).strip()
        if text and not unicodedata.category(text[-1]).startswith("P"):
            text += "."
        return text

    def _conversation(self, instruction: str | None, text: str | None = None,
                      frames: Sequence[str] | None = None) -> list[dict]:
        content: list[dict] = []
        if frames:
            frames = list(frames)
            if len(frames) > self.max_frames:
                idx = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int).tolist()
                frames = [frames[i] for i in idx]
            content.append({
                "type": "video",
                # a file path, or a PIL image (frames from a pack file)
                "video": ["file://" + f if isinstance(f, str) else f for f in frames],
                "total_pixels": MAX_TOTAL_PIXELS,
            })
        if text is not None:
            content.append({"type": "text", "text": text})
        if not content:
            content.append({"type": "text", "text": "NULL"})
        return [
            {"role": "system", "content": [{"type": "text", "text": self._instruction(instruction)}]},
            {"role": "user", "content": content},
        ]

    def _embed(self, conversations: list[list[dict]]) -> np.ndarray:
        import torch
        from qwen_vl_utils.vision_process import process_vision_info

        text = self.processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
        images, video_inputs, video_kwargs = process_vision_info(
            conversations, image_patch_size=16, return_video_metadata=True, return_video_kwargs=True)
        videos = metadata = None
        if video_inputs is not None:
            videos, metadata = (list(x) for x in zip(*video_inputs, strict=True))
        inputs = self.processor(
            text=text, images=images, videos=videos, video_metadata=metadata, truncation=True,
            max_length=MAX_LENGTH, padding=True, do_resize=False, return_tensors="pt", **video_kwargs)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            hidden = self.model(**inputs)
        mask = inputs["attention_mask"]
        last = mask.shape[1] - mask.flip(dims=[1]).argmax(dim=1) - 1  # last real token (right padding)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        emb = torch.nn.functional.normalize(hidden[rows, last].float(), p=2, dim=-1)
        return emb.cpu().numpy().astype(np.float32)

    # --- public API ------------------------------------------------------------------------------
    def embed_frame_sequences(self, sequences: Sequence[Sequence[str]], batch: int = 4) -> np.ndarray:
        """Embed each list of frame image paths as one short video."""
        out = []
        for i in range(0, len(sequences), batch):
            out.append(self._embed([self._conversation(None, frames=s) for s in sequences[i:i + batch]]))
        return np.concatenate(out) if out else np.zeros((0, self.dim), np.float32)

    def embed_texts(self, texts: Sequence[str], instruction: str | None = None, batch: int = 16) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch):
            out.append(self._embed([self._conversation(instruction, text=t) for t in texts[i:i + batch]]))
        return np.concatenate(out) if out else np.zeros((0, self.dim), np.float32)
