"""Batch feature extraction using a Hugging Face DINOv3 image model."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


class DinoV3FeatureExtractor:
    """Extract unit-length CLS-token embeddings from RGB images.

    ``model_name`` may be a Hugging Face model ID or a local model directory.
    The model and its processor are loaded lazily on construction.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        *,
        device: str | torch.device | None = None,
        batch_size: int = 32,
        use_fast: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=use_fast)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.inference_mode()
    def encode_images(self, images: Sequence[object], *, batch_size: int | None = None) -> np.ndarray:
        """Return an ``(N, D)`` float32 array of L2-normalized image embeddings."""
        size = batch_size or self.batch_size
        if size < 1:
            raise ValueError("batch_size must be positive")
        if len(images) == 0:
            hidden = int(getattr(self.model.config, "hidden_size"))
            return np.empty((0, hidden), dtype=np.float32)

        chunks: list[np.ndarray] = []
        for start in range(0, len(images), size):
            batch = [self._as_rgb(image) for image in images[start : start + size]]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            output = self.model(**inputs)
            # DINOv3 exposes the class token first in last_hidden_state.
            features = output.last_hidden_state[:, 0]
            features = F.normalize(features.float(), p=2, dim=-1)
            chunks.append(features.cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(chunks, axis=0)

    @staticmethod
    def _as_rgb(image: object):
        """Normalize common PIL/NumPy image inputs to RGB PIL images."""
        from PIL import Image

        if isinstance(image, Image.Image):
            return image.convert("RGB")
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        if array.ndim != 3 or array.shape[-1] not in (3, 4):
            raise ValueError("images must be HxW grayscale, HxWx3 RGB, or HxWx4 RGBA")
        if array.shape[-1] == 4:
            array = array[..., :3]
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating) and array.size and array.max() <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array, mode="RGB")
