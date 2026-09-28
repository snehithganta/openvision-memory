"""Embedding artifact persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


def save_embeddings(
    path: str | Path,
    embeddings: np.ndarray,
    labels: Iterable[object] | None = None,
    *,
    metadata: dict | None = None,
) -> None:
    """Save embeddings, optional labels, and reproducibility metadata in NPZ."""
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("embeddings must be a finite 2D array")
    payload: dict[str, np.ndarray] = {"embeddings": x}
    if labels is not None:
        y = np.asarray(list(labels))
        if len(y) != len(x):
            raise ValueError("embeddings and labels must have equal length")
        payload["labels"] = y
    payload["metadata_json"] = np.array(json.dumps(metadata or {}, default=str))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **payload)
