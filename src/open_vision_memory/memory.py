"""Nearest-prototype open-set recognition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.preprocessing import normalize

UNKNOWN_LABEL = -1


class OpenSetRecognizer:
    """Cosine nearest-prototype classifier with a validation-calibrated reject threshold."""

    def __init__(self, *, threshold_quantile: float = 0.95) -> None:
        if not 0.0 < threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be between 0 and 1")
        self.threshold_quantile = float(threshold_quantile)
        self.labels_: np.ndarray | None = None
        self.prototypes_: np.ndarray | None = None
        self.threshold_: float | None = None

    def fit(self, embeddings: np.ndarray, labels: Iterable[object]) -> "OpenSetRecognizer":
        x = self._validate_embeddings(embeddings)
        y = np.asarray(list(labels))
        if len(x) != len(y) or len(y) == 0:
            raise ValueError("embeddings and labels must have equal, non-zero length")
        classes = np.unique(y)
        if np.any(classes == UNKNOWN_LABEL):
            raise ValueError(f"training labels may not contain reserved unknown label {UNKNOWN_LABEL}")
        protos = []
        for label in classes:
            center = x[y == label].mean(axis=0, keepdims=True)
            protos.append(normalize(center)[0])
        self.labels_ = classes
        self.prototypes_ = np.stack(protos).astype(np.float32)
        self.threshold_ = None
        return self

    def calibrate(
        self,
        embeddings: np.ndarray,
        labels: Iterable[object],
        *,
        known_labels: Iterable[object] | None = None,
    ) -> float:
        """Set reject threshold from nearest-prototype distances of known validation samples."""
        self._require_fit()
        x = self._validate_embeddings(embeddings)
        y = np.asarray(list(labels))
        if len(x) != len(y):
            raise ValueError("embeddings and labels must have equal length")
        allowed = self.labels_ if known_labels is None else np.asarray(list(known_labels))
        mask = np.isin(y, allowed)
        if not np.any(mask):
            raise ValueError("validation data contains no samples from known classes")
        distances = self._distances(x[mask]).min(axis=1)
        self.threshold_ = float(np.quantile(distances, self.threshold_quantile))
        return self.threshold_

    def predict(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return labels and unknown scores (nearest cosine distance; higher is less familiar)."""
        self._require_fit()
        if self.threshold_ is None:
            raise RuntimeError("call calibrate() before predict()")
        x = self._validate_embeddings(embeddings)
        distances = self._distances(x)
        nearest = distances.argmin(axis=1)
        scores = distances[np.arange(len(x)), nearest]
        predictions = self.labels_[nearest].copy()
        predictions[scores > self.threshold_] = UNKNOWN_LABEL
        return predictions, scores

    def save(self, path: str | Path) -> None:
        self._require_fit()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            labels=self.labels_,
            prototypes=self.prototypes_,
            threshold=np.array(np.nan if self.threshold_ is None else self.threshold_),
            threshold_quantile=np.array(self.threshold_quantile),
        )

    @classmethod
    def load(cls, path: str | Path) -> "OpenSetRecognizer":
        with np.load(path, allow_pickle=False) as data:
            # Experiment artifacts predate the packaged API and store only
            # labels, prototypes, and the calibrated threshold.
            quantile = float(data["threshold_quantile"]) if "threshold_quantile" in data else 0.95
            model = cls(threshold_quantile=quantile)
            model.labels_ = data["labels"]
            model.prototypes_ = data["prototypes"].astype(np.float32)
            threshold = float(data["threshold"]) if "threshold" in data else float("nan")
            model.threshold_ = None if np.isnan(threshold) else threshold
        return model

    def save_config(self, path: str | Path, *, extra: dict | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        config = {"method": "cosine_nearest_prototype", "threshold_quantile": self.threshold_quantile,
                  "threshold": self.threshold_, "labels": self.labels_.tolist() if self.labels_ is not None else None}
        if extra:
            config["extra"] = extra
        target.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    def _distances(self, x: np.ndarray) -> np.ndarray:
        return np.clip(1.0 - x @ self.prototypes_.T, 0.0, 2.0)

    @staticmethod
    def _validate_embeddings(embeddings: np.ndarray) -> np.ndarray:
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("embeddings must be a finite 2D array")
        return normalize(x).astype(np.float32, copy=False)

    def _require_fit(self) -> None:
        if self.labels_ is None or self.prototypes_ is None:
            raise RuntimeError("call fit() before calibration or inference")
