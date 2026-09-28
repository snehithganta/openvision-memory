"""Dataset-independent open-set evaluation metrics."""

from __future__ import annotations

from time import perf_counter
from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .memory import OpenSetRecognizer, UNKNOWN_LABEL


def evaluate_open_set(
    recognizer: OpenSetRecognizer,
    embeddings: np.ndarray,
    labels: Iterable[object],
) -> dict[str, float | int | None]:
    """Evaluate known accuracy, unknown detection, macro F1, and prediction latency."""
    y = np.asarray(list(labels))
    if len(y) != len(embeddings):
        raise ValueError("embeddings and labels must have equal length")
    start = perf_counter()
    predictions, scores = recognizer.predict(embeddings)
    elapsed = perf_counter() - start
    known = y != UNKNOWN_LABEL
    unknown = ~known
    known_accuracy = float(accuracy_score(y[known], predictions[known])) if known.any() else None
    unknown_f1 = float(f1_score(unknown, predictions == UNKNOWN_LABEL, zero_division=0))
    try:
        auroc = float(roc_auc_score(unknown.astype(np.int8), scores)) if known.any() and unknown.any() else None
    except ValueError:
        auroc = None
    return {
        "known_accuracy": known_accuracy,
        "unknown_f1": unknown_f1,
        "unknown_auroc": auroc,
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "samples": int(len(y)),
        "inference_seconds": float(elapsed),
        "inference_ms_per_sample": float(elapsed * 1000 / max(len(y), 1)),
        "rejection_threshold": float(recognizer.threshold_),
    }
