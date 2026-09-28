"""Optional HDBSCAN exploration for normalized feature embeddings."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import normalize


def cluster_embeddings(
    embeddings: np.ndarray,
    *,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray]:
    """Return HDBSCAN cluster IDs and membership strengths.

    HDBSCAN uses Euclidean distance on unit-normalized embeddings by default,
    which preserves the simple dependency path while providing a common baseline.
    The `hdbscan` extra must be installed separately.
    """
    try:
        from hdbscan import HDBSCAN
    except ImportError as exc:
        raise ImportError("Install optional clustering support with `pip install -e '.[clustering]'`") from exc
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("embeddings must be a non-empty finite 2D array")
    if min_cluster_size < 2:
        raise ValueError("min_cluster_size must be at least 2")
    model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric)
    model.fit(normalize(x))
    return model.labels_.astype(np.int64), model.probabilities_.astype(np.float32)
