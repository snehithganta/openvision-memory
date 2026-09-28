"""Reusable core for open-set visual recognition."""

from .features import DinoV3FeatureExtractor
from .memory import OpenSetRecognizer, UNKNOWN_LABEL
from .clustering import cluster_embeddings
from .evaluation import evaluate_open_set
from .artifacts import save_embeddings

__all__ = [
    "DinoV3FeatureExtractor",
    "OpenSetRecognizer",
    "UNKNOWN_LABEL",
    "cluster_embeddings",
    "evaluate_open_set",
    "save_embeddings",
]
