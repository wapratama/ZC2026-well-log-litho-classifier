"""Competition penalty metric (FORCE 2020) and sklearn scorer."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import make_scorer

from .config import N_CLASSES


def load_penalty_matrix(path: str | Path = "data/raw/penalty_matrix.npy") -> np.ndarray:
    matrix = np.load(path)
    if matrix.shape != (N_CLASSES, N_CLASSES):
        raise ValueError(f"Expected a {N_CLASSES}x{N_CLASSES} penalty matrix, got {matrix.shape}")
    return matrix


def penalty_score(y_true, y_pred, penalty_matrix: np.ndarray) -> float:
    """Mean negative penalty; 0 is perfect, more negative is worse.

    Indexing follows the official scoring script: penalty_matrix[true, predicted].
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return float(-penalty_matrix[y_true, y_pred].mean())


def make_penalty_scorer(penalty_matrix: np.ndarray):
    """sklearn scorer (greater is better) for use in cross-validation and searches."""
    return make_scorer(penalty_score, penalty_matrix=penalty_matrix)
