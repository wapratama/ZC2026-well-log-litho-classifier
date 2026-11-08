import numpy as np
import pytest

from lithology.metrics import penalty_score


def _matrix() -> np.ndarray:
    a = np.zeros((12, 12))
    a[0, 1] = a[1, 0] = 2.0   # Sandstone <-> Sandstone/Shale
    a[0, 2] = a[2, 0] = 3.5   # Sandstone <-> Shale
    return a


def test_hand_computed_example():
    # (2.0 + 3.5 + 0) / 3 = 1.8333..., reported as a negative score
    score = penalty_score([0, 0, 2], [1, 2, 2], _matrix())
    assert score == pytest.approx(-(2.0 + 3.5) / 3)


def test_perfect_prediction_scores_zero():
    assert penalty_score([0, 1, 2, 3], [0, 1, 2, 3], _matrix()) == 0.0


def test_worse_confusions_score_lower():
    assert penalty_score([0], [2], _matrix()) < penalty_score([0], [1], _matrix())
