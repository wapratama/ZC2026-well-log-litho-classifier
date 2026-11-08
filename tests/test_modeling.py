import pytest

from lithology.modeling import build_estimator


def test_lightgbm_uses_subsample_freq_so_subsample_takes_effect():
    model = build_estimator("LightGBM", {"subsample": 0.8, "num_leaves": 63})
    assert model.get_params()["subsample_freq"] == 1
    assert model.get_params()["num_leaves"] == 63


def test_pipeline_params_are_applied():
    model = build_estimator("Random Forest", {"clf__max_depth": 12})
    assert model.named_steps["clf"].max_depth == 12


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_estimator("XGBoost", {})
