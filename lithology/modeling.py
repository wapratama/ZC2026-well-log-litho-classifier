"""Model construction and feature-matrix building shared by train.py and predict.py."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

from .features import prepare

SEED = 42
REQUIRED_CONFIG_KEYS = ("selected_model", "feature_set", "features", "best_params", "settings")


def load_config(path: str | Path) -> dict:
    """Read the configuration the notebook saved (models/best_config.json)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run notebook.ipynb through section 12 first.")
    config = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in config]
    if missing:
        raise ValueError(f"{path} is missing keys: {missing}")
    return config


def build_estimator(name: str, params: dict):
    """Rebuild the estimator selected in the notebook, with its tuned hyper-parameters.

    The base settings mirror notebook section 9/10 exactly (including subsample_freq=1
    for LightGBM, which the search used so that `subsample` takes effect).
    """
    if name == "LightGBM":
        base = dict(objective="multiclass", n_estimators=200, learning_rate=0.1, num_leaves=31,
                    random_state=SEED, n_jobs=-1, verbose=-1, subsample_freq=1)
        base.update(params)
        return lgb.LGBMClassifier(**base)

    impute = SimpleImputer(strategy="median", add_indicator=True)
    if name == "Random Forest":
        pipe = Pipeline([
            ("impute", impute),
            ("clf", RandomForestClassifier(n_estimators=150, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
        ])
    elif name == "Logistic Regression":
        pipe = Pipeline([
            ("impute", impute),
            ("scale", QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=SEED)),
            ("clf", LogisticRegression(max_iter=200, C=1.0)),
        ])
    else:
        raise ValueError(f"Unknown model name: {name!r}")
    return pipe.set_params(**params)


def build_feature_matrix(df_raw: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw log table -> (full prepared frame, float32 model matrix with exactly `features`).

    Casting to float32 matches the notebook, so training and serving see identical numbers.
    """
    frame = prepare(df_raw)
    return frame, frame[features].astype("float32")
