"""Cleaning and feature engineering for wireline-log lithology prediction.

The same functions are used in the notebook, in train.py and in the API, so
training and serving cannot drift apart. All depth-context features are computed
inside a single well, never across wells.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    ALL_LOGS, DEPTH_COL, LOG10_LOGS, MODEL_INPUT_CURVES, RAW_FEATURE_LOGS, ROLL_LOGS,
    SENTINEL_MAX, WELL_COL, WINDOWS,
)

DERIVED_FEATURES = ["PHID", "ND_SEP", "GR_NORM", "GR_GRAD", "RHOB_GRAD"]
LOG10_FEATURES = [f"LOG_{c}" for c in LOG10_LOGS]
ROLLING_FEATURES = [
    f"{c}_{stat}{w}" for w in WINDOWS for c in ROLL_LOGS for stat in ("MEAN", "STD")
]
FEATURES_LOGS = RAW_FEATURE_LOGS + LOG10_FEATURES + DERIVED_FEATURES + ROLLING_FEATURES
FEATURES_LOGS_DEPTH = FEATURES_LOGS + [DEPTH_COL]

FEATURE_SETS: dict[str, list[str]] = {
    "A_logs": FEATURES_LOGS,
    "B_logs_depth": FEATURES_LOGS_DEPTH,
}


def clean_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by well/depth, make every curve numeric, and turn LAS null sentinels into NaN.

    Curves the model needs but the input lacks become all-NaN columns, and curves that
    arrive as all-null (object dtype, e.g. from JSON) are coerced to float.
    """
    out = df.sort_values([WELL_COL, DEPTH_COL], kind="stable").reset_index(drop=True)
    for col in MODEL_INPUT_CURVES:
        if col not in out.columns:
            out[col] = np.nan
    for col in (c for c in ALL_LOGS if c in out.columns):
        numeric = pd.to_numeric(out[col], errors="coerce").astype("float64")
        out[col] = numeric.where(numeric > SENTINEL_MAX)
    return out


def _well_features(g: pd.DataFrame) -> pd.DataFrame:
    feats: dict[str, pd.Series] = {}
    for col in LOG10_LOGS:
        positive = g[col].where(g[col] > 0)
        feats[f"LOG_{col}"] = np.log10(positive).clip(-2, 4)

    phid = ((2.65 - g["RHOB"]) / 1.65).clip(-0.1, 1.0)   # density porosity, 2.65 matrix
    feats["PHID"] = phid
    feats["ND_SEP"] = g["NPHI"] - phid                    # neutron-density separation

    lo, hi = g["GR"].quantile([0.05, 0.95])
    span = hi - lo
    feats["GR_NORM"] = ((g["GR"] - lo) / span).clip(-0.5, 2.0) if span > 0 else g["GR"] * np.nan
    feats["GR_GRAD"] = g["GR"].diff(10)
    feats["RHOB_GRAD"] = g["RHOB"].diff(10)

    for w in WINDOWS:
        roll = g[ROLL_LOGS].rolling(w, center=True, min_periods=3)
        for stat, frame in (("MEAN", roll.mean()), ("STD", roll.std())):
            for col in ROLL_LOGS:
                feats[f"{col}_{stat}{w}"] = frame[col]
    return pd.DataFrame(feats, index=g.index)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append engineered features. Expects the output of clean_logs()."""
    parts = [_well_features(g) for _, g in df.groupby(WELL_COL, sort=False)]
    return pd.concat([df, pd.concat(parts)], axis=1)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """clean_logs + add_features in one call."""
    return add_features(clean_logs(df))
