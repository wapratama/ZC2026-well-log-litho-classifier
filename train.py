"""Train the final lithology model and save it for serving.

Reads the configuration chosen in notebook.ipynb (models/best_config.json), trains on the
training wells, optionally re-checks the score on the hidden-test wells, and writes
models/model.joblib.

Usage (from the project root):
    uv run python train.py                # same sampling density as the notebook's final fit
    uv run python train.py --stride 1     # use every depth sample (slower)
"""
from __future__ import annotations

import argparse
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn

from lithology.config import CLASS_NAMES, CODE_TO_CLASS, TARGET_COL, WELL_COL
from lithology.metrics import load_penalty_matrix, penalty_score
from lithology.modeling import build_estimator, build_feature_matrix, load_config

log = logging.getLogger("train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("models/best_config.json"))
    p.add_argument("--train-csv", type=Path, default=Path("data/raw/train.csv"))
    p.add_argument("--hidden-csv", type=Path, default=Path("data/raw/hidden_test.csv"))
    p.add_argument("--penalty-matrix", type=Path, default=Path("data/raw/penalty_matrix.npy"))
    p.add_argument("--output", type=Path, default=Path("models/model.joblib"))
    p.add_argument("--stride", type=int, default=None,
                   help="keep every k-th sample per well (default: stride_final from the config)")
    p.add_argument("--skip-eval", action="store_true", help="do not score the hidden-test wells")
    return p.parse_args(argv)


def load_training_data(path: Path, features: list[str], stride: int) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: uv run python download_data.py")
    raw = pd.read_csv(path, sep=";")
    frame, X = build_feature_matrix(raw, features)
    y = frame[TARGET_COL].map(CODE_TO_CLASS)
    if y.isna().any():
        raise ValueError("Training data contains lithology codes that are not in CODE_TO_CLASS")
    keep = (frame.groupby(WELL_COL).cumcount() % stride == 0).to_numpy()
    log.info("training wells: %d | rows: %d -> %d after stride %d",
             frame[WELL_COL].nunique(), len(frame), int(keep.sum()), stride)
    return X[keep], y[keep].astype(int)


def evaluate_hidden(model, features: list[str], hidden_csv: Path, penalty_matrix_path: Path) -> dict | None:
    if not hidden_csv.exists():
        log.warning("%s not found; skipping hidden-test check", hidden_csv)
        return None
    frame, X = build_feature_matrix(pd.read_csv(hidden_csv, sep=";"), features)
    y_true = frame[TARGET_COL].map(CODE_TO_CLASS).to_numpy(dtype=int)
    pred = model.predict(X)
    matrix = load_penalty_matrix(penalty_matrix_path)
    return {"penalty": penalty_score(y_true, pred, matrix),
            "accuracy": float((pred == y_true).mean()), "rows": int(len(y_true))}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    try:
        config = load_config(args.config)
        features: list[str] = config["features"]
        stride = args.stride if args.stride is not None else int(config["settings"]["stride_final"])
        if stride < 1:
            raise ValueError("--stride must be >= 1")
        log.info("model: %s | feature set: %s (%d features)",
                 config["selected_model"], config["feature_set"], len(features))

        X, y = load_training_data(args.train_csv, features, stride)
        model = build_estimator(config["selected_model"], config["best_params"])
        t0 = time.time()
        model.fit(X, y)
        log.info("fit finished in %.0fs", time.time() - t0)

        hidden = None if args.skip_eval else evaluate_hidden(model, features, args.hidden_csv, args.penalty_matrix)
        if hidden:
            log.info("hidden-test penalty: %.4f (notebook reported %.4f) | accuracy %.3f",
                     hidden["penalty"], config["hidden_test"]["penalty"], hidden["accuracy"])
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1

    artifact = {
        "model": model,
        "features": features,
        "class_names": CLASS_NAMES,
        "feature_set": config["feature_set"],
        "metadata": {
            "model_name": config["selected_model"],
            "best_params": config["best_params"],
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_train_rows": int(len(X)),
            "stride": stride,
            "cv_penalty": config.get("cv_penalty"),
            "hidden_test_penalty": hidden["penalty"] if hidden else None,
            "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                         "lightgbm": lightgbm.__version__, "numpy": np.__version__, "pandas": pd.__version__},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output, compress=3)
    log.info("saved %s (%.1f MB)", args.output, args.output.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
