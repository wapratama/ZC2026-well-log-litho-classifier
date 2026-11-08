"""FastAPI service: predict lithology from a continuous section of wireline logs.

Run locally (from the project root):
    uv run uvicorn predict:app --host 0.0.0.0 --port 8000
Interactive docs: http://localhost:8000/docs

Why a *section* and not one depth point: the model uses rolling-window context (about 2 m
and 9 m), gradients and per-well gamma-ray normalisation, all computed from neighbouring
samples. Send a continuous interval, ideally 61+ samples at ~0.15 m spacing.
"""
from __future__ import annotations

import logging
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lithology.config import DEPTH_COL, MODEL_INPUT_CURVES, TRAIN_DEPTH_STEP_M, WELL_COL
from lithology.modeling import build_feature_matrix

log = logging.getLogger("predict")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

DEFAULT_MODEL_PATH = "models/model.joblib"
MIN_SAMPLES = 15        # smallest rolling window used by the features
FULL_CONTEXT_SAMPLES = 61  # largest rolling window; fewer samples are accepted with a warning
MAX_SAMPLES = 20_000    # ~3 km of well at 0.152 m
KEY_CURVES = ("GR", "RHOB", "NPHI", "DTC", "RDEP")

Curve = Annotated[float | None, Field(default=None, description="Curve value; null if not recorded")]


class LogSample(BaseModel):
    """One depth sample. DEPTH_MD is required; every curve is optional (null = not recorded)."""

    model_config = ConfigDict(extra="forbid")

    DEPTH_MD: float = Field(description="Measured depth in metres")
    GR: Curve        # gamma ray (API)
    RHOB: Curve      # bulk density (g/cc)
    NPHI: Curve      # neutron porosity (v/v)
    DTC: Curve       # compressional sonic (us/ft)
    RDEP: Curve      # deep resistivity (ohm.m)
    RMED: Curve      # medium resistivity (ohm.m)
    RSHA: Curve      # shallow resistivity (ohm.m)
    PEF: Curve       # photoelectric factor
    CALI: Curve      # caliper (in)
    BS: Curve        # bit size (in)
    SP: Curve        # spontaneous potential (mV)
    DRHO: Curve      # density correction (g/cc)
    ROP: Curve       # rate of penetration (m/h)

    @field_validator("*")
    @classmethod
    def _finite(cls, value):
        if value is not None and not math.isfinite(value):
            raise ValueError("must be a finite number or null")
        return value


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    well: str | None = Field(default=None, max_length=100, description="Optional label echoed back")
    samples: list[LogSample] = Field(min_length=MIN_SAMPLES, max_length=MAX_SAMPLES)
    include_probabilities: bool = Field(default=False, description="Return all 12 class probabilities")

    @model_validator(mode="after")
    def _unique_depths(self):
        depths = [s.DEPTH_MD for s in self.samples]
        if len(set(depths)) != len(depths):
            raise ValueError("DEPTH_MD values must be unique within a request")
        return self


class SamplePrediction(BaseModel):
    depth_md: float
    lithology: str
    class_index: int
    confidence: float = Field(description="Probability of the predicted class")
    probabilities: dict[str, float] | None = None


class PredictResponse(BaseModel):
    well: str | None
    n_samples: int
    model: str
    predictions: list[SamplePrediction]
    warnings: list[str]


def load_artifact(path: Path) -> dict:
    """Load the model artifact written by train.py and check library versions."""
    if not path.exists():
        raise RuntimeError(f"Model file {path} not found. Run `uv run python train.py` first "
                           "or set MODEL_PATH.")
    artifact = joblib.load(path)
    missing = [k for k in ("model", "features", "class_names", "metadata") if k not in artifact]
    if missing:
        raise RuntimeError(f"{path} is not a valid model artifact (missing {missing})")
    trained = artifact["metadata"].get("versions", {})
    runtime = {"scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__}
    for lib, version in runtime.items():
        if trained.get(lib) and trained[lib] != version:
            log.warning("%s version mismatch: model trained with %s, serving with %s",
                        lib, trained[lib], version)
    log.info("loaded %s (%s, %d features)", path, artifact["metadata"]["model_name"], len(artifact["features"]))
    return artifact


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.artifact = load_artifact(Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)))
    yield


app = FastAPI(
    title="Well-Log Lithology Classifier",
    description="Predicts lithology for each depth sample of a well section (FORCE 2020 data, Norwegian Sea).",
    version="1.0.0",
    lifespan=lifespan,
)


def quality_warnings(df: pd.DataFrame) -> list[str]:
    """Non-fatal input problems that will degrade prediction quality."""
    notes: list[str] = []
    n = len(df)
    if n < FULL_CONTEXT_SAMPLES:
        notes.append(f"Only {n} samples; the long (61-sample) context window is truncated. "
                     f"Send {FULL_CONTEXT_SAMPLES}+ samples for best results.")
    steps = np.diff(np.sort(df[DEPTH_COL].to_numpy()))
    step = float(np.median(steps))
    if abs(step - TRAIN_DEPTH_STEP_M) / TRAIN_DEPTH_STEP_M > 0.25:
        notes.append(f"Median depth step is {step:.3f} m but the model was trained at {TRAIN_DEPTH_STEP_M} m; "
                     "rolling-window features assume that spacing. Resample the logs first.")
    if steps.max() > 10 * max(step, 1e-6):
        notes.append("The section contains large depth gaps; context features across gaps are unreliable.")
    for curve in KEY_CURVES:
        if df[curve].isna().all():
            notes.append(f"Curve {curve} has no values; predictions will rely on the remaining curves.")
    return notes


def predict_section(artifact: dict, request: PredictRequest) -> PredictResponse:
    df = pd.DataFrame([s.model_dump() for s in request.samples])
    df.insert(0, WELL_COL, request.well or "request")
    notes = quality_warnings(df)

    model, features = artifact["model"], artifact["features"]
    frame, X = build_feature_matrix(df, features)      # sorts by depth, builds identical features
    proba = model.predict_proba(X)
    class_ids = np.asarray(model.classes_)
    best = proba.argmax(axis=1)
    names = artifact["class_names"]

    predictions = []
    for i, depth in enumerate(frame[DEPTH_COL].to_numpy()):
        cls = int(class_ids[best[i]])
        item = SamplePrediction(depth_md=float(depth), lithology=names[cls], class_index=cls,
                                confidence=round(float(proba[i, best[i]]), 4))
        if request.include_probabilities:
            item.probabilities = {names[int(c)]: round(float(p), 4) for c, p in zip(class_ids, proba[i])}
        predictions.append(item)

    meta = artifact["metadata"]
    return PredictResponse(well=request.well, n_samples=len(predictions),
                           model=f"{meta['model_name']} ({artifact['feature_set']})",
                           predictions=predictions, warnings=notes)


@app.get("/health")
def health(request: Request) -> dict:
    """Liveness/readiness probe: ok once the model is loaded."""
    return {"status": "ok", "model_loaded": hasattr(request.app.state, "artifact")}


@app.get("/info")
def info(request: Request) -> dict:
    """Model metadata and the input contract."""
    artifact = request.app.state.artifact
    meta = artifact["metadata"]
    return {
        "model": meta["model_name"],
        "feature_set": artifact["feature_set"],
        "trained_at": meta["trained_at"],
        "classes": artifact["class_names"],
        "accepted_curves": ["DEPTH_MD", *MODEL_INPUT_CURVES],
        "limits": {"min_samples": MIN_SAMPLES, "recommended_samples": FULL_CONTEXT_SAMPLES,
                   "max_samples": MAX_SAMPLES, "training_depth_step_m": TRAIN_DEPTH_STEP_M},
        "hidden_test_penalty": meta.get("hidden_test_penalty"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    """Predict lithology for every sample of one well section."""
    try:
        return predict_section(request.app.state.artifact, body)
    except HTTPException:
        raise
    except Exception:
        log.exception("prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed; see server logs.")
