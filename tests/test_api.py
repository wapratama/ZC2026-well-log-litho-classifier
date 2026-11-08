"""API tests. They use the trained model in models/model.joblib and are skipped without it."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from lithology.config import DEPTH_COL, MODEL_INPUT_CURVES

MODEL_PATH = Path("models/model.joblib")
HIDDEN_CSV = Path("data/raw/hidden_test.csv")

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="train the model first (train.py)")


@pytest.fixture(scope="module")
def client():
    from predict import app
    with TestClient(app) as c:
        yield c


def _section(n=80, step=0.152, seed=0) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [{DEPTH_COL: 1500 + i * step, "GR": float(rng.uniform(20, 120)), "RHOB": float(rng.uniform(2.0, 2.7)),
             "NPHI": float(rng.uniform(0.1, 0.4)), "DTC": float(rng.uniform(60, 120)),
             "RDEP": float(rng.uniform(0.5, 5)), "RMED": None, "SP": -999.25} for i in range(n)]


def test_health_and_info(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    info = client.get("/info").json()
    assert len(info["classes"]) == 12
    assert set(MODEL_INPUT_CURVES) <= set(info["accepted_curves"])


def test_valid_request_returns_one_prediction_per_sample(client):
    body = client.post("/predict", json={"well": "demo", "samples": _section()}).json()
    assert body["n_samples"] == 80 and len(body["predictions"]) == 80
    depths = [p["depth_md"] for p in body["predictions"]]
    assert depths == sorted(depths)
    assert all(0.0 <= p["confidence"] <= 1.0 for p in body["predictions"])


def test_samples_may_arrive_unsorted(client):
    shuffled = _section()[::-1]
    body = client.post("/predict", json={"samples": shuffled}).json()
    depths = [p["depth_md"] for p in body["predictions"]]
    assert depths == sorted(depths)


def test_probabilities_sum_to_one_when_requested(client):
    body = client.post("/predict", json={"samples": _section(), "include_probabilities": True}).json()
    assert sum(body["predictions"][0]["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("mutate, why", [
    (lambda s: s[:5], "too few samples"),
    (lambda s: s + [dict(s[0])], "duplicate depth"),
    (lambda s: [{**x, "GRR": 1.0} for x in s], "unknown curve name (typo)"),
    (lambda s: [{**x, "GR": "high"} for x in s], "non-numeric value"),
    (lambda s: [{k: v for k, v in x.items() if k != DEPTH_COL} for x in s], "missing DEPTH_MD"),
])
def test_invalid_requests_are_rejected_with_422(client, mutate, why):
    assert client.post("/predict", json={"samples": mutate(_section())}).status_code == 422, why


def test_warns_on_wrong_depth_spacing_and_short_section(client):
    body = client.post("/predict", json={"samples": _section(n=30, step=1.0)}).json()
    text = " ".join(body["warnings"])
    assert "depth step" in text and "30 samples" in text


@pytest.mark.skipif(not HIDDEN_CSV.exists(), reason="data not downloaded")
def test_api_matches_batch_prediction_for_a_full_well(client):
    """Train/serve parity: a whole well through the API equals direct batch prediction."""
    import joblib
    from lithology.config import WELL_COL
    from lithology.modeling import build_feature_matrix

    raw = pd.read_csv(HIDDEN_CSV, sep=";")
    well = raw[WELL_COL].iloc[0]
    section = raw[raw[WELL_COL] == well]
    artifact = joblib.load(MODEL_PATH)
    _, X = build_feature_matrix(section, artifact["features"])
    expected = artifact["model"].predict(X)

    columns = [DEPTH_COL, *MODEL_INPUT_CURVES]
    import json
    samples = json.loads(section[columns].to_json(orient="records"))
    body = client.post("/predict", json={"well": str(well), "samples": samples}).json()
    got = np.array([p["class_index"] for p in body["predictions"]])
    assert (got == expected).all()
