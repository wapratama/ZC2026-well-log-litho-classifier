import numpy as np
import pandas as pd

from lithology.config import ALL_LOGS, DEPTH_COL, WELL_COL
from lithology.features import FEATURES_LOGS, add_features, clean_logs, prepare


def _synthetic(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for well in ("A", "B"):
        df = pd.DataFrame({c: rng.uniform(1, 100, n) for c in ALL_LOGS})
        df["RHOB"] = rng.uniform(1.8, 2.8, n)
        df["NPHI"] = rng.uniform(0.05, 0.5, n)
        df[WELL_COL] = well
        df[DEPTH_COL] = 1000 + 0.152 * np.arange(n)
        df["Z_LOC"] = -1000.0 - np.arange(n)      # legitimately below -990
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def test_sentinels_become_nan_but_coordinates_survive():
    df = _synthetic()
    df.loc[3, "SP"] = -999.25
    out = clean_logs(df)
    assert out["SP"].isna().sum() == 1
    assert (out["Z_LOC"] < -990).sum() > 0


def test_all_expected_features_are_created():
    out = prepare(_synthetic())
    missing = [c for c in FEATURES_LOGS if c not in out.columns]
    assert not missing


def test_features_do_not_leak_across_wells():
    df = _synthetic()
    base = add_features(clean_logs(df))
    changed = df.copy()
    changed.loc[changed[WELL_COL] == "B", "GR"] += 500.0     # perturb well B only
    after = add_features(clean_logs(changed))
    a_rows = base[WELL_COL] == "A"
    pd.testing.assert_frame_equal(base.loc[a_rows, FEATURES_LOGS], after.loc[a_rows, FEATURES_LOGS])


def test_all_null_and_absent_curves_are_handled():
    df = _synthetic()
    df["RMED"] = None                      # arrives as object dtype, e.g. from JSON
    df = df.drop(columns=["PEF"])          # curve not supplied at all
    out = prepare(df)
    assert out["RMED"].isna().all() and out["PEF"].isna().all()
    assert out["LOG_RMED"].isna().all()
    assert not out[["GR_MEAN15", "PHID"]].isna().all().any()
