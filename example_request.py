"""Send one real well section from the hidden-test set to the running API.

Usage (API must be running):
    uv run python example_request.py
    uv run python example_request.py --url http://localhost:8000 --rows 400
    uv run python example_request.py --well 16/2-7 --start 9000 --rows 600   # a section with 6 lithologies
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import requests

from lithology.config import DEPTH_COL, MODEL_INPUT_CURVES, TARGET_COL, WELL_COL, CODE_TO_NAME


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--csv", type=Path, default=Path("data/raw/hidden_test.csv"))
    p.add_argument("--well", default=None, help="well name (default: first well in the file)")
    p.add_argument("--start", type=int, default=0, help="first sample (row offset within the well, sorted by depth)")
    p.add_argument("--rows", type=int, default=300, help="number of consecutive samples to send")
    args = p.parse_args()

    if not args.csv.exists():
        print(f"{args.csv} not found. Run: uv run python download_data.py", file=sys.stderr)
        return 1
    df = pd.read_csv(args.csv, sep=";")
    well = args.well or df[WELL_COL].iloc[0]
    section = df[df[WELL_COL] == well].sort_values(DEPTH_COL).iloc[args.start:args.start + args.rows]

    if section.empty:
        print(f"No samples for well {well} at start={args.start}", file=sys.stderr)
        return 1
    columns = [DEPTH_COL, *MODEL_INPUT_CURVES]
    # to_json writes NaN as null, which is what the API expects for "not recorded"
    samples = json.loads(section[columns].to_json(orient="records"))
    payload = {"well": str(well), "samples": samples}

    try:
        response = requests.post(f"{args.url}/predict", json=payload, timeout=60)
    except requests.ConnectionError:
        print(f"Could not reach {args.url}. Is the API running?", file=sys.stderr)
        return 1
    if response.status_code != 200:
        print(response.status_code, response.text[:800], file=sys.stderr)
        return 1

    body = response.json()
    predicted = pd.Series([p["lithology"] for p in body["predictions"]])
    truth = section[TARGET_COL].map(CODE_TO_NAME).reset_index(drop=True)
    print(f"well {well}: {body['n_samples']} samples | model: {body['model']}")
    print("warnings:", body["warnings"] or "none")
    print("\npredicted lithology counts:")
    print(predicted.value_counts().to_string())
    print(f"\nagreement with interpreter labels on this section: {(predicted == truth).mean():.1%}")
    print("\nfirst 5 predictions:")
    for item in body["predictions"][:5]:
        print(f"  {item['depth_md']:9.2f} m  {item['lithology']:16s} confidence {item['confidence']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
