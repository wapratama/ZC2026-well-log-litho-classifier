"""Download the FORCE 2020 lithology competition data (idempotent).

Run from the project root:  uv run python download_data.py
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_URL = "https://github.com/bolgebrygg/Force-2020-Machine-Learning-competition.git"
REPO_DATA_DIR = "lithology_competition/data"
FILES = [
    "train.zip",
    "hidden_test.csv",
    "leaderboard_test_features.csv",
    "leaderboard_test_target.csv",
    "penalty_matrix.npy",
]
RAW_DIR = Path("data/raw")
TMP_DIR = Path("data/_clone_tmp")


def _force_remove(func, path, _exc_info) -> None:
    """Windows: git pack files are read-only, so clear the flag and retry."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def list_files() -> None:
    for p in sorted(RAW_DIR.rglob("*")):
        if p.is_file():
            print(f"{p.stat().st_size / 1e6:10.1f} MB  {p}")


def main() -> int:
    if shutil.which("git") is None:
        print("git is required: https://git-scm.com/downloads", file=sys.stderr)
        return 1

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if (RAW_DIR / "train.csv").exists() and (RAW_DIR / "penalty_matrix.npy").exists():
        print("Data already present; nothing to do.")
        list_files()
        return 0

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR, onerror=_force_remove)
    run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
         REPO_URL, str(TMP_DIR)])
    patterns = [f"/{REPO_DATA_DIR}/{name}" for name in FILES]
    run(["git", "sparse-checkout", "set", "--no-cone", *patterns], cwd=TMP_DIR)

    for name in FILES:
        shutil.move(str(TMP_DIR / REPO_DATA_DIR / name), RAW_DIR / name)
    shutil.rmtree(TMP_DIR, onerror=_force_remove)

    print("Extracting train.zip")
    with zipfile.ZipFile(RAW_DIR / "train.zip") as zf:
        zf.extractall(RAW_DIR)
    (RAW_DIR / "train.zip").unlink()

    list_files()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
