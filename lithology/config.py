"""Constants: column names, label encoding and curve groups for the FORCE 2020 dataset."""
from __future__ import annotations

WELL_COL = "WELL"
DEPTH_COL = "DEPTH_MD"
TARGET_COL = "FORCE_2020_LITHOFACIES_LITHOLOGY"
CONFIDENCE_COL = "FORCE_2020_LITHOFACIES_CONFIDENCE"

# Original lithology code -> contiguous class index (the order used by the official
# penalty matrix, so matrix[true_class, predicted_class] is valid).
CODE_TO_CLASS: dict[int, int] = {
    30000: 0, 65030: 1, 65000: 2, 80000: 3, 74000: 4, 70000: 5,
    70032: 6, 88000: 7, 86000: 8, 99000: 9, 90000: 10, 93000: 11,
}
CLASS_TO_CODE: dict[int, int] = {v: k for k, v in CODE_TO_CLASS.items()}

CODE_TO_NAME: dict[int, str] = {
    30000: "Sandstone", 65030: "Sandstone/Shale", 65000: "Shale", 80000: "Marl",
    74000: "Dolomite", 70000: "Limestone", 70032: "Chalk", 88000: "Halite",
    86000: "Anhydrite", 99000: "Tuff", 90000: "Coal", 93000: "Basement",
}
CLASS_NAMES: list[str] = [CODE_TO_NAME[CLASS_TO_CODE[i]] for i in range(len(CODE_TO_CLASS))]
N_CLASSES = len(CLASS_NAMES)

# Every measured curve in the raw files (used for sentinel cleaning).
ALL_LOGS = [
    "CALI", "RSHA", "RMED", "RDEP", "RHOB", "GR", "SGR", "NPHI", "PEF", "DTC", "SP",
    "BS", "ROP", "DTS", "DCAL", "DRHO", "MUDWEIGHT", "RMIC", "ROPA", "RXO",
]
# Curves that are missing in >70% of rows: excluded from modelling (see notebook EDA).
SPARSE_LOGS = ["SGR", "DTS", "RMIC", "ROPA", "DCAL", "MUDWEIGHT", "RXO"]

# Raw curves kept as model inputs, and curves that get a log10 transform.
RAW_FEATURE_LOGS = ["CALI", "RHOB", "GR", "NPHI", "PEF", "DTC", "SP", "BS", "DRHO"]
LOG10_LOGS = ["RSHA", "RMED", "RDEP", "ROP"]

SENTINEL_MAX = -990.0          # LAS null values such as -999.25 are <= this
ROLL_LOGS = ["GR", "RHOB", "NPHI", "DTC"]
WINDOWS = (15, 61)             # samples; ~2.3 m and ~9.3 m at the 0.152 m depth step

# Display colours for plots (one per class index).
CLASS_COLORS = [
    "#f2c14e", "#b8a46a", "#7a8b7a", "#d98b5f", "#8e6fb3", "#6fb3d9",
    "#bfe3f2", "#f2a6c8", "#c9a0dc", "#e06666", "#222222", "#8a8a8a",
]

# Curves the prediction API accepts (everything the model's features are derived from).
MODEL_INPUT_CURVES: list[str] = sorted(set(RAW_FEATURE_LOGS + LOG10_LOGS + ROLL_LOGS))
# Depth step of the training data in metres; rolling-window features assume this spacing.
TRAIN_DEPTH_STEP_M = 0.152
