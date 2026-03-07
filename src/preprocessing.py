# src/preprocessing.py
# ---------------------------------------------------------------
# Inference preprocessing — mirrors training.py exactly.
#
# Column dtype facts (from df.info() on actual CSV):
#   make         → int64  (already numeric, NO label encoding)
#   max_torque   → object → label encode
#   max_power    → object → label encode
#   engine_type  → object → label encode
#   area_cluster → object → label encode
#   model        → object → label encode
#   transmission_type → object → label encode
#   segment      → object → label encode
#   fuel_type + 19 is_*/rear_brakes/steering → object → OHE
# ---------------------------------------------------------------

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

# Columns requiring LabelEncoding (object dtype in CSV)
LABEL_ENCODE_COLS = [
    "max_torque", "max_power", "engine_type",
    "area_cluster", "model", "transmission_type", "segment"
]

# Columns for One-Hot Encoding
OHE_COLS = [
    "fuel_type", "is_esc", "is_adjustable_steering", "is_tpms",
    "is_parking_sensors", "is_parking_camera", "rear_brakes_type",
    "steering_type", "is_front_fog_lights", "is_rear_window_wiper",
    "is_rear_window_washer", "is_rear_window_defogger", "is_brake_assist",
    "is_power_door_locks", "is_central_locking", "is_power_steering",
    "is_driver_seat_height_adjustable", "is_day_night_rear_view_mirror",
    "is_ecw", "is_speed_alert"
]


def _load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _label_encode(value: str, classes: list) -> int:
    """Version-safe label encoding using a plain list."""
    s = str(value)
    return classes.index(s) if s in classes else 0


def preprocess_inference(raw_df: pd.DataFrame, models_dir: str = "models") -> np.ndarray:
    """
    Convert a raw UI input row into the scaled feature vector
    the LightGBM model expects.

    IMPORTANT: `make` stays as-is (it is already int in the CSV).
    Only the 7 LABEL_ENCODE_COLS (all object dtype) get encoded.
    """
    df = raw_df.copy()

    # 1. Drop columns unused in training ──────────────────────────
    for col in ["policy_id", "policy_tenure"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # 2. Label encode object columns (NOT make) ───────────────────
    classes_path = os.path.join(models_dir, "encoder_classes.json")
    if not os.path.exists(classes_path):
        raise FileNotFoundError(
            "encoder_classes.json not found.\n"
            "Run:  python -m src.training"
        )

    encoder_classes: dict = _load_json(classes_path)

    for col in LABEL_ENCODE_COLS:
        if col in df.columns and col in encoder_classes:
            classes = encoder_classes[col]
            df[col] = df[col].astype(str).apply(
                lambda v, c=classes: _label_encode(v, c)
            )
        # If col not in encoder_classes it was already numeric — leave it

    # 3. One-hot encode ───────────────────────────────────────────
    ohe_cols_path = os.path.join(models_dir, "ohe_columns.json")
    if not os.path.exists(ohe_cols_path):
        raise FileNotFoundError(
            "ohe_columns.json not found.\n"
            "Run:  python -m src.training"
        )

    ohe_columns: list = _load_json(ohe_cols_path)
    present_ohe = [c for c in OHE_COLS if c in df.columns]
    dms = pd.get_dummies(df[present_ohe])
    dms = dms.reindex(columns=ohe_columns, fill_value=0)

    df.drop(columns=present_ohe, inplace=True, errors="ignore")
    df = pd.concat([df, dms], axis=1)

    # 4. Drop high-correlation columns ────────────────────────────
    corr_drop_path = os.path.join(models_dir, "corr_drop_cols.json")
    if os.path.exists(corr_drop_path):
        drop_cols = _load_json(corr_drop_path)
        drop_present = [c for c in drop_cols if c in df.columns]
        df.drop(columns=drop_present, inplace=True)

    # 5. Drop target if present ───────────────────────────────────
    if "is_claim" in df.columns:
        df.drop(columns=["is_claim"], inplace=True)

    # 6. Align to exact training feature order ────────────────────
    feat_path = os.path.join(models_dir, "feature_columns.json")
    if not os.path.exists(feat_path):
        # fallback to pkl
        feat_path = os.path.join(models_dir, "feature_columns.pkl")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(
                "feature_columns.json not found.\n"
                "Run:  python -m src.training"
            )
        feature_columns = joblib.load(feat_path)
    else:
        feature_columns = _load_json(feat_path)

    df = df.reindex(columns=feature_columns, fill_value=0)

    # 7. Scale ────────────────────────────────────────────────────
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            "scaler.pkl not found.\n"
            "Run:  python -m src.training"
        )

    scaler = joblib.load(scaler_path)
    return scaler.transform(df.values.astype(float))
