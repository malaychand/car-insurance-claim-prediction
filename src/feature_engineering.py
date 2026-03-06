# -*- coding: utf-8 -*-
"""
feature_engineering.py
=======================
Car Insurance Claim Prediction — Feature Engineering Module

Handles:
  - Dropping non-informative columns
  - Label encoding high-cardinality categoricals
  - One-hot encoding binary / low-cardinality categoricals
  - Dropping highly correlated columns (threshold > 0.95)

Usage
-----
    from feature_engineering import feature_engineering

    # Training mode (fit=True): fits and returns encoders
    X_train, label_encoders, ohe_columns = feature_engineering(
        train_df, fit=True
    )

    # Inference mode (fit=False): reuse fitted encoders
    X_test, _, _ = feature_engineering(
        test_df,
        label_encoders=label_encoders,
        ohe_columns=ohe_columns,
        fit=False,
    )
"""

import warnings

import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

TARGET = "is_claim"

# Dropped before any processing — not predictive
DROP_COLS = ["policy_id", "policy_tenure"]

# High-cardinality string columns → LabelEncoder
LABEL_ENCODE_COLS = [
    "max_torque",
    "max_power",
    "engine_type",
    "area_cluster",
    "model",
    "transmission_type",
    "segment",
]

# Binary / low-cardinality columns → One-Hot Encode
OHE_COLS = [
    "fuel_type",
    "is_esc",
    "is_adjustable_steering",
    "is_tpms",
    "is_parking_sensors",
    "is_parking_camera",
    "rear_brakes_type",
    "steering_type",
    "is_front_fog_lights",
    "is_rear_window_wiper",
    "is_rear_window_washer",
    "is_rear_window_defogger",
    "is_brake_assist",
    "is_power_door_locks",
    "is_central_locking",
    "is_power_steering",
    "is_driver_seat_height_adjustable",
    "is_day_night_rear_view_mirror",
    "is_ecw",
    "is_speed_alert",
]

# Columns identified via upper-triangle correlation matrix (corr > 0.95)
HIGH_CORR_DROP_COLS = [
    "length",
    "is_esc_Yes",
    "is_adjustable_steering_No",
    "is_adjustable_steering_Yes",
    "is_tpms_No",
    "is_tpms_Yes",
    "is_parking_sensors_Yes",
    "is_parking_camera_Yes",
    "rear_brakes_type_Disc",
    "rear_brakes_type_Drum",
    "steering_type_Power",
    "is_front_fog_lights_Yes",
    "is_rear_window_wiper_No",
    "is_rear_window_wiper_Yes",
    "is_rear_window_washer_No",
    "is_rear_window_washer_Yes",
    "is_rear_window_defogger_Yes",
    "is_brake_assist_Yes",
    "is_power_door_locks_Yes",
    "is_central_locking_No",
    "is_central_locking_Yes",
    "is_power_steering_No",
    "is_power_steering_Yes",
    "is_driver_seat_height_adjustable_No",
    "is_driver_seat_height_adjustable_Yes",
    "is_day_night_rear_view_mirror_Yes",
    "is_ecw_No",
    "is_ecw_Yes",
    "is_speed_alert_Yes",
]


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def feature_engineering(
    df: pd.DataFrame,
    label_encoders: dict = None,
    ohe_columns: list = None,
    fit: bool = True,
) -> tuple:
    """
    Transform a raw input DataFrame into model-ready features.

    Processing pipeline:
      1. Drop non-informative columns (policy_id, policy_tenure)
      2. Label-encode high-cardinality categorical columns
      3. One-hot encode binary / low-cardinality categorical columns
      4. Drop highly correlated columns (upper-triangle corr > 0.95)

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data. May or may not contain the target column (is_claim).
    label_encoders : dict, optional
        Pre-fitted {column_name: LabelEncoder} mapping from training.
        Must be provided when fit=False (inference mode).
    ohe_columns : list, optional
        Exact OHE column names produced during training.
        Must be provided when fit=False to align inference schema.
    fit : bool
        True  — fit encoders from df  (training mode).
        False — reuse provided encoders (inference / test mode).

    Returns
    -------
    df_out : pd.DataFrame
        Transformed feature matrix. Target column excluded.
    label_encoders : dict
        Fitted LabelEncoders keyed by column name.
    ohe_columns : list
        OHE column names aligned to training schema.

    Examples
    --------
    >>> train_fe, le, ohe_cols = feature_engineering(train_df, fit=True)
    >>> test_fe, _, _          = feature_engineering(
    ...     test_df, label_encoders=le, ohe_columns=ohe_cols, fit=False
    ... )
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # Step 1: Drop irrelevant columns
    # ------------------------------------------------------------------
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)

    # Separate target so it is never accidentally encoded
    target_series = None
    if TARGET in df.columns:
        target_series = df.pop(TARGET)

    # ------------------------------------------------------------------
    # Step 2: Label Encoding
    # ------------------------------------------------------------------
    if fit:
        label_encoders = {}
        for col in LABEL_ENCODE_COLS:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
    else:
        if label_encoders is None:
            raise ValueError(
                "label_encoders must be provided when fit=False. "
                "Pass the encoders returned from a previous fit=True call."
            )
        for col, le in label_encoders.items():
            if col in df.columns:
                known = set(le.classes_)
                # Gracefully handle unseen labels at inference time
                df[col] = df[col].astype(str).apply(
                    lambda x: x if x in known else le.classes_[0]
                )
                df[col] = le.transform(df[col])

    # ------------------------------------------------------------------
    # Step 3: One-Hot Encoding
    # ------------------------------------------------------------------
    present_ohe = [c for c in OHE_COLS if c in df.columns]
    dummies = pd.get_dummies(df[present_ohe])
    df.drop(columns=present_ohe, inplace=True)
    df = pd.concat([df, dummies], axis=1)

    if fit:
        ohe_columns = list(dummies.columns)
    else:
        if ohe_columns is None:
            raise ValueError(
                "ohe_columns must be provided when fit=False. "
                "Pass the list returned from a previous fit=True call."
            )
        # Add any missing columns (new category not seen in this batch)
        for col in ohe_columns:
            if col not in df.columns:
                df[col] = 0
        # Remove unexpected extra columns
        extra = [c for c in dummies.columns if c not in ohe_columns]
        df.drop(columns=extra, errors="ignore", inplace=True)

    # ------------------------------------------------------------------
    # Step 4: Drop highly correlated columns
    # ------------------------------------------------------------------
    corr_drop = [c for c in HIGH_CORR_DROP_COLS if c in df.columns]
    df.drop(columns=corr_drop, inplace=True)

    # Re-attach target if it was present in the input
    if target_series is not None:
        df[TARGET] = target_series.values

    return df, label_encoders, ohe_columns