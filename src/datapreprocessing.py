# -*- coding: utf-8 -*-
"""
datapreprocessing.py
====================
Car Insurance Claim Prediction — Data Preprocessing Module

Handles:
  - Applying feature engineering (via feature_engineering module)
  - SMOTE oversampling for class imbalance
  - StandardScaler normalization
  - Train / validation split

Usage
-----
    from datapreprocessing import preprocess_train, preprocess_inference

    # Training
    result = preprocess_train(train_df)
    X_train = result["X_train"]
    X_test  = result["X_test"]
    y_train = result["y_train"]
    y_test  = result["y_test"]
    scaler  = result["scaler"]
    label_encoders = result["label_encoders"]
    ohe_columns    = result["ohe_columns"]
    feature_columns = result["feature_columns"]

    # Inference
    X_infer = preprocess_inference(
        test_df,
        label_encoders=label_encoders,
        ohe_columns=ohe_columns,
        feature_columns=feature_columns,
        scaler=scaler,
    )
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from feature_engineering import TARGET, feature_engineering

warnings.filterwarnings("ignore")

# Optional SMOTE
try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Training Preprocessing
# ---------------------------------------------------------------------------

def preprocess_train(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
    apply_smote: bool = True,
    scale_features: bool = True,
) -> dict:
    """
    Full preprocessing pipeline for training data.

    Steps:
      1. Feature engineering (label encode, OHE, correlation drop)
      2. Train / test split (stratified)
      3. SMOTE oversampling on the training split (optional)
      4. StandardScaler normalization (optional)

    Parameters
    ----------
    df : pd.DataFrame
        Raw training DataFrame. Must include the 'is_claim' target column.
    test_size : float
        Fraction of data held out for evaluation. Default 0.25.
    random_state : int
        Random seed for reproducibility. Default 42.
    apply_smote : bool
        Whether to apply SMOTE to the training split. Default True.
    scale_features : bool
        Whether to apply StandardScaler. Default True.

    Returns
    -------
    dict with keys:
        X_train, X_test : pd.DataFrame  — feature matrices
        y_train, y_test : pd.Series     — target arrays
        scaler          : StandardScaler or None
        label_encoders  : dict
        ohe_columns     : list
        feature_columns : list
    """
    if TARGET not in df.columns:
        raise ValueError(f"Training DataFrame must include target column '{TARGET}'.")

    # Step 1: Feature Engineering (fit mode)
    df_eng, label_encoders, ohe_columns = feature_engineering(df, fit=True)

    X = df_eng.drop(columns=[TARGET])
    y = df_eng[TARGET]
    feature_columns = list(X.columns)

    # Step 2: Train / test split (stratified to preserve class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Step 3: SMOTE oversampling
    if apply_smote:
        if _SMOTE_AVAILABLE:
            sm = SMOTE(random_state=random_state)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            print(
                f"[SMOTE] Class distribution after resampling: "
                f"{pd.Series(y_train).value_counts().to_dict()}"
            )
        else:
            print(
                "[WARN] imbalanced-learn not installed. "
                "Install with: pip install imbalanced-learn"
            )

    # Step 4: Scaling
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train), columns=feature_columns
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test), columns=feature_columns
        )

    print(
        f"[Preprocessing] X_train: {X_train.shape} | "
        f"X_test: {X_test.shape} | "
        f"Features: {len(feature_columns)}"
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "ohe_columns": ohe_columns,
        "feature_columns": feature_columns,
    }


# ---------------------------------------------------------------------------
# Inference Preprocessing
# ---------------------------------------------------------------------------

def preprocess_inference(
    df: pd.DataFrame,
    label_encoders: dict,
    ohe_columns: list,
    feature_columns: list,
    scaler: StandardScaler = None,
) -> pd.DataFrame:
    """
    Preprocess raw test / inference data using fitted training artefacts.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data. Target column is ignored if present.
    label_encoders : dict
        Fitted LabelEncoders from preprocess_train().
    ohe_columns : list
        OHE column schema from preprocess_train().
    feature_columns : list
        Exact feature column list from preprocess_train().
    scaler : StandardScaler, optional
        Fitted scaler from preprocess_train(). Pass None to skip scaling.

    Returns
    -------
    X : pd.DataFrame
        Preprocessed feature matrix ready for model.predict().
    """
    # Feature engineering (inference mode — reuse fitted encoders)
    df_eng, _, _ = feature_engineering(
        df,
        label_encoders=label_encoders,
        ohe_columns=ohe_columns,
        fit=False,
    )

    # Drop target if present
    if TARGET in df_eng.columns:
        df_eng.drop(columns=[TARGET], inplace=True)

    # Align to training schema
    for col in feature_columns:
        if col not in df_eng.columns:
            df_eng[col] = 0
    X = df_eng[feature_columns]

    # Apply scaling
    if scaler is not None:
        X = pd.DataFrame(scaler.transform(X), columns=feature_columns)

    return X