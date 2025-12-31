# data_preprocessing.py
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils import BOOL_COLS, OTHER_CATS, ensure_dir


def load_raw_data(
    train_path: str,
    test_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw train and test CSVs."""
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test


def clean_boolean_columns(df_train: pd.DataFrame,
                          df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Map Yes/No to 1/0 for boolean columns."""
    for col in BOOL_COLS:
        if col in df_train.columns:
            df_train[col] = df_train[col].map({"Yes": 1, "No": 0})
        if col in df_test.columns:
            df_test[col] = df_test[col].map({"Yes": 1, "No": 0})
    return df_train, df_test


def extract_power_torque_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract numeric max_power and max_torque features and drop original text columns.
    Applies to both train and test.
    """
    for df in (df_train, df_test):
        if "max_power" in df.columns:
            df["max_power_value"] = (
                df["max_power"].replace(r"([0-9.]+).*", r"\1", regex=True).astype(float)
            )
            df["max_power_rpm"] = (
                df["max_power"].replace(r".*@([0-9]+).*", r"\1", regex=True).astype(float)
            )
        if "max_torque" in df.columns:
            df["max_torque_value"] = (
                df["max_torque"].replace(r"([0-9.]+).*", r"\1", regex=True).astype(float)
            )
            df["max_torque_rpm"] = (
                df["max_torque"].replace(r".*@([0-9]+).*", r"\1", regex=True).astype(float)
            )

        # Drop original text columns if they exist
        for col in ["max_power", "max_torque"]:
            if col in df.columns:
                df.drop(col, axis=1, inplace=True)

    return df_train, df_test


def encode_categoricals(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    encoders_path: str = "models/all_encoders.pkl",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Label-encode categorical columns using train set fit and apply to both.
    Saves encoders to disk for later use in Streamlit app.
    """
    encoders: Dict[str, LabelEncoder] = {}

    for col in OTHER_CATS:
        if col not in df_train.columns:
            continue

        le = LabelEncoder()
        df_train[col] = le.fit_transform(df_train[col])
        df_test[col] = le.transform(df_test[col])
        encoders[col] = le

    ensure_dir("models")
    joblib.dump(encoders, encoders_path)
    return df_train, df_test, encoders


def preprocess_data(
    train_path: str,
    test_path: str,
    encoders_path: str = "models/all_encoders.pkl",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Full preprocessing pipeline used for model training:
    - load raw data
    - drop train policy_id (kept in test for submission)
    - clean boolean columns
    - extract power/torque numeric features
    - encode categoricals and save encoders
    """
    df_train, df_test = load_raw_data(train_path, test_path)

    # Drop policy_id only from train; keep in test for submission
    if "policy_id" in df_train.columns:
        df_train = df_train.drop("policy_id", axis=1)

    df_train, df_test = clean_boolean_columns(df_train, df_test)
    df_train, df_test = extract_power_torque_features(df_train, df_test)
    df_train, df_test, encoders = encode_categoricals(df_train, df_test, encoders_path)

    return df_train, df_test, encoders
