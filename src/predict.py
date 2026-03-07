# src/predict.py
# ---------------------------------------------------------------
# Single-row inference wrapper used by app.py
# ---------------------------------------------------------------

import os
import joblib
import pandas as pd
import numpy as np

from .preprocessing import preprocess_inference

DEFAULT_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)


def predict_claim(raw_df: pd.DataFrame, models_dir: str = DEFAULT_MODELS_DIR):
    """
    Parameters
    ----------
    raw_df     : pd.DataFrame — raw input row(s) from the Streamlit UI
    models_dir : str          — path to folder with saved .pkl files

    Returns
    -------
    prob  : float  — probability of claim (class=1)
    label : int    — predicted label (0 or 1)
    """
    model_path = os.path.join(models_dir, "lightgbm_optuna_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}.\n"
            "Run  python -m src.training  first to train and save the model."
        )

    model   = joblib.load(model_path)
    X_ready = preprocess_inference(raw_df, models_dir=models_dir)

    prob  = model.predict_proba(X_ready)[0][1]
    label = int(prob >= 0.5)
    return prob, label
