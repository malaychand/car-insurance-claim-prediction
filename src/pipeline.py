# -*- coding: utf-8 -*-
"""
pipeline.py
===========
Car Insurance Claim Prediction — End-to-End ML Pipeline

Orchestrates the full workflow by importing from the separate modules:
  - feature_engineering.py  → feature_engineering()
  - datapreprocessing.py    → preprocess_train(), preprocess_inference()
  - model_training.py       → train_final_model(), evaluate_model(), tune_lgbm()

Usage (Training)
----------------
    from pipeline import CarInsurancePipeline
    import pandas as pd

    train_df = pd.read_csv("data/train.csv")
    pipe = CarInsurancePipeline()
    pipe.fit(train_df)
    pipe.save("models/car_insurance_pipeline.pkl")

Usage (Inference)
-----------------
    pipe = CarInsurancePipeline.load("models/car_insurance_pipeline.pkl")
    test_df = pd.read_csv("data/test.csv")
    preds = pipe.predict(test_df)           # 0 or 1
    proba = pipe.predict_proba(test_df)     # [P(no claim), P(claim)]

PEP8 compliant | fixed random seeds | fully documented
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Local module imports — all in src/
from datapreprocessing import preprocess_inference, preprocess_train
from feature_engineering import TARGET
from model_training import (
    evaluate_model,
    save_model,
    train_final_model,
    tune_lgbm,
    tune_logistic,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Pipeline Class
# ---------------------------------------------------------------------------

class CarInsurancePipeline:
    """
    End-to-end pipeline for Car Insurance Claim Prediction.

    Orchestrates:
      fit()           → preprocess_train → tune (optional) → train_final_model
      predict()       → preprocess_inference → model.predict
      predict_proba() → preprocess_inference → model.predict_proba
      evaluate()      → preprocess_inference → evaluate_model
      save() / load() → pickle serialization

    Parameters
    ----------
    test_size : float
        Hold-out fraction for evaluation. Default 0.25.
    random_state : int
        Global random seed. Default 42.
    apply_smote : bool
        Apply SMOTE oversampling during preprocessing. Default True.
    scale_features : bool
        Apply StandardScaler. Default True.
    tune : bool
        Run Optuna hyperparameter tuning before final model training.
        Default False (uses pre-tuned defaults for speed).
    n_trials : int
        Number of Optuna trials if tune=True. Default 50.
    lgbm_params : dict, optional
        Override hyperparameters directly (skips tuning).
    """

    def __init__(
        self,
        test_size: float = 0.25,
        random_state: int = 42,
        apply_smote: bool = True,
        scale_features: bool = True,
        tune: bool = False,
        n_trials: int = 50,
        lgbm_params: dict = None,
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.apply_smote = apply_smote
        self.scale_features = scale_features
        self.tune = tune
        self.n_trials = n_trials
        self.lgbm_params = lgbm_params

        # Fitted artefacts — populated during fit()
        self._label_encoders = None
        self._ohe_columns = None
        self._feature_columns = None
        self._scaler = None
        self._model = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, verbose: bool = True) -> "CarInsurancePipeline":
        """
        Fit the full pipeline on raw training data.

        Steps:
          1. Preprocess (feature engineering + SMOTE + scaling)
          2. Optionally run Optuna tuning
          3. Train final LightGBM / GradientBoosting model
          4. Evaluate on hold-out test split

        Parameters
        ----------
        df      : pd.DataFrame — raw training data with 'is_claim' column
        verbose : bool         — print evaluation metrics after training

        Returns
        -------
        self
        """
        if TARGET not in df.columns:
            raise ValueError(
                f"Training DataFrame must include target column '{TARGET}'."
            )

        print("[Pipeline] Starting preprocessing...")
        prep = preprocess_train(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            apply_smote=self.apply_smote,
            scale_features=self.scale_features,
        )

        X_train = prep["X_train"]
        X_test  = prep["X_test"]
        y_train = prep["y_train"]
        y_test  = prep["y_test"]

        # Store preprocessing artefacts for inference
        self._label_encoders  = prep["label_encoders"]
        self._ohe_columns     = prep["ohe_columns"]
        self._feature_columns = prep["feature_columns"]
        self._scaler          = prep["scaler"]

        # Hyperparameter tuning (optional)
        params = self.lgbm_params or {}
        if self.tune:
            print("[Pipeline] Running Optuna hyperparameter tuning...")
            params = tune_lgbm(
                X_train, y_train, X_test, y_test,
                n_trials=self.n_trials,
                random_state=self.random_state,
            )

        # Train final model
        print("[Pipeline] Training final model...")
        self._model = train_final_model(X_train, y_train, params=params)
        self._is_fitted = True

        # Evaluate on hold-out
        if verbose:
            print("\n[Pipeline] Evaluating on hold-out test set...")
            evaluate_model(self._model, X_test, y_test, plot=False)

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict claim labels (0 = no claim, 1 = claim).

        Parameters
        ----------
        df : pd.DataFrame — raw data (target column ignored if present)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        self._check_fitted()
        X = self._preprocess_for_inference(df)
        return self._model.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return class probabilities [P(no claim), P(claim)] per row.

        Parameters
        ----------
        df : pd.DataFrame — raw data

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
        """
        self._check_fitted()
        X = self._preprocess_for_inference(df)
        return self._model.predict_proba(X)

    def evaluate(self, df: pd.DataFrame, plot: bool = True) -> dict:
        """
        Evaluate the pipeline on a labelled raw DataFrame.

        Parameters
        ----------
        df   : pd.DataFrame — must include 'is_claim' target column
        plot : bool         — generate confusion matrix & ROC curve plots

        Returns
        -------
        dict: accuracy, f1, roc_auc, report, confusion_matrix
        """
        self._check_fitted()
        if TARGET not in df.columns:
            raise ValueError(
                f"Evaluation DataFrame must include target column '{TARGET}'."
            )
        X = self._preprocess_for_inference(df)
        y = df[TARGET]
        return evaluate_model(self._model, X, y, plot=plot)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = "models/car_insurance_pipeline.pkl") -> None:
        """Pickle the entire fitted pipeline to disk."""
        self._check_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        print(f"[Pipeline] Saved  -> '{path}'")

    @classmethod
    def load(cls, path: str = "models/car_insurance_pipeline.pkl") -> "CarInsurancePipeline":
        """Load a saved pipeline from disk."""
        with open(path, "rb") as fh:
            pipe = pickle.load(fh)
        print(f"[Pipeline] Loaded <- '{path}'")
        return pipe

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess_for_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run inference preprocessing using fitted artefacts."""
        return preprocess_inference(
            df,
            label_encoders=self._label_encoders,
            ohe_columns=self._ohe_columns,
            feature_columns=self._feature_columns,
            scaler=self._scaler,
        )

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Pipeline not fitted. Call fit(train_df) first."
            )

    # ------------------------------------------------------------------
    # Properties (read-only access to internals)
    # ------------------------------------------------------------------

    @property
    def model(self):
        """The underlying trained model."""
        self._check_fitted()
        return self._model

    @property
    def feature_columns(self) -> list:
        """List of feature column names used by the model."""
        self._check_fitted()
        return self._feature_columns

    @property
    def label_encoders(self) -> dict:
        """Fitted LabelEncoders from training."""
        self._check_fitted()
        return self._label_encoders

    @property
    def ohe_columns(self) -> list:
        """One-hot encoded column names from training."""
        self._check_fitted()
        return self._ohe_columns

    @property
    def scaler(self):
        """Fitted StandardScaler from training."""
        self._check_fitted()
        return self._scaler


# ---------------------------------------------------------------------------
# Entry point — run from src/ directory
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys

    # Resolve data paths relative to project root
    BASE = Path(__file__).resolve().parent.parent
    TRAIN_PATH = BASE / "data" / "train.csv"
    TEST_PATH  = BASE / "data" / "test.csv"
    MODEL_PATH = str(BASE / "models" / "car_insurance_pipeline.pkl")

    print(f"[Pipeline] Loading data from {TRAIN_PATH}")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)
    print(f"[Pipeline] Train: {train_df.shape}  |  Test: {test_df.shape}")

    # Fit pipeline
    pipe = CarInsurancePipeline(
        apply_smote=True,
        scale_features=True,
        tune=False,           # Set True to run Optuna tuning (~50 trials)
        random_state=42,
    )
    pipe.fit(train_df, verbose=True)
    pipe.save(MODEL_PATH)

    # Inference on test set
    preds  = pipe.predict(test_df)
    probas = pipe.predict_proba(test_df)[:, 1]

    submission = pd.DataFrame({
        "policy_id": test_df["policy_id"],
        "is_claim": preds,
        "claim_probability": probas.round(4),
    })
    sub_path = BASE / "data" / "submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"\n[Pipeline] Submission saved ({len(submission):,} rows) -> {sub_path}")
    print(f"[Pipeline] Predicted claim rate: {preds.mean()*100:.2f}%")