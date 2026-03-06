# -*- coding: utf-8 -*-
"""
model_training.py
=================
Car Insurance Claim Prediction — Model Training & Evaluation Module

Handles:
  - Training multiple baseline and advanced classifiers
  - Comparing model performance
  - Hyperparameter tuning via Optuna
  - Full evaluation metrics (Accuracy, F1, ROC-AUC, Confusion Matrix)
  - Saving / loading trained models

Usage
-----
    from model_training import (
        train_all_models,
        compare_models,
        tune_lgbm,
        evaluate_model,
        save_model,
        load_model,
    )

    results = train_all_models(X_train, y_train, X_test, y_test)
    compare_models(results)

    best_params = tune_lgbm(X_train, y_train, X_test, y_test, n_trials=50)
    final_model = train_final_model(X_train, y_train, best_params)
    metrics     = evaluate_model(final_model, X_test, y_test)
    save_model(final_model, "models/best_model.pkl")
"""

import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# Optional boosting backends
try:
    from lightgbm import LGBMClassifier
    _LGBM = True
except ImportError:
    _LGBM = False

try:
    from xgboost import XGBClassifier
    _XGB = True
except ImportError:
    _XGB = False

try:
    from catboost import CatBoostClassifier
    _CAT = True
except ImportError:
    _CAT = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA = True
except ImportError:
    _OPTUNA = False

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# Default best params (from Optuna tuning in notebook)
DEFAULT_LGBM_PARAMS = {
    "n_estimators": 500,
    "num_leaves": 31,
    "max_depth": 10,
    "min_child_samples": 20,
    "learning_rate": 0.05,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}

DEFAULT_GBM_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "random_state": RANDOM_STATE,
}


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

def get_all_models() -> dict:
    """
    Return a dictionary of all classifiers to compare.

    Returns
    -------
    dict : {model_name: model_instance}
    """
    models = {
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000
        ),
        "KNN": KNeighborsClassifier(),
        "BernoulliNB": BernoulliNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
        "MLP": MLPClassifier(random_state=RANDOM_STATE, max_iter=300),
    }
    if _LGBM:
        models["LightGBM"] = LGBMClassifier(
            random_state=RANDOM_STATE, verbose=-1
        )
    if _XGB:
        models["XGBoost"] = XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss"
        )
    if _CAT:
        models["CatBoost"] = CatBoostClassifier(
            random_state=RANDOM_STATE, verbose=0
        )
    return models


# ---------------------------------------------------------------------------
# Training & Comparison
# ---------------------------------------------------------------------------

def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Train all registered models and return a comparison DataFrame.

    Parameters
    ----------
    X_train, y_train : training data
    X_test, y_test   : evaluation data
    verbose          : print results as models are trained

    Returns
    -------
    pd.DataFrame sorted by ROC-AUC (descending)
        Columns: Model, Accuracy, F1 Score, ROC-AUC
    """
    models = get_all_models()
    results = []

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            f1  = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_prob)

            results.append({
                "Model": name,
                "Accuracy": round(acc, 4),
                "F1 Score": round(f1, 4),
                "ROC-AUC": round(auc, 4),
            })
            if verbose:
                print(
                    f"  {name:<25} Acc={acc:.4f}  "
                    f"F1={f1:.4f}  AUC={auc:.4f}"
                )
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    df_results = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
    return df_results


def compare_models(results_df: pd.DataFrame) -> None:
    """
    Plot a grouped bar chart comparing model metrics.

    Parameters
    ----------
    results_df : pd.DataFrame
        Output of train_all_models().
    """
    melted = results_df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "F1 Score", "ROC-AUC"],
        var_name="Metric",
        value_name="Score",
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(results_df))
    width = 0.25
    metrics = ["Accuracy", "F1 Score", "ROC-AUC"]
    colors = ["steelblue", "tomato", "seagreen"]

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = results_df[metric].values
        bars = ax.bar(x + i * width, vals, width, label=metric, color=color, alpha=0.8)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(results_df["Model"], rotation=30, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Accuracy / F1 / ROC-AUC", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Hyperparameter Tuning
# ---------------------------------------------------------------------------

def tune_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_trials: int = 50,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Run Optuna hyperparameter search for LightGBM (or GradientBoosting).

    Parameters
    ----------
    X_train, y_train : training data
    X_test, y_test   : evaluation data
    n_trials         : number of Optuna trials. Default 50.
    random_state     : random seed for reproducibility.

    Returns
    -------
    dict : best hyperparameters
    """
    if not _OPTUNA:
        print("[WARN] optuna not installed. Returning default params.")
        return DEFAULT_LGBM_PARAMS if _LGBM else DEFAULT_GBM_PARAMS

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "random_state": random_state,
        }
        if _LGBM:
            params["num_leaves"] = trial.suggest_int("num_leaves", 10, 60)
            params["min_child_samples"] = trial.suggest_int(
                "min_child_samples", 5, 50
            )
            model = LGBMClassifier(**params, verbose=-1)
        else:
            model = GradientBoostingClassifier(**params)

        model.fit(X_train, y_train)
        return accuracy_score(y_test, model.predict(X_test))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\n[Optuna] Best Accuracy : {best.value:.4f}")
    print(f"[Optuna] Best Params   : {best.params}")
    return best.params


def tune_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_trials: int = 30,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Run Optuna hyperparameter search for Logistic Regression.

    Returns
    -------
    dict : best hyperparameters
    """
    if not _OPTUNA:
        print("[WARN] optuna not installed.")
        return {"C": 1.0, "solver": "lbfgs"}

    def objective(trial):
        params = {
            "C": trial.suggest_float("C", 0.01, 10.0, log=True),
            "solver": trial.suggest_categorical(
                "solver", ["lbfgs", "liblinear"]
            ),
            "max_iter": 1000,
            "random_state": random_state,
        }
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        return accuracy_score(y_test, model.predict(X_test))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_trial
    print(f"[Optuna LR] Best Accuracy: {best.value:.4f} | Params: {best.params}")
    return best.params


# ---------------------------------------------------------------------------
# Final Model Training
# ---------------------------------------------------------------------------

def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict = None,
) -> object:
    """
    Train the final boosting model with given hyperparameters.

    Uses LightGBM if installed, falls back to GradientBoostingClassifier.

    Parameters
    ----------
    X_train, y_train : training data
    params : dict, optional
        Hyperparameters. Defaults to tuned DEFAULT params if None.

    Returns
    -------
    Fitted model instance.
    """
    params = params or {}
    params.setdefault("random_state", RANDOM_STATE)

    if _LGBM:
        params.setdefault("verbose", -1)
        model = LGBMClassifier(**{**DEFAULT_LGBM_PARAMS, **params})
        print("[Model] Training LightGBM...")
    else:
        model = GradientBoostingClassifier(**{**DEFAULT_GBM_PARAMS, **params})
        print("[Model] LightGBM not found — training GradientBoostingClassifier...")

    model.fit(X_train, y_train)
    print("[Model] Training complete.")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    plot: bool = True,
) -> dict:
    """
    Full evaluation of a trained model.

    Parameters
    ----------
    model          : fitted sklearn-compatible model
    X_test, y_test : evaluation data
    plot           : whether to generate confusion matrix & ROC curve

    Returns
    -------
    dict: accuracy, f1, roc_auc, report (str), confusion_matrix (array)
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    cm  = confusion_matrix(y_test, y_pred)
    rep = classification_report(y_test, y_pred)

    print("=" * 55)
    print("  MODEL EVALUATION")
    print("=" * 55)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\nClassification Report:\n{rep}")
    print(f"Confusion Matrix:\n{cm}")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Confusion matrix
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No Claim", "Claim"],
        )
        disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
        axes[0].set_title("Confusion Matrix", fontsize=13)

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        axes[1].plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.4f}")
        axes[1].plot([0, 1], [0, 1], "k--", lw=1)
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("ROC Curve", fontsize=13)
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    return {
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "report": rep,
        "confusion_matrix": cm,
    }


def plot_feature_importance(model, feature_columns: list, top_n: int = 20) -> None:
    """
    Plot top-N feature importances as a horizontal bar chart.

    Parameters
    ----------
    model           : fitted model with feature_importances_ attribute
    feature_columns : list of feature names
    top_n           : number of top features to show
    """
    if not hasattr(model, "feature_importances_"):
        print("[INFO] This model does not expose feature_importances_.")
        return

    imp_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, top_n // 2))
    sns.barplot(
        x="importance", y="feature", data=imp_df,
        palette="Blues_r", orient="h",
    )
    plt.title(f"Top {top_n} Feature Importances", fontsize=13)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Model Persistence
# ---------------------------------------------------------------------------

def save_model(model, path: str = "models/best_model.pkl") -> None:
    """
    Serialize a trained model to disk.

    Parameters
    ----------
    model : fitted model
    path  : file path for the pickle file
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    print(f"[Model] Saved -> '{path}'")


def load_model(path: str = "models/best_model.pkl"):
    """
    Load a previously saved model from disk.

    Parameters
    ----------
    path : file path for the pickle file

    Returns
    -------
    Loaded model instance.
    """
    with open(path, "rb") as fh:
        model = pickle.load(fh)
    print(f"[Model] Loaded <- '{path}'")
    return model