# model_training.py
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb
from catboost import CatBoostClassifier

from data_preprocessing import preprocess_data
from feature_engineering import add_interactions
from utils import RANDOM_STATE, ensure_dir


def build_train_matrices(
    train_csv: str = "train.csv",
    test_csv: str = "test.csv",
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Preprocess data and add interaction features.
    Returns X, y for training and test_features, test_policy_ids for inference.
    """
    df_train, df_test, _ = preprocess_data(train_csv, test_csv)

    df_train = add_interactions(df_train)
    df_test = add_interactions(df_test)

    y = df_train["is_claim"].astype(int)
    X = df_train.drop("is_claim", axis=1)

    test_policy_ids = df_test["policy_id"] if "policy_id" in df_test.columns else None
    test_features = df_test.drop("policy_id", axis=1, errors="ignore")

    # align test to train columns
    test_features = test_features.reindex(columns=X.columns, fill_value=0)

    print("Train shape:", X.shape, " Test shape:", test_features.shape)
    return X, y, test_features, test_policy_ids


def cv_train_models(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> Tuple[str, Dict[str, float]]:
    """
    Cross-validate RandomForest, XGBoost, and CatBoost and return best model name and mean AUCs.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    rf_aucs, xgb_aucs, cat_aucs = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        pos_ratio = y_tr.value_counts()[0] / y_tr.value_counts()[1]

        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=16,
            min_samples_split=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        rf.fit(X_tr, y_tr)
        rf_probs = rf.predict_proba(X_val)[:, 1]
        rf_auc = roc_auc_score(y_val, rf_probs)
        rf_aucs.append(rf_auc)

        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            scale_pos_weight=pos_ratio,
            tree_method="hist",
            random_state=RANDOM_STATE,
        )
        xgb_model.fit(X_tr, y_tr)
        xgb_probs = xgb_model.predict_proba(X_val)[:, 1]
        xgb_auc = roc_auc_score(y_val, xgb_probs)
        xgb_aucs.append(xgb_auc)

        # CatBoost
        cat = CatBoostClassifier(
            iterations=700,
            depth=8,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            class_weights=[1.0, pos_ratio],
            verbose=False,
            random_seed=RANDOM_STATE,
        )
        cat.fit(X_tr, y_tr)
        cat_probs = cat.predict_proba(X_val)[:, 1]
        cat_auc = roc_auc_score(y_val, cat_probs)
        cat_aucs.append(cat_auc)

        print(
            f"Fold {fold}: RF AUC={rf_auc:.4f}, XGB AUC={xgb_auc:.4f}, CAT AUC={cat_auc:.4f}"
        )

    mean_scores: Dict[str, float] = {
        "RandomForest": float(np.mean(rf_aucs)),
        "XGBoost": float(np.mean(xgb_aucs)),
        "CatBoost": float(np.mean(cat_aucs)),
    }

    print("\nMean CV AUCs:")
    for name, score in mean_scores.items():
        print(f"{name}: {score:.5f}")

    best_name = max(mean_scores, key=mean_scores.get)
    print("\nBest model by mean CV AUC:", best_name, " -> ", mean_scores[best_name])

    return best_name, mean_scores


def train_full_model(
    X: pd.DataFrame, y: pd.Series, best_name: str
) -> object:
    """Train the chosen model on the full training data."""
    pos_ratio_full = y.value_counts()[0] / y.value_counts()[1]

    if best_name == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=16,
            min_samples_split=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    elif best_name == "XGBoost":
        model = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            scale_pos_weight=pos_ratio_full,
            tree_method="hist",
            random_state=RANDOM_STATE,
        )
    else:  # CatBoost
        model = CatBoostClassifier(
            iterations=700,
            depth=8,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            class_weights=[1.0, pos_ratio_full],
            verbose=False,
            random_seed=RANDOM_STATE,
        )

    model.fit(X, y)
    return model


def save_model(model: object, best_name: str, out_dir: str = "models") -> str:
    """Save trained model to disk and return path."""
    ensure_dir(out_dir)
    model_path = f"{out_dir}/best_model_{best_name}.pkl"
    joblib.dump(model, model_path)
    print(f"Saved best model to: {model_path}")
    return model_path


if __name__ == "__main__":
    # Full training pipeline entrypoint
    X, y, test_features, test_policy_ids = build_train_matrices(
        train_csv="train.csv",
        test_csv="test.csv",
    )

    best_name, mean_scores = cv_train_models(X, y)
    best_model = train_full_model(X, y, best_name)
    save_model(best_model, best_name)

    
    if test_policy_ids is not None:
     test_probs = best_model.predict_proba(test_features)[:, 1]
     submission = pd.DataFrame({"policy_id": test_policy_ids, "is_claim": test_probs})
     submission.to_csv("submission.csv", index=False)
