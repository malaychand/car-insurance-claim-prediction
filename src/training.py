# src/training.py
# ---------------------------------------------------------------
# LOCAL training script — mirrors the Colab notebook exactly.
#
# Key facts from df.info():
#   - make         → int64  (already numeric in CSV, NOT string)
#   - policy_id    → dropped (not in df after Colab load)
#   - policy_tenure→ float64, dropped in Colab
#   - LABEL_ENCODE_COLS = max_torque, max_power, engine_type,
#                         area_cluster, model, transmission_type, segment
#                         (all object dtype in CSV)
#   - OHE_COLS     = fuel_type, is_esc, ... (all object dtype)
#
# Usage: python -m src.training
# ---------------------------------------------------------------

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier,
                               AdaBoostClassifier,
                               GradientBoostingClassifier)
from imblearn.over_sampling import SMOTE

# ── Paths ────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Columns that are object dtype → need LabelEncoding ──────────
# (make is ALREADY int64 in CSV — do NOT label-encode it)
LABEL_ENCODE_COLS = [
    "max_torque", "max_power", "engine_type",
    "area_cluster", "model", "transmission_type", "segment"
]

# ── Columns for One-Hot Encoding ─────────────────────────────────
OHE_COLS = [
    "fuel_type", "is_esc", "is_adjustable_steering", "is_tpms",
    "is_parking_sensors", "is_parking_camera", "rear_brakes_type",
    "steering_type", "is_front_fog_lights", "is_rear_window_wiper",
    "is_rear_window_washer", "is_rear_window_defogger", "is_brake_assist",
    "is_power_door_locks", "is_central_locking", "is_power_steering",
    "is_driver_seat_height_adjustable", "is_day_night_rear_view_mirror",
    "is_ecw", "is_speed_alert"
]

# Hardcoded from Colab output (corr > 0.95 drop list)
CORR_DROP_COLS = [
    'length', 'is_esc_Yes', 'is_adjustable_steering_No',
    'is_adjustable_steering_Yes', 'is_tpms_No', 'is_tpms_Yes',
    'is_parking_sensors_Yes', 'is_parking_camera_Yes',
    'rear_brakes_type_Disc', 'rear_brakes_type_Drum',
    'steering_type_Power', 'is_front_fog_lights_Yes',
    'is_rear_window_wiper_No', 'is_rear_window_wiper_Yes',
    'is_rear_window_washer_No', 'is_rear_window_washer_Yes',
    'is_rear_window_defogger_Yes', 'is_brake_assist_Yes',
    'is_power_door_locks_Yes', 'is_central_locking_No',
    'is_central_locking_Yes', 'is_power_steering_No',
    'is_power_steering_Yes', 'is_driver_seat_height_adjustable_No',
    'is_driver_seat_height_adjustable_Yes',
    'is_day_night_rear_view_mirror_Yes', 'is_ecw_No',
    'is_ecw_Yes', 'is_speed_alert_Yes'
]


def run_training():
    print("=" * 60)
    print("CAR INSURANCE CLAIM — LOCAL TRAINING PIPELINE")
    print("=" * 60)

    # ── 1. Load ───────────────────────────────────────────────────
    print("\n📂 Loading data...")
    train_path = os.path.join(DATA_DIR, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"train.csv not found at: {train_path}\n"
            "Place train.csv inside the data/ folder."
        )

    df = pd.read_csv(train_path)
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")

    # ── 2. Drop unused columns (mirrors Colab) ────────────────────
    drop_init = [c for c in ["policy_id", "policy_tenure"] if c in df.columns]
    df.drop(columns=drop_init, inplace=True)
    print(f"   Dropped: {drop_init}")

    # ── 3. Quick EDA plots ────────────────────────────────────────
    print("\n📊 Saving EDA plots...")
    _plot_claim_dist(df)
    _plot_num_dist(df)

    # ── 4. Label Encoding (object cols only) ──────────────────────
    print("\n🔤 Label encoding (object columns)...")
    encoder_classes = {}
    le_fitted       = {}

    for col in LABEL_ENCODE_COLS:
        if col in df.columns and df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_fitted[col]       = le
            encoder_classes[col] = list(le.classes_)
            print(f"   {col}: {len(le.classes_)} classes → {le.classes_[:5]}...")
        elif col in df.columns:
            print(f"   {col}: already numeric ({df[col].dtype}), skipped")

    # Save encoder classes as JSON (version-independent)
    with open(os.path.join(MODELS_DIR, "encoder_classes.json"), "w") as f:
        json.dump(encoder_classes, f, indent=2)
    print("   ✅ encoder_classes.json saved")

    # Save fitted encoders with joblib protocol=4
    joblib.dump(le_fitted,
                os.path.join(MODELS_DIR, "label_encoders.pkl"),
                compress=3, protocol=4)
    print("   ✅ label_encoders.pkl saved")

    # ── 5. One-Hot Encoding ───────────────────────────────────────
    print("\n🔢 One-hot encoding...")
    ohe_present = [c for c in OHE_COLS if c in df.columns]
    dms = pd.get_dummies(df[ohe_present])
    print(f"   OHE produced {dms.shape[1]} columns")

    # Save OHE columns as JSON
    ohe_columns = list(dms.columns)
    with open(os.path.join(MODELS_DIR, "ohe_columns.json"), "w") as f:
        json.dump(ohe_columns, f, indent=2)
    joblib.dump(dms, os.path.join(MODELS_DIR, "dummies.pkl"),
                compress=3, protocol=4)
    print("   ✅ ohe_columns.json + dummies.pkl saved")

    df.drop(columns=ohe_present, inplace=True)
    df = pd.concat([df, dms], axis=1)

    # ── 6. Drop high-correlation columns ─────────────────────────
    print("\n✂️  Dropping high-correlation columns...")
    # Recompute to catch any extras specific to this dataset version
    cor   = df.corr(numeric_only=True).abs()
    upper = cor.where(np.triu(np.ones(cor.shape), k=1).astype(bool))
    computed_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
    all_drop      = list(set(CORR_DROP_COLS) | set(computed_drop))
    drop_present  = [c for c in all_drop if c in df.columns]
    df.drop(columns=drop_present, inplace=True)
    print(f"   Dropped {len(drop_present)} columns")

    with open(os.path.join(MODELS_DIR, "corr_drop_cols.json"), "w") as f:
        json.dump(drop_present, f, indent=2)

    # ── 7. SMOTE ──────────────────────────────────────────────────
    print("\n⚖️  Applying SMOTE (random_state=42)...")
    X = df.drop("is_claim", axis=1)
    y = df["is_claim"]

    # Save the numeric make values range for UI reference
    if "make" in X.columns:
        make_range = {"min": int(X["make"].min()), "max": int(X["make"].max())}
        with open(os.path.join(MODELS_DIR, "make_range.json"), "w") as f:
            json.dump(make_range, f)

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"   After SMOTE → {y_res.value_counts().to_dict()}")

    # ── 8. Train / test split ─────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.25, random_state=0
    )
    print(f"\n   Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    # ── 9. StandardScaler ─────────────────────────────────────────
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"),
                compress=3, protocol=4)

    # Save feature columns (JSON + pkl)
    feature_columns = list(X_train.columns)
    with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f, indent=2)
    joblib.dump(feature_columns, os.path.join(MODELS_DIR, "feature_columns.pkl"),
                compress=3, protocol=4)
    print(f"   ✅ scaler + feature_columns saved ({len(feature_columns)} features)")

    # ── 10. Baseline models ───────────────────────────────────────
    print("\n🚀 Training baseline models...")
    _run_baselines(X_train_s, X_test_s, y_train, y_test)

    # ── 11. Optuna LightGBM ───────────────────────────────────────
    print("\n🔍 Optuna LightGBM tuning (50 trials)...")
    lgb_model = _tune_lightgbm(X_train_s, X_test_s, y_train, y_test)

    # ── 12. Final CM ──────────────────────────────────────────────
    _save_lgb_cm(y_test, lgb_model.predict(X_test_s))

    print("\n" + "=" * 60)
    print("✅ Done! Run:  streamlit run app.py")
    print("=" * 60)


# ── Helpers ──────────────────────────────────────────────────────

def _plot_claim_dist(df):
    try:
        cc = df["is_claim"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(["No Claim", "Claim"], cc.values, color=["steelblue", "tomato"])
        for i, v in enumerate(cc.values):
            ax.text(i, v + 30, f"{v:,}", ha="center", fontweight="bold")
        ax.set_title("Claim Distribution")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "claim_distribution.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"   (plot skipped: {e})")


def _plot_num_dist(df):
    try:
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c != "is_claim"][:4]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for k, col in enumerate(num_cols):
            sns.histplot(df[col], kde=True, ax=axes[k // 2][k % 2])
            axes[k // 2][k % 2].set_title(col)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "numerical_distribution.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"   (plot skipped: {e})")


def _run_baselines(X_tr, X_te, y_tr, y_te):
    baselines = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN":                 KNeighborsClassifier(),
        "AdaBoost":            AdaBoostClassifier(),
        "Decision Tree":       DecisionTreeClassifier(),
        "Random Forest":       RandomForestClassifier(n_estimators=100),
        "Gradient Boosting":   GradientBoostingClassifier(),
    }
    for pkg, cls, key in [
        ("xgboost",  "XGBClassifier",   "XGBoost"),
        ("lightgbm", "LGBMClassifier",  "LightGBM"),
        ("catboost", "CatBoostClassifier", "CatBoost"),
    ]:
        try:
            mod = __import__(pkg)
            kw  = {"verbose": 0} if key == "CatBoost" else ({"verbose": -1} if key == "LightGBM" else {"eval_metric": "logloss", "verbosity": 0})
            baselines[key] = getattr(mod, cls)(**kw)
        except Exception:
            pass

    model_results = []
    for name, mdl in baselines.items():
        try:
            mdl.fit(X_tr, y_tr)
            acc = accuracy_score(y_te, mdl.predict(X_te))
            print(f"   {name:28s}  Accuracy: {acc:.4f}")
            joblib.dump(mdl,
                        os.path.join(MODELS_DIR,
                                     name.replace(" ", "_").lower() + "_baseline.pkl"),
                        compress=3, protocol=4)
            model_results.append({"Model": name, "Accuracy": round(acc, 4)})
        except Exception as e:
            print(f"   {name:28s}  SKIPPED ({e})")

    pd.DataFrame(model_results).to_csv(
        os.path.join(RESULTS_DIR, "baseline_results.csv"), index=False)
    print("   ✅ baseline_results.csv saved")


def _tune_lightgbm(X_tr, X_te, y_tr, y_te):
    import optuna
    from lightgbm import LGBMClassifier
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        p = {
            "n_estimators":      trial.suggest_int("n_estimators",      50, 500),
            "num_leaves":        trial.suggest_int("num_leaves",          2,  50),
            "max_depth":         trial.suggest_int("max_depth",           2,  15),
            "min_child_samples": trial.suggest_int("min_child_samples",   1,  50),
            "learning_rate":     trial.suggest_float("learning_rate",  0.05, 0.5, log=True),
            "verbose": -1,
        }
        m = LGBMClassifier(**p)
        m.fit(X_tr, y_tr)
        return accuracy_score(y_te, m.predict(X_te))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    best = study.best_trial
    print(f"\n   Best Accuracy : {best.value:.4f}")
    print(f"   Best Params   : {best.params}")

    lgb_final = LGBMClassifier(**best.params, verbose=-1)
    lgb_final.fit(X_tr, y_tr)

    joblib.dump(lgb_final,
                os.path.join(MODELS_DIR, "lightgbm_optuna_model.pkl"),
                compress=3, protocol=4)
    pd.DataFrame([best.params]).to_csv(
        os.path.join(RESULTS_DIR, "lightgbm_best_params.csv"), index=False)
    print("   ✅ lightgbm_optuna_model.pkl saved")
    return lgb_final


def _save_lgb_cm(y_true, y_pred):
    cm  = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    print(f"\n   Final Accuracy: {acc:.4f}")
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Claim", "Claim"],
                yticklabels=["No Claim", "Claim"])
    plt.title(f"LightGBM Confusion Matrix (acc={acc:.4f})")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "lightgbm_confusion_matrix.png"), dpi=200)
    plt.close()
    print("   ✅ lightgbm_confusion_matrix.png saved")


if __name__ == "__main__":
    run_training()
