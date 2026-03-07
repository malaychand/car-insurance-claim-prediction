# app.py — Car Insurance Claim Predictor
# Run: streamlit run app.py
# Requires: python -m src.training  (run once first)

import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.predict import predict_claim

# ── Paths ────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR    = os.path.join(BASE_DIR, "data")

# ── Page config ──────────────────────────────────────────────────

st.set_page_config(page_title="🚗 ClaimGuard AI", page_icon="🚗", layout="centered")

# ── Sidebar ──────────────────────────────────────────────────────

st.sidebar.title("🚗 ClaimGuard AI")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔍 Prediction", "📊 Data Explorer", "🛡️ Model Monitor"]
)
st.sidebar.markdown("""
<hr>
<div style="text-align:center;font-size:12px;color:gray;">
🚗 <b>ClaimGuard AI</b><br>
Car Insurance Claim Prediction System<br><br>
Machine Learning • Risk Analytics<br><br>
© 2025
</div>
""", unsafe_allow_html=True)
# ============================================================
# HOME
# ============================================================
# ============================================================
# HOME
# ============================================================
if page == "🏠 Home":

    # Image paths
    img1_path = os.path.join(BASE_DIR, "images", "image1.png")
    img2_path = os.path.join(BASE_DIR, "images", "image2.png")

    # ── Top Banner Image ─────────────────────────────────────
    if os.path.exists(img1_path):
        st.image(img1_path, use_container_width=True)

    st.title("🚗 Car Insurance Claim Predictor")
    st.markdown("---")

    col1, col2 = st.columns([1.3, 1])

    # ── Left Column (Description) ────────────────────────────
    with col1:
        st.markdown("""
        ### 🔍 Predict Insurance Claim Risk

        This application predicts the **probability that a customer will make a car insurance claim**
        in the next policy period based on:

        - Vehicle specifications
        - Policyholder demographics
        - Safety features
        - Policy information

        The model helps insurance companies with:

        - **Risk Assessment**
        - **Fraud Prevention**
        - **Premium Optimization**
        - **Operational Efficiency**
        """)

    # ── Right Column (Image + Pipeline) ──────────────────────
    with col2:
        if os.path.exists(img2_path):
            st.image(img2_path, use_container_width=True)


    # ── Model Status ─────────────────────────────────────────
    model_ok = os.path.exists(os.path.join(MODELS_DIR, "lightgbm_optuna_model.pkl"))

    if model_ok:
        st.success("✅ Model Loaded — Ready for Predictions")
    else:
        st.warning("⚠️ Model not found. Run the training script first:")
        st.code("python -m src.training", language="bash")

        st.markdown("---")
    st.markdown("#### ⚙️ Machine Learning Pipeline")

    st.markdown("""
        - Drop irrelevant columns (`policy_id`, `policy_tenure`)
        - Label Encoding of categorical variables
        - One-Hot Encoding of binary features
        - Correlation analysis and feature pruning
        - SMOTE oversampling for class imbalance
        - Feature scaling using StandardScaler
        - Hyperparameter tuning with **Optuna**
        - Final model: **LightGBM Classifier**
        """)
         
    st.markdown(

        """
        ### 📊 Features Used in the Model

        - Age of car
        - Age of policyholder
        - Vehicle model & segment
        - Fuel type
        - Engine specifications
        - Safety features (ESC, TPMS, airbags)
        - City population density
        - Vehicle dimensions & specifications

        Navigate to **🔍 Prediction** to test the model.
        """
    )

# ============================================================
# PREDICTION
# ============================================================

if page == "🔍 Prediction":
    st.title("🔍 Claim Probability Prediction")
    st.markdown("---")

    if not os.path.exists(os.path.join(MODELS_DIR, "lightgbm_optuna_model.pkl")):
        st.error("⚠️ Model not found.")
        st.code("python -m src.training", language="bash")
        st.stop()

    # ── Helper ───────────────────────────────────────────────────

    def num_input(label, default, min_v=0.0, step=1.0):
        return st.number_input(label, min_value=float(min_v),
                               value=float(default), step=float(step))

    def yn(label, key):
        return st.selectbox(label, ["Yes", "No"], key=key)

    # ── Actual value lists from CSV ───────────────────────────────

    # max_torque & max_power — object columns with specific string values
    TORQUE_OPTIONS = [
        "60Nm@3500rpm", "82.1Nm@3400rpm", "85Nm@3500rpm",
        "90Nm@3500rpm", "99Nm@3500rpm", "105Nm@4000rpm",
        "113Nm@4400rpm", "115Nm@4000rpm", "130Nm@4000rpm",
        "145Nm@3500rpm", "163Nm@1750rpm", "165Nm@4250rpm",
        "200Nm@1750rpm", "215Nm@1750rpm", "250Nm@1500rpm",
        "320Nm@1750rpm", "350Nm@1500rpm", "400Nm@1800rpm",
    ]

    POWER_OPTIONS = [
        "40.36bhp@6000rpm", "57bhp@6000rpm", "61.7bhp@6000rpm",
        "67bhp@6000rpm", "72bhp@6000rpm", "74bhp@4000rpm",
        "82bhp@6000rpm", "88.50bhp@6000rpm", "100bhp@3750rpm",
        "103.25bhp@3750rpm", "113bhp@4000rpm", "115bhp@5500rpm",
        "125bhp@5000rpm", "148bhp@5000rpm", "160bhp@5500rpm",
        "190bhp@3800rpm",
    ]

    ENGINE_OPTIONS = [
        "F8D Petrol Engine", "1.2 L K12N Dualjet", "1.2 L K12B",
        "1.0L Turbocharged", "1.5L Diesel", "1.5L Petrol",
        "2.0L Diesel", "1.4L Petrol", "1.4L Diesel",
        "1.6L Diesel", "1.6L Petrol", "2.5L Diesel",
    ]

    AREA_OPTIONS = [f"C{i}" for i in range(1, 26)]

    MODEL_OPTIONS = [f"M{i}" for i in range(1, 13)]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🚘 Vehicle Info")

        # make is int64 in CSV (pre-encoded 0–17), show as numeric
        make = int(num_input("Make (encoded 0–17, see data)", 0, min_v=0, step=1))

        model_name       = st.selectbox("Model", MODEL_OPTIONS)
        segment          = st.selectbox("Segment", ["A", "B", "C", "D", "E"])
        fuel_type        = st.selectbox("Fuel Type", ["CNG", "Diesel", "Electric", "Petrol"])
        transmission     = st.selectbox("Transmission", ["Automatic", "Manual"])
        steering         = st.selectbox("Steering Type", ["Manual", "Power"])
        rear_brakes      = st.selectbox("Rear Brakes", ["Disc", "Drum"])
        area_cluster     = st.selectbox("Area Cluster", AREA_OPTIONS)
        engine_type      = st.selectbox("Engine Type", ENGINE_OPTIONS)
        max_torque_val   = st.selectbox("Max Torque", TORQUE_OPTIONS)
        max_power_val    = st.selectbox("Max Power",  POWER_OPTIONS)

    with col2:
        st.subheader("📋 Specs & Policy")
        age_of_policyholder = num_input("Policyholder Age (normalised 0–1)", 0.5, step=0.01)
        age_of_car          = num_input("Car Age (normalised 0–1)", 0.1, step=0.01)
        population_density  = num_input("Population Density", 500, step=100)
        displacement        = num_input("Displacement (cc)", 1200, step=50)
        cylinder            = int(num_input("Cylinders", 4, step=1))
        gear_box            = int(num_input("Gear Box", 5, step=1))
        turning_radius      = num_input("Turning Radius (m)", 5.0, step=0.1)
        width               = int(num_input("Width (mm)", 1600, step=10))
        height              = int(num_input("Height (mm)", 1500, step=10))
        gross_weight        = int(num_input("Gross Weight (kg)", 1185, step=10))
        airbags             = int(num_input("Airbags", 2, step=1))
        ncap_rating         = int(num_input("NCAP Rating (0–5)", 0, step=1))

    st.markdown("---")
    st.subheader("🛡️ Safety Features (Yes / No)")
    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        is_esc       = yn("ESC",             "esc")
        is_adj_steer = yn("Adj. Steering",   "adj")
        is_tpms      = yn("TPMS",            "tpms")
        is_park_sens = yn("Parking Sensors", "psens")
        is_park_cam  = yn("Parking Camera",  "pcam")
        is_fog       = yn("Fog Lights",      "fog")
    with fc2:
        is_rr_wiper  = yn("Rear Wiper",      "rwiper")
        is_rr_washer = yn("Rear Washer",     "rwasher")
        is_rr_defog  = yn("Defogger",        "defog")
        is_brake_ast = yn("Brake Assist",    "brake")
        is_pwr_door  = yn("Power Door Locks","pdoor")
        is_central   = yn("Central Locking", "central")
    with fc3:
        is_pwr_steer = yn("Power Steering",  "psteer")
        is_seat_adj  = yn("Seat Adjustable", "seat")
        is_dn_mirror = yn("Day/Night Mirror","mirror")
        is_ecw       = yn("ECW",             "ecw")
        is_spd_alert = yn("Speed Alert",     "speed")

    st.markdown("---")
    st.info("💡 **make** is stored as an integer in the dataset (0=first brand, 1=second…). "
            "Check your `train.csv` to match the make you want.")

    if st.button("🔮 Predict Claim Probability"):
        # Build input row matching EXACT CSV column names and dtypes
        input_df = pd.DataFrame([{
            # int64 in CSV — pass as int
            "make":               make,
            "airbags":            airbags,
            "cylinder":           cylinder,
            "gear_box":           gear_box,
            "width":              width,
            "height":             height,
            "gross_weight":       gross_weight,
            "ncap_rating":        ncap_rating,
            # float64 in CSV
            "age_of_car":           age_of_car,
            "age_of_policyholder":  age_of_policyholder,
            "turning_radius":       turning_radius,
            "displacement":         float(displacement),
            "population_density":   float(population_density),
            # object → label encoded by preprocessing.py
            "max_torque":        max_torque_val,
            "max_power":         max_power_val,
            "engine_type":       engine_type,
            "area_cluster":      area_cluster,
            "model":             model_name,
            "transmission_type": transmission,
            "segment":           segment,
            # object → OHE by preprocessing.py
            "fuel_type":                          fuel_type,
            "is_esc":                             is_esc,
            "is_adjustable_steering":             is_adj_steer,
            "is_tpms":                            is_tpms,
            "is_parking_sensors":                 is_park_sens,
            "is_parking_camera":                  is_park_cam,
            "rear_brakes_type":                   rear_brakes,
            "steering_type":                      steering,
            "is_front_fog_lights":                is_fog,
            "is_rear_window_wiper":               is_rr_wiper,
            "is_rear_window_washer":              is_rr_washer,
            "is_rear_window_defogger":            is_rr_defog,
            "is_brake_assist":                    is_brake_ast,
            "is_power_door_locks":                is_pwr_door,
            "is_central_locking":                 is_central,
            "is_power_steering":                  is_pwr_steer,
            "is_driver_seat_height_adjustable":   is_seat_adj,
            "is_day_night_rear_view_mirror":       is_dn_mirror,
            "is_ecw":                             is_ecw,
            "is_speed_alert":                     is_spd_alert,
        }])

        try:
            prob, label = predict_claim(input_df, models_dir=MODELS_DIR)
            pct = prob * 100

            st.markdown("### 🎯 Result")
            if pct < 30:
                risk, color = "🟢 LOW RISK",    "green"
            elif pct < 60:
                risk, color = "🟡 MEDIUM RISK", "orange"
            else:
                risk, color = "🔴 HIGH RISK",   "red"

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Claim Probability", f"{pct:.2f}%")
            with c2:
                st.metric("Prediction", "Will Claim 🚨" if label == 1 else "No Claim ✅")

            st.markdown(f"**Risk Level:** :{color}[{risk}]")
            st.progress(min(int(pct), 100))

            if label == 1:
                st.error("⚠️ High claim likelihood — review premium or coverage.")
            else:
                st.success("✅ Low risk — standard policy terms apply.")

            st.balloons()

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            with st.expander("🔍 Debug info"):
                st.write("Input DataFrame:")
                st.dataframe(input_df)
                import traceback
                st.code(traceback.format_exc())

# ============================================================
# DATA EXPLORER
# ============================================================

if page == "📊 Data Explorer":
    import matplotlib.pyplot as plt

    st.title("📊 Dataset Explorer")

    train_path = os.path.join(DATA_DIR, "train.csv")
    if not os.path.exists(train_path):
        st.error("⚠️ train.csv not found in `data/` folder.")
        st.stop()

    @st.cache_data
    def load_train():
        return pd.read_csv(train_path)

    df = load_train()
    df_work = df.drop(columns=["policy_id", "policy_tenure"], errors="ignore")

    st.subheader("🔢 Dataset Shape")
    st.write(f"Rows: **{df.shape[0]:,}** | Columns: **{df.shape[1]}**")

    st.subheader("📋 Column Info")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Dtype":  [str(df[c].dtype) for c in df.columns],
        "Nulls":  [df[c].isnull().sum() for c in df.columns],
        "Unique": [df[c].nunique() for c in df.columns],
    })
    st.dataframe(info_df, width="stretch")

    st.subheader("📄 Sample (10 rows)")
    st.dataframe(df.head(10), width="stretch")

    st.subheader("🎯 Claim Distribution")
    cc = df["is_claim"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["No Claim (0)", "Claim (1)"], cc.values, color=["steelblue", "tomato"])
    for i, v in enumerate(cc.values):
        ax.text(i, v + 30, f"{v:,}", ha="center", fontweight="bold")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    st.warning("⚠️ Imbalanced — justifies SMOTE oversampling")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("🚘 Claim Rate by Segment")
        if "segment" in df_work.columns:
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            df_work.groupby("segment")["is_claim"].mean().sort_values().plot(
                kind="bar", ax=ax2, color="steelblue")
            ax2.set_ylabel("Claim Rate")
            st.pyplot(fig2)

    with col_b:
        st.subheader("⛽ Claim Rate by Fuel Type")
        if "fuel_type" in df_work.columns:
            fig3, ax3 = plt.subplots(figsize=(5, 3))
            df_work.groupby("fuel_type")["is_claim"].mean().sort_values().plot(
                kind="bar", ax=ax3, color="coral")
            ax3.set_ylabel("Claim Rate")
            st.pyplot(fig3)

    st.subheader("📍 Claim Rate by Area Cluster")
    if "area_cluster" in df_work.columns:
        fig4, ax4 = plt.subplots(figsize=(14, 3))
        df_work.groupby("area_cluster")["is_claim"].mean().sort_values().plot(
            kind="bar", ax=ax4, color="mediumseagreen")
        ax4.set_ylabel("Claim Rate")
        st.pyplot(fig4)

    st.subheader("📊 Numeric Stats")
    st.dataframe(df_work.select_dtypes(include=np.number).describe().T, width="stretch")

    # Show saved training plots
    st.subheader("🖼️ Training Plots")
    for fname, caption in [
        ("lightgbm_confusion_matrix.png", "LightGBM Confusion Matrix"),
        ("numerical_distribution.png",    "Numeric Distributions"),
        ("correlation_after_drop.png",    "Correlation Heatmap"),
        ("claim_distribution.png",        "Claim Distribution"),
    ]:
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(fpath):
            st.image(fpath, caption=caption, width=600)

# ============================================================
# MODEL MONITOR
# ============================================================

if page == "🛡️ Model Monitor":
    import plotly.express as px

    st.title("🛡️ Model Monitor")

    baseline_path = os.path.join(RESULTS_DIR, "baseline_results.csv")
    params_path   = os.path.join(RESULTS_DIR, "lightgbm_best_params.csv")

    if not os.path.exists(baseline_path):
        st.warning("No results yet — run training first.")
        st.code("python -m src.training", language="bash")
        st.stop()

    df_bl = pd.read_csv(baseline_path)

    st.subheader("📊 Baseline Comparison (with StandardScaler)")
    st.dataframe(df_bl.sort_values("Accuracy", ascending=False), width="stretch")

    fig1 = px.bar(df_bl.sort_values("Accuracy", ascending=False),
                  x="Model", y="Accuracy", color="Accuracy",
                  text_auto=".4f", title="Baseline Accuracy",
                  color_continuous_scale="Blues")
    st.plotly_chart(fig1)

    if os.path.exists(params_path):
        st.subheader("🔬 LightGBM Best Params (Optuna)")
        st.dataframe(pd.read_csv(params_path), width="stretch")

    cm_path = os.path.join(RESULTS_DIR, "lightgbm_confusion_matrix.png")
    if os.path.exists(cm_path):
        st.subheader("🎯 LightGBM Confusion Matrix")
        st.image(cm_path, width=400)

    best = df_bl.loc[df_bl["Accuracy"].idxmax()]
    st.success(
        f"🏆 Best Baseline: **{best['Model']}** — Accuracy: **{best['Accuracy']:.4f}**\n\n"
        f"Final model: **LightGBM (Optuna-tuned)** → `models/lightgbm_optuna_model.pkl`"
    )

    st.subheader("📁 Artifact Status")
    artifacts = [
        ("models/lightgbm_optuna_model.pkl", "LightGBM tuned model"),
        ("models/scaler.pkl",                "StandardScaler"),
        ("models/encoder_classes.json",      "LabelEncoder classes (JSON)"),
        ("models/ohe_columns.json",          "OHE column list (JSON)"),
        ("models/feature_columns.json",      "Feature column order (JSON)"),
        ("models/corr_drop_cols.json",       "Correlation drop list (JSON)"),
    ]
    rows = []
    for rel, desc in artifacts:
        full   = os.path.join(BASE_DIR, rel)
        status = "✅" if os.path.exists(full) else "❌ Missing"
        rows.append({"File": rel, "Description": desc, "Status": status})
    st.dataframe(pd.DataFrame(rows), width="stretch")
