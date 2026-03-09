
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
 <b>ClaimGuard AI</b><br>
Car Insurance Claim Prediction System<br><br>
Machine Learning • Risk Analytics<br><br>
© 2025
</div>
""", unsafe_allow_html=True)
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


    col1, col2 = st.columns([1.3, 1])

    # ── Left Column (Description) ────────────────────────────
    with col1:
        st.markdown("""
        ### 🔍 Predict Insurance Claim Risk

        This application predicts the **probability that a customer will make a car insurance claim**
        in the next policy period based on machine learning models trained on vehicle, driver,
        and policy information.

        The system helps insurance companies with:

        - **Risk Assessment**
        - **Fraud Detection**
        - **Premium Optimization**
        - **Operational Efficiency**
        """)

        st.markdown("### 🔗 Project Links")

        st.markdown("""
- 📊 **MLflow Experiments:**  
  https://dagshub.com/malaychand/car-insurance-claim-prediction.mlflow/

- 🚀 **Live Streamlit App:**  
  https://car-insurance-claim-prediction-vbsunpjsqzaeaxzs8bgjnt.streamlit.app/

- 📁 **DagsHub Repository:**  
  https://dagshub.com/malaychand/car-insurance-claim-prediction

- 💻 **GitHub Repository:**  
  https://github.com/malaychand/car-insurance-claim-prediction
        """)

    # ── Right Column (Image) ─────────────────────────────
    with col2:
        if os.path.exists(img2_path):
            st.image(img2_path, use_container_width=True)

    # ── Model Status ─────────────────────────────────────────
    model_ok = os.path.exists(os.path.join(MODELS_DIR, "lightgbm_optuna_model.pkl"))

    st.markdown("---")

    if model_ok:
        st.success("✅ Model Loaded — Ready for Predictions")
    else:
        st.warning("⚠️ Model not found. Run the training script first:")
        st.code("python -m src.training", language="bash")

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


        except Exception as e:
            st.error(f"Prediction failed: {e}")
            with st.expander("🔍 Debug info"):
                st.write("Input DataFrame:")
                st.dataframe(input_df)
                import traceback
                st.code(traceback.format_exc())


    # ============================================================
    # SAMPLE DATA TESTING
    # ============================================================

    st.markdown("---")
    st.subheader("🧪 Test Model Using Dataset Rows")

    train_path = os.path.join(DATA_DIR, "train.csv")
    pred_path  = os.path.join(DATA_DIR, "test_predictions.csv")

    if os.path.exists(train_path):

        train_df = pd.read_csv(train_path)

        # Sample rows
        claim_1 = train_df[train_df["is_claim"] == 1].sample(3, random_state=4)
        claim_0 = train_df[train_df["is_claim"] == 0].sample(3, random_state=4)

        with st.expander("🔍 Show Sample Rows (Claim = 1)", expanded=False):
            st.dataframe(claim_1)

            for i, row in claim_1.iterrows():
                if st.button(f"Check Probability for Claim Row {i}"):
                    row_df = row.drop("is_claim").to_frame().T
                    prob, label = predict_claim(row_df, models_dir=MODELS_DIR)

                    st.success(f"Predicted Probability: {prob*100:.2f}%")

        with st.expander("🔍 Show Sample Rows (Claim = 0)", expanded=False):
            st.dataframe(claim_0)

            for i, row in claim_0.iterrows():
                if st.button(f"Check Probability for Non-Claim Row {i}"):
                    row_df = row.drop("is_claim").to_frame().T
                    prob, label = predict_claim(row_df, models_dir=MODELS_DIR)

                    st.success(f"Predicted Probability: {prob*100:.2f}%")

    # ============================================================
    # SHOW TEST PREDICTIONS
    # ============================================================

    if os.path.exists(pred_path):

        with st.expander("📄 View Predicted Test File (Hidden by Default)", expanded=False):

            pred_df = pd.read_csv(pred_path)

            st.write("Preview of predicted results")
            st.dataframe(pred_df.head(10))



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

    st.subheader("📄 Sample Data")
    st.dataframe(df.head(10), width="stretch")
    # =====================================================
    # Training Artifacts (EDA Only)
    # =====================================================

    st.subheader("📊 Exploratory Data Analysis Artifacts")
    st.markdown(
        "These visualizations were generated during exploratory data analysis (EDA) "
        "to understand the dataset structure, distributions, and correlations."
    )

    plots = [
        (
            "claim_distribution.png",
            "Claim Distribution",
            "Shows the number of customers who filed insurance claims vs those who did not. "
            "The dataset is imbalanced, which is why SMOTE was applied during model training.",
        ),
        (
            "numerical_distribution.png",
            "Numeric Feature Distributions",
            "Histograms of numerical features such as vehicle age, engine displacement, "
            "and policyholder age to understand distribution patterns and potential outliers.",
        ),
        (
            "correlation_after_drop.png",
            "Correlation Heatmap",
            "Displays correlation between numerical features after removing highly correlated "
            "variables. This helps reduce multicollinearity before model training.",
        ),
    ]

    for file, caption, description in plots:
        path = os.path.join(RESULTS_DIR, file)

        if os.path.exists(path):
            with st.expander(f"📷 {caption}", expanded=True):
                st.markdown(description)
                st.image(path, caption=caption, use_container_width=True)

        else:
            with st.expander(f"📷 {caption} *(not found)*", expanded=False):
                st.markdown(description)
                st.info(f"Image `{file}` not found in results directory.")
# ============================================================
# MODEL MONITOR
# ============================================================

if page == "🛡️ Model Monitor":

    import plotly.express as px

    st.title("🛡️ Model Monitor")

    baseline_path = os.path.join(RESULTS_DIR, "baseline_results.csv")
    params_path   = os.path.join(RESULTS_DIR, "lightgbm_best_params.csv")

    if not os.path.exists(baseline_path):
        st.warning("No results found — run training first.")
        st.stop()

    df_bl = pd.read_csv(baseline_path)

    st.subheader("📊 Baseline Model Comparison")
    st.dataframe(df_bl.sort_values("ROC_AUC",ascending=False))

    # Accuracy Chart
    fig1 = px.bar(
        df_bl.sort_values("Accuracy",ascending=False),
        x="Model",
        y="Accuracy",
        color="Accuracy",
        text_auto=".4f",
        title="Model Accuracy Comparison",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig1)

    # ROC Comparison Plot
    roc_path = os.path.join(RESULTS_DIR,"roc_curves_baselines.png")
    if os.path.exists(roc_path):
        st.subheader("📉 ROC Curve Comparison")
        st.image(roc_path,use_container_width=True)

    # Model Comparison Plot
    comp_path = os.path.join(RESULTS_DIR,"model_comparison_bar.png")
    if os.path.exists(comp_path):
        st.subheader("🏆 Model Performance Comparison")
        st.image(comp_path,use_container_width=True)

    # Confusion Matrices
    st.subheader("🎯 Confusion Matrices")

    col1,col2 = st.columns(2)

    with col1:
        lgb_cm = os.path.join(RESULTS_DIR,"lightgbm_confusion_matrix.png")
        if os.path.exists(lgb_cm):
            st.image(lgb_cm,caption="LightGBM Confusion Matrix")

    with col2:
        cat_cm = os.path.join(RESULTS_DIR,"catboost_confusion_matrix.png")
        if os.path.exists(cat_cm):
            st.image(cat_cm,caption="CatBoost Confusion Matrix")

    # Feature Importance
    fi_path = os.path.join(RESULTS_DIR,"lightgbm_feature_importance.png")

    if os.path.exists(fi_path):
        st.subheader("📊 Feature Importance")
        st.image(fi_path,use_container_width=True)

    # Best Model
    best = df_bl.loc[df_bl["ROC_AUC"].idxmax()]

    st.success(
        f"""
🏆 **Best Baseline Model:** {best['Model']}

Accuracy: **{best['Accuracy']:.4f}**  
ROC-AUC: **{best['ROC_AUC']:.4f}**  
F1 Score: **{best['F1']:.4f}**

Final Production Model → **LightGBM (Optuna tuned)**
"""
    )