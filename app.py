
import io
import pickle
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

# Must import after defining path (pipeline.py must be in same directory)
from pipeline import CarInsurancePipeline, feature_engineering, TARGET

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Car Insurance Claim Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def load_or_train_pipeline(train_path: str = "train.csv"):
    """Load a saved pipeline or train a new one from train.csv."""
    try:
        pipe = CarInsurancePipeline.load("car_insurance_pipeline.pkl")
        return pipe, None
    except Exception:
        try:
            df = pd.read_csv(train_path)
            pipe = CarInsurancePipeline(apply_smote=True, scale_features=True)
            pipe.fit(df, verbose=False)
            pipe.save("car_insurance_pipeline.pkl")
            return pipe, df
        except Exception as e:
            return None, str(e)


def build_single_row(inputs: dict) -> pd.DataFrame:
    """Convert form inputs into a single-row DataFrame."""
    return pd.DataFrame([inputs])


# ---------------------------------------------------------------------------
# Sidebar — navigation
# ---------------------------------------------------------------------------
st.sidebar.image(
    "https://img.icons8.com/color/96/car--v1.png", width=80
)
st.sidebar.title("🚗 Car Insurance\nClaim Predictor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔮 Single Prediction", "📂 Batch Prediction", "📊 Model Insights"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model:** LightGBM (GradientBoosting fallback)\n\n"
    "**Pipeline:** feature_engineering → SMOTE → Scaler → Model\n\n"
    "**Target:** `is_claim` (0 = No Claim, 1 = Claim)"
)

# ---------------------------------------------------------------------------
# Load pipeline
# ---------------------------------------------------------------------------
pipe, train_df_or_err = load_or_train_pipeline()

if pipe is None:
    st.error(f"Could not load or train pipeline: {train_df_or_err}")
    st.info("Make sure `train.csv` is in the same directory as `app.py`.")
    st.stop()

# ---------------------------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------------------------
if page == "🏠 Home":
    st.title("🚗 Car Insurance Claim Prediction")
    st.markdown(
        """
        ### About This Project
        This application predicts whether a car insurance policyholder will file a
        **claim in the next 6 months** based on their vehicle and policy details.

        ---
        ### Business Use Cases
        | Use Case | Description |
        |---|---|
        | 🛡️ Fraud Prevention | Flag high-risk policyholders early |
        | 💰 Pricing Optimization | Adjust premiums based on predicted risk |
        | 🎯 Customer Targeting | Personalize campaigns for low-risk customers |
        | ⚙️ Operational Efficiency | Forecast claim volumes for resource planning |

        ---
        ### Dataset
        - **58,592** training observations  |  **43 features**
        - **Target:** `is_claim` (binary: 0 or 1)
        - **Class imbalance handled** via SMOTE oversampling

        ---
        ### Navigation
        - **Single Prediction** — fill in vehicle/policy details and get an instant prediction
        - **Batch Prediction** — upload a CSV file and download predictions
        - **Model Insights** — feature importance, ROC curve, confusion matrix
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Training Samples", "58,592")
    col2.metric("Features", "43 → ~28 (after corr. filter)")
    col3.metric("Model", "LightGBM / GBM")

# ---------------------------------------------------------------------------
# SINGLE PREDICTION PAGE
# ---------------------------------------------------------------------------
elif page == "🔮 Single Prediction":
    st.title("🔮 Single Prediction")
    st.markdown("Fill in the policyholder and vehicle details below.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Policy Details")
            policy_id = st.text_input("Policy ID", value="ID99999")
            age_of_car = st.slider("Age of Car (normalised)", 0.0, 1.0, 0.3, 0.01)
            age_of_policyholder = st.slider("Age of Policyholder (normalised)", 0.2, 1.0, 0.5, 0.01)
            population_density = st.number_input("Population Density", 100, 80000, 5000)
            area_cluster = st.selectbox("Area Cluster", [f"C{i}" for i in range(1, 23)])
            ncap_rating = st.selectbox("NCAP Rating (0-5)", [0, 1, 2, 3, 4, 5], index=3)

        with col2:
            st.subheader("Vehicle Details")
            make = st.number_input("Make (encoded)", 1, 14, 5)
            segment = st.selectbox("Segment", ["A", "B1", "B2", "C1", "C2"])
            model = st.selectbox("Model", [f"M{i}" for i in range(1, 50)])
            fuel_type = st.selectbox("Fuel Type", ["CNG", "Diesel", "Petrol"])
            engine_type = st.selectbox("Engine Type", ["K Series", "DDiS", "TDi", "CRDi", "VGT", "MPFI", "GDI"])
            displacement = st.selectbox("Displacement (cc)", [796, 998, 1197, 1248, 1373, 1498, 1596, 1998, 2498])
            cylinder = st.selectbox("Cylinders", [3, 4, 6])
            transmission_type = st.selectbox("Transmission", ["Manual", "Automatic"])
            gear_box = st.selectbox("Gear Box", [5, 6])
            airbags = st.selectbox("Airbags", [0, 2, 4, 6])

        with col3:
            st.subheader("Features")
            max_torque = st.text_input("Max Torque", "190Nm@2000rpm")
            max_power = st.text_input("Max Power", "85bhp@4000rpm")
            steering_type = st.selectbox("Steering Type", ["Power", "Manual", "Electric"])
            rear_brakes_type = st.selectbox("Rear Brakes", ["Disc", "Drum"])
            turning_radius = st.slider("Turning Radius (m)", 4.5, 6.5, 5.2, 0.1)
            width = st.number_input("Width (mm)", 1500, 1900, 1700)
            height = st.number_input("Height (mm)", 1400, 1800, 1550)
            gross_weight = st.number_input("Gross Weight (kg)", 1200, 2200, 1600)

            st.subheader("Safety Features")
            yn_features = {}
            for feat in ["is_esc", "is_adjustable_steering", "is_tpms",
                         "is_parking_sensors", "is_parking_camera",
                         "is_front_fog_lights", "is_rear_window_wiper",
                         "is_rear_window_washer", "is_rear_window_defogger",
                         "is_brake_assist", "is_power_door_locks",
                         "is_central_locking", "is_power_steering",
                         "is_driver_seat_height_adjustable",
                         "is_day_night_rear_view_mirror", "is_ecw", "is_speed_alert"]:
                yn_features[feat] = st.selectbox(
                    feat.replace("_", " ").title(), ["Yes", "No"], key=feat
                )

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        row = {
            "policy_id": policy_id,
            "policy_tenure": 0.5,
            "age_of_car": age_of_car,
            "age_of_policyholder": age_of_policyholder,
            "area_cluster": area_cluster,
            "population_density": population_density,
            "make": make,
            "segment": segment,
            "model": model,
            "fuel_type": fuel_type,
            "max_torque": max_torque,
            "max_power": max_power,
            "engine_type": engine_type,
            "airbags": airbags,
            "rear_brakes_type": rear_brakes_type,
            "displacement": displacement,
            "cylinder": cylinder,
            "transmission_type": transmission_type,
            "gear_box": gear_box,
            "steering_type": steering_type,
            "turning_radius": turning_radius,
            "width": width,
            "height": height,
            "gross_weight": gross_weight,
            "ncap_rating": ncap_rating,
            **yn_features,
        }

        input_df = build_single_row(row)
        pred = pipe.predict(input_df)[0]
        proba = pipe.predict_proba(input_df)[0][1]

        st.markdown("---")
        c1, c2 = st.columns(2)
        if pred == 1:
            c1.error(f"### ⚠️ CLAIM PREDICTED\nThis policyholder is **likely to file a claim**.")
        else:
            c1.success(f"### ✅ NO CLAIM PREDICTED\nThis policyholder is **unlikely to file a claim**.")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(proba * 100, 1),
            title={"text": "Claim Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "crimson" if pred == 1 else "green"},
                "steps": [
                    {"range": [0, 30], "color": "#d4edda"},
                    {"range": [30, 60], "color": "#fff3cd"},
                    {"range": [60, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        fig.update_layout(height=300)
        c2.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# BATCH PREDICTION PAGE
# ---------------------------------------------------------------------------
elif page == "📂 Batch Prediction":
    st.title("📂 Batch Prediction")
    st.markdown(
        "Upload a CSV file with the same columns as `test.csv` "
        "(without `is_claim`). The app will return predictions for every row."
    )

    uploaded = st.file_uploader("Upload test CSV", type="csv")

    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.write(f"**Uploaded:** {df_up.shape[0]:,} rows × {df_up.shape[1]} columns")
        st.dataframe(df_up.head(5))

        with st.spinner("Running predictions..."):
            preds = pipe.predict(df_up)
            probas = pipe.predict_proba(df_up)[:, 1]

        results = df_up[["policy_id"]].copy() if "policy_id" in df_up.columns else df_up.iloc[:, :1].copy()
        results["is_claim"] = preds
        results["claim_probability"] = probas.round(4)

        st.success(f"Done! Predicted **{preds.sum():,}** claims out of {len(preds):,} policies ({preds.mean()*100:.1f}%).")
        st.dataframe(results.head(20))

        # Distribution chart
        fig = px.histogram(
            results, x="claim_probability", nbins=50,
            color_discrete_sequence=["#1f77b4"],
            title="Distribution of Claim Probabilities",
            labels={"claim_probability": "P(Claim)"},
        )
        st.plotly_chart(fig, use_container_width=True)

        # Download button
        csv_buf = io.StringIO()
        results.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download Predictions CSV",
            data=csv_buf.getvalue(),
            file_name="predictions.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# MODEL INSIGHTS PAGE
# ---------------------------------------------------------------------------
elif page == "📊 Model Insights":
    st.title("📊 Model Insights")

    # Feature Importance
    st.subheader("Feature Importance (Top 20)")
    model = pipe.model_
    feat_names = pipe.feature_columns_

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": feat_names,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False).head(20)

        fig = px.bar(
            importance_df,
            x="importance", y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Blues",
            title="Top 20 Feature Importances",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=550)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importances not available for this model.")

    # Evaluate on uploaded data
    st.markdown("---")
    st.subheader("Evaluate on Labelled Data")
    st.markdown("Upload a CSV with `is_claim` column to see full metrics.")

    eval_file = st.file_uploader("Upload labelled CSV", type="csv", key="eval")
    if eval_file:
        eval_df = pd.read_csv(eval_file)
        if TARGET not in eval_df.columns:
            st.error(f"Uploaded file must contain '{TARGET}' column.")
        else:
            with st.spinner("Evaluating..."):
                metrics = pipe.evaluate(eval_df)

            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            c2.metric("F1 Score", f"{metrics['f1']:.4f}")
            c3.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

            st.text("Classification Report:")
            st.code(metrics["report"])

            # Confusion matrix heatmap
            cm = metrics["confusion_matrix"]
            fig_cm = px.imshow(
                cm, text_auto=True, color_continuous_scale="Blues",
                labels={"x": "Predicted", "y": "Actual"},
                x=["No Claim", "Claim"], y=["No Claim", "Claim"],
                title="Confusion Matrix",
            )
            fig_cm.update_layout(width=450, height=400)
            st.plotly_chart(fig_cm)

            # ROC curve
            X_ev, y_ev = pipe._preprocess(eval_df, fit=False)
            y_prob = pipe.model_.predict_proba(X_ev)[:, 1]
            fpr, tpr, _ = roc_curve(y_ev, y_prob)
            auc_val = roc_auc_score(y_ev, y_prob)
            fig_roc = px.line(
                x=fpr, y=tpr,
                labels={"x": "False Positive Rate", "y": "True Positive Rate"},
                title=f"ROC Curve  (AUC = {auc_val:.4f})",
            )
            fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                              line=dict(dash="dash", color="gray"))
            st.plotly_chart(fig_roc, use_container_width=True)
