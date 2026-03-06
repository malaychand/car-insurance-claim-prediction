# -*- coding: utf-8 -*-
"""
app.py — Streamlit app (see src/pipeline.py for model logic)
Run: streamlit run app.py
"""
import io, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pipeline import CarInsurancePipeline
from feature_engineering import TARGET

warnings.filterwarnings("ignore")

BASE       = Path(__file__).resolve().parent
TRAIN_PATH = BASE / "data" / "train.csv"
MODEL_PATH = BASE / "models" / "car_insurance_pipeline.pkl"

st.set_page_config(page_title="Car Insurance Claim Predictor",
                   page_icon="🚗", layout="wide")

@st.cache_resource(show_spinner="Loading model...")
def get_pipeline():
    if MODEL_PATH.exists():
        return CarInsurancePipeline.load(str(MODEL_PATH))
    if not TRAIN_PATH.exists():
        st.error("No saved model or training data found."); st.stop()
    st.info("Training model from scratch...")
    train_df = pd.read_csv(TRAIN_PATH)
    pipe = CarInsurancePipeline(apply_smote=True, scale_features=True)
    pipe.fit(train_df, verbose=False)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    pipe.save(str(MODEL_PATH))
    return pipe

pipe = get_pipeline()

st.sidebar.image("https://img.icons8.com/color/96/car--v1.png", width=72)
st.sidebar.title("🚗 Car Insurance\nClaim Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate",
    ["🏠 Overview","🔮 Single Prediction","📂 Batch Prediction","📊 Model Insights"])
st.sidebar.markdown("---")
st.sidebar.info("**Modules**\n- feature_engineering.py\n- datapreprocessing.py\n- model_training.py\n- pipeline.py")

# ── OVERVIEW ────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("🚗 Car Insurance Claim Prediction")
    st.markdown("""
### Problem Statement
Predict whether a policyholder will file a **car insurance claim in the next 6 months**.

---
### Pipeline Architecture
```
Raw CSV → feature_engineering.py → datapreprocessing.py → model_training.py → pipeline.py → app.py
```
---
### Business Use Cases
| Use Case | Description |
|---|---|
| 🛡️ Fraud Prevention | Flag high-risk policyholders early |
| 💰 Pricing Optimization | Adjust premiums by predicted risk |
| 🎯 Customer Targeting | Personalise offers for low-risk segments |
| ⚙️ Operational Efficiency | Forecast claim volumes |
""")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Train Samples","58,592"); c2.metric("Features","43 → ~28")
    c3.metric("Class Imbalance","~12% claims"); c4.metric("Model","LightGBM")

# ── SINGLE PREDICTION ───────────────────────────────────────────────────────
elif page == "🔮 Single Prediction":
    st.title("🔮 Single Prediction")
    with st.form("form"):
        c1,c2,c3 = st.columns(3)
        with c1:
            st.subheader("Policy")
            policy_id = st.text_input("Policy ID","ID99999")
            age_of_car = st.slider("Age of Car",0.0,1.0,0.3,0.01)
            age_of_policyholder = st.slider("Policyholder Age",0.2,1.0,0.5,0.01)
            population_density = st.number_input("Population Density",100,80000,5000)
            area_cluster = st.selectbox("Area Cluster",[f"C{i}" for i in range(1,23)])
            ncap_rating = st.selectbox("NCAP Rating",[0,1,2,3,4,5],index=3)
        with c2:
            st.subheader("Vehicle")
            make = st.number_input("Make",1,14,5)
            segment = st.selectbox("Segment",["A","B1","B2","C1","C2"])
            model_name = st.selectbox("Model",[f"M{i}" for i in range(1,50)])
            fuel_type = st.selectbox("Fuel",["CNG","Diesel","Petrol"])
            engine_type = st.selectbox("Engine",["K Series","DDiS","TDi","CRDi","VGT","MPFI","GDI"])
            displacement = st.selectbox("Displacement (cc)",[796,998,1197,1248,1373,1498,1596,1998,2498])
            cylinder = st.selectbox("Cylinders",[3,4,6])
            transmission_type = st.selectbox("Transmission",["Manual","Automatic"])
            gear_box = st.selectbox("Gear Box",[5,6])
            airbags = st.selectbox("Airbags",[0,2,4,6])
            max_torque = st.text_input("Max Torque","190Nm@2000rpm")
            max_power  = st.text_input("Max Power","85bhp@4000rpm")
            steering_type = st.selectbox("Steering",["Power","Manual","Electric"])
            rear_brakes_type = st.selectbox("Rear Brakes",["Disc","Drum"])
            turning_radius = st.slider("Turning Radius",4.5,6.5,5.2,0.1)
            width  = st.number_input("Width (mm)",1500,1900,1700)
            height = st.number_input("Height (mm)",1400,1800,1550)
            gross_weight = st.number_input("Gross Weight (kg)",1200,2200,1600)
        with c3:
            st.subheader("Safety Features")
            bools = ["is_esc","is_adjustable_steering","is_tpms","is_parking_sensors",
                     "is_parking_camera","is_front_fog_lights","is_rear_window_wiper",
                     "is_rear_window_washer","is_rear_window_defogger","is_brake_assist",
                     "is_power_door_locks","is_central_locking","is_power_steering",
                     "is_driver_seat_height_adjustable","is_day_night_rear_view_mirror",
                     "is_ecw","is_speed_alert"]
            bool_vals = {f: st.selectbox(f.replace("_"," ").title(),["Yes","No"],key=f) for f in bools}
        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        row = {"policy_id":policy_id,"policy_tenure":0.5,"age_of_car":age_of_car,
               "age_of_policyholder":age_of_policyholder,"area_cluster":area_cluster,
               "population_density":population_density,"make":make,"segment":segment,
               "model":model_name,"fuel_type":fuel_type,"max_torque":max_torque,
               "max_power":max_power,"engine_type":engine_type,"airbags":airbags,
               "rear_brakes_type":rear_brakes_type,"displacement":displacement,
               "cylinder":cylinder,"transmission_type":transmission_type,
               "gear_box":gear_box,"steering_type":steering_type,
               "turning_radius":turning_radius,"width":width,"height":height,
               "gross_weight":gross_weight,"ncap_rating":ncap_rating,**bool_vals}
        input_df = pd.DataFrame([row])
        pred  = pipe.predict(input_df)[0]
        proba = pipe.predict_proba(input_df)[0][1]
        st.markdown("---")
        r, g = st.columns(2)
        if pred == 1:
            r.error(f"### ⚠️ CLAIM LIKELY\n**{proba*100:.1f}%** probability of filing a claim.")
        else:
            r.success(f"### ✅ NO CLAIM LIKELY\nOnly **{proba*100:.1f}%** probability of claim.")
        fig = go.Figure(go.Indicator(mode="gauge+number",value=round(proba*100,1),
            title={"text":"Claim Probability (%)"},
            gauge={"axis":{"range":[0,100]},"bar":{"color":"crimson" if pred else "seagreen"},
                   "steps":[{"range":[0,30],"color":"#d4edda"},{"range":[30,60],"color":"#fff3cd"},
                             {"range":[60,100],"color":"#f8d7da"}],
                   "threshold":{"line":{"color":"black","width":3},"thickness":0.75,"value":50}}))
        fig.update_layout(height=280,margin=dict(t=50,b=10))
        g.plotly_chart(fig, use_container_width=True)

# ── BATCH PREDICTION ────────────────────────────────────────────────────────
elif page == "📂 Batch Prediction":
    st.title("📂 Batch Prediction")
    st.markdown("Upload a CSV file matching `test.csv` schema (no `is_claim` needed).")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.write(f"**{df_up.shape[0]:,} rows × {df_up.shape[1]} columns**")
        st.dataframe(df_up.head())
        with st.spinner("Predicting..."):
            preds  = pipe.predict(df_up)
            probas = pipe.predict_proba(df_up)[:,1]
        id_col = "policy_id" if "policy_id" in df_up.columns else df_up.columns[0]
        results = pd.DataFrame({id_col:df_up[id_col],"is_claim":preds,"claim_probability":probas.round(4)})
        st.success(f"**{int(preds.sum()):,} claims** predicted out of {len(preds):,} ({preds.mean()*100:.1f}%).")
        st.dataframe(results.head(20))
        fig = px.histogram(results,x="claim_probability",nbins=50,title="Claim Probability Distribution")
        st.plotly_chart(fig, use_container_width=True)
        buf = io.StringIO(); results.to_csv(buf,index=False)
        st.download_button("⬇️ Download Predictions CSV",buf.getvalue(),"predictions.csv","text/csv")

# ── MODEL INSIGHTS ──────────────────────────────────────────────────────────
elif page == "📊 Model Insights":
    st.title("📊 Model Insights")
    st.subheader("Feature Importance (Top 20)")
    m = pipe.model
    if hasattr(m,"feature_importances_"):
        imp = pd.DataFrame({"feature":pipe.feature_columns,"importance":m.feature_importances_}
                           ).sort_values("importance",ascending=False).head(20)
        fig = px.bar(imp,x="importance",y="feature",orientation="h",
                     color="importance",color_continuous_scale="Blues",title="Top 20 Features")
        fig.update_layout(yaxis={"categoryorder":"total ascending"},height=540)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.subheader("Evaluate on Labelled Data")
    ef = st.file_uploader("Upload labelled CSV (with is_claim)", type="csv", key="eval")
    if ef:
        edf = pd.read_csv(ef)
        if TARGET not in edf.columns:
            st.error(f"File must contain '{TARGET}' column.")
        else:
            with st.spinner("Evaluating..."):
                metrics = pipe.evaluate(edf, plot=False)
            c1,c2,c3 = st.columns(3)
            c1.metric("Accuracy",f"{metrics['accuracy']:.4f}")
            c2.metric("F1 Score",f"{metrics['f1']:.4f}")
            c3.metric("ROC-AUC",f"{metrics['roc_auc']:.4f}")
            st.code(metrics["report"])
            cm = metrics["confusion_matrix"]
            fig_cm = px.imshow(cm,text_auto=True,color_continuous_scale="Blues",
                x=["No Claim","Claim"],y=["No Claim","Claim"],title="Confusion Matrix")
            st.plotly_chart(fig_cm)