# Car Insurance Claim Prediction

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://car-insurance-claim-prediction-vbsunpjsqzaeaxzs8bgjnt.streamlit.app/)
[![MLflow](https://img.shields.io/badge/MLflow-DagsHub_Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://dagshub.com/malaychand/car-insurance-claim-prediction.mlflow/)
[![DagsHub](https://img.shields.io/badge/DagsHub-Repository-orange?style=for-the-badge&logo=git&logoColor=white)](https://dagshub.com/malaychand/car-insurance-claim-prediction)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/malaychand/car-insurance-claim-prediction)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

---

## 📌 Project Overview

Car insurance companies face significant financial risk due to fraudulent or unexpected claims. This project builds a **machine learning–based predictive system** to determine whether a customer is likely to file an insurance claim in the next policy period, using demographic, vehicle, and policy-related features.

The solution is **end-to-end**, covering EDA, preprocessing, modeling, evaluation, and deployment via Streamlit.

---

## 🔗 Project Links & Live Demo

| Resource | Link |
|---|---|
| 🚀 Live Streamlit App | [car-insurance-claim-prediction.streamlit.app](https://car-insurance-claim-prediction-vbsunpjsqzaeaxzs8bgjnt.streamlit.app/) |
| 📊 MLflow Experiment Tracking | [DagsHub MLflow](https://dagshub.com/malaychand/car-insurance-claim-prediction.mlflow/) |
| 📁 DagsHub Repository | [dagshub.com/malaychand](https://dagshub.com/malaychand/car-insurance-claim-prediction) |
| 💻 GitHub Repository | [github.com/malaychand](https://github.com/malaychand/car-insurance-claim-prediction) |

---

## 🎯 Problem Statement

The objective is to predict whether a customer will make a car insurance claim (`is_claim`) using historical insurance data.

This is a **binary classification** problem where:

- `1` → Claim will occur
- `0` → No claim

---

## 🏢 Business Use Cases

| Use Case | Description |
|---|---|
| 🛡️ Fraud Prevention | Identify high-risk customers early and tailor policies to reduce losses |
| 💰 Pricing Optimization | Adjust premiums based on predicted likelihood of claims |
| 🎯 Customer Targeting | Focus marketing campaigns on low-risk, profitable customers |
| ⚙️ Operational Efficiency | Forecast claim volumes for better resource allocation |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.9+ |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost, LightGBM, CatBoost |
| Hyperparameter Tuning | Optuna, GridSearchCV, RandomizedSearchCV |
| Experiment Tracking | MLflow (via DagsHub) |
| Imbalanced Data | imbalanced-learn (SMOTE) |
| Visualization | Matplotlib, Seaborn, Plotly |
| Deployment | Streamlit |
| Version Control | Git, GitHub, DagsHub |


---

## 📊 Dataset

**Source:** [Google Drive Dataset](https://drive.google.com/file/d/1RP5vqMcI9SIFW3LsdacdHoTtrAgylC8l/view?usp=sharing)

**Target Column:** `is_claim` (Binary: 0 or 1)

**Size:** ~58,000+ rows × 44 columns

The dataset contains policyholder demographics, vehicle specifications, and safety feature flags. Key features include:

| Feature | Description |
|---|---|
| `age_of_car` | Normalized age of the car in years |
| `age_of_policyholder` | Normalized age of the policyholder |
| `segment` | Car segment (A / B1 / B2 / C1 / C2) |
| `fuel_type` | Type of fuel used |
| `max_power` | Max power (bhp@rpm) |
| `displacement` | Engine displacement (cc) |
| `ncap_rating` | NCAP safety rating (out of 5) |
| `airbags` | Number of airbags |
| `is_esc`, `is_tpms`, etc. | Boolean safety feature flags |
| `is_claim` | **Target**: Did the policyholder file a claim? |

> See the full data dictionary in `notebooks/01_eda.ipynb`.

---

## 🔄 Approach & Methodology

```
Raw Data
   ↓
EDA & Visualization
   ↓
Preprocessing (Null handling, Encoding, Scaling, SMOTE)
   ↓
Baseline Models (Logistic Regression, Decision Tree)
   ↓
Advanced Models (Random Forest, XGBoost, LightGBM, CatBoost)
   ↓
Hyperparameter Tuning (Optuna / GridSearchCV)
   ↓
Model Evaluation (AUC-ROC, F1, Confusion Matrix)
   ↓
Best Model Saved → Streamlit Deployment
```

**Handling Class Imbalance:** The dataset is significantly imbalanced (~93% no-claim vs ~7% claim). SMOTE was applied on the training set only to avoid data leakage.

---

## 📈 Model Results

| Model | Accuracy | F1-Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | ~baseline | — | — |
| Decision Tree | ~baseline | — | — |
| Random Forest | ✅ | ✅ | ✅ |
| XGBoost | ✅ | ✅ | ✅ |
| LightGBM | ✅✅ | ✅✅ | ✅✅ |
| CatBoost | ✅✅ | ✅✅ | ✅✅ |
| **LightGBM + Optuna** | 🏆 **Best** | 🏆 **Best** | 🏆 **Best** |

> Full metrics logged to [MLflow on DagsHub](https://dagshub.com/malaychand/car-insurance-claim-prediction.mlflow/).

---

## 🚀 Running Locally

### 1. Clone the Repository

```bash
git clone https://github.com/malaychand/car-insurance-claim-prediction.git
cd car-insurance-claim-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the Dataset

Download `train.csv` from the [dataset link](https://drive.google.com/file/d/1RP5vqMcI9SIFW3LsdacdHoTtrAgylC8l/view?usp=sharing) and place it in the `data/` folder.

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🧪 Evaluation Metrics

The following metrics were used to evaluate model performance:

- **Accuracy** — Overall correct predictions
- **Precision & Recall** — Critical on imbalanced data; recall prioritized for catching claims
- **F1-Score** — Harmonic mean of Precision and Recall
- **ROC-AUC** — Ability to rank high-risk customers above low-risk ones
- **Confusion Matrix** — Breakdown of TP, TN, FP, FN

---

## 🔑 Key Findings

- **Vehicle age** and **policyholder age** are among the strongest predictors of claims.
- Cars with **fewer safety features** (airbags, ESC, TPMS) show higher claim rates.
- **Segment B1/B2** vehicles exhibit higher claim frequency compared to premium segments.
- **Diesel vehicles** showed marginally different claim rates compared to petrol.
- The dataset is heavily imbalanced; SMOTE significantly improved recall for the minority class.

---


## 👤 Author

**Malay Chand**

[![GitHub](https://img.shields.io/badge/GitHub-malaychand-181717?style=flat&logo=github)](https://github.com/malaychand)
[![DagsHub](https://img.shields.io/badge/DagsHub-malaychand-orange?style=flat)](https://dagshub.com/malaychand)
