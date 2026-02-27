# ⚡ Energy Consumption Forecasting Dashboard

An **end-to-end machine learning dashboard** for forecasting hourly electricity demand using classical time-series models, tree ensembles, and deep learning architectures.

The project combines **feature engineering, explainable ML, anomaly detection, synthetic data generation, and conversational analytics** inside an interactive Streamlit dashboard.

---

## 📌 Problem Statement

Accurate electricity demand forecasting is essential for:

* Grid stability and load balancing
* Renewable energy integration
* Energy market optimization
* Infrastructure planning
* Cost and emission reduction

Electricity demand exhibits **multi-scale seasonality** (hourly, daily, weekly, yearly) and structural shifts, making forecasting a complex time-series problem.

---

## 📊 Dataset

* **Time range:** 2016–2021
* **Granularity:** Hourly consumption
* **Size:** ~50,000 observations
* **Average consumption:** ~1,350 units/hour
* **Notable pattern:** Structural demand drop during COVID-19 lockdowns (2020)

---

## 🚀 Dashboard Features

### 🔎 Data Explorer

* Interactive time-series visualization
* Trend and seasonality inspection
* Distribution and variance analysis

---

### 📈 Forecasting

Models implemented:

* Naive Baseline
* ARIMA
* Prophet
* Random Forest
* XGBoost
* LSTM

Users can generate forecasts and compare predictions interactively.

---

### 🧠 Model Comparison

* RMSE, MAE, MAPE, and R² comparison
* Residual analysis
* Error distribution visualization

---

### 🔬 Explainability

* SHAP feature importance
* Model interpretation
* Temporal dependency insights

---

### 🚨 Anomaly Detection

* Isolation Forest based anomaly detection
* Detection of sudden demand spikes and drops
* Visualization of abnormal consumption behavior

---

### 🧪 Synthetic Data Generator

* Bootstrapped time-series generation
* Scenario simulation capability
* Useful for stress testing models

---

### 💬 Conversational Chatbot

Natural language querying of:

* Forecast trends
* Model performance
* Dataset insights

---

## 🧩 Feature Engineering

To capture temporal dependencies and prevent leakage:

### Lag Features

* 24-hour lag
* 168-hour (weekly) lag

---

### Rolling Statistics

* Rolling mean
* Rolling standard deviation

---

### Cyclical Encoding

* Hour of day
* Day of week
* Month of year

---

### Leakage Prevention

* Lag features shifted forward
* Time-based split (no shuffling)
* Strict chronological evaluation

---

## 📐 Evaluation Strategy

* **Train period:** 2016–2020
* **Test period:** 2021
* **Time-based split** to preserve temporal structure
* No future information leakage
* Metrics used:

| Metric | Meaning                  |
| ------ | ------------------------ |
| RMSE   | Penalizes large errors   |
| MAE    | Average prediction error |
| MAPE   | Percentage error         |
| R²     | Explained variance       |

---

## 🏆 Model Performance Summary

Key observation:

**XGBoost consistently outperformed other models**, achieving:

* Lowest RMSE and MAE
* Best percentage accuracy
* Highest explained variance

This highlights the effectiveness of **boosting with engineered temporal features**.

---

## 🔎 Key Findings

* Lag features dominate feature importance
* Demand drop during COVID introduced structural break
* Tree ensembles outperform classical models
* LSTM underperformed due to limited data scale
* Electricity demand exhibits strong weekly periodicity
* Forecast uncertainty increases during anomalies

---

## 🏗 System Architecture

1. Raw time-series ingestion
2. Feature engineering pipeline
3. Model training layer
4. Evaluation and explainability module
5. Streamlit visualization dashboard

---

## 📦 Installation

```bash
git clone https://github.com/PrarthiPatel/Predicting-Energy-Consumption.git 
cd energy-forecasting-dashboard
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deployment

The dashboard is deployed on **Streamlit Cloud**.

⚠️ Recommendations:

* Use CPU-compatible models
* Prefer lightweight deep learning models
* Cache model loading for faster startup

---
# 🚀 Live Demo

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-ff4b4b?logo=streamlit&logoColor=white)](https://predicting-energy-consumption.streamlit.app/)

## 📈 Future Improvements

* Weather and exogenous variable integration
* Multivariate forecasting
* Transformer-based models
* Online model retraining
* Probabilistic forecasting
* Real-time data streaming

---

## ⚠️ Limitations

* Univariate dataset (no weather or economic variables)
* Structural break during pandemic affects generalization
* LSTM constrained by compute and dataset size
* No automated retraining pipeline

---

## 👨‍💻 Author

**Prarthi Patel**
Gandhinagar, Gujarat
February 2026

---

## ⭐ Project Highlights

✅ End-to-end ML pipeline
✅ Hybrid modeling approach
✅ Explainable forecasting
✅ Interactive analytics dashboard
✅ Real-world energy use case
✅ Research-grade feature engineering

---

