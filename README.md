# Staten Island Railway OTP Forecasting & Explainable AI Dashboard

An end-to-end machine learning project for forecasting Staten Island Railway (SIR) On-Time Performance (OTP) using real MTA Open Data.

This project combines:

* Time-series forecasting
* Explainable AI (SHAP)
* Operational analytics
* Interactive Streamlit dashboards
* XGBoost machine learning models

The dashboard provides operational insights into railway reliability trends and identifies the key drivers influencing future OTP performance.

---

## Project Features

* Interactive Streamlit dashboard
* OTP trend analysis
* Actual vs Predicted OTP visualization
* Explainable AI using SHAP
* Dynamic filtering by operational periods
* Time-series feature engineering
* XGBoost forecasting model

---

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP
* Streamlit
* Plotly
* Matplotlib
* Seaborn

---

## Project Structure

```text
staten-island-otp-forecasting/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── raw/
│
├── models/
│   └── xgboost_otp_model.pkl
│
├── notebooks/
│   ├── 01_data_loading_and_initial_review.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_explainability.ipynb
│
├── outputs/
│   ├── figures/
│   ├── predictions/
│   └── reports/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Machine Learning Workflow

1. Data cleaning and preprocessing
2. Exploratory data analysis
3. Time-series feature engineering
4. Model training and evaluation
5. Explainable AI analysis using SHAP
6. Dashboard deployment with Streamlit

---

## Models Used

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor

XGBoost achieved the best forecasting performance for OTP prediction.

---

## Explainable AI

SHAP (SHapley Additive Explanations) was used to identify the most influential operational variables impacting OTP predictions.

Key drivers identified:

* Month/Seasonality
* Delay Rate
* Recent OTP Momentum
* Rolling Operational Trends

---

## Dashboard Preview

The dashboard includes:

* KPI monitoring
* OTP trend visualization
* Forecasting comparison
* Explainable AI insights

---

## Dataset

Source:
MTA Open Data

The dataset contains historical Staten Island Railway operational performance metrics including:

* Delayed trains
* On-time trips
* Scheduled trips
* OTP metrics
* Day time operational categories

---

## Future Improvements

* Multi-step future forecasting
* Real-time MTA API integration
* Operational risk scoring
* Automated executive insights
* Cloud deployment

---

## Author

Chinmay Naringrekar
