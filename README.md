# Staten Island Railway OTP Forecasting & Explainable AI Dashboard

An end-to-end machine learning project for analyzing, forecasting, and explaining Staten Island Railway (SIR) On-Time Performance (OTP) using real MTA Open Data.

This project combines machine learning, operational analytics, explainable AI, future forecasting, and interactive dashboard development to better understand railway reliability trends and the factors influencing OTP performance.

---

## Project Overview

The goal of this project is to move beyond traditional historical reporting and build an applied machine learning system that can:

- Analyze historical Staten Island Railway OTP trends
- Predict OTP using machine learning models
- Compare actual vs predicted performance
- Explain model behavior using SHAP
- Forecast future OTP trends
- Classify future forecast periods into operational risk levels
- Present results through an interactive Streamlit dashboard

This project is still a work in progress and is being developed as an applied AI and transportation analytics project.

---

## Key Features

- Interactive Streamlit dashboard
- Historical OTP trend analysis
- Dynamic filtering by operational period
- Actual vs Predicted OTP visualization
- XGBoost-based OTP forecasting model
- Recursive multi-month future OTP forecasting
- Operational risk scoring for forecasted months
- SHAP explainability for model interpretation
- Executive-style KPI cards
- Forecast outlook messaging

---

## Dashboard Capabilities

The dashboard currently includes:

### 1. Executive Overview

- Average OTP
- Average Delay Rate
- Best OTP
- Worst OTP

### 2. Historical OTP Trend Analysis

- Interactive line chart showing OTP over time
- Sidebar filter for different operational periods such as Day Time categories

### 3. Actual vs Predicted OTP

- Compares real OTP values against model-predicted OTP
- Allows users to select a specific Day Time category for clearer model evaluation

### 4. Future OTP Forecasting

- Recursive future forecasting using the trained XGBoost model
- User-controlled forecast horizon from 3 to 12 months
- Forecasted OTP trend chart
- Forecast KPI cards:
  - Average Forecasted OTP
  - Lowest Forecasted OTP
  - Highest Forecasted OTP

### 5. Operational Risk Scoring

Forecasted months are classified into operational risk levels:

| Forecasted OTP | Risk Level |
|---|---|
| >= 95% | Low Risk |
| 90% - 95% | Medium Risk |
| < 90% | High Risk |

The dashboard also provides a simple operational outlook message based on the forecasted risk levels.

### 6. Explainable AI

SHAP (SHapley Additive Explanations) is used to identify which features most influence the model’s OTP predictions.

Key operational drivers identified include:

- Seasonality
- Delay Rate
- Recent OTP Momentum
- Rolling Operational Trends

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Plotly
- Matplotlib
- Seaborn
- Joblib
- OpenPyXL

---

## Machine Learning Workflow

The project follows a structured machine learning workflow:

1. Data loading and initial review
2. Data cleaning and preprocessing
3. Exploratory data analysis
4. Time-series feature engineering
5. Model training and evaluation
6. Explainable AI analysis using SHAP
7. Future OTP forecasting
8. Streamlit dashboard development
9. Operational risk scoring

---

## Models Used

The following regression models were tested:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

XGBoost performed the best among the tested models and was selected as the final model for dashboard integration.

---

## Why These Models Were Used

### Linear Regression

Used as a baseline model to understand whether simple linear relationships exist between operational features and future OTP.

### Random Forest Regressor

Used to capture non-linear relationships and operational threshold effects that a simple linear model may miss.

### XGBoost Regressor

Used as the final model because it performs well on structured/tabular data and can capture complex interactions between delay metrics, lag features, rolling averages, and seasonal patterns.

---

## Feature Engineering

The model uses several time-series and operational features, including:

- OTP lag features
- Delayed train lag features
- Rolling OTP averages
- Rolling delayed train averages
- Delay rate
- Year
- Month number
- Quarter
- Season
- Day Time category

These features help the model understand historical operational momentum and seasonal reliability patterns.

---

## Future Forecasting

The project includes a recursive multi-month forecasting engine.

The forecasting approach works by:

1. Taking the latest available operational state
2. Predicting the next month’s OTP
3. Feeding that prediction back into the model as a lag feature
4. Repeating the process for the selected forecast horizon

This allows the dashboard to generate future OTP projections for 3 to 12 months.

---

## Explainable AI

SHAP was used to interpret the XGBoost model and understand which features most influenced OTP predictions.

This adds transparency to the machine learning process and helps move the project beyond black-box forecasting.

Instead of only asking:

> What did the model predict?

The project also asks:

> Why did the model make that prediction?

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
│   ├── 05_explainability.ipynb
│   └── 06_future_forecasting.ipynb
│
├── outputs/
│   ├── figures/
│   ├── forecasts/
│   ├── predictions/
│   └── reports/
│
├── README.md
├── requirements.txt
└── .gitignore