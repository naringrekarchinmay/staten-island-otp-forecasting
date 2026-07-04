"""Cached data loaders and forecasting logic (ported from the single-page app)."""
from pathlib import Path
import pandas as pd
import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

# Phase 15 residual bounds (percentage points), from outputs/reports/phase15_prediction_interval_summary.csv
PI80 = (-3.1265, 3.3576)
PI90 = (-4.4797, 4.1961)

@st.cache_data
def load_features():
    df = pd.read_csv(ROOT / "outputs/predictions/staten_island_otp_features.csv")
    df["Month"] = pd.to_datetime(df["Month"])
    return df

@st.cache_data
def load_clean():
    df = pd.read_csv(ROOT / "data/raw/cleaned_staten_island_otp.csv")
    df["Month"] = pd.to_datetime(df["Month"])
    return df

@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models/xgboost_otp_model.pkl")

@st.cache_data
def load_report(name):
    return pd.read_csv(ROOT / "outputs/reports" / name)

@st.cache_data
def load_cv_predictions():
    df = pd.read_csv(ROOT / "outputs/predictions/phase15_prediction_intervals.csv")
    df["month"] = pd.to_datetime(df["month"])
    return df

def seven_day(df):
    return df[df["Day Time"] == "7-Day"].sort_values("Month")

def _encode(df, model):
    enc = pd.get_dummies(df.copy(), columns=["Day Time", "Season"], drop_first=True)
    cols = model.get_booster().feature_names
    for c in cols:
        if c not in enc.columns:
            enc[c] = 0
    return enc[cols]

def predict(df, model):
    return model.predict(_encode(df, model))

def future_forecast(df, model, months=6):
    """Recursive multi-month forecast from the latest available row."""
    X = _encode(df, model)
    row = X.iloc[-1:].copy()
    preds = []
    for _ in range(months):
        p = float(model.predict(row)[0])
        preds.append(p)
        row["OTP_Lag_3"] = row["OTP_Lag_2"]
        row["OTP_Lag_2"] = row["OTP_Lag_1"]
        row["OTP_Lag_1"] = p
        row["OTP_Rolling_3"] = row[["OTP_Lag_1", "OTP_Lag_2", "OTP_Lag_3"]].mean(axis=1)
    dates = pd.date_range(df["Month"].max() + pd.DateOffset(months=1), periods=months, freq="MS")
    return pd.DataFrame({"Month": dates, "Forecasted_OTP": preds})

def scenario_forecast(df, model, delay_increase=0.0, months=6):
    s = df.copy()
    for col in ["Delay_Rate", "Delayed Trains", "Delayed_Trains_Lag_1", "Delayed_Trains_Rolling_3"]:
        if col in s.columns:
            s[col] = s[col] * (1 + delay_increase)
    return future_forecast(s, model, months)

def risk_level(otp01):
    if otp01 >= 0.95:
        return "Low"
    if otp01 >= 0.90:
        return "Medium"
    return "High"

RISK_COLOR = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}

def style_risk(df, col="Risk Level"):
    return df.style.map(lambda v: f"color: {RISK_COLOR.get(v, '#fff')}; font-weight: 600", subset=[col])
