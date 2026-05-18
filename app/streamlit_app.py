import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.metrics import mean_absolute_error
from PIL import Image

st.set_page_config(
    page_title="SIR OTP Intelligence Dashboard",
    layout="wide"
)

st.title("Staten Island Railway OTP Intelligence Dashboard")
st.write(
    "An explainable machine learning dashboard for analyzing and forecasting "
    "On-Time Performance using MTA Open Data."
)

df = pd.read_csv("outputs/predictions/staten_island_otp_features.csv")
df["Month"] = pd.to_datetime(df["Month"])
model = joblib.load("models/xgboost_otp_model.pkl")

def generate_future_forecast(df, model, forecast_months=6):
    df_encoded = pd.get_dummies(
        df.copy(),
        columns=["Day Time", "Season"],
        drop_first=True
    )

    expected_cols = model.get_booster().feature_names

    for col in expected_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    X_all = df_encoded[expected_cols]

    latest_row = X_all.iloc[-1:].copy()

    future_predictions = []
    current_row = latest_row.copy()

    for i in range(forecast_months):
        pred = model.predict(current_row)[0]
        future_predictions.append(pred)

        current_row["OTP_Lag_3"] = current_row["OTP_Lag_2"]
        current_row["OTP_Lag_2"] = current_row["OTP_Lag_1"]
        current_row["OTP_Lag_1"] = pred

        current_row["OTP_Rolling_3"] = (
            current_row[["OTP_Lag_1", "OTP_Lag_2", "OTP_Lag_3"]]
            .mean(axis=1)
        )

    last_date = df["Month"].max()

    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=forecast_months,
        freq="MS"
    )

    forecast_df = pd.DataFrame({
        "Month": future_dates,
        "Forecasted_OTP": future_predictions
    })

    return forecast_df

shap_image = Image.open("outputs/figures/shap_summary.png")

st.sidebar.header("Dashboard Filters")

day_time_options = df["Day Time"].unique()

selected_day_times = st.sidebar.multiselect(
    "Select Day Time",
    options=day_time_options,
    default=day_time_options
)

filtered_df = df[df["Day Time"].isin(selected_day_times)]
df_encoded = pd.get_dummies(
    filtered_df.copy(),
    columns=["Day Time", "Season"],
    drop_first=True
)

expected_cols = model.get_booster().feature_names

for col in expected_cols:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

X_dashboard = df_encoded[expected_cols]

filtered_df = filtered_df.copy()
filtered_df["Predicted_OTP"] = model.predict(X_dashboard)

avg_otp = filtered_df["On-Time Performance"].mean()
avg_delay_rate = filtered_df["Delay_Rate"].mean()
best_otp = filtered_df["On-Time Performance"].max()
worst_otp = filtered_df["On-Time Performance"].min()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Average OTP", f"{avg_otp:.2%}")
col2.metric("Average Delay Rate", f"{avg_delay_rate:.2%}")
col3.metric("Best OTP", f"{best_otp:.2%}")
col4.metric("Worst OTP", f"{worst_otp:.2%}")

st.subheader("OTP Trend Over Time")

fig = px.line(
    filtered_df,
    x="Month",
    y="On-Time Performance",
    color="Day Time",
    title="On-Time Performance Trend by Day Time"
)

st.plotly_chart(fig, use_container_width=True)

selected_prediction_group = st.selectbox(
    "Select Day Time for Prediction View",
    options=filtered_df["Day Time"].unique()
)

prediction_df = filtered_df[
    filtered_df["Day Time"] == selected_prediction_group
]
st.subheader("Actual vs Predicted OTP")

forecast_fig = px.line(
    prediction_df,
    x="Month",
    y=["On-Time Performance", "Predicted_OTP"],
    title=f"Actual vs Predicted OTP - {selected_prediction_group}"
)

st.plotly_chart(forecast_fig, use_container_width=True)



st.divider()

st.subheader("Future OTP Forecast")

forecast_months = st.slider(
    "Select Forecast Horizon",
    min_value=3,
    max_value=12,
    value=6
)

forecast_df = generate_future_forecast(
    df=filtered_df,
    model=model,
    forecast_months=forecast_months
)

future_fig = px.line(
    forecast_df,
    x="Month",
    y="Forecasted_OTP",
    markers=True,
    title=f"{forecast_months}-Month Future OTP Forecast"
)

future_fig.update_yaxes(
    tickformat=".1%",
    title="Forecasted OTP"
)

st.divider()
st.plotly_chart(
    future_fig,
    use_container_width=True
)

forecast_avg = forecast_df["Forecasted_OTP"].mean()
forecast_min = forecast_df["Forecasted_OTP"].min()
forecast_max = forecast_df["Forecasted_OTP"].max()

col1, col2, col3 = st.columns(3)

col1.metric("Average Forecasted OTP", f"{forecast_avg:.2%}")
col2.metric("Lowest Forecasted OTP", f"{forecast_min:.2%}")
col3.metric("Highest Forecasted OTP", f"{forecast_max:.2%}")

if forecast_min >= 0.95:
    st.success(
        "Operational Outlook: Forecasted OTP remains strong across the selected forecast horizon."
    )
elif forecast_min >= 0.90:
    st.warning(
        "Operational Outlook: Forecasted OTP shows moderate reliability risk. Monitoring is recommended."
    )
else:
    st.error(
        "Operational Outlook: Forecasted OTP indicates high reliability risk. Operational review is recommended."
    )
st.subheader("AI Explainability & Operational Drivers")

st.write(
    """
    SHAP (SHapley Additive Explanations) identifies which operational variables
    most influence OTP predictions generated by the XGBoost forecasting model.
    """
)

st.image(
    shap_image,
    caption="Global Feature Importance using SHAP",
    width=600
)
st.info(
    """
    Key operational drivers identified by the XGBoost forecasting model include:
    seasonality, delay rate, and recent OTP momentum.
    """
)