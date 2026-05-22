# Import required libraries for dashboard creation, data handling, plotting, model loading, and image display.
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.metrics import mean_absolute_error
from PIL import Image

# Configure the Streamlit page title and set the dashboard to wide layout.
st.set_page_config(
    page_title="SIR OTP Intelligence Dashboard",
    layout="wide"
)

# Main dashboard title and short project description.
st.title("Staten Island Railway OTP Intelligence Dashboard")
st.write(
    "An explainable machine learning dashboard for analyzing and forecasting "
    "On-Time Performance using MTA Open Data."
)

# Load the feature-engineered Staten Island Railway OTP dataset.
df = pd.read_csv("outputs/predictions/staten_island_otp_features.csv")

# Convert the Month column to datetime format so it can be used properly in time-series charts.
df["Month"] = pd.to_datetime(df["Month"])

# Load the trained XGBoost model that was saved from the model training notebook.
model = joblib.load("models/xgboost_otp_model.pkl")

# Function to generate recursive future OTP forecasts using the trained model.
def generate_future_forecast(df, model, forecast_months=6):
    # One-hot encode categorical columns so the data matches the format used during model training.
    df_encoded = pd.get_dummies(
        df.copy(),
        columns=["Day Time", "Season"],
        drop_first=True
    )

    # Get the exact feature columns expected by the trained XGBoost model.
    expected_cols = model.get_booster().feature_names

    # Add any missing encoded feature columns with a value of 0 to avoid prediction errors.
    for col in expected_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Reorder/select columns so the prediction dataset exactly matches the model's training features.
    X_all = df_encoded[expected_cols]

    # Use the most recent available row as the starting point for future forecasting.
    latest_row = X_all.iloc[-1:].copy()

    # Store future OTP predictions.
    future_predictions = []

    # Current row is updated recursively after each future prediction.
    current_row = latest_row.copy()

    # Generate predictions month by month for the selected forecast horizon.
    for i in range(forecast_months):
        # Predict the next month's OTP.
        pred = model.predict(current_row)[0]

        # Store the forecasted OTP value.
        future_predictions.append(pred)

        # Shift lag features forward so the predicted OTP becomes part of the next prediction cycle.
        current_row["OTP_Lag_3"] = current_row["OTP_Lag_2"]
        current_row["OTP_Lag_2"] = current_row["OTP_Lag_1"]
        current_row["OTP_Lag_1"] = pred

        # Recalculate the 3-month rolling OTP feature using the updated lag values.
        current_row["OTP_Rolling_3"] = (
            current_row[["OTP_Lag_1", "OTP_Lag_2", "OTP_Lag_3"]]
            .mean(axis=1)
        )

    # Identify the last month available in the historical dataset.
    last_date = df["Month"].max()

    # Create future monthly dates after the latest historical month.
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=forecast_months,
        freq="MS"
    )

    # Create a forecast dataframe with future dates and predicted OTP values.
    forecast_df = pd.DataFrame({
        "Month": future_dates,
        "Forecasted_OTP": future_predictions
    })

    # Return the future forecast dataframe for dashboard visualization.
    return forecast_df

def generate_scenario_forecast(df, model, delay_rate_increase=0.0, forecast_months=6):
    # Create a copy of the filtered data so the original dashboard data is not changed.
    scenario_df = df.copy()

    # Increase delay-related features based on the selected scenario.
    scenario_df["Delay_Rate"] = scenario_df["Delay_Rate"] * (1 + delay_rate_increase)

    if "Delayed Trains" in scenario_df.columns:
        scenario_df["Delayed Trains"] = scenario_df["Delayed Trains"] * (1 + delay_rate_increase)

    if "Delayed_Trains_Lag_1" in scenario_df.columns:
        scenario_df["Delayed_Trains_Lag_1"] = scenario_df["Delayed_Trains_Lag_1"] * (1 + delay_rate_increase)

    if "Delayed_Trains_Rolling_3" in scenario_df.columns:
        scenario_df["Delayed_Trains_Rolling_3"] = scenario_df["Delayed_Trains_Rolling_3"] * (1 + delay_rate_increase)

    # Reuse the same future forecasting logic after applying the scenario adjustment.
    scenario_forecast_df = generate_future_forecast(
        df=scenario_df,
        model=model,
        forecast_months=forecast_months
    )

    return scenario_forecast_df

def assign_risk_level(otp_value):
    # Classify forecasted OTP into operational risk levels.
    if otp_value >= 0.95:
        return "Low Risk"
    elif otp_value >= 0.90:
        return "Medium Risk"
    else:
        return "High Risk"


# Load the SHAP summary image generated in the explainability notebook.
shap_image = Image.open("outputs/figures/shap_summary.png")

# Sidebar section for dashboard filters.
st.sidebar.header("Dashboard Filters")

# Get available Day Time categories from the dataset.
day_time_options = df["Day Time"].unique()

# Multi-select filter that allows users to choose which Day Time categories to view.
selected_day_times = st.sidebar.multiselect(
    "Select Day Time",
    options=day_time_options,
    default=day_time_options
)

# Filter the dataframe based on the selected Day Time categories.
filtered_df = df[df["Day Time"].isin(selected_day_times)]

# One-hot encode the filtered data for dashboard model predictions.
df_encoded = pd.get_dummies(
    filtered_df.copy(),
    columns=["Day Time", "Season"],
    drop_first=True
)

# Get the exact feature columns expected by the trained model.
expected_cols = model.get_booster().feature_names

# Add missing feature columns if the user's selected filters remove certain encoded categories.
for col in expected_cols:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

# Select and order dashboard features exactly as expected by the model.
X_dashboard = df_encoded[expected_cols]

# Copy the filtered dataframe to safely add prediction results.
filtered_df = filtered_df.copy()

# Generate model predictions for the filtered dashboard data.
filtered_df["Predicted_OTP"] = model.predict(X_dashboard)

# Calculate KPI values based on the filtered dataset.
avg_otp = filtered_df["On-Time Performance"].mean()
avg_delay_rate = filtered_df["Delay_Rate"].mean()
best_otp = filtered_df["On-Time Performance"].max()
worst_otp = filtered_df["On-Time Performance"].min()

# Create four KPI columns at the top of the dashboard.
col1, col2, col3, col4 = st.columns(4)

# Display operational KPI cards.
col1.metric("Average OTP", f"{avg_otp:.2%}")
col2.metric("Average Delay Rate", f"{avg_delay_rate:.2%}")
col3.metric("Best OTP", f"{best_otp:.2%}")
col4.metric("Worst OTP", f"{worst_otp:.2%}")

# Section for historical OTP trend exploration.
st.subheader("OTP Trend Over Time")

# Create an interactive line chart showing OTP over time by Day Time category.
fig = px.line(
    filtered_df,
    x="Month",
    y="On-Time Performance",
    color="Day Time",
    title="On-Time Performance Trend by Day Time"
)

# Display the OTP trend chart in the dashboard.
st.plotly_chart(fig, use_container_width=True)

# Separate selector for the model evaluation chart so users can inspect one Day Time group at a time.
selected_prediction_group = st.selectbox(
    "Select Day Time for Prediction View",
    options=filtered_df["Day Time"].unique()
)

# Filter the prediction view to the selected Day Time group.
prediction_df = filtered_df[
    filtered_df["Day Time"] == selected_prediction_group
]

# Section for model evaluation using actual vs predicted OTP.
st.subheader("Actual vs Predicted OTP")

# Create a line chart comparing actual OTP against model-predicted OTP.
forecast_fig = px.line(
    prediction_df,
    x="Month",
    y=["On-Time Performance", "Predicted_OTP"],
    title=f"Actual vs Predicted OTP - {selected_prediction_group}"
)

# Display the actual vs predicted chart.
st.plotly_chart(forecast_fig, use_container_width=True)


# Add a visual divider before the future forecasting section.
st.divider()

# Section for future OTP forecasting.
st.subheader("Future OTP Forecast")

# Slider allowing the user to choose the forecast horizon between 3 and 12 months.
forecast_months = st.slider(
    "Select Forecast Horizon",
    min_value=3,
    max_value=12,
    value=6
)

# Generate future OTP forecasts using the recursive forecasting function.
forecast_df = generate_future_forecast(
    df=filtered_df,
    model=model,
    forecast_months=forecast_months
)

forecast_df["Risk_Level"] = forecast_df["Forecasted_OTP"].apply(assign_risk_level)

# Create an interactive line chart for future OTP forecasts.
future_fig = px.line(
    forecast_df,
    x="Month",
    y="Forecasted_OTP",
    markers=True,
    title=f"{forecast_months}-Month Future OTP Forecast"
)

# Format the y-axis as percentages.
future_fig.update_yaxes(
    tickformat=".1%",
    title="Forecasted OTP"
)

# Add a divider before displaying the forecast chart.
st.divider()

# Display the future forecast chart.
st.plotly_chart(
    future_fig,
    use_container_width=True
)

# Calculate summary KPIs for the forecasted OTP values.
forecast_avg = forecast_df["Forecasted_OTP"].mean()
forecast_min = forecast_df["Forecasted_OTP"].min()
forecast_max = forecast_df["Forecasted_OTP"].max()

# Create three KPI columns for forecast summary metrics.
col1, col2, col3 = st.columns(3)

# Display future forecast KPI cards.
col1.metric("Average Forecasted OTP", f"{forecast_avg:.2%}")
col2.metric("Lowest Forecasted OTP", f"{forecast_min:.2%}")
col3.metric("Highest Forecasted OTP", f"{forecast_max:.2%}")

st.subheader("Forecast Risk Breakdown")

risk_table = forecast_df.copy()
risk_table["Forecasted_OTP"] = risk_table["Forecasted_OTP"].map(lambda x: f"{x:.2%}")
risk_table["Month"] = risk_table["Month"].dt.strftime("%b %Y")

st.dataframe(
    risk_table,
    use_container_width=True,
    hide_index=True
)

low_risk_count = (forecast_df["Risk_Level"] == "Low Risk").sum()
medium_risk_count = (forecast_df["Risk_Level"] == "Medium Risk").sum()
high_risk_count = (forecast_df["Risk_Level"] == "High Risk").sum()

col1, col2, col3 = st.columns(3)

col1.metric("Low Risk Months", low_risk_count)
col2.metric("Medium Risk Months", medium_risk_count)
col3.metric("High Risk Months", high_risk_count)

# Display an operational outlook message based on the lowest forecasted OTP.
if high_risk_count > 0:
    st.error(
        "Operational Outlook: High-risk forecast period detected. Operational review is recommended."
    )
elif medium_risk_count > 0:
    st.warning(
        "Operational Outlook: Moderate reliability risk detected. Continued monitoring is recommended."
    )
else:
    st.success(
        "Operational Outlook: Forecasted OTP remains strong across the selected forecast horizon."
    )

st.divider()

st.subheader("Scenario Testing: Delay Impact Simulation")

st.write(
    """
    This section simulates how future OTP forecasts may change if delay-related 
    conditions increase compared to the current operational pattern.
    """
)

delay_rate_increase = st.slider(
    "Simulated Delay Increase",
    min_value=0,
    max_value=50,
    value=20,
    step=5,
    format="%d%%"
)

scenario_forecast_df = generate_scenario_forecast(
    df=filtered_df,
    model=model,
    delay_rate_increase=delay_rate_increase / 100,
    forecast_months=forecast_months
)

scenario_forecast_df["Risk_Level"] = scenario_forecast_df["Forecasted_OTP"].apply(assign_risk_level)

scenario_fig = px.line(
    scenario_forecast_df,
    x="Month",
    y="Forecasted_OTP",
    markers=True,
    title=f"Scenario Forecast with {delay_rate_increase}% Delay Increase"
)

scenario_fig.update_yaxes(
    tickformat=".1%",
    title="Scenario Forecasted OTP"
)

st.plotly_chart(
    scenario_fig,
    use_container_width=True
)

scenario_risk_table = scenario_forecast_df.copy()
scenario_risk_table["Forecasted_OTP"] = scenario_risk_table["Forecasted_OTP"].map(lambda x: f"{x:.2%}")
scenario_risk_table["Month"] = scenario_risk_table["Month"].dt.strftime("%b %Y")

st.subheader("Scenario Risk Breakdown")

st.dataframe(
    scenario_risk_table,
    use_container_width=True,
    hide_index=True
)

scenario_low_risk_count = (scenario_forecast_df["Risk_Level"] == "Low Risk").sum()
scenario_medium_risk_count = (scenario_forecast_df["Risk_Level"] == "Medium Risk").sum()
scenario_high_risk_count = (scenario_forecast_df["Risk_Level"] == "High Risk").sum()

col1, col2, col3 = st.columns(3)

col1.metric("Scenario Low Risk Months", scenario_low_risk_count)
col2.metric("Scenario Medium Risk Months", scenario_medium_risk_count)
col3.metric("Scenario High Risk Months", scenario_high_risk_count)

if scenario_high_risk_count > 0:
    st.error(
        "Scenario Outlook: High-risk periods appear under this simulated delay increase."
    )
elif scenario_medium_risk_count > 0:
    st.warning(
        "Scenario Outlook: Moderate reliability risk appears under this simulated delay increase."
    )
else:
    st.success(
        "Scenario Outlook: OTP remains low risk even under this simulated delay increase."
    )

# Section for model explainability and operational driver interpretation.
st.subheader("AI Explainability & Operational Drivers")

# Explain what SHAP is doing in the context of this dashboard.
st.write(
    """
    SHAP (SHapley Additive Explanations) identifies which operational variables
    most influence OTP predictions generated by the XGBoost forecasting model.
    """
)

# Display the SHAP summary plot generated from the explainability notebook.
st.image(
    shap_image,
    caption="Global Feature Importance using SHAP",
    width=600
)

# Add an executive-friendly interpretation of the SHAP results.
st.info(
    """
    Key operational drivers identified by the XGBoost forecasting model include:
    seasonality, delay rate, and recent OTP momentum.
    """
)
