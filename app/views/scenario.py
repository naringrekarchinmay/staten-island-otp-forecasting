import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from shared.styles import page_header, breadcrumb, chart_title
from shared import data
from shared.charts import base_layout

breadcrumb("SCENARIO LAB", prev=("views/forecast.py", "04 AI Forecast"), nxt=("views/research.py", "06 Research"))
page_header("05", "Scenario Lab", "What If?",
            "Simulate a rise in delay conditions and model the impact on forecast OTP.")

if "scenario_delay" not in st.session_state:
    st.session_state.scenario_delay = 20

increase = st.slider("Simulated Delay Increase (%)", 0, 100, step=10, key="scenario_delay")

df7 = data.seven_day(data.load_features())
model = data.load_model()
months = 6
base = data.future_forecast(df7, model, months)
scen = data.scenario_forecast(df7, model, increase / 100, months)

base_avg = base["Forecasted_OTP"].mean()
scen_avg = scen["Forecasted_OTP"].mean()
delta_pts = (scen_avg - base_avg) * 100

if scen_avg >= 0.95:
    st.success(f"With **+{increase}%** delay conditions, average forecast OTP moves from "
               f"**{base_avg:.2%}** to **{scen_avg:.2%}** ({delta_pts:+.2f} pts) — within the acceptable range.")
elif scen_avg >= 0.90:
    st.warning(f"With **+{increase}%** delay conditions, average forecast OTP moves from "
               f"**{base_avg:.2%}** to **{scen_avg:.2%}** ({delta_pts:+.2f} pts) — approaching the 90% service threshold.")
else:
    st.error(f"With **+{increase}%** delay conditions, average forecast OTP moves from "
             f"**{base_avg:.2%}** to **{scen_avg:.2%}** ({delta_pts:+.2f} pts) — breaching the 90% service threshold.")

chart_title("Scenario Forecast vs. Baseline")
fig = go.Figure()
fig.add_trace(go.Scatter(x=base["Month"], y=base["Forecasted_OTP"] * 100, name="Baseline",
                         line=dict(color="#4a8ab0", width=2), mode="lines+markers"))
fig.add_trace(go.Scatter(x=scen["Month"], y=scen["Forecasted_OTP"] * 100,
                         name=f"+{increase}% Delay Scenario",
                         line=dict(color="#ef4444", width=2.5, dash="dash"), mode="lines+markers"))
base_layout(fig, height=270)
fig.update_yaxes(tickformat=".1f")
st.plotly_chart(fig, use_container_width=True)

chart_title("Scenario Risk Comparison")
table = pd.DataFrame({
    "Month": base["Month"].dt.strftime("%b %Y"),
    "Baseline OTP": base["Forecasted_OTP"].map(lambda v: f"{v:.2%}"),
    "Scenario OTP": scen["Forecasted_OTP"].map(lambda v: f"{v:.2%}"),
    "Risk Level": scen["Forecasted_OTP"].apply(data.risk_level),
})
st.dataframe(data.style_risk(table), use_container_width=True, hide_index=True)

st.markdown(
    "<p style=\"font-family:'Barlow',sans-serif;font-size:12px;color:rgba(255,255,255,0.35);\">"
    "The scenario scales the delay-related features (delay rate, delayed-train counts and their "
    "lags/rolling means) before re-running the recursive forecast — a model-driven what-if, not a "
    "fixed rule of thumb. Responses are modest because the model leans most heavily on OTP momentum "
    "and seasonality rather than delay levels alone (see SHAP, page 06).</p>", unsafe_allow_html=True)
