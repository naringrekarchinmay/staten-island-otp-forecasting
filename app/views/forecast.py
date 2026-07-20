import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from shared.styles import page_header, breadcrumb, chart_title
from shared import data
from shared.charts import base_layout

breadcrumb("AI FORECAST", prev=("views/trends.py", "03 OTP Trends"), nxt=("views/scenario.py", "05 Scenario Lab"))

tcol, scol = st.columns([3, 2])
with tcol:
    page_header("04", "AI Forecast", "What Comes Next")
with scol:
    st.markdown("<div style='height:26px;'></div>", unsafe_allow_html=True)
    horizon = st.slider("Forecast Horizon (months)", 3, 12, 6)

df7 = data.seven_day(data.load_features())
model = data.load_model()
fc = data.future_forecast(df7, model, horizon)
fc["pct"] = fc["Forecasted_OTP"] * 100
fc["lower80"], fc["upper80"] = data.apply_pi(fc["pct"], data.PI80)
fc["Risk Level"] = fc["Forecasted_OTP"].apply(data.risk_level)

hist = df7.tail(36)
now = df7["Month"].max()

chart_title(f"{horizon}-Month OTP Forecast with 80% Prediction Interval")
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist["Month"], y=hist["On-Time Performance"] * 100,
                         name="Historical", line=dict(color="#4a8ab0", width=2)))
fig.add_trace(go.Scatter(x=fc["Month"], y=fc["upper80"], mode="lines", line=dict(width=0),
                         showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=fc["Month"], y=fc["lower80"], mode="lines", fill="tonexty",
                         fillcolor="rgba(233,184,32,0.12)", line=dict(width=0), name="80% interval"))
fig.add_trace(go.Scatter(x=fc["Month"], y=fc["pct"], name="Forecast",
                         line=dict(color="#e9b820", width=2.5, dash="dash"), mode="lines+markers"))
fig.add_shape(type="line", x0=now, x1=now, y0=0, y1=1, yref="paper",
              line=dict(color="rgba(255,255,255,0.15)", dash="dot"))
fig.add_annotation(x=now, y=1.04, yref="paper", text="NOW", showarrow=False,
                   font=dict(family="DM Mono, monospace", size=9, color="rgba(255,255,255,0.4)"))
base_layout(fig, height=280, yrange=[85, 101])
st.plotly_chart(fig, use_container_width=True)

c1, c2, c3 = st.columns(3)
with c1, st.container(key="gold_fc"):
    st.metric("Avg Forecast OTP", f"{fc['Forecasted_OTP'].mean():.2%}")
c2.metric("Lowest Month Forecast", f"{fc['Forecasted_OTP'].min():.2%}",
          f"{fc.loc[fc['Forecasted_OTP'].idxmin(), 'Month']:%b %Y}", delta_color="off")
c3.metric("Highest Month Forecast", f"{fc['Forecasted_OTP'].max():.2%}",
          f"{fc.loc[fc['Forecasted_OTP'].idxmax(), 'Month']:%b %Y}", delta_color="off")

st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
chart_title("Forecast Risk Breakdown")

table = pd.DataFrame({
    "Month": fc["Month"].dt.strftime("%b %Y"),
    "Forecast OTP": fc["pct"].map(lambda v: f"{v:.2f}%"),
    "80% Interval": [f"{lo:.1f} – {hi:.1f}%" for lo, hi in zip(fc["lower80"], fc["upper80"])],
    "Risk Level": fc["Risk Level"],
})
st.dataframe(data.style_risk(table), use_container_width=True, hide_index=True)

st.markdown(
    "<p style=\"font-family:'Barlow',sans-serif;font-size:12px;color:rgba(255,255,255,0.35);\">"
    "Recursive forecast: each prediction feeds back as a lag feature. The 80% band uses Phase 15 "
    "residual bounds (−3.1 / +3.4 pts, observed coverage 79.5%); interval calibration applies to "
    "the one-step forecast and widens in practice at longer horizons.</p>", unsafe_allow_html=True)
