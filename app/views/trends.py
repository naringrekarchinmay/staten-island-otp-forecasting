import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from shared.styles import page_header, breadcrumb, chart_title
from shared import data
from shared.charts import base_layout

breadcrumb("OTP TRENDS", prev=("views/health.py", "02 System Health"), nxt=("views/forecast.py", "04 AI Forecast"))

tcol, rcol = st.columns([2.7, 2.3])
with tcol:
    page_header("03", "OTP Trends", "The Story So Far")
with rcol:
    st.markdown("<div style='height:34px;'></div>", unsafe_allow_html=True)
    rng = st.radio("Range", ["All Time", "10 Yr", "5 Yr", "2 Yr"], horizontal=True,
                   label_visibility="collapsed")

df = data.load_clean()
end = df["Month"].max()
years = {"All Time": None, "10 Yr": 10, "5 Yr": 5, "2 Yr": 2}[rng]
if years:
    df = df[df["Month"] >= end - pd.DateOffset(years=years)]

chart_title("OTP by Service Category")
series = [("7-Day", "#e9b820", None, 2.5), ("Weekday", "#4a8ab0", "dot", 1.5), ("Weekend", "#7b6caa", "dot", 1.5)]
fig = go.Figure()
for cat, color, dash, w in series:
    d = df[df["Day Time"] == cat]
    kw = dict(fill="tozeroy", fillcolor="rgba(233,184,32,0.07)") if cat == "7-Day" else {}
    fig.add_trace(go.Scatter(x=d["Month"], y=d["On-Time Performance"] * 100, name=cat,
                             line=dict(color=color, width=w, dash=dash), **kw))
base_layout(fig, height=260, yrange=[80, 101])
st.plotly_chart(fig, use_container_width=True)

cv = data.load_cv_predictions()

c1, c2 = st.columns(2)
with c1:
    chart_title("Actual vs. Predicted OTP (out-of-sample CV)")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=cv["month"], y=cv["actual_otp"], name="Actual",
                              line=dict(color="#4a8ab0", width=2)))
    fig2.add_trace(go.Scatter(x=cv["month"], y=cv["predicted_otp"], name="Predicted",
                              line=dict(color="#e9b820", width=1.5, dash="dot")))
    base_layout(fig2, height=230)
    st.plotly_chart(fig2, use_container_width=True)
with c2:
    chart_title("Prediction Residuals Distribution")
    fig3 = go.Figure(go.Histogram(x=cv["residual"], marker_color="#4a8ab0", opacity=0.75, nbinsx=30))
    fig3.add_vline(x=0, line_color="rgba(233,184,32,0.5)", line_dash="dot")
    base_layout(fig3, height=230, pct_axis=False, legend=False)
    fig3.update_xaxes(ticksuffix=" pts")
    st.plotly_chart(fig3, use_container_width=True)

st.markdown(
    "<p style=\"font-family:'Barlow',sans-serif;font-size:12px;color:rgba(255,255,255,0.35);\">"
    "Predictions are out-of-sample from Phase 14 time-series cross-validation: the model never saw "
    "the months it is predicting. Residuals in OTP percentage points.</p>", unsafe_allow_html=True)
