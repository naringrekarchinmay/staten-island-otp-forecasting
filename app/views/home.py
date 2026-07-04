import streamlit as st
from shared.styles import GOLD, NAVY, breadcrumb
from shared import data

df7 = data.seven_day(data.load_features())
avg_otp = df7["On-Time Performance"].mean()
avg_delay = df7["Delay_Rate"].mean()

breadcrumb("HOME", nxt=("views/health.py", "02 System Health"))

st.markdown(f"""
<div style="position:relative;width:100%;height:400px;overflow:hidden;border-radius:12px;margin-bottom:24px;">
  <img src="app/static/sir_train.png" style="width:100%;height:100%;object-fit:cover;object-position:center 55%;">
  <div style="position:absolute;inset:0;background:linear-gradient(108deg,#0c1824 0%,rgba(12,24,36,0.86) 42%,rgba(12,24,36,0.38) 100%);"></div>
  <div style="position:absolute;bottom:0;left:0;right:0;padding:0 40px 36px;">
    <p style="font-family:'DM Mono',monospace;font-size:10px;color:{GOLD};letter-spacing:0.16em;text-transform:uppercase;margin-bottom:10px;">Staten Island Railway · Operational Intelligence</p>
    <h1 style="font-family:'Barlow Condensed',sans-serif;font-weight:800;font-size:80px;line-height:0.86;color:#fff;text-transform:uppercase;margin:0;">ON-TIME<br><span style="color:{GOLD};">PERFORMANCE</span></h1>
    <p style="font-family:'Barlow',sans-serif;font-size:14px;color:rgba(255,255,255,0.6);margin:14px 0 0;max-width:520px;">AI-powered forecasting, twenty years of trends, and scenario simulation for the SIR system.</p>
  </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1, st.container(key="gold_home"):
    st.metric("Avg OTP (7-Day)", f"{avg_otp:.2%}")
c2.metric("Avg Delay Rate", f"{avg_delay:.2%}")
c3.metric("Fair-Test MAE", "1.21 pts", "-18.7% error vs best baseline", delta_color="inverse")
c4.metric("80% Interval Coverage", "79.5%")

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

chapters = [
    ("views/health.py", "02", "System Health", "Current OTP score, delay burden, and reliability status."),
    ("views/trends.py", "03", "OTP Trends", "Twenty years of history, actual vs. predicted, residuals."),
    ("views/forecast.py", "04", "AI Forecast", "Recursive OTP forecast with calibrated intervals."),
    ("views/scenario.py", "05", "Scenario Lab", "Simulate delay increases and model the OTP impact."),
    ("views/research.py", "06", "Research", "Phase 13–15 validation and SHAP explainability."),
]
cols = st.columns(5)
for col, (page, num, title, desc) in zip(cols, chapters):
    with col:
        st.markdown(f"""
<div style="background:#fff;border:1px solid rgba(27,45,82,0.1);border-radius:8px;padding:18px 16px;min-height:150px;">
  <div style="font-family:'DM Mono',monospace;font-size:10px;color:{GOLD};margin-bottom:8px;">{num}</div>
  <div style="font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:18px;color:{NAVY};text-transform:uppercase;margin-bottom:6px;line-height:1.05;">{title}</div>
  <div style="font-family:'Barlow',sans-serif;font-size:12px;color:rgba(27,45,82,0.55);line-height:1.45;">{desc}</div>
</div>
""", unsafe_allow_html=True)
        st.page_link(page, label=f"Open {num} →")
