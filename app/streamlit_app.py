"""SIR OTP Intelligence Platform — multi-page entry point (Dispatch design)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from shared.styles import inject_styles, GOLD

st.set_page_config(
    page_title="SIR OTP Intelligence",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()

PAGES = [
    ("views/home.py", "01  Home", "home", True),
    ("views/health.py", "02  System Health", "health", False),
    ("views/trends.py", "03  OTP Trends", "trends", False),
    ("views/forecast.py", "04  AI Forecast", "forecast", False),
    ("views/scenario.py", "05  Scenario Lab", "scenario", False),
    ("views/research.py", "06  Research", "research", False),
]

nav = st.navigation(
    [st.Page(f, title=t, url_path=u, default=d) for f, t, u, d in PAGES],
    position="hidden",
)

with st.sidebar:
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;padding:8px 0 14px;">
  <div style="width:38px;height:38px;background:{GOLD};border-radius:7px;display:flex;align-items:center;justify-content:center;flex:none;">
    <span style="font-family:'Barlow Condensed',sans-serif;font-weight:800;font-size:13px;color:#0c1824;">SIR</span>
  </div>
  <div>
    <div style="font-family:'Barlow Condensed',sans-serif;font-weight:700;font-size:15px;color:#fff;text-transform:uppercase;line-height:1.1;">Staten Island</div>
    <div style="font-family:'Barlow Condensed',sans-serif;font-weight:600;font-size:11px;color:#3a6a80;text-transform:uppercase;letter-spacing:0.06em;">Railway</div>
  </div>
</div>
<div style="font-family:'DM Mono',monospace;font-size:9px;color:#24506a;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:14px;border-top:1px solid rgba(255,255,255,0.05);padding-top:12px;">OTP Intelligence Platform</div>
""", unsafe_allow_html=True)

    for f, t, _, _ in PAGES:
        st.page_link(f, label=t)

    st.markdown("<div style='border-top:1px solid rgba(255,255,255,0.05);margin:18px 0 12px;'></div>", unsafe_allow_html=True)
    st.markdown("""
<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
  <div style="width:7px;height:7px;border-radius:50%;background:#22c55e;box-shadow:0 0 6px #22c55e;"></div>
  <span style="font-family:'DM Mono',monospace;font-size:10px;color:#22c55e;">System Live</span>
</div>
<div style="font-family:'DM Mono',monospace;font-size:9px;color:#245060;">MTA Open Data · 2006–2026</div>
""", unsafe_allow_html=True)

nav.run()
