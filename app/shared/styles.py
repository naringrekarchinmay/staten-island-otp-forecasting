"""Dispatch design system: fonts, global CSS, and shared page chrome."""
import streamlit as st

GOLD = "#e9b820"
BLUE = "#4a8ab0"
PURPLE = "#7b6caa"
NAVY = "#1b2d52"
BG = "#0c1824"
BG_CARD = "#0a1520"
GREEN = "#22c55e"
RED = "#ef4444"
AMBER = "#f59e0b"
MUTED = "#2e5a6e"

def inject_styles():
    st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=Barlow:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');
.stApp { background-color: #0c1824; }
[data-testid="stSidebar"] { background-color: #0e1f30; border-right: 1px solid rgba(255,255,255,0.05); }
.block-container { padding-top: 4rem; padding-bottom: 2rem; max-width: 1240px; }
div[data-testid="stHeader"] { background: transparent; }
h1, h2, h3 { font-family: 'Barlow Condensed', sans-serif !important; text-transform: uppercase; letter-spacing: 0.04em; }
p, li, span, div { font-family: 'Barlow', sans-serif; }

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 14px 16px;
}
[data-testid="stMetric"] label p {
    font-family: 'DM Mono', monospace !important;
    font-size: 9.5px !important;
    color: #4a7a90 !important;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
[data-testid="stMetricValue"] {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 800;
    font-size: 42px !important;
    color: #ffffff;
    line-height: 1.05;
}
[data-testid="stMetricDelta"] { font-family: 'Barlow', sans-serif !important; font-size: 12px !important; }
div[class*="st-key-gold"] [data-testid="stMetricValue"] { color: #e9b820 !important; }

/* Sidebar page links */
[data-testid="stSidebar"] [data-testid="stPageLink"] a,
[data-testid="stSidebar"] [data-testid="stPageLink"] a p {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15.5px !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #4a7a90;
}
[data-testid="stSidebar"] [data-testid="stPageLink"] a { border-radius: 6px; padding: 2px 10px; }
[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] { background: rgba(233,184,32,0.08); }
[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] p { color: #e9b820 !important; }

/* Breadcrumb page links (top bar) */
.crumbrow [data-testid="stPageLink"] a p, .main [data-testid="stPageLink"] a p { font-family: 'DM Mono', monospace !important; font-size: 10.5px !important; letter-spacing: 0.08em; text-transform: uppercase; color: #4a8ab0; }

/* Charts + tables */
[data-testid="stPlotlyChart"] { background: #0a1520; border: 1px solid rgba(255,255,255,0.07); border-radius: 8px; padding: 6px; }
[data-testid="stDataFrame"] { background: #0a1520; border: 1px solid rgba(255,255,255,0.07); border-radius: 8px; }

/* Radio as segmented chips */
[data-testid="stRadio"] div[role="radiogroup"] { gap: 6px; }
[data-testid="stRadio"] label { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 4px; padding: 3px 12px; }
[data-testid="stRadio"] label p { font-family: 'DM Mono', monospace !important; font-size: 10px !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stRadio"] label:has(input:checked) { background: rgba(233,184,32,0.12); border-color: rgba(233,184,32,0.5); }

/* Tabs */
button[data-baseweb="tab"] p { font-family: 'DM Mono', monospace !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.08em; }

/* Alerts */
[data-testid="stAlert"] { border-radius: 8px; font-family: 'Barlow', sans-serif; }

hr { border-color: rgba(255,255,255,0.06); }
</style>
""")

def eyebrow(text, color=GOLD, mb=6):
    st.markdown(
        f"<p style=\"font-family:'DM Mono',monospace;font-size:10px;color:{color};"
        f"letter-spacing:0.16em;text-transform:uppercase;margin:0 0 {mb}px;\">{text}</p>",
        unsafe_allow_html=True)

def page_header(num, title, tagline, subtitle=None):
    st.markdown(f"""
<div style="margin:2px 0 18px;">
  <p style="font-family:'DM Mono',monospace;font-size:10px;color:{GOLD};letter-spacing:0.16em;text-transform:uppercase;margin:0 0 6px;">{num} · {title}</p>
  <h1 style="font-family:'Barlow Condensed',sans-serif;font-weight:800;font-size:38px;line-height:1;color:#fff;text-transform:uppercase;margin:0;">{tagline}</h1>
  {f'<p style="font-family:Barlow,sans-serif;font-size:13.5px;color:rgba(255,255,255,0.45);margin:8px 0 0;">{subtitle}</p>' if subtitle else ''}
</div>""", unsafe_allow_html=True)

def breadcrumb(page_label, prev=None, nxt=None):
    """Top bar: SIR / OTP_INTELLIGENCE / PAGE with prev/next links."""
    left, p, n = st.columns([6, 2, 2])
    with left:
        st.markdown(
            f"<p style=\"font-family:'DM Mono',monospace;font-size:10px;color:{MUTED};"
            f"letter-spacing:0.1em;text-transform:uppercase;margin:6px 0 0;\">"
            f"SIR / OTP_INTELLIGENCE / <span style='color:{GOLD};'>{page_label}</span></p>",
            unsafe_allow_html=True)
    if prev:
        with p:
            st.page_link(prev[0], label=f"← {prev[1]}")
    if nxt:
        with n:
            st.page_link(nxt[0], label=f"{nxt[1]} →")
    st.markdown("<div style='border-bottom:1px solid rgba(255,255,255,0.06);margin:6px 0 18px;'></div>",
                unsafe_allow_html=True)

def chart_title(text):
    eyebrow(text, color="#4a7a90", mb=4)
