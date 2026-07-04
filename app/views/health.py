import streamlit as st
import plotly.graph_objects as go
from shared.styles import page_header, breadcrumb, chart_title
from shared import data
from shared.charts import base_layout, threshold_90

breadcrumb("SYSTEM HEALTH", prev=("views/home.py", "01 Home"), nxt=("views/trends.py", "03 OTP Trends"))
page_header("02", "System Health", "How Are We Doing?",
            "Current system performance across 241 months of 7-Day service data.")

df7 = data.seven_day(data.load_clean()).copy()
otp = df7.set_index("Month")["On-Time Performance"]

recent12 = otp.tail(12).mean()
prior12 = otp.tail(24).head(12).mean()
delta = (recent12 - prior12) * 100
below90 = int((otp < 0.90).sum())

feat7 = data.seven_day(data.load_features())

c1, c2, c3, c4 = st.columns(4)
with c1, st.container(key="gold_health"):
    st.metric("Average OTP", f"{otp.mean():.2%}", f"{delta:+.1f} pts vs. prior 12 mo")
c2.metric("Average Delay Rate", f"{feat7['Delay_Rate'].mean():.2%}")
c3.metric("Best Month OTP", f"{otp.max():.1%}", f"{otp.idxmax():%b %Y}", delta_color="off")
c4.metric("Latest Month OTP", f"{otp.iloc[-1]:.2%}", f"{otp.index[-1]:%b %Y}", delta_color="off")

st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
chart_title("OTP — Monthly with 6-Month Rolling Average")

rolling = otp.rolling(6).mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=otp.index, y=otp.values * 100, name="Monthly OTP",
                         line=dict(color="rgba(74,138,176,0.55)", width=1)))
fig.add_trace(go.Scatter(x=rolling.index, y=rolling.values * 100, name="6-Mo Rolling",
                         fill="tozeroy", fillcolor="rgba(233,184,32,0.08)",
                         line=dict(color="#e9b820", width=2.5)))
base_layout(fig, height=240, yrange=[82, 101])
threshold_90(fig)
st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns(2)
c1.metric("Total Service Months Analyzed", f"{len(otp)}", "2006 – 2026", delta_color="off")
c2.metric("Months Below 90% OTP", f"{below90}", f"{below90/len(otp):.1%} of all months",
          delta_color="inverse")
