import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from shared.styles import page_header, breadcrumb, chart_title, GOLD
from shared import data
from shared.charts import base_layout

breadcrumb("RESEARCH", prev=("views/scenario.py", "05 Scenario Lab"))
page_header("06", "Research", "The Science Behind It",
            "Phase 13–15 validation of the XGBoost forecaster. Target: next-month 7-Day "
            "On-Time Performance (without boat). All numbers from saved research artifacts.")

tab13, tab14, tab15 = st.tabs(["Phase 13 · Baseline", "Phase 14 · Cross-Validation", "Phase 15 · Intervals"])

with tab13:
    p13 = data.load_report("phase13_model_comparison.csv").copy()
    p13["Rank"] = ["1st ⭐", "2nd", "3rd", "4th", "5th", "Baseline"]
    st.dataframe(
        p13.style.apply(lambda r: ["background-color: rgba(233,184,32,0.12)"] * len(r)
                        if r["Model"] == "XGBoost Fair" else [""] * len(r), axis=1)
           .format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R2": "{:.4f}", "MAPE": "{:.4f}"}),
        use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        chart_title("Model Comparison — MAE (pts, lower is better)")
        colors = [GOLD if m == "XGBoost Fair" else "rgba(74,138,176,0.6)" for m in p13["Model"]]
        fig = go.Figure(go.Bar(x=p13["Model"], y=p13["MAE"], marker_color=colors))
        base_layout(fig, height=230, pct_axis=False, legend=False)
        fig.update_xaxes(tickfont=dict(size=9))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        chart_title("Model Comparison — RMSE (pts)")
        fig = go.Figure(go.Bar(x=p13["Model"], y=p13["RMSE"], marker_color=colors))
        base_layout(fig, height=230, pct_axis=False, legend=False)
        fig.update_xaxes(tickfont=dict(size=9))
        st.plotly_chart(fig, use_container_width=True)

    st.info("Fairly retrained XGBoost (trained only on pre-test data) cut MAE by ~18.7% vs the "
            "6-month moving average and was the only model with positive R² on the "
            "Aug 2025 – Jan 2026 test window.")

with tab14:
    folds = data.load_report("phase14_timeseries_cv_results.csv")
    summary = data.load_report("phase14_timeseries_cv_summary.csv")
    st.dataframe(summary.style.format({"Mean": "{:.4f}", "Standard_Deviation": "{:.4f}",
                                       "Minimum": "{:.4f}", "Maximum": "{:.4f}"}),
                 use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        chart_title("MAE by Chronological Fold")
        fig = go.Figure(go.Bar(x=folds["Fold"], y=folds["MAE"],
                               marker_color=["rgba(74,138,176,0.6)"] * 4 + [GOLD]))
        base_layout(fig, height=230, pct_axis=False, legend=False)
        fig.update_xaxes(title="Fold (1 = earliest era)", dtick=1)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        chart_title("Error vs. Training History")
        fig = go.Figure(go.Scatter(x=folds["Train_Rows"], y=folds["MAE"], mode="lines+markers",
                                   line=dict(color=GOLD, width=2.5)))
        base_layout(fig, height=230, pct_axis=False, legend=False)
        fig.update_xaxes(title="Training rows available")
        st.plotly_chart(fig, use_container_width=True)
    st.info("Average MAE 2.11 ± 0.59 pts across 5 folds. Accuracy improves almost monotonically "
            "with training history (fold 1: 2.93 → fold 5: 1.30), so the Phase 13 result reads as "
            "a recent-era estimate.")

with tab15:
    p15 = data.load_report("phase15_prediction_interval_summary.csv")
    st.dataframe(p15.style.format({"Lower_Residual_Bound": "{:.4f}", "Upper_Residual_Bound": "{:.4f}",
                                   "Coverage_Percentage": "{:.2f}", "Average_Interval_Width": "{:.4f}"}),
                 use_container_width=True, hide_index=True)
    chart_title("Observed vs. Expected Coverage")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["80% Interval", "90% Interval"], y=[80, 90], name="Expected",
                         marker_color="rgba(74,138,176,0.5)"))
    fig.add_trace(go.Bar(x=["80% Interval", "90% Interval"], y=list(p15["Coverage_Percentage"]),
                         name="Observed", marker_color=GOLD))
    base_layout(fig, height=240, pct_axis=False)
    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    st.plotly_chart(fig, use_container_width=True)
    st.info("Residual-based intervals are calibrated within ~0.5 pts of nominal: 79.49% observed "
            "coverage at 80%, 89.74% at 90%. A 96% point forecast reads as roughly 93–99% at 80% "
            "confidence.")

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
chart_title("AI Explainability — Top Operational Drivers (mean |SHAP|)")
shap_imp = data.load_report("shap_importance.csv").head(6).iloc[::-1]
colors = ["rgba(74,138,176,0.5)"] * 5 + [GOLD]
fig = go.Figure(go.Bar(x=shap_imp["MeanAbsSHAP"], y=shap_imp["Feature"], orientation="h",
                       marker_color=colors))
base_layout(fig, height=220, pct_axis=False, legend=False)
fig.update_yaxes(gridcolor="rgba(255,255,255,0)")
st.plotly_chart(fig, use_container_width=True)
st.markdown(
    "<p style=\"font-family:'Barlow',sans-serif;font-size:12px;color:rgba(255,255,255,0.35);\">"
    "Seasonality (month number) dominates by a wide margin, followed by the current delay rate and "
    "long-run era effects — drivers a transit planner would recognize.</p>", unsafe_allow_html=True)
