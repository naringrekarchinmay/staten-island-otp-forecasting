# PRODUCT.md

**Product:** SIR OTP Intelligence Platform — a 6-page Streamlit dashboard forecasting Staten Island Railway on-time performance with XGBoost, SHAP explainability, and prediction intervals.

**Register:** product (dashboard/tool — design serves the task).

**Users:** transit planners and course evaluators reviewing reliability trends, forecasts, and model validation evidence.

**Narrative arc:** Home (hero + chapters) → System Health → OTP Trends → AI Forecast → Scenario Lab → Research.

**Data ground truth:** monthly MTA open data 2006–2026; headline metrics from Phase 13–15 artifacts in `outputs/reports/` (fair-test MAE 1.2087 pts, CV MAE 2.1107, 80/90% interval coverage 79.49/89.74%). All displayed numbers must come from repo artifacts — never invent metrics.

**Constraints:** Streamlit 1.45 + Plotly only; run from repo root (`streamlit run app/streamlit_app.py`); no model retraining; keep target phrasing "next-month 7-Day On-Time Performance (without boat)".
