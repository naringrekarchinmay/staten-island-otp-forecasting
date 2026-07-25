# Deploying to Streamlit Community Cloud

The repository is deploy-ready: all data/model artifacts are committed, every
file path is relative, and the root `requirements.txt` holds only the app
runtime (no Prophet/cmdstan build in the cloud).

## One-time setup (manual, in the browser)

1. Go to **https://share.streamlit.io** and sign in with the GitHub account
   that owns `naringrekarchinmay/staten-island-otp-forecasting`.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `naringrekarchinmay/staten-island-otp-forecasting`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
   - **App URL:** pick a slug, e.g. `staten-island-otp` (→ `https://staten-island-otp.streamlit.app`)
4. (Optional) **Advanced settings → Python version:** 3.12 or 3.13 — either
   resolves the pinned wheels cleanly. No secrets are required.
5. Click **Deploy**. First build takes ~2–4 minutes (slim requirements, no
   Prophet). Watch the build log; it should end with the app running.

## After it goes live

- Paste the public URL back here and I'll verify all 6 pages render on the
  live container (Home, System Health, Trends, AI Forecast, Scenario Lab,
  Research), then add the link + badge to the README (Phase 4).
- Set the GitHub repo **About** → website field to the same URL.

## Redeploys

Streamlit Cloud auto-redeploys on every push to `main`. To force a rebuild
(e.g. after changing `requirements.txt`), use **Manage app → Reboot**.

## Notes

- The dashboard reads committed artifacts under `data/`, `models/`, and
  `outputs/`; nothing is fetched at runtime.
- To reproduce the notebooks locally, install the full stack instead:
  `pip install -r requirements-notebooks.txt`.
