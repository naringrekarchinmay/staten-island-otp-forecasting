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
4. **Required — do not skip:** click **Advanced settings** and set
   **Python version = 3.13**. The pinned `numpy`/`pandas` wheels only exist
   up to Python 3.13 (cp313); if the build lands on 3.14 it will try to
   compile numpy/pandas from source and hang at "Your app is in the oven"
   forever. Streamlit Cloud cannot change the Python version after deploy and
   does not read a `runtime.txt`, so this must be set here. No secrets needed.
5. Click **Deploy**. With Python 3.13 it installs prebuilt wheels (no
   compiling) and is live in ~2 minutes.

> If an app is already stuck on 3.14: delete it (**⋮ → Delete app**) and
> redeploy from step 2 — the version is fixed at first deploy and can only be
> changed by redeploying.

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
