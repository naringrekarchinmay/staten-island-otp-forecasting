# PROJECT_ROADMAP.md — Staten Island Railway OTP Forecasting

**Portfolio role:** Centerpiece #1 — improve this project FIRST (Month 1 of `../AI_Career_Project_Roadmap.md`)
**Total effort:** ~4 weeks at 6–8 hrs/week
**Usage:** Open this folder and say — *"Claude, read PROJECT_ROADMAP.md and execute Phase 1."*

---

## 1. Project Summary

Applied ML system forecasting Staten Island Railway on-time performance from 20 years of real MTA open data (1,206 rows, 2006–2026). XGBoost beats 5 baselines (SARIMA, Prophet, naive, 3/6-month moving averages) with MAE 1.21 — an 18.7% improvement over the best baseline — validated with 5-fold expanding-window time-series CV, explained with SHAP, and wrapped in calibrated 80/90% prediction intervals (empirical coverage 79.5%/89.7%). Served through a 6-page "Dispatch" dark-theme Streamlit dashboard (Home, System Health, Trends, AI Forecast, Scenario Lab, Research).

## 2. Current State Audit (2026-07-06)

**Strengths**
- Real, authoritative data (MTA open data), actively maintained (last commit July 5).
- Rigorous methodology: fair baseline comparison (Phase 13), time-series CV (Phase 14), calibrated prediction intervals (Phase 15), SHAP explainability.
- Well-architected app: `app/views/` + `app/shared/` separation, type hints, caching, consistent design system; PRODUCT.md and DESIGN.md exist.
- 10 well-documented notebooks with clear progression; clean git history; pushed to GitHub.

**Weaknesses**
- **No test suite at all** — validation is notebook-based and manual.
- `requirements.txt` (14 packages) is **not version-pinned**; Prophet/statsmodels API drift will eventually break notebooks.
- **Not deployed** — local-only despite being demo-ready.
- **Target-naming inconsistency:** README/notebook 10 conflict on "7-Day OTP" vs "7-Day OTP With Boat" (already flagged in `docs/plans/2026-07-03-submission-readiness.md`).
- No CI/CD, no API, no automated retraining when MTA publishes new months; PI bounds hardcoded in `app/shared/data.py`; feature logic duplicated between notebook 03 and `app/shared/data.py`.
- README is 23 KB — thorough but buries the headline results.

**Existing plans:** `docs/plans/2026-07-03-submission-readiness.md` covers course-submission tasks (report, deck, naming fix). This roadmap absorbs its naming/pinning fixes into Phase 1 and goes beyond it for portfolio purposes — treat the submission plan as complementary, not duplicated.

## 3. Target Final Version

A **live, tested, CI-verified public forecasting app**: deployed on Streamlit Cloud with a link at the top of a results-first README; pytest suite covering feature engineering, forecast logic, and risk scoring; GitHub Actions running tests on every push; consistent target naming everywhere; optional FastAPI endpoint as a stretch. The interview line: *"I forecast the railway I help operate, beat SARIMA and Prophet by 18%, quantified my uncertainty, and it's live — here's the link."*

## 4. Must-Do Improvements

1. Fix target-variable naming inconsistency across README, notebook 10, and report.
2. Pin all versions in `requirements.txt` (import-check Prophet/statsmodels compatibility).
3. Add pytest suite: lag/rolling feature construction, recursive forecast logic, risk-score thresholds, model loading.
4. Deploy to Streamlit Cloud; verify the public link works cold (fresh container).
5. Add GitHub Actions CI running the test suite.
6. Rewrite README results-first (headline metrics table + dashboard screenshot + live link in the first screen).

## 5. Should-Do Improvements

1. Extract shared feature logic into a small `src/` module imported by both notebooks and `app/shared/data.py` (kills the duplication risk).
2. Document a manual "new MTA month" refresh procedure (which notebooks to re-run, in what order) in `docs/`.
3. Add a limitations section to README (R² variance across CV folds 0.18 ± 0.39, external factors unmodeled, one-step PI bounds).
4. Store PI bounds + model metadata in a JSON artifact instead of hardcoded constants in `data.py`.

## 6. Nice-to-Have Improvements

1. FastAPI `/forecast` endpoint returning forecast + intervals as JSON (good rehearsal for the Disease project's API phase).
2. Dockerfile (can wait — the Disease project is the designated Docker learning ground).
3. Automated monthly ingestion via scheduled GitHub Action (this graduates into Future Project #2 in the master roadmap — don't build it here yet).

## 7–10. Execution Phases (with Claude Code instructions, model type, and skills per phase)

### Phase 1 — Correctness & Reproducibility (≈4–6 hrs)
- **Goal:** Naming consistency + pinned environment.
- **Tasks:** Fix "7-Day OTP (without boat)" naming in README, notebook 10, and report; pin every package in `requirements.txt` to currently-installed versions; verify app boots and notebooks' imports still resolve in a fresh venv.
- **Claude Code instruction:** *"Read PROJECT_ROADMAP.md Phase 1. Grep for every occurrence of 'With Boat' and '7-Day' across README.md, notebooks/, reports/, and app/; make the target naming consistent as '7-Day OTP (without boat)'. Then pin requirements.txt to the versions in the current environment, create a fresh venv, install, and run `streamlit run app/streamlit_app.py` headless to verify it boots. Use verification-before-completion before declaring done."*
- **Model type:** Fast/cheap (claude-haiku-4-5) for the renaming sweep; coding model (claude-sonnet-5) for the environment verification.
- **Skills:** `executing-plans`, `verification-before-completion`.

### Phase 2 — Test Suite (≈6–8 hrs)
- **Goal:** Real pytest coverage of the logic an interviewer would probe.
- **Tasks:** Create `tests/` with: lag/rolling feature correctness on a synthetic mini-series; recursive forecast feeds predictions back correctly and returns the requested horizon; risk-score thresholds map to Low/Medium/High as intended; model artifact loads and predicts within sane OTP bounds (0–100); PI bounds applied correctly.
- **Claude Code instruction:** *"Read PROJECT_ROADMAP.md Phase 2. Use test-driven-development: write pytest tests for the feature engineering and forecast/risk logic in app/shared/data.py, refactoring minimally only if needed for testability (no behavior changes). Target the five test areas listed in the roadmap. Run pytest until green."*
- **Model type:** Coding-focused (claude-sonnet-5 or claude-opus-4-8).
- **Skills:** `test-driven-development`, `systematic-debugging` (if refactoring surfaces issues), `verification-before-completion`.

### Phase 3 — Deployment (≈4–6 hrs)
- **Goal:** Public live app.
- **Tasks:** Deploy to Streamlit Cloud from the GitHub repo; confirm data/model artifacts are in-repo and paths are relative; test all 6 pages on the live URL; add the URL to README and repo description.
- **Claude Code instruction:** *"Read PROJECT_ROADMAP.md Phase 3. Prepare the repo for Streamlit Cloud deployment: verify all paths are relative, artifacts committed, requirements install cleanly. Walk me through the Streamlit Cloud steps I must do in the browser, then use agent-browser or webapp-testing to verify every page of the live URL renders without errors."* (The Streamlit Cloud account clicks are **human/manual** work.)
- **Model type:** Coding-focused for prep; human for the cloud console.
- **Skills:** `webapp-testing`, `agent-browser`, `verification-before-completion`.

### Phase 4 — CI + README Rewrite (≈5–7 hrs)
- **Goal:** Green badge + a README that sells in 30 seconds.
- **Tasks:** Add `.github/workflows/ci.yml` (install pinned reqs, run pytest) and confirm it passes on push; rewrite README top: one-paragraph problem, headline metrics table (MAE 1.21 vs baselines, CV summary, PI coverage), dashboard screenshot, live-app link, then existing detail below; add limitations section; add CI badge.
- **Claude Code instruction:** *"Read PROJECT_ROADMAP.md Phase 4. Create a GitHub Actions workflow that installs requirements.txt and runs pytest on push/PR. Then restructure README.md to be results-first per the roadmap — keep all existing content but reorganize; verify every metric you state against outputs/reports/*.csv. Use humanizer on the intro paragraph. Finish with requesting-code-review for a final pass."*
- **Model type:** Coding-focused for CI; fast/cheap (claude-haiku-4-5) for README restructuring; highest-reasoning (claude-fable-5) for the final review.
- **Skills:** `executing-plans`, `humanizer`, `requesting-code-review`, `receiving-code-review`, `finishing-a-development-branch`, `verification-before-completion`.

### Phase 5 (Stretch, only if Month 1 has hours left) — FastAPI endpoint (≈6–8 hrs)
- **Goal:** `/forecast?months=6` returning JSON forecast + intervals; pytest for the endpoint.
- **Claude Code instruction:** *"Read PROJECT_ROADMAP.md Phase 5. Add a minimal FastAPI app (api/main.py) exposing the recursive forecast with prediction intervals as JSON, reusing app/shared/data.py logic. TDD the endpoint with httpx test client."*
- **Model type:** Coding-focused.
- **Skills:** `tdd`, `verification-before-completion`.

## 11. Recommended Skills by Phase

| Phase | Goal | Model Type | Skills to Use | Why These Skills |
|-------|------|------------|---------------|------------------|
| 1 | Naming fix + pinned env | Fast/cheap (haiku-4-5) + coding (sonnet-5) | `executing-plans`, `verification-before-completion` | Mechanical sweep + proof the env still works |
| 2 | Pytest suite | Coding (sonnet-5 / opus-4-8) | `test-driven-development`, `systematic-debugging`, `verification-before-completion` | Tests written test-first stay honest; debugging if refactors bite |
| 3 | Live deployment | Coding + human (cloud console) | `webapp-testing`, `agent-browser`, `verification-before-completion` | Automated verification of the live app, page by page |
| 4 | CI + results-first README | Coding + fast/cheap + highest-reasoning review | `humanizer`, `requesting-code-review`, `receiving-code-review`, `finishing-a-development-branch` | Natural-sounding copy; structured final review before calling it done |
| 5 (stretch) | FastAPI endpoint | Coding (sonnet-5) | `tdd`, `verification-before-completion` | API contract enforced by tests |

## 12. Skills Not Needed for This Project

`shadcn`, `vercel-react-best-practices` (no React), `banner-design`, `brand`, `design-system` (Dispatch theme already done — do not restyle), `pptx`/`slides` (deck exists via submission plan), `xlsx`, `pdf` (unless the course report needs export), `dispatching-parallel-agents` / `subagent-driven-development` (project too small), `using-git-worktrees` (no risky refactor planned), `brainstorming` (scope is fixed).

## 13. Files Likely to Be Modified

- `README.md` (naming, results-first rewrite, badges, live link, limitations)
- `requirements.txt` (pinning)
- `notebooks/10_research_question_methodology.ipynb` (naming)
- `reports/final_report.md` (naming)
- `app/shared/data.py` (only if Phase 2 needs testability refactors; Phase 5 reuse)

## 14. New Files Likely to Be Created

- `tests/test_features.py`, `tests/test_forecast.py`, `tests/test_risk.py`
- `.github/workflows/ci.yml`
- `docs/data_refresh.md` (should-do)
- `api/main.py`, `tests/test_api.py` (stretch Phase 5)
- Model/PI metadata JSON (should-do)

## 15. Testing / Validation Checklist

- [ ] Fresh venv from pinned `requirements.txt` installs cleanly and app boots
- [ ] `pytest` green locally: features, forecast recursion, risk thresholds, model load, PI application
- [ ] GitHub Actions run green on push
- [ ] All 6 pages of the **live** app render with data (checked on the public URL, not localhost)
- [ ] Every metric in README matches `outputs/reports/*.csv` exactly (MAE 1.2087, coverage 79.49%/89.74%, etc.)
- [ ] No occurrence of the wrong "With Boat" phrasing remains (`grep -ri "with boat"`)

## 16. README & Portfolio Update Checklist

- [ ] README: live-app link + CI badge + headline metrics table + screenshot in the first screen
- [ ] README: limitations section added
- [ ] GitHub repo description + topics set (forecasting, xgboost, transit, streamlit)
- [ ] Portfolio site `staten-island-otp.html`: add live-app button, refresh metrics
- [ ] Repo pinned on GitHub profile

## 17. LinkedIn Post Idea

*"I work in transit analytics — so I built a forecasting system for the Staten Island Railway using 20 years of public MTA data. XGBoost beat SARIMA, Prophet, and moving-average baselines by 18.7% (MAE 1.21), validated with time-series cross-validation, with prediction intervals that actually achieve their stated coverage (80% interval → 79.5% observed). It's live — link in comments. The most useful lesson: the boring baselines were embarrassingly hard to beat."* (Run through `humanizer`; attach dashboard screenshot.)

## 18. Definition of Done

Live public URL renders all 6 pages · pytest suite green in CI on GitHub · requirements pinned · naming consistent everywhere · results-first README with verified metrics, limitations, live link, CI badge · portfolio page updated · LinkedIn post drafted.

## 19. Risks & Things to Avoid

- **Do not restyle the dashboard.** The Dispatch theme is finished; design tinkering is the #1 time sink risk here.
- **Do not build automated retraining yet** — that's Future Project #2; scope creep kills Month 1.
- Prophet is heavy and only needed by notebook 07 — if Streamlit Cloud deploy struggles, split app requirements from notebook requirements rather than debugging Prophet in the cloud.
- Model artifact (686 KB) and data are committed — fine; don't "clean" them out of git or the deployed app breaks.
- Keep the submission-readiness plan's course deliverables separate — don't let portfolio work break the assignment artifacts before submission.
- If a week is lost, cut Phase 5 (stretch), never Phase 2 (tests) or 3 (deploy).
