# ANLY 530 Submission-Readiness Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. One task per session unless the user says continue.

**Goal:** Make the Staten Island Railway OTP forecasting project submission-ready per `CourseProject_General.pdf`: internally consistent, reproducible, with a final report and presentation.

**Architecture:** No new modeling. Fix narrative/dependency inconsistencies, then author `reports/final_report.md` and `reports/final_presentation.pptx` from existing repository evidence (Phases 13–15 are the headline results). Optional scoped dashboard upgrade, then final verification.

**Tech Stack:** Existing Jupyter notebooks (01–10), Streamlit app, python-pptx for the deck.

## Global Constraints

- Do not rewrite the project; keep every change minimal and scoped.
- Authoritative target: **next-month 7-Day On-Time Performance (without boat)**. Evidence: `notebooks/03_feature_engineering.ipynb` builds `Next_Month_OTP` from the `On-Time Performance` column; the raw data has a separate `On-Time Performance (With Boat)` column that is NOT used as the target. `notebooks/10_research_question_methodology.ipynb` §6 wrongly says "7-Day OTP With Boat" — it must be corrected, not the code.
- Preserve Phase 13–15 results verbatim: Phase 13 XGBoost Fair MAE 1.2087, RMSE 1.4974, R² 0.4821, MAPE 1.2394 (best baseline: 6-month MA MAE 1.4861, ≈18.7% MAE reduction); Phase 14 CV avg MAE 2.1107, RMSE 2.8548, R² 0.1820, MAPE 2.2677%; Phase 15 coverage 79.49% (80% PI) and 89.74% (90% PI).
- Weather, incidents, maintenance, ridership = **future work only**; never describe as done.
- Feature dataset ends at 2025-12 because `Next_Month_OTP` (shift -1) drops the last month (2026-01) from the feature table — state this wherever data range is described.
- Course PDF deliverables: (1) presentation, (2) final report following APA or IEEE guidelines, (3) code. Report must cover: introduction/motivation, related work, data + EDA, technical approach, test & evaluation (validity, metrics, baseline comparison, how well it worked, future work, replicability assessment).

---

### Task 1: Consistency + reproducibility fixes (prompts.md task 2)

**Files:**
- Modify: `README.md` (target naming ~lines 231, 285; add How to Run + notebook order + data-range note)
- Modify: `notebooks/10_research_question_methodology.ipynb` (§6 Data Source markdown cell)
- Modify: `requirements.txt`

**Interfaces:**
- Produces: canonical target phrasing "next-month 7-Day On-Time Performance (without boat)" reused verbatim by Tasks 2–4.

- [ ] **Step 1: Fix notebook 10 §6.** In the markdown cell containing "**7-Day OTP With Boat**", change the primary target to **7-Day OTP (without boat)** and move "7-Day OTP With Boat" into the future-extensions list (replacing the "Without Boat" bullet). Edit via `json` load/dump or NotebookEdit — do not touch code cells.
- [ ] **Step 2: Fix README target naming.** Change "7-Day OTP" at the two target-definition sites to "7-Day On-Time Performance (without boat)". Keep all Phase 13–15 numbers untouched.
- [ ] **Step 3: Add README "How to Run" section** with: `pip install -r requirements.txt`; notebook execution order 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 (01–03 required before 04+; 07–09 are Phases 13–15); `streamlit run app/streamlit_app.py`; note that the feature table ends 2025-12 due to the Next_Month_OTP shift.
- [ ] **Step 4: Update requirements.txt** — append `prophet` and `statsmodels` only (verified imports in `notebooks/07_baseline_model_comparison.ipynb`; no other missing packages found across notebooks/app).
- [ ] **Step 5: Verify.** `grep -rn "With Boat" README.md notebooks/10_*.ipynb` shows only the future-work mention; `python3 -c "import statsmodels, prophet"` (or note if env lacks them); confirm no code cells changed (`git diff --stat`).
- [ ] **Step 6: Commit** `docs: fix target-variable naming, add run instructions and missing deps`.

### Task 2: Final report (prompts.md task 3)

**Files:**
- Create: `reports/final_report.md`

**Interfaces:**
- Consumes: canonical target phrasing from Task 1; metrics from Global Constraints; figures in `outputs/figures/`, tables in `outputs/reports/`.
- Produces: report sections 1–17 that Task 3 slides mirror.

- [ ] **Step 1: Draft all 17 sections** (Introduction/motivation; Problem definition & research question; Related work; Data description; EDA & preprocessing; Feature engineering; Technical approach; Modeling methods; Validation & metrics; Results & findings; SHAP explainability; Prediction intervals; Dashboard; Limitations; Future work; Replicability; Conclusion). Academic APA-ish tone. Use only repository evidence; numbers verbatim from Global Constraints. Figure/table placeholders must reference real files, e.g. `outputs/figures/phase13_model_comparison_mae.png`, `outputs/reports/phase13_model_comparison.csv`.
- [ ] **Step 2: Verify** every metric against `outputs/reports/*.csv|txt`; every referenced figure exists (`ls outputs/figures/`); target phrasing matches Task 1; weather/incidents appear only under Future Work.
- [ ] **Step 3: Commit** `docs: add final project report`.

### Task 3: Presentation (prompts.md task 4)

**Files:**
- Create: `reports/final_presentation.pptx`

**Interfaces:**
- Consumes: report sections + these figures (priority order): `phase13_model_comparison_mae.png`, `phase13_actual_vs_key_models.png`, `phase14_cv_mae_by_fold.png`, `phase15_interval_coverage.png`, `phase15_prediction_intervals.png`, `shap_summary.png` (all in `outputs/figures/`).

- [ ] **Step 1: Build 10–12 slides** per the fixed outline (Title; Problem & motivation; Research question; Data & target; EDA; Features & model; Baseline comparison; Time-series CV; Prediction intervals; SHAP; Dashboard value; Limitations/future work/conclusion). Concise bullets; detail goes in speaker notes.
- [ ] **Step 2: Verify** deck opens, every image renders, numbers match the report exactly.
- [ ] **Step 3: Commit** `docs: add final presentation deck`.

### Task 4 (optional): Dashboard upgrade (prompts.md task 5)

**Files:**
- Modify: `app/streamlit_app.py`

- [ ] **Step 1: Add sections** reading `outputs/reports/phase13_model_comparison.csv`, `phase14_timeseries_cv_summary.csv`, `phase15_prediction_interval_summary.csv`, and displaying the Phase 13–15 PNGs. Keep existing forecast + SHAP sections; no retraining; use paths relative to repo root with existence checks.
- [ ] **Step 2: Verify** with `streamlit run app/streamlit_app.py` (webapp-testing skill) — no file-path/runtime errors.
- [ ] **Step 3: Commit** `feat: surface Phase 13-15 results in dashboard`.

### Task 5: Final verification (prompts.md task 6)

- [ ] **Step 1: Cross-check** README ↔ report ↔ deck ↔ notebooks ↔ app: same target, same numbers, same story; requirements.txt complete; notebook order documented; report has limitations/future work + replicability statement (course PDF requires it explicitly).
- [ ] **Step 2: Produce checklist** — Completed / Remaining risks / Files changed / Presentation-day talking points. No major changes at this stage.

## Self-Review Notes

- Course-PDF coverage: problem definition → report §2; EDA/preprocessing → §5; modeling/validation/metrics → §8–9; model comparison tables → §10 + Phase 13 CSV; future work → §15; replicability evaluation → §16; APA/IEEE style → report tone; three deliverables → code (repo), report (Task 2), presentation (Task 3). The PDF's "computer vision algorithm" line is boilerplate from another offering — do not force CV framing.
- Known non-issue: `README.md` already documents Phases 13–16; only naming and run-instructions change.
