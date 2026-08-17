# Data Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. One task per session unless the user says continue.

**Goal:** Refresh the committed MTA data from 2026-01 to the latest published month, regenerate every downstream artifact and metric, and replace the manual download with a reproducible, tested fetch script before the paper is drafted.

**Architecture:** Replace the hand-downloaded `.xlsx` with a script that pulls the same dataset from the data.ny.gov Socrata API and writes the identical Excel file that `notebooks/01` already reads, so no notebook logic changes. Then re-run notebooks 01 to 09 in order, which regenerates the cleaned data, feature table, model artifact, and all Phase 13 to 15 metric CSVs. Finally propagate the new numbers into the app constants, README, and report, using the existing pytest suite as the drift detector.

**Tech Stack:** Python 3.13, pandas, openpyxl, requests, existing Jupyter notebooks 01 to 09, pytest.

## Global Constraints

- **Source of truth:** data.ny.gov Socrata dataset `fccm-griq` ("MTA Staten Island Railway On-Time Performance: Beginning 2006"). CSV endpoint: `https://data.ny.gov/resource/fccm-griq.csv`. This is the same data behind the `metrics.mta.info` dashboard the original file came from.
- **Source state (updated 2026-08-04):** the source published 2026-06 while Task 1 was in progress, so the refresh applied was **25 rows = 5 new months** (2026-02 through 2026-06), not the 20 rows/4 months first measured on 2026-07-27. Task 2 refreshed the repository to 1,230 raw rows (246 in the 7-Day series) and 1,195 feature rows ending 2026-05. Re-run the dry run before assuming any figure here is current.
- **Socrata field names are not the Excel column names.** The CSV export returns snake_case field names, and `on_time_performance_with` is truncated by Socrata's field-name length limit. It maps to `On-Time Performance (With Boat)`. Any script must map all 11 columns explicitly and fail loudly if the source schema changes.
- **Preserve source row order.** The shipped `.xlsx` is ordered by Month, then by the fixed category sequence `Weekday, AM Rush, PM Rush, Weekend, 7-Day`. Sorting alphabetically instead would rewrite every row of `cleaned_staten_island_otp.csv` and produce a 1,200-line spurious diff. Feature correctness is unaffected either way because `notebooks/03` re-sorts with `sort_values(['Day Time', 'Month'])`, but the diff must stay readable.
- **Every headline metric will change.** The Phase 13 test window is `iloc[-6:]` of the feature table, so it shifts automatically from Aug 2025–Jan 2026 to Nov 2025–Apr 2026. MAE 1.2087, the 18.7% improvement, CV MAE 2.1107, and interval coverage 79.49%/89.74% are all invalidated by this refresh. No document may keep the old numbers alongside new data.
- **`app/shared/data.py` PI constants are hardcoded** (`PI80 = (-3.1265, 3.3576)`, `PI90 = (-4.4797, 4.1961)`) and must be updated from the regenerated `outputs/reports/phase15_prediction_interval_summary.csv`. `tests/test_forecast.py::TestPredictionIntervals::test_pi_constants_match_phase15_artifact` fails if they drift, so the test suite is the safety net. Do not delete or weaken that test.
- **Paper sequencing:** the paper is targeted for next month. Complete this entire refresh *before* drafting results, so the paper is written once against final numbers. Once the paper draft quotes numbers, freeze the data and record the frozen month in the report.
- Do not build the scheduled GitHub Action in this plan. `PROJECT_ROADMAP.md` §19 treats automated ingestion as scope creep for the current phase; this plan delivers the reproducible script the Action would later call.

---

### Task 1: Reproducible fetch script

**Files:**
- Create: `scripts/fetch_mta_data.py`
- Create: `tests/test_fetch.py`
- Modify: `requirements-notebooks.txt` (add `requests`)

**Interfaces:**
- Produces: `normalize(raw: pd.DataFrame) -> pd.DataFrame`, which renames the 11 Socrata columns to the Excel names, parses `Month` to datetime, and orders rows by Month then the fixed category sequence. Task 2 calls the script's CLI.
- Produces: `fetch(limit: int = 50000) -> pd.DataFrame` and `main() -> None` writing `data/raw/MTA_Staten_Island_Railway_On-Time_Performance.xlsx`.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_fetch.py
"""The Socrata fetch script maps the source schema onto the Excel layout."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.fetch_mta_data import CATEGORY_ORDER, COLUMN_MAP, EXCEL_COLUMNS, normalize


def _raw_row(month, day_time, **over):
    row = {
        "month": month, "day_time": day_time, "delayed_trains": 10,
        "on_time_trips": 100, "on_time_performance": 0.95,
        "delayed_trains_with_boat": 12, "on_time_trips_with_boat": 98,
        "on_time_performance_with": 0.94, "scheduled_trips": 110,
        "incomplete_trips": "N/A", "trip_complete_percentage": "N/A",
    }
    row.update(over)
    return row


def test_renames_socrata_fields_to_excel_columns():
    raw = pd.DataFrame([_raw_row("2026-02-01T00:00:00.000", "7-Day")])
    out = normalize(raw)
    assert list(out.columns) == EXCEL_COLUMNS


def test_truncated_boat_field_maps_to_full_column_name():
    # Socrata truncates the field to `on_time_performance_with`.
    raw = pd.DataFrame([_raw_row("2026-02-01T00:00:00.000", "7-Day",
                                 on_time_performance_with=0.912)])
    out = normalize(raw)
    assert out.loc[0, "On-Time Performance (With Boat)"] == 0.912


def test_month_is_parsed_to_datetime():
    raw = pd.DataFrame([_raw_row("2026-02-01T00:00:00.000", "7-Day")])
    out = normalize(raw)
    assert out["Month"].dtype.kind == "M"
    assert out.loc[0, "Month"] == pd.Timestamp("2026-02-01")


def test_rows_ordered_by_month_then_source_category_sequence():
    raw = pd.DataFrame([
        _raw_row("2026-03-01T00:00:00.000", "7-Day"),
        _raw_row("2026-02-01T00:00:00.000", "Weekend"),
        _raw_row("2026-02-01T00:00:00.000", "Weekday"),
    ])
    out = normalize(raw)
    assert list(out["Day Time"]) == ["Weekday", "Weekend", "7-Day"]
    assert out["Month"].is_monotonic_increasing


def test_category_order_matches_the_shipped_workbook():
    assert CATEGORY_ORDER == ["Weekday", "AM Rush", "PM Rush", "Weekend", "7-Day"]


def test_missing_source_column_raises_rather_than_silently_dropping():
    raw = pd.DataFrame([_raw_row("2026-02-01T00:00:00.000", "7-Day")]).drop(
        columns=["on_time_performance_with"])
    with pytest.raises(ValueError, match="on_time_performance_with"):
        normalize(raw)


def test_column_map_covers_all_eleven_source_fields():
    assert len(COLUMN_MAP) == 11
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_fetch.py -v`
Expected: FAIL at import, `ModuleNotFoundError: No module named 'scripts'`.

- [x] **Step 3: Write the script**

```python
# scripts/fetch_mta_data.py
"""Download the MTA Staten Island Railway OTP dataset from data.ny.gov.

Writes the same Excel workbook that notebooks/01 reads, so refreshing the
data requires no notebook changes. The dataset behind metrics.mta.info is
published as Socrata dataset fccm-griq.
"""
from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

DATASET_ID = "fccm-griq"
SOURCE_URL = f"https://data.ny.gov/resource/{DATASET_ID}.csv"
RAW_XLSX = (Path(__file__).resolve().parents[1]
            / "data/raw/MTA_Staten_Island_Railway_On-Time_Performance.xlsx")

# Socrata field name -> column name used by the workbook and notebook 01.
# `on_time_performance_with` is truncated by Socrata's field-name limit.
COLUMN_MAP = {
    "month": "Month",
    "day_time": "Day Time",
    "delayed_trains": "Delayed Trains",
    "on_time_trips": "On-Time Trips",
    "on_time_performance": "On-Time Performance",
    "delayed_trains_with_boat": "Delayed Trains (With Boat)",
    "on_time_trips_with_boat": "On-Time Trips (With Boat)",
    "on_time_performance_with": "On-Time Performance (With Boat)",
    "scheduled_trips": "Scheduled Trips",
    "incomplete_trips": "Incomplete Trips",
    "trip_complete_percentage": "Trip Complete Percentage",
}
EXCEL_COLUMNS = list(COLUMN_MAP.values())

# Order the shipped workbook uses within each month. Preserved so a refresh
# produces a readable diff instead of reordering every row.
CATEGORY_ORDER = ["Weekday", "AM Rush", "PM Rush", "Weekend", "7-Day"]


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Map the Socrata export onto the workbook layout."""
    missing = [c for c in COLUMN_MAP if c not in raw.columns]
    if missing:
        raise ValueError(
            f"source schema changed; missing columns: {missing}. "
            f"Check https://data.ny.gov/d/{DATASET_ID}")
    df = raw[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)
    df["Month"] = pd.to_datetime(df["Month"])
    order = pd.Categorical(df["Day Time"], categories=CATEGORY_ORDER, ordered=True)
    df = df.assign(_order=order).sort_values(["Month", "_order"], kind="mergesort")
    return df.drop(columns="_order").reset_index(drop=True)


def fetch(limit: int = 50000) -> pd.DataFrame:
    """Download the full dataset and return it in workbook layout."""
    response = requests.get(SOURCE_URL, params={"$limit": limit, "$order": "month"},
                            timeout=60)
    response.raise_for_status()
    return normalize(pd.read_csv(StringIO(response.text)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RAW_XLSX,
                        help="destination workbook path")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change without writing")
    args = parser.parse_args()

    fresh = fetch()
    latest = fresh["Month"].max().strftime("%Y-%m")
    print(f"fetched {len(fresh)} rows from {DATASET_ID}, latest month {latest}")

    if args.out.exists():
        current = pd.read_excel(args.out)
        current_latest = pd.to_datetime(current["Month"]).max().strftime("%Y-%m")
        print(f"existing workbook: {len(current)} rows, latest month {current_latest}")
        print(f"delta: {len(fresh) - len(current)} new rows")

    if args.dry_run:
        print("dry run, nothing written")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fresh.to_excel(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_fetch.py -v`
Expected: 7 passed.

- [x] **Step 5: Confirm the whole suite is still green**

Run: `pytest tests/ -q`
Expected: `51 passed` (the existing 44 plus the 7 new).

- [x] **Step 6: Add `requests` to the notebook requirements**

In `requirements-notebooks.txt`, add below the `-r requirements.txt` line:

```
requests==2.32.3
```

Then confirm the pin resolves: `pip download requests==2.32.3 --no-deps -d /tmp/reqcheck` and expect a successful download. If that version is unavailable, pin the newest 2.x that is, and record the version used.

- [x] **Step 7: Commit**

```bash
git add scripts/fetch_mta_data.py tests/test_fetch.py requirements-notebooks.txt
git commit -m "feat: add reproducible MTA data fetch script

Replaces the manual dashboard download with a pull from the data.ny.gov
Socrata dataset fccm-griq, writing the same workbook notebook 01 reads.
Maps the truncated on_time_performance_with field and preserves the
source row order so refreshes produce readable diffs."
```

---

### Task 2: Refresh the raw data and rebuild the feature table

**Files:**
- Modify: `data/raw/MTA_Staten_Island_Railway_On-Time_Performance.xlsx` (regenerated)
- Modify: `data/raw/cleaned_staten_island_otp.csv` (regenerated by notebook 01)
- Modify: `outputs/predictions/staten_island_otp_features.csv` (regenerated by notebook 03)
- Run: `notebooks/01_data_loading_and_initial_review.ipynb`, `02_eda.ipynb`, `03_feature_engineering.ipynb`

**Interfaces:**
- Consumes: `scripts/fetch_mta_data.py` from Task 1.
- Produces: refreshed feature table that Tasks 3 and 4 train and evaluate against.

- [x] **Step 1: Preview the delta without writing**

Run: `python scripts/fetch_mta_data.py --dry-run`
Expected: reports roughly 1,225 fetched rows against 1,205 existing, delta about 20 rows. If the delta is 0, the source has not advanced and the rest of this plan can wait. If the delta is far larger than 20, the source may have been restated; inspect before continuing.

- [x] **Step 2: Write the refreshed workbook**

Run: `python scripts/fetch_mta_data.py`
Then confirm the diff is additive rather than a reordering:

```bash
git diff --stat data/raw/MTA_Staten_Island_Railway_On-Time_Performance.xlsx
```

Excel is binary so the stat is not meaningful on its own. Verify content instead:

```bash
python -c "
import pandas as pd
x = pd.read_excel('data/raw/MTA_Staten_Island_Railway_On-Time_Performance.xlsx')
print(len(x), 'rows;', x['Month'].min().date(), 'to', x['Month'].max().date())
print(x[['Month','Day Time']].head(5).to_string(index=False))
"
```
Expected: about 1,225 rows, 2006-01 to the new latest month, and the first five rows still ordered Weekday, AM Rush, PM Rush, Weekend, 7-Day.

- [x] **Step 3: Re-run notebooks 01 to 03 in order**

```bash
pip install -r requirements-notebooks.txt
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/01_data_loading_and_initial_review.ipynb \
  notebooks/02_eda.ipynb \
  notebooks/03_feature_engineering.ipynb
```
Expected: all three execute without error. Notebook 01 rewrites `cleaned_staten_island_otp.csv`; notebook 03 rewrites `staten_island_otp_features.csv`.

- [x] **Step 4: Verify the rebuilt tables**

```bash
python -c "
import pandas as pd
c = pd.read_csv('data/raw/cleaned_staten_island_otp.csv')
f = pd.read_csv('outputs/predictions/staten_island_otp_features.csv')
c['Month'] = pd.to_datetime(c['Month']); f['Month'] = pd.to_datetime(f['Month'])
print('clean:', len(c), c['Month'].max().strftime('%Y-%m'))
print('7-Day rows:', (c['Day Time']=='7-Day').sum())
print('features:', len(f), f['Month'].max().strftime('%Y-%m'))
"
```
Expected: clean row count matches the workbook; the feature table's latest month is exactly one month behind the clean data, because `Next_Month_OTP` is a `shift(-1)`. Record all four numbers, they are quoted in Task 5.

- [x] **Step 5: Confirm the feature-engineering tests still pass**

Run: `pytest tests/test_features.py -v`
Expected: all pass. These assert lag and rolling relationships hold on the regenerated artifact, so a pass here proves notebook 03 rebuilt the features correctly on the larger series.

- [x] **Step 6: Commit**

```bash
git add data/raw outputs/predictions/staten_island_otp_features.csv notebooks/01_data_loading_and_initial_review.ipynb notebooks/02_eda.ipynb notebooks/03_feature_engineering.ipynb outputs/figures/otp_trend.png
git commit -m "data: refresh MTA OTP data through <YYYY-MM>

Adds <N> new months from data.ny.gov and rebuilds the cleaned dataset and
feature table. Downstream metrics are regenerated in the following commits."
```

---

### Task 3: Retrain the model and regenerate explainability

**Files:**
- Modify: `models/xgboost_otp_model.pkl` (regenerated by notebook 04)
- Modify: `outputs/figures/shap_bar.png`, `outputs/figures/shap_summary.png` (notebook 05)
- Modify: `outputs/reports/shap_importance.csv` (notebook 05)
- Run: `notebooks/04_model_training.ipynb`, `05_explainability.ipynb`, `06_future_forecasting.ipynb`

**Interfaces:**
- Consumes: refreshed `staten_island_otp_features.csv` from Task 2.
- Produces: retrained model artifact that the dashboard loads and Task 4 evaluates.

- [x] **Step 1: Re-run notebooks 04 to 06**

```bash
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/04_model_training.ipynb \
  notebooks/05_explainability.ipynb \
  notebooks/06_future_forecasting.ipynb
```
Expected: all execute without error; `models/xgboost_otp_model.pkl` is rewritten.

- [x] **Step 2: Verify the model artifact loads and predicts sanely**

Run: `pytest tests/test_model.py -v`
Expected: 3 passed. These load the real artifact and assert predictions stay inside plausible OTP bounds, catching a corrupt or mistrained save.

- [x] **Step 3: Check the SHAP ranking for narrative drift**

```bash
python -c "
import pandas as pd
print(pd.read_csv('outputs/reports/shap_importance.csv').head(6).to_string(index=False))
"
```
Expected: a ranking. Compare against the README's claimed drivers (month/seasonality, delay rate, recent OTP momentum, rolling trends). If the top features have reordered materially, note it now; Task 5 updates the wording.

- [x] **Step 4: Commit**

```bash
git add models/ outputs/figures/shap_bar.png outputs/figures/shap_summary.png outputs/reports/shap_importance.csv notebooks/04_model_training.ipynb notebooks/05_explainability.ipynb notebooks/06_future_forecasting.ipynb
git commit -m "model: retrain XGBoost on refreshed data through <YYYY-MM>"
```

---

### Task 4: Regenerate the Phase 13 to 15 evaluation metrics

**Files:**
- Modify: `outputs/reports/phase13_model_comparison.csv`, `phase14_timeseries_cv_results.csv`, `phase14_timeseries_cv_summary.csv`, `phase15_prediction_interval_summary.csv`
- Modify: `outputs/predictions/phase13_baseline_predictions.csv`, `phase14_cv_predictions.csv`, `phase15_prediction_intervals.csv`
- Modify: the Phase 13 to 15 figures under `outputs/figures/`
- Run: `notebooks/07_baseline_model_comparison.ipynb`, `08_timeseries_cross_validation.ipynb`, `09_prediction_intervals.ipynb`

**Interfaces:**
- Consumes: refreshed feature table and retrained model.
- Produces: the metric CSVs that Task 5 copies into `data.py`, the README, and the report.

- [x] **Step 1: Re-run notebooks 07 to 09**

```bash
jupyter nbconvert --to notebook --execute --inplace \
  notebooks/07_baseline_model_comparison.ipynb \
  notebooks/08_timeseries_cross_validation.ipynb \
  notebooks/09_prediction_intervals.ipynb
```
Expected: all execute. Notebook 07 needs Prophet and statsmodels, which is why `requirements-notebooks.txt` is the environment for this task. Note that its split is `iloc[-6:]`, so the test window shifts automatically to the last six months of the refreshed feature table.

- [x] **Step 2: Capture the new headline numbers**

```bash
python -c "
import pandas as pd
comp = pd.read_csv('outputs/reports/phase13_model_comparison.csv').set_index('Model')
cv   = pd.read_csv('outputs/reports/phase14_timeseries_cv_summary.csv').set_index('Metric')
pi   = pd.read_csv('outputs/reports/phase15_prediction_interval_summary.csv')
xgb  = comp.loc['XGBoost Fair','MAE']
best = comp.drop('XGBoost Fair').sort_values('MAE').iloc[0]
print('XGBoost MAE        ', round(xgb,4))
print('best baseline      ', best.name, round(best['MAE'],4))
print('improvement        ', f\"{(best['MAE']-xgb)/best['MAE']*100:.2f}%\")
print('CV MAE mean/sd     ', round(cv.loc['MAE','Mean'],4), round(cv.loc['MAE','Standard_Deviation'],4))
print('CV R2 mean/sd      ', round(cv.loc['R2','Mean'],4), round(cv.loc['R2','Standard_Deviation'],4))
print('folds with R2 < 0  ', (pd.read_csv('outputs/reports/phase14_timeseries_cv_results.csv')['R2']<0).sum())
print(pi[['Interval','Lower_Residual_Bound','Upper_Residual_Bound','Coverage_Percentage']].to_string(index=False))
"
```
Expected: a complete set of new figures. Record every one; Task 5 writes exactly these values and nothing rounded by hand.

- [x] **Step 3: Sanity-check that XGBoost still wins**

Confirm from Step 2 that `XGBoost Fair` has the lowest MAE. If a baseline now beats it, **stop and report before continuing.** That is a real finding that changes the paper's claim, not a bug to patch over, and the README, report, and abstract all need rethinking rather than a numeric find-and-replace.

- [x] **Step 4: Watch the PI test fail, proving it detects drift**

Run: `pytest tests/test_forecast.py::TestPredictionIntervals::test_pi_constants_match_phase15_artifact -v`
Expected: FAIL, because `app/shared/data.py` still holds the old bounds while the CSV now holds new ones. This failure is the intended signal and Task 5 Step 1 resolves it. If it unexpectedly passes, the interval bounds did not change; confirm notebook 09 actually re-ran.

- [x] **Step 5: Commit**

```bash
git add outputs/ notebooks/07_baseline_model_comparison.ipynb notebooks/08_timeseries_cross_validation.ipynb notebooks/09_prediction_intervals.ipynb
git commit -m "eval: regenerate Phase 13-15 metrics on refreshed data

Test window shifts to <new window>. Headline MAE moves from 1.2087 to
<new>, improvement over the best baseline from 18.7% to <new>%."
```

---

### Task 5: Propagate the new numbers through code and documents

**Files:**
- Modify: `app/shared/data.py:10-11` (PI80, PI90)
- Modify: `app/views/health.py:9` ("241 months")
- Modify: `app/views/research.py:41` ("Aug 2025 – Jan 2026 test window")
- Modify: `README.md` (headline table, limitations, Data section, Phase 13/14/15 tables, data-range note)
- Modify: `reports/final_report.md` (sections 2, 4, 10, 14 numbers)
- Modify: `PROJECT_ROADMAP.md` §1 summary line if it quotes metrics

**Interfaces:**
- Consumes: the metric values captured in Task 4 Step 2.
- Produces: a repository where every quoted number matches `outputs/reports/*.csv`.

- [x] **Step 1: Update the prediction-interval constants**

In `app/shared/data.py`, replace the two constants with the regenerated bounds, keeping the comment pointing at the source CSV:

```python
# Phase 15 residual bounds (percentage points), from outputs/reports/phase15_prediction_interval_summary.csv
PI80 = (<new_lower_80>, <new_upper_80>)
PI90 = (<new_lower_90>, <new_upper_90>)
```

Run: `pytest tests/test_forecast.py -v`
Expected: all pass, including the constant-versus-artifact check that failed in Task 4.

- [x] **Step 2: Update the two hardcoded strings in the app**

`app/views/health.py:9`, replace `241 months` with the new 7-Day row count from Task 2 Step 4.
`app/views/research.py:41`, replace `Aug 2025 – Jan 2026 test window` with the new window from Task 4.

Run: `pytest tests/test_app_pages.py -v`
Expected: 6 passed, confirming every page still renders after the edits.

- [x] **Step 3: Update the README**

Replace in place: the six rows of the Headline Results table, the improvement percentages, the sample size and R² spread in Limitations, the fold-1 and fold-5 MAE values, the row counts and date range in the Data section, the Phase 13/14/15 result tables, and the data-range note near How to Run. Keep the structure; change only numbers, dates, and the window label.

- [x] **Step 4: Update the final report**

Apply the same replacements in `reports/final_report.md`: dataset size and range in section 4, the model table and improvement claim in section 10, the CV and interval numbers, and the sample-size and era-dependence bullets in section 14. If `reports/final_report.pdf` is regenerated from the Markdown, regenerate it; otherwise note in the commit that the PDF is stale.

- [x] **Step 5: Verify every quoted number against the CSVs**

```bash
python - <<'EOF'
import pandas as pd, re, pathlib
comp = pd.read_csv("outputs/reports/phase13_model_comparison.csv").set_index("Model")
cv   = pd.read_csv("outputs/reports/phase14_timeseries_cv_summary.csv").set_index("Metric")
pi   = pd.read_csv("outputs/reports/phase15_prediction_interval_summary.csv")
xgb  = comp.loc["XGBoost Fair","MAE"]
best = comp.drop("XGBoost Fair").sort_values("MAE").iloc[0]
expected = {
    "XGBoost MAE": f"{xgb:.2f}",
    "improvement": f"{(best['MAE']-xgb)/best['MAE']*100:.1f}",
    "CV MAE": f"{cv.loc['MAE','Mean']:.2f}",
    "80% coverage": f"{pi.loc[0,'Coverage_Percentage']:.1f}",
    "90% coverage": f"{pi.loc[1,'Coverage_Percentage']:.1f}",
}
for doc in ["README.md", "reports/final_report.md"]:
    text = pathlib.Path(doc).read_text()
    print(f"--- {doc} ---")
    for label, value in expected.items():
        print(f"  {'OK ' if value in text else 'MISSING'} {label}: {value}")
    for stale in ["1.2087", "18.7", "2.1107", "79.49", "89.74", "1.21 OTP"]:
        if stale in text:
            print(f"  STALE VALUE STILL PRESENT: {stale}")
EOF
```
Expected: every metric marked OK, and no line beginning `STALE VALUE STILL PRESENT`. Any stale hit means a number was missed in Step 3 or 4. Note that a stale string is legitimate only if the refreshed value happens to be identical, which is very unlikely; investigate rather than assume.

- [x] **Step 6: Run the full suite**

Run: `pytest tests/ -q`
Expected: `51 passed`.

- [x] **Step 7: Commit**

```bash
git add app/ README.md reports/ PROJECT_ROADMAP.md
git commit -m "docs: update all metrics for the refreshed dataset

Propagates the regenerated Phase 13-15 numbers into the dashboard
constants, README, and final report so every quoted figure matches
outputs/reports/*.csv."
```

---

### Task 6: Verify the deployed app and freeze for the paper

**Files:**
- Modify: `docs/deployment.md` (add a data-refresh note)
- Modify: `reports/final_report.md` (record the frozen data vintage)

- [x] **Step 1: Push and confirm CI passes**

```bash
git push
```
Then check the run: `curl -s "https://api.github.com/repos/naringrekarchinmay/staten-island-otp-forecasting/actions/runs?per_page=1" | python -c "import json,sys; r=json.load(sys.stdin)['workflow_runs'][0]; print(r['status'], r['conclusion'])"`
Expected: `completed success`.

- [x] **Step 2: Verify the live app serves the refreshed data**

Streamlit Cloud redeploys automatically on push. Wait about two minutes, then confirm the deployed app reflects the new data rather than a cached artifact:

```bash
python -c "
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page()
    pg.goto('https://staten-island-otp-forecasting.streamlit.app/health', wait_until='networkidle', timeout=90000)
    pg.wait_for_timeout(8000)
    print(pg.inner_text('body')[:600])
    b.close()
"
```
Expected: the System Health page shows the new 7-Day month count from Task 5 Step 2, not 241. If it still shows the old value, force a rebuild with Manage app then Reboot.

- [x] **Step 3: Refresh the README screenshots**

```bash
python -c "
from playwright.sync_api import sync_playwright
from PIL import Image
pages = {'dashboard_home': '/', 'dashboard_forecast': '/forecast'}
with sync_playwright() as p:
    b = p.chromium.launch()
    for name, path in pages.items():
        pg = b.new_page(viewport={'width':1600,'height':1000}, device_scale_factor=2)
        pg.goto('https://staten-island-otp-forecasting.streamlit.app'+path, wait_until='networkidle', timeout=90000)
        pg.wait_for_timeout(9000)
        pg.screenshot(path=f'docs/images/{name}.png')
        pg.close()
    b.close()
for name in pages:
    im = Image.open(f'docs/images/{name}.png'); w,h = im.size
    im.resize((1600,int(h*1600/w)), Image.LANCZOS).convert('RGB').save(f'docs/images/{name}.png','PNG',optimize=True)
"
```
Expected: both files rewritten and each well under 1 MB.

- [x] **Step 4: Record the data vintage for the paper**

Add one line to `reports/final_report.md` section 4 stating the exact data vintage, for example: `Data vintage: MTA Socrata dataset fccm-griq, retrieved 2026-08-XX, covering 2006-01 through <YYYY-MM>.` A paper needs the retrieval date because the source is updated in place and is not versioned.

Add to `docs/deployment.md` under a new "Refreshing the data" heading: run `python scripts/fetch_mta_data.py`, re-run notebooks 01 to 09 in order, then follow this plan's Task 5 to propagate numbers.

- [x] **Step 5: Freeze**

Once the paper draft quotes these numbers, do not re-run the refresh until the paper is submitted. A mid-draft refresh silently invalidates every number already written. If new months arrive during drafting, record them as a footnote instead.

- [x] **Step 6: Commit**

```bash
git add docs/ reports/ README.md
git commit -m "docs: record data vintage and refresh procedure"
git push
```

## Self-Review Notes

- **Coverage:** the fetch script replaces the manual download (Task 1); data, model, and metrics are regenerated in dependency order (Tasks 2 to 4); every hardcoded number is updated with a mechanical staleness check (Task 5); deployment and paper-freeze are handled (Task 6).
- **The riskiest step is Task 4 Step 3.** Adding four months moves the six-month test window entirely off the period the current 18.7% claim was measured on. XGBoost winning again is likely but not guaranteed, and the plan deliberately stops rather than papering over a reversal, since the paper's central claim depends on it.
- **Known non-issue:** row ordering. Notebook 03 sorts by `['Day Time','Month']` before building lags, so lag correctness does not depend on workbook order. The categorical ordering in `normalize()` exists only to keep the CSV diff readable.
- **Deferred deliberately:** the scheduled GitHub Action. Task 1 delivers the script it would call, which is the part with real reuse value, without taking on the scheduling, credential, and PR-automation surface during a paper month.
- **Not covered:** `notebooks/10_research_question_methodology.ipynb` is narrative only and writes no artifacts, so it needs no re-run unless the refresh changes the research claim (see Task 4 Step 3).
