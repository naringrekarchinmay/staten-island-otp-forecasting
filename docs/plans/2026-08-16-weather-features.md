# Weather Feature Enhancement (Phase 17) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. One task per session unless the user says continue.

**Goal:** Test whether monthly weather features (snowfall, precipitation, temperature, wind) improve the next-month OTP forecast over the current internal-features-only model, using the same rolling-origin protocol and paired significance tests that Phase 22 established, with February 2026's snowstorm as the motivating validation case.

**Architecture:** Weather is added as an augmentation, not a rewrite. A keyless fetch script pulls daily weather for Staten Island from the Open-Meteo historical archive and aggregates it to a monthly table. That table joins onto the existing 7-Day feature series in two clearly separated framings (operational and oracle, defined below). A controlled experiment reuses `scripts/rolling_origin_eval.py` to compare the base XGBoost against weather-augmented variants, paired by month. The frozen base pipeline and the Phase 22 result are left untouched, so the comparison is clean.

**Tech Stack:** Python 3.13, pandas, requests, xgboost, scipy, matplotlib, existing `scripts/rolling_origin_eval.py`, pytest. Runs in the `requirements-notebooks.txt` environment.

## Global Constraints

- **The operational / oracle distinction is the core of this phase and must never be blurred.** At forecast time for month *t + 1*, next month's weather is not known. Two variants answer two different questions, and every table, figure, and sentence must label which one it is:
  - **Operational:** attach the weather of the feature month *t* (known when the forecast is made) to the row predicting *t + 1*. This is a real forecast and tests whether recent weather carries predictive signal forward.
  - **Oracle (explanatory):** attach the weather of the target month *t + 1*. This is **not a forecast** (it assumes perfect foreknowledge of next month's weather); it measures how much of the OTP the base model misses is explainable by weather. The February 2026 validation case is an oracle test: given that a 30 cm snow month occurred, could the model have predicted the 94% drop?
- **Weather source:** Open-Meteo historical archive (`https://archive-api.open-meteo.com/v1/archive`), ERA5 reanalysis, keyless. Verified 2026-08-16 to cover 2006-01 through 2026-06 for Staten Island and to capture the February 2026 storm (30.4 cm monthly snowfall, 17.85 cm on 2026-02-23). Location: Staten Island centroid, latitude 40.58, longitude -74.13. Freeze the weather data at the same vintage as the MTA data (through 2026-06); see `docs/deployment.md` and the data-refresh freeze.
- **Do not modify the base feature pipeline.** Notebooks 01 to 09, `outputs/predictions/staten_island_otp_features.csv`, the model artifact, and the Phase 13 to 15 and Phase 22 outputs stay as they are. Weather features live in new files. The base XGBoost in the comparison must use exactly the Phase 22 feature set so that any difference is attributable to weather alone.
- **Reuse the Phase 22 harness.** Import `rolling_origins`, `xgboost_forecast`, `summarize`, and `paired_comparison` from `scripts/rolling_origin_eval.py` rather than reimplementing them. The comparison must be paired by target month and use the Wilcoxon signed-rank test, consistent with Phase 22.
- **A negative result is a valid result.** If weather does not significantly improve the operational forecast, report that plainly. The oracle result is expected to help (that is nearly tautological for shock months); the operational result is the real open question. Do not present an oracle improvement as if it were an operational one.
- **Every reported number must be reproduced from a committed CSV**, checked mechanically as in the data-refresh plan.

---

### Task 1: Weather data fetch script

**Files:**
- Create: `scripts/fetch_weather_data.py`
- Create: `tests/test_weather_fetch.py`
- Create (output): `data/raw/staten_island_weather_monthly.csv`

**Interfaces:**
- Produces: `aggregate_monthly(daily: pd.DataFrame) -> pd.DataFrame` mapping daily records to one row per month with the columns in `WEATHER_COLUMNS`.
- Produces: `fetch(start="2006-01-01", end="2026-06-30") -> pd.DataFrame` and `main()` writing the monthly CSV.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_weather_fetch.py
"""Monthly aggregation of daily Staten Island weather."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.fetch_weather_data import WEATHER_COLUMNS, aggregate_monthly


def _daily(dates, snow=0.0, precip=0.0, tmean=5.0, tmin=1.0, wind=10.0):
    n = len(dates)
    def col(v): return v if isinstance(v, list) else [v] * n
    return pd.DataFrame({
        "time": pd.to_datetime(dates),
        "snowfall_sum": col(snow), "precipitation_sum": col(precip),
        "temperature_2m_mean": col(tmean), "temperature_2m_min": col(tmin),
        "windspeed_10m_max": col(wind),
    })


def test_one_row_per_month():
    daily = _daily(pd.date_range("2026-01-01", "2026-02-28", freq="D"))
    out = aggregate_monthly(daily)
    assert list(out["Month"]) == [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01")]


def test_snowfall_is_summed_and_peak_is_max():
    daily = _daily(["2026-02-22", "2026-02-23"], snow=[8.26, 17.85])
    out = aggregate_monthly(daily).iloc[0]
    assert out["total_snowfall_cm"] == pytest.approx(26.11)
    assert out["max_daily_snowfall_cm"] == pytest.approx(17.85)


def test_snow_days_counts_days_above_zero():
    daily = _daily(["2026-02-01", "2026-02-02", "2026-02-03"], snow=[0.0, 2.0, 5.0])
    assert aggregate_monthly(daily).iloc[0]["snow_days"] == 2


def test_mean_temperature_is_averaged():
    daily = _daily(["2026-01-01", "2026-01-02"], tmean=[-2.0, 4.0])
    assert aggregate_monthly(daily).iloc[0]["mean_temp_c"] == pytest.approx(1.0)


def test_freezing_days_count_days_below_zero_min():
    daily = _daily(["2026-01-01", "2026-01-02", "2026-01-03"], tmin=[-1.0, 0.5, -3.0])
    assert aggregate_monthly(daily).iloc[0]["freezing_days"] == 2


def test_output_columns_are_exactly_the_declared_schema():
    out = aggregate_monthly(_daily(["2026-01-01"]))
    assert list(out.columns) == ["Month"] + WEATHER_COLUMNS


def test_missing_daily_column_raises():
    daily = _daily(["2026-01-01"]).drop(columns=["snowfall_sum"])
    with pytest.raises(ValueError, match="snowfall_sum"):
        aggregate_monthly(daily)
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_weather_fetch.py -v`
Expected: FAIL at import, `ModuleNotFoundError: No module named 'scripts.fetch_weather_data'`.

- [x] **Step 3: Write the script**

```python
# scripts/fetch_weather_data.py
"""Download daily Staten Island weather and aggregate it to a monthly table.

Source: Open-Meteo historical archive (ERA5 reanalysis), keyless. Written to
mirror scripts/fetch_mta_data.py: a reproducible pull to a committed CSV, so
the weather join has the same provenance guarantees as the OTP data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
# Staten Island centroid. Monthly weather is near-uniform across the island,
# so one representative point is sufficient (noted as a limitation).
LATITUDE, LONGITUDE = 40.58, -74.13
DAILY_VARS = ["snowfall_sum", "precipitation_sum", "temperature_2m_mean",
              "temperature_2m_min", "windspeed_10m_max"]
OUT_CSV = (Path(__file__).resolve().parents[1]
           / "data/raw/staten_island_weather_monthly.csv")

WEATHER_COLUMNS = [
    "total_snowfall_cm", "max_daily_snowfall_cm", "snow_days",
    "total_precip_mm", "mean_temp_c", "min_temp_c",
    "freezing_days", "max_wind_kmh",
]


def aggregate_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    required = {"time", *DAILY_VARS}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"daily frame missing columns: {sorted(missing)}")
    d = daily.copy()
    d["time"] = pd.to_datetime(d["time"])
    d["Month"] = d["time"].dt.to_period("M").dt.to_timestamp()
    g = d.groupby("Month")
    out = pd.DataFrame({
        "total_snowfall_cm": g["snowfall_sum"].sum(),
        "max_daily_snowfall_cm": g["snowfall_sum"].max(),
        "snow_days": g["snowfall_sum"].apply(lambda s: int((s > 0).sum())),
        "total_precip_mm": g["precipitation_sum"].sum(),
        "mean_temp_c": g["temperature_2m_mean"].mean(),
        "min_temp_c": g["temperature_2m_min"].min(),
        "freezing_days": g["temperature_2m_min"].apply(lambda s: int((s < 0).sum())),
        "max_wind_kmh": g["windspeed_10m_max"].max(),
    }).reset_index()
    return out[["Month"] + WEATHER_COLUMNS]


def fetch(start: str = "2006-01-01", end: str = "2026-06-30") -> pd.DataFrame:
    resp = requests.get(ARCHIVE_URL, params={
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "start_date": start, "end_date": end,
        "daily": ",".join(DAILY_VARS), "timezone": "America/New_York"},
        timeout=90)
    resp.raise_for_status()
    daily = pd.DataFrame(resp.json()["daily"])
    return aggregate_monthly(daily)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    parser.add_argument("--end", default="2026-06-30",
                        help="freeze date; keep at the MTA data vintage")
    args = parser.parse_args()
    monthly = fetch(end=args.end)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(args.out, index=False)
    print(f"wrote {len(monthly)} months to {args.out} "
          f"({monthly['Month'].min():%Y-%m} to {monthly['Month'].max():%Y-%m})")
    feb = monthly[monthly["Month"] == "2026-02-01"]
    if not feb.empty:
        print(f"Feb 2026 sanity: {feb['total_snowfall_cm'].iloc[0]:.1f} cm snowfall, "
              f"peak day {feb['max_daily_snowfall_cm'].iloc[0]:.1f} cm")


if __name__ == "__main__":
    main()
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_weather_fetch.py -v`
Expected: 7 passed.

- [x] **Step 5: Fetch the data and sanity-check the storm**

Run: `python scripts/fetch_weather_data.py`
Expected: 246 months, 2006-01 to 2026-06, and the printed Feb 2026 line shows roughly 30 cm total with a ~18 cm peak day. If Feb 2026 snowfall is near zero, the location or date handling is wrong; stop and fix before continuing.

- [x] **Step 6: Confirm the full suite is green and commit**

Run: `pytest tests/ -q` (expect all prior tests plus 7 new).

```bash
git add scripts/fetch_weather_data.py tests/test_weather_fetch.py data/raw/staten_island_weather_monthly.csv
git commit -m "feat: add Staten Island monthly weather fetch (Open-Meteo)

Keyless pull of daily weather aggregated to monthly snowfall, precipitation,
temperature, and wind features, 2006-01 to 2026-06. Frozen at the MTA data
vintage. Feb 2026 captures the 30 cm snow month that the OTP model misses."
```

---

### Task 2: Weather-augmented feature tables and EDA

**Files:**
- Create: `notebooks/11_weather_features.ipynb`
- Create (output): `outputs/predictions/otp_weather_operational.csv`, `outputs/predictions/otp_weather_oracle.csv`
- Create (output): `outputs/figures/weather_otp_correlation.png`

**Interfaces:**
- Consumes: `outputs/predictions/staten_island_otp_features.csv` (7-Day rows) and `data/raw/staten_island_weather_monthly.csv`.
- Produces: two feature tables identical to the base 7-Day feature set plus the eight `WEATHER_COLUMNS`, one joined operationally (weather of month *t*) and one joined as oracle (weather of month *t + 1*). Task 3 reads both.

- [x] **Step 1: Build the two joins in the notebook**

Load the 7-Day feature rows and the monthly weather. For each row whose `Month` is *t* and whose target `Next_Month_OTP` is the OTP of *t + 1*:
- Operational table: left-join weather on `Month == t`.
- Oracle table: left-join weather on `Month == t + 1` (compute `target_month = Month + MonthBegin(1)` and join weather on that).

Write both to the CSVs above. Assert that neither introduces NaNs in the weather columns on the interior rows (the first/last month may be unavoidable and should be documented, not filled).

- [x] **Step 2: Verify the Feb 2026 alignment explicitly**

In the oracle table, the row with `target_month == 2026-02-01` (i.e. `Month == 2026-01-01`) must carry February's ~30 cm `total_snowfall_cm`. In the operational table, the row with `Month == 2026-02-01` carries February's weather. Print both rows and confirm the snowfall lands where intended. This is the single most important correctness check in the phase: getting the offset backwards would silently invert the whole result.

```python
oracle = pd.read_csv("../outputs/predictions/otp_weather_oracle.csv", parse_dates=["Month"])
row = oracle[oracle["Month"] == "2026-01-01"]
assert row["total_snowfall_cm"].iloc[0] > 25, "oracle join misaligned"
```

- [x] **Step 3: EDA on weather vs OTP**

Correlate each weather feature with same-month OTP (contemporaneous) and with next-month OTP. Save a correlation bar chart to `weather_otp_correlation.png`. Expect snowfall and freezing-days to correlate negatively with same-month OTP and weakly or not at all with next-month OTP; that contrast is the operational-versus-oracle story in one figure.

- [x] **Step 4: Commit**

```bash
git add notebooks/11_weather_features.ipynb outputs/predictions/otp_weather_operational.csv outputs/predictions/otp_weather_oracle.csv outputs/figures/weather_otp_correlation.png
git commit -m "feat: join weather onto the OTP feature series (operational and oracle)

Two feature tables augmenting the 7-Day base features with monthly weather:
operational (weather of the feature month, a real forecast) and oracle
(weather of the target month, an explanatory upper bound). Verified the
Feb 2026 snowfall lands on the correct row in both."
```

---

### Task 3: Controlled rolling-origin comparison

**Files:**
- Create: `scripts/weather_experiment.py`
- Create: `tests/test_weather_experiment.py`
- Create (output): `outputs/reports/phase23_weather_comparison.csv`, `outputs/reports/phase23_weather_paired.csv`
- Create (output): `outputs/reports/phase23_shock_months.csv`, `outputs/figures/phase23_weather_errors.png`

**Interfaces:**
- Consumes: the two weather feature tables and the Phase 22 base feature set.
- Reuses: `rolling_origins`, `xgboost_forecast`, `summarize`, `paired_comparison` from `scripts/rolling_origin_eval.py`.
- Produces: rolling-origin errors for three models on identical origins: `XGBoost` (base, Phase 22 features), `XGBoost+Weather (operational)`, `XGBoost+Weather (oracle)`.

- [ ] **Step 1: Write the failing tests**

Test the two behaviours specific to this experiment (origin alignment across the three feature matrices, and a shock-month filter), reusing the harness for everything else.

```python
# tests/test_weather_experiment.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.weather_experiment import shock_months, align_matrices


def test_shock_months_selects_high_snow_targets():
    weather = pd.DataFrame({
        "Month": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"]),
        "total_snowfall_cm": [0.0, 30.4, 1.0],
    })
    assert shock_months(weather, threshold_cm=15) == [pd.Timestamp("2026-02-01")]


def test_align_matrices_share_one_row_index_per_month():
    idx = pd.date_range("2020-01-01", periods=5, freq="MS")
    base = pd.DataFrame({"Month": idx, "f": range(5)})
    weather = pd.DataFrame({"Month": idx, "f": range(5), "snow": range(5)})
    b, w, months = align_matrices(base, weather)
    assert list(months) == list(idx)
    assert len(b) == len(w) == 5
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_weather_experiment.py -v`
Expected: FAIL at import.

- [ ] **Step 3: Write the experiment**

Build three aligned feature matrices over the shared 7-Day months. At each rolling origin (reuse `rolling_origins` with the same `min_train=120` as Phase 22), train each variant on rows before the origin and predict the origin month, so the only difference between variants is the presence of the weather columns. Collect predictions long-format with a `model` column, then call `summarize` and `paired_comparison(reference="XGBoost")`. Additionally, compute the same metrics restricted to `shock_months` (target months with `total_snowfall_cm` above a threshold, e.g. 15 cm) and write them to `phase23_shock_months.csv`. Save an errors-over-time and boxplot figure as in Phase 22.

`shock_months(weather, threshold_cm)` returns the sorted list of target months above the snowfall threshold. `align_matrices(base, weather)` inner-joins on `Month` and returns the two feature frames plus the shared month index.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_weather_experiment.py -v` then `pytest tests/ -q`.
Expected: all green.

- [ ] **Step 5: Run the experiment and read the result**

Run: `python scripts/weather_experiment.py`
Capture, for each variant, overall MAE and the paired Wilcoxon p-value against base XGBoost, plus the same on shock months only. Interpretation gates:
- If **operational** weather significantly beats base overall (p < 0.05 after considering the two comparisons), weather has real forecasting value: this is a positive, publishable operational result.
- If operational does not beat base overall but the **oracle** variant sharply reduces error on shock months (expected for Feb 2026), the finding is that weather explains the misses but is only useful given a weather forecast: state exactly that, and frame operational value as contingent on forecast weather inputs.
- If neither helps, report the negative result. That would mean monthly weather aggregates do not carry enough signal at this horizon, which is itself worth stating.

Do not tune features or thresholds to manufacture significance. If the first feature set is not significant, that is the result; any additional feature engineering must be pre-declared here as a follow-up, not a silent retrofit.

- [ ] **Step 6: Commit**

```bash
git add scripts/weather_experiment.py tests/test_weather_experiment.py outputs/reports/phase23_*.csv outputs/figures/phase23_weather_errors.png
git commit -m "feat: rolling-origin weather experiment (base vs operational vs oracle)

Compares the base XGBoost against weather-augmented variants on identical
origins, paired by month, reusing the Phase 22 harness. Reports overall and
shock-month results. <one line stating the actual finding>."
```

---

### Task 4: SHAP, dashboard, and write-up

**Files:**
- Modify: `notebooks/11_weather_features.ipynb` (append SHAP on the weather model)
- Create (output): `outputs/figures/phase23_weather_shap.png`, `outputs/reports/phase23_weather_shap_importance.csv`
- Modify: `README.md` (a Phase 23 section under the results), `app/views/research.py` (a weather tab, only if the result warrants surfacing)
- Modify: `docs/plans/2026-08-16-weather-features.md` (record the outcome)

**Interfaces:**
- Consumes: the Phase 23 outputs and the weather feature tables.

- [ ] **Step 1: SHAP on the weather model**

Fit the oracle weather XGBoost on the full series, compute SHAP, and confirm whether snowfall and freezing-days rank among the influential features. Save the bar plot and an importance CSV. On the February 2026 row specifically, produce a SHAP explanation showing snowfall pushing the prediction down; this is the figure that makes the weather argument concrete.

- [ ] **Step 2: Verify every number against the CSVs**

Mechanically check that each figure quoted in the README matches `outputs/reports/phase23_*.csv`, the same guard used in the data-refresh plan. No hand-entered numbers.

- [ ] **Step 3: Write the README section**

Add a Phase 23 section reporting the operational and oracle results with their paired p-values and the shock-month table, in the honest framing the result supports. If operational weather did not help, say so directly; the paper's contribution is the controlled test, not a guaranteed win. Update the limitations note about "no external drivers" to reflect what was actually found.

- [ ] **Step 4: Dashboard (conditional)**

Only if the operational result is significant, add a weather tab to the Research page in the existing style. If the result is oracle-only or negative, leave the dashboard as is and keep the finding in the README and report; do not imply an operational gain the data does not support.

- [ ] **Step 5: Record the outcome and commit**

Fill in the outcome at the top of this plan's self-review, then commit README, notebook, figures, and any dashboard change. Push and confirm CI is green.

## Self-Review Notes

- **Outcome (fill in on completion):** <operational significant? oracle shock-month effect? negative?>
- **The one correctness risk that matters:** the target-month offset in the oracle join (Task 2 Step 2). Getting it backwards would attach the wrong month's weather and silently produce a wrong conclusion in either direction; the explicit Feb 2026 assertion is the guard.
- **Leakage:** the oracle variant intentionally uses the target month's weather. This is sound because weather is exogenous to OTP and no OTP value from the future is used; the framing (not a real forecast) is what keeps it honest. The operational variant has no such subtlety.
- **Frozen data:** weather is frozen at the 2026-06 MTA vintage. Do not extend the weather series past the OTP series, or the join gains target months with no OTP.
- **Reuse:** the comparison deliberately reuses `scripts/rolling_origin_eval.py` so the base XGBoost here is byte-for-byte the Phase 22 model on the Phase 22 features; only the added weather columns differ between variants.
- **Deferred:** a station-level or multi-point weather source, sub-monthly weather alignment, and any weather-conditioned prediction intervals are out of scope for this phase.
