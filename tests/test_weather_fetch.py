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
