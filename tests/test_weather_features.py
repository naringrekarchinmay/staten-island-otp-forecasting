"""Joining monthly weather onto the OTP feature series, operational vs oracle.

The offset between the two framings is the load-bearing correctness point of
the weather phase: operational attaches the feature month's weather, oracle
attaches the target (next) month's weather.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.fetch_weather_data import WEATHER_COLUMNS
from scripts.weather_features import join_weather


def _features(months):
    m = pd.to_datetime(months)
    return pd.DataFrame({"Month": m, "Day Time": "7-Day",
                         "On-Time Performance": np.linspace(0.95, 0.97, len(m))})


def _weather(months, snow):
    m = pd.to_datetime(months)
    df = pd.DataFrame({"Month": m})
    for c in WEATHER_COLUMNS:
        df[c] = 0.0
    df["total_snowfall_cm"] = snow
    return df


def test_operational_attaches_the_feature_months_weather():
    feats = _features(["2026-01-01", "2026-02-01"])
    weather = _weather(["2026-01-01", "2026-02-01"], snow=[1.0, 30.0])
    out = join_weather(feats, weather, "operational").set_index("Month")
    assert out.loc["2026-01-01", "total_snowfall_cm"] == 1.0
    assert out.loc["2026-02-01", "total_snowfall_cm"] == 30.0


def test_oracle_attaches_the_target_months_weather():
    feats = _features(["2026-01-01", "2026-02-01"])
    weather = _weather(["2026-01-01", "2026-02-01", "2026-03-01"], snow=[1.0, 30.0, 2.0])
    out = join_weather(feats, weather, "oracle").set_index("Month")
    # The row for month t carries the weather of t+1.
    assert out.loc["2026-01-01", "total_snowfall_cm"] == 30.0  # Feb's snow
    assert out.loc["2026-02-01", "total_snowfall_cm"] == 2.0   # March's snow


def test_feb_2026_lands_on_the_right_row_in_each_framing():
    feats = _features(["2026-01-01", "2026-02-01"])
    weather = _weather(["2026-01-01", "2026-02-01"], snow=[1.0, 30.0])
    op = join_weather(feats, weather, "operational").set_index("Month")
    orc = join_weather(feats, weather, "oracle").set_index("Month")
    # Operational: the snowy month is February's own row.
    assert op.loc["2026-02-01", "total_snowfall_cm"] == 30.0
    # Oracle: February's snow attaches to January's row (which predicts Feb).
    assert orc.loc["2026-01-01", "total_snowfall_cm"] == 30.0


def test_output_is_base_columns_plus_weather_columns():
    feats = _features(["2026-01-01"])
    weather = _weather(["2026-01-01"], snow=[1.0])
    out = join_weather(feats, weather, "operational")
    assert list(out.columns) == list(feats.columns) + WEATHER_COLUMNS


def test_unknown_framing_raises():
    feats = _features(["2026-01-01"])
    weather = _weather(["2026-01-01"], snow=[1.0])
    with pytest.raises(ValueError, match="framing"):
        join_weather(feats, weather, "sideways")


def test_missing_weather_column_raises():
    feats = _features(["2026-01-01"])
    weather = _weather(["2026-01-01"], snow=[1.0]).drop(columns=["mean_temp_c"])
    with pytest.raises(ValueError, match="mean_temp_c"):
        join_weather(feats, weather, "operational")
