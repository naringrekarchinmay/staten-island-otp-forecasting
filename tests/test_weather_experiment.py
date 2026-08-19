"""Origin alignment and shock-month selection for the weather experiment."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.weather_experiment import align_matrices, shock_months


def test_shock_months_selects_high_snow_targets():
    weather = pd.DataFrame({
        "Month": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"]),
        "total_snowfall_cm": [0.0, 30.4, 1.0],
    })
    assert shock_months(weather, threshold_cm=15) == [pd.Timestamp("2026-02-01")]


def test_shock_months_returns_sorted_and_empty_when_none_exceed():
    weather = pd.DataFrame({
        "Month": pd.to_datetime(["2026-03-01", "2026-01-01", "2026-02-01"]),
        "total_snowfall_cm": [20.0, 40.0, 5.0],
    })
    assert shock_months(weather, threshold_cm=15) == [
        pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-01")]
    assert shock_months(weather, threshold_cm=100) == []


def test_align_matrices_share_one_row_index_per_month():
    idx = pd.date_range("2020-01-01", periods=5, freq="MS")
    base = pd.DataFrame({"Month": idx, "f": range(5)})
    weather = pd.DataFrame({"Month": idx, "f": range(5), "snow": range(5)})
    b, w, months = align_matrices(base, weather)
    assert list(months) == list(idx)
    assert len(b) == len(w) == 5


def test_align_matrices_keeps_only_shared_months():
    base = pd.DataFrame({"Month": pd.date_range("2020-01-01", periods=4, freq="MS"), "f": range(4)})
    weather = pd.DataFrame({"Month": pd.date_range("2020-02-01", periods=4, freq="MS"),
                            "f": range(4), "snow": range(4)})
    b, w, months = align_matrices(base, weather)
    assert list(months) == list(pd.date_range("2020-02-01", periods=3, freq="MS"))
    assert len(b) == len(w) == 3
