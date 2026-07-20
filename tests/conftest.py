"""Shared fixtures: path setup, synthetic mini-series, and a stub model."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
# Views and shared modules import as `from shared import data`, so the app
# directory must be on sys.path exactly as when streamlit runs from repo root.
sys.path.insert(0, str(ROOT / "app"))

STUB_FEATURES = ["OTP_Lag_1", "OTP_Lag_2", "OTP_Lag_3", "OTP_Rolling_3", "Delay_Rate"]


class _StubBooster:
    def __init__(self, names):
        self.feature_names = names


class StubModel:
    """Duck-types the XGBRegressor surface used by app/shared/data.py.

    `predict_fn` receives one encoded row (pd.Series) and returns a float.
    Every frame passed to predict() is recorded in `calls` for inspection.
    """

    def __init__(self, predict_fn, feature_names=None):
        self._fn = predict_fn
        self._names = list(feature_names or STUB_FEATURES)
        self.calls = []

    def get_booster(self):
        return _StubBooster(self._names)

    def predict(self, X):
        self.calls.append(X.copy())
        return np.array([float(self._fn(row)) for _, row in X.iterrows()])


@pytest.fixture
def stub_model():
    """Model whose prediction echoes OTP_Lag_1 + 0.01 — makes feedback visible."""
    return StubModel(lambda row: row["OTP_Lag_1"] + 0.01)


@pytest.fixture
def synthetic_df():
    """Six months of a synthetic 7-Day series with known lag/rolling values."""
    months = pd.date_range("2025-01-01", periods=6, freq="MS")
    otp = [0.90, 0.92, 0.94, 0.96, 0.95, 0.93]
    df = pd.DataFrame(
        {
            "Month": months,
            "Day Time": "7-Day",
            "Season": ["Winter", "Winter", "Spring", "Spring", "Spring", "Summer"],
            "On-Time Performance": otp,
            "Delay_Rate": [0.10, 0.08, 0.06, 0.04, 0.05, 0.07],
            "Delayed Trains": [50, 40, 30, 20, 25, 35],
            "Delayed_Trains_Lag_1": [55, 50, 40, 30, 20, 25],
            "Delayed_Trains_Rolling_3": [50.0, 48.0, 45.0, 40.0, 30.0, 25.0],
        }
    )
    s = pd.Series(otp)
    df["OTP_Lag_1"] = s.shift(1)
    df["OTP_Lag_2"] = s.shift(2)
    df["OTP_Lag_3"] = s.shift(3)
    df["OTP_Rolling_3"] = s.shift(1).rolling(3).mean()
    return df


@pytest.fixture(scope="session")
def features_df():
    from shared import data

    return data.load_features()


@pytest.fixture(scope="session")
def model():
    from shared import data

    return data.load_model()
