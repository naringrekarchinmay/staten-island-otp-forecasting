"""Rolling-origin evaluation: origin generation, forecasters, and scoring.

The protocol under test gives every model the same task: at each origin the
model sees OTP up to and including month t and predicts month t+1.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.rolling_origin_eval import (
    moving_average_forecast,
    naive_forecast,
    rolling_origins,
    summarize,
)


class TestRollingOrigins:
    def test_first_origin_respects_minimum_training_size(self):
        assert list(rolling_origins(n_obs=10, min_train=6))[0] == 6

    def test_every_origin_leaves_a_target_month_available(self):
        origins = list(rolling_origins(n_obs=10, min_train=6))
        # Target for origin i is index i, so i must be a valid index.
        assert max(origins) == 9
        assert all(0 <= i < 10 for i in origins)

    def test_origin_count_is_observations_minus_minimum_training(self):
        assert len(list(rolling_origins(n_obs=100, min_train=60))) == 40

    def test_no_origins_when_history_is_too_short(self):
        assert list(rolling_origins(n_obs=5, min_train=5)) == []


class TestNaiveForecast:
    def test_predicts_the_most_recent_observation(self):
        assert naive_forecast(np.array([90.0, 92.0, 95.5])) == 95.5

    def test_uses_only_the_history_it_is_given(self):
        history = np.array([90.0, 92.0, 95.5])
        naive_forecast(history)
        assert list(history) == [90.0, 92.0, 95.5]


class TestMovingAverageForecast:
    def test_averages_the_trailing_window(self):
        history = np.array([90.0, 93.0, 96.0, 99.0])
        assert moving_average_forecast(history, window=3) == pytest.approx(96.0)

    def test_window_longer_than_history_uses_all_of_it(self):
        history = np.array([90.0, 94.0])
        assert moving_average_forecast(history, window=6) == pytest.approx(92.0)

    def test_window_of_one_matches_the_naive_forecast(self):
        history = np.array([90.0, 93.0, 97.0])
        assert moving_average_forecast(history, window=1) == naive_forecast(history)


class TestSummarize:
    def _predictions(self):
        # Model A is exactly 1.0 off every month; model B alternates 0 and 2.
        return pd.DataFrame({
            "target_month": pd.to_datetime(["2025-01-01", "2025-02-01"] * 2),
            "model": ["A", "A", "B", "B"],
            "actual": [95.0, 96.0, 95.0, 96.0],
            "predicted": [94.0, 97.0, 95.0, 94.0],
        })

    def test_computes_mae_per_model(self):
        out = summarize(self._predictions()).set_index("Model")
        assert out.loc["A", "MAE"] == pytest.approx(1.0)
        assert out.loc["B", "MAE"] == pytest.approx(1.0)

    def test_rmse_penalises_the_larger_single_error(self):
        out = summarize(self._predictions()).set_index("Model")
        # Same MAE, but B's errors are 0 and 2, so its RMSE is higher.
        assert out.loc["B", "RMSE"] > out.loc["A", "RMSE"]

    def test_reports_the_number_of_origins_scored(self):
        out = summarize(self._predictions()).set_index("Model")
        assert out.loc["A", "N"] == 2

    def test_orders_models_best_mae_first(self):
        preds = self._predictions()
        preds.loc[preds["model"] == "A", "predicted"] = [95.0, 96.0]  # A now perfect
        out = summarize(preds)
        assert list(out["Model"])[0] == "A"
        assert out.loc[0, "MAE"] == pytest.approx(0.0)


class TestNoLookahead:
    def test_forecasters_never_receive_the_target_value(self):
        """The history passed at origin i must end before the target month."""
        series = np.arange(100.0)
        for i in rolling_origins(n_obs=len(series), min_train=10):
            history = series[:i]
            target = series[i]
            assert target not in history
            assert len(history) == i
            assert history[-1] == series[i - 1]


class TestPairedComparison:
    """A ranking claim needs a paired test, not just a lower mean."""

    def _preds(self, ref_errors, other_errors):
        months = pd.date_range("2020-01-01", periods=len(ref_errors), freq="MS")
        rows = []
        for m, a, b in zip(months, ref_errors, other_errors):
            rows.append({"target_month": m, "model": "Ref", "actual": 100.0,
                         "predicted": 100.0 - a})
            rows.append({"target_month": m, "model": "Other", "actual": 100.0,
                         "predicted": 100.0 - b})
        return pd.DataFrame(rows)

    def test_reports_mean_absolute_error_difference(self):
        from scripts.rolling_origin_eval import paired_comparison
        # Ref is off by 1 every month, Other by 3: Other is 2.0 worse.
        preds = self._preds([1.0] * 12, [3.0] * 12)
        out = paired_comparison(preds, reference="Ref").set_index("Model")
        assert out.loc["Other", "MeanAE_Diff"] == pytest.approx(2.0)

    def test_negative_difference_means_the_baseline_is_better(self):
        from scripts.rolling_origin_eval import paired_comparison
        preds = self._preds([4.0] * 12, [1.0] * 12)
        out = paired_comparison(preds, reference="Ref").set_index("Model")
        assert out.loc["Other", "MeanAE_Diff"] < 0

    def test_consistent_gap_is_flagged_significant(self):
        from scripts.rolling_origin_eval import paired_comparison
        preds = self._preds([1.0] * 20, [3.0] * 20)
        out = paired_comparison(preds, reference="Ref").set_index("Model")
        assert out.loc["Other", "p_value"] < 0.05

    def test_noise_around_zero_is_not_flagged_significant(self):
        from scripts.rolling_origin_eval import paired_comparison
        rng = np.random.default_rng(0)
        base = rng.uniform(0.5, 3.0, 30)
        jitter = base + rng.normal(0, 0.01, 30) * np.where(np.arange(30) % 2, 1, -1)
        preds = self._preds(base, jitter)
        out = paired_comparison(preds, reference="Ref").set_index("Model")
        assert out.loc["Other", "p_value"] > 0.05

    def test_reference_model_is_excluded_from_its_own_comparison(self):
        from scripts.rolling_origin_eval import paired_comparison
        preds = self._preds([1.0] * 12, [2.0] * 12)
        assert "Ref" not in list(paired_comparison(preds, reference="Ref")["Model"])
