"""Recursive forecast mechanics and prediction-interval application."""
import numpy as np
import pandas as pd
import pytest

from shared import data


class TestFutureForecast:
    @pytest.mark.parametrize("months", [1, 3, 6, 12])
    def test_returns_requested_horizon(self, synthetic_df, stub_model, months):
        fc = data.future_forecast(synthetic_df, stub_model, months=months)
        assert len(fc) == months
        assert list(fc.columns) == ["Month", "Forecasted_OTP"]

    def test_forecast_months_continue_from_last_observation(self, synthetic_df, stub_model):
        fc = data.future_forecast(synthetic_df, stub_model, months=4)
        expected = pd.date_range("2025-07-01", periods=4, freq="MS")
        assert list(fc["Month"]) == list(expected)

    def test_predictions_are_fed_back_into_lag_1(self, synthetic_df, stub_model):
        # The stub predicts OTP_Lag_1 + 0.01. Step 1 uses the last feature row
        # unchanged (its OTP_Lag_1 is 0.95); each later step must see the prior
        # prediction in OTP_Lag_1, so the series climbs by exactly 0.01 per step.
        fc = data.future_forecast(synthetic_df, stub_model, months=5)
        expected = 0.95 + 0.01 * np.arange(1, 6)
        assert np.allclose(fc["Forecasted_OTP"], expected)

    def test_lags_shift_down_one_slot_each_step(self, synthetic_df, stub_model):
        data.future_forecast(synthetic_df, stub_model, months=3)
        first, second = stub_model.calls[0].iloc[0], stub_model.calls[1].iloc[0]
        pred_1 = first["OTP_Lag_1"] + 0.01
        assert second["OTP_Lag_1"] == pytest.approx(pred_1)
        assert second["OTP_Lag_2"] == pytest.approx(first["OTP_Lag_1"])
        assert second["OTP_Lag_3"] == pytest.approx(first["OTP_Lag_2"])

    def test_rolling3_is_mean_of_current_lags_each_step(self, synthetic_df, stub_model):
        data.future_forecast(synthetic_df, stub_model, months=3)
        for call in stub_model.calls[1:]:
            row = call.iloc[0]
            lags_mean = np.mean([row["OTP_Lag_1"], row["OTP_Lag_2"], row["OTP_Lag_3"]])
            assert row["OTP_Rolling_3"] == pytest.approx(lags_mean)


class TestScenarioForecast:
    def test_zero_delay_increase_matches_base_forecast(self, synthetic_df, stub_model):
        base = data.future_forecast(synthetic_df, stub_model, months=6)
        scen = data.scenario_forecast(synthetic_df, stub_model, delay_increase=0.0, months=6)
        assert np.allclose(base["Forecasted_OTP"], scen["Forecasted_OTP"])

    def test_delay_columns_are_scaled_by_increase(self, synthetic_df, stub_model):
        data.scenario_forecast(synthetic_df, stub_model, delay_increase=0.20, months=1)
        row = stub_model.calls[0].iloc[0]
        last_delay_rate = synthetic_df["Delay_Rate"].iloc[-1]
        assert row["Delay_Rate"] == pytest.approx(last_delay_rate * 1.20)

    def test_scenario_does_not_mutate_input_frame(self, synthetic_df, stub_model):
        before = synthetic_df["Delay_Rate"].copy()
        data.scenario_forecast(synthetic_df, stub_model, delay_increase=0.5, months=2)
        assert np.allclose(synthetic_df["Delay_Rate"], before)


class TestPredictionIntervals:
    def test_pi_constants_match_phase15_artifact(self):
        summary = data.load_report("phase15_prediction_interval_summary.csv")
        pi80 = summary[summary["Interval"].str.startswith("80%")].iloc[0]
        pi90 = summary[summary["Interval"].str.startswith("90%")].iloc[0]
        assert data.PI80 == (pi80["Lower_Residual_Bound"], pi80["Upper_Residual_Bound"])
        assert data.PI90 == (pi90["Lower_Residual_Bound"], pi90["Upper_Residual_Bound"])

    def test_apply_pi_offsets_forecast_by_residual_bounds(self):
        pct = pd.Series([90.0, 95.0])
        lower, upper = data.apply_pi(pct, data.PI80)
        assert np.allclose(lower, pct + data.PI80[0])
        assert np.allclose(upper, pct + data.PI80[1])

    def test_apply_pi_caps_upper_bound_at_100(self):
        pct = pd.Series([99.5])
        _, upper = data.apply_pi(pct, data.PI80)
        assert upper.iloc[0] == 100.0

    def test_apply_pi_wider_interval_for_higher_confidence(self):
        pct = pd.Series([90.0])
        lo80, up80 = data.apply_pi(pct, data.PI80)
        lo90, up90 = data.apply_pi(pct, data.PI90)
        assert lo90.iloc[0] < lo80.iloc[0]
        assert up90.iloc[0] > up80.iloc[0]
