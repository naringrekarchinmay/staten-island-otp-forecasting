"""Lag/rolling feature correctness — synthetic mini-series and the shipped artifact."""
import numpy as np
import pandas as pd

from shared import data

# The features artifact is the post-dropna ML frame: the first 6 rows of each
# Day Time group (rolling-6 warm-up) and the final row (Next_Month_OTP target)
# reference months that were dropped, so invariants are asserted on the interior.
WARMUP = 6


def _seven_day(features_df):
    return data.seven_day(features_df).reset_index(drop=True)


class TestSevenDayFilter:
    def test_keeps_only_seven_day_rows_sorted_by_month(self, synthetic_df):
        extra = synthetic_df.copy()
        extra["Day Time"] = "Weekday"
        mixed = pd.concat([extra, synthetic_df.iloc[::-1]], ignore_index=True)

        out = data.seven_day(mixed)

        assert (out["Day Time"] == "7-Day").all()
        assert len(out) == len(synthetic_df)
        assert out["Month"].is_monotonic_increasing

    def test_artifact_seven_day_series_is_monthly_with_no_gaps(self, features_df):
        d7 = _seven_day(features_df)
        gaps = d7["Month"].diff().dropna()
        assert (gaps <= pd.Timedelta(days=31)).all()


class TestLagFeatures:
    def test_synthetic_lags_are_previous_months_otp(self, synthetic_df):
        otp = synthetic_df["On-Time Performance"]
        for k in (1, 2, 3):
            expected = otp.shift(k)
            assert np.allclose(
                synthetic_df[f"OTP_Lag_{k}"], expected, equal_nan=True
            ), f"OTP_Lag_{k} must equal OTP shifted by {k} months"

    def test_artifact_lags_match_shifted_otp(self, features_df):
        d7 = _seven_day(features_df)
        otp = d7["On-Time Performance"]
        for k in (1, 2, 3):
            got = d7[f"OTP_Lag_{k}"].iloc[WARMUP:]
            expected = otp.shift(k).iloc[WARMUP:]
            assert np.allclose(got, expected), f"OTP_Lag_{k} drifted from shift({k})"


class TestRollingFeatures:
    def test_synthetic_rolling3_is_mean_of_prior_three_months(self, synthetic_df):
        otp = synthetic_df["On-Time Performance"]
        expected = otp.shift(1).rolling(3).mean()
        assert np.allclose(synthetic_df["OTP_Rolling_3"], expected, equal_nan=True)

    def test_artifact_rolling_means_exclude_current_month(self, features_df):
        d7 = _seven_day(features_df)
        otp = d7["On-Time Performance"]
        for window in (3, 6):
            got = d7[f"OTP_Rolling_{window}"].iloc[WARMUP:]
            expected = otp.shift(1).rolling(window).mean().iloc[WARMUP:]
            assert np.allclose(got, expected), (
                f"OTP_Rolling_{window} must be the mean of the prior {window} months"
            )


class TestDerivedColumns:
    def test_artifact_delay_rate_is_delayed_over_scheduled(self, features_df):
        d7 = _seven_day(features_df)
        expected = d7["Delayed Trains"] / d7["Scheduled Trips"]
        assert np.allclose(d7["Delay_Rate"], expected)

    def test_artifact_target_is_next_months_otp(self, features_df):
        d7 = _seven_day(features_df)
        got = d7["Next_Month_OTP"].iloc[WARMUP:-1]
        expected = d7["On-Time Performance"].shift(-1).iloc[WARMUP:-1]
        assert np.allclose(got, expected)


class TestEncode:
    def test_encode_returns_exactly_model_feature_columns(self, synthetic_df, stub_model):
        X = data._encode(synthetic_df, stub_model)
        assert list(X.columns) == stub_model.get_booster().feature_names

    def test_encode_fills_absent_model_features_with_zero(self, synthetic_df):
        from conftest import STUB_FEATURES, StubModel

        stub = StubModel(lambda row: 0.9, feature_names=STUB_FEATURES + ["Day Time_Weekend"])
        X = data._encode(synthetic_df, stub)
        assert (X["Day Time_Weekend"] == 0).all()
