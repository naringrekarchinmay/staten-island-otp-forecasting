"""The shipped model artifact loads and predicts sane OTP values."""
import numpy as np

from shared import data


class TestModelArtifact:
    def test_model_loads_as_xgboost_regressor(self, model):
        assert type(model).__name__ == "XGBRegressor"
        assert len(model.get_booster().feature_names) > 0

    def test_predictions_on_historical_features_stay_in_otp_range(self, features_df, model):
        d7 = data.seven_day(features_df)
        preds = data.predict(d7, model)
        # OTP is a 0–1 fraction; regression output may overshoot marginally.
        assert np.all(preds > 0.5), "predicted OTP below any historically observed level"
        assert np.all(preds < 1.05), "predicted OTP far above the 100% ceiling"

    def test_recursive_forecast_with_real_model_is_sane(self, features_df, model):
        d7 = data.seven_day(features_df)
        fc = data.future_forecast(d7, model, months=6)
        assert len(fc) == 6
        assert fc["Forecasted_OTP"].between(0.5, 1.05).all()
        assert fc["Month"].iloc[0] > d7["Month"].max()
