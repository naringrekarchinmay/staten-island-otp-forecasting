"""Rolling-origin evaluation of the OTP forecasters.

Phase 13 scored every model on a single six-month hold-out. Six points
cannot separate six models, and the ranking proved unstable: XGBoost led by
18.7% on data through 2026-01 and trailed Prophet by 35% once five more
months arrived. This module replaces that single window with a rolling
origin: at each month t a model sees on-time performance up to and
including t and predicts t+1, scored over every origin with enough history.

It also removes an inconsistency in the Phase 13 protocol. There the naive
and moving-average baselines advanced month by month on true history and
XGBoost received each test month's real lag features, while SARIMA and
Prophet forecast six months ahead from the training cut without ever seeing
an actual. Those are different tasks. Every model here performs the same
one-step-ahead forecast from the same information.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "outputs/predictions/staten_island_otp_features.csv"
REPORT_DIR = ROOT / "outputs/reports"
PREDICTION_DIR = ROOT / "outputs/predictions"

# Ten years of monthly history before the first origin, so early folds are
# not scored on a model that has barely seen a seasonal cycle.
DEFAULT_MIN_TRAIN = 120


def rolling_origins(n_obs: int, min_train: int):
    """Yield each index whose value is forecast from the values before it."""
    return range(min_train, n_obs)


def naive_forecast(history: np.ndarray) -> float:
    """Carry the most recent observation forward."""
    return float(history[-1])


def moving_average_forecast(history: np.ndarray, window: int) -> float:
    """Average the trailing `window` observations."""
    return float(np.mean(history[-window:]))


def sarima_forecast(history: np.ndarray, months: pd.DatetimeIndex) -> float:
    """One-step SARIMA(1,1,1)(1,1,1,12) forecast, matching notebook 07."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    series = pd.Series(history, index=months).asfreq("MS").interpolate()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                      enforce_stationarity=False, enforce_invertibility=False
                      ).fit(disp=False)
        return float(fit.forecast(steps=1).iloc[0])


def prophet_forecast(history: np.ndarray, months: pd.DatetimeIndex) -> float:
    """One-step Prophet forecast with yearly seasonality, matching notebook 07."""
    from prophet import Prophet

    frame = pd.DataFrame({"ds": months, "y": history})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                        daily_seasonality=False)
        model.fit(frame)
        future = model.make_future_dataframe(periods=1, freq="MS")
        return float(model.predict(future)["yhat"].iloc[-1])


def xgboost_forecast(X_train: pd.DataFrame, y_train: pd.Series,
                     X_next: pd.DataFrame) -> float:
    """One-step XGBoost forecast, hyperparameters matching notebook 07."""
    from xgboost import XGBRegressor

    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=3,
                         subsample=0.8, colsample_bytree=0.8, random_state=42,
                         objective="reg:squarederror")
    model.fit(X_train, y_train)
    return float(model.predict(X_next)[0])


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate long-format predictions into per-model error metrics."""
    rows = []
    for model, group in predictions.groupby("model"):
        error = group["actual"] - group["predicted"]
        rows.append({
            "Model": model,
            "MAE": np.abs(error).mean(),
            "RMSE": float(np.sqrt((error ** 2).mean())),
            "MedianAE": float(np.abs(error).median()),
            "MaxAE": float(np.abs(error).max()),
            "N": int(len(group)),
        })
    return (pd.DataFrame(rows).sort_values("MAE")
            .reset_index(drop=True))


def win_rate(predictions: pd.DataFrame) -> pd.DataFrame:
    """Share of origins on which each model had the smallest absolute error."""
    wide = predictions.assign(ae=(predictions["actual"] - predictions["predicted"]).abs())
    best = wide.loc[wide.groupby("target_month")["ae"].idxmin(), "model"]
    counts = best.value_counts()
    total = predictions["target_month"].nunique()
    return (pd.DataFrame({"Model": counts.index, "Wins": counts.values,
                          "WinRate": (counts.values / total * 100).round(1)})
            .reset_index(drop=True))


def paired_comparison(predictions: pd.DataFrame,
                      reference: str = "XGBoost") -> pd.DataFrame:
    """Compare each model against `reference` on the same origins.

    Errors are paired by target month, so this asks whether one model beats
    another on the same forecasting problems rather than comparing two
    independent averages. A positive MeanAE_Diff means the model is worse
    than the reference. The Wilcoxon signed-rank test is used because
    absolute forecast errors are skewed and heavy-tailed.
    """
    from scipy.stats import wilcoxon

    wide = predictions.assign(
        ae=(predictions["actual"] - predictions["predicted"]).abs())
    pivot = wide.pivot_table(index="target_month", columns="model", values="ae")
    if reference not in pivot.columns:
        raise ValueError(f"reference model {reference!r} not in predictions")

    rows = []
    for model in pivot.columns:
        if model == reference:
            continue
        paired = pivot[[model, reference]].dropna()
        diff = (paired[model] - paired[reference]).to_numpy()
        try:
            p_value = float(wilcoxon(paired[model], paired[reference]).pvalue)
        except ValueError:  # all differences zero
            p_value = 1.0
        rows.append({
            "Model": model,
            "MeanAE_Diff": float(diff.mean()),
            "MedianAE_Diff": float(np.median(diff)),
            "BeatsReference_Pct": float((diff < 0).mean() * 100),
            "p_value": p_value,
            "N": int(len(diff)),
        })
    return pd.DataFrame(rows).sort_values("MeanAE_Diff").reset_index(drop=True)


def load_series() -> tuple[pd.DataFrame, list[str]]:
    """Return the 7-Day series with encoded features and the feature columns."""
    df = pd.read_csv(FEATURES_CSV)
    df["Month"] = pd.to_datetime(df["Month"])
    df = df[df["Day Time"] == "7-Day"].sort_values("Month").reset_index(drop=True)

    otp = pd.to_numeric(df["On-Time Performance"], errors="coerce")
    df["otp"] = otp * 100 if otp.max() <= 1.5 else otp

    encoded = pd.get_dummies(df, columns=["Season"], drop_first=True)
    feature_cols = [c for c in [
        "Delayed Trains", "On-Time Trips", "Scheduled Trips",
        "OTP_Lag_1", "OTP_Lag_2", "OTP_Lag_3",
        "OTP_Rolling_3", "OTP_Rolling_6",
        "Delayed_Trains_Lag_1", "Delayed_Trains_Rolling_3",
        "Delay_Rate", "Year", "Month_Number", "Quarter",
    ] if c in encoded.columns]
    feature_cols += [c for c in encoded.columns if c.startswith("Season_")]
    return encoded, feature_cols


def run(min_train: int = DEFAULT_MIN_TRAIN, skip_slow: bool = False) -> pd.DataFrame:
    """Score every model one step ahead at each rolling origin."""
    df, feature_cols = load_series()
    otp = df["otp"].to_numpy()
    months = pd.DatetimeIndex(df["Month"])
    X_all = df[feature_cols].astype(float)

    records = []
    origins = list(rolling_origins(len(df), min_train))
    print(f"{len(df)} months available; scoring {len(origins)} origins "
          f"({months[min_train].strftime('%Y-%m')} to {months[-1].strftime('%Y-%m')})")

    for n, i in enumerate(origins, 1):
        history, hist_months = otp[:i], months[:i]
        target_month, actual = months[i], otp[i]

        preds = {
            "Naive": naive_forecast(history),
            "3-Month MA": moving_average_forecast(history, 3),
            "6-Month MA": moving_average_forecast(history, 6),
        }
        # XGBoost predicts month i from the row whose target is month i,
        # i.e. the feature row for month i-1, using only earlier rows to train.
        X_train, y_train = X_all.iloc[: i - 1], pd.Series(otp[1:i])
        preds["XGBoost"] = xgboost_forecast(X_train, y_train, X_all.iloc[[i - 1]])

        if not skip_slow:
            preds["SARIMA"] = sarima_forecast(history, hist_months)
            preds["Prophet"] = prophet_forecast(history, hist_months)

        for model, value in preds.items():
            records.append({"origin_index": i, "target_month": target_month,
                            "model": model, "actual": actual, "predicted": value})
        if n % 20 == 0 or n == len(origins):
            print(f"  {n}/{len(origins)} origins done")

    return pd.DataFrame(records)


def plot_results(predictions: pd.DataFrame, path: Path) -> None:
    """Rolling error over time, and the spread of errors per model."""
    import matplotlib.pyplot as plt

    wide = predictions.assign(
        ae=(predictions["actual"] - predictions["predicted"]).abs())
    pivot = wide.pivot_table(index="target_month", columns="model", values="ae")
    order = pivot.mean().sort_values().index

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 9))
    for model in order:
        pivot[model].rolling(12, min_periods=6).mean().plot(
            ax=top, linewidth=2 if model == "XGBoost" else 1.2, label=model)
    top.set_title("Rolling 12-month mean absolute error, one-step-ahead forecasts")
    top.set_ylabel("MAE (OTP points)")
    top.set_xlabel("")
    top.legend(ncol=3, fontsize=9)
    top.grid(alpha=0.3)

    bottom.boxplot([pivot[m].dropna() for m in order], tick_labels=list(order),
                   showfliers=True)
    bottom.set_title("Distribution of absolute errors across 119 origins")
    bottom.set_ylabel("Absolute error (OTP points)")
    bottom.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-train", type=int, default=DEFAULT_MIN_TRAIN)
    parser.add_argument("--skip-slow", action="store_true",
                        help="omit SARIMA and Prophet for a fast smoke run")
    parser.add_argument("--report-only", action="store_true",
                        help="rebuild reports and the figure from saved predictions")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    predictions_path = PREDICTION_DIR / "phase22_rolling_origin_predictions.csv"

    if args.report_only:
        predictions = pd.read_csv(predictions_path)
        predictions["target_month"] = pd.to_datetime(predictions["target_month"])
    else:
        predictions = run(min_train=args.min_train, skip_slow=args.skip_slow)
        predictions.to_csv(predictions_path, index=False)

    summary = summarize(predictions).merge(win_rate(predictions), on="Model", how="left")
    summary.to_csv(REPORT_DIR / "phase22_rolling_origin_summary.csv", index=False)

    paired = paired_comparison(predictions, reference="XGBoost")
    paired.to_csv(REPORT_DIR / "phase22_rolling_origin_paired_vs_xgboost.csv", index=False)

    plot_results(predictions, ROOT / "outputs/figures/phase22_rolling_origin_errors.png")

    print("\n=== Rolling-origin summary (one-step-ahead, identical task) ===")
    print(summary.to_string(index=False))
    print("\n=== Paired against XGBoost (positive means worse than XGBoost) ===")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
