"""Controlled rolling-origin test of whether weather improves the OTP forecast.

Compares three models on identical origins, paired by target month, reusing the
Phase 22 harness so the base XGBoost here is byte-for-byte the Phase 22 model on
the Phase 22 features. The only difference between variants is the weather
columns, so any error difference is attributable to weather alone.

- XGBoost                    : base, internal features only (Phase 22)
- XGBoost+Weather operational: base + weather of the feature month (a forecast)
- XGBoost+Weather oracle     : base + weather of the target month (explanatory)

See docs/plans/2026-08-16-weather-features.md for the operational/oracle framing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.fetch_weather_data import WEATHER_COLUMNS
from scripts.rolling_origin_eval import (
    DEFAULT_MIN_TRAIN,
    load_series,
    paired_comparison,
    rolling_origins,
    summarize,
    xgboost_forecast,
)
from scripts.weather_features import join_weather

ROOT = Path(__file__).resolve().parents[1]
WEATHER_CSV = ROOT / "data/raw/staten_island_weather_monthly.csv"
REPORT_DIR = ROOT / "outputs/reports"
PREDICTION_DIR = ROOT / "outputs/predictions"
FIGURE = ROOT / "outputs/figures/phase23_weather_errors.png"

SHOCK_THRESHOLD_CM = 15.0
MODELS = ["XGBoost", "XGBoost+Weather (operational)", "XGBoost+Weather (oracle)"]


def shock_months(weather: pd.DataFrame, threshold_cm: float = SHOCK_THRESHOLD_CM):
    """Target months whose total snowfall exceeds the threshold, sorted."""
    w = weather.copy()
    w["Month"] = pd.to_datetime(w["Month"])
    hit = w.loc[w["total_snowfall_cm"] > threshold_cm, "Month"]
    return sorted(hit)


def align_matrices(base: pd.DataFrame, weather: pd.DataFrame):
    """Inner-join two Month-indexed frames; return both aligned plus the months."""
    b = base.copy(); b["Month"] = pd.to_datetime(b["Month"])
    w = weather.copy(); w["Month"] = pd.to_datetime(w["Month"])
    months = pd.Index(sorted(set(b["Month"]) & set(w["Month"])))
    b = b[b["Month"].isin(months)].sort_values("Month").reset_index(drop=True)
    w = w[w["Month"].isin(months)].sort_values("Month").reset_index(drop=True)
    return b, w, months


def _feature_matrices():
    """Base and weather-augmented feature matrices over the shared 7-Day months."""
    df, feature_cols = load_series()
    weather = pd.read_csv(WEATHER_CSV)
    op = join_weather(df, weather, "operational")
    orc = join_weather(df, weather, "oracle")
    matrices = {
        "XGBoost": df[feature_cols].astype(float),
        "XGBoost+Weather (operational)": op[feature_cols + WEATHER_COLUMNS].astype(float),
        "XGBoost+Weather (oracle)": orc[feature_cols + WEATHER_COLUMNS].astype(float),
    }
    return df, matrices


def run(min_train: int = DEFAULT_MIN_TRAIN) -> pd.DataFrame:
    """Score all three variants one step ahead at each rolling origin."""
    df, matrices = _feature_matrices()
    otp = df["otp"].to_numpy()
    months = pd.DatetimeIndex(df["Month"])
    origins = list(rolling_origins(len(df), min_train))
    print(f"{len(df)} months; {len(origins)} origins "
          f"({months[min_train]:%Y-%m} to {months[-1]:%Y-%m})")

    records = []
    for n, i in enumerate(origins, 1):
        target_month, actual = months[i], otp[i]
        y_train = pd.Series(otp[1:i])
        for name, X in matrices.items():
            pred = xgboost_forecast(X.iloc[: i - 1], y_train, X.iloc[[i - 1]])
            records.append({"origin_index": i, "target_month": target_month,
                            "model": name, "actual": actual, "predicted": pred})
        if n % 20 == 0 or n == len(origins):
            print(f"  {n}/{len(origins)} origins")
    return pd.DataFrame(records)


def plot_results(predictions: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    wide = predictions.assign(ae=(predictions["actual"] - predictions["predicted"]).abs())
    pivot = wide.pivot_table(index="target_month", columns="model", values="ae")
    order = [m for m in MODELS if m in pivot.columns]

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 9))
    for m in order:
        pivot[m].rolling(12, min_periods=6).mean().plot(ax=top, linewidth=1.8, label=m)
    top.set_title("Rolling 12-month MAE: base vs weather-augmented forecasts")
    top.set_ylabel("MAE (OTP points)"); top.set_xlabel("")
    top.legend(fontsize=9); top.grid(alpha=0.3)
    bottom.boxplot([pivot[m].dropna() for m in order],
                   tick_labels=[m.replace("XGBoost+Weather ", "") for m in order],
                   showfliers=True)
    bottom.set_title("Absolute-error distribution across origins")
    bottom.set_ylabel("Absolute error (OTP points)"); bottom.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)


def _summary_with_paired(predictions: pd.DataFrame) -> pd.DataFrame:
    summary = summarize(predictions)
    paired = paired_comparison(predictions, reference="XGBoost")[
        ["Model", "MeanAE_Diff", "p_value"]]
    return summary.merge(paired, on="Model", how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-train", type=int, default=DEFAULT_MIN_TRAIN)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    preds_path = PREDICTION_DIR / "phase23_weather_predictions.csv"
    if args.report_only:
        predictions = pd.read_csv(preds_path, parse_dates=["target_month"])
    else:
        predictions = run(min_train=args.min_train)
        predictions.to_csv(preds_path, index=False)

    overall = _summary_with_paired(predictions)
    overall.to_csv(REPORT_DIR / "phase23_weather_comparison.csv", index=False)
    paired_comparison(predictions, reference="XGBoost").to_csv(
        REPORT_DIR / "phase23_weather_paired.csv", index=False)

    weather = pd.read_csv(WEATHER_CSV)
    shocks = shock_months(weather)
    shock_preds = predictions[predictions["target_month"].isin(shocks)]
    shock_summary = _summary_with_paired(shock_preds)
    shock_summary.insert(0, "n_shock_months", shock_preds["target_month"].nunique())
    shock_summary.to_csv(REPORT_DIR / "phase23_shock_months.csv", index=False)

    plot_results(predictions, FIGURE)

    print("\n=== Overall (119 origins) ===")
    print(overall.to_string(index=False))
    print(f"\n=== Shock months only (snowfall > {SHOCK_THRESHOLD_CM:.0f} cm target, "
          f"n = {shock_preds['target_month'].nunique()}) ===")
    print(shock_summary.to_string(index=False))


if __name__ == "__main__":
    main()
