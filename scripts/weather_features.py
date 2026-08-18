"""Join monthly weather onto the 7-Day OTP feature series.

Two framings, kept strictly separate (see docs/plans/2026-08-16-weather-features.md):

- operational: a feature row for month t carries the weather of month t, which
  is known when the month t+1 forecast is made. A real forecast input.
- oracle: a feature row for month t carries the weather of month t+1 (its
  target month). Not a forecast; it measures how much of the OTP the base
  model misses is explainable by weather, given foreknowledge of the weather.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.fetch_weather_data import WEATHER_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "outputs/predictions/staten_island_otp_features.csv"
WEATHER_CSV = ROOT / "data/raw/staten_island_weather_monthly.csv"
OPERATIONAL_CSV = ROOT / "outputs/predictions/otp_weather_operational.csv"
ORACLE_CSV = ROOT / "outputs/predictions/otp_weather_oracle.csv"

FRAMINGS = ("operational", "oracle")


def join_weather(features: pd.DataFrame, weather: pd.DataFrame,
                 framing: str) -> pd.DataFrame:
    """Attach the weather columns to each feature row under the given framing."""
    if framing not in FRAMINGS:
        raise ValueError(f"unknown framing {framing!r}; expected one of {FRAMINGS}")
    missing = [c for c in WEATHER_COLUMNS if c not in weather.columns]
    if missing:
        raise ValueError(f"weather frame missing columns: {missing}")

    f = features.copy()
    f["Month"] = pd.to_datetime(f["Month"])
    w = weather[["Month"] + WEATHER_COLUMNS].copy()
    w["Month"] = pd.to_datetime(w["Month"])

    offset = 1 if framing == "oracle" else 0
    f["_join_month"] = f["Month"] + pd.DateOffset(months=offset)
    merged = f.merge(w.rename(columns={"Month": "_join_month"}),
                     on="_join_month", how="left")
    return merged.drop(columns="_join_month")


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produce the operational and oracle tables for the 7-Day series."""
    feats = pd.read_csv(FEATURES_CSV)
    feats["Month"] = pd.to_datetime(feats["Month"])
    feats = feats[feats["Day Time"] == "7-Day"].sort_values("Month").reset_index(drop=True)
    weather = pd.read_csv(WEATHER_CSV)
    return (join_weather(feats, weather, "operational"),
            join_weather(feats, weather, "oracle"))


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    operational, oracle = build()
    operational.to_csv(OPERATIONAL_CSV, index=False)
    oracle.to_csv(ORACLE_CSV, index=False)
    for name, df in (("operational", operational), ("oracle", oracle)):
        na = int(df[WEATHER_COLUMNS].isna().sum().sum())
        print(f"{name}: {len(df)} rows, {na} weather NaNs")
    # The one alignment check that matters: February 2026's snow.
    orc = oracle.set_index("Month")
    print(f"oracle row for 2026-01 (predicts Feb): "
          f"{orc.loc[pd.Timestamp('2026-01-01'), 'total_snowfall_cm']:.1f} cm snow")


if __name__ == "__main__":
    main()
