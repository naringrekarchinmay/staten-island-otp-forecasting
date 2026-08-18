"""Download daily Staten Island weather and aggregate it to a monthly table.

Source: Open-Meteo historical archive (ERA5 reanalysis), keyless. Written to
mirror scripts/fetch_mta_data.py: a reproducible pull to a committed CSV, so
the weather join has the same provenance guarantees as the OTP data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
# Staten Island centroid. Monthly weather is near-uniform across the island,
# so one representative point is sufficient (noted as a limitation).
LATITUDE, LONGITUDE = 40.58, -74.13
DAILY_VARS = ["snowfall_sum", "precipitation_sum", "temperature_2m_mean",
              "temperature_2m_min", "windspeed_10m_max"]
OUT_CSV = (Path(__file__).resolve().parents[1]
           / "data/raw/staten_island_weather_monthly.csv")

WEATHER_COLUMNS = [
    "total_snowfall_cm", "max_daily_snowfall_cm", "snow_days",
    "total_precip_mm", "mean_temp_c", "min_temp_c",
    "freezing_days", "max_wind_kmh",
]


def aggregate_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    required = {"time", *DAILY_VARS}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"daily frame missing columns: {sorted(missing)}")
    d = daily.copy()
    d["time"] = pd.to_datetime(d["time"])
    d["Month"] = d["time"].dt.to_period("M").dt.to_timestamp()
    g = d.groupby("Month")
    out = pd.DataFrame({
        "total_snowfall_cm": g["snowfall_sum"].sum(),
        "max_daily_snowfall_cm": g["snowfall_sum"].max(),
        "snow_days": g["snowfall_sum"].apply(lambda s: int((s > 0).sum())),
        "total_precip_mm": g["precipitation_sum"].sum(),
        "mean_temp_c": g["temperature_2m_mean"].mean(),
        "min_temp_c": g["temperature_2m_min"].min(),
        "freezing_days": g["temperature_2m_min"].apply(lambda s: int((s < 0).sum())),
        "max_wind_kmh": g["windspeed_10m_max"].max(),
    }).reset_index()
    return out[["Month"] + WEATHER_COLUMNS]


def fetch(start: str = "2006-01-01", end: str = "2026-06-30") -> pd.DataFrame:
    resp = requests.get(ARCHIVE_URL, params={
        "latitude": LATITUDE, "longitude": LONGITUDE,
        "start_date": start, "end_date": end,
        "daily": ",".join(DAILY_VARS), "timezone": "America/New_York"},
        timeout=90)
    resp.raise_for_status()
    daily = pd.DataFrame(resp.json()["daily"])
    return aggregate_monthly(daily)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    parser.add_argument("--end", default="2026-06-30",
                        help="freeze date; keep at the MTA data vintage")
    args = parser.parse_args()
    monthly = fetch(end=args.end)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(args.out, index=False)
    print(f"wrote {len(monthly)} months to {args.out} "
          f"({monthly['Month'].min():%Y-%m} to {monthly['Month'].max():%Y-%m})")
    feb = monthly[monthly["Month"] == "2026-02-01"]
    if not feb.empty:
        print(f"Feb 2026 sanity: {feb['total_snowfall_cm'].iloc[0]:.1f} cm snowfall, "
              f"peak day {feb['max_daily_snowfall_cm'].iloc[0]:.1f} cm")


if __name__ == "__main__":
    main()
