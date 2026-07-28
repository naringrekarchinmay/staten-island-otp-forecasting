"""Download the MTA Staten Island Railway OTP dataset from data.ny.gov.

Writes the same Excel workbook that notebooks/01 reads, so refreshing the
data requires no notebook changes. The dataset behind metrics.mta.info is
published as Socrata dataset fccm-griq.
"""
from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

DATASET_ID = "fccm-griq"
SOURCE_URL = f"https://data.ny.gov/resource/{DATASET_ID}.csv"
RAW_XLSX = (Path(__file__).resolve().parents[1]
            / "data/raw/MTA_Staten_Island_Railway_On-Time_Performance.xlsx")

# Socrata field name -> column name used by the workbook and notebook 01.
# `on_time_performance_with` is truncated by Socrata's field-name limit.
COLUMN_MAP = {
    "month": "Month",
    "day_time": "Day Time",
    "delayed_trains": "Delayed Trains",
    "on_time_trips": "On-Time Trips",
    "on_time_performance": "On-Time Performance",
    "delayed_trains_with_boat": "Delayed Trains (With Boat)",
    "on_time_trips_with_boat": "On-Time Trips (With Boat)",
    "on_time_performance_with": "On-Time Performance (With Boat)",
    "scheduled_trips": "Scheduled Trips",
    "incomplete_trips": "Incomplete Trips",
    "trip_complete_percentage": "Trip Complete Percentage",
}
EXCEL_COLUMNS = list(COLUMN_MAP.values())

# Order the shipped workbook uses within each month. Preserved so a refresh
# produces a readable diff instead of reordering every row.
CATEGORY_ORDER = ["Weekday", "AM Rush", "PM Rush", "Weekend", "7-Day"]


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Map the Socrata export onto the workbook layout."""
    missing = [c for c in COLUMN_MAP if c not in raw.columns]
    if missing:
        raise ValueError(
            f"source schema changed; missing columns: {missing}. "
            f"Check https://data.ny.gov/d/{DATASET_ID}")
    df = raw[list(COLUMN_MAP)].rename(columns=COLUMN_MAP)
    df["Month"] = pd.to_datetime(df["Month"])
    order = pd.Categorical(df["Day Time"], categories=CATEGORY_ORDER, ordered=True)
    df = df.assign(_order=order).sort_values(["Month", "_order"], kind="mergesort")
    return df.drop(columns="_order").reset_index(drop=True)


def fetch(limit: int = 50000) -> pd.DataFrame:
    """Download the full dataset and return it in workbook layout."""
    response = requests.get(SOURCE_URL, params={"$limit": limit, "$order": "month"},
                            timeout=60)
    response.raise_for_status()
    return normalize(pd.read_csv(StringIO(response.text)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RAW_XLSX,
                        help="destination workbook path")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change without writing")
    args = parser.parse_args()

    fresh = fetch()
    latest = fresh["Month"].max().strftime("%Y-%m")
    print(f"fetched {len(fresh)} rows from {DATASET_ID}, latest month {latest}")

    if args.out.exists():
        current = pd.read_excel(args.out)
        current_latest = pd.to_datetime(current["Month"]).max().strftime("%Y-%m")
        print(f"existing workbook: {len(current)} rows, latest month {current_latest}")
        print(f"delta: {len(fresh) - len(current)} new rows")

    if args.dry_run:
        print("dry run, nothing written")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fresh.to_excel(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
