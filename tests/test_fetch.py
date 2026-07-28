"""The Socrata fetch script maps the source schema onto the Excel layout."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.fetch_mta_data import CATEGORY_ORDER, COLUMN_MAP, EXCEL_COLUMNS, normalize


def _raw_row(month, day_time, **over):
    row = {
        "month": month, "day_time": day_time, "delayed_trains": 10,
        "on_time_trips": 100, "on_time_performance": 0.95,
        "delayed_trains_with_boat": 12, "on_time_trips_with_boat": 98,
        "on_time_performance_with": 0.94, "scheduled_trips": 110,
        "incomplete_trips": "N/A", "trip_complete_percentage": "N/A",
    }
    row.update(over)
    return row


def test_renames_socrata_fields_to_excel_columns():
    raw = pd.DataFrame([_raw_row("2026-02-01T00:00:00.000", "7-Day")])
    out = normalize(raw)
    assert list(out.columns) == EXCEL_COLUMNS


def test_truncated_boat_field_maps_to_full_column_name():
    # Socrata truncates the field to `on_time_performance_with`.
    raw = pd.DataFrame([_raw_row("2026-02-01T00:00:00.000", "7-Day",
                                 on_time_performance_with=0.912)])
    out = normalize(raw)
    assert out.loc[0, "On-Time Performance (With Boat)"] == 0.912


def test_month_is_parsed_to_datetime():
    raw = pd.DataFrame([_raw_row("2026-02-01T00:00:00.000", "7-Day")])
    out = normalize(raw)
    assert out["Month"].dtype.kind == "M"
    assert out.loc[0, "Month"] == pd.Timestamp("2026-02-01")


def test_rows_ordered_by_month_then_source_category_sequence():
    raw = pd.DataFrame([
        _raw_row("2026-03-01T00:00:00.000", "7-Day"),
        _raw_row("2026-02-01T00:00:00.000", "Weekend"),
        _raw_row("2026-02-01T00:00:00.000", "Weekday"),
    ])
    out = normalize(raw)
    assert list(out["Day Time"]) == ["Weekday", "Weekend", "7-Day"]
    assert out["Month"].is_monotonic_increasing


def test_category_order_matches_the_shipped_workbook():
    assert CATEGORY_ORDER == ["Weekday", "AM Rush", "PM Rush", "Weekend", "7-Day"]


def test_missing_source_column_raises_rather_than_silently_dropping():
    raw = pd.DataFrame([_raw_row("2026-02-01T00:00:00.000", "7-Day")]).drop(
        columns=["on_time_performance_with"])
    with pytest.raises(ValueError, match="on_time_performance_with"):
        normalize(raw)


def test_column_map_covers_all_eleven_source_fields():
    assert len(COLUMN_MAP) == 11
