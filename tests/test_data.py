from datetime import date

import numpy as np
import pandas as pd
import pytest

from supply_pipeline import schema
from supply_pipeline.calendar_features import buen_fin_dates, build_calendar, semana_santa_dates
from supply_pipeline.config import load_config
from supply_pipeline.data import aggregate_inventory_to_cedis, complete_daily_calendar, flag_outliers
from supply_pipeline.features import build_weekly


# ----------------------------------------------------------------------------- calendar
def test_buen_fin_2024_is_nov_15_to_18() -> None:
    assert buen_fin_dates(2024) == {date(2024, 11, 15), date(2024, 11, 16), date(2024, 11, 17), date(2024, 11, 18)}


def test_semana_santa_2025_ends_on_easter() -> None:
    days = semana_santa_dates(2025)
    assert date(2025, 4, 20) in days  # Easter Sunday 2025
    assert date(2025, 4, 13) in days  # Palm Sunday
    assert len(days) == 8


def test_calendar_paydays_and_holidays() -> None:
    cal = build_calendar(date(2025, 2, 1), date(2025, 3, 3)).set_index("date")
    assert cal.loc["2025-02-15", "is_payday"] == 1
    assert cal.loc["2025-02-28", "is_payday"] == 1
    assert cal.loc["2025-02-20", "is_payday"] == 0
    assert cal.loc["2025-02-03", "is_holiday"] == 1  # Constitution Day, observed first Monday of February
    assert cal.loc["2025-02-17", "is_holiday"] == 0
    assert cal.loc["2025-02-10", "week_start"] == pd.Timestamp("2025-02-10")
    assert cal.loc["2025-02-16", "week_start"] == pd.Timestamp("2025-02-10")


# ----------------------------------------------------------------------------- schema
def test_schema_rejects_missing_column() -> None:
    df = pd.DataFrame({"store_nbr": [1]})
    with pytest.raises(schema.SchemaError):
        schema.validate(df, schema.STORE_CATALOG_DTYPES, "store_catalog")


def test_schema_rejects_duplicate_key() -> None:
    df = pd.DataFrame({"store_nbr": [1, 1], "cedis": ["A", "A"]})
    with pytest.raises(schema.SchemaError):
        schema.validate(df, schema.STORE_CATALOG_DTYPES, "x", key=["store_nbr"])


# ----------------------------------------------------------------------------- inventory
def test_inventory_aggregation_clips_and_counts() -> None:
    inv = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-04-01"] * 3),
            "upc": [1, 1, 1],
            "store_nbr": [10, 11, 12],
            "cedis": ["X", "X", "X"],
            "on_hand_qty": [10.0, -5.0, 0.0],
        }
    )
    out = aggregate_inventory_to_cedis(inv, clip_negative=True)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["on_hand"] == 10.0
    assert row["on_hand_raw"] == 5.0
    assert row["stores_reporting"] == 3
    assert row["stores_at_or_below_zero"] == 2
    assert row["stores_negative"] == 1
    assert row["stockout_store_share"] == pytest.approx(2 / 3)


# ----------------------------------------------------------------------------- daily / weekly
def _toy_sell_out() -> pd.DataFrame:
    dates = pd.date_range("2024-03-18", "2024-04-14", freq="D")  # 4 full weeks, Monday start
    df = pd.DataFrame({"date": dates})
    df["upc"] = 1
    df["cedis"] = pd.Series(["X"] * len(df), dtype="string")
    df["sell_out_pzs"] = 10.0
    df["final_price"] = 20.0
    df["promo_flag"] = 0
    df["promo_flag"] = df["promo_flag"].astype("int8")
    # Drop two days in week 2 (still >= 5 observed) and four days in week 3 (< 5 observed).
    drop = list(pd.date_range("2024-03-26", "2024-03-27")) + list(pd.date_range("2024-04-01", "2024-04-04"))
    return df[~df["date"].isin(drop)].reset_index(drop=True)


def test_complete_calendar_keeps_gaps_as_nan() -> None:
    daily = complete_daily_calendar(_toy_sell_out(), date(2024, 4, 14))
    assert len(daily) == 28
    assert daily["is_missing"].sum() == 6
    assert daily["sell_out_pzs"].isna().sum() == 6
    assert daily["promo_flag"].isna().sum() == 0


def test_weekly_scaling_and_min_days(tmp_path) -> None:
    cfg = load_config()
    cfg_data = cfg.data.__class__(
        **{**cfg.data.__dict__, "first_week_start": date(2024, 3, 18), "last_complete_week_start": date(2024, 4, 8)}
    )
    cfg = cfg.__class__(**{**cfg.__dict__, "data": cfg_data})
    daily = complete_daily_calendar(_toy_sell_out(), date(2024, 4, 14))
    daily = flag_outliers(daily, z=5.0)
    cal = build_calendar(date(2024, 3, 18), date(2024, 4, 30))
    daily = daily.merge(cal.drop(columns=["dow", "is_weekend"]), on="date", how="left")
    catalog = pd.DataFrame(
        {
            "upc": [1],
            "cluster": ["A-X"],
            "abc_class": ["A"],
            "xyz_class": ["X"],
            "lead_time_days": [7],
            "moq": [100],
            "safety_stock_days": [7],
        }
    )
    weekly = build_weekly(daily, cal, catalog, cfg)
    weekly = weekly.set_index("week_start")
    assert weekly.loc["2024-03-18", "y"] == pytest.approx(70.0)
    assert weekly.loc["2024-03-25", "days_observed"] == 5
    assert weekly.loc["2024-03-25", "y"] == pytest.approx(70.0)  # 5 days * 10 scaled to 7
    assert np.isnan(weekly.loc["2024-04-01", "y"])  # only 3 days observed
    assert weekly.loc["2024-04-08", "y"] == pytest.approx(70.0)
    assert weekly.loc["2024-03-25", "series_age_weeks"] == 1


def test_outlier_flag_caps_training_copy_only() -> None:
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "upc": 1,
            "cedis": pd.Series(["X"] * 80, dtype="string"),
            "sell_out_pzs": 100.0,
            "final_price": 1.0,
            "promo_flag": 0,
            "is_missing": 0,
        }
    )
    df.loc[70, "sell_out_pzs"] = 5000.0
    df["sell_out_pzs"] += np.linspace(0, 5, 80)  # tiny variation so MAD > 0
    out = flag_outliers(df, z=5.0)
    assert out.loc[70, "outlier_flag"] == 1
    assert out.loc[70, "sell_out_pzs"] > 4000
    assert out.loc[70, "sell_out_capped"] < 200
    assert out["outlier_flag"].sum() == 1
