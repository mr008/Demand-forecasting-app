"""Weekly modelling table and per-series metadata.

The weekly table has one row per (upc, cedis, week_start) for every complete ISO
week between ``data.first_week_start`` and ``data.last_complete_week_start``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from supply_pipeline.calendar_features import weekly_calendar
from supply_pipeline.config import Config

SERIES_KEY = ["upc", "cedis"]
CATALOG_COLS = ["upc", "cluster", "abc_class", "xyz_class", "lead_time_days", "moq", "safety_stock_days"]
MIN_DAYS_FOR_WEEK = 5


def build_weekly(daily: pd.DataFrame, calendar: pd.DataFrame, catalog: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Aggregate daily rows into the weekly modelling table.

    ``y`` is the weekly sell-out. Weeks with fewer than ``MIN_DAYS_FOR_WEEK``
    observed days are NaN; weeks with 5-6 observed days are scaled to 7 days and
    flagged via ``days_observed``.
    """
    d = daily.copy()
    d["week_start"] = (d["date"] - pd.to_timedelta(d["date"].dt.dayofweek, unit="D")).dt.normalize()
    start = pd.Timestamp(cfg.data.first_week_start)
    last = pd.Timestamp(cfg.data.last_complete_week_start)
    d = d[(d["week_start"] >= start) & (d["week_start"] <= last)]

    observed = d["sell_out_pzs"].notna()
    d["_obs"] = observed.astype("int8")
    d["_promo_obs"] = np.where(observed, d["promo_flag"], np.nan)
    d["_price_obs"] = np.where(observed, d["final_price"], np.nan)

    g = d.groupby(SERIES_KEY + ["week_start"], as_index=False)
    w = g.agg(
        y_sum=("sell_out_pzs", "sum"),
        y_capped_sum=("sell_out_capped", "sum"),
        days_observed=("_obs", "sum"),
        price_mean=("_price_obs", "mean"),
        promo_share=("_promo_obs", "mean"),
        n_outliers=("outlier_flag", "sum"),
        zero_days=("sell_out_pzs", lambda s: int((s == 0).sum())),
    )
    scale = 7.0 / w["days_observed"].replace(0, np.nan)
    enough = w["days_observed"] >= MIN_DAYS_FOR_WEEK
    w["y"] = np.where(enough, w["y_sum"] * scale, np.nan)
    w["y_capped"] = np.where(enough, w["y_capped_sum"] * scale, np.nan)
    w = w.drop(columns=["y_sum", "y_capped_sum"])

    # Reindex to the full week grid per series from its first complete week.
    weeks = pd.date_range(start, last, freq="7D")
    first_week = w.dropna(subset=["y"]).groupby(SERIES_KEY)["week_start"].min().rename("first_week").reset_index()
    grid = first_week.merge(pd.DataFrame({"week_start": weeks}), how="cross")
    grid = grid[grid["week_start"] >= grid["first_week"]].drop(columns="first_week")
    w = grid.merge(w, on=SERIES_KEY + ["week_start"], how="left", validate="one_to_one")
    w["days_observed"] = w["days_observed"].fillna(0).astype("int8")
    w["n_outliers"] = w["n_outliers"].fillna(0).astype("int8")
    w["zero_days"] = w["zero_days"].fillna(0).astype("int8")
    w["promo_share"] = w["promo_share"].fillna(0.0)
    w["price_mean"] = w.groupby(SERIES_KEY)["price_mean"].ffill()

    w = w.merge(weekly_calendar(calendar), on="week_start", how="left", validate="many_to_one")
    w = w.merge(catalog[CATALOG_COLS], on="upc", how="left", validate="many_to_one")
    w["series_age_weeks"] = w.groupby(SERIES_KEY).cumcount().astype("int16")
    w["cedis"] = w["cedis"].astype("string")
    return w.sort_values(SERIES_KEY + ["week_start"]).reset_index(drop=True)


def build_series_metadata(
    daily: pd.DataFrame, weekly: pd.DataFrame, inv_cedis: pd.DataFrame, catalog: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Per-series facts and the flags that gate model trust."""
    obs = daily[daily["sell_out_pzs"].notna()]
    meta = (
        obs.groupby(SERIES_KEY)
        .agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            n_days=("date", "size"),
            mean_daily=("sell_out_pzs", "mean"),
        )
        .reset_index()
    )
    span = (meta["last_date"] - meta["first_date"]).dt.days + 1
    meta["interior_gap_days"] = (span - meta["n_days"]).astype("int32")

    wk = weekly.dropna(subset=["y"])
    wmeta = (
        wk.groupby(SERIES_KEY)
        .agg(
            first_week=("week_start", "min"),
            n_weeks=("week_start", "size"),
            mean_weekly=("y", "mean"),
        )
        .reset_index()
    )
    tail = wk[wk["week_start"] > wk["week_start"].max() - pd.Timedelta(weeks=8)]
    tail_stats = (
        tail.groupby(SERIES_KEY)
        .agg(
            tail8_mean=("y", "mean"),
            tail8_promo_share=("promo_share", "mean"),
        )
        .reset_index()
    )
    tail4 = wk[wk["week_start"] > wk["week_start"].max() - pd.Timedelta(weeks=4)]
    tail4_stats = tail4.groupby(SERIES_KEY).agg(tail4_mean=("y", "mean")).reset_index()

    meta = meta.merge(wmeta, on=SERIES_KEY, how="left").merge(tail_stats, on=SERIES_KEY, how="left")
    meta = meta.merge(tail4_stats, on=SERIES_KEY, how="left")
    meta = meta.merge(catalog[CATALOG_COLS], on="upc", how="left", validate="many_to_one")

    as_of = pd.Timestamp(cfg.data.as_of)
    inv_asof = inv_cedis[inv_cedis["date"] == as_of][SERIES_KEY + ["on_hand"]].rename(columns={"on_hand": "on_hand_as_of"})
    meta = meta.merge(inv_asof, on=SERIES_KEY, how="left")
    meta["has_inventory"] = meta["on_hand_as_of"].notna()

    meta["is_cold_start"] = meta["n_weeks"].fillna(0) < cfg.data.cold_start_weeks
    # Discontinued: essentially no sales in the last 4 weeks while the series used to sell.
    meta["is_discontinued"] = (meta["tail4_mean"].fillna(0) <= 0.05 * meta["mean_weekly"].fillna(0)) & (
        meta["mean_weekly"].fillna(0) > 0
    )
    meta["is_high_promo"] = meta["tail8_promo_share"].fillna(0) >= 0.4
    meta["cedis"] = meta["cedis"].astype("string")
    return meta.sort_values(SERIES_KEY).reset_index(drop=True)
