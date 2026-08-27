"""Stage 'prepare': load, validate, clean and cache the four raw files.

Outputs (all under ``data/interim``):

- ``catalog.parquet``            one row per UPC (deduplicated on ``upc``)
- ``stores.parquet``             store -> cedis
- ``daily.parquet``              daily sell-out per (upc, cedis) on a completed calendar
- ``inventory_cedis_daily.parquet``  store inventory rolled up to cedis per day
- ``weekly.parquet``             the modelling table (see ``features.py``)
- ``series.parquet``             per-series metadata and flags

and ``reports/tables/coverage_*.csv`` describing what the raw data covers.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from supply_pipeline import schema
from supply_pipeline.calendar_features import build_calendar
from supply_pipeline.config import Config
from supply_pipeline.features import build_series_metadata, build_weekly

log = logging.getLogger(__name__)

SERIES_KEY = ["upc", "cedis"]


# --------------------------------------------------------------------------- loaders
def load_upc_catalog(cfg: Config) -> pd.DataFrame:
    raw = pd.read_csv(cfg.paths.upc_catalog)
    df = schema.validate(raw, schema.UPC_CATALOG_DTYPES, "upc_catalog", non_null=["upc"])
    n_before = len(df)
    df = df.sort_values(["upc", "prime_item_nbr"]).drop_duplicates("upc", keep="first")
    if len(df) != n_before:
        log.info("upc_catalog: dropped %d duplicate rows sharing a upc", n_before - len(df))
    df["cluster"] = df["abc_class"].astype(str) + "-" + df["xyz_class"].astype(str)
    return df.reset_index(drop=True)


def load_store_catalog(cfg: Config) -> pd.DataFrame:
    raw = pd.read_csv(cfg.paths.store_catalog)
    return schema.validate(raw, schema.STORE_CATALOG_DTYPES, "store_catalog", key=["store_nbr"], non_null=["store_nbr", "cedis"])


def load_sell_out(cfg: Config) -> pd.DataFrame:
    raw = pd.read_csv(cfg.paths.sell_out)
    df = schema.validate(
        raw,
        schema.SELL_OUT_DTYPES,
        "sell_out",
        key=["date", "upc", "cedis"],
        non_null=["date", "upc", "cedis", "sell_out_pzs"],
    )
    if (df["sell_out_pzs"] < 0).any():
        raise schema.SchemaError("sell_out: negative sell_out_pzs")
    return df


def load_inventory(cfg: Config, stores: pd.DataFrame) -> pd.DataFrame:
    """Store-level inventory, deduplicated across prime_item_nbr, joined to cedis."""
    raw = pd.read_csv(
        cfg.paths.inventory,
        usecols=["date", "prime_item_nbr", "upc", "store_nbr", "on_hand_qty"],
    )
    df = schema.validate(raw, schema.INVENTORY_DTYPES, "inventory", non_null=["date", "upc", "store_nbr"])
    # Three UPCs carry two prime_item_nbr; the same physical stock is reported under both.
    df = df.groupby(["date", "upc", "store_nbr"], as_index=False)["on_hand_qty"].sum()
    df = df.merge(stores, on="store_nbr", how="left", validate="many_to_one")
    if df["cedis"].isna().any():
        raise schema.SchemaError("inventory: stores missing from the store catalog")
    return df


# --------------------------------------------------------------------------- transforms
def aggregate_inventory_to_cedis(inv: pd.DataFrame, clip_negative: bool) -> pd.DataFrame:
    """Roll store-level on-hand up to cedis per day, keeping stock-out signals."""
    oh = inv["on_hand_qty"]
    work = inv.assign(
        on_hand_clipped=oh.clip(lower=0) if clip_negative else oh,
        at_or_below_zero=(oh <= 0).astype("int32"),
        negative=(oh < 0).astype("int32"),
    )
    out = work.groupby(["date", "upc", "cedis"], as_index=False).agg(
        on_hand=("on_hand_clipped", "sum"),
        on_hand_raw=("on_hand_qty", "sum"),
        stores_reporting=("store_nbr", "nunique"),
        stores_at_or_below_zero=("at_or_below_zero", "sum"),
        stores_negative=("negative", "sum"),
    )
    out["stockout_store_share"] = out["stores_at_or_below_zero"] / out["stores_reporting"]
    return out


def complete_daily_calendar(sell_out: pd.DataFrame, end: date) -> pd.DataFrame:
    """One row per series per day from the series' first observation to ``end``.

    Gaps are kept as NaN (never zero-filled) and flagged with ``is_missing``.
    """
    first = sell_out.groupby(SERIES_KEY)["date"].min().rename("first_date").reset_index()
    frames = []
    end_ts = pd.Timestamp(end)
    for row in first.itertuples(index=False):
        idx = pd.date_range(row.first_date, end_ts, freq="D")
        frames.append(pd.DataFrame({"upc": row.upc, "cedis": row.cedis, "date": idx}))
    grid = pd.concat(frames, ignore_index=True)
    grid["cedis"] = grid["cedis"].astype("string")
    daily = grid.merge(sell_out, on=["upc", "cedis", "date"], how="left", validate="one_to_one")
    daily["is_missing"] = daily["sell_out_pzs"].isna().astype("int8")
    daily["promo_flag"] = daily["promo_flag"].fillna(0).astype("int8")
    daily["final_price"] = daily.groupby(SERIES_KEY)["final_price"].ffill()
    return daily.sort_values(SERIES_KEY + ["date"]).reset_index(drop=True)


def flag_outliers(daily: pd.DataFrame, z: float, window: int = 56) -> pd.DataFrame:
    """Robust rolling z-score on a trailing window; cap flagged values for training use only.

    ``sell_out_pzs`` is never modified; ``sell_out_capped`` is the training-side copy.
    """
    g = daily.groupby(SERIES_KEY)["sell_out_pzs"]
    med = g.transform(lambda s: s.shift(1).rolling(window, min_periods=14).median())
    mad = g.transform(
        lambda s: (
            (s.shift(1) - s.shift(1).rolling(window, min_periods=14).median()).abs().rolling(window, min_periods=14).median()
        )
    )
    scale = 1.4826 * mad
    robust_z = (daily["sell_out_pzs"] - med) / scale.replace(0, np.nan)
    daily["outlier_flag"] = ((robust_z.abs() > z) & daily["sell_out_pzs"].notna()).astype("int8")
    cap = med + z * scale
    daily["sell_out_capped"] = np.where(daily["outlier_flag"] == 1, np.minimum(daily["sell_out_pzs"], cap), daily["sell_out_pzs"])
    return daily


# --------------------------------------------------------------------------- stage
def run(cfg: Config) -> None:
    p = cfg.paths
    catalog = load_upc_catalog(cfg)
    stores = load_store_catalog(cfg)
    sell_out = load_sell_out(cfg)
    log.info(
        "sell_out: %d rows, %d series, %s..%s",
        len(sell_out),
        sell_out.groupby(SERIES_KEY).ngroups,
        sell_out["date"].min().date(),
        sell_out["date"].max().date(),
    )

    unknown_upc = set(sell_out["upc"]) - set(catalog["upc"])
    if unknown_upc:
        raise schema.SchemaError(f"sell_out has {len(unknown_upc)} upcs missing from the catalog")

    end = sell_out["date"].max().date()
    calendar = build_calendar(cfg.data.first_week_start, end + timedelta(days=70))
    daily = complete_daily_calendar(sell_out, end)
    daily = flag_outliers(daily, cfg.data.outlier_mad_z)
    daily = daily.merge(calendar.drop(columns=["dow", "is_weekend"]), on="date", how="left")
    log.info(
        "daily: %d rows, %.2f%% missing, %d outliers flagged",
        len(daily),
        100 * daily["is_missing"].mean(),
        int(daily["outlier_flag"].sum()),
    )

    inv = load_inventory(cfg, stores)
    inv_cedis = aggregate_inventory_to_cedis(inv, cfg.data.clip_negative_on_hand)
    log.info(
        "inventory: %d store-rows -> %d cedis-day rows, %s..%s",
        len(inv),
        len(inv_cedis),
        inv_cedis["date"].min().date(),
        inv_cedis["date"].max().date(),
    )

    weekly = build_weekly(daily, calendar, catalog, cfg)
    series = build_series_metadata(daily, weekly, inv_cedis, catalog, cfg)
    log.info(
        "weekly: %d rows, %d series, %d weeks", len(weekly), weekly.groupby(SERIES_KEY).ngroups, weekly["week_start"].nunique()
    )

    # Coverage tables for the report.
    inv_cov = inv.groupby("date").agg(rows=("upc", "size"), upcs=("upc", "nunique"), stores=("store_nbr", "nunique"))
    inv_cov.to_csv(p.tables_dir / "coverage_inventory_by_date.csv")
    series_cov = series[
        ["upc", "cedis", "cluster", "first_date", "n_days", "n_weeks", "is_cold_start", "is_discontinued", "has_inventory"]
    ]
    series_cov.to_csv(p.tables_dir / "coverage_series.csv", index=False)

    catalog.to_parquet(p.interim_dir / "catalog.parquet", index=False)
    stores.to_parquet(p.interim_dir / "stores.parquet", index=False)
    daily.to_parquet(p.interim_dir / "daily.parquet", index=False)
    inv_cedis.to_parquet(p.interim_dir / "inventory_cedis_daily.parquet", index=False)
    weekly.to_parquet(p.interim_dir / "weekly.parquet", index=False)
    series.to_parquet(p.interim_dir / "series.parquet", index=False)
    calendar.to_parquet(p.interim_dir / "calendar.parquet", index=False)
