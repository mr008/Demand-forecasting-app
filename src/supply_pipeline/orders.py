"""Stage 'orders': supply-order recommendation per SKU x distribution center.

Policy (weekly ordering, order-up-to)
-------------------------------------
* Protection period = lead time + review period (7 + 7 = 14 days = 2 forecast weeks).
* Demand over the protection period ``D``: quantiles are the (comonotonic) sum
  of the two weekly quantile forecasts; a lognormal is fitted to (p50, p90).
* Safety stock = max(catalog policy, forecast-uncertainty policy):
    - policy:  ``safety_stock_days`` x expected daily demand
    - quantile: D_{service level target} - D_p50, target set per ABC class.
* Order-up-to level S = D_p50 + safety stock.
* Order = max(0, S - projected on-hand), rounded **up** to a multiple of MOQ.
* Projected on-hand = as-of DC on-hand minus sell-out observed between the
  snapshot and the start of the protection period (no receipts data exists, so
  this is a lower bound and is flagged).
* Implied service level = P(D <= on-hand + order); expected lost sales
  = E[max(D - stock, 0)]; working capital = order x unit value.

Baselines: 4-week moving average and last-year-same-weeks, each with the same
catalog safety-stock policy, evaluated under the model's demand distribution so
that the comparison is like-for-like.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from supply_pipeline import distributions as dist
from supply_pipeline.config import Config

log = logging.getLogger(__name__)

SERIES_KEY = ["upc", "cedis"]


def moq_round_up(qty: np.ndarray, moq: np.ndarray) -> np.ndarray:
    qty = np.asarray(qty, dtype=float)
    moq = np.asarray(moq, dtype=float)
    return np.where(qty <= 0, 0.0, np.ceil(qty / moq) * moq)


def protection_demand(fc: pd.DataFrame, weeks: int) -> pd.DataFrame:
    """Sum the first ``weeks`` weekly quantile forecasts per series."""
    f = fc[fc["h"] <= weeks]
    qcols = [c for c in f.columns if c.startswith("q") and c[1:].isdigit()]
    d = f.groupby(SERIES_KEY, as_index=False)[qcols].sum()
    d["weeks_available"] = f.groupby(SERIES_KEY).size().to_numpy()
    return d


def naive_demands(weekly: pd.DataFrame, origin: pd.Timestamp, weeks: int) -> pd.DataFrame:
    """Moving-average and last-year baselines for demand over the protection period."""
    hist = weekly[weekly["week_start"] <= origin].dropna(subset=["y"])
    ma4 = hist.sort_values("week_start").groupby(SERIES_KEY)["y"].apply(lambda s: s.tail(4).mean()).rename("ma4_weekly")
    targets = [origin + pd.Timedelta(weeks=h) for h in range(1, weeks + 1)]
    ly_weeks = [t - pd.Timedelta(weeks=52) for t in targets]
    ly = weekly[weekly["week_start"].isin(ly_weeks)].groupby(SERIES_KEY)["y"].sum(min_count=1).rename("ly_demand")
    out = ma4.reset_index().merge(ly.reset_index(), on=SERIES_KEY, how="left")
    out["ma4_demand"] = out["ma4_weekly"] * weeks
    out["ly_demand"] = out["ly_demand"].fillna(out["ma4_demand"])
    return out


def build_orders(
    fc: pd.DataFrame, series: pd.DataFrame, weekly: pd.DataFrame, inv: pd.DataFrame, daily: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    origin = pd.Timestamp(cfg.data.last_complete_week_start)
    as_of = pd.Timestamp(cfg.data.as_of)
    period_start = origin + pd.Timedelta(weeks=1)
    lead = int(series["lead_time_days"].iloc[0])
    protection_days = lead + cfg.orders.review_period_days
    weeks = int(np.ceil(protection_days / 7))

    d = protection_demand(fc, weeks)
    d = d.merge(
        series[
            SERIES_KEY
            + [
                "cluster",
                "abc_class",
                "xyz_class",
                "lead_time_days",
                "moq",
                "safety_stock_days",
                "is_discontinued",
                "is_high_promo",
                "on_hand_as_of",
                "mean_weekly",
            ]
        ],
        on=SERIES_KEY,
        how="left",
    )
    d = d.merge(fc[fc["h"] == 1][SERIES_KEY + ["model", "is_cold_start"]], on=SERIES_KEY, how="left")
    d = d.merge(naive_demands(weekly, origin, weeks), on=SERIES_KEY, how="left")

    # Unit value: recent shelf price x cost ratio.
    price = (
        weekly[weekly["week_start"] > origin - pd.Timedelta(weeks=4)]
        .groupby(SERIES_KEY)["price_mean"]
        .mean()
        .rename("unit_price")
    )
    d = d.merge(price.reset_index(), on=SERIES_KEY, how="left")
    d["unit_value"] = d["unit_price"] * cfg.orders.cost_ratio

    # Projected on-hand at the start of the protection period.
    gap_sales = (
        daily[(daily["date"] > as_of) & (daily["date"] < period_start)]
        .groupby(SERIES_KEY)["sell_out_pzs"]
        .sum()
        .rename("sales_after_snapshot")
    )
    d = d.merge(gap_sales.reset_index(), on=SERIES_KEY, how="left")
    d["sales_after_snapshot"] = d["sales_after_snapshot"].fillna(0.0)
    d["no_inventory_snapshot"] = d["on_hand_as_of"].isna()
    d["on_hand_as_of"] = d["on_hand_as_of"].fillna(0.0)
    d["on_hand_projected"] = (d["on_hand_as_of"] - d["sales_after_snapshot"]).clip(lower=0.0)

    # Demand distribution over the protection period.
    mu, sigma = dist.lognormal_params(d["q50"].to_numpy(), d["q90"].to_numpy())
    d["demand_p50"] = d["q50"]
    d["demand_p90"] = d["q90"]
    d["demand_mean"] = dist.mean(mu, sigma)
    d["daily_mean"] = d["demand_p50"] / protection_days
    d["service_level_target"] = d["abc_class"].map(cfg.orders.service_level).fillna(min(cfg.orders.service_level.values()))
    d["safety_stock_policy"] = d["safety_stock_days"] * d["daily_mean"]
    d["safety_stock_quantile"] = dist.quantile(mu, sigma, d["service_level_target"].to_numpy()) - d["demand_p50"]
    d["safety_stock"] = np.maximum(d["safety_stock_policy"], d["safety_stock_quantile"])
    d["safety_stock_binding"] = np.where(
        d["safety_stock_policy"] >= d["safety_stock_quantile"], "policy_days", "forecast_quantile"
    )
    d["order_up_to"] = d["demand_p50"] + d["safety_stock"]
    d["order_raw"] = (d["order_up_to"] - d["on_hand_projected"]).clip(lower=0.0)
    d["order_qty"] = moq_round_up(d["order_raw"], d["moq"])
    # Discontinued lines: recommend zero and send to the planner.
    d.loc[d["is_discontinued"], "order_qty"] = 0.0

    stock = d["on_hand_projected"] + d["order_qty"]
    d["implied_service_level"] = 1.0 - dist.prob_demand_exceeds(stock.to_numpy(), mu, sigma)
    d["expected_lost_sales"] = dist.expected_shortfall(stock.to_numpy(), mu, sigma)
    d["expected_fill_rate"] = 1.0 - d["expected_lost_sales"] / d["demand_mean"].clip(lower=1e-6)
    d["working_capital"] = d["order_qty"] * d["unit_value"]
    d["stock_value_after_order"] = stock * d["unit_value"]
    d["days_of_cover_after_order"] = stock / d["daily_mean"].clip(lower=1e-6)

    # Variant: forecast-driven safety stock only (drop the catalog day rule) - shows the stock the
    # service-level targets actually require.
    d["fq_order_up_to"] = d["demand_p50"] + np.maximum(d["safety_stock_quantile"], 0.0)
    fq_raw = (d["fq_order_up_to"] - d["on_hand_projected"]).clip(lower=0.0)
    d["fq_order_qty"] = np.where(d["is_discontinued"], 0.0, moq_round_up(fq_raw, d["moq"]))
    fq_stock = d["on_hand_projected"] + d["fq_order_qty"]
    d["fq_service_level"] = 1.0 - dist.prob_demand_exceeds(fq_stock.to_numpy(), mu, sigma)
    d["fq_expected_lost_sales"] = dist.expected_shortfall(fq_stock.to_numpy(), mu, sigma)
    d["fq_working_capital"] = d["fq_order_qty"] * d["unit_value"]
    d["target_stock_value"] = d["order_up_to"] * d["unit_value"]
    d["fq_target_stock_value"] = d["fq_order_up_to"] * d["unit_value"]

    # Baselines under the same distribution and policy.
    for name, dem in (("ma4", d["ma4_demand"]), ("ly", d["ly_demand"])):
        ss = d["safety_stock_days"] * dem / protection_days
        raw = (dem + ss - d["on_hand_projected"]).clip(lower=0.0)
        qty = moq_round_up(raw, d["moq"])
        qty = np.where(d["is_discontinued"], 0.0, qty)
        st = d["on_hand_projected"] + qty
        d[f"{name}_order_qty"] = qty
        d[f"{name}_service_level"] = 1.0 - dist.prob_demand_exceeds(st.to_numpy(), mu, sigma)
        d[f"{name}_expected_lost_sales"] = dist.expected_shortfall(st.to_numpy(), mu, sigma)
        d[f"{name}_working_capital"] = qty * d["unit_value"]
    d["delta_working_capital_vs_ma4"] = d["working_capital"] - d["ma4_working_capital"]
    d["delta_service_level_vs_ma4"] = d["implied_service_level"] - d["ma4_service_level"]
    d["delta_lost_sales_vs_ma4"] = d["expected_lost_sales"] - d["ma4_expected_lost_sales"]

    flags = []
    for r in d.itertuples(index=False):
        f = []
        if r.is_cold_start:
            f.append("cold_start")
        if r.is_discontinued:
            f.append("discontinued_review")
        if r.is_high_promo:
            f.append("high_promo")
        if r.no_inventory_snapshot:
            f.append("no_inventory_snapshot")
        if r.weeks_available < weeks:
            f.append("short_horizon")
        flags.append(";".join(f))
    d["flags"] = flags
    d["origin"] = origin
    d["protection_period_days"] = protection_days
    return d


ORDER_COLUMNS = [
    "upc",
    "cedis",
    "cluster",
    "model",
    "origin",
    "protection_period_days",
    "on_hand_as_of",
    "sales_after_snapshot",
    "on_hand_projected",
    "demand_p50",
    "demand_p90",
    "demand_mean",
    "service_level_target",
    "safety_stock_policy",
    "safety_stock_quantile",
    "safety_stock",
    "safety_stock_binding",
    "order_up_to",
    "moq",
    "order_qty",
    "implied_service_level",
    "expected_fill_rate",
    "expected_lost_sales",
    "days_of_cover_after_order",
    "unit_value",
    "working_capital",
    "stock_value_after_order",
    "ma4_order_qty",
    "ma4_service_level",
    "ma4_expected_lost_sales",
    "ma4_working_capital",
    "ly_order_qty",
    "ly_service_level",
    "ly_working_capital",
    "delta_working_capital_vs_ma4",
    "delta_service_level_vs_ma4",
    "delta_lost_sales_vs_ma4",
    "fq_order_up_to",
    "fq_order_qty",
    "fq_service_level",
    "fq_expected_lost_sales",
    "fq_working_capital",
    "target_stock_value",
    "fq_target_stock_value",
    "flags",
]


def summarize(d: pd.DataFrame, by: list[str] | None) -> pd.DataFrame:
    """Portfolio (``by=None``) or per-group totals with demand-weighted service levels."""
    keys = d[by[0]] if by else pd.Series("portfolio", index=d.index, name="_all")
    g = d.groupby(by) if by else d.assign(_all="portfolio").groupby("_all")

    def demand_weighted(col: str) -> np.ndarray:
        num = (d[col] * d["demand_mean"]).groupby(keys).sum()
        den = d["demand_mean"].groupby(keys).sum()
        return (num / den).to_numpy()

    out = g.agg(
        n_series=("upc", "size"),
        order_units=("order_qty", "sum"),
        working_capital=("working_capital", "sum"),
        expected_lost_units=("expected_lost_sales", "sum"),
        demand_mean_units=("demand_mean", "sum"),
        ma4_order_units=("ma4_order_qty", "sum"),
        ma4_working_capital=("ma4_working_capital", "sum"),
        ma4_expected_lost_units=("ma4_expected_lost_sales", "sum"),
        ly_order_units=("ly_order_qty", "sum"),
        ly_working_capital=("ly_working_capital", "sum"),
        fq_order_units=("fq_order_qty", "sum"),
        fq_working_capital=("fq_working_capital", "sum"),
        fq_expected_lost_units=("fq_expected_lost_sales", "sum"),
        target_stock_value=("target_stock_value", "sum"),
        fq_target_stock_value=("fq_target_stock_value", "sum"),
        lines_below_90=("implied_service_level", lambda s: int((s < 0.9).sum())),
        ma4_lines_below_90=("ma4_service_level", lambda s: int((s < 0.9).sum())),
        fq_lines_below_90=("fq_service_level", lambda s: int((s < 0.9).sum())),
    ).reset_index()
    out["service_level_weighted"] = demand_weighted("implied_service_level")
    out["ma4_service_level_weighted"] = demand_weighted("ma4_service_level")
    out["fq_service_level_weighted"] = demand_weighted("fq_service_level")
    out["fill_rate"] = 1 - out["expected_lost_units"] / out["demand_mean_units"]
    out["ma4_fill_rate"] = 1 - out["ma4_expected_lost_units"] / out["demand_mean_units"]
    out["fq_fill_rate"] = 1 - out["fq_expected_lost_units"] / out["demand_mean_units"]
    out["delta_working_capital_vs_ma4"] = out["working_capital"] - out["ma4_working_capital"]
    out["lost_units_reduction_vs_ma4"] = out["ma4_expected_lost_units"] - out["expected_lost_units"]
    out["target_stock_value_freed_by_fq"] = out["target_stock_value"] - out["fq_target_stock_value"]
    return out


def run(cfg: Config) -> None:
    p = cfg.paths
    series = pd.read_parquet(p.interim_dir / "series.parquet")
    weekly = pd.read_parquet(p.interim_dir / "weekly.parquet")
    inv = pd.read_parquet(p.interim_dir / "inventory_cedis_daily.parquet")
    daily = pd.read_parquet(p.interim_dir / "daily.parquet")
    fc = pd.read_csv(p.output_dir / f"forecast_{cfg.data.last_complete_week_start}.csv", parse_dates=["target_week", "origin"])
    fc["cedis"] = fc["cedis"].astype("string")

    orders = build_orders(fc, series, weekly, inv, daily, cfg)
    out = orders[ORDER_COLUMNS].sort_values(["cluster", "upc", "cedis"])
    out_path = p.output_dir / f"supply_order_{cfg.data.as_of}.csv"
    out.to_csv(out_path, index=False, float_format="%.4f")

    portfolio = summarize(orders, None)
    by_cluster = summarize(orders, ["cluster"])
    by_cedis = summarize(orders, ["cedis"])
    portfolio.to_csv(p.tables_dir / "order_summary_portfolio.csv", index=False)
    by_cluster.to_csv(p.tables_dir / "order_summary_cluster.csv", index=False)
    by_cedis.to_csv(p.tables_dir / "order_summary_cedis.csv", index=False)
    log.info("orders written to %s (%d lines, %d with flags)", out_path.name, len(out), int((out["flags"] != "").sum()))
    log.info("portfolio:\n%s", portfolio.round(3).T.to_string())
