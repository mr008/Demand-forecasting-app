"""Stage 'risk': stock-out / service-level risk detection (Track B).

Alert definition (what planners see)
-----------------------------------
For a SKU at a distribution center on day *d*, an alert says: "on current
stock and the demand we expect, this SKU is likely to run short at stores
within the next lead time (7 days)". Three scorers produce it:

1. ``cover``   days of cover = on-hand / expected daily demand, compared with
               lead time + the SKU's safety-stock days.
2. ``prob``    P(demand over the next 7 days > on-hand), from the selected
               forecast's quantiles (lognormal fit). Alert if above
               ``risk.prob_threshold``.
3. ``iforest`` Isolation Forest over cover, probability, sales-vs-forecast
               ratios and on-hand trend; flags unusual combinations.

Label
-----
A cedis-day is a *stock-out event* when at least ``data.stockout_store_share``
of its reporting stores have on-hand <= 0. The label for an alert issued on day
*d* is "any event in d+1 .. d+7" and is only evaluable when those seven days are
present in the inventory file for that series.

Outputs: ``reports/tables/risk_eval_methods.csv``, ``risk_threshold_sweep.csv``,
``risk_lead_time.csv``, ``data/interim/risk_scored_window.parquet`` and the
as-of alert list ``data/output/risk_alerts_<as_of>.csv``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from supply_pipeline import distributions as dist
from supply_pipeline.config import Config
from supply_pipeline.metrics import alert_metrics, lead_time_to_alert

log = logging.getLogger(__name__)

SERIES_KEY = ["upc", "cedis"]
LOOKAHEAD_DAYS = 7  # = lead time for every SKU in the catalog
IFOREST_FEATURES = ["log_cover", "p_stockout_7d", "sales_ratio_1d", "sales_ratio_7d", "on_hand_change_3d", "promo_flag"]


# --------------------------------------------------------------------------- demand at daily grain
def weekday_shares(daily: pd.DataFrame, end: pd.Timestamp, weeks: int = 26) -> pd.DataFrame:
    """Share of a week's sales that falls on each weekday, per series (uniform fallback)."""
    d = daily[(daily["date"] <= end) & (daily["date"] > end - pd.Timedelta(weeks=weeks)) & daily["sell_out_pzs"].notna()].copy()
    d["dow"] = d["date"].dt.dayofweek
    tot = d.groupby(SERIES_KEY + ["dow"])["sell_out_pzs"].sum().rename("dow_sum").reset_index()
    tot["share"] = tot["dow_sum"] / tot.groupby(SERIES_KEY)["dow_sum"].transform("sum")
    grid = tot[SERIES_KEY].drop_duplicates().merge(pd.DataFrame({"dow": range(7)}), how="cross")
    out = grid.merge(tot[SERIES_KEY + ["dow", "share"]], on=SERIES_KEY + ["dow"], how="left")
    out["share"] = out["share"].fillna(1 / 7)
    return out


def daily_forecast(fc: pd.DataFrame, shares: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Expand weekly quantile forecasts to daily q50/q90 per series for the given dates.

    Dates earlier than the first forecast week borrow the first forecast week
    (they sit inside the training window, where no forecast row exists).
    """
    fc = fc.copy()
    first_week = fc["target_week"].min()
    grid = fc[SERIES_KEY].drop_duplicates().merge(pd.DataFrame({"date": dates}), how="cross")
    grid["week_start"] = (grid["date"] - pd.to_timedelta(grid["date"].dt.dayofweek, unit="D")).dt.normalize()
    grid["week_start"] = grid["week_start"].clip(lower=first_week)
    grid["dow"] = grid["date"].dt.dayofweek
    fcw = fc[SERIES_KEY + ["target_week", "q50", "q90"]].rename(columns={"target_week": "week_start"})
    out = grid.merge(fcw, on=SERIES_KEY + ["week_start"], how="left").merge(shares, on=SERIES_KEY + ["dow"], how="left")
    out["share"] = out["share"].fillna(1 / 7)
    out["d50"] = out["q50"] * out["share"]
    out["d90"] = out["q90"] * out["share"]
    return out[SERIES_KEY + ["date", "d50", "d90"]]


# --------------------------------------------------------------------------- scoring
def score_days(inv: pd.DataFrame, dfc: pd.DataFrame, daily: pd.DataFrame, series: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compute cover, stock-out probability and anomaly features for every (series, day) in ``inv``."""
    # Expected demand over the next LOOKAHEAD_DAYS days (comonotonic sum of daily quantiles).
    dfc = dfc.sort_values(SERIES_KEY + ["date"])
    g = dfc.groupby(SERIES_KEY, sort=False)
    dfc["fwd50"] = g["d50"].transform(lambda s: s[::-1].rolling(LOOKAHEAD_DAYS, min_periods=1).sum()[::-1].shift(-1))
    dfc["fwd90"] = g["d90"].transform(lambda s: s[::-1].rolling(LOOKAHEAD_DAYS, min_periods=1).sum()[::-1].shift(-1))

    x = inv.merge(dfc, on=SERIES_KEY + ["date"], how="inner")
    x = x.merge(series[SERIES_KEY + ["cluster", "lead_time_days", "safety_stock_days"]], on=SERIES_KEY, how="left")
    x = x.merge(daily[SERIES_KEY + ["date", "sell_out_pzs", "promo_flag"]], on=SERIES_KEY + ["date"], how="left")

    x["daily_mean_7d"] = x["fwd50"] / LOOKAHEAD_DAYS
    x["cover_days"] = x["on_hand"] / x["daily_mean_7d"].clip(lower=1e-6)
    x["cover_threshold"] = x["lead_time_days"] + x["safety_stock_days"]
    mu, sigma = dist.lognormal_params(x["fwd50"].to_numpy(), x["fwd90"].to_numpy())
    x["p_stockout_7d"] = dist.prob_demand_exceeds(x["on_hand"].to_numpy(), mu, sigma)

    # Anomaly features: sales vs forecast, on-hand trend.
    x = x.sort_values(SERIES_KEY + ["date"])
    gx = x.groupby(SERIES_KEY, sort=False)
    x["sales_ratio_1d"] = x["sell_out_pzs"] / x["d50"].clip(lower=1e-6)
    x["sales_7d"] = gx["sell_out_pzs"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    x["exp_7d"] = gx["d50"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    x["sales_ratio_7d"] = x["sales_7d"] / x["exp_7d"].clip(lower=1e-6)
    x["on_hand_change_3d"] = (x["on_hand"] - gx["on_hand"].shift(3)) / gx["on_hand"].shift(3).clip(lower=1.0)
    x["on_hand_change_3d"] = x["on_hand_change_3d"].fillna(0.0)
    x["log_cover"] = np.log1p(x["cover_days"].clip(upper=365))
    x["promo_flag"] = x["promo_flag"].fillna(0)
    x["sales_ratio_1d"] = x["sales_ratio_1d"].fillna(1.0).clip(upper=10)
    x["sales_ratio_7d"] = x["sales_ratio_7d"].fillna(1.0).clip(upper=10)

    x["alert_cover"] = x["cover_days"] < x["cover_threshold"]
    x["alert_prob"] = x["p_stockout_7d"] > cfg.risk.prob_threshold
    iso = IsolationForest(contamination=cfg.risk.contamination, random_state=cfg.forecast.seed)
    x["iforest_score"] = -iso.fit(x[IFOREST_FEATURES]).score_samples(x[IFOREST_FEATURES])
    x["alert_iforest"] = iso.predict(x[IFOREST_FEATURES]) == -1
    return x


def label_lookahead(scored: pd.DataFrame, share_threshold: float) -> pd.DataFrame:
    """Event flag per day and the 'event within the next 7 days' label (NaN if not fully observed)."""
    s = scored.sort_values(SERIES_KEY + ["date"]).copy()
    s["event"] = s["stockout_store_share"] >= share_threshold
    labels = []
    for _, g in s.groupby(SERIES_KEY, sort=False):
        ev = pd.Series(g["event"].to_numpy(), index=g["date"])
        lab = []
        for d in g["date"]:
            window = ev[(ev.index > d) & (ev.index <= d + pd.Timedelta(days=LOOKAHEAD_DAYS))]
            lab.append(float(window.any()) if len(window) == LOOKAHEAD_DAYS else np.nan)
        labels.append(pd.Series(lab, index=g.index))
    s["label_7d"] = pd.concat(labels)
    prev = s.groupby(SERIES_KEY, sort=False)["event"].shift(1)
    s["episode_onset"] = s["event"] & (prev == False)  # noqa: E712 - NaN (no prior day) must not count
    return s


def evaluate(labeled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ev = labeled.dropna(subset=["label_7d"])
    truth = ev["label_7d"].astype(bool).to_numpy()
    rows: list[dict[str, object]] = []
    for method in ("cover", "prob", "iforest"):
        m = alert_metrics(ev[f"alert_{method}"].to_numpy(), truth)
        rows.append({"method": method, **m, "median_cover_at_alert": float(ev.loc[ev[f"alert_{method}"], "cover_days"].median())})
    methods = pd.DataFrame(rows)[
        ["method", "n", "n_alerts", "n_events", "precision", "recall", "f1", "false_alarm_rate", "median_cover_at_alert"]
    ]

    sweep = []
    for k in (3, 5, 7, 10, 14, 21, 28):
        m = alert_metrics((ev["cover_days"] < k).to_numpy(), truth)
        sweep.append({"method": "cover", "threshold": k, **m})
    for p in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
        m = alert_metrics((ev["p_stockout_7d"] > p).to_numpy(), truth)
        sweep.append({"method": "prob", "threshold": p, **m})
    sweep_df = pd.DataFrame(sweep)

    onsets = labeled[labeled["episode_onset"]][SERIES_KEY + ["date"]]
    lt_rows = []
    for method in ("cover", "prob", "iforest"):
        alerts = labeled[labeled[f"alert_{method}"]][SERIES_KEY + ["date"]]
        lt = lead_time_to_alert(alerts, onsets, LOOKAHEAD_DAYS) if len(onsets) else pd.Series(dtype=float)
        lt_rows.append(
            {
                "method": method,
                "n_onsets": len(onsets),
                "n_onsets_alerted": int(lt.notna().sum()),
                "median_lead_time_days": float(lt.median()) if lt.notna().any() else np.nan,
            }
        )
    return methods, sweep_df, pd.DataFrame(lt_rows)


def severity(row: pd.Series) -> str:
    if row["p_stockout_7d"] > 0.75 or row["cover_days"] < row["lead_time_days"]:
        return "high"
    if row["alert_cover"] or row["alert_prob"] or row["alert_iforest"]:
        return "medium"
    return "none"


# --------------------------------------------------------------------------- stage
def run(cfg: Config) -> None:
    p = cfg.paths
    inv = pd.read_parquet(p.interim_dir / "inventory_cedis_daily.parquet")
    daily = pd.read_parquet(p.interim_dir / "daily.parquet")
    series = pd.read_parquet(p.interim_dir / "series.parquet")
    fc_eval = pd.read_csv(p.output_dir / f"forecast_{cfg.risk.eval_origin}.csv", parse_dates=["target_week"])
    fc_final = pd.read_csv(p.output_dir / f"forecast_{cfg.data.last_complete_week_start}.csv", parse_dates=["target_week"])
    inv["cedis"] = inv["cedis"].astype("string")
    for df in (fc_eval, fc_final):
        df["cedis"] = df["cedis"].astype("string")

    # --- evaluation on the inventory window, using the forecast available at eval_origin
    eval_origin = pd.Timestamp(cfg.risk.eval_origin)
    dates = pd.date_range(inv["date"].min(), inv["date"].max() + pd.Timedelta(days=LOOKAHEAD_DAYS), freq="D")
    shares = weekday_shares(daily, eval_origin + pd.Timedelta(days=6))
    dfc = daily_forecast(fc_eval, shares, dates)
    scored = score_days(inv, dfc, daily, series, cfg)
    labeled = label_lookahead(scored, cfg.data.stockout_store_share)
    methods, sweep, lead = evaluate(labeled)
    labeled.to_parquet(p.interim_dir / "risk_scored_window.parquet", index=False)
    methods.to_csv(p.tables_dir / "risk_eval_methods.csv", index=False)
    sweep.to_csv(p.tables_dir / "risk_threshold_sweep.csv", index=False)
    lead.to_csv(p.tables_dir / "risk_lead_time.csv", index=False)
    n_ev_series = labeled[labeled["event"]].groupby(SERIES_KEY).ngroups
    log.info(
        "window: %d series-days scored, %d evaluable, %d event days over %d series, %d episode onsets",
        len(labeled),
        int(labeled["label_7d"].notna().sum()),
        int(labeled["event"].sum()),
        n_ev_series,
        int(labeled["episode_onset"].sum()),
    )
    log.info("methods:\n%s", methods.round(3).to_string(index=False))

    # --- as-of alerts with the final forecast
    as_of = pd.Timestamp(cfg.data.as_of)
    inv_asof = inv[inv["date"] == as_of]
    dates_asof = pd.date_range(as_of, as_of + pd.Timedelta(days=LOOKAHEAD_DAYS), freq="D")
    daily_forecast(fc_final, weekday_shares(daily, as_of), dates_asof)
    # Need a few trailing days for the trend/ratio features.
    inv_tail = inv[(inv["date"] > as_of - pd.Timedelta(days=7)) & (inv["date"] <= as_of)]
    dfc_tail = daily_forecast(fc_final, weekday_shares(daily, as_of), pd.date_range(as_of - pd.Timedelta(days=7), dates_asof[-1]))
    scored_asof = score_days(inv_tail, dfc_tail, daily, series, cfg)
    scored_asof = scored_asof[scored_asof["date"] == as_of].copy()
    scored_asof["severity"] = scored_asof.apply(severity, axis=1)
    cols = SERIES_KEY + [
        "cluster",
        "date",
        "on_hand",
        "stores_reporting",
        "stockout_store_share",
        "daily_mean_7d",
        "fwd50",
        "fwd90",
        "cover_days",
        "cover_threshold",
        "p_stockout_7d",
        "alert_cover",
        "alert_prob",
        "alert_iforest",
        "iforest_score",
        "severity",
    ]
    out = scored_asof[cols].rename(columns={"fwd50": "demand_7d_p50", "fwd90": "demand_7d_p90"})
    out = out.sort_values(["severity", "p_stockout_7d"], ascending=[True, False])
    out.to_csv(p.output_dir / f"risk_alerts_{as_of.date()}.csv", index=False)
    log.info("as-of %s alerts: %s", as_of.date(), out["severity"].value_counts().to_dict())
    _ = inv_asof  # retained for clarity: the as-of snapshot is the last row of inv_tail per series
