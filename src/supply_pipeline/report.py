"""Stage 'report': figures and ``reports/summary.md``.

Reads only the tables and parquet files written by earlier stages, so the
report always reflects the last run.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from supply_pipeline import plots
from supply_pipeline.backtest import apply_calibration
from supply_pipeline.config import Config

log = logging.getLogger(__name__)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(floatfmt.format(v) if pd.notna(v) else "")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def build_figures(cfg: Config) -> dict[str, Path]:
    p = cfg.paths
    t = p.tables_dir
    f = p.figures_dir
    weekly = pd.read_parquet(p.interim_dir / "weekly.parquet")
    series = pd.read_parquet(p.interim_dir / "series.parquet")
    pred = pd.read_parquet(p.interim_dir / "backtest_predictions.parquet")
    calib = pd.read_csv(t / "interval_calibration.csv")
    pred_cal = apply_calibration(pred, calib, cfg.forecast.quantiles)
    selection = pd.read_csv(t / "model_selection.csv")
    fc = pd.read_csv(p.output_dir / f"forecast_{cfg.data.last_complete_week_start}.csv", parse_dates=["target_week"])
    scored = pd.read_parquet(p.interim_dir / "risk_scored_window.parquet")
    figs = {
        "model_comparison": plots.model_comparison(
            pd.read_csv(t / "backtest_metrics_cluster.csv"), selection, f / "model_comparison.png"
        ),
        "wape_by_horizon": plots.wape_by_horizon(pd.read_csv(t / "backtest_metrics_cluster_h.csv"), f / "wape_by_horizon.png"),
        "coverage": plots.coverage_calibration(
            pd.read_csv(t / "backtest_metrics_holdout_raw.csv"),
            pd.read_csv(t / "backtest_metrics_holdout_calibrated.csv"),
            f / "coverage_calibration.png",
        ),
        "cluster_fva": plots.cluster_forecast_vs_actual(pred_cal, weekly, selection, f / "cluster_forecast_vs_actual.png"),
        "residuals": plots.residuals(pred, selection, f / "residuals.png"),
        "risk": plots.risk_window(scored, pd.read_csv(t / "risk_threshold_sweep.csv"), f / "risk_window.png"),
        "examples": plots.series_examples(weekly, fc, series, f / "series_examples.png"),
        "portfolio": plots.portfolio_forecast(weekly, fc, f / "portfolio_forecast.png"),
        "orders": plots.orders_summary(pd.read_csv(t / "order_summary_cluster.csv"), f / "orders_summary.png"),
        "blind": f / "blind_test.png",
        "risk_timeline": plots.risk_timeline(scored, f / "risk_timeline.png"),
        "inv_cov": plots.inventory_coverage(
            pd.read_csv(t / "coverage_inventory_by_date.csv", index_col=0), f / "inventory_coverage.png"
        ),
    }
    return figs


def build_summary(cfg: Config, figs: dict[str, Path]) -> str:
    p = cfg.paths
    t = p.tables_dir
    series = pd.read_parquet(p.interim_dir / "series.parquet")
    overall = pd.read_csv(t / "backtest_metrics_overall.csv")
    cluster = pd.read_csv(t / "backtest_metrics_cluster.csv")
    selection = pd.read_csv(t / "model_selection.csv")
    hold_raw = pd.read_csv(t / "backtest_metrics_holdout_calibrated_overall.csv")
    hold_cal_cluster = pd.read_csv(t / "backtest_metrics_holdout_calibrated.csv")
    risk_methods = pd.read_csv(t / "risk_eval_methods.csv")
    risk_lead = pd.read_csv(t / "risk_lead_time.csv")
    alerts = pd.read_csv(p.output_dir / f"risk_alerts_{cfg.data.as_of}.csv")
    orders = pd.read_csv(p.output_dir / f"supply_order_{cfg.data.as_of}.csv")
    port = pd.read_csv(t / "order_summary_portfolio.csv").iloc[0]
    by_cluster = pd.read_csv(t / "order_summary_cluster.csv")
    scored = pd.read_parquet(p.interim_dir / "risk_scored_window.parquet")
    blind_overall = pd.read_csv(t / "blind_test_overall.csv")
    blind_sel = pd.read_csv(t / "blind_test_selection.csv")
    pred = pd.read_parquet(p.interim_dir / "backtest_predictions.parquet")
    sealed = pred[pred["fold"] == pred["fold"].max()]
    blind_origin = sealed["origin"].iloc[0].date()
    blind_start, blind_end = sealed["target_week"].min().date(), sealed["target_week"].max().date()
    blind_w = float(np.average(blind_sel["blind_wape_pre_registered"], weights=blind_sel["n"]))
    blind_w_best = float(np.average(blind_sel["blind_wape_best"], weights=blind_sel["n"]))

    rel = lambda path: Path("figures") / path.name  # noqa: E731
    n_series = len(series)
    sel_txt = ", ".join(f"{r.cluster}: {r.selected_model}" for r in selection.itertuples())
    lgbm = overall.set_index("model").loc["lgbm"]
    best_naive = overall[overall["model"].isin(["ma4", "seasonal_naive"])].sort_values("wape").iloc[0]
    n_events_series = scored[scored["event"]].groupby(["upc", "cedis"]).ngroups
    per_series = (
        scored.assign(any_alert=scored["alert_cover"] | scored["alert_prob"] | scored["alert_iforest"])
        .groupby(["upc", "cedis"])
        .agg(events=("event", "sum"), alerts=("any_alert", "sum"), cover=("cover_days", "median"))
    )
    silent = per_series[(per_series["events"] > 0) & (per_series["alerts"] == 0)]
    n_silent = len(silent)
    if n_silent:
        silent_text = (
            f"{n_silent} of the {n_events_series} series with stock-outs never triggered any alert. Their DC held a median of "
            f"{float(silent['cover'].median()):.0f} days of cover while at least a quarter of their stores were empty: the stock "
            "existed but was not reaching the shelves. That is an allocation problem between the DC and its stores, invisible "
            "to any DC-level signal, and the strongest argument for adding store-level cover to the alert once store stock "
            "history is available beyond 21 days."
        )
    else:
        silent_text = f"Every one of the {n_events_series} series with stock-outs triggered at least one alert during the window."
    flagged = orders[orders["flags"].fillna("") != ""]

    md = f"""# Demand forecasting & supply order - results summary

Generated by `python -m supply_pipeline run`. Numbers below come from `reports/tables/`; figures from `reports/figures/`.

## 1. Scope and data

- {n_series} SKU x distribution-center series (67 UPCs, 6 CEDIS), daily sell-out 2024-03-18 .. 2026-04-10, modelled at weekly grain (107 complete ISO weeks).
- Store inventory covers only 2026-03-20 .. 2026-04-09; the last day with full coverage is **{cfg.data.as_of}** and is used as "current on-hand". The final week of the file has 4 UPCs only (see `figures/inventory_coverage.png`).
- Clusters (ABC x XYZ): {", ".join(f"{k} = {v}" for k, v in series["cluster"].value_counts().sort_index().items())}.
- Flags at the as-of date: {int(series["is_discontinued"].sum())} discontinued series, {int((~series["has_inventory"]).sum())} without an inventory snapshot, {int(series["is_cold_start"].sum())} cold-start (< {cfg.data.cold_start_weeks} weeks).

Data policies (all configurable in `config.toml`): catalog and inventory are de-duplicated on `upc` (three UPCs carry two item numbers); negative store on-hand is clipped to zero for stock maths but counted as a stock-out signal; missing days stay missing (never zero-filled); daily outliers (robust z > {cfg.data.outlier_mad_z:g}) are winsorised for model training only; future price and promotions are treated as unknown, so models use lagged price/promo plus known calendar effects (Mexican federal holidays, quincena paydays, Semana Santa, El Buen Fin, December peak).

## 2. Forecasting (Track A)

**Protocol.** Expanding-window backtest, {cfg.forecast.backtest_folds} folds, origins every {cfg.forecast.backtest_step_weeks} weeks from 2025-07-21 to 2026-02-02, horizon {cfg.forecast.horizon_weeks} weeks, quantiles {list(cfg.forecast.quantiles)}. Metrics are against raw actuals. Series with fewer than {cfg.data.cold_start_weeks} weeks of history at an origin are excluded from model comparison.

**Models.** Seasonal naive (52 wk), 4-week moving average, per-series ETS (additive damped trend, simulated intervals), and a global LightGBM quantile model (direct multi-horizon; lags 1-8 and 52, rolling means, lagged price/promo, calendar counts, cluster/DC/UPC categoricals; target scaled by recent level).

**Overall (all folds, all clusters):**

{md_table(overall[["model", "n", "wape", "mape", "bias", "pinball", "coverage_90", "coverage_80"]])}

**Per cluster (WAPE):**

{md_table(cluster.pivot(index="cluster", columns="model", values="wape").reset_index())}

**Selection.** {sel_txt}. Rule: lowest mean WAPE across folds unless a simpler or steadier model is within 0.02 WAPE. LightGBM reduces WAPE from {best_naive["wape"]:.3f} (best naive, {best_naive["model"]}) to {lgbm["wape"]:.3f} overall, with bias {lgbm["bias"]:+.3f}. Seasonal naive is strongly biased ({overall.set_index("model").loc["seasonal_naive", "bias"]:+.2f}): last year's level is not a usable guide for this portfolio.

**Uncertainty.** Raw LightGBM quantiles were over-confident (90% interval covered {_pct(lgbm["coverage_90"])} of actuals). A conformal-style width calibration per model x cluster is fitted on the first {cfg.forecast.backtest_folds - 3} folds and checked on the last 3:

{md_table(hold_raw[["model", "coverage_90", "coverage_80", "width_rel"]])}

Calibrated coverage by cluster for the selected models:

{md_table(hold_cal_cluster.merge(selection[["cluster", "selected_model"]], left_on=["cluster", "model"], right_on=["cluster", "selected_model"])[["cluster", "model", "wape", "bias", "coverage_90", "coverage_80"]])}

**Blind test.** The last replay (origin {blind_origin}, weeks {blind_start} to {blind_end}) is sealed; model selection and interval calibration are redone using only folds whose targets end before it, then every model is scored on the sealed weeks. The pre-registered choice held in {int(blind_sel["choice_held"].sum())} of {len(blind_sel)} clusters; its blind WAPE is {blind_w:.3f} against {blind_w_best:.3f} for the best model in hindsight (regret {blind_w - blind_w_best:.3f}). Where it did not hold (A-Z), ETS edged LightGBM by {abs(blind_sel.set_index("cluster").loc["A-Z", "regret"]):.3f} WAPE; the two are within the selection tolerance and the pipeline would switch automatically if that persists.

{md_table(blind_overall[["model", "n", "wape", "bias", "coverage_90_raw", "coverage_90", "coverage_80"]])}

![blind test]({rel(figs["blind"])})
![model comparison]({rel(figs["model_comparison"])})
![wape by horizon]({rel(figs["wape_by_horizon"])})
![coverage]({rel(figs["coverage"])})
![cluster forecast vs actual]({rel(figs["cluster_fva"])})
![residuals]({rel(figs["residuals"])})
![examples]({rel(figs["examples"])})

## 3. Stock-out risk (Track B)

**Alert definition (for planners).** "On current DC stock and the demand we expect, this SKU is likely to be short at stores within the next 7 days (the lead time)." Three scorers: (1) *cover rule* - days of cover below lead time + the SKU's safety-stock days; (2) *forecast probability* - P(demand over the next 7 days > on-hand) > {cfg.risk.prob_threshold:g}, from the calibrated quantile forecast; (3) *Isolation Forest* over cover, probability, sales-vs-forecast ratios and on-hand trend. Severity is *high* when P > 0.75 or cover < lead time, *medium* when any scorer fires.

**Label.** A DC-day is a stock-out event when >= {_pct(cfg.data.stockout_store_share)} of reporting stores have on-hand <= 0. Alerts issued on day d are scored against "any event in d+1..d+7", using the forecast a planner would have had at {cfg.risk.eval_origin}.

**What the window contains.** {int(scored["event"].sum())} event DC-days across {n_events_series} series; every one of them is *chronic* - those series are short for the whole 14-day window - so only {int(risk_lead["n_onsets"].iloc[0])} episode onsets exist to measure lead-time-to-alert on. Sell-out does not collapse during these events (stores keep selling what they get), so residual-based anomaly detection has little signal here.

{md_table(risk_methods)}

{md_table(risk_lead)}

The cover rule and the forecast-probability scorer both identify the chronic short series with high recall; precision is limited by DCs that hold thin stock but keep stores supplied. The threshold sweep (`risk_threshold_sweep.csv`, right panel of the figure) is the dial planners would use. Alerts as of {cfg.data.as_of}: {alerts["severity"].value_counts().to_dict()} (`data/output/risk_alerts_{cfg.data.as_of}.csv`).

![risk]({rel(figs["risk"])})

Per-series view of the window: shaded squares are days when at least 25% of the DC's stores were empty; markers are the days each scorer raised an alert. Rows above the dotted line had at least one stock-out; rows below are the most-alerted series that never did (the false alarms).

![risk timeline]({rel(figs["risk_timeline"])})

{silent_text}

## 4. Supply order recommendation

Weekly order-up-to policy over a {int(orders["protection_period_days"].iloc[0])}-day protection period (lead time 7 + review 7). Demand over the period comes from the selected model's calibrated quantiles (lognormal fit); safety stock is the larger of the catalog policy (`safety_stock_days` x daily demand) and the forecast-uncertainty stock needed for the ABC service-level target ({", ".join(f"{k} {_pct(v)}" for k, v in cfg.orders.service_level.items())}). Orders are rounded up to MOQ multiples; discontinued lines are set to zero and flagged; on-hand is projected from the {cfg.data.as_of} snapshot minus sell-out observed before the period starts. Baselines: 4-week moving average and last-year-same-weeks with the same policy, judged under the same demand distribution.

| | Recommended | 4-wk moving average | Last year |
|---|---|---|---|
| Order units | {port["order_units"]:,.0f} | {port["ma4_order_units"]:,.0f} | {port["ly_order_units"]:,.0f} |
| Working capital (MXN at shelf price x cost_ratio {cfg.orders.cost_ratio:g}) | {port["working_capital"]:,.0f} | {port["ma4_working_capital"]:,.0f} | {port["ly_working_capital"]:,.0f} |
| Demand-weighted implied service level | {_pct(port["service_level_weighted"])} | {_pct(port["ma4_service_level_weighted"])} | - |
| Expected fill rate over the period | {_pct(port["fill_rate"])} | {_pct(port["ma4_fill_rate"])} | - |
| Expected lost sales (units) | {port["expected_lost_units"]:,.0f} | {port["ma4_expected_lost_units"]:,.0f} | - |

Incremental working capital vs the moving-average policy: **{port["delta_working_capital_vs_ma4"]:+,.0f}**; expected lost-sales reduction: **{port["lost_units_reduction_vs_ma4"]:,.0f} units**. Portfolio averages hide the tail: the moving-average policy leaves **{int(port["ma4_lines_below_90"])}** SKU x DC lines below 90% service level for the coming period, the recommendation **{int(port["lines_below_90"])}** (discontinued lines, forced to zero, count among them).

**Where the money is.** DCs already hold about {(orders["on_hand_projected"].sum() / orders["demand_p50"].sum() * 14):.0f} days of demand, and the catalog's day-based safety stock (3/7/14 days) binds on {int((orders["safety_stock_binding"] == "policy_days").sum())} of {len(orders)} lines. A forecast-driven safety stock sized for the same ABC service-level targets (`fq_*` columns) would need a target stock worth **{port["fq_target_stock_value"]:,.0f}** instead of **{port["target_stock_value"]:,.0f}** - {port["target_stock_value_freed_by_fq"]:,.0f} MXN ({_pct(port["target_stock_value_freed_by_fq"] / port["target_stock_value"])}) of target inventory freed - at a demand-weighted service level of {_pct(port["fq_service_level_weighted"])} and fill rate {_pct(port["fq_fill_rate"])}, with {int(port["fq_lines_below_90"])} lines below 90%. The recommended order keeps the catalog policy as a floor (the conservative default); the forecast-only variant is the lever to negotiate with supply planning.

{md_table(by_cluster[["cluster", "n_series", "order_units", "ma4_order_units", "service_level_weighted", "ma4_service_level_weighted", "lines_below_90", "ma4_lines_below_90", "delta_working_capital_vs_ma4", "target_stock_value_freed_by_fq"]], "{:,.2f}")}

{len(flagged)} of {len(orders)} order lines carry a review flag ({flagged["flags"].value_counts().to_dict()}). Output: `data/output/supply_order_{cfg.data.as_of}.csv`.

![orders]({rel(figs["orders"])})
![portfolio]({rel(figs["portfolio"])})

## 5. Where not to trust the model

- **Discontinued lines** ({int(series["is_discontinued"].sum())} series): history is real, the future is zero. Orders are forced to 0 and flagged `discontinued_review`.
- **Cold-start series**: forecast with the 4-week moving average fallback and flagged; none at the current origin, but the rule is active in backtests.
- **High-promo weeks**: promotions are not known ahead, so promo-driven spikes are systematically under-forecast (bias is negative in A-Z). Planners should overlay the promo calendar; the `high_promo` flag marks series with >= 40% promo share over the last 8 weeks.
- **A-Z cluster**: highest WAPE ({cluster[(cluster["cluster"] == "A-Z") & (cluster["model"] == "lgbm")]["wape"].iloc[0]:.2f}) and widest calibrated bands; use the p90, not the median, for stock decisions.
- **Series with no inventory snapshot** ({int((~series["has_inventory"]).sum())}): on-hand assumed zero; flagged `no_inventory_snapshot`.
- **Stock-out labels** exist for 21 days only; Track B precision/recall are indicative, and lead-time-to-alert cannot be established from this file.
- **Receipts are not in the data**: projected on-hand ignores in-transit orders, so the recommendation is an upper bound when an order is already on its way.

## 6. Productionization sketch

Weekly scoring after the sales close (Monday), monthly retraining of LightGBM, ETS refit at every scoring run; drift monitors on rolling 4-week WAPE and bias per cluster against the backtest envelope, interval coverage against nominal, and input checks (schema, coverage per DC, negative-stock share); alerts feed a planner queue with the risk severity and the order line, and every override is logged to become training signal. See the deck for the architecture and next steps.
"""
    return md


def run(cfg: Config) -> None:
    figs = build_figures(cfg)
    md = build_summary(cfg, figs)
    out = cfg.paths.reports_dir / "summary.md"
    out.write_text(md, encoding="utf-8")
    log.info("wrote %s and %d figures", out, len(figs))
