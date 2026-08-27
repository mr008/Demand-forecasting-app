"""Stage 'forecast': produce forecasts with the selected model per cluster.

Two origins are forecast:

* the **final** origin (``data.last_complete_week_start``) feeding the supply
  order, and
* the **risk-evaluation** origin (``risk.eval_origin``), placed so that its
  horizon covers the 21-day inventory window; Track B is scored on the
  forecast a planner would actually have had at that time.

All models are run at both origins (for plots and comparison); the selected
model's rows are written to ``data/output/forecast_<origin>.csv``.
"""

from __future__ import annotations

import logging

import pandas as pd

from supply_pipeline.backtest import apply_calibration
from supply_pipeline.calendar_features import weekly_calendar
from supply_pipeline.config import Config
from supply_pipeline.models import FALLBACK_MODEL, Forecaster, build_models, make_future

log = logging.getLogger(__name__)

SERIES_KEY = ["upc", "cedis"]


def forecast_at_origin(
    models: dict[str, Forecaster], origin: pd.Timestamp, weekly: pd.DataFrame, weekly_cal: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    train = weekly[weekly["week_start"] <= origin]
    future = make_future(train, origin, cfg.forecast.horizon_weeks, weekly_cal)
    n_weeks = train.dropna(subset=["y"]).groupby(SERIES_KEY).size().rename("n_weeks_at_origin").reset_index()
    preds = []
    for name, model in models.items():
        p = model.fit_predict(train, future, cfg.forecast.quantiles)
        p["model"] = name
        preds.append(p)
        log.info("origin %s model %-14s -> %d rows", origin.date(), name, len(p))
    out = pd.concat(preds, ignore_index=True)
    out["origin"] = origin
    out = out.merge(n_weeks, on=SERIES_KEY, how="left")
    return out


def apply_selection(pred: pd.DataFrame, selection: pd.DataFrame, series: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Keep the selected model per cluster; cold-start series use the fallback model."""
    sel = selection.set_index("cluster")["selected_model"].to_dict()
    p = pred.merge(series[SERIES_KEY + ["cluster", "is_discontinued", "is_high_promo"]], on=SERIES_KEY, how="left")
    p["is_cold_start"] = p["n_weeks_at_origin"].fillna(0) < cfg.data.cold_start_weeks
    p["selected_model"] = p["cluster"].map(sel).fillna(FALLBACK_MODEL)
    p.loc[p["is_cold_start"], "selected_model"] = FALLBACK_MODEL
    chosen = p[p["model"] == p["selected_model"]].copy()
    return chosen.drop(columns=["selected_model"])


def run(cfg: Config) -> None:
    p = cfg.paths
    weekly = pd.read_parquet(p.interim_dir / "weekly.parquet")
    calendar = pd.read_parquet(p.interim_dir / "calendar.parquet")
    series = pd.read_parquet(p.interim_dir / "series.parquet")
    selection = pd.read_csv(p.tables_dir / "model_selection.csv")
    calib = pd.read_csv(p.tables_dir / "interval_calibration.csv")
    weekly_cal = weekly_calendar(calendar)
    models = build_models(cfg)

    origins = {
        "final": pd.Timestamp(cfg.data.last_complete_week_start),
        "risk_eval": pd.Timestamp(cfg.risk.eval_origin),
    }
    all_preds = []
    for label, origin in origins.items():
        pred = forecast_at_origin(models, origin, weekly, weekly_cal, cfg)
        pred["origin_label"] = label
        pred = pred.merge(series[SERIES_KEY + ["cluster"]], on=SERIES_KEY, how="left")
        pred = apply_calibration(pred, calib, cfg.forecast.quantiles).drop(columns=["cluster"])
        all_preds.append(pred)
        chosen = apply_selection(pred, selection, series, cfg)
        chosen.to_csv(p.output_dir / f"forecast_{origin.date()}.csv", index=False)
        log.info(
            "origin %s (%s): %d selected rows, %d cold-start series",
            origin.date(),
            label,
            len(chosen),
            int(chosen[chosen["h"] == 1]["is_cold_start"].sum()),
        )
    pd.concat(all_preds, ignore_index=True).to_parquet(p.interim_dir / "forecasts_all_models.parquet", index=False)
