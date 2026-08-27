"""Stage 'backtest': expanding-window evaluation and per-cluster model selection.

Protocol
--------
* Forecast origins are the last week of each training window. The latest origin
  is ``horizon_weeks`` before the last complete week, so every fold's full
  horizon is observed; earlier origins step back ``backtest_step_weeks`` at a
  time (``backtest_folds`` folds in total).
* At each origin every model is trained on all series with ``week_start <=
  origin`` and asked for ``horizon_weeks`` weeks of quantile forecasts.
* Series with fewer than ``cold_start_weeks`` observed weeks at the origin are
  excluded from model comparison (they are forecast with the fallback model in
  production and flagged).
* Metrics are computed against raw actuals (never the winsorised copy).

Outputs: ``data/interim/backtest_predictions.parquet``,
``reports/tables/backtest_metrics_{cluster,cluster_h,fold}.csv``,
``reports/tables/model_selection.csv``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from supply_pipeline.calendar_features import weekly_calendar
from supply_pipeline.config import Config
from supply_pipeline.metrics import quantile_col, summarize_forecasts
from supply_pipeline.models import FALLBACK_MODEL, MODEL_ORDER, Forecaster, build_models, make_future

log = logging.getLogger(__name__)

SERIES_KEY = ["upc", "cedis"]
SELECTION_TOLERANCE = 0.02  # absolute WAPE points within which a simpler / steadier model wins
CALIBRATION_HOLDOUT_FOLDS = 3  # last folds are kept out of interval calibration to report honest coverage


def fold_origins(cfg: Config) -> list[pd.Timestamp]:
    last = pd.Timestamp(cfg.data.last_complete_week_start)
    latest = last - pd.Timedelta(weeks=cfg.forecast.horizon_weeks)
    step = pd.Timedelta(weeks=cfg.forecast.backtest_step_weeks)
    return sorted(latest - k * step for k in range(cfg.forecast.backtest_folds))


def eligible_series(train: pd.DataFrame, min_weeks: int) -> pd.DataFrame:
    n = train.dropna(subset=["y"]).groupby(SERIES_KEY).size()
    return n[n >= min_weeks].reset_index()[SERIES_KEY]


def run_fold(
    models: dict[str, Forecaster], fold: int, origin: pd.Timestamp, weekly: pd.DataFrame, weekly_cal: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    horizon = cfg.forecast.horizon_weeks
    train_all = weekly[weekly["week_start"] <= origin]
    ok = eligible_series(train_all, cfg.data.cold_start_weeks)
    train = train_all.merge(ok, on=SERIES_KEY)
    future = make_future(train, origin, horizon, weekly_cal)
    actual = weekly[(weekly["week_start"] > origin) & (weekly["week_start"] <= origin + pd.Timedelta(weeks=horizon))]
    actual = actual[SERIES_KEY + ["week_start", "y", "cluster"]].rename(columns={"week_start": "target_week"})

    preds = []
    for name, model in models.items():
        p = model.fit_predict(train, future, cfg.forecast.quantiles)
        p["model"] = name
        preds.append(p)
        log.info("fold %d origin %s model %-14s -> %d rows", fold, origin.date(), name, len(p))
    out = pd.concat(preds, ignore_index=True)
    out["fold"] = fold
    out["origin"] = origin
    out = out.merge(actual, on=SERIES_KEY + ["target_week"], how="left")
    return out


def calibrate_intervals(pred: pd.DataFrame, quantiles: tuple[float, ...]) -> pd.DataFrame:
    """Conformal-style width factors per (model, cluster, interval).

    For each nominal interval (q10-q90, q05-q95) the non-conformity score is how
    far outside the interval the actual fell, in units of the interval width:
    ``E = max(lo - y, y - hi) / (hi - lo)``. The factor ``k`` is the nominal-level
    empirical quantile of ``E``; calibrated bounds are ``lo - k*w`` and ``hi + k*w``.
    Negative ``k`` narrows an over-wide interval.
    """
    qs = sorted(quantiles)
    pairs = [(qs[1], qs[-2]), (qs[0], qs[-1])]
    rows = []
    for lo_q, hi_q in pairs:
        lo, hi = quantile_col(lo_q), quantile_col(hi_q)
        nominal = hi_q - lo_q
        for (model, cluster), g in pred.dropna(subset=["y"]).groupby(["model", "cluster"]):
            w = (g[hi] - g[lo]).clip(lower=1e-6)
            e = np.maximum(g[lo] - g["y"], g["y"] - g[hi]) / w
            k = float(np.quantile(e, nominal))
            rows.append(
                {"model": model, "cluster": cluster, "lo": lo, "hi": hi, "nominal": nominal, "k": max(k, -0.4), "n": len(g)}
            )
    return pd.DataFrame(rows)


def apply_calibration(pred: pd.DataFrame, calib: pd.DataFrame, quantiles: tuple[float, ...]) -> pd.DataFrame:
    """Widen/narrow interval bounds by the fitted factors; q50 is untouched."""
    out = pred.copy()
    for _, r in calib.iterrows():
        mask = (out["model"] == r["model"]) & (out["cluster"] == r["cluster"])
        if not mask.any():
            continue
        w = (out.loc[mask, r["hi"]] - out.loc[mask, r["lo"]]).clip(lower=1e-6)
        out.loc[mask, r["lo"]] = out.loc[mask, r["lo"]] - r["k"] * w
        out.loc[mask, r["hi"]] = out.loc[mask, r["hi"]] + r["k"] * w
    cols = [quantile_col(q) for q in sorted(quantiles)]
    out[cols] = np.sort(np.clip(out[cols].to_numpy(dtype=float), 0.0, None), axis=1)
    return out


def select_models(metrics_fold: pd.DataFrame) -> pd.DataFrame:
    """Pick one model per cluster from per-fold metrics.

    Rule: lowest mean WAPE across folds wins, unless a model earlier in
    ``MODEL_ORDER`` (simpler) or with a lower fold-to-fold WAPE spread is within
    ``SELECTION_TOLERANCE``; then the steadier / simpler one wins.
    """
    agg = (
        metrics_fold.groupby(["cluster", "model"])
        .agg(
            wape_mean=("wape", "mean"),
            wape_std=("wape", "std"),
            bias_mean=("bias", "mean"),
            pinball_mean=("pinball", "mean"),
            coverage_90=("coverage_90", "mean"),
            coverage_80=("coverage_80", "mean"),
            n_folds=("wape", "size"),
        )
        .reset_index()
    )
    agg["rank_simplicity"] = agg["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    rows = []
    for cluster, g in agg.groupby("cluster"):
        best = g["wape_mean"].min()
        cands = g[g["wape_mean"] <= best + SELECTION_TOLERANCE]
        chosen = cands.sort_values(["wape_std", "rank_simplicity"]).iloc[0]
        if chosen["wape_mean"] == best:
            why = "lowest mean WAPE"
        else:
            why = f"within {SELECTION_TOLERANCE:.2f} WAPE of best ({best:.3f}) with lower fold spread"
        rows.append(
            {
                "cluster": cluster,
                "selected_model": chosen["model"],
                "wape_mean": chosen["wape_mean"],
                "wape_std": chosen["wape_std"],
                "bias_mean": chosen["bias_mean"],
                "coverage_90": chosen["coverage_90"],
                "best_wape": best,
                "rationale": why,
            }
        )
    return pd.DataFrame(rows)


def run(cfg: Config) -> None:
    p = cfg.paths
    weekly = pd.read_parquet(p.interim_dir / "weekly.parquet")
    calendar = pd.read_parquet(p.interim_dir / "calendar.parquet")
    weekly_cal = weekly_calendar(calendar)
    models = build_models(cfg)
    origins = fold_origins(cfg)
    log.info(
        "backtest: %d folds, origins %s .. %s, horizon %d weeks",
        len(origins),
        origins[0].date(),
        origins[-1].date(),
        cfg.forecast.horizon_weeks,
    )

    folds = [run_fold(models, i, o, weekly, weekly_cal, cfg) for i, o in enumerate(origins)]
    pred = pd.concat(folds, ignore_index=True)
    pred.to_parquet(p.interim_dir / "backtest_predictions.parquet", index=False)

    q = cfg.forecast.quantiles
    m_fold = summarize_forecasts(pred, ["model", "cluster", "fold"], q)
    m_cluster = summarize_forecasts(pred, ["model", "cluster"], q)
    m_cluster_h = summarize_forecasts(pred, ["model", "cluster", "h"], q)
    m_overall = summarize_forecasts(pred, ["model"], q)
    m_cedis = summarize_forecasts(pred, ["model", "cedis"], q)
    selection = select_models(m_fold)

    # Interval calibration: fit on the earlier folds, report on the last CALIBRATION_HOLDOUT_FOLDS.
    n_fit = max(len(origins) - CALIBRATION_HOLDOUT_FOLDS, 1)
    calib = calibrate_intervals(pred[pred["fold"] < n_fit], q)
    holdout = apply_calibration(pred[pred["fold"] >= n_fit], calib, q)
    m_holdout_raw = summarize_forecasts(pred[pred["fold"] >= n_fit], ["model", "cluster"], q)
    m_holdout_cal = summarize_forecasts(holdout, ["model", "cluster"], q)
    m_holdout_overall = summarize_forecasts(holdout, ["model"], q)
    calib.to_csv(p.tables_dir / "interval_calibration.csv", index=False)
    m_holdout_raw.to_csv(p.tables_dir / "backtest_metrics_holdout_raw.csv", index=False)
    m_holdout_cal.to_csv(p.tables_dir / "backtest_metrics_holdout_calibrated.csv", index=False)
    m_holdout_overall.to_csv(p.tables_dir / "backtest_metrics_holdout_calibrated_overall.csv", index=False)
    log.info(
        "holdout folds >= %d, calibrated coverage:\n%s",
        n_fit,
        m_holdout_overall[["model", "coverage_90", "coverage_80", "width_rel"]].round(3).to_string(index=False),
    )

    m_fold.to_csv(p.tables_dir / "backtest_metrics_fold.csv", index=False)
    m_cluster.to_csv(p.tables_dir / "backtest_metrics_cluster.csv", index=False)
    m_cluster_h.to_csv(p.tables_dir / "backtest_metrics_cluster_h.csv", index=False)
    m_overall.to_csv(p.tables_dir / "backtest_metrics_overall.csv", index=False)
    m_cedis.to_csv(p.tables_dir / "backtest_metrics_cedis.csv", index=False)
    selection.to_csv(p.tables_dir / "model_selection.csv", index=False)
    log.info("overall:\n%s", m_overall.round(3).to_string(index=False))
    log.info("selection:\n%s", selection.round(3).to_string(index=False))
    if selection["selected_model"].isna().any():
        selection["selected_model"] = selection["selected_model"].fillna(FALLBACK_MODEL)
