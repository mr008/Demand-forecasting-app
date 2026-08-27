"""Stage 'blindtest': sealed hold-out comparison of all models.

The backtest already trains every model only on data before each replay
origin, so its predictions are out-of-sample. What is *not* blind in the
backtest is the model *selection*, which uses every fold. This stage fixes
that:

* the latest fold (whose 8 target weeks end at the last complete week) is
  sealed as the test set;
* model selection and interval calibration are re-done using only folds whose
  target weeks all end before the sealed origin (folds that overlap the sealed
  window are dropped);
* every model is scored on the sealed weeks, and the pre-registered choice is
  compared with the best model in hindsight ("regret").

No model is re-fitted; the stage reads ``backtest_predictions.parquet``.

Outputs: ``reports/tables/blind_test_overall.csv``, ``blind_test_cluster.csv``,
``blind_test_selection.csv`` and ``reports/figures/blind_test.png``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from supply_pipeline.backtest import apply_calibration, calibrate_intervals, select_models
from supply_pipeline.config import Config
from supply_pipeline.metrics import summarize_forecasts

log = logging.getLogger(__name__)


def split_folds(pred: pd.DataFrame, horizon_weeks: int) -> tuple[list[int], int]:
    """Folds usable for selection (targets end before the sealed origin) and the sealed fold."""
    origins = pred.groupby("fold")["origin"].first().sort_index()
    sealed_fold = int(origins.index.max())
    sealed_origin = origins.loc[sealed_fold]
    usable = [int(f) for f, o in origins.items() if o + pd.Timedelta(weeks=horizon_weeks) <= sealed_origin]
    return usable, sealed_fold


def run(cfg: Config) -> None:
    p = cfg.paths
    q = cfg.forecast.quantiles
    pred = pd.read_parquet(p.interim_dir / "backtest_predictions.parquet")
    usable, sealed_fold = split_folds(pred, cfg.forecast.horizon_weeks)
    if not usable:
        raise RuntimeError("blindtest: no backtest fold ends before the sealed fold; increase backtest_folds")
    train_pred = pred[pred["fold"].isin(usable)]
    sealed = pred[pred["fold"] == sealed_fold]
    sealed_origin = sealed["origin"].iloc[0]
    log.info(
        "blind test: selection on folds %s (targets end <= %s), sealed fold %d = weeks %s .. %s",
        usable,
        sealed_origin.date(),
        sealed_fold,
        sealed["target_week"].min().date(),
        sealed["target_week"].max().date(),
    )

    # Pre-registered choice and calibration from the earlier folds only.
    pre_selection = select_models(summarize_forecasts(train_pred, ["model", "cluster", "fold"], q))
    calib = calibrate_intervals(train_pred, q)
    sealed_cal = apply_calibration(sealed, calib, q)

    overall = summarize_forecasts(sealed_cal, ["model"], q).sort_values("wape")
    cluster = summarize_forecasts(sealed_cal, ["model", "cluster"], q)
    overall_raw_cov = summarize_forecasts(sealed, ["model"], q)[["model", "coverage_90", "coverage_80"]]
    overall = overall.merge(
        overall_raw_cov.rename(columns={"coverage_90": "coverage_90_raw", "coverage_80": "coverage_80_raw"}), on="model"
    )

    # Did the pre-registered choice hold up? Compare with the best model in hindsight per cluster.
    rows = []
    for cl, g in cluster.groupby("cluster"):
        g = g.set_index("model")
        chosen = pre_selection.set_index("cluster").loc[cl, "selected_model"]
        best = g["wape"].idxmin()
        rows.append(
            {
                "cluster": cl,
                "pre_registered_model": chosen,
                "blind_wape_pre_registered": float(g.loc[chosen, "wape"]),
                "best_model_in_hindsight": best,
                "blind_wape_best": float(g.loc[best, "wape"]),
                "regret": float(g.loc[chosen, "wape"] - g.loc[best, "wape"]),
                "coverage_90_pre_registered": float(g.loc[chosen, "coverage_90"]),
                "n": int(g.loc[chosen, "n"]),
            }
        )
    selection = pd.DataFrame(rows)
    selection["choice_held"] = selection["pre_registered_model"] == selection["best_model_in_hindsight"]

    overall.to_csv(p.tables_dir / "blind_test_overall.csv", index=False)
    cluster.to_csv(p.tables_dir / "blind_test_cluster.csv", index=False)
    selection.to_csv(p.tables_dir / "blind_test_selection.csv", index=False)
    _figure(cluster, selection, p.figures_dir / "blind_test.png")

    w = np.average(selection["blind_wape_pre_registered"], weights=selection["n"])
    w_best = np.average(selection["blind_wape_best"], weights=selection["n"])
    log.info(
        "blind test overall:\n%s",
        overall[["model", "n", "wape", "bias", "coverage_90_raw", "coverage_90", "coverage_80"]].round(3).to_string(index=False),
    )
    log.info(
        "pre-registered choice held in %d of %d clusters; blind WAPE %.3f vs %.3f with hindsight",
        int(selection["choice_held"].sum()),
        len(selection),
        w,
        w_best,
    )


def _figure(cluster: pd.DataFrame, selection: pd.DataFrame, out: Path) -> None:
    from supply_pipeline import plots

    clusters = sorted(cluster["cluster"].unique())
    models = [m for m in plots.COLORS if m in set(cluster["model"])]
    fig, ax = plots.plt.subplots(figsize=(8, 3.6))
    width = 0.8 / len(models)
    x = np.arange(len(clusters))
    for i, m in enumerate(models):
        vals = [cluster[(cluster["model"] == m) & (cluster["cluster"] == c)]["wape"].mean() for c in clusters]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=plots.MODEL_LABELS[m], color=plots.COLORS[m])
    sel = selection.set_index("cluster")
    for j, c in enumerate(clusters):
        m = sel.loc[c, "pre_registered_model"]
        if m in models:
            xi = x[j] + models.index(m) * width - 0.4 + width / 2
            ax.annotate("pre-registered", (xi, 0.01), ha="center", va="bottom", fontsize=6.5, color="white", rotation=90)
    ax.set_xticks(x, clusters)
    ax.set_ylabel("WAPE on sealed weeks")
    ax.set_ylim(0, min(1.0, ax.get_ylim()[1]))
    ax.set_title("Blind test: sealed last 8 weeks, selection made without them")
    ax.legend(ncol=2, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plots.plt.close(fig)
