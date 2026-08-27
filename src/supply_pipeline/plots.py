"""Matplotlib figures for the report and the deck. All functions save a PNG and return its path."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SERIES_KEY = ["upc", "cedis"]
COLORS = {"lgbm": "#0E6B58", "ets": "#B9911E", "ma4": "#5B6862", "seasonal_naive": "#9A4A1E"}
MODEL_LABELS = {
    "lgbm": "LightGBM (global, quantile)",
    "ets": "ETS (damped trend)",
    "ma4": "Moving average (4 wk)",
    "seasonal_naive": "Seasonal naive (52 wk)",
}
plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "legend.frameon": False,
    }
)


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def model_comparison(metrics_cluster: pd.DataFrame, selection: pd.DataFrame, out: Path) -> Path:
    clusters = sorted(metrics_cluster["cluster"].unique())
    models = [m for m in COLORS if m in set(metrics_cluster["model"])]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    width = 0.8 / len(models)
    x = np.arange(len(clusters))
    for i, m in enumerate(models):
        vals = [
            metrics_cluster[(metrics_cluster["model"] == m) & (metrics_cluster["cluster"] == c)]["wape"].mean() for c in clusters
        ]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=MODEL_LABELS[m], color=COLORS[m])
    sel = selection.set_index("cluster")["selected_model"]
    for j, c in enumerate(clusters):
        m = sel.get(c)
        if m in models:
            xi = x[j] + models.index(m) * width - 0.4 + width / 2
            ax.annotate("selected", (xi, 0.01), ha="center", va="bottom", fontsize=7, color="white", rotation=90)
    ax.set_xticks(x, clusters)
    ax.set_ylabel("WAPE (8 folds, 8-week horizon)")
    ax.set_title("Backtest accuracy by SKU cluster")
    ax.legend(ncol=2, fontsize=8, loc="upper left")
    return _save(fig, out)


def wape_by_horizon(metrics_cluster_h: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for m, g in metrics_cluster_h.groupby("model"):
        byh = g.groupby("h").apply(lambda d: np.average(d["wape"], weights=d["n"]), include_groups=False)
        ax.plot(byh.index, byh.values, marker="o", ms=3, label=MODEL_LABELS.get(m, m), color=COLORS.get(m))
    ax.set_xlabel("weeks ahead")
    ax.set_ylabel("WAPE")
    ax.set_ylim(0, min(1.0, ax.get_ylim()[1]))
    ax.set_title("Error grows with horizon")
    ax.legend(fontsize=8)
    return _save(fig, out)


def coverage_calibration(raw: pd.DataFrame, cal: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, (label, df) in zip(axes, (("raw quantiles", raw), ("calibrated (held-out folds)", cal)), strict=False):
        for m, g in df.groupby("model"):
            c90 = np.average(g["coverage_90"], weights=g["n"])
            c80 = np.average(g["coverage_80"], weights=g["n"])
            ax.plot([0.8, 0.9], [c80, c90], marker="o", label=MODEL_LABELS.get(m, m), color=COLORS.get(m))
        ax.plot([0.75, 0.95], [0.75, 0.95], ls="--", color="#999", lw=1)
        ax.set_xticks([0.8, 0.9])
        ax.set_xlabel("nominal interval")
        ax.set_title(label)
    axes[0].set_ylabel("achieved coverage")
    axes[0].legend(fontsize=7)
    return _save(fig, out)


def cluster_forecast_vs_actual(pred: pd.DataFrame, weekly: pd.DataFrame, selection: pd.DataFrame, out: Path) -> Path:
    """Last backtest fold, aggregated to cluster level, selected model with 80% band."""
    last_fold = pred["fold"].max()
    sel = selection.set_index("cluster")["selected_model"]
    clusters = sorted(sel.index)
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5))
    for ax, c in zip(axes.ravel(), clusters, strict=False):
        p = pred[(pred["fold"] == last_fold) & (pred["cluster"] == c) & (pred["model"] == sel[c])]
        origin = p["origin"].iloc[0]
        keys = p[SERIES_KEY].drop_duplicates()
        hist = weekly.merge(keys, on=SERIES_KEY)
        hist = hist[(hist["week_start"] > origin - pd.Timedelta(weeks=26)) & (hist["week_start"] <= origin)]
        h = hist.groupby("week_start")["y"].sum()
        agg = p.groupby("target_week").agg(y=("y", "sum"), q50=("q50", "sum"), q10=("q10", "sum"), q90=("q90", "sum"))
        ax.plot(h.index, h.values, color="#333", lw=1.2, label="actual")
        ax.plot(agg.index, agg["y"], color="#333", lw=1.2)
        ax.plot(agg.index, agg["q50"], color=COLORS[sel[c]], lw=1.5, label=f"{sel[c]} median")
        ax.fill_between(agg.index, agg["q10"], agg["q90"], color=COLORS[sel[c]], alpha=0.2, label="80% band")
        ax.axvline(origin, color="#999", ls=":", lw=1)
        ax.set_title(f"{c}  ({len(keys)} series)")
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        ax.set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Forecast vs actual, last backtest fold (origin {origin.date()}), cluster totals", fontweight="bold")
    return _save(fig, out)


def residuals(pred: pd.DataFrame, selection: pd.DataFrame, out: Path) -> Path:
    sel = selection.set_index("cluster")["selected_model"]
    p = pred[pred.apply(lambda r: r["model"] == sel.get(r["cluster"]), axis=1)].dropna(subset=["y"])
    p = p[p["y"] > 0]
    rel = (p["q50"] - p["y"]) / p["y"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].hist(rel.clip(-1, 1), bins=50, color="#0E6B58", alpha=0.85)
    axes[0].axvline(0, color="#333", lw=1)
    axes[0].set_title("Relative residual (forecast - actual) / actual, selected models")
    axes[0].set_xlabel("clipped to [-1, 1]")
    byc = p.assign(rel=rel).groupby("cluster")["rel"].agg(["median", lambda s: s.quantile(0.1), lambda s: s.quantile(0.9)])
    byc.columns = ["median", "p10", "p90"]
    y = np.arange(len(byc))
    axes[1].errorbar(
        byc["median"], y, xerr=[byc["median"] - byc["p10"], byc["p90"] - byc["median"]], fmt="o", color="#0E6B58", capsize=3
    )
    axes[1].axvline(0, color="#333", lw=1)
    axes[1].set_yticks(y, byc.index)
    axes[1].set_title("Residual median and p10-p90 by cluster")
    return _save(fig, out)


def risk_window(scored: pd.DataFrame, sweep: pd.DataFrame, out: Path) -> Path:
    ev = scored.dropna(subset=["label_7d"]).copy()
    ev["label"] = np.where(ev["label_7d"] == 1, "stock-out within 7 days", "no stock-out")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    data = [ev.loc[ev["label_7d"] == 0, "cover_days"].clip(upper=120), ev.loc[ev["label_7d"] == 1, "cover_days"].clip(upper=120)]
    axes[0].boxplot(data, tick_labels=["no stock-out", "stock-out in 7d"], showfliers=False, widths=0.5)
    axes[0].axhline(14, color="#9A4A1E", ls="--", lw=1, label="lead time + 7 safety days")
    axes[0].set_ylabel("DC days of cover (capped at 120)")
    axes[0].set_title("Cover separates the two outcomes")
    axes[0].legend(fontsize=7)
    for m, g in sweep.groupby("method"):
        axes[1].plot(g["recall"], g["precision"], marker="o", ms=3, label=f"{m} threshold sweep")
        for _, r in g.iterrows():
            axes[1].annotate(
                str(r["threshold"]), (r["recall"], r["precision"]), fontsize=6, xytext=(2, 2), textcoords="offset points"
            )
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_xlim(0, 1.02)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Alert precision vs recall")
    axes[1].legend(fontsize=7)
    return _save(fig, out)


def risk_timeline(scored: pd.DataFrame, out: Path, max_rows: int = 30) -> Path:
    """Per-series timeline of the inventory window: stock-out days shaded, alert days marked.

    Rows are every series with at least one stock-out event, followed by the
    series with the most alert days but no event (the false alarms), up to
    ``max_rows`` in total.
    """
    s = scored.copy()
    s["key"] = s["upc"].astype(str) + " @ " + s["cedis"].astype(str)
    s["any_alert"] = s["alert_cover"] | s["alert_prob"]
    per = s.groupby("key").agg(events=("event", "sum"), alerts=("any_alert", "sum"))
    hits = per[per["events"] > 0].sort_values("events", ascending=False)
    misses = per[per["events"] == 0].sort_values("alerts", ascending=False).head(max(0, max_rows - len(hits)))
    keys = list(hits.index) + list(misses.index)
    days = sorted(s["date"].unique())
    day_idx = {d: i for i, d in enumerate(days)}
    fig, ax = plt.subplots(figsize=(10, 0.28 * len(keys) + 1.6))
    for r, k in enumerate(keys):
        g = s[s["key"] == k]
        ev = g[g["event"]]
        ax.scatter([day_idx[d] for d in ev["date"]], [r] * len(ev), marker="s", s=60, color="#D9C7B8", linewidths=0, zorder=1)
        cov = g[g["alert_cover"]]
        ax.scatter([day_idx[d] for d in cov["date"]], [r] * len(cov), marker="_", s=70, color="#B9911E", linewidths=1.6, zorder=2)
        pr = g[g["alert_prob"]]
        ax.scatter([day_idx[d] for d in pr["date"]], [r] * len(pr), marker="o", s=14, color="#0E6B58", zorder=3)
    ax.axhline(len(hits) - 0.5, color="#999", lw=0.8, ls=":")
    ax.set_yticks(range(len(keys)), keys, fontsize=6.5)
    ax.set_ylim(len(keys) - 0.5, -0.5)
    ax.set_xticks(range(len(days)), [pd.Timestamp(d).strftime("%d %b") for d in days], rotation=60, fontsize=7)
    ax.set_xlim(-0.5, len(days) - 0.5)
    ax.scatter([], [], marker="s", s=60, color="#D9C7B8", label="stock-out day (>= 25% of stores empty)")
    ax.scatter([], [], marker="_", s=70, color="#B9911E", linewidths=1.6, label="cover-rule alert")
    ax.scatter([], [], marker="o", s=14, color="#0E6B58", label="forecast-probability alert")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.set_title(
        f"Alerts vs actual stock-outs, {len(hits)} series with events (top) and {len(misses)} most-alerted without (bottom)"
    )
    ax.grid(axis="x", color="#eee", lw=0.5)
    return _save(fig, out)


def series_examples(weekly: pd.DataFrame, fc: pd.DataFrame, series: pd.DataFrame, out: Path) -> Path:
    """Four representative series with the final 8-week forecast fan."""
    s = series.copy()
    picks = []
    for cond, label in (
        (s["cluster"] == "A-X", "A-X, steady"),
        (s["cluster"] == "A-Z", "A-Z, erratic"),
        (s["cluster"] == "B-Z", "B-Z"),
        (s["is_discontinued"], "discontinued (do not trust)"),
    ):
        cand = s[cond].sort_values("mean_weekly", ascending=False)
        if len(cand):
            picks.append((cand.iloc[0], label))
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.5))
    for ax, (row, label) in zip(axes.ravel(), picks, strict=False):
        h = weekly[(weekly["upc"] == row["upc"]) & (weekly["cedis"] == row["cedis"])].tail(52)
        f = fc[(fc["upc"] == row["upc"]) & (fc["cedis"] == row["cedis"])].sort_values("target_week")
        ax.plot(h["week_start"], h["y"], color="#333", lw=1.2, label="actual")
        promo = h[h["promo_share"] > 0.5]
        ax.scatter(promo["week_start"], promo["y"], s=12, color="#B9911E", zorder=3, label="promo week")
        if len(f):
            model = f["model"].iloc[0]
            ax.plot(f["target_week"], f["q50"], color=COLORS.get(model, "#0E6B58"), lw=1.5, label=f"{model} median")
            ax.fill_between(
                f["target_week"], f["q05"], f["q95"], color=COLORS.get(model, "#0E6B58"), alpha=0.12, label="90% band"
            )
            ax.fill_between(
                f["target_week"], f["q10"], f["q90"], color=COLORS.get(model, "#0E6B58"), alpha=0.25, label="80% band"
            )
        ax.set_title(f"{label}: upc {row['upc']} @ {row['cedis']}")
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        ax.set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=7)
    return _save(fig, out)


def portfolio_forecast(weekly: pd.DataFrame, fc: pd.DataFrame, out: Path) -> Path:
    hist = weekly[weekly["week_start"] > weekly["week_start"].max() - pd.Timedelta(weeks=52)].groupby("week_start")["y"].sum()
    agg = fc.groupby("target_week")[["q05", "q10", "q50", "q90", "q95"]].sum()
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(hist.index, hist.values / 1e3, color="#333", lw=1.2, label="actual (all 328 series)")
    ax.plot(agg.index, agg["q50"] / 1e3, color="#0E6B58", lw=1.6, label="forecast median")
    ax.fill_between(agg.index, agg["q05"] / 1e3, agg["q95"] / 1e3, color="#0E6B58", alpha=0.12, label="90% band (sum of series)")
    ax.fill_between(agg.index, agg["q10"] / 1e3, agg["q90"] / 1e3, color="#0E6B58", alpha=0.25, label="80% band")
    ax.set_ylabel("thousand units / week")
    ax.set_ylim(bottom=0)
    ax.set_title("Portfolio weekly sell-out: last 52 weeks and the next 8")
    ax.legend(fontsize=7, ncol=2)
    return _save(fig, out)


def orders_summary(by_cluster: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    x = np.arange(len(by_cluster))
    w = 0.27
    axes[0].bar(x - w, by_cluster["ma4_order_units"] / 1e3, w, label="4-wk moving average", color="#5B6862")
    axes[0].bar(x, by_cluster["ly_order_units"] / 1e3, w, label="last year same weeks", color="#9A4A1E")
    axes[0].bar(x + w, by_cluster["order_units"] / 1e3, w, label="recommended", color="#0E6B58")
    axes[0].set_xticks(x, by_cluster["cluster"])
    axes[0].set_ylabel("order units (thousands)")
    axes[0].set_title("Order quantity by cluster")
    axes[0].legend(fontsize=7)
    axes[1].bar(x - w / 2, by_cluster["ma4_service_level_weighted"], w, label="4-wk moving average", color="#5B6862")
    axes[1].bar(x + w / 2, by_cluster["service_level_weighted"], w, label="recommended", color="#0E6B58")
    axes[1].set_xticks(x, by_cluster["cluster"])
    axes[1].set_ylim(0.5, 1.0)
    axes[1].set_ylabel("implied cycle service level")
    axes[1].set_title("Service level over the 14-day protection period")
    axes[1].legend(fontsize=7, loc="lower right")
    return _save(fig, out)


def inventory_coverage(cov: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.bar(pd.to_datetime(cov.index), cov["upcs"], color="#0E6B58")
    ax.set_ylabel("UPCs reported")
    ax.set_title("Inventory file: UPC coverage per day (last full snapshot 2026-04-02)")
    ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    return _save(fig, out)
