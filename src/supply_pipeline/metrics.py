"""Forecast and alert metrics. Pure functions over numpy arrays / pandas frames.

Forecast metrics take actual ``y`` and predictions; ``summarize_forecasts`` applies
them to the long prediction frame produced by the backtest.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- point metrics
def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Weighted absolute percentage error: sum|y - yhat| / sum(y)."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    denom = np.abs(y).sum()
    return float(np.abs(y - yhat).sum() / denom) if denom > 0 else float("nan")


def mape(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean absolute percentage error over rows with y > 0 (undefined otherwise)."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = y > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y[mask] - yhat[mask]) / y[mask]))


def bias(y: np.ndarray, yhat: np.ndarray) -> float:
    """Relative bias: sum(yhat - y) / sum(y). Positive = over-forecast."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    denom = np.abs(y).sum()
    return float((yhat - y).sum() / denom) if denom > 0 else float("nan")


# --------------------------------------------------------------------------- probabilistic metrics
def pinball(y: np.ndarray, q_pred: np.ndarray, alpha: float) -> float:
    """Mean pinball (quantile) loss for quantile level ``alpha``."""
    y = np.asarray(y, dtype=float)
    q_pred = np.asarray(q_pred, dtype=float)
    diff = y - q_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def scaled_pinball(y: np.ndarray, q_pred: np.ndarray, alpha: float) -> float:
    """Pinball loss divided by mean |y| so it is comparable across series."""
    y = np.asarray(y, dtype=float)
    denom = np.abs(y).mean()
    return pinball(y, q_pred, alpha) / denom if denom > 0 else float("nan")


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Share of actuals inside [lo, hi]."""
    y = np.asarray(y, dtype=float)
    return float(np.mean((y >= np.asarray(lo)) & (y <= np.asarray(hi))))


def quantile_col(q: float) -> str:
    return f"q{round(q * 100):02d}"


def summarize_forecasts(
    pred: pd.DataFrame,
    group_cols: Sequence[str],
    quantiles: Sequence[float],
    y_col: str = "y",
    point_col: str = "q50",
) -> pd.DataFrame:
    """Metrics per group from a long prediction frame with columns y, q05..q95."""
    qs = sorted(quantiles)
    lo_col, hi_col = quantile_col(qs[0]), quantile_col(qs[-1])
    inner_lo, inner_hi = quantile_col(qs[1]), quantile_col(qs[-2])
    nominal_outer = round((qs[-1] - qs[0]) * 100)
    nominal_inner = round((qs[-2] - qs[1]) * 100)

    rows = []
    for key, g in pred.dropna(subset=[y_col]).groupby(list(group_cols), observed=True):
        y = g[y_col].to_numpy()
        yhat = g[point_col].to_numpy()
        row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,), strict=False))
        row.update(
            n=len(g),
            wape=wape(y, yhat),
            mape=mape(y, yhat),
            bias=bias(y, yhat),
            pinball=float(np.mean([scaled_pinball(y, g[quantile_col(q)].to_numpy(), q) for q in qs])),
            **{f"coverage_{nominal_outer}": coverage(y, g[lo_col], g[hi_col])},
            **{f"coverage_{nominal_inner}": coverage(y, g[inner_lo], g[inner_hi])},
            width_rel=float(((g[hi_col] - g[lo_col]) / np.maximum(g[point_col], 1e-9)).median()),
        )
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- alert metrics
def alert_metrics(alert: np.ndarray, event: np.ndarray) -> dict[str, float]:
    """Precision / recall / F1 / false-alarm rate for binary alerts vs events."""
    alert = np.asarray(alert, dtype=bool)
    event = np.asarray(event, dtype=bool)
    tp = int((alert & event).sum())
    fp = int((alert & ~event).sum())
    fn = int((~alert & event).sum())
    tn = int((~alert & ~event).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (tp + fp) and (tp + fn) and (precision + recall) else float("nan")
    far = fp / (fp + tn) if fp + tn else float("nan")
    return {
        "n": len(alert),
        "n_alerts": tp + fp,
        "n_events": tp + fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": far,
    }


def lead_time_to_alert(alert_days: pd.DataFrame, event_days: pd.DataFrame, max_lead: int) -> pd.Series:
    """For each event-start (series, date), days from the earliest alert in the preceding window.

    ``alert_days`` and ``event_days`` have columns upc, cedis, date. Returns one
    value per event (NaN if no alert fired within ``max_lead`` days before it).
    """
    out = []
    alerts = {k: g["date"].sort_values().to_numpy() for k, g in alert_days.groupby(["upc", "cedis"])}
    for row in event_days.itertuples(index=False):
        a = alerts.get((row.upc, row.cedis))
        if a is None:
            out.append(np.nan)
            continue
        window_start = row.date - pd.Timedelta(days=max_lead)
        prior = a[(a >= window_start) & (a < row.date)]
        out.append(float((row.date - prior.min()).days) if len(prior) else np.nan)
    return pd.Series(out, index=event_days.index, name="lead_time_days")
