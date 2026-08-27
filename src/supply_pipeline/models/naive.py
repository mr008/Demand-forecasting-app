"""Naive baselines: seasonal naive and moving average.

Intervals come from the empirical distribution of each method's own in-sample
h-step-ahead errors, per series, so they are honest about how wrong the naive
method has been on that series rather than assuming a shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from supply_pipeline.metrics import quantile_col
from supply_pipeline.models.base import SERIES_KEY, finalize_quantiles

MIN_RESIDUALS = 8


def _interval_from_residuals(point: float, resid: np.ndarray, quantiles: tuple[float, ...]) -> dict[str, float]:
    resid = resid[~np.isnan(resid)]
    out: dict[str, float] = {}
    for q in quantiles:
        if len(resid) >= MIN_RESIDUALS:
            out[quantile_col(q)] = point + float(np.quantile(resid, q))
        else:
            # Too little history: symmetric +-50% band as a crude fallback.
            out[quantile_col(q)] = point * (1 + (q - 0.5))
    out["q50"] = point  # median is the point forecast itself
    return out


class MovingAverage:
    """Mean of the last ``window`` observed weeks, flat over the horizon."""

    def __init__(self, window: int = 4) -> None:
        self.window = window
        self.name = f"ma{window}"

    def fit_predict(self, train: pd.DataFrame, future: pd.DataFrame, quantiles: tuple[float, ...]) -> pd.DataFrame:
        rows = []
        horizon = int(future["h"].max())
        for key, g in train.groupby(SERIES_KEY, sort=False):
            y = g.sort_values("week_start")["y"].to_numpy(dtype=float)
            obs = y[~np.isnan(y)]
            point = float(obs[-self.window :].mean()) if len(obs) else 0.0
            s = pd.Series(y)
            for h in range(1, horizon + 1):
                # in-sample h-step forecast at t: mean of y[t-h-window+1 .. t-h]
                fitted = s.shift(h).rolling(self.window, min_periods=1).mean()
                resid = (s - fitted).to_numpy()
                rows.append({"upc": key[0], "cedis": key[1], "h": h, **_interval_from_residuals(point, resid, quantiles)})
        pred = pd.DataFrame(rows)
        pred = future[SERIES_KEY + ["h", "target_week"]].merge(pred, on=SERIES_KEY + ["h"], how="left")
        return finalize_quantiles(pred, quantiles)


class SeasonalNaive:
    """Same week last year (lag 52); falls back to the last observed value."""

    name = "seasonal_naive"
    season = 52

    def fit_predict(self, train: pd.DataFrame, future: pd.DataFrame, quantiles: tuple[float, ...]) -> pd.DataFrame:
        rows = []
        horizon = int(future["h"].max())
        for key, g in train.groupby(SERIES_KEY, sort=False):
            y = g.sort_values("week_start")["y"].to_numpy(dtype=float)
            obs = y[~np.isnan(y)]
            last = float(obs[-1]) if len(obs) else 0.0
            s = pd.Series(y)
            resid = (s - s.shift(self.season)).to_numpy()
            for h in range(1, horizon + 1):
                idx = len(y) - self.season + (h - 1)
                point = float(y[idx]) if 0 <= idx < len(y) and not np.isnan(y[idx]) else last
                rows.append({"upc": key[0], "cedis": key[1], "h": h, **_interval_from_residuals(point, resid, quantiles)})
        pred = pd.DataFrame(rows)
        pred = future[SERIES_KEY + ["h", "target_week"]].merge(pred, on=SERIES_KEY + ["h"], how="left")
        return finalize_quantiles(pred, quantiles)
