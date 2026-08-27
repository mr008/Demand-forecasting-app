"""Common interface for forecasting models.

Every model receives the weekly training table for *all* series (rows with
``week_start <= origin``) and a ``future`` frame with one row per series and
horizon step carrying the known-in-advance calendar features. It returns one
row per (upc, cedis, h) with a column per requested quantile (``q05`` .. ``q95``).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from supply_pipeline.metrics import quantile_col

SERIES_KEY = ["upc", "cedis"]
FUTURE_KEY = SERIES_KEY + ["h", "target_week"]


class Forecaster(Protocol):
    name: str

    def fit_predict(self, train: pd.DataFrame, future: pd.DataFrame, quantiles: tuple[float, ...]) -> pd.DataFrame: ...


def make_future(train: pd.DataFrame, origin: pd.Timestamp, horizon: int, weekly_cal: pd.DataFrame) -> pd.DataFrame:
    """Series x horizon grid with calendar features for the target weeks."""
    series = train[SERIES_KEY].drop_duplicates()
    hs = pd.DataFrame({"h": np.arange(1, horizon + 1)})
    fut = series.merge(hs, how="cross")
    fut["target_week"] = origin + pd.to_timedelta(fut["h"] * 7, unit="D")
    fut = fut.merge(weekly_cal.rename(columns={"week_start": "target_week"}), on="target_week", how="left")
    return fut


def finalize_quantiles(df: pd.DataFrame, quantiles: tuple[float, ...]) -> pd.DataFrame:
    """Clip at zero and enforce monotone quantiles across the row."""
    cols = [quantile_col(q) for q in sorted(quantiles)]
    arr = np.clip(df[cols].to_numpy(dtype=float), 0.0, None)
    arr = np.sort(arr, axis=1)
    out = df.copy()
    out[cols] = arr
    return out
