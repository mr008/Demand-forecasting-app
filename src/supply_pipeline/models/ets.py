"""Per-series exponential smoothing (ETS, additive damped trend).

Weekly series here are too short (107 weeks) for a 52-period seasonal ETS, so
the classical model is non-seasonal; yearly seasonality is covered by the
seasonal-naive baseline and by the LightGBM lag-52 feature. Prediction
intervals come from simulating the fitted state-space model.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from supply_pipeline.metrics import quantile_col
from supply_pipeline.models.base import SERIES_KEY, finalize_quantiles

MIN_OBS = 10
N_SIMS = 500


def _fit_one(key: tuple, y: np.ndarray, horizon: int, quantiles: tuple[float, ...], seed: int) -> list[dict]:
    y = pd.Series(y, dtype=float).interpolate(limit_direction="both").to_numpy()
    rows: list[dict] = []
    if len(y) < MIN_OBS or np.nanstd(y) == 0:
        point = float(np.nanmean(y)) if len(y) else 0.0
        for h in range(1, horizon + 1):
            rows.append({"upc": key[0], "cedis": key[1], "h": h, **{quantile_col(q): point for q in quantiles}})
        return rows

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", FutureWarning)
        model = ETSModel(y, error="add", trend="add", damped_trend=True, seasonal=None)
        res = model.fit(disp=False, maxiter=200)
        path = np.asarray(res.forecast(horizon), dtype=float)
        sims = res.simulate(nsimulations=horizon, repetitions=N_SIMS, anchor="end", random_state=seed)
    sims = np.asarray(sims, dtype=float)  # (horizon, N_SIMS)
    for h in range(1, horizon + 1):
        row: dict[str, object] = {"upc": key[0], "cedis": key[1], "h": h}
        for q in quantiles:
            row[quantile_col(q)] = float(np.quantile(sims[h - 1], q))
        row["q50"] = float(path[h - 1])
        rows.append(row)
    return rows


class ETSDamped:
    name = "ets"

    def __init__(self, seed: int = 42, n_jobs: int = -1) -> None:
        self.seed = seed
        self.n_jobs = n_jobs

    def fit_predict(self, train: pd.DataFrame, future: pd.DataFrame, quantiles: tuple[float, ...]) -> pd.DataFrame:
        horizon = int(future["h"].max())
        groups = [(key, g.sort_values("week_start")["y_capped"].to_numpy()) for key, g in train.groupby(SERIES_KEY, sort=False)]
        results = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(_fit_one)(key, y, horizon, quantiles, self.seed) for key, y in groups
        )
        pred = pd.DataFrame([r for rows in results for r in rows])
        pred = future[SERIES_KEY + ["h", "target_week"]].merge(pred, on=SERIES_KEY + ["h"], how="left")
        return finalize_quantiles(pred, quantiles)
