"""Global LightGBM quantile model with lag and calendar features.

One model per quantile is trained across all series at once (direct
multi-horizon: the horizon ``h`` is a feature and every training origin
contributes one row per horizon step). Targets and lag features are scaled by
the series' recent level so that a 70k-unit series and a 50-unit series share
the same tree structure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from supply_pipeline.metrics import quantile_col
from supply_pipeline.models.base import SERIES_KEY, finalize_quantiles

LAGS = (1, 2, 3, 4, 5, 6, 7, 8)
ROLLS = (4, 8, 13, 26)
CAL_COLS = ("payday_days", "holiday_days", "semana_santa_days", "buen_fin_days", "december_peak_days", "week_of_year", "month")
CAT_COLS = ("cluster", "abc_class", "xyz_class", "cedis", "upc")


def _series_features(g: pd.DataFrame) -> pd.DataFrame:
    """Features observable at week t for one series (sorted by week_start)."""
    y = g["y_capped"].astype(float)
    out = pd.DataFrame(index=g.index)
    scale = y.rolling(13, min_periods=4).mean().clip(lower=1.0)
    out["scale"] = scale
    out["log_scale"] = np.log1p(scale)
    for lag in LAGS:
        out[f"lag{lag}"] = y.shift(lag - 1) / scale  # lag1 == current week t
    for w in ROLLS:
        out[f"roll{w}"] = y.rolling(w, min_periods=2).mean() / scale
    out["roll8_std"] = y.rolling(8, min_periods=3).std() / scale
    out["trend4_13"] = out["roll4"] - out["roll13"]
    out["price_last"] = g["price_mean"]
    out["price_ratio4"] = g["price_mean"] / g["price_mean"].rolling(4, min_periods=1).mean()
    out["promo_last"] = g["promo_share"]
    out["promo_roll4"] = g["promo_share"].rolling(4, min_periods=1).mean()
    out["zero_days_last"] = g["zero_days"]
    out["series_age"] = g["series_age_weeks"]
    return out


def _lag52_lookup(train: pd.DataFrame) -> pd.DataFrame:
    """y_capped keyed by (series, week + 52w) so it can be joined onto target weeks."""
    look = train[SERIES_KEY + ["week_start", "y_capped"]].copy()
    look["target_week"] = look["week_start"] + pd.Timedelta(weeks=52)
    return look.drop(columns="week_start").rename(columns={"y_capped": "lag52_target_raw"})


class LightGBMQuantile:
    name = "lgbm"

    def __init__(self, seed: int = 42, n_estimators: int = 400, learning_rate: float = 0.05) -> None:
        self.seed = seed
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self._categories: dict[str, list] = {}

    # ------------------------------------------------------------------ frames
    def _base_features(self, train: pd.DataFrame) -> pd.DataFrame:
        train = train.sort_values(SERIES_KEY + ["week_start"])
        feats = train.groupby(SERIES_KEY, group_keys=False, sort=False).apply(_series_features, include_groups=False)
        meta = train[SERIES_KEY + ["week_start", "cluster", "abc_class", "xyz_class"]]
        return pd.concat([meta, feats], axis=1)

    def _supervised(self, train: pd.DataFrame, feats: pd.DataFrame, weekly_cal: pd.DataFrame, horizon: int) -> pd.DataFrame:
        target = train[SERIES_KEY + ["week_start", "y_capped"]].rename(
            columns={"week_start": "target_week", "y_capped": "y_target"}
        )
        lag52 = _lag52_lookup(train)
        parts = []
        for h in range(1, horizon + 1):
            f = feats.copy()
            f["h"] = h
            f["target_week"] = f["week_start"] + pd.Timedelta(weeks=h)
            f = f.merge(target, on=SERIES_KEY + ["target_week"], how="inner")
            parts.append(f)
        sup = pd.concat(parts, ignore_index=True)
        sup = sup.merge(weekly_cal.rename(columns={"week_start": "target_week"}), on="target_week", how="left")
        sup = sup.merge(lag52, on=SERIES_KEY + ["target_week"], how="left")
        sup["lag52_target"] = sup["lag52_target_raw"] / sup["scale"]
        sup["y_scaled"] = sup["y_target"] / sup["scale"]
        return sup.dropna(subset=["y_scaled", "lag1"])

    def _inference(self, train: pd.DataFrame, feats: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        last = feats.sort_values("week_start").groupby(SERIES_KEY, sort=False).tail(1)
        inf = future.merge(last.drop(columns=["week_start"]), on=SERIES_KEY, how="left")
        inf = inf.merge(_lag52_lookup(train), on=SERIES_KEY + ["target_week"], how="left")
        inf["lag52_target"] = inf["lag52_target_raw"] / inf["scale"]
        return inf

    def _feature_cols(self) -> list[str]:
        cols = ["h", "log_scale"] + [f"lag{lag}" for lag in LAGS] + [f"roll{w}" for w in ROLLS]
        cols += (
            [
                "roll8_std",
                "trend4_13",
                "price_last",
                "price_ratio4",
                "promo_last",
                "promo_roll4",
                "zero_days_last",
                "series_age",
                "lag52_target",
            ]
            + list(CAL_COLS)
            + list(CAT_COLS)
        )
        return cols

    def _encode(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        x = df[self._feature_cols()].copy()
        for c in CAT_COLS:
            if fit:
                self._categories[c] = sorted(x[c].dropna().unique().tolist())
            x[c] = pd.Categorical(x[c], categories=self._categories[c])
        return x

    # ------------------------------------------------------------------ api
    def fit_predict(self, train: pd.DataFrame, future: pd.DataFrame, quantiles: tuple[float, ...]) -> pd.DataFrame:
        horizon = int(future["h"].max())
        weekly_cal = (
            future[["target_week"] + list(CAL_COLS)].drop_duplicates("target_week").rename(columns={"target_week": "week_start"})
        )
        # Calendar for historical target weeks comes from the training table itself.
        hist_cal = train[["week_start"] + list(CAL_COLS)].drop_duplicates("week_start")
        cal = pd.concat([hist_cal, weekly_cal]).drop_duplicates("week_start")

        feats = self._base_features(train)
        sup = self._supervised(train, feats, cal, horizon)
        inf = self._inference(
            train,
            feats,
            future.drop(columns=[c for c in CAL_COLS if c in future.columns]).merge(
                cal.rename(columns={"week_start": "target_week"}), on="target_week", how="left"
            ),
        )

        x_train = self._encode(sup, fit=True)
        x_inf = self._encode(inf, fit=False)
        y_train = sup["y_scaled"].to_numpy()

        pred = inf[SERIES_KEY + ["h", "target_week"]].copy()
        for q in quantiles:
            model = LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=31,
                min_child_samples=40,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=self.seed,
                verbose=-1,
            )
            model.fit(x_train, y_train, categorical_feature=list(CAT_COLS))
            pred[quantile_col(q)] = model.predict(x_inf) * inf["scale"].to_numpy()
        return finalize_quantiles(pred, quantiles)
