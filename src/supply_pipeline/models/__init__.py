"""Forecasting models sharing a common fit/predict interface."""

from __future__ import annotations

from supply_pipeline.config import Config
from supply_pipeline.models.base import Forecaster, make_future
from supply_pipeline.models.ets import ETSDamped
from supply_pipeline.models.lgbm import LightGBMQuantile
from supply_pipeline.models.naive import MovingAverage, SeasonalNaive

# Simpler models first: ties in model selection resolve toward the earlier entry.
MODEL_ORDER = ("ma4", "seasonal_naive", "ets", "lgbm")
FALLBACK_MODEL = "ma4"


def build_models(cfg: Config) -> dict[str, Forecaster]:
    models: list[Forecaster] = [
        MovingAverage(4),
        SeasonalNaive(),
        ETSDamped(seed=cfg.forecast.seed),
        LightGBMQuantile(seed=cfg.forecast.seed),
    ]
    return {m.name: m for m in models}


__all__ = ["FALLBACK_MODEL", "MODEL_ORDER", "Forecaster", "build_models", "make_future"]
