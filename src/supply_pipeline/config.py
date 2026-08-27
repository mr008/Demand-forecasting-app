"""Typed configuration loaded from ``config.toml``.

Every tunable in the pipeline lives here so that the README can point at one file
for "assumptions". Paths are resolved relative to the config file's directory.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class PathsConfig:
    root: Path
    raw_dir: Path
    interim_dir: Path
    output_dir: Path
    reports_dir: Path
    sell_out: Path
    inventory: Path
    store_catalog: Path
    upc_catalog: Path

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.reports_dir / "tables"

    def ensure_dirs(self) -> None:
        for d in (self.interim_dir, self.output_dir, self.figures_dir, self.tables_dir):
            d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DataConfig:
    as_of: date
    first_week_start: date
    last_complete_week_start: date
    cold_start_weeks: int
    outlier_mad_z: float
    clip_negative_on_hand: bool
    stockout_store_share: float


@dataclass(frozen=True)
class ForecastConfig:
    horizon_weeks: int
    quantiles: tuple[float, ...]
    backtest_folds: int
    backtest_step_weeks: int
    seed: int


@dataclass(frozen=True)
class RiskConfig:
    eval_origin: date
    prob_threshold: float
    contamination: float


@dataclass(frozen=True)
class OrdersConfig:
    review_period_days: int
    cost_ratio: float
    service_level: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalConfig:
    max_weekly_missing_share: float
    min_series_with_inventory_share: float
    max_wape_selected: float
    min_rel_improvement_vs_naive: float
    max_abs_bias: float
    coverage_90_range: tuple[float, float]
    coverage_80_range: tuple[float, float]
    min_risk_recall: float
    max_prob_false_alarm_rate: float
    min_share_lines_at_target_sl: float


@dataclass(frozen=True)
class Config:
    paths: PathsConfig
    data: DataConfig
    forecast: ForecastConfig
    risk: RiskConfig
    orders: OrdersConfig
    eval: EvalConfig


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.toml"


def load_config(path: Path | str | None = None) -> Config:
    """Parse ``config.toml`` into frozen dataclasses."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with cfg_path.open("rb") as fh:
        raw = tomllib.load(fh)

    root = cfg_path.resolve().parent
    p = raw["paths"]
    raw_dir = root / p["raw_dir"]
    paths = PathsConfig(
        root=root,
        raw_dir=raw_dir,
        interim_dir=root / p["interim_dir"],
        output_dir=root / p["output_dir"],
        reports_dir=root / p["reports_dir"],
        sell_out=raw_dir / p["sell_out"],
        inventory=raw_dir / p["inventory"],
        store_catalog=raw_dir / p["store_catalog"],
        upc_catalog=raw_dir / p["upc_catalog"],
    )

    d = raw["data"]
    data = DataConfig(
        as_of=date.fromisoformat(d["as_of"]),
        first_week_start=date.fromisoformat(d["first_week_start"]),
        last_complete_week_start=date.fromisoformat(d["last_complete_week_start"]),
        cold_start_weeks=int(d["cold_start_weeks"]),
        outlier_mad_z=float(d["outlier_mad_z"]),
        clip_negative_on_hand=bool(d["clip_negative_on_hand"]),
        stockout_store_share=float(d["stockout_store_share"]),
    )

    f = raw["forecast"]
    forecast = ForecastConfig(
        horizon_weeks=int(f["horizon_weeks"]),
        quantiles=tuple(float(q) for q in f["quantiles"]),
        backtest_folds=int(f["backtest_folds"]),
        backtest_step_weeks=int(f["backtest_step_weeks"]),
        seed=int(f["seed"]),
    )

    r = raw["risk"]
    risk = RiskConfig(
        eval_origin=date.fromisoformat(r["eval_origin"]),
        prob_threshold=float(r["prob_threshold"]),
        contamination=float(r["contamination"]),
    )

    o = raw["orders"]
    orders = OrdersConfig(
        review_period_days=int(o["review_period_days"]),
        cost_ratio=float(o["cost_ratio"]),
        service_level={k: float(v) for k, v in o["service_level"].items()},
    )

    e = raw["eval"]
    eval_cfg = EvalConfig(
        max_weekly_missing_share=float(e["max_weekly_missing_share"]),
        min_series_with_inventory_share=float(e["min_series_with_inventory_share"]),
        max_wape_selected=float(e["max_wape_selected"]),
        min_rel_improvement_vs_naive=float(e["min_rel_improvement_vs_naive"]),
        max_abs_bias=float(e["max_abs_bias"]),
        coverage_90_range=(float(e["coverage_90_range"][0]), float(e["coverage_90_range"][1])),
        coverage_80_range=(float(e["coverage_80_range"][0]), float(e["coverage_80_range"][1])),
        min_risk_recall=float(e["min_risk_recall"]),
        max_prob_false_alarm_rate=float(e["max_prob_false_alarm_rate"]),
        min_share_lines_at_target_sl=float(e["min_share_lines_at_target_sl"]),
    )

    return Config(paths=paths, data=data, forecast=forecast, risk=risk, orders=orders, eval=eval_cfg)
