"""Stage 'evaluate': quality gates over the artifacts of a completed run.

Each gate compares one measured value against a threshold from ``[eval]`` in
``config.toml`` (or a structural invariant) and is either *hard* (the run is
not fit to ship; the stage exits non-zero after writing the scorecard) or
*soft* (reported as a warning). The scorecard is written to
``reports/tables/eval_scorecard.csv`` and ``reports/eval_report.md`` so a CI
job or a reviewer can read pass/fail without re-running anything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from pptx import Presentation

from supply_pipeline.config import Config
from supply_pipeline.metrics import quantile_col

log = logging.getLogger(__name__)

SERIES_KEY = ["upc", "cedis"]
Severity = Literal["hard", "soft"]
Op = Literal["<=", ">=", "==", "in"]

REQUIRED_SUMMARY_SECTIONS = (
    "## 1. Scope and data",
    "## 2. Forecasting",
    "## 3. Stock-out risk",
    "## 4. Supply order",
    "## 5. Where not to trust the model",
)
EXPECTED_FIGURES = 10
EXPECTED_SLIDES = 10


class EvalError(RuntimeError):
    """Raised when at least one hard gate fails."""


@dataclass
class Gate:
    area: str
    name: str
    value: float
    op: Op
    threshold: float | tuple[float, float]
    severity: Severity
    note: str = ""
    passed: bool = field(init=False)

    def __post_init__(self) -> None:
        v = self.value
        thr = self.threshold
        if np.isnan(v):
            self.passed = False
        elif isinstance(thr, tuple):
            if self.op != "in":  # pragma: no cover
                raise ValueError(f"range threshold requires op 'in', got {self.op!r}")
            lo, hi = thr
            self.passed = bool(lo <= v <= hi)
        elif self.op == "<=":
            self.passed = bool(v <= thr)
        elif self.op == ">=":
            self.passed = bool(v >= thr)
        elif self.op == "==":
            self.passed = bool(abs(v - thr) < 1e-9)
        else:  # pragma: no cover
            raise ValueError(self.op)

    @property
    def status(self) -> str:
        if self.passed:
            return "pass"
        return "FAIL" if self.severity == "hard" else "warn"

    def row(self) -> dict[str, object]:
        thr = f"[{self.threshold[0]}, {self.threshold[1]}]" if isinstance(self.threshold, tuple) else str(self.threshold)
        return {
            "area": self.area,
            "gate": self.name,
            "value": self.value,
            "op": self.op,
            "threshold": thr,
            "severity": self.severity,
            "status": self.status,
            "note": self.note,
        }


def _weighted(df: pd.DataFrame, col: str) -> float:
    return float(np.average(df[col], weights=df["n"])) if len(df) else float("nan")


# --------------------------------------------------------------------------- gate groups
def data_gates(cfg: Config) -> list[Gate]:
    p = cfg.paths
    weekly = pd.read_parquet(p.interim_dir / "weekly.parquet")
    daily = pd.read_parquet(p.interim_dir / "daily.parquet")
    series = pd.read_parquet(p.interim_dir / "series.parquet")
    gates: list[Gate] = []

    dup = int(weekly.duplicated(SERIES_KEY + ["week_start"]).sum())
    gates.append(Gate("data", "weekly_duplicate_keys", dup, "==", 0, "hard"))

    # Weekly y must equal the daily sum for fully observed weeks.
    d = daily.copy()
    d["week_start"] = (d["date"] - pd.to_timedelta(d["date"].dt.dayofweek, unit="D")).dt.normalize()
    ds = d.groupby(SERIES_KEY + ["week_start"])["sell_out_pzs"].agg(["sum", "count"]).reset_index()
    m = weekly.merge(ds, on=SERIES_KEY + ["week_start"], how="inner")
    full = m[m["count"] == 7]
    rel = ((full["y"] - full["sum"]).abs() / full["sum"].clip(lower=1e-9)).max() if len(full) else float("nan")
    gates.append(
        Gate(
            "data",
            "weekly_reconciles_to_daily_max_rel_diff",
            float(rel),
            "<=",
            1e-6,
            "hard",
            f"{len(full)} fully observed weeks compared",
        )
    )

    gates.append(
        Gate("data", "weekly_y_missing_share", float(weekly["y"].isna().mean()), "<=", cfg.eval.max_weekly_missing_share, "soft")
    )
    gates.append(
        Gate(
            "data",
            "series_with_inventory_share",
            float(series["has_inventory"].mean()),
            ">=",
            cfg.eval.min_series_with_inventory_share,
            "soft",
        )
    )
    gates.append(Gate("data", "n_series", float(len(series)), ">=", 1, "hard"))
    return gates


def forecast_gates(cfg: Config) -> list[Gate]:
    p = cfg.paths
    t = p.tables_dir
    q = cfg.forecast.quantiles
    qcols = [quantile_col(x) for x in sorted(q)]
    fc = pd.read_csv(p.output_dir / f"forecast_{cfg.data.last_complete_week_start}.csv")
    series = pd.read_parquet(p.interim_dir / "series.parquet")
    cluster = pd.read_csv(t / "backtest_metrics_cluster.csv")
    selection = pd.read_csv(t / "model_selection.csv")
    hold = pd.read_csv(t / "backtest_metrics_holdout_calibrated.csv")
    calib = pd.read_csv(t / "interval_calibration.csv")
    gates: list[Gate] = []

    # Structure of the final forecast.
    arr = fc[qcols].to_numpy(dtype=float)
    mono = float(np.mean(np.all(np.diff(arr, axis=1) >= -1e-9, axis=1)))
    gates.append(Gate("forecast", "quantiles_monotone_share", mono, "==", 1.0, "hard"))
    gates.append(Gate("forecast", "quantiles_non_negative_share", float(np.mean(arr >= 0)), "==", 1.0, "hard"))
    gates.append(Gate("forecast", "quantiles_finite_share", float(np.mean(np.isfinite(arr))), "==", 1.0, "hard"))
    per_series = fc.groupby(SERIES_KEY)["h"].nunique()
    gates.append(
        Gate(
            "forecast",
            "series_with_full_horizon_share",
            float((per_series == cfg.forecast.horizon_weeks).mean()),
            "==",
            1.0,
            "hard",
        )
    )
    gates.append(
        Gate(
            "forecast",
            "series_covered",
            float(per_series.shape[0]),
            "==",
            float(len(series)),
            "hard",
            "every series in series.parquet gets a forecast",
        )
    )
    one_model = fc.groupby(SERIES_KEY)["model"].nunique()
    gates.append(Gate("forecast", "one_model_per_series_share", float((one_model == 1).mean()), "==", 1.0, "hard"))

    # Accuracy of the selected models, weighted by evaluated rows.
    sel = cluster.merge(selection[["cluster", "selected_model"]], on="cluster")
    sel = sel[sel["model"] == sel["selected_model"]]
    naive = cluster[cluster["model"].isin(["ma4", "seasonal_naive"])]
    best_naive = min(_weighted(naive[naive["model"] == m], "wape") for m in naive["model"].unique())
    wape_sel = _weighted(sel, "wape")
    gates.append(Gate("forecast", "selected_wape_overall", wape_sel, "<=", cfg.eval.max_wape_selected, "hard"))
    gates.append(
        Gate(
            "forecast",
            "rel_improvement_vs_best_naive",
            float(1 - wape_sel / best_naive),
            ">=",
            cfg.eval.min_rel_improvement_vs_naive,
            "soft",
            f"best naive WAPE {best_naive:.3f}",
        )
    )
    gates.append(Gate("forecast", "selected_abs_bias", abs(_weighted(sel, "bias")), "<=", cfg.eval.max_abs_bias, "soft"))

    # Calibrated interval coverage on held-out folds.
    hs = hold.merge(selection[["cluster", "selected_model"]], on="cluster")
    hs = hs[hs["model"] == hs["selected_model"]]
    gates.append(
        Gate("forecast", "holdout_coverage_90_calibrated", _weighted(hs, "coverage_90"), "in", cfg.eval.coverage_90_range, "hard")
    )
    gates.append(
        Gate("forecast", "holdout_coverage_80_calibrated", _weighted(hs, "coverage_80"), "in", cfg.eval.coverage_80_range, "soft")
    )
    n_models = cluster["model"].nunique()
    n_clusters = cluster["cluster"].nunique()
    gates.append(
        Gate(
            "forecast",
            "calibration_table_complete",
            float(len(calib)),
            "==",
            float(2 * n_models * n_clusters),
            "hard",
            "two intervals per model x cluster",
        )
    )
    gates.append(
        Gate("forecast", "selection_covers_all_clusters", float(selection["cluster"].nunique()), "==", float(n_clusters), "hard")
    )
    return gates


def risk_gates(cfg: Config) -> list[Gate]:
    p = cfg.paths
    methods = pd.read_csv(p.tables_dir / "risk_eval_methods.csv").set_index("method")
    alerts = pd.read_csv(p.output_dir / f"risk_alerts_{cfg.data.as_of}.csv")
    series = pd.read_parquet(p.interim_dir / "series.parquet")
    gates: list[Gate] = []
    gates.append(Gate("risk", "best_method_recall", float(methods["recall"].max()), ">=", cfg.eval.min_risk_recall, "soft"))
    gates.append(
        Gate(
            "risk",
            "prob_method_false_alarm_rate",
            float(methods.loc["prob", "false_alarm_rate"]),
            "<=",
            cfg.eval.max_prob_false_alarm_rate,
            "soft",
        )
    )
    gates.append(Gate("risk", "methods_evaluated", float(len(methods)), "==", 3.0, "hard"))
    gates.append(
        Gate(
            "risk", "alert_rows_vs_series_with_inventory", float(len(alerts)), "==", float(series["has_inventory"].sum()), "hard"
        )
    )
    gates.append(
        Gate(
            "risk",
            "severity_values_valid_share",
            float(alerts["severity"].isin(["none", "medium", "high"]).mean()),
            "==",
            1.0,
            "hard",
        )
    )
    gates.append(
        Gate("risk", "p_stockout_in_unit_interval_share", float(alerts["p_stockout_7d"].between(0, 1).mean()), "==", 1.0, "hard")
    )
    gates.append(
        Gate(
            "risk",
            "high_severity_implies_prob_or_cover",
            float(
                (
                    alerts.loc[alerts["severity"] == "high"].pipe(lambda a: (a["p_stockout_7d"] > 0.75) | (a["cover_days"] < 7))
                ).mean()
                if (alerts["severity"] == "high").any()
                else 1.0
            ),
            "==",
            1.0,
            "hard",
        )
    )
    return gates


def order_gates(cfg: Config) -> list[Gate]:
    p = cfg.paths
    o = pd.read_csv(p.output_dir / f"supply_order_{cfg.data.as_of}.csv")
    series = pd.read_parquet(p.interim_dir / "series.parquet")
    port = pd.read_csv(p.tables_dir / "order_summary_portfolio.csv").iloc[0]
    gates: list[Gate] = []
    gates.append(Gate("orders", "one_line_per_series", float(len(o)), "==", float(len(series)), "hard"))
    gates.append(Gate("orders", "duplicate_lines", float(o.duplicated(SERIES_KEY).sum()), "==", 0, "hard"))
    key_cols = ["order_qty", "implied_service_level", "working_capital", "demand_p50", "on_hand_projected", "unit_value"]
    gates.append(Gate("orders", "key_columns_nan_count", float(o[key_cols].isna().sum().sum()), "==", 0, "hard"))
    gates.append(Gate("orders", "order_qty_non_negative_share", float((o["order_qty"] >= 0).mean()), "==", 1.0, "hard"))
    moq_ok = ((o["order_qty"] % o["moq"]).abs() < 1e-6) | (o["order_qty"] == 0)
    gates.append(Gate("orders", "order_qty_moq_multiple_share", float(moq_ok.mean()), "==", 1.0, "hard"))
    disc = o[o["flags"].fillna("").str.contains("discontinued")]
    gates.append(
        Gate(
            "orders",
            "discontinued_orders_zero_share",
            float((disc["order_qty"] == 0).mean()) if len(disc) else 1.0,
            "==",
            1.0,
            "hard",
        )
    )
    wc_ok = (o["working_capital"] - o["order_qty"] * o["unit_value"]).abs() <= 1e-3 * o["working_capital"].abs().clip(lower=1)
    gates.append(Gate("orders", "working_capital_consistent_share", float(wc_ok.mean()), "==", 1.0, "hard"))
    gates.append(
        Gate(
            "orders",
            "service_level_in_unit_interval_share",
            float(o["implied_service_level"].between(0, 1).mean()),
            "==",
            1.0,
            "hard",
        )
    )
    clean = o[o["flags"].fillna("") == ""]
    at_target = float((clean["implied_service_level"] >= clean["service_level_target"] - 1e-6).mean()) if len(clean) else 1.0
    gates.append(
        Gate(
            "orders",
            "unflagged_lines_at_target_service_level_share",
            at_target,
            ">=",
            cfg.eval.min_share_lines_at_target_sl,
            "soft",
            f"{len(clean)} unflagged lines",
        )
    )
    gates.append(
        Gate(
            "orders",
            "portfolio_service_level_vs_ma4",
            float(port["service_level_weighted"] - port["ma4_service_level_weighted"]),
            ">=",
            -1e-9,
            "soft",
        )
    )
    gates.append(
        Gate(
            "orders",
            "lines_below_90_vs_ma4",
            float(port["ma4_lines_below_90"] - port["lines_below_90"]),
            ">=",
            0,
            "soft",
            "recommendation should not leave more lines below 90% than the naive policy",
        )
    )
    return gates


def report_gates(cfg: Config) -> list[Gate]:
    p = cfg.paths
    gates: list[Gate] = []
    summary = p.reports_dir / "summary.md"
    text = summary.read_text(encoding="utf-8") if summary.exists() else ""
    gates.append(
        Gate(
            "report",
            "summary_sections_present",
            float(sum(s in text for s in REQUIRED_SUMMARY_SECTIONS)),
            "==",
            float(len(REQUIRED_SUMMARY_SECTIONS)),
            "hard",
        )
    )
    gates.append(
        Gate("report", "figures_written", float(len(list(p.figures_dir.glob("*.png")))), ">=", float(EXPECTED_FIGURES), "hard")
    )
    deck = p.reports_dir / "deck.pptx"
    n_slides = len(Presentation(str(deck)).slides) if deck.exists() else 0
    gates.append(Gate("report", "deck_slides", float(n_slides), "==", float(EXPECTED_SLIDES), "hard"))
    return gates


# --------------------------------------------------------------------------- stage
def run_gates(cfg: Config) -> pd.DataFrame:
    gates = data_gates(cfg) + forecast_gates(cfg) + risk_gates(cfg) + order_gates(cfg) + report_gates(cfg)
    return pd.DataFrame([g.row() for g in gates])


def write_scorecard(cfg: Config, scorecard: pd.DataFrame) -> None:
    p = cfg.paths
    scorecard.to_csv(p.tables_dir / "eval_scorecard.csv", index=False)
    n_fail = int((scorecard["status"] == "FAIL").sum())
    n_warn = int((scorecard["status"] == "warn").sum())
    lines = [
        "# Evaluation scorecard",
        "",
        f"{len(scorecard)} gates: {int((scorecard['status'] == 'pass').sum())} pass, {n_warn} warn, {n_fail} FAIL.",
        "",
        "| area | gate | value | op | threshold | severity | status | note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in scorecard.itertuples(index=False):
        lines.append(f"| {r.area} | {r.gate} | {r.value:.4g} | {r.op} | {r.threshold} | {r.severity} | {r.status} | {r.note} |")
    (p.reports_dir / "eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cfg: Config) -> None:
    scorecard = run_gates(cfg)
    write_scorecard(cfg, scorecard)
    for r in scorecard.itertuples(index=False):
        if r.status != "pass":
            log.warning("%s gate %s/%s: value %.4g %s %s", r.status, r.area, r.gate, r.value, r.op, r.threshold)
    log.info("eval: %s", scorecard["status"].value_counts().to_dict())
    failed = scorecard[scorecard["status"] == "FAIL"]
    if len(failed):
        raise EvalError(f"{len(failed)} hard gate(s) failed: {failed['gate'].tolist()}")
