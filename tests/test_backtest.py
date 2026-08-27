import itertools

import numpy as np
import pandas as pd

from supply_pipeline.backtest import apply_calibration, calibrate_intervals, fold_origins, select_models
from supply_pipeline.config import load_config

Q = (0.05, 0.10, 0.50, 0.90, 0.95)


def test_fold_origins_leave_full_horizon_observed() -> None:
    cfg = load_config()
    origins = fold_origins(cfg)
    assert len(origins) == cfg.forecast.backtest_folds
    last = pd.Timestamp(cfg.data.last_complete_week_start)
    assert origins[-1] + pd.Timedelta(weeks=cfg.forecast.horizon_weeks) == last
    steps = {(b - a).days for a, b in itertools.pairwise(origins)}
    assert steps == {7 * cfg.forecast.backtest_step_weeks}
    assert all(o.dayofweek == 0 for o in origins)  # Mondays


def _pred(n: int, width: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.normal(100, 10, n)
    return pd.DataFrame(
        {
            "model": "m",
            "cluster": "c",
            "y": y,
            "q05": 100 - 1.645 * width,
            "q10": 100 - 1.28 * width,
            "q50": 100.0,
            "q90": 100 + 1.28 * width,
            "q95": 100 + 1.645 * width,
        }
    )


def test_calibration_widens_overconfident_and_narrows_overwide() -> None:
    narrow = _pred(4000, width=3.0)  # true sd is 10 -> intervals far too narrow
    wide = _pred(4000, width=30.0)  # intervals far too wide
    k_narrow = calibrate_intervals(narrow, Q)
    k_wide = calibrate_intervals(wide, Q)
    assert (k_narrow["k"] > 0).all()
    assert (k_wide["k"] < 0).all()
    cal = apply_calibration(narrow, k_narrow, Q)
    cov80 = ((cal["y"] >= cal["q10"]) & (cal["y"] <= cal["q90"])).mean()
    cov90 = ((cal["y"] >= cal["q05"]) & (cal["y"] <= cal["q95"])).mean()
    assert 0.76 <= cov80 <= 0.84
    assert 0.86 <= cov90 <= 0.94
    # quantiles stay ordered and non-negative, q50 untouched
    assert (cal["q05"] <= cal["q10"]).all() and (cal["q10"] <= cal["q50"]).all() and (cal["q90"] <= cal["q95"]).all()
    assert (cal["q50"] == 100.0).all()


def test_select_models_prefers_steadier_within_tolerance() -> None:
    rows = []
    for fold in range(4):
        rows.append(
            {
                "cluster": "c",
                "model": "lgbm",
                "fold": fold,
                "wape": [0.10, 0.20, 0.10, 0.20][fold],
                "bias": 0.0,
                "pinball": 0.0,
                "coverage_90": 0.9,
                "coverage_80": 0.8,
            }
        )
        rows.append(
            {
                "cluster": "c",
                "model": "ets",
                "fold": fold,
                "wape": 0.16,
                "bias": 0.0,
                "pinball": 0.0,
                "coverage_90": 0.9,
                "coverage_80": 0.8,
            }
        )
    sel = select_models(pd.DataFrame(rows))
    # lgbm mean 0.15 vs ets 0.16: within 0.02, ets has zero spread -> ets wins
    assert sel.loc[0, "selected_model"] == "ets"
    assert "lower fold spread" in sel.loc[0, "rationale"]
