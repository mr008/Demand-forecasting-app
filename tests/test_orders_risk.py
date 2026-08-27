import numpy as np
import pandas as pd
import pytest

from supply_pipeline import distributions as dist
from supply_pipeline.orders import moq_round_up, protection_demand
from supply_pipeline.risk import daily_forecast, label_lookahead, weekday_shares


def test_moq_round_up() -> None:
    qty = np.array([0.0, 1.0, 100.0, 101.0, 250.0])
    moq = np.full(5, 100.0)
    assert moq_round_up(qty, moq).tolist() == [0.0, 100.0, 100.0, 200.0, 300.0]


def test_lognormal_fit_recovers_quantiles() -> None:
    mu, sigma = dist.lognormal_params(np.array([100.0]), np.array([150.0]))
    assert dist.quantile(mu, sigma, 0.5)[0] == pytest.approx(100.0)
    assert dist.quantile(mu, sigma, 0.9)[0] == pytest.approx(150.0, rel=1e-6)


def test_prob_and_shortfall_monotone() -> None:
    mu, sigma = dist.lognormal_params(np.array([100.0, 100.0, 100.0]), np.array([160.0] * 3))
    stock = np.array([0.0, 100.0, 400.0])
    p = dist.prob_demand_exceeds(stock, mu, sigma)
    assert p[0] == 1.0
    assert p[1] == pytest.approx(0.5, abs=1e-6)
    assert p[2] < 0.01
    es = dist.expected_shortfall(stock, mu, sigma)
    assert es[0] == pytest.approx(dist.mean(mu, sigma)[0])
    assert es[0] > es[1] > es[2] >= 0


def test_protection_demand_sums_first_weeks() -> None:
    fc = pd.DataFrame(
        {"upc": [1, 1, 1], "cedis": ["X"] * 3, "h": [1, 2, 3], "q50": [10.0, 20.0, 30.0], "q90": [15.0, 25.0, 35.0]}
    )
    d = protection_demand(fc, 2)
    assert d.loc[0, "q50"] == 30.0 and d.loc[0, "q90"] == 40.0 and d.loc[0, "weeks_available"] == 2


def test_daily_forecast_uses_weekday_shares() -> None:
    dates = pd.date_range("2026-03-16", "2026-03-22")
    daily = pd.DataFrame({"upc": 1, "cedis": "X", "date": pd.date_range("2026-01-05", "2026-03-15")})
    daily["sell_out_pzs"] = np.where(daily["date"].dt.dayofweek == 5, 70.0, 0.0)  # all sales on Saturdays
    shares = weekday_shares(daily, pd.Timestamp("2026-03-15"))
    fc = pd.DataFrame({"upc": [1], "cedis": ["X"], "target_week": [pd.Timestamp("2026-03-16")], "q50": [700.0], "q90": [900.0]})
    df = daily_forecast(fc, shares, dates).set_index("date")
    assert df.loc["2026-03-21", "d50"] == pytest.approx(700.0)
    assert df.loc["2026-03-18", "d50"] == pytest.approx(0.0)


def test_label_lookahead_requires_full_window() -> None:
    dates = pd.date_range("2026-03-20", "2026-03-30")
    s = pd.DataFrame({"upc": 1, "cedis": "X", "date": dates, "stockout_store_share": 0.0})
    s.loc[s["date"] == "2026-03-27", "stockout_store_share"] = 0.5
    lab = label_lookahead(s, 0.25).set_index("date")
    assert lab.loc["2026-03-20", "label_7d"] == 1.0  # 03-27 is within 7 days
    assert lab.loc["2026-03-21", "label_7d"] == 1.0
    assert lab.loc["2026-03-23", "label_7d"] == 1.0
    assert np.isnan(lab.loc["2026-03-24", "label_7d"])  # window 03-25..03-31 not fully observed
    assert lab.loc["2026-03-27", "episode_onset"] == True  # noqa: E712
    assert lab["episode_onset"].sum() == 1
