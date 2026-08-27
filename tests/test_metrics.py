import numpy as np
import pandas as pd
import pytest

from supply_pipeline.metrics import alert_metrics, bias, coverage, lead_time_to_alert, mape, pinball, wape


def test_wape_and_bias() -> None:
    y = np.array([100.0, 200.0, 300.0])
    yhat = np.array([110.0, 190.0, 330.0])
    assert wape(y, yhat) == pytest.approx(50 / 600)
    assert bias(y, yhat) == pytest.approx(30 / 600)
    assert wape(y, y) == 0.0


def test_mape_ignores_zero_actuals() -> None:
    y = np.array([0.0, 100.0])
    yhat = np.array([5.0, 150.0])
    assert mape(y, yhat) == pytest.approx(0.5)
    assert np.isnan(mape(np.zeros(3), np.ones(3)))


def test_pinball_is_asymmetric() -> None:
    y = np.array([10.0])
    assert pinball(y, np.array([8.0]), 0.9) == pytest.approx(0.9 * 2)  # under-forecast at q90 costs 0.9/unit
    assert pinball(y, np.array([12.0]), 0.9) == pytest.approx(0.1 * 2)  # over-forecast costs 0.1/unit


def test_coverage() -> None:
    y = np.array([1.0, 5.0, 9.0, 20.0])
    assert coverage(y, np.zeros(4), np.full(4, 10.0)) == 0.75


def test_alert_metrics_counts() -> None:
    alert = np.array([1, 1, 0, 0, 1], dtype=bool)
    event = np.array([1, 0, 1, 0, 0], dtype=bool)
    m = alert_metrics(alert, event)
    assert m["n_alerts"] == 3 and m["n_events"] == 2
    assert m["precision"] == pytest.approx(1 / 3)
    assert m["recall"] == pytest.approx(1 / 2)
    assert m["false_alarm_rate"] == pytest.approx(2 / 3)


def test_lead_time_to_alert_picks_earliest_alert_in_window() -> None:
    alerts = pd.DataFrame(
        {"upc": [1, 1, 1], "cedis": ["X"] * 3, "date": pd.to_datetime(["2026-03-20", "2026-03-22", "2026-03-30"])}
    )
    events = pd.DataFrame({"upc": [1, 2], "cedis": ["X", "X"], "date": pd.to_datetime(["2026-03-25", "2026-03-25"])})
    lt = lead_time_to_alert(alerts, events, max_lead=7)
    assert lt.iloc[0] == 5.0  # 03-20 is the earliest alert within 7 days before 03-25
    assert np.isnan(lt.iloc[1])
