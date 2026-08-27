import pandas as pd
import pytest

from supply_pipeline.blindtest import split_folds
from supply_pipeline.models import MODEL_ORDER


def test_split_folds_excludes_overlapping_folds() -> None:
    origins = pd.to_datetime(["2025-12-08", "2026-01-05", "2026-02-02"])
    pred = pd.DataFrame({"fold": [0, 1, 2], "origin": origins})
    usable, sealed = split_folds(pred, horizon_weeks=8)
    assert sealed == 2
    # fold 1's targets (01-12 .. 03-02) overlap the sealed window, so only fold 0 is usable
    assert usable == [0]


@pytest.mark.e2e
def test_blind_test_outputs(pipeline_run) -> None:
    t = pipeline_run.paths.tables_dir
    overall = pd.read_csv(t / "blind_test_overall.csv")
    cluster = pd.read_csv(t / "blind_test_cluster.csv")
    sel = pd.read_csv(t / "blind_test_selection.csv")
    assert set(overall["model"]) == set(MODEL_ORDER)
    assert (overall["n"] == overall["n"].iloc[0]).all()  # every model scored on the same sealed rows
    assert set(sel["cluster"]) == set(cluster["cluster"])
    assert (sel["regret"] >= -1e-9).all()
    assert sel["pre_registered_model"].isin(MODEL_ORDER).all()
    assert (pipeline_run.paths.figures_dir / "blind_test.png").exists()
