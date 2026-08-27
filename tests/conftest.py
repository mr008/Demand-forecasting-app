"""Session fixtures: a synthetic dataset and one full pipeline run over it.

The pipeline run is expensive (~1-2 min) so it is executed once per session and
shared by every end-to-end test. Tests that only need the raw synthetic files
use ``synthetic`` and stay fast.
"""

from __future__ import annotations

import logging

import pytest

from supply_pipeline import cli
from supply_pipeline.config import Config, load_config
from tests.synthetic import SyntheticDataset, make_dataset


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> SyntheticDataset:
    root = tmp_path_factory.mktemp("synthetic")
    return make_dataset(root)


@pytest.fixture(scope="session")
def synthetic_cfg(synthetic: SyntheticDataset) -> Config:
    return load_config(synthetic.config_path)


@pytest.fixture(scope="session")
def pipeline_run(synthetic: SyntheticDataset, synthetic_cfg: Config) -> Config:
    """Run every stage except ``evaluate`` (tested explicitly so a failing gate reads as a test failure)."""
    logging.getLogger("supply_pipeline").setLevel(logging.WARNING)
    for stage in cli.STAGE_ORDER:
        if stage == "evaluate":
            continue
        assert cli.main(["--config", str(synthetic.config_path), stage]) == 0, stage
    return synthetic_cfg
