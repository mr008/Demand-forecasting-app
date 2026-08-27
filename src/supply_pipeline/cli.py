"""Command-line entry point.

``python -m supply_pipeline run`` executes every stage in order; each stage is also
exposed as its own sub-command so a single step can be re-run after a change.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable
from pathlib import Path

from supply_pipeline.config import Config, load_config

log = logging.getLogger("supply_pipeline")

StageFn = Callable[[Config], None]


def _stage_registry() -> dict[str, StageFn]:
    # Imported lazily so that ``--help`` stays fast and a broken stage module
    # does not prevent other stages from running.
    from supply_pipeline import backtest, data, deck, evaluate, forecast, orders, report, risk

    return {
        "prepare": data.run,
        "backtest": backtest.run,
        "forecast": forecast.run,
        "risk": risk.run,
        "orders": orders.run,
        "report": report.run,
        "deck": deck.run,
        "evaluate": evaluate.run,
    }


STAGE_ORDER = ("prepare", "backtest", "forecast", "risk", "orders", "report", "deck", "evaluate")


def _run_stage(name: str, fn: StageFn, cfg: Config) -> None:
    log.info("=== stage: %s ===", name)
    t0 = time.perf_counter()
    fn(cfg)
    log.info("=== stage: %s done in %.1fs ===", name, time.perf_counter() - t0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="supply_pipeline", description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="path to config.toml")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="run every stage in order")
    for name in STAGE_ORDER:
        sub.add_parser(name, help=f"run only the '{name}' stage")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config(args.config)
    cfg.paths.ensure_dirs()
    stages = _stage_registry()

    if args.command == "run":
        for name in STAGE_ORDER:
            _run_stage(name, stages[name], cfg)
    else:
        _run_stage(args.command, stages[args.command], cfg)
    return 0
