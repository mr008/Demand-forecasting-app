from datetime import date

from supply_pipeline.config import DEFAULT_CONFIG_PATH, load_config


def test_default_config_loads() -> None:
    cfg = load_config()
    assert DEFAULT_CONFIG_PATH.exists()
    assert cfg.forecast.horizon_weeks == 8
    assert cfg.data.as_of == date(2026, 4, 2)
    assert cfg.paths.sell_out.name == "challenge_daily_sell_out_pricing.csv"
    assert set(cfg.orders.service_level) == {"A", "B", "C"}


def test_cli_help_lists_stages() -> None:
    from supply_pipeline.cli import STAGE_ORDER, build_parser

    parser = build_parser()
    help_text = parser.format_help()
    for stage in STAGE_ORDER:
        assert stage in help_text
