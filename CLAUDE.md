# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A take-home data-science challenge: a reproducible pipeline that forecasts weekly demand per SKU x distribution center, flags stock-out risk, and emits a recommended supply order. The brief is `docs/brief.pdf` (the em dash in its original filename breaks the Read tool; it has been renamed). The `CLAUDE.md` in the home directory (`C:\Users\KOS`) describes an unrelated Playwright workspace - ignore it here.

Design decisions and stage checkpoints are in the approved plan; the short version: both tracks (forecasting + risk) plus the bonus order engine, weekly grain, LightGBM global quantile model vs statsmodels ETS vs naive baselines, three risk methods (days-of-cover rule, forecast-probability, Isolation Forest), python-pptx deck.

## Commands

All from the repo root. The venv lives in `.venv/`; `run.ps1` / `run.sh` create it and install the package on first use.

- One-command full run: `.\run.ps1` (PowerShell) or `./run.sh` (bash)
- Full pipeline from an existing venv: `.venv\Scripts\python -m supply_pipeline run`
- Single stage: `.venv\Scripts\python -m supply_pipeline <prepare|backtest|forecast|risk|orders|report|deck|evaluate>` (add `-v` for debug logs, `--config path` for an alternative config)
- Tests: `.venv\Scripts\python -m pytest` (~20 s; `-m "not e2e"` for the ~2 s unit subset; single test: `-k test_name`). The e2e tests build a synthetic dataset via `tests/synthetic.py` and run the whole pipeline once per session (`pipeline_run` fixture in `tests/conftest.py`); change the generator, not the tests, when a new data quirk needs covering.
- Quality gates on the real run: `.venv\Scripts\python -m supply_pipeline evaluate` (also the last stage of `run`; hard-gate failure exits non-zero). Thresholds are in `[eval]` of `config.toml`; the synthetic config loosens accuracy gates in `tests/synthetic.py::_write_config`.
- Install after editing `pyproject.toml`: `.venv\Scripts\python -m pip install -e ".[dev]"`
- Notebooks are generated, not hand-edited: `.venv\Scripts\python notebooks\build_notebooks.py` then `.venv\Scripts\python -m nbconvert --execute --to notebook --inplace notebooks\01_eda.ipynb notebooks\02_results.ipynb` (needs a completed pipeline run). The `jupyter` metapackage is deliberately not installed (jupyterlab exceeds Windows path limits); open notebooks in VS Code/Cursor.

- Lint / format / types (all must be clean; CI enforces them): `.venv\Scripts\python -m ruff check src tests notebooks\build_notebooks.py`, `.venv\Scripts\python -m ruff format src tests notebooks\build_notebooks.py`, `.venv\Scripts\python -m mypy`. Config is in `pyproject.toml`; `RUF005` is deliberately ignored (`KEY + ["h"]` style), mypy targets 3.12 syntax because numpy's stubs need it, and `pandas-stubs` is deliberately *not* installed (its overloads crash mypy 2.3 and fight pandas idioms) - pandas is typed as `Any`.
- CI: `.github/workflows/ci.yml` runs lint+mypy, pytest on 3.11 and 3.13, and a full synthetic `run` (`python -m tests.synthetic build/synthetic` then `python -m supply_pipeline --config build/synthetic/config.toml run`). Real data never reaches CI.

Full run takes ~2.5 minutes (backtest ~1.5 min: 8 folds x 4 models; ETS fits are parallelised with joblib). `reports/` and `data/output/` are committed on purpose so reviewers can read results without the raw CSVs; regenerate them before committing pipeline changes.

## Architecture

`src/supply_pipeline/` is a flat package; `cli.py` runs stages in the fixed order `prepare -> backtest -> forecast -> risk -> orders -> report -> deck -> evaluate`, each stage being a module with a `run(cfg: Config) -> None`. Stages communicate only through files: `data/interim/*.parquet` (prepared tables, backtest predictions), `data/output/` (forecasts, alerts, the order CSV), `reports/tables|figures` (what the report and deck consume). Re-running one stage therefore requires its upstream artifacts to exist.

- `config.py` loads `config.toml` (stdlib `tomllib`) into frozen dataclasses. Every tunable and every data-policy assumption (as-of date, cold-start threshold, outlier z, stock-out label share, service-level targets, cost ratio) is a config value - add new knobs there, not as module constants.
- `schema.py` / `data.py` / `calendar_features.py` / `features.py`: load + validate raw CSVs, dedupe on `upc`, aggregate inventory store -> cedis via the store catalog, complete the daily calendar per series (gaps stay NaN), flag outliers, build the weekly table (ISO weeks, Monday start).
- `models/`: each model exposes the same `fit_predict(train, future, quantiles)` interface (`models/base.py`) so `backtest.py` can loop over them and select per ABC/XYZ cluster. `MODEL_ORDER` in `models/__init__.py` doubles as the simplicity tie-break order.
- `backtest.py` also fits the conformal-style interval calibration (`interval_calibration.csv`) on the first folds and reports held-out coverage on the last `CALIBRATION_HOLDOUT_FOLDS`; `forecast.py` applies it to every model before selection, so downstream stages always see calibrated quantiles.
- `distributions.py`: lognormal fitted to (q50, q90) - the one place that turns quantiles into P(stock-out), service level and expected lost sales for both `risk.py` and `orders.py`. Sigma is floored/capped there; do not add ad-hoc distribution maths elsewhere.
- `metrics.py`: WAPE, MAPE, bias, pinball, coverage; precision/recall/lead-time-to-alert/false-alarm rate. Pure functions, unit-tested.
- `risk.py` scores the 21-day inventory window with the forecast from `risk.eval_origin` (so evaluation is "as a planner would have seen it") and the as-of day with the final forecast. `orders.py` writes `data/output/supply_order_<as_of>.csv` with the recommended policy, two naive baselines and the forecast-only (`fq_*`) variant.
- Known data reality that shapes Track B: stock-outs in the window are chronic (same series short all 14 days), so lead-time-to-alert has almost no onsets to measure on, and sell-out does not drop during them. Report this rather than tuning around it.
- `plots.py`, `report.py`, `deck.py`: matplotlib PNGs, CSV/markdown tables, `reports/summary.md`, `reports/deck.pptx`.
- `notebooks/` are thin: they import from the package and display; no logic lives there.

## Data

Raw CSVs live in `data/raw/` (gitignored, 88 MB). All four join cleanly on `upc`, `store_nbr`, `cedis`; six DCs: `CCUL`, `CMD2`, `CMTY`, `MXLI`, `TLJ`, `VHSA`.

| File | Rows | Key | Notes |
|---|---|---|---|
| `challenge_daily_sell_out_pricing.csv` | 245,216 | `date, upc, cedis` | 2024-03-18 (a Monday) to 2026-04-10; 67 UPCs, 328 series; `sell_out_pzs` fractional; `promo_flag` on 12% |
| `challenge_inventory.csv` (75 MB) | 1,223,744 | `date, upc, store_nbr` | 2026-03-20 to 04-09 only; `on_hand_qty` negative in 7,116 rows |
| `challenge_store_catalog.csv` | 2,858 | `store_nbr` | store -> `cedis` |
| `challenge_upc_catalog.csv` | 70 | `prime_item_nbr` | 67 unique `upc`; `lead_time_days`=7 and `moq`=100 for all; `safety_stock_days` in {3,7,14}; `abc_class`/`xyz_class` |

Quirks the code must respect (verified against the raw files):

- Inventory coverage by date is uneven: 03-20..03-26 have 63 UPCs, 03-27..04-02 all 67, **04-03..04-09 only 4 UPCs / 962 stores**. `data.as_of = 2026-04-02` is the last complete snapshot.
- Three UPCs (`750101310119`, `750101310340`, `750101310341`) map to two `prime_item_nbr` each -> 70 catalog rows and 50 duplicate inventory keys. Everything is deduped on `upc`.
- 25% of sell-out series start late (shortest 301 days); 52-59 UPCs per cedis. Gaps are missing, not zero.
- 107 complete ISO weeks (2024-03-18 .. 2026-04-05); the partial week 04-06..04-10 is excluded from modelling.
- Loading the inventory CSV takes ~20-30 s; `prepare` caches parquet so later stages never re-read it.
