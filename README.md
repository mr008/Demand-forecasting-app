# Demand Forecasting & Supply Order Generation

[![CI](https://github.com/mr008/Demand-forecasting-app/actions/workflows/ci.yml/badge.svg)](https://github.com/mr008/Demand-forecasting-app/actions/workflows/ci.yml)

A reproducible pipeline for a CPG beverages portfolio (67 UPCs x 6 distribution centers) that

1. forecasts weekly demand per SKU x DC with calibrated prediction intervals (Track A),
2. flags stock-out / service-level risk (Track B), and
3. recommends the next replenishment order per SKU x DC with implied service level and working-capital impact versus naive policies (bonus).

Results are written to `reports/summary.md` (narrative + tables + figures), `reports/deck.pptx` (10 slides) and `data/output/supply_order_2026-04-02.csv` (the order file).

## How to run

Requirements: Python 3.11+ (developed on 3.13), Windows or Linux/macOS. Put the four challenge CSVs in `data/raw/`.

```powershell
.\run.ps1          # Windows: creates .venv, installs, runs every stage (~3 min)
./run.sh           # bash equivalent
```

Single stages (after the first run): `.venv\Scripts\python -m supply_pipeline <prepare|backtest|blindtest|forecast|risk|orders|report|deck|evaluate>`.
Notebooks: `notebooks/01_eda.ipynb`, `notebooks/02_results.ipynb` (thin; they import the package).

## Tests and evaluation

```powershell
.venv\Scripts\python -m pytest              # everything: 26 unit tests + 32 end-to-end tests, ~20 s
.venv\Scripts\python -m pytest -m "not e2e" # unit tests only, ~2 s
.venv\Scripts\python -m supply_pipeline evaluate   # quality gates over the real run -> reports/eval_report.md
.venv\Scripts\python -m ruff check src tests && .venv\Scripts\python -m ruff format --check src tests && .venv\Scripts\python -m mypy
```

* **Unit tests** (`tests/test_*.py`): metrics, calendar features, schema validation, store->DC aggregation, weekly build, MOQ rounding, lognormal service-level maths, risk labels, fold construction, interval calibration, model-selection tie-break.
* **End-to-end tests** (`tests/test_pipeline_e2e.py`, fixtures in `tests/conftest.py`): `tests/synthetic.py` generates a small challenge-shaped dataset with *known ground truth* and planted messiness (late-start series, discontinued line, duplicate item numbers, negative stock, partial last inventory week, a chronic stock-out and a stock-out onset, promo weeks, outlier spikes). The whole pipeline runs on it once per session; tests then check every stage's outputs - schemas, keys, monotone quantiles, MOQ multiples, flags, labels and onsets, deck/report structure - plus behaviour: the steady series' forecast tracks the true process, the discontinued line forecasts near zero and orders zero, the cold-start series falls back to the moving average, the stocked-out DC is alerted and re-ordered, and re-running downstream stages is byte-for-byte deterministic.
* **Static checks**: `ruff check`, `ruff format --check` and `mypy` (config in `pyproject.toml`; the package is fully annotated, third-party libraries are treated as untyped).
* **Continuous integration** (`.github/workflows/ci.yml`): on every push and pull request, three jobs run on Ubuntu - lint + type-check; the full test suite on Python 3.11 and 3.13; and a one-command `run` on a freshly generated synthetic dataset (the challenge CSVs are not committed), whose report, deck, order file and eval scorecard are uploaded as a build artifact. A hard quality-gate failure fails the build.
* **Evaluation suite** (`src/supply_pipeline/evaluate.py`, last stage of `run`): 39 quality gates over the produced artifacts - data reconciliation, forecast structure and accuracy, held-out interval coverage, risk metrics, order-policy invariants, report completeness. Thresholds live in `[eval]` of `config.toml`; hard gates make the run exit non-zero, soft gates warn. Output: `reports/tables/eval_scorecard.csv` and `reports/eval_report.md`. The same gates run against the synthetic dataset inside the test suite.

## Layout

```
config.toml                 every assumption and tunable (see below)
src/supply_pipeline/
  cli.py                    stage runner: prepare -> backtest -> forecast -> risk -> orders -> report -> deck
  schema.py, data.py        validation, cleaning, store -> DC aggregation, calendar completion, outlier flags
  calendar_features.py      MX holidays, quincena paydays, Semana Santa, Buen Fin, December peak
  features.py               weekly modelling table and per-series metadata / flags
  models/                   ma4, seasonal_naive, ets (statsmodels), lgbm (global quantile LightGBM)
  backtest.py               expanding-window folds, metrics, per-cluster selection, interval calibration
  blindtest.py              sealed last fold: selection without it, all models scored on it, regret per cluster
  forecast.py               forecasts at the final and risk-evaluation origins with the selected models
  distributions.py          lognormal fit to forecast quantiles (P(stock-out), service level, lost sales)
  risk.py                   Track B scorers, labels, evaluation, as-of alert list
  orders.py                 order-up-to policy, MOQ rounding, baselines, portfolio summaries
  plots.py, report.py       figures and summary.md      deck.py   python-pptx deck
  evaluate.py               quality gates -> eval_scorecard.csv / eval_report.md (fails the run on hard gates)
tests/                      unit tests, synthetic dataset generator, end-to-end + eval-gate tests
data/{raw,interim,output}   inputs (gitignored) / parquet cache / forecasts, alerts, order file
reports/{figures,tables}    everything the report and deck are built from
```

## Method in brief

**Data.** Daily sell-out is completed onto a calendar per series (gaps stay missing), outliers are flagged with a trailing robust z-score and winsorised for training only, and aggregated to ISO weeks (107 complete weeks). Store inventory is de-duplicated (three UPCs carry two item numbers), negative on-hand is clipped to zero for stock maths but counted as a stock-out signal, and rolled up to DC via the store catalog. Inventory covers 21 days; 2026-04-02 is the last day with full coverage and is the "current on-hand".

**Forecasting.** Four models under one interface: seasonal naive, 4-week moving average, per-series ETS (additive damped trend, simulated intervals) and a global LightGBM quantile model (direct multi-horizon; lags, rolling stats, lagged price/promo, calendar counts, cluster/DC/UPC categoricals; level-scaled target). Expanding-window backtest: 8 origins every 4 weeks (2025-07-21 .. 2026-02-02), 8-week horizon, quantiles 5/10/50/90/95. Metrics: WAPE, MAPE, bias, scaled pinball, 80/90% coverage. Selection per ABC x XYZ cluster on mean WAPE with a stability/simplicity tie-break. Intervals are calibrated conformal-style per model x cluster on folds 1-5 and reported on folds 6-8.

**Blind test.** The backtest's predictions are out-of-sample, but its model *selection* sees every fold. The `blindtest` stage seals the last fold (the final 8 weeks), redoes selection and calibration using only folds whose targets end before it, and scores all four models on the sealed weeks. It reports each cluster's pre-registered choice against the best model in hindsight ("regret"). No model is refitted; it reads `backtest_predictions.parquet`.

**Risk.** Alert = likely short at stores within the 7-day lead time. Scorers: days-of-cover rule, P(7-day demand > on-hand) from the calibrated forecast, Isolation Forest over cover / probability / sales-vs-forecast / stock trend. Labels: >= 25% of a DC's stores at zero; alerts on day d are scored against d+1..d+7, using the forecast available at 2026-03-09. Precision, recall, false-alarm rate, threshold sweep and lead-time-to-alert (where onsets exist) are reported.

**Orders.** Weekly order-up-to over a 14-day protection period. Demand over the period from the selected model's calibrated quantiles (lognormal fit); safety stock = max(catalog `safety_stock_days` x daily demand, stock for the ABC service-level target); order = max(0, order-up-to - projected on-hand) rounded up to MOQ. Implied service level, expected lost sales, fill rate and working capital are computed for the recommendation and for two baselines (4-week moving average, last-year-same-weeks) under the same demand distribution.

## Assumptions and trade-offs

- **Weekly grain.** Lead time is exactly 7 days and MOQ is 100 for every SKU, so weekly buckets match the ordering cycle; daily data is kept for Track B and for calendar features. Trade-off: intra-week (payday) timing is handled by weekday shares rather than modelled.
- **Future price and promo are unknown** at forecast time. Models use lagged price/promo and known calendar effects only; promo-driven weeks are under-forecast and flagged. A promo calendar is the single biggest accuracy lever.
- **Non-seasonal ETS.** 107 weeks is too short for a 52-period seasonal ETS; yearly seasonality is carried by the seasonal-naive comparator and the LightGBM lag-52 feature.
- **Stock-out label** uses store-level zeros because DC-level stock rarely hits zero. The 21-day window contains only chronic stock-outs, so lead-time-to-alert cannot be established from this file; precision/recall are indicative.
- **Projected on-hand** subtracts sell-out observed after the snapshot; receipts / in-transit orders are not in the data, so recommendations are upper bounds where an order is already on its way.
- **Working capital** is valued at recent shelf price x `cost_ratio` (default 1.0). Set `cost_ratio` to the cost share of price for a cost basis.
- **Lognormal demand** over the protection period is fitted to the p50/p90 of the (comonotonic) sum of weekly quantiles - conservative for correlated forecast errors across weeks.
- **Duplicated item numbers** are collapsed on UPC; if the two item numbers are genuinely different physical products this double-counts nothing in sales (sales are per UPC) but sums their stock.
- **No neural model.** With 328 short series, LSTM/TFT would cost a day for little expected gain; the interface allows adding one.

## Configuration (`config.toml`)

`data.as_of`, `data.last_complete_week_start`, `data.cold_start_weeks`, `data.outlier_mad_z`, `data.stockout_store_share`, `forecast.horizon_weeks / quantiles / backtest_folds / backtest_step_weeks / seed`, `risk.eval_origin / prob_threshold / contamination`, `orders.review_period_days / cost_ratio / service_level.{A,B,C}`.

## Outputs

| Path | Content |
|---|---|
| `data/output/supply_order_2026-04-02.csv` | one line per SKU x DC: on-hand, demand p50/p90, safety stock, order quantity, implied service level, expected lost sales, working capital, baselines, deltas, flags |
| `data/output/forecast_2026-03-30.csv` | selected-model 8-week quantile forecast per SKU x DC |
| `data/output/risk_alerts_2026-04-02.csv` | as-of risk scores and severity per SKU x DC |
| `reports/summary.md`, `reports/figures/*.png`, `reports/tables/*.csv` | full results |
| `reports/deck.pptx` | executive deck |
