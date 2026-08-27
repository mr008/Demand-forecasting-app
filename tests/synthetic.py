"""Synthetic challenge dataset with known ground truth, in the exact raw-file format.

The generator plants the messiness the pipeline must handle:

* eight UPCs across three DCs with different ABC/XYZ behaviour and noise levels,
* one UPC absent from one DC, one late-starting (cold-start) series, one discontinued line,
* one UPC carried under two ``prime_item_nbr`` (catalog + inventory duplicates),
* negative store on-hand rows, an inventory file whose last week covers 2 UPCs only,
* one chronic DC-level stock-out and one stock-out *onset* inside the window,
* promo weeks with a known lift, payday bumps, weekday pattern, yearly seasonality, outlier spikes.

``truth`` holds the deterministic daily mean (no promo, no noise) so forecasts can be
checked against what the process actually was, not just against noisy actuals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
START = pd.Timestamp("2024-03-18")
END = pd.Timestamp("2026-04-10")
INV_START = pd.Timestamp("2026-03-20")
INV_END = pd.Timestamp("2026-04-09")
LAST_FULL_INV_DAY = pd.Timestamp("2026-04-02")
ONSET_DAY = pd.Timestamp("2026-03-27")
DISCONTINUE_FROM = pd.Timestamp("2025-11-03")
CEDIS_MULT = {"X1": 1.0, "X2": 0.6, "X3": 0.35}
STORES_PER_CEDIS = 10
DOW_EFFECT = np.array([0.9, 0.9, 0.95, 1.0, 1.1, 1.25, 0.9])
PROMO_LIFT = 1.5
PROMO_PRICE = 0.8
PROMO_WEEK_PROB = 0.12


@dataclass
class SeriesSpec:
    upc: int
    level: float
    abc: str
    xyz: str
    sigma: float
    safety_stock_days: int
    moq: int = 100
    start: pd.Timestamp = START
    discontinued: bool = False
    dual_prime: bool = False
    skip_cedis: tuple[str, ...] = ()
    stockout_cedis: str | None = None  # chronic: half the stores empty all window
    onset_cedis: str | None = None  # stock-out starts on ONSET_DAY
    base_price: float = 25.0

    @property
    def cluster(self) -> str:
        return f"{self.abc}-{self.xyz}"


SPECS: list[SeriesSpec] = [
    SeriesSpec(750000000001, 3000, "A", "X", 0.08, 7, base_price=22.0),
    SeriesSpec(750000000002, 2000, "A", "Y", 0.20, 7, dual_prime=True, base_price=30.0),
    SeriesSpec(750000000003, 1500, "A", "Z", 0.45, 14, stockout_cedis="X1", base_price=18.0),
    SeriesSpec(750000000004, 400, "B", "Y", 0.20, 7, onset_cedis="X2", base_price=40.0),
    SeriesSpec(750000000005, 300, "B", "Z", 0.45, 14, skip_cedis=("X3",), base_price=35.0),
    SeriesSpec(750000000006, 120, "C", "Z", 0.45, 3, discontinued=True, base_price=45.0),
    SeriesSpec(750000000007, 800, "A", "Y", 0.20, 7, start=pd.Timestamp("2025-11-03"), base_price=27.0),  # cold start
    SeriesSpec(750000000008, 250, "B", "Y", 0.20, 3, moq=50, base_price=50.0),
]
STOCKOUT_SPEC = SPECS[2]
ONSET_SPEC = SPECS[3]
DISCONTINUED_SPEC = SPECS[5]
COLD_START_SPEC = SPECS[6]
STEADY_SPEC = SPECS[0]
DUAL_PRIME_SPEC = SPECS[1]


@dataclass
class SyntheticDataset:
    root: Path
    config_path: Path
    truth: pd.DataFrame  # date, upc, cedis, true_mean, promo_flag
    specs: list[SeriesSpec] = field(default_factory=lambda: list(SPECS))

    @property
    def n_series(self) -> int:
        return sum(len(CEDIS_MULT) - len(s.skip_cedis) for s in self.specs)


def _prime(spec: SeriesSpec, second: bool = False) -> int:
    return 100000000 + (spec.upc % 1000) + (100 if second else 0)


TRUTH_END = END + pd.Timedelta(days=70)  # truth extends over the forecast horizon


def _daily_series(spec: SeriesSpec, cedis: str, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range(spec.start, TRUTH_END, freq="D")
    t = (dates - START).days.to_numpy()
    season = 1 + 0.2 * np.sin(2 * np.pi * t / 364)
    dow = DOW_EFFECT[dates.dayofweek]
    is_payday = (dates.day == 15) | dates.is_month_end
    payday = 1 + 0.15 * (is_payday | np.roll(is_payday, 1))
    base = spec.level * CEDIS_MULT[cedis] * season * dow * payday
    if spec.discontinued:
        base = base * np.exp(-np.maximum(0, (dates - DISCONTINUE_FROM).days.to_numpy()) / 30)
    week_idx = (dates - START).days.to_numpy() // 7
    promo_weeks = rng.random(week_idx.max() + 1) < PROMO_WEEK_PROB
    promo = promo_weeks[week_idx].astype(int)
    noise = rng.lognormal(-(spec.sigma**2) / 2, spec.sigma, len(dates))
    sales = base * np.where(promo == 1, PROMO_LIFT, 1.0) * noise
    for i in rng.choice(len(dates), size=3, replace=False):
        sales[i] *= 8.0
    price = spec.base_price * np.where(promo == 1, PROMO_PRICE, 1.0) * (1 + rng.normal(0, 0.01, len(dates)))
    sell = pd.DataFrame(
        {
            "date": dates,
            "sell_out_pzs": np.round(sales, 3),
            "upc": spec.upc,
            "cedis": cedis,
            "final_price": np.round(price, 2),
            "promo_flag": promo,
        }
    )
    truth = pd.DataFrame({"date": dates, "upc": spec.upc, "cedis": cedis, "true_mean": base, "promo_flag": promo})
    sell = sell[sell["date"] <= END].reset_index(drop=True)  # observations stop at the data end; truth continues
    return sell, truth


def _stores() -> pd.DataFrame:
    rows = []
    n = 1001
    for cedis in CEDIS_MULT:
        for _ in range(STORES_PER_CEDIS):
            rows.append({"store_nbr": n, "store_name": f"STORE {n}", "cedis": cedis})
            n += 1
    df = pd.DataFrame(rows)
    df.loc[df.index[-1], "store_name"] = np.nan  # one null name, like the real file
    return df


def _inventory(specs: list[SeriesSpec], stores: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.date_range(INV_START, INV_END, freq="D")
    rows = []
    for spec in specs:
        for st in stores.itertuples(index=False):
            if st.cedis in spec.skip_cedis:
                continue
            daily_store = spec.level * CEDIS_MULT[st.cedis] / STORES_PER_CEDIS
            store_pos = (st.store_nbr - 1001) % STORES_PER_CEDIS
            for d in dates:
                if d > LAST_FULL_INV_DAY and spec.upc not in (SPECS[0].upc, SPECS[1].upc):
                    continue  # last week of the file only covers two UPCs
                oh = daily_store * 20 * rng.lognormal(0, 0.3)
                if spec.stockout_cedis == st.cedis and store_pos < 5:
                    oh = 0.0
                if spec.onset_cedis == st.cedis and store_pos < 4 and d >= ONSET_DAY:
                    oh = 0.0
                if rng.random() < 0.03:
                    oh = -float(rng.integers(1, 20))
                rows.append(
                    {
                        "date": d,
                        "prime_item_nbr": _prime(spec),
                        "upc": spec.upc,
                        "store_nbr": st.store_nbr,
                        "store_name": st.store_name,
                        "on_hand_qty": float(np.round(oh)),
                    }
                )
                if spec.dual_prime and store_pos % 3 == 0:
                    rows.append(
                        {
                            "date": d,
                            "prime_item_nbr": _prime(spec, True),
                            "upc": spec.upc,
                            "store_nbr": st.store_nbr,
                            "store_name": st.store_name,
                            "on_hand_qty": 0.0,
                        }
                    )
    return pd.DataFrame(rows)


def _upc_catalog(specs: list[SeriesSpec], sell: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
    total = sell["sell_out_pzs"].sum()
    rows = []
    for spec in specs:
        s = sell[sell["upc"] == spec.upc]
        i = inv[inv["upc"] == spec.upc]
        daily = s.groupby("date")["sell_out_pzs"].sum()
        row = {
            "prime_item_nbr": _prime(spec),
            "upc": spec.upc,
            "total_sales": s["sell_out_pzs"].sum(),
            "avg_daily_sales": daily.mean(),
            "median_daily_sales": daily.median(),
            "std_daily_sales": daily.std(),
            "active_cedis": s["cedis"].nunique(),
            "sales_days": daily.shape[0],
            "cv_demand": daily.std() / daily.mean(),
            "avg_inventory": i["on_hand_qty"].mean(),
            "max_inventory": i["on_hand_qty"].max(),
            "min_inventory": i["on_hand_qty"].min(),
            "active_stores": i["store_nbr"].nunique(),
            "sales_share": s["sell_out_pzs"].sum() / total,
            "cumulative_share": np.nan,
            "abc_class": spec.abc,
            "xyz_class": spec.xyz,
            "lead_time_days": 7,
            "moq": spec.moq,
            "safety_stock_days": spec.safety_stock_days,
        }
        rows.append(row)
        if spec.dual_prime:
            rows.append({**row, "prime_item_nbr": _prime(spec, True)})
    df = pd.DataFrame(rows).sort_values("sales_share", ascending=False)
    df["cumulative_share"] = df["sales_share"].cumsum()
    return df


def _write_config(root: Path, backtest_folds: int) -> Path:
    text = (REPO_ROOT / "config.toml").read_text(encoding="utf-8")
    overrides = {
        "backtest_folds": str(backtest_folds),
        # Synthetic Z-class noise is deliberately heavy; loosen accuracy-style gates, keep structural ones.
        "max_wape_selected": "0.60",
        "min_rel_improvement_vs_naive": "-0.50",
        "max_abs_bias": "0.30",
        "coverage_90_range": "[0.70, 1.00]",
        "coverage_80_range": "[0.55, 1.00]",
        "min_risk_recall": "0.30",
        "max_prob_false_alarm_rate": "0.50",
        "min_share_lines_at_target_sl": "0.70",
    }
    for key, val in overrides.items():
        text, n = re.subn(rf"^{key}\s*=.*$", f"{key} = {val}", text, flags=re.M)
        assert n == 1, key
    path = root / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def make_dataset(root: Path, seed: int = 7, backtest_folds: int = 3) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    sells, truths = [], []
    for spec in SPECS:
        for cedis in CEDIS_MULT:
            if cedis in spec.skip_cedis:
                continue
            s, t = _daily_series(spec, cedis, rng)
            sells.append(s)
            truths.append(t)
    sell = pd.concat(sells, ignore_index=True)
    truth = pd.concat(truths, ignore_index=True)
    stores = _stores()
    inv = _inventory(SPECS, stores, rng)
    catalog = _upc_catalog(SPECS, sell, inv)

    sell.to_csv(raw / "challenge_daily_sell_out_pricing.csv", index=False, date_format="%Y-%m-%d")
    inv.to_csv(raw / "challenge_inventory.csv", index=False, date_format="%Y-%m-%d")
    stores.to_csv(raw / "challenge_store_catalog.csv", index=False)
    catalog.to_csv(raw / "challenge_upc_catalog.csv", index=False)
    config_path = _write_config(root, backtest_folds)
    return SyntheticDataset(root=root, config_path=config_path, truth=truth)


if __name__ == "__main__":  # python -m tests.synthetic <output_dir>  (used by CI to exercise the one-command run)
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build") / "synthetic"
    dataset = make_dataset(target)
    print(dataset.config_path)
