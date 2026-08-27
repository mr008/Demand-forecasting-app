"""Generate the two thin notebooks (they import from the package; no logic lives here).

Run: python notebooks/build_notebooks.py && python -m nbconvert --execute --to notebook --inplace notebooks/*.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

SETUP = """import sys, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from supply_pipeline.config import load_config
cfg = load_config(ROOT / "config.toml")
p = cfg.paths
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 40)
%matplotlib inline"""


def eda() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    c = nb.cells
    c.append(
        nbf.v4.new_markdown_cell(
            "# 01 - Exploratory data analysis\n\nReads the parquet cache written by `python -m supply_pipeline prepare`."
        )
    )
    c.append(nbf.v4.new_code_cell(SETUP))
    c.append(
        nbf.v4.new_code_cell("""weekly = pd.read_parquet(p.interim_dir / "weekly.parquet")
series = pd.read_parquet(p.interim_dir / "series.parquet")
daily = pd.read_parquet(p.interim_dir / "daily.parquet")
inv = pd.read_parquet(p.interim_dir / "inventory_cedis_daily.parquet")
print(weekly.shape, series.shape, daily.shape, inv.shape)""")
    )
    c.append(nbf.v4.new_markdown_cell("## Coverage"))
    c.append(
        nbf.v4.new_code_cell("""display(series["cluster"].value_counts().sort_index().to_frame("series"))
display(series[["n_weeks", "mean_weekly", "on_hand_as_of"]].describe().round(1))
display(pd.read_csv(p.tables_dir / "coverage_inventory_by_date.csv", index_col=0))""")
    )
    c.append(nbf.v4.new_markdown_cell("## Cluster-level weekly sell-out"))
    c.append(
        nbf.v4.new_code_cell("""fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
for ax, (cl, g) in zip(axes.ravel(), weekly.groupby("cluster")):
    tot = g.groupby("week_start")["y"].sum()
    ax.plot(tot.index, tot.values / 1e3, color="#0E6B58"); ax.set_title(f"{cl} ({g[['upc','cedis']].drop_duplicates().shape[0]} series)"); ax.set_ylim(0)
fig.suptitle("Weekly sell-out by cluster (thousand units)"); fig.tight_layout()""")
    )
    c.append(nbf.v4.new_markdown_cell("## Promo lift and price"))
    c.append(
        nbf.v4.new_code_cell("""d = daily[daily["sell_out_pzs"].notna()].copy()
base = d[d["promo_flag"] == 0].groupby(["upc", "cedis"])["sell_out_pzs"].mean().rename("base")
promo = d[d["promo_flag"] == 1].groupby(["upc", "cedis"])["sell_out_pzs"].mean().rename("promo")
lift = pd.concat([base, promo], axis=1).dropna()
lift["lift"] = lift["promo"] / lift["base"]
print("median promo lift:", round(lift["lift"].median(), 2), " share of promo days:", round(d["promo_flag"].mean(), 3))
lift["lift"].clip(0, 5).hist(bins=40, color="#B9911E"); plt.title("Promo-day lift vs non-promo days, per series");""")
    )
    c.append(nbf.v4.new_markdown_cell("## Calendar effects"))
    c.append(
        nbf.v4.new_code_cell("""d["dow"] = d["date"].dt.dayofweek
rel = d.groupby(["upc", "cedis"])["sell_out_pzs"].transform(lambda s: s / s.mean())
print("weekday index:", d.assign(rel=rel).groupby("dow")["rel"].mean().round(3).to_dict())
for col in ["is_payday_window", "is_holiday", "is_semana_santa", "is_buen_fin", "is_december_peak"]:
    print(f"{col:20s} index = {d.assign(rel=rel).groupby(col)['rel'].mean().round(3).to_dict()}")""")
    )
    c.append(nbf.v4.new_markdown_cell("## Inventory window"))
    c.append(
        nbf.v4.new_code_cell("""print(inv["stockout_store_share"].describe(percentiles=[.5, .9, .95, .99]).round(3))
print("series with >=25% stores at zero on any day:", inv[inv["stockout_store_share"] >= 0.25].groupby(["upc", "cedis"]).ngroups)
display(series.loc[series["is_discontinued"], ["upc", "cedis", "cluster", "mean_weekly", "tail4_mean"]])""")
    )
    return nb


def results() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    c = nb.cells
    c.append(nbf.v4.new_markdown_cell("# 02 - Results walkthrough\n\nReads the tables produced by the full pipeline run."))
    c.append(nbf.v4.new_code_cell(SETUP))
    c.append(nbf.v4.new_markdown_cell("## Backtest"))
    c.append(
        nbf.v4.new_code_cell("""t = p.tables_dir
display(pd.read_csv(t / "backtest_metrics_overall.csv").round(3))
display(pd.read_csv(t / "backtest_metrics_cluster.csv").pivot(index="cluster", columns="model", values="wape").round(3))
display(pd.read_csv(t / "model_selection.csv").round(3))""")
    )
    c.append(
        nbf.v4.new_code_cell("""display(pd.read_csv(t / "backtest_metrics_holdout_calibrated_overall.csv").round(3))
display(pd.read_csv(t / "interval_calibration.csv").pivot(index=["model", "cluster"], columns="nominal", values="k").round(2))""")
    )
    c.append(nbf.v4.new_markdown_cell("## Figures"))
    c.append(
        nbf.v4.new_code_cell("""from IPython.display import Image, display as show
for name in ["model_comparison", "wape_by_horizon", "coverage_calibration", "cluster_forecast_vs_actual", "residuals", "series_examples"]:
    show(Image(filename=str(p.figures_dir / f"{name}.png")))""")
    )
    c.append(nbf.v4.new_markdown_cell("## Stock-out risk"))
    c.append(
        nbf.v4.new_code_cell("""display(pd.read_csv(t / "risk_eval_methods.csv").round(3))
display(pd.read_csv(t / "risk_threshold_sweep.csv").round(3))
display(pd.read_csv(t / "risk_lead_time.csv"))
alerts = pd.read_csv(p.output_dir / f"risk_alerts_{cfg.data.as_of}.csv")
print(alerts["severity"].value_counts())
display(alerts[alerts["severity"] == "high"].head(15).round(2))
show(Image(filename=str(p.figures_dir / "risk_window.png")))""")
    )
    c.append(nbf.v4.new_markdown_cell("## Supply order"))
    c.append(
        nbf.v4.new_code_cell("""orders = pd.read_csv(p.output_dir / f"supply_order_{cfg.data.as_of}.csv")
display(pd.read_csv(t / "order_summary_portfolio.csv").round(3).T)
display(pd.read_csv(t / "order_summary_cluster.csv").round(3))
display(orders.sort_values("working_capital", ascending=False).head(10)[["upc", "cedis", "cluster", "on_hand_projected", "demand_p50", "demand_p90", "safety_stock", "order_qty", "implied_service_level", "working_capital", "ma4_order_qty", "flags"]].round(2))
show(Image(filename=str(p.figures_dir / "orders_summary.png")))""")
    )
    return nb


if __name__ == "__main__":
    nbf.write(eda(), HERE / "01_eda.ipynb")
    nbf.write(results(), HERE / "02_results.ipynb")
    print("wrote 01_eda.ipynb, 02_results.ipynb")
