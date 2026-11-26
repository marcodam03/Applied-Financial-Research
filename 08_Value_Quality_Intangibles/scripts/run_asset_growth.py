"""
Asset growth decile portfolios (Cooper, Gulen and Schill 2008), used to check
that our portfolio engine reproduces the professor's numbers.
Also the regression test of the pipeline: rerun it after any change to clean.py.
Run from the project folder:  python scripts/run_asset_growth.py
"""
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))  # lets Python find the "src" folder

from src.data.panel import load_panel  # noqa: E402
from src.portfolio.sorts import sort_portfolios, summarize  # noqa: E402
from src.signals.asset_growth import COMPUSTAT_COLS, REBALANCE, asset_growth  # noqa: E402

OUT_DIR = BASE_DIR / "results" / "asset_growth"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START, END = "1963-07", "2025-12"  # professor's sample

# Professor's numbers (slides 18, 19, 22 and 27), in %
PROF_JUL63_EW = [-1.34, -0.91, -1.05, -1.41, -0.18, -0.01, -0.31, -1.35, -0.24, -0.62]
PROF_JUL63_VW = [-2.47, -0.30, -1.11, -0.70, 1.34, 0.36, 0.80, -0.34, 0.56, -0.47]
PROF_JUL63_N = [57, 58, 58, 58, 58, 58, 58, 59, 58, 59]
PROF_AVG_EW = [1.50, 1.46, 1.37, 1.27, 1.26, 1.23, 1.23, 1.17, 1.11, 0.66, 0.85]
PROF_AVG_VW = [1.14, 1.14, 1.08, 0.97, 0.98, 0.95, 0.97, 1.00, 0.99, 0.88, 0.26]

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:.2f}")

# 1. Load only the columns we need and compute the signal
print("Loading panel...")
panel = load_panel(
    crsp_cols=["PERMNO", "MthCalDt", "formation_year", "PrimaryExch", "MthRet", "mkt_cap"],
    comp_cols=COMPUSTAT_COLS,
)
panel["asset_growth"] = asset_growth(panel)

# 2. Build the portfolios
print("Building asset growth deciles (annual rebalancing, NYSE breakpoints)...")
res = sort_portfolios(panel, signal="asset_growth", n_portfolios=10, rebalance=REBALANCE)

ew = res["ew"].loc[START:END]
vw = res["vw"].loc[START:END]
n_stocks = res["n_stocks"].loc[START:END]
print(f"Sample: {ew.index.min()} to {ew.index.max()} ({len(ew)} months)")

# 3. Check the first month against slides 18, 19 and 22
first = pd.Period("1963-07", "M")
check = pd.DataFrame(
    {
        "EW ours": ew.loc[first] * 100, "EW prof": PROF_JUL63_EW,
        "VW ours": vw.loc[first] * 100, "VW prof": PROF_JUL63_VW,
        "N ours": n_stocks.loc[first], "N prof": PROF_JUL63_N,
    }
)
print("\n===== July 1963, returns in % (slides 18, 19, 22) =====")
print(check.to_string())

# 4. Full-sample averages against slide 27 (spread = decile 1 minus decile 10)
sum_ew = summarize(ew, long_port=1, short_port=10)
sum_vw = summarize(vw, long_port=1, short_port=10)
for name, table, prof in [("EQUAL", sum_ew, PROF_AVG_EW), ("VALUE", sum_vw, PROF_AVG_VW)]:
    table.loc["Prof avg %"] = prof
    print(f"\n===== {name}-WEIGHTED, full sample (slide 27) =====")
    print(table.to_string())

# 5. Save everything for later steps (factor regressions, charts)
ew.to_csv(OUT_DIR / "ew_returns.csv")
vw.to_csv(OUT_DIR / "vw_returns.csv")
n_stocks.to_csv(OUT_DIR / "n_stocks.csv")
res["breakpoints"].to_csv(OUT_DIR / "breakpoints.csv")
sum_ew.to_csv(OUT_DIR / "summary_ew.csv")
sum_vw.to_csv(OUT_DIR / "summary_vw.csv")
print(f"\nResults saved in: {OUT_DIR}")