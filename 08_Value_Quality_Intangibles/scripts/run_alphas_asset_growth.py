"""
Alphas of the asset growth decile portfolios under all benchmark models.
Needs: results/asset_growth/ew_returns.csv and vw_returns.csv (run_asset_growth.py)
       data/external/factors_monthly.csv                     (src/data/factors.py)
Run from the project folder:  python scripts/run_alphas_asset_growth.py
"""
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.evaluation.alpha import (  # noqa: E402
    alpha_table, grs_test, load_factors, load_portfolio_returns, regress,
)

RES_DIR = BASE_DIR / "results" / "asset_growth"
FACTORS = BASE_DIR / "data" / "external" / "factors_monthly.csv"

FULL = ("1963-07", "2025-12")     # our portfolio sample (French factors available)
COMMON = ("1967-01", "2025-12")   # sample where ALL models are available (q-factors start 1967)
FRENCH_MODELS = ["CAPM", "FF3", "FF4", "FF5", "FF6"]
ALL_MODELS = FRENCH_MODELS + ["q4", "q5"]
SPREAD = "1-10"                   # long low growth (decile 1), short high growth (decile 10)

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:.2f}")

factors = load_factors(FACTORS)
results = {}

for weighting in ["ew", "vw"]:
    rets = load_portfolio_returns(RES_DIR / f"{weighting}_returns.csv")
    rets[SPREAD] = rets[1] - rets[10]
    label = "EQUAL-WEIGHTED" if weighting == "ew" else "VALUE-WEIGHTED"

    # 1. The long-short spread under every model, on both samples
    rows = []
    for sample_name, (start, end), models in [
        ("1963-2025", FULL, FRENCH_MODELS),
        ("1967-2025", COMMON, ALL_MODELS),
    ]:
        for model in models:
            r = regress(rets[SPREAD], factors, model, long_short=True, start=start, end=end)
            rows.append({"sample": sample_name, "model": model, "alpha %": r["alpha %"],
                         "t(alpha)": r["t(alpha)"], "R2": r["R2"], "months": r["months"]})
    spread_table = pd.DataFrame(rows).set_index(["sample", "model"])
    print(f"\n===== {label}: spread {SPREAD} (Newey-West t, 6 lags) =====")
    print(spread_table.to_string())

    # 2. Alphas of every decile, common sample, all models
    deciles = rets[list(range(1, 11))]
    table = alpha_table(rets[list(range(1, 11)) + [SPREAD]], factors, ALL_MODELS,
                        long_short_cols=[SPREAD], start=COMMON[0], end=COMMON[1])
    alphas_only = table.xs("alpha %", axis=1, level=1)
    tstats_only = table.xs("t", axis=1, level=1)
    print(f"\n----- {label}: alpha (% per month) of each decile, 1967-2025 -----")
    print(alphas_only.to_string())
    print(f"\n----- {label}: t-stats -----")
    print(tstats_only.to_string())

    # 3. GRS: are the 10 decile alphas jointly zero?
    grs_rows = {m: grs_test(deciles, factors, m, start=COMMON[0], end=COMMON[1]) for m in ALL_MODELS}
    grs_table = pd.DataFrame(grs_rows).T
    print(f"\n----- {label}: GRS test on the 10 deciles, 1967-2025 -----")
    print(grs_table.to_string(float_format=lambda x: f"{x:.4f}"))

    spread_table.to_csv(RES_DIR / f"alpha_spread_{weighting}.csv")
    table.to_csv(RES_DIR / f"alpha_deciles_{weighting}.csv")
    grs_table.to_csv(RES_DIR / f"grs_{weighting}.csv")

print(f"\nResults saved in: {RES_DIR}")
