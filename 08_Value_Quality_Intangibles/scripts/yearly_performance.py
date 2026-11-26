"""
Year-by-year performance of the final strategy as a real portfolio.

Run from the project folder, after run_signal.py for the signal (it reads whatever
that run saved in results/signals/<signal>/):
    python scripts/yearly_performance.py                     (final strategy, qvi_timely)
    python scripts/yearly_performance.py --signal qvi_intan_full

Three portfolios, value-weighted, 1963-07 to 2015-12 (the holdout is never read):
  long_short   a market-neutral fund: the cash sits in T-bills and earns rf, on top
               the fund is long decile 10 and short decile 1 (net of trading costs)
               -> monthly return = rf + net spread
  long_only    only the long leg (decile 10), net of its own trading costs
  market       the US stock market (Fama-French MKT_RF + rf), the benchmark
Prints every calendar year (1963 starts in July), then a summary: average yearly
return (compounded), volatility, Sharpe, worst year, maximum drawdown, share of
positive years, and what $100 invested in July 1963 became.
"""
import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.evaluation.alpha import load_factors  # noqa: E402

END = pd.Period("2015-12", freq="M")
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:.1f}")


def read(path):
    x = pd.read_csv(path, index_col=0)
    x.index = pd.PeriodIndex(x.index, freq="M")
    return x[x.index <= END]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", default="qvi_timely")
    args = ap.parse_args()
    m = importlib.import_module(f"src.signals.{args.signal}")
    L, S = str(m.LONG), str(m.SHORT)
    d = BASE_DIR / "results" / "signals" / args.signal
    r, c = read(d / "returns_vw.csv"), read(d / "costs_vw.csv")
    f = load_factors(BASE_DIR / "data" / "external" / "factors_monthly.csv").reindex(r.index)

    port = pd.DataFrame({
        "long_short": f["RF"] + r[f"{L}-{S} net"],
        "long_only": r[L] - c[L],
        "market": f["MKT_RF"] + f["RF"],
    }).dropna()

    yearly = (1 + port).groupby(port.index.year).prod() - 1
    print(f"{args.signal}: calendar-year returns, % (1963 = July to December)")
    print((yearly * 100).to_string())

    n_years = len(port) / 12
    wealth = (1 + port).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    rf = f["RF"].reindex(port.index)
    summary = pd.DataFrame({
        "Return per year %": (wealth.iloc[-1] ** (1 / n_years) - 1) * 100,
        "Volatility per year %": port.std() * np.sqrt(12) * 100,
        "Sharpe": port.sub(rf, axis=0).mean() / port.std() * np.sqrt(12),
        "Best year %": yearly.max() * 100,
        "Worst year %": yearly.min() * 100,
        "Max drawdown %": drawdown.min() * 100,
        "Positive years %": (yearly > 0).mean() * 100,
        "$100 became": wealth.iloc[-1] * 100,
    }).T
    print(f"\n===== Summary, {port.index[0]} to {port.index[-1]} ({n_years:.1f} years) =====")
    print(summary.to_string(float_format=lambda x: f"{x:,.1f}"))
    print(f"\nCorrelation of monthly returns: long_short vs market "
          f"{port['long_short'].corr(port['market']):+.2f}")
    print("Note: long_short assumes the short sale proceeds earn rf and no fees; "
          "long_only is a 10%-of-stocks portfolio, not diversified like the market.")


if __name__ == "__main__":
    main()
