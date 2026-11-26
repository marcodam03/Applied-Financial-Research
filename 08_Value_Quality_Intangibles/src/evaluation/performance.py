"""
Performance of a long-short spread, before and after trading costs.

t-stat:  mean / (std / sqrt(months)), as in the slides (no Newey-West here;
         the alphas in alpha.py have Newey-West t-stats).
Break-even: the one-way cost per trade (bps) that would wipe out the average
         spread; useful whatever the cost assumption is.
Sharpe:  annualised, monthly mean / monthly std * sqrt(12). A long-short spread
         costs nothing to set up, so no risk-free rate is subtracted.
Net:     the spread after the trading costs of src/portfolio/costs.py (half
         spread + price impact). "Net months" is the number of months with a known
         cost, "Cost %" the average monthly cost of both legs over those months.
"""
from typing import Dict

import numpy as np
import pandas as pd


def _mean_t_sharpe(r: pd.Series):
    r = r.dropna()
    mean, std = r.mean(), r.std()
    return mean, mean / (std / np.sqrt(len(r))), mean / std * np.sqrt(12), len(r)


def breakeven_bps(gross: pd.Series, turnover_long: pd.Series, turnover_short: pd.Series) -> float:
    """One-way trading cost (basis points) at which the average net spread is zero:
    average gross spread / (2 x average turnover of both legs), i.e. if every trade
    paid the same cost. Independent of the spreads data."""
    traded = 2 * (turnover_long.mean() + turnover_short.mean())
    return gross.mean() / traded * 10_000 if traded > 0 else float("nan")


def spread_stats(
    gross: pd.Series, net: pd.Series, turnover_long: pd.Series, turnover_short: pd.Series
) -> Dict[str, float]:
    """Average (% per month), t-stat and Sharpe ratio of the spread before and
    after costs, the average monthly cost (%) and the average monthly one-way
    turnover (%) of each leg."""
    g_mean, g_t, g_sharpe, months = _mean_t_sharpe(gross)
    n_mean, n_t, n_sharpe, net_months = _mean_t_sharpe(net)
    return {
        "Avg %": g_mean * 100,
        "t": g_t,
        "Sharpe": g_sharpe,
        "Net avg %": n_mean * 100,
        "Net t": n_t,
        "Net Sharpe": n_sharpe,
        "Cost %": (gross - net).mean() * 100,
        "Break-even bps": breakeven_bps(gross, turnover_long, turnover_short),
        "Turnover % long": turnover_long.mean() * 100,
        "Turnover % short": turnover_short.mean() * 100,
        "Months": months,
        "Net months": net_months,
    }
