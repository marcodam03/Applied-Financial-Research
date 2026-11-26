"""
Momentum signal (Jegadeesh and Titman 1993), as in Ken French's "Prior 12-2"
portfolios: the cumulative return over months m-11 to m-1, skipping month m.

Timing: the value in the row of month m is known at the end of month m, the
portfolio is formed then and held in month m+1. So the signal uses returns from
12 months to 2 months before the holding month, skipping the most recent one
(short-term reversal).

All 11 monthly returns must exist, in 11 consecutive months (no gap in the
stock's history); otherwise the signal is missing.
"""
import numpy as np
import pandas as pd

COMPUSTAT_COLS = []
REBALANCE = "monthly"
LONG, SHORT = 10, 1  # past winners beat past losers
FRENCH_BENCHMARK = "10_Portfolios_Prior_12_2_CSV.zip"

WINDOW = 11  # months m-11 to m-1


def momentum(panel: pd.DataFrame) -> pd.Series:
    """panel must be sorted by PERMNO and MthCalDt (load_panel does it)."""
    stock = panel["PERMNO"]
    month_id = panel["MthCalDt"].dt.year * 12 + panel["MthCalDt"].dt.month

    with np.errstate(divide="ignore"):  # a -100% return gives log(0) = -inf, i.e. momentum -100%
        log_ret = np.log1p(panel["MthRet"])
    past = log_ret.groupby(stock).shift(1)  # return of month m-1 in the row of month m
    total = past.groupby(stock).rolling(WINDOW, min_periods=WINDOW).sum().droplevel(0)

    # The row 11 rows back must be exactly 11 months back (no gap in between)
    consecutive = (month_id - month_id.groupby(stock).shift(WINDOW)) == WINDOW
    return np.expm1(total).where(consecutive)
