"""
Laggard with a one-month gap: robustness test of laggard.py for bid-ask bounce.

Within an industry, the leaders' return is the same for every follower, so the
laggard signal ranks followers mostly on minus their own return of month t. Part
of that reversal is bid-ask bounce: a stock whose month-end price is at the bid
looks like it fell in t and rose in t+1, a return nobody can earn.

Here the row of month t holds the laggard signal of month t-1 (same stock, the
month just before), so the portfolio formed at the end of t is held in t+1: the
signal of t-1 is held in t+1, one month later than in laggard.py. The bounce is
gone after a month. If the spread survives the gap, the effect is not just bounce;
if it disappears, it may be bounce or a real effect that fades within a month
(Hou 2007 finds the lead-lag effect within weeks), and only daily data can tell
the two apart.
"""
import pandas as pd

from src.signals.laggard import laggard

COMPUSTAT_COLS = []
REBALANCE = "monthly"
LONG, SHORT = 10, 1


def laggard_skip(panel: pd.DataFrame) -> pd.Series:
    """panel must be sorted by PERMNO and MthCalDt (load_panel does it)."""
    stock = panel["PERMNO"]
    mid = panel["MthCalDt"].dt.year * 12 + panel["MthCalDt"].dt.month
    previous = laggard(panel).groupby(stock).shift(1)
    return previous.where(mid - mid.groupby(stock).shift(1) == 1)  # t-1 must be the month before
