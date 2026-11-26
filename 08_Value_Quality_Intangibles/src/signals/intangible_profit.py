"""
Intangible-adjusted profitability. Ingredient of intangible_qv (also tested on
its own).

Idea: firms that invest in R&D and organisation capital look unprofitable,
because accounting subtracts that investment from earnings. Adding it back (and
adding the capital to assets) measures the profitability of the business itself.
Related: gross profitability (Novy-Marx 2013), cash-based operating profitability
(Ball, Gerakos, Linnainmaa and Nikolaev 2016), intangibles as in Peters and Taylor
(2017). The q5 ROE factor uses reported earnings, so it penalises these firms.

For formation year t (held July t - June t+1), fiscal year ending in t-1:
  signal = (sale - cogs - xsga + R&D + 0.30 x SG&A without R&D)
           / (assets + knowledge capital + organisation capital)
Long high, short low.
"""
import pandas as pd

from src.signals._intangibles import COMPUSTAT_COLS, CRSP_COLS, intangible_table  # noqa: F401
from src.signals._yearly import to_rows

REBALANCE = "annual"
LONG, SHORT = 10, 1


def intangible_profit_yearly(panel: pd.DataFrame) -> pd.Series:
    t = intangible_table(panel)
    profit = t["sale"] - t["cogs"] - t["xsga"].fillna(0) + t["rd"] + t["org_inv"]
    capital = t["at"] + t["k_know"] + t["k_org"]
    return (profit / capital).where(t["at"] > 0)


def intangible_profit(panel: pd.DataFrame) -> pd.Series:
    return to_rows(panel, intangible_profit_yearly(panel))
