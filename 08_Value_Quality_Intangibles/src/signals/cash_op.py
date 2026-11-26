"""
Cash-based operating profitability (Ball, Gerakos, Linnainmaa and Nikolaev 2016):
operating profits minus the part not yet received in cash (accruals). They find it
subsumes both operating profitability and accruals as a predictor of returns.

From the fiscal year ending in calendar year t-1 (sorted at the end of June t):
    operating profit  = sale - cogs - xsga + xrd      (R&D added back, as in the paper;
                                                       missing xsga or xrd count as 0)
    cash-based OP     = operating profit
                        - change in receivables (rect)
                        - change in inventory (invt)
                        + change in accounts payable (ap)
    signal            = cash-based OP / total assets (at)
The paper also adjusts for prepaid expenses and accrued expenses, which our
Compustat file does not have; the three items used are the main ones.
High = profitable in cash. Ranked within the stocks with a positive June market cap.
"""
import numpy as np
import pandas as pd

from src.signals._quality import restrict, universe_june
from src.signals._yearly import to_rows, yearly

ITEMS = ["sale", "cogs", "xsga", "xrd", "at", "rect", "invt", "ap"]
COMPUSTAT_COLS = ITEMS + ["rect_lag", "invt_lag", "ap_lag"]
REBALANCE = "annual"
LONG, SHORT = 10, 1


def cash_op(panel: pd.DataFrame) -> pd.Series:
    a = yearly(panel, COMPUSTAT_COLS)
    op = a["sale"] - a["cogs"] - a["xsga"].fillna(0) + a["xrd"].fillna(0)
    d = lambda c: (a[c] - a[f"{c}_lag"]).fillna(0)  # noqa: E731  missing item or lag = no change
    cop = op - d("rect") - d("invt") + d("ap")
    s = (cop / a["at"].where(a["at"] > 0)).replace([np.inf, -np.inf], np.nan).dropna()
    return to_rows(panel, restrict(s, universe_june(panel)))
