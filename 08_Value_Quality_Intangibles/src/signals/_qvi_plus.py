"""
Builder of the annual value + quality score (qvi_intan_full.py); its quality part is
also used by the final strategy (_qvi_ext.py). One score per (PERMNO,
formation_year t): accounting of the fiscal year ending in t-1, market equity of
December t-1, sorted at the end of June t.

Pieces (each ranked within (formation year, FF12 industry), industries >= 10 stocks):
  value      value_industry_year(intangible=True): B/M with intangible-adjusted book
             (be - goodwill + knowledge + organisation capital), E/P, CF/P, EBITDA/EV
             (financials: B/M and E/P only). Eisfeldt, Kim and Papanikolaou (2022).
  quality    intangible-adjusted cash profitability:
               (sale - cogs - xsga + xrd + 0.30 x SG&A without R&D
                - change in receivables - change in inventory + change in payables)
               / (assets + knowledge capital + organisation capital)
             = cash-based operating profitability of Ball et al. (2016) with the
             organisation-capital investment added back and the intangible capital in
             the assets (Peters and Taylor 2017). A missing item or lag counts as no
             change.
  score      average of the two ranks, ranked again within the industry.
"""
import numpy as np
import pandas as pd

from src.signals._industry import industry_june, industry_rank
from src.signals._intangibles import intangible_table
from src.signals._quality import restrict, universe_june
from src.signals._yearly import yearly
from src.signals.value_industry import value_industry_year

CASH_ITEMS = ["sale", "cogs", "xsga", "xrd", "at", "rect", "invt", "ap",
              "rect_lag", "invt_lag", "ap_lag"]
COMPUSTAT_COLS = sorted(set(CASH_ITEMS) | {
    "be", "ib", "dp", "oibdp", "dltt", "dlc", "pstk", "che", "gdwl"})
CRSP_COLS = ["me_firm"]
MIN_STOCKS = 10


def _clean(s: pd.Series) -> pd.Series:
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s[~s.index.duplicated()]


def intangible_cash_profit(panel: pd.DataFrame) -> pd.Series:
    a = yearly(panel, CASH_ITEMS)
    t = intangible_table(panel).reindex(a.index)
    op = (a["sale"] - a["cogs"] - a["xsga"].fillna(0) + a["xrd"].fillna(0)
          + t["org_inv"].fillna(0))
    change = lambda c: (a[c] - a[f"{c}_lag"]).fillna(0)  # noqa: E731
    cop = op - change("rect") - change("invt") + change("ap")
    capital = a["at"] + t["k_know"].fillna(0) + t["k_org"].fillna(0)
    s = cop / capital.where(a["at"] > 0)
    return restrict(_clean(s), universe_june(panel))


def qvi_plus_year(panel: pd.DataFrame, scheme: str = "ff12") -> pd.Series:
    parts = pd.concat({
        "value": _clean(value_industry_year(panel, scheme, intangible=True)),
        "quality": intangible_cash_profit(panel),
    }, axis=1).dropna()
    ind = industry_june(panel, scheme).reindex(parts.index)
    keep = ind.notna()
    parts, ind = parts[keep], ind[keep]
    ranks = industry_rank(parts, ind, MIN_STOCKS)
    score = ranks.mean(axis=1, skipna=False).dropna()
    final = industry_rank(score.to_frame("s"), ind.reindex(score.index), MIN_STOCKS)["s"]
    return final.dropna()
