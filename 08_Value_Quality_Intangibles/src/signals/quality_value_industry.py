"""
THE STRATEGY (docs/STRATEGY.md): quality + value, both measured against the stock's
own industry.

Signal, one value per stock and formation year t (sorted at the end of June t):
    average of two ranks computed within (formation year, FF12 industry):
        value_industry   (four valuation ratios ranked within the industry;
                          banks: book/price and earnings/price only)
        cash_op          (cash-based operating profitability)
    Both parts are required; industries need at least 10 stocks that year.
High = cheap AND profitable compared with the industry's other firms.
Always 50/50, no other neutralisation.
"""
import pandas as pd

from src.signals._industry import industry_june, industry_rank
from src.signals._yearly import KEY, to_rows
from src.signals.cash_op import COMPUSTAT_COLS as _COP_COLS
from src.signals.cash_op import cash_op
from src.signals.value_industry import COMPUSTAT_COLS as _VI_COLS
from src.signals.value_industry import value_industry_year

COMPUSTAT_COLS = sorted(set(_VI_COLS) | set(_COP_COLS))
CRSP_COLS = ["me_firm"]
REBALANCE = "annual"
LONG, SHORT = 10, 1


def _per_year(panel: pd.DataFrame, values: pd.Series) -> pd.Series:
    """Row-level annual signal -> one value per (PERMNO, formation_year)."""
    s = pd.Series(values.to_numpy(), index=pd.MultiIndex.from_arrays(
        [panel["PERMNO"], panel["formation_year"]], names=KEY)).dropna()
    return s[~s.index.duplicated()]


def quality_value_industry(panel: pd.DataFrame) -> pd.Series:
    return quality_value_industry_scores(panel, "ff12")


def quality_value_industry_scores(panel: pd.DataFrame, scheme: str) -> pd.Series:
    """scheme: "ff12" (main) or "ff49" (robustness, quality_value_industry_ff49.py)."""
    return to_rows(panel, quality_value_industry_year(panel, scheme))


def quality_value_industry_year(panel: pd.DataFrame, scheme: str = "ff12", intangible: bool = False) -> pd.Series:
    """One score per (PERMNO, formation_year). scheme, intangible: see value_industry_year."""
    parts = pd.concat({
        "value": value_industry_year(panel, scheme, intangible),
        "quality": _per_year(panel, cash_op(panel)),
    }, axis=1).dropna()
    ind = industry_june(panel, scheme).reindex(parts.index)
    parts, ind = parts[ind.notna()], ind[ind.notna()]
    ranks = industry_rank(parts, ind, min_stocks=10)
    return ranks.mean(axis=1, skipna=False).dropna()
