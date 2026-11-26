"""
Robustness of the core strategy: quality_value_industry with the 49 Fama-French
industries instead of 12.

Everything else is identical: the same four valuation ratios (banks and other
financials: book/price and earnings/price only), the same cash profitability, ranks
within (formation year, industry), industries with at least 10 stocks that year,
50/50 average, long decile 10, short decile 1, annual rebalancing at the end of June.
Finer industries compare a bank with banks instead of with every financial firm,
but some industry-years have fewer than 10 stocks and drop out.
"""
import pandas as pd

from src.signals.quality_value_industry import COMPUSTAT_COLS, CRSP_COLS, LONG, REBALANCE, SHORT  # noqa: F401
from src.signals.quality_value_industry import quality_value_industry_scores


def quality_value_industry_ff49(panel: pd.DataFrame) -> pd.Series:
    return quality_value_industry_scores(panel, "ff49")
