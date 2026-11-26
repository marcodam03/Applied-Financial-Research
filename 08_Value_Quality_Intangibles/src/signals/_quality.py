"""
Helpers for the annual value and quality scores (universe, December market equity, ranks).

- universe_june(panel):  (PERMNO, formation_year) pairs with a positive market cap
                         at the end of June t, i.e. the stocks the sort can use.
                         Signals are ranked within this universe, like the sort itself.
- me_december(panel):    firm market equity in December t-1, in $ millions
                         (same units as Compustat), for formation year t.
- rank_mean(frame, min_count): average of the columns' yearly percentile ranks,
                         requiring at least min_count non-missing columns.
"""
import numpy as np
import pandas as pd

from src.signals._yearly import KEY, december, june, year_rank


def universe_june(panel: pd.DataFrame) -> pd.Index:
    cap = june(panel, panel["mkt_cap"])
    cap = cap[~cap.index.duplicated()]
    return cap[cap > 0].index


def me_december(panel: pd.DataFrame) -> pd.Series:
    me = december(panel, "me_firm") / 1000  # CRSP thousands -> millions
    me = me[~me.index.duplicated()]
    return me.where(me > 0)


def rank_mean(frame: pd.DataFrame, min_count: int) -> pd.Series:
    frame = frame.replace([np.inf, -np.inf], np.nan)
    ranks = frame.apply(year_rank)
    return ranks.mean(axis=1).where(ranks.notna().sum(axis=1) >= min_count)


def restrict(s: pd.Series, universe: pd.Index) -> pd.Series:
    s = s[~s.index.duplicated()]
    return s.reindex(s.index.intersection(universe))


__all__ = ["KEY", "universe_june", "me_december", "rank_mean", "restrict"]
