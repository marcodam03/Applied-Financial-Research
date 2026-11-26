"""
Value composite (Lakonishok, Shleifer and Vishny 1994; Asness, Moskowitz and
Pedersen 2013 use a similar mix): book value alone has become a weaker measure of
cheapness (intangibles, buybacks), so cheapness is measured four ways and averaged.

From the fiscal year ending in calendar year t-1, with firm market equity ME of
December t-1 (Fama-French timing), sorted at the end of June t:
    book-to-market        be / ME                        (be > 0)
    earnings yield        ib / ME
    cash-flow yield       (ib + dp) / ME
    EBITDA / EV           oibdp / (ME + dltt + dlc + pstk - che)   (EV > 0)
Signal = average of the four yearly percentile ranks, at least 3 available.
High = cheap. Ranked within the stocks with a positive June market cap.
Negative earnings or cash flows simply get low ranks (expensive on that measure).
"""
import pandas as pd

from src.signals._quality import me_december, rank_mean, restrict, universe_june
from src.signals._yearly import to_rows, yearly

COMPUSTAT_COLS = ["be", "ib", "dp", "oibdp", "dltt", "dlc", "pstk", "che"]
CRSP_COLS = ["me_firm"]
REBALANCE = "annual"
LONG, SHORT = 10, 1


def value_composite(panel: pd.DataFrame) -> pd.Series:
    a = yearly(panel, COMPUSTAT_COLS)
    me = me_december(panel).reindex(a.index)
    ev = me + a["dltt"].fillna(0) + a["dlc"].fillna(0) + a["pstk"].fillna(0) - a["che"].fillna(0)
    parts = pd.DataFrame({
        "bm": (a["be"] / me).where(a["be"] > 0),
        "ep": a["ib"] / me,
        "cfp": (a["ib"] + a["dp"].fillna(0)) / me,
        "ebitda_ev": a["oibdp"] / ev.where(ev > 0),
    })
    parts = restrict(parts[me.notna()], universe_june(panel))
    return to_rows(panel, rank_mean(parts, min_count=3))
