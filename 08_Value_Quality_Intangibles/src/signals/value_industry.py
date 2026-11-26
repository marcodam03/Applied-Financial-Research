"""
Industry-relative value: cheap compared with its own industry, not with the market.

Why: industries are valued on different accounting numbers and at different levels
(a bank always looks "cheap" on book value next to a software firm). The plain value
composite therefore makes big industry bets: long banks and utilities, short tech.
Research finds that most of the value premium comes from cheapness WITHIN an
industry, while the industry bet adds risk but little return (Asness, Porter and
Stevens 2000; Cohen and Polk 1998). Removing the industry bet should give a
smoother value sleeve that the HML factor (which keeps the industry bet) does not
fully explain.

Same four ratios as value_composite.py (fiscal year ending in t-1, ME of December t-1):
    be/ME, ib/ME, (ib + dp)/ME, oibdp/EV
Rule fixed in advance, from textbook valuation practice (no data fitting):
    Financials (FF12 "Money", SIC 6000-6999): only be/ME and ib/ME, because for banks
    and insurers debt is raw material, so EV and EBITDA have no meaning.
    All other industries: the four ratios, at least 3 available.
Each ratio is ranked within (formation year, FF12 industry), industries with at least
10 stocks; the signal is the average rank. High = cheap for its industry.
Deciles with NYSE breakpoints as usual, so every decile holds every industry.
"""
import pandas as pd

from src.signals._industry import financial_june, industry_june, industry_rank
from src.signals._quality import me_december, restrict, universe_june
from src.signals._yearly import to_rows, yearly

COMPUSTAT_COLS = ["be", "ib", "dp", "oibdp", "dltt", "dlc", "pstk", "che"]
CRSP_COLS = ["me_firm"]
REBALANCE = "annual"
LONG, SHORT = 10, 1

def value_industry(panel: pd.DataFrame) -> pd.Series:
    return value_industry_scores(panel, "ff12")


def value_industry_scores(panel: pd.DataFrame, scheme: str) -> pd.Series:
    """scheme: "ff12" (main) or "ff49" (robustness)."""
    return to_rows(panel, value_industry_year(panel, scheme))


def value_industry_year(panel: pd.DataFrame, scheme: str = "ff12", intangible: bool = False) -> pd.Series:
    """One score per (PERMNO, formation_year t), accounting of the fiscal year ending in t-1,
    price = firm market equity of December t-1 (Fama-French timing), sorted at the end of June t.
    scheme: "ff12" (main) or "ff49" (robustness).
    intangible = True: the book value in book/price is the intangible-adjusted book of
    Eisfeldt, Kim and Papanikolaou (2022): be - goodwill + knowledge capital + organisation
    capital (src/signals/_intangibles.py). Used by quality_value_industry_intan.py."""
    a = yearly(panel, COMPUSTAT_COLS)
    book = a["be"]
    if intangible:
        from src.signals._intangibles import intangible_table
        t = intangible_table(panel).reindex(a.index)
        gw = yearly(panel, ["gdwl"])["gdwl"].reindex(a.index).fillna(0).clip(lower=0)
        book = a["be"] - gw + t["k_know"].fillna(0) + t["k_org"].fillna(0)
    me = me_december(panel).reindex(a.index)
    ev = me + a["dltt"].fillna(0) + a["dlc"].fillna(0) + a["pstk"].fillna(0) - a["che"].fillna(0)
    parts = pd.DataFrame({
        "bm": (book / me).where(book > 0),
        "ep": a["ib"] / me,
        "cfp": (a["ib"] + a["dp"].fillna(0)) / me,
        "ebitda_ev": a["oibdp"] / ev.where(ev > 0),
    })
    parts = restrict(parts[me.notna()], universe_june(panel))
    ind = industry_june(panel, scheme).reindex(parts.index)
    parts = parts[ind.notna()]
    ind = ind[ind.notna()]

    bank = financial_june(panel).reindex(ind.index).fillna(0).astype(bool).to_numpy()
    parts.loc[bank, ["cfp", "ebitda_ev"]] = float("nan")

    ranks = industry_rank(parts, ind, min_stocks=10)
    need = pd.Series(3, index=ranks.index).where(~bank, 2)
    return ranks.mean(axis=1).where(ranks.notna().sum(axis=1) >= need).dropna()
