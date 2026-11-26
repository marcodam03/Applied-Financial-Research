"""
Monthly ("timely") version of the value + quality score of qvi_intan_full
(_qvi_plus.py), used by qvi_timely.py.

Asness and Frazzini (2013), "The devil in HML's details": the cheapness ratios use
the market equity of the CURRENT month instead of December t-1, so a stock that has
fallen becomes cheaper at once instead of up to 18 months later. The accounting data
are unchanged (fiscal year ending in t-1, used from July t).

Every month, within each FF12 industry (at least 10 stocks):
  value    average rank of four cheapness ratios, book adjusted for intangible
           capital (Peters and Taylor 2017, _intangibles.py):
             book / ME       book = be - goodwill + knowledge + organisation capital
             earnings / ME   ib
             cash flow / ME  ib + dp
             EBITDA / EV     EV = ME + dltt + dlc + pstk - che
           financials (SIC 6000-6999): book / ME and earnings / ME only.
           At least 3 ratios (2 for financials) must be available.
  quality  intangible-adjusted cash profitability (intangible_cash_profit in
           _qvi_plus.py), ranked within (month, industry)
  score    average of the two ranks, ranked again within (month, industry).
"""
import numpy as np
import pandas as pd

from src.signals._industry import ff12, is_financial
from src.signals._intangibles import intangible_table
from src.signals._qvi_plus import COMPUSTAT_COLS as _BASE_COLS
from src.signals._qvi_plus import MIN_STOCKS, intangible_cash_profit
from src.signals._yearly import to_rows

VALUE_ITEMS = ["be", "ib", "dp", "oibdp", "dltt", "dlc", "pstk", "che", "gdwl"]
COMPUSTAT_COLS = sorted(set(_BASE_COLS) | set(VALUE_ITEMS))
CRSP_COLS = ["me_firm"]


def _finite(df):
    return df.replace([np.inf, -np.inf], np.nan)


def intangible_capital(panel: pd.DataFrame) -> pd.Series:
    """Knowledge + organisation capital of the fiscal year ending in t-1, one value
    per panel row."""
    t = intangible_table(panel)
    t = t[~t.index.duplicated()]
    cap = t["k_know"] + t["k_org"]
    keys = pd.MultiIndex.from_arrays([panel["PERMNO"], panel["formation_year"]],
                                     names=["PERMNO", "formation_year"])
    return pd.Series(cap.reindex(keys).to_numpy(), index=panel.index)


def value_ratios(a: pd.DataFrame, capital: pd.Series, me: pd.Series) -> pd.DataFrame:
    """The four cheapness ratios. a: accounting items, capital: intangible capital,
    me: market equity ($ millions)."""
    book = a["be"] - a["gdwl"].fillna(0).clip(lower=0) + capital.fillna(0)
    ev = me + a["dltt"].fillna(0) + a["dlc"].fillna(0) + a["pstk"].fillna(0) - a["che"].fillna(0)
    return _finite(pd.DataFrame({
        "bm": (book / me).where(book > 0),
        "ep": a["ib"] / me,
        "cfp": (a["ib"] + a["dp"].fillna(0)) / me,
        "ebitda_ev": a["oibdp"] / ev.where(ev > 0),
    }))


def month_rank(panel: pd.DataFrame, values: pd.Series, industry: pd.Series,
               min_stocks: int = MIN_STOCKS) -> pd.Series:
    """Percentile rank within each (month, industry), from -0.5 to +0.5; NaN when
    the group has fewer than min_stocks stocks."""
    g = values.groupby([panel["MthCalDt"], industry])
    r = g.rank(pct=True) - 0.5
    return r.where(g.transform("count") >= min_stocks)


def timely_value_quality(panel: pd.DataFrame) -> pd.Series:
    ok = panel["mkt_cap"] > 0
    me = (panel["me_firm"] / 1000).where(panel["me_firm"] > 0)  # current month, $ millions
    ratios = value_ratios(panel[VALUE_ITEMS], intangible_capital(panel), me)
    industry = ff12(panel["SICCD"])
    bank = is_financial(panel["SICCD"]).to_numpy()
    ratios.loc[bank, ["cfp", "ebitda_ev"]] = np.nan
    ratios = ratios.where(ok & industry.notna())
    ranks = pd.DataFrame({c: month_rank(panel, ratios[c], industry) for c in ratios.columns})
    need = pd.Series(np.where(bank, 2, 3), index=panel.index)
    value = ranks.mean(axis=1).where(ranks.notna().sum(axis=1) >= need)
    quality = month_rank(panel, to_rows(panel, intangible_cash_profit(panel)).where(ok), industry)
    score = pd.concat([value, quality], axis=1).mean(axis=1, skipna=False)
    return month_rank(panel, score, industry)
