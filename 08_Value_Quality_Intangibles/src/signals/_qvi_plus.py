"""
Builder for the extensions of Roberto's quality_value_industry (QVI), used by the
qvi_*.py signals. One score per (PERMNO, formation_year t), accounting of the
fiscal year ending in t-1, market equity of December t-1, sorted at the end of June t.

Pieces (each ranked within (formation year, industry), industries >= 10 stocks):
  value      value_industry_year(intangible=True): B/M with intangible-adjusted book
             (be - goodwill + knowledge + organisation capital), E/P, CF/P, EBITDA/EV
             (financials: B/M and E/P only). Eisfeldt, Kim and Papanikolaou (2022).
  quality    intangible-adjusted cash profitability:
               (sale - cogs - xsga + xrd + 0.30 x SG&A without R&D
                - change in receivables - change in inventory + change in payables)
               / (assets + knowledge capital + organisation capital)
             = cash_op of Ball et al. (2016) with the organisation-capital investment
             added back and the intangible capital in the assets (Peters and Taylor
             2017). A missing item or lag counts as no change.
  intensity  (optional) intangible investment relative to price, the signals that the
             q-factor model fails to price in Hou, Xue and Zhang (2020): R&D / ME and
             advertising / ME (missing xrd or xad = 0), averaged ranks.

Options:
  scheme     "ff12" or "ff49" industries
  drop_fin   drop financials (SIC 6000-6999): accounting ratios mean little for them
  neutral    remove from the final score what the q5 characteristics explain: each
             formation year, regress the score on the industry ranks of size (log ME),
             investment (asset growth), ROE (ib / lagged book equity) and the change in
             ROE (a proxy for q5's expected growth), keep the residual, rank it again
             within the industry. What is left cannot load on q5 by construction.
"""
import numpy as np
import pandas as pd

from src.signals._industry import financial_june, industry_june, industry_rank
from src.signals._intangibles import intangible_table
from src.signals._quality import me_december, restrict, universe_june
from src.signals._yearly import lag_years, yearly
from src.signals.value_industry import value_industry_year

CASH_ITEMS = ["sale", "cogs", "xsga", "xrd", "at", "rect", "invt", "ap",
              "rect_lag", "invt_lag", "ap_lag"]
COMPUSTAT_COLS = sorted(set(CASH_ITEMS) | {
    "be", "be_lag", "ib", "dp", "oibdp", "dltt", "dlc", "pstk", "che", "gdwl", "xad",
    "at_lag"})
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


def intangible_intensity(panel: pd.DataFrame) -> pd.DataFrame:
    a = yearly(panel, ["xrd", "xad"])
    me = me_december(panel).reindex(a.index)
    parts = pd.DataFrame({"rd_me": a["xrd"].fillna(0).clip(lower=0) / me,
                          "ad_me": a["xad"].fillna(0).clip(lower=0) / me})
    parts = parts[me.notna()].replace([np.inf, -np.inf], np.nan)
    return restrict(parts, universe_june(panel))


def q5_characteristics(panel: pd.DataFrame) -> pd.DataFrame:
    a = yearly(panel, ["at", "at_lag", "ib", "be_lag"])
    me = me_december(panel).reindex(a.index)
    roe = (a["ib"] / a["be_lag"].where(a["be_lag"] > 0)).replace([np.inf, -np.inf], np.nan)
    c = pd.DataFrame({
        "size": np.log(me),
        "inv": a["at"] / a["at_lag"].where(a["at_lag"] > 0) - 1,
        "roe": roe,
        "droe": roe - lag_years(roe, 1).reindex(roe.index),
    }, index=a.index)
    return c.replace([np.inf, -np.inf], np.nan)


def _neutralise(score: pd.Series, controls: pd.DataFrame, ind: pd.Series) -> pd.Series:
    """Per formation year: regress the score on the industry ranks of the controls
    (missing = 0, the median), keep the residual."""
    X = industry_rank(controls.reindex(score.index), ind, MIN_STOCKS).fillna(0.0)
    out = pd.Series(np.nan, index=score.index)
    years = score.index.get_level_values("formation_year")
    for _, rows in pd.Series(np.arange(len(score))).groupby(years).groups.items():
        rows = np.asarray(rows)
        if len(rows) < 50:
            continue
        Xi = np.column_stack([np.ones(len(rows)), X.iloc[rows].to_numpy()])
        yi = score.iloc[rows].to_numpy()
        beta = np.linalg.lstsq(Xi, yi, rcond=None)[0]
        out.iloc[rows] = yi - Xi @ beta
    return out


def qvi_plus_year(panel: pd.DataFrame, scheme: str = "ff12", drop_fin: bool = False,
                  intensity: bool = False, neutral: bool = False) -> pd.Series:
    parts = pd.concat({
        "value": _clean(value_industry_year(panel, scheme, intangible=True)),
        "quality": intangible_cash_profit(panel),
    }, axis=1).dropna()
    if intensity:
        inten = intangible_intensity(panel)
        parts = parts.join(inten, how="inner")
    ind = industry_june(panel, scheme).reindex(parts.index)
    keep = ind.notna()
    if drop_fin:
        fin = financial_june(panel).reindex(parts.index).fillna(0).astype(bool)
        keep &= ~fin
    parts, ind = parts[keep], ind[keep]

    ranks = industry_rank(parts, ind, MIN_STOCKS)
    if intensity:  # intensity = one third: average of its two ratios first
        ranks = pd.DataFrame({"value": ranks["value"], "quality": ranks["quality"],
                              "intensity": ranks[["rd_me", "ad_me"]].mean(axis=1)})
    score = ranks.mean(axis=1, skipna=False).dropna()
    if neutral:
        score = _neutralise(score, q5_characteristics(panel), ind.reindex(score.index)).dropna()
    final = industry_rank(score.to_frame("s"), ind.reindex(score.index), MIN_STOCKS)["s"]
    return final.dropna()
