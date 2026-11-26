"""
Extensions of the final strategy qvi_intan_full (_qvi_plus.py), used by the
qvi_*.py and rd_ability.py signals of the October 14 batch.

Annual pieces (one value per (PERMNO, formation_year t): accounting of the fiscal
year ending in t-1, market equity of December t-1, sorted at the end of June t):
  value_scores(panel, adjust, payout)
      adjust="book": the four ratios of value_industry (book adjusted for intangibles)
      adjust="all":  all four ratios adjusted for intangibles: book = be - goodwill + K + O,
                     earnings = ib + intangible investment - its amortisation,
                     cash flow = ib + dp + intangible investment,
                     EBITDA = oibdp + intangible investment (EV unchanged)
      payout=True:   adds net payout yield (dividends + buybacks - issues) / ME
                     (Boudoukh, Michaely, Richardson and Roberts 2007)
      Financials (SIC 6000-6999): book/price, earnings/price (+ payout) only.
  safety(panel):     minus the volatility of the last 36 monthly returns (at least 24),
                     measured at the end of June t (low risk = safe)
  rd_ability(panel): Cohen, Diether and Malloy (2013) "Misvaluing innovation": how much
                     a firm's past R&D turned into sales growth. Each year, slope of the
                     firm's sales growth on its R&D intensity (R&D / sales) 1 and 2 years
                     earlier, over its last 8 fiscal years (at least 5), averaged over the
                     two lags. Only firms with R&D in the latest year.
  combo_year(panel, ...): value + quality (intangible cash profitability) + optional
                     extra legs, each ranked within (formation year, FF12 industry),
                     averaged, ranked again within the industry.

Monthly pieces (one value per panel row, known at the end of that month):
  residual_momentum(panel): Blitz, Huij and Martens (2011): sum of the FF5 residual
                     returns of months t-11..t-1 divided by their standard deviation
                     (at least 8 months), so the most recent month is skipped.
  timely_value_quality(panel): Asness and Frazzini (2013) "The devil in HML's details":
                     the same intangible-adjusted ratios but with the market equity of
                     the current month instead of December t-1, so the score moves with
                     the price every month; quality as in the annual version. Ranked
                     within (month, FF12 industry).
"""
import numpy as np
import pandas as pd

from src.signals._common import factor_residuals, from_matrix
from src.signals._industry import ff12, financial_june, industry_june, industry_rank, is_financial
from src.signals._intangibles import DELTA_ORG, DELTA_RD, intangible_table
from src.signals._qvi_plus import COMPUSTAT_COLS as _BASE_COLS
from src.signals._qvi_plus import MIN_STOCKS, intangible_cash_profit
from src.signals._quality import me_december, restrict, universe_june
from src.signals._yearly import june, lag_years, to_rows, yearly
from src.signals.value_industry import value_industry_year

VALUE_ITEMS = ["be", "ib", "dp", "oibdp", "dltt", "dlc", "pstk", "che", "gdwl"]
PAYOUT_ITEMS = ["dvc", "prstkc", "sstk"]
COMPUSTAT_COLS = sorted(set(_BASE_COLS) | set(VALUE_ITEMS) | set(PAYOUT_ITEMS))
CRSP_COLS = ["me_firm"]


def _finite(df):
    return df.replace([np.inf, -np.inf], np.nan)


def _intangible_flows(panel: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Intangible investment, capital and amortisation per (PERMNO, formation_year)."""
    t = intangible_table(panel)
    t = t[~t.index.duplicated()]
    inv = t["rd"] + t["org_inv"]
    cap = t["k_know"] + t["k_org"]
    cap_lag = lag_years(cap, 1).reindex(cap.index)
    # amortisation = investment - change in capital; first year: depreciation rates x capital
    amort = (inv - (cap - cap_lag)).fillna(DELTA_RD * t["k_know"] + DELTA_ORG * t["k_org"])
    return pd.DataFrame({"inv": inv, "cap": cap, "amort": amort}).reindex(index)


def adjusted_ratios(a: pd.DataFrame, f: pd.DataFrame, me: pd.Series, adjust: str) -> pd.DataFrame:
    """The four valuation ratios. a: accounting items, f: intangible flows, me: price."""
    inv = f["inv"].fillna(0) if adjust == "all" else 0.0
    amort = f["amort"].fillna(0) if adjust == "all" else 0.0
    book = a["be"] - a["gdwl"].fillna(0).clip(lower=0) + f["cap"].fillna(0)
    ev = me + a["dltt"].fillna(0) + a["dlc"].fillna(0) + a["pstk"].fillna(0) - a["che"].fillna(0)
    return _finite(pd.DataFrame({
        "bm": (book / me).where(book > 0),
        "ep": (a["ib"] + inv - amort) / me,
        "cfp": (a["ib"] + a["dp"].fillna(0) + inv) / me,
        "ebitda_ev": (a["oibdp"] + inv) / ev.where(ev > 0),
    }))


def value_scores(panel: pd.DataFrame, adjust: str = "book", payout: bool = False,
                 scheme: str = "ff12") -> pd.Series:
    if adjust == "book" and not payout:
        return value_industry_year(panel, scheme, intangible=True)
    a = yearly(panel, VALUE_ITEMS + PAYOUT_ITEMS)
    me = me_december(panel).reindex(a.index)
    parts = adjusted_ratios(a, _intangible_flows(panel, a.index), me, adjust)
    if payout:
        net_payout = (a["dvc"].fillna(0) + a["prstkc"].fillna(0) - a["sstk"].fillna(0))
        parts["npy"] = _finite(net_payout / me)
    parts = restrict(parts[me.notna()], universe_june(panel))
    ind = industry_june(panel, scheme).reindex(parts.index)
    parts, ind = parts[ind.notna()], ind[ind.notna()]
    bank = financial_june(panel).reindex(ind.index).fillna(0).astype(bool).to_numpy()
    parts.loc[bank, ["cfp", "ebitda_ev"]] = np.nan
    ranks = industry_rank(parts, ind, MIN_STOCKS)
    need = pd.Series(3, index=ranks.index).where(~bank, 2)
    return ranks.mean(axis=1).where(ranks.notna().sum(axis=1) >= need).dropna()


def safety(panel: pd.DataFrame) -> pd.Series:
    p = panel.sort_values(["PERMNO", "MthCalDt"])
    vol = p.groupby("PERMNO")["MthRet"].rolling(36, min_periods=24).std().reset_index(level=0, drop=True)
    s = june(panel, (-vol).reindex(panel.index))
    return restrict(_finite(s).dropna(), universe_june(panel))


def rd_ability(panel: pd.DataFrame, window: int = 8, min_years: int = 5) -> pd.Series:
    a = yearly(panel, ["sale", "xrd"]).sort_index()
    sale = a["sale"].where(a["sale"] > 0)
    growth = np.log(sale / lag_years(sale, 1).reindex(sale.index))
    intensity = (a["xrd"].fillna(0).clip(lower=0) / sale)
    slopes = []
    for k in (1, 2):
        x = lag_years(intensity, k).reindex(sale.index)
        ok = growth.notna() & x.notna()
        d = pd.DataFrame({"n": ok.astype(float), "x": x.where(ok, 0), "y": growth.where(ok, 0)})
        d["xx"], d["xy"] = d["x"] ** 2, d["x"] * d["y"]
        r = d.groupby(level="PERMNO").rolling(window, min_periods=1).sum().droplevel(0)
        n = r["n"]
        var = r["xx"] - r["x"] ** 2 / n
        slope = (r["xy"] - r["x"] * r["y"] / n) / var.where(var > 1e-8)
        slopes.append(slope.where(n >= min_years))
    ability = pd.concat(slopes, axis=1).mean(axis=1)
    ability = ability.where(a["xrd"].fillna(0) > 0)
    return restrict(_finite(ability).dropna(), universe_june(panel))


def combo_year(panel: pd.DataFrame, adjust: str = "book", payout: bool = False,
               with_safety: bool = False, with_ability: bool = False) -> pd.Series:
    parts = pd.concat({"value": value_scores(panel, adjust, payout),
                       "quality": intangible_cash_profit(panel)}, axis=1).dropna()
    if with_safety:
        parts = parts.join(safety(panel).rename("safety"), how="inner")
    if with_ability:
        parts = parts.join(rd_ability(panel).rename("ability"), how="left")
    ind = industry_june(panel).reindex(parts.index)
    parts, ind = parts[ind.notna()], ind[ind.notna()]
    ranks = industry_rank(parts, ind, MIN_STOCKS)
    if with_ability:  # firms without R&D: neutral on this leg
        ranks["ability"] = ranks["ability"].fillna(0.0)
    score = ranks.mean(axis=1, skipna=False).dropna()
    return industry_rank(score.to_frame("s"), ind.reindex(score.index), MIN_STOCKS)["s"].dropna()


# ------------------------------------------------------------------------------
# Monthly pieces
# ------------------------------------------------------------------------------
def month_rank(panel: pd.DataFrame, values: pd.Series, industry: pd.Series = None,
               min_stocks: int = MIN_STOCKS) -> pd.Series:
    """Percentile rank within each month (and industry), from -0.5 to +0.5."""
    keys = [panel["MthCalDt"]] if industry is None else [panel["MthCalDt"], industry]
    g = values.groupby(keys)
    r = g.rank(pct=True) - 0.5
    return r.where(g.transform("count") >= min_stocks)


def residual_momentum(panel: pd.DataFrame, months: int = 11, min_obs: int = 8) -> pd.Series:
    R, month_ids, permnos = factor_residuals(panel, "FF5", 36)
    R = R.astype(float)
    shifted = np.vstack([np.full((1, R.shape[1]), np.nan), R[:-1]])  # row t holds month t-1
    has = ~np.isnan(shifted)
    x = np.nan_to_num(shifted)
    cs = lambda M: np.vstack([np.zeros((1, M.shape[1])), np.cumsum(M, axis=0)])  # noqa: E731
    S, S2, N = cs(x), cs(x ** 2), cs(has.astype(float))
    T = R.shape[0]
    lo = np.maximum(np.arange(1, T + 1) - months, 0)
    hi = np.arange(1, T + 1)
    s, s2, n = S[hi] - S[lo], S2[hi] - S2[lo], N[hi] - N[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / n
        sd = np.sqrt(np.maximum(s2 / n - mean ** 2, 0) * n / np.maximum(n - 1, 1))
        score = np.where((n >= min_obs) & (sd > 0), s / sd, np.nan)
    return from_matrix(panel, score, month_ids, permnos)


def timely_value_quality(panel: pd.DataFrame, adjust: str = "book", payout: bool = False,
                         ability: bool = False) -> pd.Series:
    """payout: net payout yield (dividends + buybacks - issues) / current ME as an extra
    cheapness ratio. ability: innovation ability (rd_ability) as a third leg for R&D
    firms, neutral (median) for the others."""
    ok = panel["mkt_cap"] > 0
    me = (panel["me_firm"] / 1000).where(panel["me_firm"] > 0)  # current month, $ millions
    a = panel[VALUE_ITEMS]
    flows = _intangible_flows(panel, pd.MultiIndex.from_arrays(
        [panel["PERMNO"], panel["formation_year"]], names=["PERMNO", "formation_year"]))
    flows.index = panel.index
    ratios = adjusted_ratios(a, flows, me, adjust)
    if payout:
        net_payout = (panel["dvc"].fillna(0) + panel["prstkc"].fillna(0) - panel["sstk"].fillna(0))
        ratios["npy"] = _finite(net_payout / me)
    industry = ff12(panel["SICCD"])
    bank = is_financial(panel["SICCD"]).to_numpy()
    ratios.loc[bank, ["cfp", "ebitda_ev"]] = np.nan
    ratios = ratios.where(ok & industry.notna())
    ranks = pd.DataFrame({c: month_rank(panel, ratios[c], industry) for c in ratios.columns})
    need = pd.Series(np.where(bank, 2, 3), index=panel.index)
    value = ranks.mean(axis=1).where(ranks.notna().sum(axis=1) >= need)
    quality = month_rank(panel, to_rows(panel, intangible_cash_profit(panel)).where(ok), industry)
    legs = [value, quality]
    if ability:
        abil = to_rows(panel, rd_ability(panel)).where(ok)
        legs.append(month_rank(panel, abil, industry, min_stocks=5).fillna(0.0).where(value.notna()))
    score = pd.concat(legs, axis=1).mean(axis=1, skipna=False)
    return month_rank(panel, score, industry)
