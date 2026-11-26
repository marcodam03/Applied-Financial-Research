"""
Helpers for annual signals (Fama-French timing): one value per stock and
formation year t, known at the end of June t, held July t - June t+1.

In the monthly panel, the accounting columns of formation year t come from the
fiscal year ending in calendar year t-1 (clean.py / panel.py), so they are the
same in every month of that formation year.

- yearly(panel, cols):          one row per (PERMNO, formation_year) with those columns
- december(panel, col):         value of a CRSP column in December of t-1, for formation year t
- june(panel, values):          value of a monthly series in June of t, for formation year t
- lag_years(s, k):              value k formation years earlier (same stock, no gap)
- to_rows(panel, s):            spread a (PERMNO, formation_year) series over the monthly rows
- neutralise_yearly(...):       cross-sectional regression on control ranks, each year
"""
import numpy as np
import pandas as pd

KEY = ["PERMNO", "formation_year"]


def yearly(panel: pd.DataFrame, cols) -> pd.DataFrame:
    return panel.groupby(KEY)[list(cols)].first()


def december(panel: pd.DataFrame, col: str) -> pd.Series:
    d = panel.loc[panel["MthCalDt"].dt.month == 12, ["PERMNO", "MthCalDt", col]]
    d = d.assign(formation_year=d["MthCalDt"].dt.year + 1)
    return d.set_index(KEY)[col]


def june(panel: pd.DataFrame, values: pd.Series) -> pd.Series:
    """values: one per panel row. Returns the June t value for formation year t."""
    jun = panel["MthCalDt"].dt.month == 6
    d = pd.DataFrame({"PERMNO": panel.loc[jun, "PERMNO"],
                      "formation_year": panel.loc[jun, "MthCalDt"].dt.year,
                      "v": values[jun]})
    return d.set_index(KEY)["v"]


def lag_years(s: pd.Series, k: int) -> pd.Series:
    """Value k formation years earlier for the same stock; NaN if a year is missing."""
    s = s.sort_index()
    years = pd.Series(s.index.get_level_values("formation_year"), index=s.index)
    prev = s.groupby(level="PERMNO").shift(k)
    prev_year = years.groupby(level="PERMNO").shift(k)
    return prev.where(years - prev_year == k)


def to_rows(panel: pd.DataFrame, s: pd.Series) -> pd.Series:
    idx = pd.MultiIndex.from_arrays([panel["PERMNO"], panel["formation_year"]], names=KEY)
    s = s[~s.index.duplicated()]
    return pd.Series(s.reindex(idx).to_numpy(), index=panel.index)


def year_rank(s: pd.Series) -> pd.Series:
    """Percentile rank within each formation year, from -0.5 to +0.5."""
    return s.groupby(level="formation_year").rank(pct=True) - 0.5


def neutralise_yearly(signal: pd.Series, controls: pd.DataFrame, min_obs: int = 30) -> pd.Series:
    """Each formation year: regress the signal's rank on the ranks of the controls
    (missing control = median), keep the residual, rank it again."""
    y = year_rank(signal.dropna())
    X = controls.reindex(y.index).apply(year_rank).fillna(0.0)
    out = pd.Series(np.nan, index=y.index)
    for _, idx in y.groupby(level="formation_year").groups.items():
        if len(idx) < min_obs:
            continue
        Xi = np.column_stack([np.ones(len(idx)), X.loc[idx].to_numpy()])
        yi = y.loc[idx].to_numpy()
        beta = np.linalg.lstsq(Xi, yi, rcond=None)[0]
        out.loc[idx] = yi - Xi @ beta
    return year_rank(out.dropna())


def annual_controls(panel: pd.DataFrame) -> pd.DataFrame:
    """Characteristics behind the benchmark factors, one row per (PERMNO, formation_year),
    all known at the end of June t: size and book-to-market with December t-1 firm
    market cap, operating profitability and asset growth of fiscal year t-1,
    momentum 12-2 at the end of June t."""
    from src.signals.momentum import momentum

    acc = yearly(panel, ["be", "at", "at_lag", "sale", "cogs", "xsga"])
    me = december(panel, "me_firm") / 1000          # millions, like Compustat
    me = me[~me.index.duplicated()].reindex(acc.index)
    at = acc["at"].where(acc["at"] > 0)
    c = pd.DataFrame(index=acc.index)
    c["size"] = np.log(me.where(me > 0))
    c["bm"] = acc["be"] / me.where(me > 0)
    c["op"] = (acc["sale"] - acc["cogs"] - acc["xsga"].fillna(0)) / at
    c["inv"] = acc["at"] / acc["at_lag"].where(acc["at_lag"] > 0) - 1
    mom = june(panel, momentum(panel))
    c["mom"] = mom[~mom.index.duplicated()].reindex(acc.index)
    return c.replace([np.inf, -np.inf], np.nan)
