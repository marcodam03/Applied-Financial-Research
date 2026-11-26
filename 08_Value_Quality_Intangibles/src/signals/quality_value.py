"""
Quality + value (Novy-Marx 2013; Asness, Frazzini and Pedersen 2019): buy stocks that
are both cheap and profitable in cash, sell stocks that are both expensive and
unprofitable.

Why: value earns a return but is explained by the value factor; cash-based
profitability earns an alpha against FF5 but little raw return; and the two tend to
move in opposite directions, so the combination should keep part of both, with a
smoother return than either.

Signal, one value per stock and formation year t (sorted at the end of June t):
    average of the yearly percentile ranks of
        value_composite   (B/M, E/P, CF/P, EBITDA/EV; see value_composite.py)
        cash_op           (cash-based operating profitability; see cash_op.py)
Both are required. High = cheap and profitable. No neutralisation against the
factors (slow_combo showed it can strip out the return together with the exposure).
"""
import pandas as pd

from src.signals._yearly import KEY, to_rows, year_rank
import src.signals.cash_op as _cop
import src.signals.value_composite as _vc

COMPUSTAT_COLS = sorted(set(_vc.COMPUSTAT_COLS) | set(_cop.COMPUSTAT_COLS))
CRSP_COLS = ["me_firm"]
REBALANCE = "annual"
LONG, SHORT = 10, 1


def _per_year(panel: pd.DataFrame, values: pd.Series) -> pd.Series:
    """Row-level annual signal -> one value per (PERMNO, formation_year)."""
    s = pd.Series(values.to_numpy(), index=pd.MultiIndex.from_arrays(
        [panel["PERMNO"], panel["formation_year"]], names=KEY))
    s = s.dropna()
    return s[~s.index.duplicated()]


def quality_value(panel: pd.DataFrame) -> pd.Series:
    parts = pd.concat({
        "value": _per_year(panel, _vc.value_composite(panel)),
        "quality": _per_year(panel, _cop.cash_op(panel)),
    }, axis=1).dropna()
    combo = parts.apply(year_rank).mean(axis=1)
    return to_rows(panel, combo)
