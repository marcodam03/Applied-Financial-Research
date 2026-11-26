"""
Combined strategy: three signals, each first "neutralised" against the
characteristics the benchmark factors are built on, then averaged.

Signals (formulas imported from their own files, so they are identical):
  seasonality_idio, laggard, dividend_month

Step 1, neutralisation (every month, separately for each signal):
  regress the signal's rank on the ranks of
    size (log firm market cap), book-to-market, operating profitability,
    asset growth, momentum 12-2, last-month return (short-term reversal)
  and keep the residual: the part of the signal unrelated to the characteristics
  behind FF3-FF6 and q4/q5. Missing characteristics count as the median (rank 0)
  so that no stock is dropped because of them.
  Exceptions:
  - laggard is NOT neutralised against last-month return: its signal is the
    industry leaders' return minus the stock's own return, so that would remove
    most of it. Whether what remains is just short-term reversal is tested with
    the FF6+STR and q5+STR models (src/evaluation/alpha.py).
  - no industry dummies: they would remove the industry information of laggard.
Step 2: rank each neutralised signal again (-0.5 to +0.5) and average the ones
  available for the stock (at least MIN_SIGNALS of the 3; dividend_month only
  exists for regular dividend payers).

Inspiration: Lewellen (2015) "The cross-section of expected stock returns"
(cross-sectional regressions on many characteristics); DeMiguel, Martin-Utrera,
Nogales and Uppal (2020) "A transaction-cost perspective on the multitude of firm
characteristics" (combining signals nets trades and cuts costs); Stambaugh and
Yuan (2017) "Mispricing factors" (composite of anomaly ranks).
"""
import numpy as np
import pandas as pd

from src.signals._common import cross_rank
from src.signals.dividend_month import dividend_month
from src.signals.laggard import laggard
from src.signals.momentum import momentum
from src.signals.seasonality_idio import seasonality_idio

COMPUSTAT_COLS = ["be", "at", "at_lag", "sale", "cogs", "xsga"]
CRSP_COLS = ["me_firm", "DisDivAmt"]
REBALANCE = "monthly"
LONG, SHORT = 10, 1
MIN_SIGNALS = 2

# signal name: (function, neutralise against last-month return?)
SIGNALS = {
    "seasonality_idio": (seasonality_idio, True),
    "laggard": (laggard, False),
    "dividend_month": (dividend_month, True),
}


def controls(panel: pd.DataFrame) -> pd.DataFrame:
    """Characteristics behind the benchmark factors, known at the end of month t."""
    me = panel["me_firm"].where(panel["me_firm"] > 0)
    at = panel["at"].where(panel["at"] > 0)
    c = pd.DataFrame(index=panel.index)
    c["size"] = np.log(me)
    c["bm"] = panel["be"] / (me / 1000)              # Compustat millions, CRSP thousands
    c["op"] = (panel["sale"] - panel["cogs"] - panel["xsga"].fillna(0)) / at
    c["inv"] = panel["at"] / panel["at_lag"].where(panel["at_lag"] > 0) - 1
    c["mom"] = momentum(panel)
    c["rev"] = panel["MthRet"]
    return c.replace([np.inf, -np.inf], np.nan)


def neutralise(panel: pd.DataFrame, signal: pd.Series, ctrl_ranks: pd.DataFrame) -> pd.Series:
    """Residual of the signal's monthly rank on the control ranks, re-ranked."""
    y_all = cross_rank(panel, signal).to_numpy()
    X_all = ctrl_ranks.fillna(0.0).to_numpy()
    resid = np.full(len(panel), np.nan)
    rows_ok = np.flatnonzero(~np.isnan(y_all))
    months = panel["MthCalDt"].to_numpy()[rows_ok]
    for rows in pd.Series(rows_ok).groupby(months).indices.values():
        rows = rows_ok[rows]
        if len(rows) < 30:
            continue
        X = np.column_stack([np.ones(len(rows)), X_all[rows]])
        beta = np.linalg.lstsq(X, y_all[rows], rcond=None)[0]
        resid[rows] = y_all[rows] - X @ beta
    return cross_rank(panel, pd.Series(resid, index=panel.index))


def alpha_combo2(panel: pd.DataFrame) -> pd.Series:
    ctrl_ranks = controls(panel).apply(lambda col: cross_rank(panel, col))
    parts = {}
    for name, (fn, with_rev) in SIGNALS.items():
        print(f"  alpha_combo2: {name}...", flush=True)
        ctrl = ctrl_ranks if with_rev else ctrl_ranks.drop(columns="rev")
        parts[name] = neutralise(panel, fn(panel), ctrl)
    parts = pd.DataFrame(parts)
    return parts.mean(axis=1).where(parts.notna().sum(axis=1) >= MIN_SIGNALS)