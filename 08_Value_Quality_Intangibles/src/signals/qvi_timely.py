"""
FINAL STRATEGY (docs/STRATEGY.md): cheap and cash-profitable stocks within their
industry, with intangible capital counted, at current prices.

Same value + quality score as qvi_intan_full, but the cheapness ratios use the market
equity of the current month instead of December t-1 (Asness and Frazzini 2013, "The
devil in HML's details"), so a stock that has fallen becomes cheaper at once instead
of up to 18 months later. Accounting data unchanged (fiscal year ending in t-1, used
from July t). Ranked within (month, FF12 industry); built in _qvi_ext.py.
Portfolios formed every 3 months (configs/final.yaml):
    python scripts/run_signal.py --signal qvi_timely --config configs/final.yaml
"""
import pandas as pd

from src.signals._qvi_ext import COMPUSTAT_COLS, CRSP_COLS, timely_value_quality  # noqa: F401

REBALANCE = "monthly"
LONG, SHORT = 10, 1


def qvi_timely(panel: pd.DataFrame) -> pd.Series:
    return timely_value_quality(panel)
