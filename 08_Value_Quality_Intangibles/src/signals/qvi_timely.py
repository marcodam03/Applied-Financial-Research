"""
qvi_intan_full with timely prices (Asness and Frazzini 2013, "The devil in HML's
details"): the cheapness ratios use the market equity of the current month instead of
December t-1, so a stock that has fallen becomes cheaper at once instead of up to 18
months later. Accounting data unchanged (fiscal year t-1, from July t). Ranked within
(month, FF12 industry) every month. They find the timely version much better, above
all next to momentum. Run with configs/fast_monthly.yaml (hold/buffer variants).
"""
import pandas as pd

from src.signals._qvi_ext import COMPUSTAT_COLS, CRSP_COLS, timely_value_quality  # noqa: F401

REBALANCE = "monthly"
LONG, SHORT = 10, 1


def qvi_timely(panel: pd.DataFrame) -> pd.Series:
    return timely_value_quality(panel)
