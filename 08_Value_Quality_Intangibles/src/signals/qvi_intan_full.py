"""
Building block 4 of the final strategy (Table 5 of the paper): its annual version.
Sorted once a year, at the end of June t, with the accounts of fiscal year t-1 and
the market equity of December t-1:
  - value:   four cheapness ratios, book value adjusted for intangible capital
  - quality: cash profitability adjusted for intangibles (organisation-capital
             investment added back, intangible capital added to assets)
Both ranked within FF12 industries, 50/50. Details in _qvi_plus.py.
Why: book value and earnings are wrong exactly for intangible-heavy firms (tech,
pharma), the firms where q5's ROE and investment factors, built on reported
numbers, misjudge them.
"""
import pandas as pd

from src.signals._qvi_plus import COMPUSTAT_COLS, CRSP_COLS, qvi_plus_year  # noqa: F401
from src.signals._yearly import to_rows

REBALANCE = "annual"
LONG, SHORT = 10, 1


def qvi_intan_full(panel: pd.DataFrame) -> pd.Series:
    return to_rows(panel, qvi_plus_year(panel))
