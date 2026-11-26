"""
Building block 3 of the final strategy (Table 5 of the paper): intangible-adjusted book value.

One change to quality_value_industry: in the book/price ratio of the cheapness score,
book value becomes the intangible-adjusted book of Eisfeldt, Kim and Papanikolaou (2022):
    book = be - goodwill + knowledge capital + organisation capital
Knowledge capital = past R&D, organisation capital = 30% of past SG&A (without R&D), both
accumulated with depreciation (15% and 20% a year), as in Peters and Taylor (2017); the
code is the team's src/signals/_intangibles.py. Missing goodwill counts as 0.

Why: accounting expenses R&D and brand-building, so firms built on intangibles (tech,
pharma) always look expensive on book value. Arnott, Harvey, Kalesnik and Linnainmaa (2021) and Eisfeldt, Kim and Papanikolaou (2022)
find that value works better, and more steadily, with intangibles added back.
Everything else is unchanged: E/P, CF/P and EBITDA/EV, cash profitability, FF12 industry
ranks, 50/50, long decile 10, short decile 1, annual rebalancing at the end of June.
""" 
import pandas as pd

from src.signals._intangibles import COMPUSTAT_COLS as _INT_COLS
from src.signals._yearly import to_rows
from src.signals.quality_value_industry import COMPUSTAT_COLS as _CORE_COLS
from src.signals.quality_value_industry import quality_value_industry_year

COMPUSTAT_COLS = sorted(set(_CORE_COLS) | set(_INT_COLS) | {"gdwl"})
CRSP_COLS = ["me_firm"]
REBALANCE = "annual"
LONG, SHORT = 10, 1


def quality_value_industry_intan(panel: pd.DataFrame) -> pd.Series:
    return to_rows(panel, quality_value_industry_year(panel, "ff12", intangible=True))
