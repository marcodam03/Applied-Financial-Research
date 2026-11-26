"""
Dividend month premium: stocks earn higher returns in the months in which they
are expected to pay a dividend (dividend-seeking investors buy around the
payment). Payment months are predictable: firms keep a regular calendar.

Source: Hartzmark and Solomon (2013) "The dividend month premium".
Our twist: the short side is not "the market" but other regular dividend payers
that are NOT expected to pay that month, so long and short are similar firms
(both dividend payers) and the spread is a clean calendar effect, not a
value/size/profitability tilt.

In the row of month t (portfolio held in t+1), among regular payers (at least one
dividend in months t-11..t, 12 consecutive months of data):
  signal = 1 if the stock paid a dividend in month t-11 (12 months before the
           holding month, so a payment is expected in t+1), else 0
Two portfolios: 2 = expected payers (long), 1 = other payers (short).
Dividend = CRSP DisDivAmt > 0 in the month.
"""
import pandas as pd

COMPUSTAT_COLS = []
CRSP_COLS = ["DisDivAmt"]
REBALANCE = "monthly"
N_PORTFOLIOS = 2           # binary signal: overrides the config's n_portfolios
LONG, SHORT = 2, 1


def dividend_month(panel: pd.DataFrame) -> pd.Series:
    stock = panel["PERMNO"]
    mid = panel["MthCalDt"].dt.year * 12 + panel["MthCalDt"].dt.month
    paid = (pd.to_numeric(panel["DisDivAmt"], errors="coerce").fillna(0) > 0).astype(float)

    paid_12m = paid.groupby(stock).rolling(12, min_periods=12).sum().droplevel(0)
    paid_t11 = paid.groupby(stock).shift(11)
    consecutive = (mid - mid.groupby(stock).shift(11)) == 11   # rows t-11..t are 12 straight months
    regular = consecutive & (paid_12m > 0)
    return paid_t11.where(regular)
