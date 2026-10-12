"""Price download and monthly feature preparation."""
import datetime as dt
import os
from dataclasses import dataclass

import pandas as pd
import yfinance as yf


def download_prices(tickers, years=20, cache_path=None, verbose=True):
    """Daily adjusted close prices from Yahoo Finance, forward-filled over market holidays."""
    end = dt.date.today()
    start = end - dt.timedelta(days=years * 365)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
    prices = raw["Adj Close"].copy()

    if verbose:
        print(f"Downloaded {start} -> {end}. First valid date per ETF:")
        for t in tickers:
            first = prices[t].first_valid_index()
            print(f"  {t:<5} {first.date() if first is not None else 'no data'}")

    prices = prices.ffill()
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        prices.to_csv(cache_path)
    return prices


@dataclass
class MonthlyData:
    prices: pd.DataFrame          # month-end prices
    sma: pd.DataFrame             # daily SMA sampled at month end
    returns: pd.DataFrame         # monthly returns
    mom_3m: pd.DataFrame          # 3-month total return
    daily_returns: pd.DataFrame


def prepare_monthly(prices_daily, sma_window):
    """Month-end features. The last row is dropped if the month is not complete yet,
    so a signal is never computed on a partial month."""
    sma_d = prices_daily.rolling(window=sma_window, min_periods=sma_window).mean()
    daily_returns = prices_daily.pct_change()

    prices_m = prices_daily.resample("ME").last()
    sma_m = sma_d.resample("ME").last()
    if not prices_daily.index[-1].is_month_end:
        prices_m, sma_m = prices_m.iloc[:-1], sma_m.iloc[:-1]

    return MonthlyData(
        prices=prices_m,
        sma=sma_m,
        returns=prices_m.pct_change(periods=1),
        mom_3m=prices_m.pct_change(periods=3),
        daily_returns=daily_returns,
    )
