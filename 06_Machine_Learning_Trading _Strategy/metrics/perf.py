# metrics/perf.py
import numpy as np

def cagr(series):
    """
    Compute CAGR (annualized growth rate) from a series of monthly returns +1.
    """
    if len(series)==0:
        return np.nan
    n_years = len(series)/12
    return series.prod()**(1/n_years)-1

def ann_vol(returns):
    """
    Annualized volatility from monthly returns.
    """
    return returns.std(ddof=0)*np.sqrt(12)

def sharpe(returns, rf=0.0):
    """
    Sharpe ratio: (annual return - rf) / annual vol.

    rf is set to 0 because in this project all strategies are
    evaluated over the same period, and the risk-free rate has a negligible
    impact on the relative ranking. If needed, a non-zero annual rf
    can be used, but it does not change the conclusions.
    """
    v = ann_vol(returns)
    return np.nan if v==0 or np.isnan(v) else (cagr(1+returns)-rf)/v


def max_drawdown(cum):
    """
    Max drawdown from a cumulative curve.
    """
    peak=cum.cummax()
    dd=(cum/peak)-1
    return dd.min()

def rolling_sharpe(returns, window=36):
    """
    Approx rolling Sharpe over a window (in months).
    """
    m = returns.rolling(window).mean()
    s = returns.rolling(window).std(ddof=0)
    return (m / s) * np.sqrt(12)

def max_dd(cum_curve):
    """
    Same as max_drawdown: min relative drop from previous peaks.
    """
    peak = cum_curve.cummax()
    dd = (cum_curve / peak) - 1.0
    return dd.min()
