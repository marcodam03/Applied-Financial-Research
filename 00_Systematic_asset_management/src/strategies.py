"""Signal and portfolio construction for the two engines.

Both the research notebooks and the live notebooks import these functions,
so the portfolio traded on the paper account is exactly the one backtested.
Convention: weights are decided at month end t and earn the returns of month t+1.
"""
import numpy as np
import pandas as pd


def trend_momentum_signal(md):
    """Eligible if month-end price > daily SMA and 3-month return > 0."""
    return (md.prices > md.sma) & (md.mom_3m > 0)


def _apply_cap(weights, cap, max_iter):
    """Cap each weight and redistribute the excess equally across uncapped names."""
    w = weights.copy()
    for _ in range(max_iter):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = w < cap
        if under.any():
            w[under] += excess / under.sum()
        else:
            break
    return w


# ---------------------------------------------------------------------
# SAFE: multi-asset dual momentum
# ---------------------------------------------------------------------
def correlation_filter(signal, mom_3m, daily_returns, max_assets=5, corr_threshold=0.80,
                       lookback_days=180):
    """Among eligible assets, drop the weaker-momentum asset of every pair whose
    daily-return correlation over the lookback exceeds the threshold; keep the top N."""
    selected = pd.DataFrame(0, index=signal.index, columns=signal.columns)

    for month in signal.index:
        row = signal.loc[month]
        eligible = row[row == True].index.tolist()  # noqa: E712
        if not eligible:
            continue

        mom = mom_3m.loc[month, eligible].dropna()
        eligible = mom.index.tolist()

        if len(eligible) > 1:
            window = daily_returns.loc[month - pd.Timedelta(days=lookback_days):month, eligible]
            corr = window.corr()
            dropped = set()
            for i in range(len(eligible)):
                for j in range(i + 1, len(eligible)):
                    a, b = eligible[i], eligible[j]
                    if corr.loc[a, b] > corr_threshold and a not in dropped and b not in dropped:
                        dropped.add(b if mom[a] > mom[b] else a)
            eligible = [e for e in eligible if e not in dropped]

        winners = mom[eligible].sort_values(ascending=False).head(max_assets).index.tolist()
        selected.loc[month, winners] = 1

    return selected


def _apply_caps(weights, caps, max_iter=10):
    """Like _apply_cap but with a different cap for each asset (Series aligned on weights)."""
    w = weights.copy()
    for _ in range(max_iter):
        over = w > caps + 1e-12
        if not over.any():
            break
        excess = (w[over] - caps[over]).sum()
        w[over] = caps[over]
        under = w < caps - 1e-12
        if under.any():
            w[under] += excess / under.sum()
        else:
            break
    return w


def momentum_weights(selected, mom_3m, base_cap=0.25, turbo_cap=0.40, turbo_trigger=0.05,
                     cash="SHV", turbo_min_assets=None, turbo_top_only=False):
    """Weights proportional to 3-month momentum. Cap is 25%, raised to 40% when the
    strongest asset is up more than 5% over 3 months. Unallocated weight goes to cash.
    v2 option: turbo_min_assets=3 allows the 40% cap only when at least 3 risky assets
    are selected, so two ETFs alone can never reach 80% of the portfolio."""
    W = pd.DataFrame(0.0, index=selected.index, columns=selected.columns)

    for month in selected.index:
        winners = selected.loc[month][selected.loc[month] == 1].index.tolist()
        if not winners:
            W.loc[month, cash] = 1.0
            continue

        strength = mom_3m.loc[month, winners]
        turbo_ok = (turbo_min_assets is None
                    or len([a for a in winners if a != cash]) >= turbo_min_assets)
        cap = turbo_cap if (strength.max() > turbo_trigger and turbo_ok) else base_cap
        if turbo_top_only and cap > base_cap:
            # v2b: only the strongest asset may reach the turbo cap, the others stay at base_cap
            caps = pd.Series(base_cap, index=strength.index)
            caps[strength.idxmax()] = turbo_cap
            w = _apply_caps(strength / strength.sum(), caps)
        else:
            w = _apply_cap(strength / strength.sum(), cap, max_iter=5)
        for etf in winners:
            W.loc[month, etf] = w[etf]

        residual = 1.0 - W.loc[month].sum()
        if residual > 0.001:
            W.loc[month, cash] += residual

    return W


def inverse_vol_weights(selected, daily_returns, max_weight=0.25, vol_window=60, cash="SHV"):
    """Alternative sizing: inverse 60-day volatility, 25% cap, residual to cash."""
    vol_m = daily_returns.rolling(window=vol_window, min_periods=vol_window).std().resample("ME").last()
    W = pd.DataFrame(0.0, index=selected.index, columns=selected.columns)

    for month in selected.index:
        winners = selected.loc[month][selected.loc[month] == 1].index.tolist()
        if not winners:
            W.loc[month, cash] = 1.0
            continue

        inv_vol = 1.0 / vol_m.loc[month, winners]
        w = _apply_cap(inv_vol / inv_vol.sum(), max_weight, max_iter=10)
        for etf in winners:
            W.loc[month, etf] = w[etf]

        residual = 1.0 - W.loc[month].sum()
        if residual > 0.001:
            W.loc[month, cash] += residual

    return W


# ---------------------------------------------------------------------
# SAFE v2: portfolio-level risk controls on top of the v1 weights
# ---------------------------------------------------------------------
SAFE_EQUITY = ["SPY", "QQQ", "IWM", "VWO"]


def risk_controlled_weights(weights, daily_returns, target_vol=0.10, max_equity=0.60,
                            equity=SAFE_EQUITY, vol_lookback=63, cash="SHV"):
    """Apply two portfolio-level limits at each month end, moving the excess to cash:
    1. total equity weight <= max_equity (scaled proportionally across equity ETFs);
    2. ex-ante volatility of the risky sleeve, from the last `vol_lookback` daily returns,
       <= target_vol (all risky weights scaled down; never levered up).
    Uses only data available at the month end, like the v1 signals."""
    W = weights.copy()
    eq = [c for c in equity if c in W.columns]
    for month in W.index:
        w = W.loc[month].copy()

        eq_sum = w[eq].sum()
        if max_equity is not None and eq_sum > max_equity:
            w[eq] *= max_equity / eq_sum

        risky = [c for c in w.index if c != cash and w[c] > 0]
        if target_vol is not None and risky:
            hist = daily_returns.loc[:month, risky].tail(vol_lookback).dropna()
            if len(hist) >= vol_lookback * 0.8:
                vol = float(np.sqrt(w[risky].values @ hist.cov().values @ w[risky].values * 252))
                if vol > target_vol:
                    w[risky] *= target_vol / vol

        w[cash] = 0.0
        w[cash] = max(0.0, 1.0 - w.sum())
        W.loc[month] = w
    return W


def safe_v2_weights(md, max_assets=5, corr_threshold=0.80, base_cap=0.25, turbo_cap=0.40,
                    turbo_trigger=0.05, turbo_min_assets=3, target_vol=0.10, max_equity=0.60):
    """Safe v2 = v1 selection + turbo only with >= 3 risky assets + equity cap + vol target.
    Parameters fixed a priori (10% is roughly the v1 long-run volatility, 60% the equity share
    of a 60/40), not optimised on the backtest."""
    signal = trend_momentum_signal(md)
    selected = correlation_filter(signal, md.mom_3m, md.daily_returns, max_assets, corr_threshold)
    w = momentum_weights(selected, md.mom_3m, base_cap, turbo_cap, turbo_trigger,
                         turbo_min_assets=turbo_min_assets)
    return risk_controlled_weights(w, md.daily_returns, target_vol=target_vol, max_equity=max_equity)


def safe_v2b_weights(md, max_assets=5, corr_threshold=0.80, base_cap=0.25, turbo_cap=0.40,
                     turbo_trigger=0.05, max_equity=0.60):
    """Safe v2b = v1 selection + only the strongest asset can reach 40% (others 25%)
    + total equity <= 60%. No volatility target: every limit is a simple, observable weight."""
    signal = trend_momentum_signal(md)
    selected = correlation_filter(signal, md.mom_3m, md.daily_returns, max_assets, corr_threshold)
    w = momentum_weights(selected, md.mom_3m, base_cap, turbo_cap, turbo_trigger, turbo_top_only=True)
    return risk_controlled_weights(w, md.daily_returns, target_vol=None, max_equity=max_equity)


def safe_v3_weights(md, max_assets=5, corr_threshold=0.80, max_weight=0.25, max_equity=0.60):
    """Safe v3 (idea under test) = v1 selection + inverse-volatility weights (25% cap)
    + total equity <= 60%. Sizes positions by risk instead of by momentum."""
    signal = trend_momentum_signal(md)
    selected = correlation_filter(signal, md.mom_3m, md.daily_returns, max_assets, corr_threshold)
    w = inverse_vol_weights(selected, md.daily_returns, max_weight=max_weight)
    return risk_controlled_weights(w, md.daily_returns, target_vol=None, max_equity=max_equity)


# ---------------------------------------------------------------------
# ALPHA CHASER: concentrated equity momentum
# ---------------------------------------------------------------------
def alpha_chaser_weights(signal, mom_3m, top_n=2, buffer=0.10, max_weight=0.50, cash="SHV"):
    """Hold the top-N eligible ETFs by 3-month momentum, equally weighted.
    Current holdings get a momentum bonus (hysteresis) to cut turnover.
    One eligible ETF: 50% in it, 50% cash. None eligible: 100% cash."""
    W = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    previous = []

    for month in signal.index:
        eligible = [c for c in signal.columns[signal.loc[month]] if c != cash]
        if not eligible:
            W.loc[month, cash] = 1.0
            previous = [cash]
            continue

        strength = mom_3m.loc[month, eligible].copy()
        for asset in previous:
            if asset in strength.index:
                strength[asset] += buffer
        top = strength.nlargest(top_n)

        if len(top) == 1:
            W.loc[month, top.index[0]] = max_weight
            W.loc[month, cash] = 1.0 - max_weight
        else:
            W.loc[month, top.index] = 1.0 / len(top)
        previous = top.index.tolist()

    return W
