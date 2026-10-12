"""Signal and portfolio construction for the two engines.

Both the research notebooks and the live notebooks import these functions,
so the portfolio traded on the paper account is exactly the one backtested.
Convention: weights are decided at month end t and earn the returns of month t+1.
"""
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


def momentum_weights(selected, mom_3m, base_cap=0.25, turbo_cap=0.40, turbo_trigger=0.05,
                     cash="SHV"):
    """Weights proportional to 3-month momentum. Cap is 25%, raised to 40% when the
    strongest asset is up more than 5% over 3 months. Unallocated weight goes to cash."""
    W = pd.DataFrame(0.0, index=selected.index, columns=selected.columns)

    for month in selected.index:
        winners = selected.loc[month][selected.loc[month] == 1].index.tolist()
        if not winners:
            W.loc[month, cash] = 1.0
            continue

        strength = mom_3m.loc[month, winners]
        cap = turbo_cap if strength.max() > turbo_trigger else base_cap
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
