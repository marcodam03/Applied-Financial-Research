"""Benchmarks, factor models and significance tests used to check whether a strategy adds alpha.

All benchmarks follow the backtest convention: weights decided at the end of month t,
applied to the returns of month t+1. Benchmarks are shown without costs; strategies net of costs.
"""
import io
import math
import os
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

FF_URLS = {
    "ff5": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "mom": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip",
}


# ---------------------------------------------------------------------
# 1. Fama-French data (Kenneth French Data Library, monthly, decimals)
# ---------------------------------------------------------------------
def _parse_french_csv(text):
    """Read only the monthly section (YYYYMM rows) of a French Library CSV."""
    header, rows = None, []
    for line in text.splitlines():
        s = line.strip()
        if header is None:
            if s.startswith(",") and len(s) > 1:
                header = [h.strip() for h in s.split(",")[1:]]
            continue
        fields = [c.strip() for c in s.split(",")]
        if len(fields[0]) == 6 and fields[0].isdigit():
            rows.append(fields[:len(header) + 1])
        elif rows:                                   # annual section starts: stop
            break
    df = pd.DataFrame(rows, columns=["date"] + header)
    df.index = pd.to_datetime(df.pop("date"), format="%Y%m") + pd.offsets.MonthEnd(0)
    return df.astype(float) / 100


def load_fama_french(cache=os.path.join("..", "data", "ff_factors_monthly.csv")):
    """FF5 + momentum (UMD). Falls back to the cached copy if the download fails."""
    try:
        parts = []
        for url in FF_URLS.values():
            r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            parts.append(_parse_french_csv(z.read(z.namelist()[0]).decode("latin-1")))
        ff = parts[0].join(parts[1], how="inner")
        ff = ff.rename(columns={c: "UMD" for c in ff.columns if c.lower().startswith("mom")})
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        ff.to_csv(cache)
        source = "downloaded"
    except Exception as err:
        if not os.path.exists(cache):
            print(f"Fama-French data unavailable ({err}); factor regressions skipped.")
            return None
        ff = pd.read_csv(cache, index_col=0, parse_dates=True)
        source = f"cached copy (download failed: {err})"
    print(f"Fama-French: {ff.index[0]:%Y-%m} -> {ff.index[-1]:%Y-%m}, {source}")
    return ff


# ---------------------------------------------------------------------
# 2. Benchmark construction
# ---------------------------------------------------------------------
def returns_from_weights(W, R):
    W = W.reindex(index=R.index, columns=R.columns).fillna(0.0)
    active = W.shift(1).sum(axis=1) > 0
    return (W.shift(1) * R).sum(axis=1)[active]


def fixed_mix(dates, weights):
    """Static mix rebalanced monthly, e.g. {'SPY': 0.6, 'IEF': 0.4}."""
    return pd.DataFrame({k: v for k, v in weights.items()}, index=dates)


def equal_weight(prices_m, assets):
    """1/N over the ETFs available in each month."""
    avail = prices_m[assets].notna().astype(float)
    return avail.div(avail.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def naive_trend(prices_m, assets, months=10, cash="SHV"):
    """Textbook trend-following (Faber GTAA): each ETF gets 1/N if above its 10-month
    moving average, otherwise its slot goes to T-bills."""
    P = prices_m[assets]
    sma = P.rolling(months, min_periods=months).mean()
    n = sma.notna().sum(axis=1).replace(0, np.nan)
    W = (P > sma).astype(float).div(n, axis=0).fillna(0.0)
    W[cash] = np.where(n.notna(), 1 - W.sum(axis=1), 0.0)
    return W


def _erc_weights(cov, tol=1e-10, max_iter=1000):
    """Equal Risk Contribution via cyclical coordinate descent (Griveau-Billion et al., 2013)."""
    n = cov.shape[0]
    b = np.ones(n) / n
    y = 1 / np.sqrt(np.diag(cov))
    for _ in range(max_iter):
        y_old = y.copy()
        for i in range(n):
            c = cov[i] @ y - cov[i, i] * y[i]
            y[i] = (-c + np.sqrt(c * c + 4 * cov[i, i] * b[i])) / (2 * cov[i, i])
        if np.max(np.abs(y - y_old)) < tol:
            break
    return y / y.sum()


def erc(daily_returns, dates, assets, window=126):
    """Long-only ERC on the universe, covariance from the last 6 months of daily returns."""
    W = pd.DataFrame(0.0, index=dates, columns=assets)
    for d in dates:
        blk = daily_returns.loc[:d, assets].tail(window)
        if len(blk) < window:
            continue
        avail = blk.columns[blk.notna().sum() >= 0.9 * window]
        if len(avail) < 2:
            continue
        cov = blk[avail].dropna().cov().values * 252
        W.loc[d, avail] = _erc_weights(cov)
    return W


def tsmom_factor(prices_m, daily_returns, monthly_returns, assets, rf, target_vol=0.40):
    """Long-short time-series momentum factor, vol-scaled (Moskowitz, Ooi & Pedersen, 2012)."""
    sign = np.sign(prices_m[assets].pct_change(12))
    vol = daily_returns[assets].rolling(60, min_periods=50).std().resample("ME").last() * np.sqrt(252)
    W = sign * (target_vol / vol.reindex(sign.index))
    W = W.div(W.notna().sum(axis=1).replace(0, np.nan), axis=0)
    excess = monthly_returns[assets].sub(rf.reindex(monthly_returns.index), axis=0)
    return (W.shift(1) * excess).sum(axis=1, min_count=1).dropna()


# ---------------------------------------------------------------------
# 3. Statistics: OLS with Newey-West (HAC) errors, Sharpe difference test
# ---------------------------------------------------------------------
def _p_norm(t):
    return math.erfc(abs(t) / math.sqrt(2))


def ols_hac(y, X, lags=None):
    d = pd.concat([y.rename("_y"), X], axis=1).dropna()
    yv = d["_y"].values
    Xv = np.column_stack([np.ones(len(d)), d[X.columns].values])
    T, k = Xv.shape
    lags = int(np.floor(4 * (T / 100) ** (2 / 9))) if lags is None else lags
    XtX_inv = np.linalg.inv(Xv.T @ Xv)
    b = XtX_inv @ Xv.T @ yv
    e = yv - Xv @ b
    u = Xv * e[:, None]
    S = u.T @ u
    for lag in range(1, lags + 1):
        G = u[lag:].T @ u[:-lag]
        S += (1 - lag / (lags + 1)) * (G + G.T)
    V = XtX_inv @ S @ XtX_inv * T / (T - k)
    se = np.sqrt(np.diag(V))
    t = b / se
    tab = pd.DataFrame({"coef": b, "se": se, "t": t, "p": [_p_norm(x) for x in t]},
                       index=["alpha"] + list(X.columns))
    r2 = 1 - (e @ e) / ((yv - yv.mean()) @ (yv - yv.mean()))
    return tab, r2, T, lags


def sharpe_diff_pvalue(ra, rb, rf):
    """Jobson-Korkie test with Memmel (2003) correction. H0: equal Sharpe ratios."""
    d = pd.concat([ra - rf, rb - rf], axis=1).dropna()
    a, b = d.iloc[:, 0], d.iloc[:, 1]
    sa, sb, rho, T = a.mean() / a.std(), b.mean() / b.std(), a.corr(b), len(d)
    var = (2 - 2 * rho + 0.5 * (sa**2 + sb**2 - 2 * sa * sb * rho**2)) / T
    return _p_norm((sa - sb) / np.sqrt(var))


def perf_stats(r, rf):
    excess = r - rf
    years = len(r) / 12
    eq = (1 + r).cumprod()
    mdd = (eq / eq.cummax() - 1).min()
    cagr = eq.iloc[-1] ** (1 / years) - 1
    return {"CAGR": cagr, "Vol": r.std() * np.sqrt(12),
            "Sharpe": excess.mean() / excess.std() * np.sqrt(12),
            "MaxDD": mdd, "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan}


def _significance(t):
    a = abs(t)
    return ("significant at 1%" if a >= 2.58 else "significant at 5%" if a >= 1.96
            else "significant at 10%" if a >= 1.65 else "not significant")


# ---------------------------------------------------------------------
# 4. Reports
# ---------------------------------------------------------------------
def check_weights(W):
    active = W[W.sum(axis=1) > 0].sum(axis=1)
    wrong = active[(active - 1).abs() > 0.01]
    if len(wrong):
        print(f"WARNING: {len(wrong)} of {len(active)} months have weights not summing to 100% "
              f"(min {active.min():.0%}); the missing part earns zero in the backtest.\n")
    else:
        print("Weights check: every month sums to 100%.\n")


def align_sample(strat, benchmarks, rf, weights):
    """Common sample: from the first month the strategy takes risk, while all series exist."""
    first = weights.drop(columns="SHV", errors="ignore").sum(axis=1).gt(0).idxmax()
    idx = strat.dropna().index
    idx = idx[idx > first].intersection(rf.dropna().index)
    for b in benchmarks.values():
        idx = idx.intersection(b.dropna().index)
    print(f"Common sample: {idx[0]:%Y-%m} -> {idx[-1]:%Y-%m} ({len(idx)} months)\n")
    return strat.loc[idx], {k: v.loc[idx] for k, v in benchmarks.items()}, rf.loc[idx]


def benchmark_comparison(strat, benchmarks, rf, name):
    rows, verdicts = {name: perf_stats(strat, rf)}, []
    for b_name, b in benchmarks.items():
        active = strat - b
        te = active.std() * np.sqrt(12)
        reg, _, _, _ = ols_hac(strat - rf, (b - rf).to_frame("bench"))
        rows[b_name] = {**perf_stats(b, rf),
                        "Active": active.mean() * 12, "TE": te, "IR": active.mean() * 12 / te,
                        "Beta": reg.loc["bench", "coef"], "Alpha": reg.loc["alpha", "coef"] * 12,
                        "t(α)": reg.loc["alpha", "t"], "p(ΔSR)": sharpe_diff_pvalue(strat, b, rf)}
        verdicts.append(f"  vs {b_name:<26} alpha {reg.loc['alpha', 'coef']*12:+6.2%} p.a. "
                        f"(t = {reg.loc['alpha', 't']:5.2f}) -> {_significance(reg.loc['alpha', 't'])}")
    tab = pd.DataFrame(rows).T
    fmt = {c: "{:.2%}" for c in ["CAGR", "Vol", "MaxDD", "Active", "TE", "Alpha"]}
    fmt.update({c: "{:.2f}" for c in ["Sharpe", "Calmar", "IR", "Beta", "t(α)", "p(ΔSR)"]})
    out = tab.copy().astype(object)
    for c, f in fmt.items():
        out[c] = [("" if pd.isna(v) else f.format(v)) for v in tab[c]]

    print(f"BENCHMARK COMPARISON: {name}")
    print(out.to_string())
    print("Active = strategy - benchmark (ann.) | TE = tracking error | IR = Active / TE | "
          "Alpha, Beta from regressing strategy excess returns on benchmark excess returns (Newey-West t)")
    print("p(ΔSR) = p-value of the Sharpe difference (Jobson-Korkie / Memmel). "
          "Benchmarks have no costs; the strategy is net of costs.\n")
    print("Beta-adjusted alpha:")
    print("\n".join(verdicts) + "\n")
    return tab


def factor_regression(strat, factors, rf, title):
    y = strat - rf.reindex(strat.index)
    reg, r2, T, lags = ols_hac(y, factors.reindex(strat.index))
    out = pd.DataFrame({
        "Coefficient": [f"{reg.loc['alpha', 'coef']*12:+.2%} p.a."] +
                       [f"{v:+.3f}" for v in reg["coef"].iloc[1:]],
        "t-stat (NW)": [f"{v:.2f}" for v in reg["t"]],
        "p-value": [f"{v:.3f}" for v in reg["p"]],
    }, index=["Alpha"] + list(factors.columns))
    print(f"FACTOR MODEL: {title}")
    print(out.to_string())
    print(f"R² = {r2:.2f} | {T} months | Newey-West lags = {lags}")
    print(f"Alpha {reg.loc['alpha', 'coef']*12:+.2%} p.a., {_significance(reg.loc['alpha', 't'])}\n")
    return reg, r2


def rolling_alpha(strat, factors, rf, window=36):
    d = pd.concat([strat - rf.reindex(strat.index), factors.reindex(strat.index)], axis=1).dropna()
    out = {}
    for i in range(window, len(d) + 1):
        blk = d.iloc[i - window:i]
        X = np.column_stack([np.ones(window), blk.iloc[:, 1:].values])
        out[d.index[i - 1]] = np.linalg.lstsq(X, blk.iloc[:, 0].values, rcond=None)[0][0] * 12
    return pd.Series(out)


def _log_axis(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(plt.matplotlib.ticker.LogLocator(base=10, subs=(1, 1.5, 2, 3, 5, 7)))
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter("%.0f"))
    ax.yaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())


def equity_curve(r):
    """Index starting at 100 the month before the first return."""
    c = (1 + r).cumprod() * 100
    return pd.concat([pd.Series([100.0], index=[r.index[0] - pd.offsets.MonthEnd(1)]), c])


def plot_alpha(strat, benchmarks, name, alpha_roll, model_name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 2]})
    ax1.plot(equity_curve(strat), color="black", lw=2.5, label=name)
    for b_name, b in benchmarks.items():
        ax1.plot(equity_curve(b), lw=1.4, alpha=0.85, label=b_name)
    _log_axis(ax1)
    ax1.set_title(f"{name} vs benchmarks (log scale, base 100)", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)
    ax2.plot(alpha_roll, color="#00441b", lw=2)
    ax2.axhline(0, color="grey", lw=1)
    ax2.fill_between(alpha_roll.index, alpha_roll, 0, where=alpha_roll > 0, color="#2E7D32", alpha=0.2)
    ax2.fill_between(alpha_roll.index, alpha_roll, 0, where=alpha_roll < 0, color="#C62828", alpha=0.2)
    ax2.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    ax2.set_title(f"Rolling 36-month alpha (annualised), {model_name}", fontsize=12, fontweight="bold")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
