"""Backtest with transaction costs, allocation statistics, stress tests and sub-period checks."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def perf_summary(r, rf=0.02):
    """CAGR, volatility, simplified Sharpe ((CAGR - rf) / vol) and max drawdown from monthly returns."""
    years = len(r) / 12
    cagr = (1 + r).prod() ** (1 / years) - 1
    vol = r.std() * np.sqrt(12)
    eq = (1 + r).cumprod()
    max_dd = (eq / eq.cummax() - 1).min()
    sharpe = (cagr - rf) / vol if vol > 0 else 0.0
    return {"CAGR": f"{cagr*100:.2f}%", "Volatility": f"{vol*100:.2f}%",
            "Sharpe": f"{sharpe:.2f}", "Max DD": f"{max_dd*100:.1f}%"}


def run_backtest(weights, monthly_returns, ticker_costs, name, benchmark="SPY",
                 cost_low=0.0001, cost_high=0.0010, rf=0.02, colors=("#238b45", "#00441b", "#c7e9c0"),
                 plot=True):
    """Gross and net returns. Costs: a 1-10 bps uncertainty band on total turnover, plus a
    'realistic' case with a per-ETF one-way cost. Weights are lagged one month (no look-ahead)."""
    first = weights[weights.sum(axis=1) > 0].index[0]
    gross = (weights.shift(1) * monthly_returns).sum(axis=1).loc[first:]
    bench = monthly_returns[benchmark].loc[first:]

    turnover = weights.diff().abs().sum(axis=1)
    turnover.loc[first] = 1.0                                   # initial purchase
    net_low = gross - (turnover.shift(1) * cost_low).fillna(0).loc[first:]
    net_high = gross - (turnover.shift(1) * cost_high).fillna(0).loc[first:]

    turnover_by_etf = weights.diff().abs()
    turnover_by_etf.loc[first] = weights.loc[first]
    realistic_cost = (turnover_by_etf * pd.Series(ticker_costs)).sum(axis=1).shift(1).fillna(0).loc[first:]
    net = gross - realistic_cost

    stats = pd.DataFrame({
        "Gross (no costs)": perf_summary(gross, rf),
        "Net (per-ETF costs)": perf_summary(net, rf),
        f"Benchmark ({benchmark})": perf_summary(bench, rf),
    }).T
    print(f"PERFORMANCE: {name}  ({first:%Y-%m} -> {gross.index[-1]:%Y-%m}, Sharpe uses rf = {rf:.0%})")
    print(stats.to_string(), "\n")

    if plot:
        curve = lambda r: (1 + r).cumprod() * 100  # noqa: E731
        plt.figure(figsize=(14, 7))
        sns.set_style("whitegrid")
        plt.fill_between(net_low.index, curve(net_high), curve(net_low), color=colors[2], alpha=0.4,
                         label="Cost range (1-10 bps on turnover)")
        plt.plot(curve(gross), label=f"{name} (gross)", color=colors[0], lw=1, ls="--")
        plt.plot(curve(net), label=f"{name} (net)", color=colors[1], lw=2.5)
        plt.plot(curve(bench), label=f"{benchmark}", color="#1f77b4", lw=1.8, alpha=0.7)
        plt.title(f"{name}: backtest net of costs", fontsize=15, fontweight="bold")
        plt.ylabel("Value (base 100)")
        plt.legend(loc="upper left")
        plt.show()

    return {"gross": gross, "net": net, "net_low": net_low, "net_high": net_high,
            "benchmark": bench, "stats": stats}


def allocation_stats(weights, save_path=None):
    """How often each ETF is held and at what weight."""
    first = weights[weights.sum(axis=1) > 0].index[0]
    w = weights.loc[first:]
    out = pd.DataFrame(index=w.columns)
    out["Months held"] = (w > 0).sum()
    out["Frequency %"] = (out["Months held"] / len(w) * 100).round(1)
    out["Avg weight when held %"] = (w[w > 0].mean() * 100).round(1)
    out["Max weight %"] = (w.max() * 100).round(1)
    out = out.sort_values("Months held", ascending=False)

    print(f"ALLOCATION HISTORY: {len(w)} months since {first.date()}")
    print(out.to_string(), "\n")
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        out.to_csv(save_path)
    return out


def stress_test(weights, monthly_returns, name, scenarios, unmapped_shock=-0.05, n_paths=1000,
                horizon=12):
    """(1) Deterministic shocks applied to the average portfolio.
    (2) 12-month bootstrap of the average portfolio's monthly returns: VaR 95% / 99%."""
    avg_w = weights[weights.sum(axis=1) > 0].mean()

    impacts = {}
    for scen, shocks in scenarios.items():
        impact = sum(avg_w.get(a, 0) * s for a, s in shocks.items())
        unmapped = [a for a in avg_w.index if a not in shocks and avg_w[a] > 0.01]
        impact += sum(avg_w[a] * unmapped_shock for a in unmapped)
        impacts[scen] = impact * 100

    port = (monthly_returns * avg_w).sum(axis=1).dropna()
    paths = np.array([np.cumprod(1 + np.random.choice(port, size=horizon, replace=True)) * 100
                      for _ in range(n_paths)])
    var95 = 100 - np.percentile(paths[:, -1], 5)
    var99 = 100 - np.percentile(paths[:, -1], 1)

    print(f"STRESS TEST: {name}")
    print(f"{'Scenario':<32} | {'Impact on average portfolio':>28}")
    for scen, imp in impacts.items():
        print(f"{scen:<32} | {imp:>27.2f}%")

    plt.figure(figsize=(13, 6))
    plt.plot(paths.T, color="#d62728", alpha=0.03)
    plt.plot(np.percentile(paths, 50, axis=0), color="#1f77b4", lw=2.5, label="Median path")
    plt.plot(np.percentile(paths, 5, axis=0), color="#ff7f0e", ls="--", lw=2, label="5th percentile")
    plt.plot(np.percentile(paths, 1, axis=0), color="#8c564b", ls="--", lw=2, label="1st percentile")
    plt.title(f"{name}: 12-month bootstrap of monthly returns", fontsize=14, fontweight="bold")
    plt.xlabel("Months ahead")
    plt.ylabel("Value (base 100)")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.show()

    print(f"12-month VaR 95%: {var95:.2f}%   |   12-month VaR 99%: {var99:.2f}%\n")
    return impacts, var95, var99


def subperiod_table(weights, monthly_returns, years=(10, 5, 3), benchmark="SPY", rf=0.02):
    """Gross performance over the last N years versus the benchmark."""
    def metrics(r):
        cagr = (1 + r).prod() ** (12 / len(r)) - 1
        vol = r.std() * np.sqrt(12)
        dd = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()
        return cagr, (cagr - rf) / vol, dd

    rows = []
    for y in years:
        start = weights.index[-1] - pd.DateOffset(years=y)
        strat = (weights.loc[start:].shift(1) * monthly_returns.loc[start:]).sum(axis=1).dropna()
        bench = monthly_returns.loc[strat.index, benchmark]
        s_cagr, s_sharpe, s_dd = metrics(strat)
        b_cagr, _, _ = metrics(bench)
        rows.append({"Period": f"Last {y}Y", "Strategy CAGR": f"{s_cagr*100:.2f}%",
                     f"{benchmark} CAGR": f"{b_cagr*100:.2f}%",
                     "Excess CAGR": f"{(s_cagr - b_cagr)*100:.2f}%",
                     "Sharpe": f"{s_sharpe:.2f}", "Max DD": f"{s_dd*100:.1f}%"})
    out = pd.DataFrame(rows)
    print("SUB-PERIODS (gross of costs)")
    print(out.to_string(index=False), "\n")
    return out
