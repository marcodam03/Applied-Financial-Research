"""
Laggard catch-up: news about an industry reaches the prices of its large, closely
followed firms first and the smaller firms later. A firm that lagged its
industry leaders this month tends to catch up next month.

Inspiration: Hou (2007) "Industry information diffusion and the lead-lag effect
in stock returns"; Moskowitz and Grinblatt (1999) "Do industries explain
momentum?"; Cohen and Frazzini (2008) "Economic links and predictable returns".
Hou works with weekly, portfolio-level data; this is a stock-level monthly
version.

In the row of month t (portfolio held in t+1), for each industry (SIC 2 digits)
with at least MIN_FIRMS stocks:
  leaders    = the largest third of the industry by market cap at the end of t-1
  leader_ret = their value-weighted return in month t
  signal     = leader_ret - the stock's own return in month t
               (only for the other two thirds: the followers)
Long the followers that lagged their leaders most, short those that ran ahead.

Caveat for the report: the signal contains minus the stock's own last-month
return, i.e. part of the known short-term reversal effect. The standard benchmark
models have no reversal factor, so check the FF6+STR and q5+STR alphas (they add
Ken French's short-term reversal factor, see src/evaluation/alpha.py).
"""
import numpy as np
import pandas as pd

COMPUSTAT_COLS = []
REBALANCE = "monthly"
LONG, SHORT = 10, 1
MIN_FIRMS = 10


def laggard(panel: pd.DataFrame) -> pd.Series:
    p = panel[["PERMNO", "MthCalDt", "MthRet", "mkt_cap", "SICCD"]].copy()
    mid = p["MthCalDt"].dt.year * 12 + p["MthCalDt"].dt.month
    prev_cap = p.groupby("PERMNO")["mkt_cap"].shift(1)
    prev_mid = mid.groupby(p["PERMNO"]).shift(1)
    p["w"] = prev_cap.where((mid - prev_mid == 1) & (prev_cap > 0))  # cap at the end of t-1
    p["sic2"] = pd.to_numeric(p["SICCD"], errors="coerce") // 100
    p = p[p["w"].notna() & p["MthRet"].notna() & p["sic2"].notna()]

    grp = [p["MthCalDt"], p["sic2"]]
    n = p.groupby(grp)["w"].transform("count")
    is_leader = p.groupby(grp)["w"].rank(pct=True) > 2 / 3
    wr = (p["w"] * p["MthRet"]).where(is_leader, 0.0)
    wl = p["w"].where(is_leader, 0.0)
    leader_ret = wr.groupby(grp).transform("sum") / wl.groupby(grp).transform("sum")

    signal = (leader_ret - p["MthRet"]).where((n >= MIN_FIRMS) & ~is_leader)
    return signal.reindex(panel.index)