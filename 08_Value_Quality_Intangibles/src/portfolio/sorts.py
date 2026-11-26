"""
Univariate portfolio sorts, following the method of the Applied Investments slides.

How it works, in plain words:
1. At each formation date, keep the stocks that have a valid signal and a
   positive market cap on that date (and pass the optional universe filters).
2. Compute breakpoints (e.g. deciles) using NYSE stocks only.
3. Put every stock (NYSE, AMEX, NASDAQ) into a portfolio: stocks with a signal
   <= the first breakpoint go in portfolio 1, and so on.
4. Hold the portfolios until the next formation date and compute monthly
   equal-weighted and value-weighted returns. Value weights use the market cap
   at the end of the previous month (for July: the June market cap).

Two rebalancing modes:
- "annual": sort at the end of June of year t, hold July t to June t+1.
  Use this for accounting signals. The signal must be constant within each
  July-June year (the check below enforces it, to avoid look-ahead bias).
- "monthly": sort at the end of every month using that month's signal, hold
  the next month. Use this for market signals (momentum, volatility, ...).
  The signal in a row of month m must be known at the end of month m.
  hold_months = k > 1: sort only every k months (end of March, June, September
  and December for k = 3) and hold the same stocks for the next k months:
  fewer trades, but the signal gets older within the k months.

Optional universe filters (applied at each formation date, all off by default):
- min_price:          drop stocks with price below this value (e.g. 5)
- min_nyse_size_pct:  drop stocks smaller than this NYSE market-cap percentile (e.g. 0.2)
- exclude_sic:        list of (low, high) SIC ranges to drop, e.g. FINANCIALS + UTILITIES
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLS = ["PERMNO", "MthCalDt", "formation_year", "PrimaryExch", "MthRet", "mkt_cap"]
OPTIONAL_COLS = ["MthPrc", "SICCD"]  # needed only by the filters

FINANCIALS = [(6000, 6999)]
UTILITIES = [(4900, 4999)]


def prepare_panel(panel: pd.DataFrame, signal: str) -> pd.DataFrame:
    """Keep the needed columns, sort, and add the lagged market cap (weights)."""
    missing = [c for c in REQUIRED_COLS + [signal] if c not in panel.columns]
    if missing:
        raise KeyError(f"Columns missing from the panel: {missing}")

    optional = [c for c in OPTIONAL_COLS if c in panel.columns]
    df = panel[REQUIRED_COLS + optional + [signal]].copy()
    df["MthCalDt"] = pd.to_datetime(df["MthCalDt"])
    df = df.sort_values(["PERMNO", "MthCalDt"]).reset_index(drop=True)

    # A simple month counter (Jan 1963 -> 23556, Feb 1963 -> 23557, ...)
    df["month_id"] = df["MthCalDt"].dt.year * 12 + df["MthCalDt"].dt.month - 1

    # Market cap at the end of the PREVIOUS month, only if that month exists
    prev_month = df.groupby("PERMNO")["month_id"].shift(1)
    prev_mcap = df.groupby("PERMNO")["mkt_cap"].shift(1)
    df["mcap_lag"] = np.where(df["month_id"] - prev_month == 1, prev_mcap, np.nan)
    return df


def _formation_snapshot(df: pd.DataFrame, signal: str, rebalance: str,
                        hold_months: int = 1) -> pd.DataFrame:
    """One row per stock and formation date, with the value used for sorting.
    Also sets df["form_key"]: which formation date each holding month belongs to."""
    info_cols = ["PERMNO", "MthCalDt", "PrimaryExch", "mkt_cap"] + [
        c for c in OPTIONAL_COLS if c in df.columns
    ]
    if rebalance == "annual":
        # Look-ahead protection: the signal must not change within July-June
        n_values = (
            df.dropna(subset=[signal]).groupby(["PERMNO", "formation_year"])[signal].nunique()
        )
        if (n_values > 1).any():
            raise ValueError(
                f"'{signal}' changes within a July-June year for {int((n_values > 1).sum())} "
                "stock-years. Annual rebalancing needs a signal fixed within the year "
                "(accounting data). Use rebalance='monthly' for market-based signals."
            )
        # Market cap, exchange, price, SIC measured at the end of June of year t
        snap = df.loc[df["MthCalDt"].dt.month == 6, info_cols].copy()
        snap["form_key"] = snap["MthCalDt"].dt.year
        sig = (
            df.dropna(subset=[signal])
            .groupby(["PERMNO", "formation_year"])[signal].first()
            .rename("sort_value").reset_index()
            .rename(columns={"formation_year": "form_key"})
        )
        snap = snap.merge(sig, on=["PERMNO", "form_key"], how="inner")
        df["form_key"] = df["formation_year"]

    elif rebalance == "monthly":
        snap = df[info_cols + [signal, "month_id"]].copy()
        snap = snap.rename(columns={signal: "sort_value", "month_id": "form_key"})
        # hold in month m what was formed at the last formation date before m
        # (every month if hold_months = 1: formed in m-1)
        k = int(hold_months)
        snap = snap[(snap["form_key"] + 1) % k == 0]
        df["form_key"] = df["month_id"] - 1 - df["month_id"] % k

    else:
        raise ValueError("rebalance must be 'annual' or 'monthly'")

    ok = (snap["mkt_cap"] > 0) & np.isfinite(snap["sort_value"])
    return snap[ok].copy()


def _apply_filters(
    snap: pd.DataFrame,
    breakpoint_exchanges: Sequence[str],
    min_price: Optional[float],
    min_nyse_size_pct: Optional[float],
    exclude_sic: Optional[List[Tuple[int, int]]],
) -> pd.DataFrame:
    """Universe filters at the formation date (no filter = no change)."""
    if min_nyse_size_pct is not None:
        nyse = snap[snap["PrimaryExch"].isin(breakpoint_exchanges)]
        cut = nyse.groupby("form_key")["mkt_cap"].quantile(min_nyse_size_pct)
        snap = snap[snap["mkt_cap"] >= snap["form_key"].map(cut)]
    if min_price is not None:
        if "MthPrc" not in snap.columns:
            raise KeyError("min_price needs the column 'MthPrc' in the panel")
        snap = snap[snap["MthPrc"] >= min_price]
    if exclude_sic:
        if "SICCD" not in snap.columns:
            raise KeyError("exclude_sic needs the column 'SICCD' in the panel")
        drop = pd.Series(False, index=snap.index)
        for low, high in exclude_sic:
            drop |= snap["SICCD"].between(low, high)
        snap = snap[~drop]
    return snap


def _apply_buffer(snap: pd.DataFrame, n_portfolios: int, buffer: int, step: int = 1) -> np.ndarray:
    """Portfolio numbers after the buffer rule (see sort_portfolios). Only stocks
    held in an extreme portfolio at the previous formation date (form_key - step)
    can be kept there."""
    if not 0 < buffer < n_portfolios / 2:
        raise ValueError("buffer must be between 1 and n_portfolios/2")
    port = snap["port"].to_numpy().copy()
    ids = snap["PERMNO"].to_numpy()
    top, bottom = n_portfolios, 1
    prev_key, prev_top, prev_bottom = None, np.array([]), np.array([])
    for key, idx in sorted(snap.groupby("form_key").indices.items()):
        if prev_key is not None and key == prev_key + step:
            stay_top = np.isin(ids[idx], prev_top) & (port[idx] >= top - buffer)
            stay_bottom = np.isin(ids[idx], prev_bottom) & (port[idx] <= bottom + buffer)
            port[idx[stay_top]] = top
            port[idx[stay_bottom]] = bottom
        prev_top, prev_bottom = ids[idx][port[idx] == top], ids[idx][port[idx] == bottom]
        prev_key = key
    return port


def sort_portfolios(
    panel: pd.DataFrame,
    signal: str,
    n_portfolios: int = 10,
    rebalance: str = "annual",
    breakpoint_exchanges: Sequence[str] = ("N",),
    min_price: Optional[float] = None,
    min_nyse_size_pct: Optional[float] = None,
    exclude_sic: Optional[List[Tuple[int, int]]] = None,
    return_holdings: bool = False,
    buffer: int = 0,
    hold_months: int = 1,
) -> Dict[str, pd.DataFrame]:
    """Build portfolios sorted on `signal`.

    buffer > 0 lowers turnover in the two extreme portfolios (Novy-Marx and
    Velikov 2016, "A taxonomy of anomalies and their trading costs"): a stock
    enters portfolio n (or 1) only when it reaches it, but stays there as long as
    it is within `buffer` portfolios of it (buffer=2 with deciles: it stays in the
    top portfolio while it is in deciles 8-10). Kept stocks leave the middle
    portfolio they would otherwise be in, so portfolios 2..n-1 change slightly.

    Returns a dictionary with:
      ew          monthly equal-weighted returns (rows: months, columns: portfolios)
      vw          monthly value-weighted returns
      n_stocks    number of stocks with a return, per month and portfolio
      breakpoints breakpoints used at each formation date
      assignments which portfolio each stock was in at each formation date
      holdings    only if return_holdings=True: one row per stock and holding
                  month (PERMNO, month, port, MthRet, mcap_lag), used for turnover
    """
    df = prepare_panel(panel, signal)
    hold_months = int(hold_months or 1) if rebalance == "monthly" else 1
    snap = _formation_snapshot(df, signal, rebalance, hold_months)
    snap = _apply_filters(snap, breakpoint_exchanges, min_price, min_nyse_size_pct, exclude_sic)

    # Breakpoints from NYSE stocks only
    quantiles = [i / n_portfolios for i in range(1, n_portfolios)]
    bp_universe = snap[snap["PrimaryExch"].isin(breakpoint_exchanges)]
    breakpoints = bp_universe.groupby("form_key")["sort_value"].quantile(quantiles).unstack()
    breakpoints.columns = [f"P{round(q * 100)}" for q in quantiles]

    # Assign portfolios: portfolio = 1 + number of breakpoints strictly below the value
    snap = snap[snap["form_key"].isin(breakpoints.index)]
    bp_matrix = breakpoints.loc[snap["form_key"]].to_numpy()
    values = snap["sort_value"].to_numpy()[:, None]
    snap["port"] = (values > bp_matrix).sum(axis=1) + 1
    if buffer:
        snap["port"] = _apply_buffer(snap, n_portfolios, buffer, step=hold_months)

    # Holding months: attach each stock-month to its portfolio
    hold = df.merge(snap[["PERMNO", "form_key", "port"]], on=["PERMNO", "form_key"], how="inner")
    hold = hold[hold["MthRet"].notna()]
    hold["month"] = hold["MthCalDt"].dt.to_period("M")

    ew = hold.groupby(["month", "port"])["MthRet"].mean().unstack()

    hv = hold[hold["mcap_lag"] > 0].copy()
    hv["w_ret"] = hv["MthRet"] * hv["mcap_lag"]
    grouped = hv.groupby(["month", "port"])
    vw = (grouped["w_ret"].sum() / grouped["mcap_lag"].sum()).unstack()

    n_stocks = hold.groupby(["month", "port"]).size().unstack()

    out = {
        "ew": ew,
        "vw": vw,
        "n_stocks": n_stocks,
        "breakpoints": breakpoints,
        "assignments": snap[["PERMNO", "form_key", "sort_value", "port", "mkt_cap"]],
    }
    if return_holdings:
        out["holdings"] = hold[["PERMNO", "month", "port", "MthRet", "mcap_lag"]].reset_index(drop=True)
    return out


def summarize(returns: pd.DataFrame, long_port: int, short_port: int) -> pd.DataFrame:
    """Average, standard deviation (both in % per month), t-stat and number of
    months for each portfolio and for the long-short spread."""
    r = returns.copy()
    spread = f"{long_port}-{short_port}"
    r[spread] = r[long_port] - r[short_port]
    n = r.count()
    mean = r.mean()
    std = r.std()
    tstat = mean / (std / np.sqrt(n))
    out = pd.DataFrame(
        {"Average %": mean * 100, "StdDev %": std * 100, "t-stat": tstat, "Months": n}
    ).T
    return out