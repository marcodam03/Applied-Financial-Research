"""
Portfolio turnover and trading costs.

Turnover of a portfolio in month t, in plain words: the fraction of the portfolio
you have to sell (and buy back with other stocks) at the start of month t.
    turnover(t) = 1/2 * sum over stocks of |w_start(t) - w_end(t-1)|
w_start(t) is the weight the portfolio wants at the start of month t, w_end(t-1)
is last month's weight after one month of returns (weights drift with prices).
0 = nothing traded, 1 = every stock replaced.

Equal-weighted portfolios go back to equal weights every month, so they trade a
little every month even when the list of stocks does not change. Value-weighted
portfolios mostly trade at the formation date (plus delistings and share issues).

Trading cost of every dollar bought or sold = half spread + price impact.

1. Half spread: buying or selling a stock costs half its bid-ask spread, relative
to the midpoint, (ask - bid) / (ask + bid). The return of month t starts from the
close of the last trading day of month t-1, so that close is when the backtest
trades, and every trade pays the stock's closing spread of that day
(data/interim/spreads_clean.parquet, written by src/data/clean.py; before 1993 an
estimate for NYSE/AMEX stocks, see src/data/cost_inputs.py):
    spread cost(t) = sum over stocks of |w_start(t) - w_end(t-1)| * half_spread(end of t-1)
A stock with no spread is charged the average spread of the portfolio's other
trades that month.

2. Price impact (professor's slides 21-24): trades move prices, more so for small
stocks and for trades that are big compared with the volume:
    impact per dollar = lambda x participation / 10        (shape "linear", slides)
                      = lambda / 10 x sqrt(participation / 100)        ("sqrt")
    participation     = dollars traded in the stock / its daily dollar volume
                        (per day, if the trade is spread over execution_days)
    lambda            = 2.5 micro, 1.5 small, 1.0 large stocks (NYSE size
                        percentiles 20 and 50): trading 1% of a day's volume costs
                        25, 15 or 10 bps, with either shape.
"sqrt": costs grow with the square root of the trade, as Frazzini, Israel and
Moskowitz (2018) find on real institutional trades; the linear rule gets very
expensive for trades bigger than a few % of the volume.
Dollars traded = fund size (per leg) x |weight change|. The fund size is
`aum_musd` in `aum_ref_month` and moves back in time with the market's total dollar
volume (aum_scale "volume"), so the fund is the same share of the market's trading
in 1965 and in 2015. With "market_cap" it is the same share of the market's value
instead: since stocks traded far less in the 1960s-80s (monthly volume about 1% of
market cap, 10-20% today), the same fund is then a much bigger part of each day's
volume back then. A trade bigger than one day's volume is spread over several days
(participation at most 100% per day). A stock with no volume gets the median volume
of the portfolio's trades that month.
"""
import numpy as np
import pandas as pd

LAMBDA = {"micro": 2.5, "small": 1.5, "large": 1.0}


def weight_changes(holdings: pd.DataFrame, weighting: str) -> pd.DataFrame:
    """One row per stock, portfolio and month t: change = |w_start(t) - w_end(t-1)|.
    holdings: the 'holdings' table of sort_portfolios(..., return_holdings=True).
    The weights are the ones sort_portfolios uses for the returns. Only months where
    the portfolio is held this month AND was held last month (the first month of a
    portfolio has nothing to compare with)."""
    if weighting == "vw":
        h = holdings[holdings["mcap_lag"] > 0]
        size = h["mcap_lag"]
    elif weighting == "ew":
        h = holdings
        size = pd.Series(1.0, index=h.index)
    else:
        raise ValueError("weighting must be 'ew' or 'vw'")

    group = [h["month"], h["port"]]
    w_start = size / size.groupby(group).transform("sum")
    grown = w_start * (1 + h["MthRet"])
    w_end = grown / grown.groupby(group).transform("sum")

    # Compare the start weights of month t with the end weights of month t-1
    keys = ["PERMNO", "port", "month"]
    now = h[keys].assign(w_start=w_start)
    before = h[keys].assign(month=h["month"] + 1, w_end_prev=w_end)
    trades = now.merge(before, on=keys, how="outer").fillna({"w_start": 0.0, "w_end_prev": 0.0})
    trades["change"] = (trades["w_start"] - trades["w_end_prev"]).abs()

    held = h[["month", "port"]].drop_duplicates()
    held_both = held.merge(held.assign(month=held["month"] + 1), on=["month", "port"])
    return trades.merge(held_both, on=["month", "port"])[keys + ["change"]]


def turnover(changes: pd.DataFrame) -> pd.DataFrame:
    """Monthly one-way turnover (rows: months, columns: portfolios).
    changes: weight_changes(holdings, weighting)."""
    return 0.5 * changes.groupby(["month", "port"])["change"].sum().unstack()


def aum_path(spreads: pd.DataFrame, panel: pd.DataFrame, aum_usd: float, ref_month: str,
             scale: str = "volume") -> pd.Series:
    """Fund size per leg ($) each month: aum_usd in ref_month, moved back in time with
    the market's total dollar volume (scale="volume", from spreads) or total market
    cap (scale="market_cap", from panel)."""
    if scale == "volume":
        total = spreads.groupby("month")["adv"].sum()
    elif scale == "market_cap":
        total = panel["mkt_cap"].groupby(pd.to_datetime(panel["MthCalDt"]).dt.to_period("M")).sum()
    else:
        raise ValueError("costs.aum_scale must be 'volume' or 'market_cap'")
    ref = pd.Period(ref_month, freq="M")
    return aum_usd * total / (total.loc[ref] if ref in total.index else total.iloc[-1])


def trading_cost(changes: pd.DataFrame, spreads: pd.DataFrame, impact: dict = None) -> dict:
    """Monthly trading cost of each portfolio, as a return (rows: months, columns:
    portfolios): {"total", "spread", "impact"}.
    changes: weight_changes(holdings, weighting); spreads: load_spreads() of
    src/data/panel.py; impact: None (spread only) or {"aum": aum_path(...),
    "shape": "linear" or "sqrt", "execution_days": number of days}."""
    # the trades of month t are made at the last close of month t-1
    quotes = spreads.assign(month=spreads["month"] + 1)
    t = changes.merge(quotes, on=["PERMNO", "month"], how="left")
    group = [t["month"], t["port"]]
    paid = (t["change"] * t["half_spread"]).groupby(group).sum()
    traded = t["change"].groupby(group).sum()
    traded_quoted = t["change"].where(t["half_spread"].notna(), 0.0).groupby(group).sum()
    # trades without a spread pay the average of the others; none: 0/0 = NaN
    spread_cost = (paid / traded_quoted * traded).where(traded > 0, 0.0)

    impact_cost = spread_cost * 0.0
    if impact is not None:
        lam = t["size_group"].map(LAMBDA)
        lam = lam.fillna(lam.groupby(group).transform("median")).fillna(LAMBDA["small"])
        adv = t["adv"].where(t["adv"] > 0)
        adv = adv.fillna(adv.groupby(group).transform("median"))
        dollars = impact["aum"].reindex(t["month"]).to_numpy() * t["change"]
        participation = (dollars / impact.get("execution_days", 1) / adv).clip(upper=1.0)
        if impact.get("shape", "linear") == "linear":
            per_dollar = lam * participation / 10
        elif impact["shape"] == "sqrt":
            per_dollar = lam / 10 * np.sqrt(participation / 100)
        else:
            raise ValueError("costs.impact_shape must be 'linear' or 'sqrt'")
        impact_cost = (t["change"] * per_dollar).groupby(group).sum().reindex(spread_cost.index)

    return {"total": (spread_cost + impact_cost).unstack(),
            "spread": spread_cost.unstack(), "impact": impact_cost.unstack()}
