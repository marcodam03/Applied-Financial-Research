"""
Trading cost inputs for every stock-month of CRSP, called by clean.py (clean_spreads):
the half bid-ask spread (with an estimate where CRSP has no quote) and what the
price impact needs (daily dollar volume and size group).

1. SPREAD BEFORE 1993 (assumption on the period)
CRSP has closing quotes for NASDAQ stocks before 1993 but for NYSE/AMEX stocks only
from 1992-12. Quotes used: NYSE/AMEX from 1992-12, NASDAQ from NASDAQ_FROM (1983).
Where there is no usable quote before 1993, the half spread is estimated:
  a. Cross-section, fitted on 1993-01..1996-12 (every exchange quoted, and still the
     1/8 dollar tick of 1963-1997, so prices and spreads relate as before 1993):
        log(half spread) = b0 + b1 size + b2 size^2 + b3 log(price) + b4 log(volatility)
                           + b5 NASDAQ + b6 AMEX
     size = percentile of market cap among NYSE stocks that month (0..1, so a
     "large" stock means the same in 1965 and 1995), volatility = std of the last 12
     monthly returns (new listings: the median of their size group). Stoll and
     Whaley (1983) and Chen and Velikov (2023, they match missing spreads on size
     and volatility) use the same determinants.
  b. Level by year: the year effect is the average residual of the NASDAQ stocks
     quoted that year (the only quotes before 1993), from NASDAQ_FROM (the National
     Market System, 1982; CRSP has no NASDAQ volume before it either: checked on our
     file, 99.9% missing 1972-1981). The NYSE/AMEX gap is the one of 1993-1996.
     Years before NASDAQ_FROM get the average of its first 5 years: Jones (2002)
     finds that spreads of large NYSE stocks were at similar levels (about 0.6%)
     in the 1950s and the 1980s.
  c. Back from logs with Duan's smearing factor (mean of exp(residual)), so the
     estimate is a mean, not a median.
After 1993, a stock without a quote keeps the fill of src/portfolio/costs.py (the
average spread of the portfolio's quoted trades that month), as before.

2. PRICE IMPACT INPUTS (assumption on size)
  size_group  micro / small / large: NYSE 20th and 50th percentiles of market cap
              (Fama-French); src/portfolio/costs.py gives each its lambda.
  adv         average daily dollar volume = monthly volume x price / 21 trading
              days, NASDAQ volume divided by 2.0 before 2001-02, 1.8 in the rest
              of 2001 and 1.6 in 2002-2003 (Gao and Ritter 2010: NASDAQ counted
              dealer trades twice). Missing (NASDAQ before 1983): the median of the
              same size group that month.
"""
import numpy as np
import pandas as pd

QUOTES_FROM = pd.Period("1992-12", freq="M")  # NYSE/AMEX quotes are usable from here
FIT_MONTHS = ("1993-01", "1996-12")           # estimation window of the cross-section
MIN_QUOTES_PER_YEAR = 200                     # NASDAQ quotes needed to estimate a year
NASDAQ_FROM = 1983                            # first year of NASDAQ quotes used
TRADING_DAYS = 21
MIN_HALF_SPREAD, MAX_HALF_SPREAD = 0.0005, 0.25
SIZE_CUTS = (0.2, 0.5)  # NYSE percentiles: micro | small | large


def stock_characteristics(crsp: pd.DataFrame) -> pd.DataFrame:
    """One row per stock-month of crsp_clean: exchange, NYSE size percentile, size
    group, price, 12-month volatility, ADV in dollars."""
    if "MthVol" not in crsp.columns:
        print("  !! No MthVol in the CRSP table: no dollar volume, so no price impact.")
        crsp = crsp.assign(MthVol=np.nan)
    c = crsp[["PERMNO", "MthCalDt", "PrimaryExch", "mkt_cap", "MthPrc", "MthRet", "MthVol"]].copy()
    c = c.sort_values(["PERMNO", "MthCalDt"])
    c["month"] = c["MthCalDt"].dt.to_period("M")

    # NYSE size percentile, month by month
    def percentile(g: pd.DataFrame) -> pd.Series:
        nyse = np.sort(g.loc[(g["PrimaryExch"] == "N") & (g["mkt_cap"] > 0), "mkt_cap"].to_numpy())
        if len(nyse) < 20:
            return pd.Series(np.nan, index=g.index)
        return pd.Series(np.searchsorted(nyse, g["mkt_cap"].to_numpy(), side="right") / len(nyse),
                         index=g.index)
    c["size_pct"] = c.groupby("month", group_keys=False)[["PrimaryExch", "mkt_cap"]].apply(percentile)
    c["size_group"] = pd.cut(c["size_pct"], [-np.inf, *SIZE_CUTS, np.inf],
                             labels=["micro", "small", "large"], right=False).astype(object)

    # Volatility: std of the last 12 monthly returns (at least 6)
    c["vol12"] = (c.groupby("PERMNO")["MthRet"]
                  .rolling(12, min_periods=6).std().reset_index(level=0, drop=True))
    # new listings (fewer than 6 months): median volatility of their size group that month
    c["vol12"] = (c["vol12"].fillna(c.groupby(["month", "size_group"])["vol12"].transform("median"))
                  .fillna(c.groupby("size_group")["vol12"].transform("median")))

    # ADV in dollars, NASDAQ double counting removed
    ym = c["month"].dt.year * 100 + c["month"].dt.month
    nasdaq_div = np.select([ym < 200102, ym < 200201, ym < 200401], [2.0, 1.8, 1.6], 1.0)
    nasdaq_div = np.where(c["PrimaryExch"] == "Q", nasdaq_div, 1.0)
    adv = c["MthVol"].abs() * c["MthPrc"].abs() / TRADING_DAYS / nasdaq_div
    adv = adv.where(adv > 0)
    c["adv"] = adv.fillna(adv.groupby([c["month"], c["size_group"]]).transform("median"))
    return c


def _design(df: pd.DataFrame) -> np.ndarray:
    s = df["size_pct"].clip(0, 1).to_numpy()
    return np.column_stack([
        np.ones(len(df)), s, s ** 2,
        np.log(df["MthPrc"].abs().clip(lower=0.1)),
        np.log(df["vol12"].clip(lower=0.005)),
        (df["PrimaryExch"] == "Q").to_numpy(float),
        (df["PrimaryExch"] == "A").to_numpy(float),
    ])


def _usable(df: pd.DataFrame) -> pd.Series:
    return df["size_pct"].notna() & df["vol12"].notna() & (df["MthPrc"].abs() > 0)


def estimate_spreads(table: pd.DataFrame) -> pd.DataFrame:
    """table: stock characteristics + quoted half_spread (NaN if none) + quote_ok.
    Adds half_spread_model for every usable stock-month and prints the checks."""
    fit_lo, fit_hi = (pd.Period(m, freq="M") for m in FIT_MONTHS)
    obs = table["quote_ok"] & (table["half_spread"] > 0) & _usable(table)
    fit = table[obs & table["month"].between(fit_lo, fit_hi)]
    if len(fit) < 1000:
        raise ValueError(f"Only {len(fit)} quoted stock-months in {FIT_MONTHS}: cannot fit the spread model.")
    X, y = _design(fit), np.log(fit["half_spread"].to_numpy())
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    smear = np.mean(np.exp(resid))
    r2 = 1 - resid.var() / y.var()

    # Out-of-sample check: fit on 1993-1994, predict NYSE 1995-1996
    half = fit["month"] <= pd.Period("1994-12", freq="M")
    b_half, *_ = np.linalg.lstsq(_design(fit[half]), np.log(fit.loc[half, "half_spread"]), rcond=None)
    test = fit[~half & (fit["PrimaryExch"] == "N")]
    pred_test = np.exp(_design(test) @ b_half) * np.mean(np.exp(
        np.log(fit.loc[half, "half_spread"]) - _design(fit[half]) @ b_half))

    # Year effects from NASDAQ quotes before 1993
    year = table["month"].dt.year
    early_q = table[obs & (table["month"] < QUOTES_FROM) & (table["PrimaryExch"] == "Q")
                    & (table["month"].dt.year >= NASDAQ_FROM)]
    eff = pd.Series(np.log(early_q["half_spread"].to_numpy()) - _design(early_q) @ beta,
                    index=early_q.index).groupby(early_q["month"].dt.year)
    effects = eff.mean()[eff.size() >= MIN_QUOTES_PER_YEAR]
    first = effects.sort_index().iloc[:5].mean() if len(effects) else 0.0
    years = np.arange(year.min(), QUOTES_FROM.year + 1)
    effects = effects.reindex(years)
    effects[effects.index < (effects.first_valid_index() or QUOTES_FROM.year)] = first
    effects = effects.ffill().fillna(0.0)
    shift = year.map(effects).where(table["month"] < QUOTES_FROM, 0.0).fillna(0.0)

    ok = _usable(table)
    pred = pd.Series(np.nan, index=table.index)
    pred[ok] = np.exp(_design(table[ok]) @ beta + shift[ok].to_numpy()) * smear
    table["half_spread_model"] = pred.clip(MIN_HALF_SPREAD, MAX_HALF_SPREAD)

    names = ["const", "size", "size^2", "log price", "log vol", "NASDAQ", "AMEX"]
    print(f"  -> Spread model, fitted on {FIT_MONTHS[0]}..{FIT_MONTHS[1]} ({len(fit):,} quoted "
          f"stock-months, R2 {r2:.2f}): " + ", ".join(f"{n} {b:+.2f}" for n, b in zip(names, beta)))
    print(f"     Out of sample (fit 1993-94, NYSE 1995-96): median half spread actual "
          f"{test['half_spread'].median() * 1e4:.0f} bps, predicted {np.median(pred_test) * 1e4:.0f} bps, "
          f"median abs log error {np.median(np.abs(np.log(pred_test) - np.log(test['half_spread']))):.2f}")
    shown = effects.loc[effects.index % 5 == 0]
    print("     Year effect (log points vs 1993-96, from NASDAQ quotes): "
          + ", ".join(f"{y}: {v:+.2f}" for y, v in shown.items()))
    # Check against Jones (2002): Dow-like stocks (NYSE, above the 90th size percentile)
    big = table[(table["PrimaryExch"] == "N") & (table["size_pct"] >= 0.9)]
    full = (2 * big["half_spread_model"]).groupby(big["month"].dt.year // 10 * 10).median() * 100
    print("     Large NYSE stocks (top 10%), estimated FULL spread by decade, %: "
          + ", ".join(f"{d}s: {v:.2f}" for d, v in full.items() if d < 1990)
          + " (Jones 2002: about 0.6% for Dow stocks in the 1950s and 1980s)")
    return table


def add_cost_inputs(spreads: pd.DataFrame, crsp: pd.DataFrame) -> pd.DataFrame:
    """spreads: the quotes of clean_spreads (PERMNO, month, month_end, quote_day,
    half_spread, stopped). Returns one row per stock-month of crsp_clean with the
    final half_spread, its source, size_group and adv."""
    c = stock_characteristics(crsp)
    t = c.merge(spreads, on=["PERMNO", "month"], how="left")
    t["stopped"] = t["stopped"].astype("boolean").fillna(False).astype(bool)
    # NYSE/AMEX quotes before 1992-12 exist only on days without trades, NASDAQ quotes
    # before NASDAQ_FROM come before the National Market System: neither is used
    t["quote_ok"] = t["half_spread"].notna() & (
        (t["month"] >= QUOTES_FROM)
        | ((t["PrimaryExch"] == "Q") & (t["month"].dt.year >= NASDAQ_FROM)))
    t = estimate_spreads(t)

    t["spread_source"] = np.select(
        [t["stopped"], t["quote_ok"], (t["month"] < QUOTES_FROM) & t["half_spread_model"].notna()],
        ["delisted", "quote", "estimate"], "none")
    t["half_spread"] = np.select(
        [t["stopped"], t["quote_ok"], t["spread_source"] == "estimate"],
        [0.0, t["half_spread"], t["half_spread_model"]], np.nan)
    # month_end of the months the daily file does not cover: the CRSP date
    t["month_end"] = t["month_end"].fillna(t["MthCalDt"])

    share = pd.crosstab(t["month"].dt.year // 10 * 10, t["spread_source"], normalize="index") * 100
    print("     Source of the half spread by decade (% of stock-months):")
    print(share.round(1).to_string())
    adv_check = t.loc[(t["PERMNO"] == 14593) & (t["month"] == pd.Period("2015-12", freq="M")), "adv"]
    if len(adv_check):
        print(f"  -> UNIT CHECK: Apple daily dollar volume 2015-12 = ${adv_check.iloc[0] / 1e9:.1f}bn "
              "(should be a few $bn; if not, MthVol is not in shares)")
    return t[["PERMNO", "month_end", "quote_day", "half_spread", "spread_source", "size_group", "adv"]]
