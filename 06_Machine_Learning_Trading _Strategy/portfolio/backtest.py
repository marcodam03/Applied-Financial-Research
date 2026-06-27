# portfolio/backtest.py
import pandas as pd
import numpy as np
from metrics.perf import cagr, ann_vol, sharpe, max_dd
from settings import SUBPERIODS
from metrics.perf import cagr, sharpe

def backtest_long_top(df_pred, top_frac=0.10, cost=0.0):
    """
    Select each month the top 'top_frac' of stocks by p_up and build an
    equal-weight portfolio. Compare ML with equal-weight and momentum
    benchmarks, using realized next-month returns and optional trading costs.
    """

    # build monthly return matrix (wide format)
    ret_m = df_pred.pivot_table(
        index="Date", columns="Ticker", values="ret_next1m"
        ).sort_index()

    # lag returns so they are known at the start of each month
    ret_hist = ret_m.shift(1)

    # compute 12–1 momentum from product of past 12 returns minus 1
    mom_12_1 = (1 + ret_hist).rolling(12).apply(lambda x: x.prod() - 1, raw=True)

    # reshape momentum back to long format and merge into df_pred
    mom_long = (
        mom_12_1
        .stack(future_stack=True)
        .rename("mom_12_1")
        .reset_index()
    )

    df_pred = df_pred.merge(mom_long, on=["Date", "Ticker"], how="left")

    # months where model predictions are available
    months = sorted(df_pred.loc[df_pred["p_up"].notna(), "Date"].unique())

    records = []
    for d in months:
        m = df_pred[df_pred["Date"] == d]

        # ML top bucket
        cutoff_ml = m["p_up"].quantile(1 - top_frac)
        longs_ml  = m[m["p_up"] >= cutoff_ml]
        r_ml = longs_ml["ret_next1m"].mean() - cost if len(longs_ml) else 0.0

        #  equal-weight benchmark for the whole universe
        r_ew = m["ret_next1m"].mean()

        # momentum top bucket
        cutoff_mom = m["mom_12_1"].quantile(1 - top_frac)
        longs_mom  = m[m["mom_12_1"] >= cutoff_mom]
        r_mom = longs_mom["ret_next1m"].mean() - cost if len(longs_mom) else 0.0

        records.append({"Date": d, "ret_ml": r_ml, "ret_ew": r_ew, "ret_mom": r_mom})

    # build output with cumulative curves
    out = pd.DataFrame(records).set_index("Date").sort_index()
    for col in ["ret_ml", "ret_ew", "ret_mom"]:
        out[f"cum_{col}"] = (1 + out[col]).cumprod()

    return out

def backtest_long_short(df_pred, top_frac=0.10, cost=0.0, score_col="p_up"):
    """
    Long the top 'top_frac' and short the bottom 'top_frac' based
    on the chosen score. Compute market-neutral returns net of estimated
    trading costs to isolate the pure result of the model.
    """

    # months where scores are available
    months = sorted(df_pred.loc[df_pred[score_col].notna(), "Date"].unique())

    records = []
    for d in months:
        m = df_pred[df_pred["Date"] == d].copy()

        # thresholds for long and short sides
        thr_long  = m[score_col].quantile(1 - top_frac)
        thr_short = m[score_col].quantile(top_frac)

        longs  = m[m[score_col] >= thr_long]
        shorts = m[m[score_col] <= thr_short]

        # skip month if we do not have enough stocks on either side
        if len(longs) == 0 or len(shorts) == 0:
            continue

        r_long  = longs["ret_next1m"].mean()
        r_short = shorts["ret_next1m"].mean()

        # cost on both legs (long and short)
        r_ls = (r_long - r_short) - 2 * cost

        records.append({
            "Date": d,
            "ret_long": r_long - cost,
            "ret_short": -(r_short - cost),  # short leg: gain when prices fall
            "ret_long_short": r_ls,
        })

    out = pd.DataFrame(records).set_index("Date").sort_index()

    if not out.empty:
        out["cum_ret_long_short"] = (1 + out["ret_long_short"]).cumprod()
        out["cum_ret_long"] = (1 + out["ret_long"]).cumprod()
        out["cum_ret_short"] = (1 + out["ret_short"]).cumprod()

    return out

# performance metrics used in the backtests
def cagr(series):
    """
    Computes the annualized growth rate assuming the series represents
    a sequence of cumulative return factors (1+r) over monthly intervals.
    """

    if len(series)==0:
        return np.nan
    n_years = len(series)/12
    return series.prod()**(1/n_years)-1

def ann_vol(returns):
    """
    Annualized volatility of monthly returns, using standard deviation
    and assuming 12 periods per year.
    """

    return returns.std(ddof=0)*np.sqrt(12)

def sharpe(returns, rf=0.0):
    """
    Annualized Sharpe ratio based on CAGR and annualized volatility.
    Uses a constant risk-free rate for simplicity.
    """

    v = ann_vol(returns);
    return np.nan if v==0 or np.isnan(v) else (cagr(1+returns)-rf)/v

def max_dd(cum):
    """
    Computes the maximum drawdown of a cumulative return curve,
    measuring the worst peak-to-trough decline.
    """

    return (cum/cum.cummax()-1).min()


def compute_turnover(df_pred, top_frac=0.10):
    """
    Estimates monthly turnover by tracking changes in the ML top-bucket.
    Uses the symmetric difference between consecutive sets of selected tickers.
    """

    # months where model predictions are available
    months = sorted(df_pred.loc[df_pred["p_up"].notna(), "Date"].unique())
    prev = set()
    rows = []

    for d in months:
        m = df_pred[df_pred["Date"] == d]
        thr = m["p_up"].quantile(1 - top_frac)
        # current top group of tickers
        now = set(m.loc[m["p_up"] >= thr, "Ticker"])

        if prev:
            # symmetric difference |A Δ B| / |B| as turnover proxy
            chg = len(now.symmetric_difference(prev)) / max(1, len(now))
            rows.append({"Date": d, "turnover": chg})

        prev = now

    return pd.DataFrame(rows).set_index("Date").sort_index()


def sensitivity_grid(df_pred,
                     top_fracs=(0.10, 0.20),
                     costs=(0.0, 0.0005, 0.0010),
                     subperiods=None):
    """
    Runs a sensitivity analysis over multiple top_frac and cost settings.
    Returns a table with CAGR and Sharpe for the full sample and each subperiod,
    useful for testing robustness of the strategy.
    """
    # fallback to global subperiods if none are provided
    if subperiods is None:
        subperiods = SUBPERIODS

    rec = []

    # full-sample backtests for all parameter combinations
    for tf in top_fracs:
        for c in costs:
            bt_full = backtest_long_top(df_pred, top_frac=tf, cost=c)
            if bt_full.empty:
                continue
            rec.append({
                "period": "FULL",
                "top_frac": tf,
                "cost": c,
                "CAGR_ml": cagr(1 + bt_full["ret_ml"]),
                "Sharpe_ml": sharpe(bt_full["ret_ml"]),
            })

            # subperiod backtests
            for (s, e) in subperiods:
                s_dt = pd.to_datetime(s)
                e_dt = pd.to_datetime(e)

                # restrict data to subperiod
                sub = df_pred[(df_pred["Date"] >= s_dt) & (df_pred["Date"] <= e_dt)]
                if sub.empty or sub["p_up"].notna().sum() == 0:
                    continue

                bt_sub = backtest_long_top(sub, top_frac=tf, cost=c)
                if bt_sub.empty:
                    continue

                rec.append({
                    "period": f"{s_dt.date()}–{e_dt.date()}",
                    "top_frac": tf,
                    "cost": c,
                    "CAGR_ml": cagr(1 + bt_sub["ret_ml"]),
                    "Sharpe_ml": sharpe(bt_sub["ret_ml"]),
                })

    return pd.DataFrame(rec)
