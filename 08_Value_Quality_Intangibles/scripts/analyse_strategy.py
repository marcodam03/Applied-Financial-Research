"""
Is the core strategy luck? Does it all come from one year? How does it move with the
market? (docs/STRATEGY.md, section 4)

Run from the project folder, after run_signal.py for the signal, with the config whose
results you want to analyse (results/signals/<signal>/ holds the LAST run):
    python scripts/run_signal.py --signal qvi_timely --config configs/final.yaml
    python scripts/analyse_strategy.py                     (final strategy, qvi_timely)
    python scripts/analyse_strategy.py --signal qvi_intan_full --config configs/annual.yaml

Uses the value-weighted spread AFTER costs (long decile 10, short decile 1), 1963-07 to
2015-12; the holdout is never read.

A. LUCK     t-stat against the Harvey, Liu and Zhu (2016) bar of 3; block bootstrap
            (12-month blocks, 10,000 resamples) of the mean and the Sharpe ratio;
B. ONE YEAR? every calendar year; the result without the best 1, 3 and 5 years, without
            2008, without recessions; leave-one-year-out; rolling 10-year windows; decades.
C. MARKET   correlations with the market and every factor; FF6 loadings; behaviour in up
            and down markets, in the 10% worst market months and in recessions; rolling
            3-year correlation with the market.
D. MIX      the US market plus k x the strategy (an overlay: the long-short is
            self-financing), k = 0 to 1; best k estimated on the TRAINING period only and
            then checked on validation.
"""
import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.evaluation.alpha import load_factors, newey_west_ols  # noqa: E402

END = pd.Period("2015-12", freq="M")
N_BOOT, BLOCK, SEED = 10_000, 12, 7
RECESSIONS = [("1969-12", "1970-11"), ("1973-11", "1975-03"), ("1980-01", "1980-07"),
              ("1981-07", "1982-11"), ("1990-07", "1991-03"), ("2001-03", "2001-11"),
              ("2007-12", "2009-06")]   # NBER peak to trough months
FACTORS = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM", "STREV", "Q_ME", "Q_IA", "Q_ROE", "Q_EG"]
OVERLAYS = [0, 0.25, 0.5, 0.75, 1.0]

pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda x: f"{x:.2f}")


def tstat(x):
    return x.mean() / x.std() * np.sqrt(len(x))


def sharpe(x):
    return x.mean() / x.std() * np.sqrt(12)


def max_dd(x):
    w = (1 + x).cumprod()
    return (w / w.cummax() - 1).min()


def block_bootstrap(x: np.ndarray, rng) -> tuple:
    n = len(x)
    starts = rng.integers(0, n - BLOCK + 1, size=(N_BOOT, int(np.ceil(n / BLOCK))))
    idx = (starts[:, :, None] + np.arange(BLOCK)).reshape(N_BOOT, -1)[:, :n]
    s = x[idx]
    return s.mean(axis=1), s.mean(axis=1) / s.std(axis=1, ddof=1) * np.sqrt(12)


def recession_mask(index: pd.PeriodIndex) -> pd.Series:
    m = pd.Series(False, index=index)
    for a, b in RECESSIONS:
        m.loc[a:b] = True
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal", default="qvi_timely")
    ap.add_argument("--config", default="configs/final.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load((BASE_DIR / args.config).read_text())
    train, valid = cfg["periods"]["train"], cfg["periods"]["validation"]
    mod = importlib.import_module(f"src.signals.{args.signal}")
    spread = f"{mod.LONG}-{mod.SHORT}"
    r = pd.read_csv(BASE_DIR / "results" / "signals" / args.signal / "returns_vw.csv", index_col=0)
    r.index = pd.PeriodIndex(r.index, freq="M")
    r = r[r.index <= END]
    f = load_factors(BASE_DIR / cfg["data"]["factors"]).reindex(r.index)
    s = r[f"{spread} net"].dropna()
    f = f.reindex(s.index)
    mkt_ex, rf = f["MKT_RF"], f["RF"]
    rec = recession_mask(s.index)
    out = {}

    print(f"{args.signal}, value-weighted spread {spread} AFTER costs, {s.index[0]} to {s.index[-1]}, "
          f"config {args.config}\n")

    # ---------------- A. Luck ----------------
    rng = np.random.default_rng(SEED)
    bm, bs = block_bootstrap(s.to_numpy(), rng)
    t = tstat(s)
    from scipy.stats import norm
    p_one = 1 - norm.cdf(t)
    luck = pd.Series({
        "Mean % per month": s.mean() * 100,
        "t-stat": t,
        "Passes Harvey-Liu-Zhu bar (t > 3)": float(t > 3),
        "Sharpe (annual)": sharpe(s),
        "Bootstrap 95% CI of the mean, low %": np.percentile(bm, 2.5) * 100,
        "Bootstrap 95% CI of the mean, high %": np.percentile(bm, 97.5) * 100,
        "Bootstrap 95% CI of the Sharpe, low": np.percentile(bs, 2.5),
        "Bootstrap 95% CI of the Sharpe, high": np.percentile(bs, 97.5),
        "Bootstrap share of resamples with mean <= 0 %": (bm <= 0).mean() * 100,
    })
    print("===== A. Is it luck? =====")
    print(luck.to_string(float_format=lambda x: f"{x:.3f}"))
    out["A_luck"] = luck

    # ---------------- B. One year? ----------------
    yearly = (1 + s).groupby(s.index.year).prod() - 1
    logs = np.log1p(s).groupby(s.index.year).sum()
    share = logs / logs.sum() * 100
    ranked = pd.DataFrame({"Return %": yearly * 100, "Share of total log-return %": share}).sort_values(
        "Return %", ascending=False)
    print("\n===== B1. Best and worst calendar years (share of the whole 1963-2015 gain) =====")
    print(pd.concat([ranked.head(5), ranked.tail(5)]).to_string())

    best = ranked.index.tolist()
    rows = {"All months": s}
    for k in (1, 3, 5):
        rows[f"Without the best {k} year(s) ({', '.join(map(str, best[:k]))})"] = s[~s.index.year.isin(best[:k])]
    rows["Without 2008"] = s[s.index.year != 2008]
    rows["Without recession months"] = s[~rec.to_numpy()]
    rows["Recession months only"] = s[rec.to_numpy()]
    drop = pd.DataFrame({k: {"Mean % per month": v.mean() * 100, "t": tstat(v), "Sharpe": sharpe(v),
                             "Months": len(v)} for k, v in rows.items()}).T
    print("\n===== B2. Remove the best years =====")
    print(drop.to_string())

    loo = pd.Series({y: tstat(s[s.index.year != y]) for y in sorted(set(s.index.year))})
    print(f"\nLeave-one-year-out: the lowest t-stat is {loo.min():.2f} (dropping {loo.idxmin()}), "
          f"the highest {loo.max():.2f}; t > 2 in {(loo > 2).mean():.0%} of the cases.")

    roll = s.rolling(120)
    roll_mean = roll.mean() * 1200
    roll_t = roll.mean() / roll.std() * np.sqrt(120)
    print(f"Rolling 10-year windows ({roll_mean.notna().sum()}): positive in {(roll_mean > 0).mean() * 100:.0f}% "
          f"of them, t > 2 in {(roll_t > 2).sum() / roll_t.notna().sum():.0%}; worst window "
          f"{roll_mean.min():.1f}% per year (ending {roll_mean.idxmin()}), best {roll_mean.max():.1f}% "
          f"(ending {roll_mean.idxmax()}).")
    dec = s.groupby((s.index.year // 10) * 10)
    decades = pd.DataFrame({"Mean % per year": dec.mean() * 1200, "t": dec.apply(tstat), "Sharpe": dec.apply(sharpe),
                            "Months": dec.size()})
    print("\nBy decade:")
    print(decades.to_string())
    print(f"\nMonths: {(s > 0).mean():.0%} positive, median {s.median() * 100:.2f}% vs mean {s.mean() * 100:.2f}%, "
          f"worst {s.min() * 100:.1f}% ({s.idxmin()}), best {s.max() * 100:.1f}% ({s.idxmax()}), "
          f"skewness {s.skew():.2f}, max drawdown {max_dd(s) * 100:.1f}%.")
    out["B_drop"], out["B_years"], out["B_decades"] = drop, ranked, decades

    # ---------------- C. Market and factors ----------------
    avail = [c for c in FACTORS if c in f.columns]
    tr = s.loc[train[0]:train[1]]
    corr = pd.DataFrame({
        "Correlation, all": f[avail].corrwith(s),
        "Correlation, training": f[avail].loc[tr.index].corrwith(tr),
    })
    X = np.column_stack([np.ones(len(s)), f[["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]].to_numpy()])
    b, se, _, r2 = newey_west_ols(s.to_numpy(), X, 6)
    load = pd.DataFrame({"Loading": b, "t": b / se},
                        index=["alpha (% x 100)", "MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"])
    load.iloc[0, 0] *= 100
    print("\n===== C1. Correlation with the market and the factors =====")
    print(corr.to_string())
    print(f"\nFF6 loadings of the net spread (R2 {r2:.2f}):")
    print(load.to_string())

    up, down = mkt_ex > 0, mkt_ex <= 0
    worst10 = mkt_ex <= mkt_ex.quantile(0.10)

    def beta(mask):
        x, y = mkt_ex[mask], s[mask]
        return np.polyfit(x, y, 1)[0]

    cond = pd.DataFrame({
        "Market up months": {"Strategy %": s[up].mean() * 100, "Market %": mkt_ex[up].mean() * 100,
                             "Beta": beta(up), "Months": up.sum()},
        "Market down months": {"Strategy %": s[down].mean() * 100, "Market %": mkt_ex[down].mean() * 100,
                               "Beta": beta(down), "Months": down.sum()},
        "10% worst market months": {"Strategy %": s[worst10].mean() * 100, "Market %": mkt_ex[worst10].mean() * 100,
                                    "Beta": beta(worst10), "Months": worst10.sum()},
        "Recessions": {"Strategy %": s[rec].mean() * 100, "Market %": mkt_ex[rec].mean() * 100,
                       "Beta": beta(rec), "Months": rec.sum()},
        "Expansions": {"Strategy %": s[~rec].mean() * 100, "Market %": mkt_ex[~rec].mean() * 100,
                       "Beta": beta(~rec), "Months": (~rec).sum()},
    }).T
    rc = s.rolling(36).corr(mkt_ex)
    print("\n===== C2. Good and bad markets (market = excess return of the US market) =====")
    print(cond.to_string())
    print(f"Rolling 3-year correlation with the market: average {rc.mean():+.2f}, "
          f"from {rc.min():+.2f} ({rc.idxmin()}) to {rc.max():+.2f} ({rc.idxmax()}).")
    out["C_corr"], out["C_loadings"], out["C_conditional"] = corr, load, cond

    # ---------------- D. Market + strategy ----------------
    mkt_tot = mkt_ex + rf

    def stats(x, sub):
        x = x.loc[sub[0]:sub[1]] if sub else x
        yrs = len(x) / 12
        yr = (1 + x).groupby(x.index.year).prod() - 1
        return {"Return per year %": ((1 + x).prod() ** (1 / yrs) - 1) * 100,
                "Vol per year %": x.std() * np.sqrt(12) * 100,
                "Sharpe": sharpe(x - rf.loc[x.index]),
                "Max drawdown %": max_dd(x) * 100, "Worst year %": yr.min() * 100}

    tr_s, tr_m = s.loc[train[0]:train[1]], mkt_ex.loc[train[0]:train[1]]
    mu_s, mu_m = tr_s.mean(), tr_m.mean()
    v_s, v_m, c = tr_s.var(), tr_m.var(), np.cov(tr_s, tr_m)[0, 1]
    k_star = (mu_s * v_m - mu_m * c) / (mu_m * v_s - mu_s * c)
    k_star = float(np.clip(k_star, 0, 2))
    rows = []
    for label, sub in [("all", None), ("training", train), ("validation", valid)]:
        ks = sorted(set(OVERLAYS + [round(k_star, 2)]))
        for k in ks:
            name = f"market + {k:.2f} x strategy" + (" (best k on training)" if k == round(k_star, 2) else "")
            rows.append({"period": label, "portfolio": name, **stats(mkt_tot + k * s, sub)})
        rows.append({"period": label, "portfolio": "strategy alone (cash + spread)", **stats(rf + s, sub)})
    mix = pd.DataFrame(rows).set_index(["period", "portfolio"])
    print("\n===== D. The US market plus the strategy as an overlay =====")
    print(f"Best k (maximum Sharpe), estimated on training {train[0]} to {train[1]} only: {k_star:.2f}")
    print(mix.to_string())
    out["D_mix"] = mix

    d = BASE_DIR / "results" / "analysis" / f"{args.signal}_{Path(args.config).stem}"
    d.mkdir(parents=True, exist_ok=True)
    for k, v in out.items():
        v.to_csv(d / f"{k}.csv")
    print(f"\nResults saved in: {d}")


if __name__ == "__main__":
    main()
