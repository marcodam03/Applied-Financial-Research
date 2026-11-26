"""
The numbers quoted in the text of the paper that are not in one of its tables.

Run from the project folder, after run_final_analysis.py and paper_outputs.py
(it only reads files those scripts write; about 2 seconds):
    python scripts/paper_numbers.py

Reads results/signals/qvi_timely/ (in-sample and holdout returns and costs),
results/compare/ (building blocks of Table 5), paper/rolling_10y.csv (Figure 3)
and data/external/factors_monthly.csv. Prints everything and saves
results/paper/paper_numbers.txt.

Regressions: OLS with Newey-West standard errors, six lags, as in
src/evaluation/alpha.py (same formula, copied so this script needs only pandas).
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SIG = BASE_DIR / "results" / "signals" / "qvi_timely"
CMP = BASE_DIR / "results" / "compare"
OUT = BASE_DIR / "results" / "paper" / "paper_numbers.txt"
IN_SAMPLE = ("1963-07", "2015-12")
HOLDOUT = ("2016-01", "2025-12")
FF6 = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]
BLOCKS = {"Value + quality, ranked across all stocks": "quality_value",
          "+ ranked within industries": "quality_value_industry",
          "+ intangible capital in book value": "quality_value_industry_intan",
          "+ intangible capital in profitability (annual version)": "qvi_intan_full"}

lines = []


def say(text=""):
    print(text)
    lines.append(text)


def monthly(path):
    r = pd.read_csv(path, index_col=0)
    r.index = pd.PeriodIndex(r.index, freq="M")
    return r


def nw_ols(y, X, lags=6):
    """OLS with Newey-West (Bartlett) standard errors; X includes the constant."""
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    s = X * u[:, None]
    S = s.T @ s
    for lag in range(1, lags + 1):
        g = s[lag:].T @ s[:-lag]
        S += (1 - lag / (lags + 1)) * (g + g.T)
    se = np.sqrt(np.diag(XtX_inv @ S @ XtX_inv))
    return beta, beta / se


def regress(y, X):
    d = pd.concat([y.rename("y"), X], axis=1).dropna()
    Xm = np.column_stack([np.ones(len(d)), d.drop(columns="y").to_numpy()])
    b, t = nw_ols(d["y"].to_numpy(), Xm)
    return pd.DataFrame({"coef": b, "t": t}, index=["alpha"] + list(X.columns)), len(d)


def cut(x, period):
    return x.loc[period[0]:period[1]]


f = monthly(BASE_DIR / "data" / "external" / "factors_monthly.csv")
net = cut(monthly(SIG / "returns_vw.csv")["10-1 net"], IN_SAMPLE)
full = monthly(SIG / "holdout" / "returns_vw.csv")["10-1 net"]
assert np.allclose(cut(full, IN_SAMPLE), net), "holdout run does not reproduce 1963-2015"

say("Numbers quoted in the text of the paper (qvi_timely, value-weighted, net of costs)")
say("=" * 78)

say("\nSection 5.2: the q5 alpha is estimated from 1967; mean net return over the same months")
say(f"  net return 1967-01 to 2015-12: {cut(net, ('1967-01', '2015-12')).mean() * 100:.2f}% per month")

say("\nSection 5.2 and Figure 3: rolling 10-year windows (paper/rolling_10y.csv, % per year)")
roll = pd.read_csv(BASE_DIR / "paper" / "rolling_10y.csv", index_col=0)
roll.index = pd.PeriodIndex(roll.index, freq="M")
a = roll["FF6 alpha % per year"]
say(f"  windows: {len(a)} ({a.index.min()} to {a.index.max()})")
say(f"  FF6 alpha > 0 in {(a > 0).mean() * 100:.0f}% of all windows, "
    f"{(a.loc[:'2015-12'] > 0).mean() * 100:.0f}% of those ending by 2015-12")
since = a.loc["2000-12":]
say(f"  windows ending 2000-12 onwards: FF6 alpha between {since.min():.1f}% and {since.max():.1f}% per year")
say(f"  window ending 2000-12: net return {roll.loc['2000-12', 'net return % per year']:.1f}%, "
    f"FF6 alpha {roll.loc['2000-12', 'FF6 alpha % per year']:.1f}% per year")

say("\nSection 5.4: building blocks (results/compare/, same universe and costs, 1963-2015)")
blocks = {name: cut(monthly(CMP / folder / "returns_vw.csv")["10-1 net"], IN_SAMPLE)
          for name, folder in BLOCKS.items()}
blocks["+ current prices, every 3 months (final)"] = net
for name, x in blocks.items():
    mom = regress(x, f[FF6])[0].loc["MOM"]
    say(f"  {name:58s} net Sharpe {x.mean() / x.std() * np.sqrt(12):.2f}   "
        f"FF6 momentum loading {mom['coef']:+.2f} (t = {mom['t']:.2f})")
annual = blocks["+ intangible capital in profitability (annual version)"]
fin_on_ann, _ = regress(net, pd.concat([f[FF6], annual.rename("annual")], axis=1))
ann_on_fin, _ = regress(annual, pd.concat([f[FF6], net.rename("final")], axis=1))
say(f"  alpha of the final strategy on FF6 + annual version: {fin_on_ann.loc['alpha', 'coef'] * 100:.2f}% "
    f"(t = {fin_on_ann.loc['alpha', 't']:.2f})")
say(f"  alpha of the annual version on FF6 + final strategy: {ann_on_fin.loc['alpha', 'coef'] * 100:.2f}% "
    f"(t = {ann_on_fin.loc['alpha', 't']:.2f})")

say("\nSection 5.5: the holdout 2016-2025 (results/signals/qvi_timely/holdout/)")
ho = cut(full, HOLDOUT)
years = (1 + ho).groupby(ho.index.year).prod() - 1
say("  calendar-year net return (%): " + ", ".join(f"{y} {v * 100:+.0f}" for y, v in years.items()))
say(f"  positive years: {(years > 0).sum()} of {len(years)}")
mkt = cut(f["MKT_RF"] + f["RF"], HOLDOUT)
myears = (1 + mkt).groupby(mkt.index.year).prod() - 1
say("  market falls in: " + ", ".join(str(y) for y, v in myears.items() if v < 0))
gross = cut(monthly(SIG / "holdout" / "returns_vw.csv")["10-1"], HOLDOUT)
say(f"  trading costs (gross minus net): {(gross - ho).mean() * 100:.2f}% per month")

say("\nSection 5.6: market exposure, 1963-2015")
capm, n = regress(net, f[["MKT_RF"]])
say(f"  CAPM beta {capm.loc['MKT_RF', 'coef']:.2f} (t = {capm.loc['MKT_RF', 't']:.1f}), {n} months")
say(f"  correlation with the market excess return: {pd.concat([net, f['MKT_RF']], axis=1).dropna().corr().iloc[0, 1]:.2f}")

say("\nSection 7: borrowing fees (D'Avolio 2002: 91% at 0.17%, 9% at 4.3% per year)")
fee = 0.91 * 0.17 + 0.09 * 4.3
say(f"  average fee {fee:.2f}% per year = {fee / 12:.3f}% per month on the short leg")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(lines) + "\n")
print(f"\nSaved {OUT.relative_to(BASE_DIR)}")
