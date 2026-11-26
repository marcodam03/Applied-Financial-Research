"""
Tables (LaTeX) and figures (PDF) of the paper, from the results already on disk.
Nothing is backtested here: run the backtests first.

Needs (run from the project folder):
    python scripts/run_final_analysis.py                                  results/ of qvi_timely
    python scripts/run_signal.py --signal qvi_timely --config configs/final.yaml --holdout
                                                                          (done once already)
    python scripts/compare_strategies.py                                  building blocks (Table 5)
    python scripts/describe_legs.py                                       legs (Table 2, Figure 1)
Then:
    python scripts/paper_outputs.py

Output, to upload to Overleaf with the same folder names:
    paper/tables/*.tex     one tabular per table, \\input{} by main.tex
    paper/figures/*.pdf    the figures
Every number is computed from results/signals/<signal>/returns_*.csv and the factor
file, or read from the csv files of run_capacity.py and overfit_tests.py.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.evaluation.alpha import MODELS, load_factors, newey_west_ols, regress  # noqa: E402

SIGNAL = "qvi_timely"
RES = BASE_DIR / "results"
SIG = RES / "signals" / SIGNAL
OUT_T = BASE_DIR / "paper" / "tables"
OUT_F = BASE_DIR / "paper" / "figures"
IN_SAMPLE = ("1963-07", "2015-12")
HOLDOUT = ("2016-01", "2025-12")
FULL = ("1963-07", "2025-12")
SUB = {"1963--2015": IN_SAMPLE, "1963--1994": ("1963-07", "1994-12"), "1995--2015": ("1995-01", "2015-12")}
ALPHA_MODELS = ["CAPM", "FF5", "FF6", "q5", "q5+STR"]
RECESSIONS = [("1969-12", "1970-11"), ("1973-11", "1975-03"), ("1980-01", "1980-07"),
              ("1981-07", "1982-11"), ("1990-07", "1991-03"), ("2001-03", "2001-11"),
              ("2007-12", "2009-06"), ("2020-02", "2020-04")]   # NBER peak to trough


# ------------------------------------------------------------------ helpers
def read(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, index_col=0)
    x.index = pd.PeriodIndex(x.index, freq="M")
    return x


def need(path: Path) -> Path:
    if not path.exists():
        sys.exit(f"Missing {path.relative_to(BASE_DIR)}: run the script that writes it first (see the docstring).")
    return path


def first(*paths: Path) -> Path:
    """The first file that exists (file names changed when the configs were renamed)."""
    for p in paths:
        if p.exists():
            return p
    sys.exit("Missing all of: " + ", ".join(str(p.relative_to(BASE_DIR)) for p in paths))


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    return x.mean() / x.std() * np.sqrt(len(x))


def f2(x, d=2) -> str:
    return "" if pd.isna(x) else f"{x:.{d}f}"


def est(mean, t, d=2) -> str:
    """'0.41 (2.58)': estimate with its t-stat."""
    return "" if pd.isna(mean) else f"{mean:.{d}f} ({t:.2f})"


def alpha(x: pd.Series, factors: pd.DataFrame, model: str, period) -> tuple:
    r = regress(x.dropna(), factors, model, long_short=True, nw_lags=6, start=period[0], end=period[1])
    return r["alpha %"], r["t(alpha)"]


def mean_t(x: pd.Series, period) -> tuple:
    s = x.loc[period[0]:period[1]].dropna()
    return s.mean() * 100, tstat(s)


def tabular(rows: list, header: list, align: str, groups: dict = None) -> str:
    """Booktabs tabular. rows: lists of cells, or a string = a panel title row."""
    n = len(header)
    out = [f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        if isinstance(r, str):
            out.append(f"\\multicolumn{{{n}}}{{l}}{{\\textit{{{r}}}}} \\\\")
        elif r == []:
            out.append("\\midrule")
        else:
            out.append(" & ".join(str(c) for c in r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def save(name: str, text: str) -> None:
    (OUT_T / f"{name}.tex").write_text(text)
    print(f"  tables/{name}.tex")


# ------------------------------------------------------------------ tables
def table_sample(n: pd.DataFrame) -> None:
    total = n.sum(axis=1)
    rows = []
    for label, (a, b) in {"1963--1969": ("1963-07", "1969-12"), "1970--1979": ("1970-01", "1979-12"),
                          "1980--1989": ("1980-01", "1989-12"), "1990--1999": ("1990-01", "1999-12"),
                          "2000--2009": ("2000-01", "2009-12"), "2010--2015": ("2010-01", "2015-12"),
                          "2016--2025 (holdout)": HOLDOUT}.items():
        t = total.loc[a:b]
        long_, short_ = n[n.columns[-1]].loc[a:b], n[n.columns[0]].loc[a:b]
        rows.append([label, len(t), f"{t.mean():,.0f}", f"{t.min():,.0f}", f"{t.max():,.0f}",
                     f"{long_.mean():,.0f}", f"{short_.mean():,.0f}"])
    save("tab_sample", tabular(rows, ["Period", "Months", "Stocks (mean)", "Min", "Max",
                                      "Decile 10", "Decile 1"], "lrrrrrr"))


def table_deciles(r: pd.DataFrame, factors: pd.DataFrame) -> None:
    ports = [c for c in r.columns if c.isdigit()]
    cols = ports + ["10-1", "10-1 net"]
    lines = {"Mean return": [], "$t$": [], "CAPM $\\alpha$": [], "$t$ ": [], "FF6 $\\alpha$": [],
             "$t$  ": [], "q5 $\\alpha$": [], "$t$   ": []}
    for c in cols:
        x = r[c]
        m, t = mean_t(x, IN_SAMPLE)
        lines["Mean return"].append(f2(m)); lines["$t$"].append(f"({t:.2f})")
        for key, model in [("CAPM $\\alpha$", "CAPM"), ("FF6 $\\alpha$", "FF6"), ("q5 $\\alpha$", "q5")]:
            a, ta = alpha(x, factors, model, IN_SAMPLE) if "-" in c else _excess_alpha(x, factors, model)
            lines[key].append(f2(a))
            tkey = {"CAPM $\\alpha$": "$t$ ", "FF6 $\\alpha$": "$t$  ", "q5 $\\alpha$": "$t$   "}[key]
            lines[tkey].append(f"({ta:.2f})")
    header = [""] + [("Low" if c == ports[0] else "High" if c == ports[-1] else c) for c in ports] + ["10--1", "10--1 net"]
    rows = [[k.strip() if k.startswith("$t$") else k] + v for k, v in lines.items()]
    save("tab_deciles", tabular(rows, header, "l" + "r" * len(cols)))


def _excess_alpha(x, factors, model):
    r = regress(x.dropna(), factors, model, long_short=False, nw_lags=6, start=IN_SAMPLE[0], end=IN_SAMPLE[1])
    return r["alpha %"], r["t(alpha)"]


def table_spread(rets: dict, turnover: dict, costs: dict, factors: pd.DataFrame) -> None:
    rows = []
    for w, label in [("vw", "Panel A: value-weighted"), ("ew", "Panel B: equal-weighted")]:
        rows.append(label)
        r = rets[w]
        to = (turnover[w]["10"] + turnover[w]["1"]) / 2 * 100
        cost = (costs[w]["10"] + costs[w]["1"]) * 100
        for name, fn in [("Gross return", lambda p: mean_t(r["10-1"], p)),
                         ("Net return", lambda p: mean_t(r["10-1 net"], p))] + \
                        [(f"{m} net $\\alpha$", (lambda m: lambda p: alpha(r["10-1 net"], factors, m, p))(m))
                         for m in ALPHA_MODELS]:
            rows.append([name] + [est(*fn(p)) for p in SUB.values()])
        rows.append(["Turnover (\\% per side)"] + [f2(to.loc[a:b].mean(), 1) for a, b in SUB.values()])
        rows.append(["Trading cost (\\%)"] + [f2(cost.loc[a:b].mean()) for a, b in SUB.values()])
        if w == "vw":
            rows.append([])
    save("tab_spread", tabular(rows, [""] + list(SUB), "lrrr"))


def table_holdout(rets_in: dict, rets_ho: dict, factors: pd.DataFrame) -> None:
    rows = []
    periods = {"1963--2015": IN_SAMPLE, "2016--2025 (holdout)": HOLDOUT, "1963--2025": FULL}
    for w, label in [("vw", "Panel A: value-weighted"), ("ew", "Panel B: equal-weighted")]:
        rows.append(label)
        r = rets_ho[w]
        rows.append(["Gross return"] + [est(*mean_t(r["10-1"], p)) for p in periods.values()])
        rows.append(["Net return"] + [est(*mean_t(r["10-1 net"], p)) for p in periods.values()])
        for m in ALPHA_MODELS:
            rows.append([f"{m} net $\\alpha$"] + [est(*alpha(r["10-1 net"], factors, m, p)) for p in periods.values()])
        s = r["10-1 net"]
        rows.append(["Net Sharpe ratio (annual)"] + [f2(s.loc[a:b].mean() / s.loc[a:b].std() * np.sqrt(12))
                                                     for a, b in periods.values()])
        if w == "vw":
            rows.append([])
    # the holdout file must reproduce the in-sample run exactly
    gap = (rets_ho["vw"]["10-1 net"].loc[IN_SAMPLE[0]:IN_SAMPLE[1]]
           - rets_in["vw"]["10-1 net"].loc[IN_SAMPLE[0]:IN_SAMPLE[1]]).abs().max()
    if gap > 1e-10:
        print(f"  !! the holdout run differs from the in-sample run before 2016 (max gap {gap:.2e})")
    save("tab_holdout", tabular(rows, [""] + list(periods), "lrrr"))


BLOCKS = [("quality_value", "Value + quality, ranked across all stocks"),
          ("quality_value_industry", "+ ranked within industries"),
          ("quality_value_industry_intan", "+ intangible capital in book value"),
          ("qvi_intan_full", "+ intangible capital in profitability"),
          (SIGNAL, "+ current prices, formed every 3 months (final)")]


def table_blocks(factors: pd.DataFrame) -> None:
    """Table 5: the strategy built one correction at a time (compare_strategies.py)."""
    rows = []
    for sig, label in BLOCKS:
        f = RES / "signals" / sig / "returns_vw.csv"
        if not f.exists():
            print(f"  !! no results for {sig}: run python scripts/compare_strategies.py "
                  f"(table tab_blocks incomplete)")
            continue
        s = read(f)["10-1 net"]
        rows.append([label, est(*mean_t(s, IN_SAMPLE))]
                    + [est(*alpha(s, factors, m, IN_SAMPLE)) for m in ["FF5", "FF6", "q5"]])
    save("tab_blocks", tabular(rows, ["", "Net return", "FF5 net $\\alpha$", "FF6 net $\\alpha$",
                                      "q5 net $\\alpha$"], "lrrrr"))


def table_legs() -> None:
    """Table 2: characteristics of the two legs (describe_legs.py)."""
    f = RES / "paper" / "legs_characteristics.csv"
    if not f.exists():
        print("  !! no results/paper/legs_characteristics.csv: run python scripts/describe_legs.py")
        return
    c = pd.read_csv(f, index_col=0)
    rows = [[k.replace("&", "\\&").replace("$", "\\$").replace("%", "\\%").replace("-12 to -2", "$-12$ to $-2$")]
            + [f2(v) for v in c.loc[k]] for k in c.index]
    save("tab_legs", tabular(rows, [""] + list(c.columns), "lrrr"))


def table_robust() -> None:
    rb = pd.read_csv(need(SIG / "robustness.csv"))
    rb = rb[(rb["sample"] == "full") & (rb["weighting"] == "vw")].set_index("variant")
    names = {"main": "Main specification", "all_stocks": "With microcaps",
             "large_caps": "Large caps only (above NYSE median)", "hold1": "Formed every month",
             "hold6": "Formed every 6 months", "hold3_buffer2": "Every 3 months + turnover buffer"}
    rows = []
    for v, label in names.items():
        if v not in rb.index:
            continue
        x = rb.loc[v]
        to = (x["Turnover % long"] + x["Turnover % short"]) / 2
        rows.append([label, est(x["Net avg %"], x["Net t"]), est(x["FF6 net alpha %"], x["FF6 net t"]),
                     est(x["q5 net alpha %"], x["q5 net t"]), f2(to, 1)])
    save("tab_robust", tabular(rows, ["", "Net return", "FF6 net $\\alpha$", "q5 net $\\alpha$", "Turnover"],
                               "lrrrr"))


def table_costs() -> None:
    f = first(RES / "capacity" / f"{SIGNAL}_final_scenarios.csv",
              RES / "capacity" / f"{SIGNAL}_fast_quarterly_scenarios.csv")
    c = pd.read_csv(f)
    c = c[c["period"] == "train+validation"].set_index("cost scenario")
    nice = {"spread only": "Half spread only", "linear $10m": "Linear impact, \\$10m",
            "linear $100m (main)": "Linear impact, \\$100m (main)", "sqrt $100m": "Square-root impact, \\$100m",
            "linear $500m": "Linear impact, \\$500m", "linear $1bn": "Linear impact, \\$1bn",
            "linear $2.5bn": "Linear impact, \\$2.5bn",
            "linear $100m, 5 days to trade": "Linear impact, \\$100m, 5 days to trade"}
    rows = [[nice.get(k, k.replace("$", "\\$")), f2(x["Cost %"]), est(x["Net %"], x["Net t"]),
             est(x["FF6 net alpha %"], x["FF6 t"]), est(x["q5 net alpha %"], x["q5 t"])]
            for k, x in c.iterrows()]
    save("tab_costs", tabular(rows, ["Cost model / fund size per side", "Cost", "Net return",
                                     "FF6 net $\\alpha$", "q5 net $\\alpha$"], "lrrrr"))


def table_overfit() -> None:
    folder = RES / "overfit" / SIGNAL
    pl = pd.read_csv(first(folder / "A_placebo.csv", folder / "C_placebo.csv"), index_col=0)
    hv = pd.read_csv(first(folder / "B_half_universes.csv", folder / "D_half_universes.csv"))
    rows = ["Panel A: placebo, score shuffled across stocks within each month (gross, VW)"]
    for key, label in [("gross %", "Gross return"), ("FF6 gross alpha %", "FF6 $\\alpha$"),
                       ("q5 gross alpha %", "q5 $\\alpha$")]:
        x = pl.loc[key]
        rows.append([label, f2(x["real"]), f2(x["placebo mean"]), f2(x["placebo 5%"]), f2(x["placebo 95%"])])
    rows.append([])
    rows.append("Panel B: random halves of the stocks (net of costs, VW)")
    rows.append(["Split / half", "Net return", "FF6 net $\\alpha$", "q5 net $\\alpha$", ""])
    for _, x in hv.iterrows():
        rows.append([f"{int(x['split'])} / {x['half']}", f2(x["net %"]),
                     est(x["FF6 net alpha %"], x["FF6 net t"]), est(x["q5 net alpha %"], x["q5 net t"]), ""])
    save("tab_overfit", tabular(rows, ["", "Real", "Placebo mean", "Placebo 5\\%", "Placebo 95\\%"], "lrrrr"))


def recession_mask(idx: pd.PeriodIndex) -> np.ndarray:
    m = np.zeros(len(idx), dtype=bool)
    for a, b in RECESSIONS:
        m |= (idx >= pd.Period(a, "M")) & (idx <= pd.Period(b, "M"))
    return m


def table_behaviour(r: pd.DataFrame, factors: pd.DataFrame) -> None:
    s = r["10-1 net"].loc[IN_SAMPLE[0]:IN_SAMPLE[1]].dropna()
    f = factors.reindex(s.index)
    cols = MODELS["FF6"]["factors"]
    ok = f[cols].notna().all(axis=1)
    X = np.column_stack([np.ones(ok.sum()), f.loc[ok, cols].to_numpy()])
    b, se, _, r2 = newey_west_ols(s[ok].to_numpy(), X, 6)
    names = ["$\\alpha$ (\\% per month)", "Market", "SMB (size)", "HML (value)", "RMW (profitability)",
             "CMA (investment)", "MOM (momentum)"]
    left = [[n, f2(b[i] * (100 if i == 0 else 1)), f"({b[i] / se[i]:.2f})"] for i, n in enumerate(names)]
    left.append(["$R^2$", f2(r2), ""])
    mkt = f["MKT_RF"]
    worst = mkt <= mkt.quantile(0.10)
    rec = recession_mask(s.index)
    right = [[lab, f2(s[m].mean() * 100), f2(mkt[m].mean() * 100), int(m.sum())]
             for lab, m in [("Market up", (mkt > 0).to_numpy()), ("Market down", (mkt <= 0).to_numpy()),
                            ("10\\% worst market months", worst.to_numpy()), ("NBER recessions", rec),
                            ("Expansions", ~rec), ("All months", np.ones(len(s), bool))]]
    save("tab_loadings", tabular(left, ["FF6 regression", "Coefficient", "$t$"], "lrr"))
    save("tab_conditional", tabular(right, ["Months", "Strategy (\\%)", "Market excess (\\%)", "N"], "lrrr"))


# ------------------------------------------------------------------ figures
def style():
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})


def fig_growth(r_ho: pd.DataFrame, factors: pd.DataFrame) -> None:
    f = factors.reindex(r_ho.index)
    spread = r_ho["10-1 net"].loc[FULL[0]:FULL[1]]
    mkt = (f["RF"] + f["MKT_RF"]).loc[FULL[0]:FULL[1]]
    ls = f["RF"].loc[FULL[0]:FULL[1]] + spread          # market-neutral fund: T-bill + spread
    overlay = mkt + spread                              # market + 1 x long-short spread
    t = ls.index.to_timestamp()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(t, (1 + overlay).cumprod(), color="#1f4e79", lw=1.5, label="Market + strategy (overlay)")
    ax.plot(t, (1 + mkt).cumprod(), color="#999999", lw=1.2, label="US stock market")
    ax.plot(t, (1 + ls).cumprod(), color="#c0504d", lw=1.2, ls="--",
            label="Strategy alone (T-bill + long-short, net)")
    ax.set_yscale("log")
    ax.axvspan(pd.Timestamp("2016-01-01"), t[-1], color="#f2c14e", alpha=0.25, lw=0, label="Holdout 2016-2025")
    ax.set_ylabel("Value of $1 invested in July 1963 (log scale)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_F / "fig_growth.pdf")
    plt.close(fig)
    print("  figures/fig_growth.pdf")


def fig_deciles(r: pd.DataFrame, factors: pd.DataFrame) -> None:
    ports = [c for c in r.columns if c.isdigit()]
    means = [r[c].loc[IN_SAMPLE[0]:IN_SAMPLE[1]].mean() * 100 for c in ports]
    q5 = [_excess_alpha(r[c], factors, "q5")[0] for c in ports]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(ports))
    ax.bar(x - 0.2, means, 0.4, color="#1f4e79", label="Mean return")
    ax.bar(x + 0.2, q5, 0.4, color="#f2c14e", label="q5 alpha")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x, ["1\nexpensive,\nunprofitable"] + ports[1:-1] + ["10\ncheap,\nprofitable"])
    ax.set_ylabel("% per month")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_F / "fig_deciles.pdf")
    plt.close(fig)
    print("  figures/fig_deciles.pdf")


def fig_years(r_ho: pd.DataFrame, factors: pd.DataFrame) -> None:
    s = r_ho["10-1 net"].loc[FULL[0]:FULL[1]]
    mkt = (factors["MKT_RF"] + factors["RF"]).reindex(s.index)
    ys = ((1 + s).groupby(s.index.year).prod() - 1) * 100
    ym = ((1 + mkt).groupby(mkt.index.year).prod() - 1) * 100
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    colors = ["#f2c14e" if y >= 2016 else "#1f4e79" for y in ys.index]
    ax.bar(ys.index, ys.values, color=colors, width=0.8, label="Long-short, net (gold: holdout)")
    ax.plot(ym.index, ym.values, color="#999999", lw=0, marker="o", ms=2.5, label="US stock market")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Calendar-year return (%)")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.set_ylim(-45, 55)
    fig.tight_layout()
    fig.savefig(OUT_F / "fig_years.pdf")
    plt.close(fig)
    print("  figures/fig_years.pdf")


def fig_industries() -> None:
    """Figure 1: value weight of each industry in the universe and in the two legs."""
    f = RES / "paper" / "industry_weights.csv"
    if not f.exists():
        print("  !! no results/paper/industry_weights.csv: run python scripts/describe_legs.py")
        return
    w = pd.read_csv(f, index_col=0).iloc[::-1]
    order = (["All sorted stocks"] + [c for c in w.columns if c.startswith("Long")]
             + [c for c in w.columns if c.startswith("Short")])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    y = np.arange(len(w))
    for i, (col, c) in enumerate(zip(order, ["#999999", "#1f4e79", "#c0504d"])):
        ax.barh(y + (i - 1) * 0.27, w[col], 0.27, color=c, label=col)
    ax.set_yticks(y, w.index)
    ax.set_xlabel("Average value weight (%)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_F / "fig_industries.pdf")
    plt.close(fig)
    print("  figures/fig_industries.pdf")


def fig_rolling(r_ho: pd.DataFrame, factors: pd.DataFrame, window: int = 120) -> None:
    """Figure 3: net return and net FF6 alpha over the previous 10 years, % per year."""
    s = r_ho["10-1 net"].loc[FULL[0]:FULL[1]]
    cols = MODELS["FF6"]["factors"]
    f = factors.reindex(s.index)[cols]
    ok = s.notna() & f.notna().all(axis=1)
    s, f = s[ok], f[ok]
    ret, alp = [], []
    for i in range(window, len(s) + 1):
        y = s.iloc[i - window:i].to_numpy()
        X = np.column_stack([np.ones(window), f.iloc[i - window:i].to_numpy()])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        ret.append(y.mean() * 1200)
        alp.append(b[0] * 1200)
    ends = s.index[window - 1:]
    pd.DataFrame({"net return % per year": ret, "FF6 alpha % per year": alp},
                 index=ends).to_csv(OUT_T.parent / "rolling_10y.csv")
    idx = ends.to_timestamp()
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(idx, ret, color="#999999", lw=1.2, label="Net return")
    ax.plot(idx, alp, color="#1f4e79", lw=1.5, label="Net FF6 alpha")
    ax.axhline(0, color="black", lw=0.6)
    ax.axvspan(pd.Timestamp("2016-01-01"), idx[-1], color="#f2c14e", alpha=0.25, lw=0,
               label="Windows ending in the holdout")
    ax.set_ylabel("% per year, previous 10 years")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_F / "fig_rolling.pdf")
    plt.close(fig)
    print("  figures/fig_rolling.pdf  (numbers in paper/rolling_10y.csv)")


def main() -> None:
    OUT_T.mkdir(parents=True, exist_ok=True)
    OUT_F.mkdir(parents=True, exist_ok=True)
    factors = load_factors(BASE_DIR / "data" / "external" / "factors_monthly.csv")
    ins = {w: read(need(SIG / f"returns_{w}.csv")) for w in ["vw", "ew"]}
    ho = {w: read(need(SIG / "holdout" / f"returns_{w}.csv")) for w in ["vw", "ew"]}
    to = {w: read(need(SIG / f"turnover_{w}.csv")) for w in ["vw", "ew"]}
    co = {w: read(need(SIG / f"costs_{w}.csv")) for w in ["vw", "ew"]}
    n = read(need(SIG / "holdout" / "n_stocks.csv"))
    print("Tables:")
    table_sample(n)
    table_legs()
    table_deciles(ins["vw"], factors)
    table_spread(ins, to, co, factors)
    table_blocks(factors)
    table_holdout(ins, ho, factors)
    table_robust()
    table_costs()
    table_overfit()
    table_behaviour(ins["vw"], factors)
    print("Figures:")
    style()
    fig_industries()
    fig_rolling(ho["vw"], factors)
    fig_growth(ho["vw"], factors)
    fig_deciles(ins["vw"], factors)
    fig_years(ho["vw"], factors)
    print(f"\nDone: upload paper/tables/ and paper/figures/ to Overleaf (same folder names).")


if __name__ == "__main__":
    main()
