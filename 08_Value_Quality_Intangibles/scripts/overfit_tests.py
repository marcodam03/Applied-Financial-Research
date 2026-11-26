"""
Is the final strategy a real signal or the result of overfitting (we tried many ideas
and kept the best)? Three tests, 1963-07 to 2015-12 (the holdout is never read).

Run from the project folder, after the backtest of the signal with its config
(results/signals/<signal>/):
    python scripts/overfit_tests.py                         (qvi_timely, configs/final.yaml)
    python scripts/overfit_tests.py --quick                 (only C, no backtests: a few seconds)
    python scripts/overfit_tests.py --n-placebo 50 --n-splits 5

A. PLACEBO            the score is shuffled across stocks within each month (same
                      universe, same portfolios, same timing, same costs code), N times.
                      If the pipeline itself made alpha (look-ahead, size or industry
                      tilt, a bug), the placebos would have it too. p-value = share of
                      placebos with a gross q5 / FF6 alpha at least as high as the real one.
B. HALF UNIVERSES     the stocks are split at random into two halves (by PERMNO); the
                      same strategy is run on each half separately, net of costs. A real
                      signal works in both halves; an overfit one (driven by a few
                      stocks) does not.
C. NEIGHBOURS         the strategy with the parameters around the chosen ones (hold 1 /
                      6 months, buffer, large caps, all stocks) from robustness.csv:
                      a real signal does not depend on one precise parameter.
The last, decisive test is the holdout 2016-2025: run ONCE, after freezing the rules
(python scripts/run_signal.py --signal qvi_timely --config configs/final.yaml --holdout).
Output: results/overfit/<signal>/ (csv tables and summary.txt).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "scripts"))

from src.evaluation.alpha import load_factors, regress  # noqa: E402

START, END = "1963-07", "2015-12"
SEED = 11

pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")


def say(text, lines):
    print(text, flush=True)
    lines.append(text)


def read_returns(path: Path) -> pd.DataFrame:
    r = pd.read_csv(path, index_col=0)
    r.index = pd.PeriodIndex(r.index, freq="M")
    return r.loc[START:END]


def net_col(r: pd.DataFrame) -> str:
    return [c for c in r.columns if c.endswith(" net")][0]


def gross_col(r: pd.DataFrame) -> str:
    return [c for c in r.columns if "-" in c and not c.endswith("net")][0]


def tstat(x) -> float:
    x = pd.Series(x).dropna()
    return x.mean() / x.std() * np.sqrt(len(x))


# ------------------------------------------------------------------ A/B. backtests
def shuffle_within_month(panel, col, rng):
    s = panel[col]
    ok = s.notna().to_numpy()
    month = panel["MthCalDt"].to_numpy()[ok]
    month = pd.factorize(month)[0]
    vals = s.to_numpy()[ok]
    base = np.argsort(month, kind="stable")
    mixed = np.lexsort((rng.random(len(month)), month))
    out = np.empty_like(vals)
    out[base] = vals[mixed]
    full = np.full(len(s), np.nan)
    full[ok] = out
    return full


def backtests(signal, cfg, n_placebo, n_splits, lines):
    import run_signal as rs
    from src.data.panel import load_spreads
    from src.signals import build_panel, load_signal

    spec = load_signal(signal)
    portfolio_cfg = {**cfg["portfolio"], "n_portfolios": getattr(spec, "N_PORTFOLIOS", cfg["portfolio"]["n_portfolios"])}
    opts = rs.sort_options(cfg["universe"], portfolio_cfg)
    factors = load_factors(BASE_DIR / cfg["data"]["factors"])
    spread = f"{spec.LONG}-{spec.SHORT}"
    say(f"\nLoading the panel and computing '{signal}' (once)...", lines)
    t0 = time.time()
    panel = build_panel(spec)
    spreads = load_spreads()
    impact, label = rs.cost_settings(panel, spreads, cfg.get("costs"))
    say(f"  done in {time.time() - t0:.0f} s; costs: {label}", lines)

    def run(col, with_impact):
        out = rs.backtest(panel, col, spec, opts, ["vw"], spreads, START, END,
                          impact if with_impact else None)["vw"]
        row = {"gross %": out[spread].mean() * 100, "net %": out[f"{spread} net"].mean() * 100}
        for model in ["FF6", "q5"]:
            for kind, x in [("gross", out[spread]), ("net", out[f"{spread} net"])]:
                res = regress(x.dropna(), factors, model, long_short=True, nw_lags=6)
                row[f"{model} {kind} alpha %"], row[f"{model} {kind} t"] = res["alpha %"], res["t(alpha)"]
        return row

    rng = np.random.default_rng(SEED)
    placebo_tab = None
    if n_placebo:
        say(f"\n===== A. Placebo: score shuffled within each month, {n_placebo} times "
            f"(spread costs only, to be faster) =====", lines)
        t0 = time.time()
        real = run(signal, False)
        rows = []
        for k in range(n_placebo):
            panel["__placebo"] = shuffle_within_month(panel, signal, rng)
            rows.append(run("__placebo", False))
            print(f"  placebo {k + 1}/{n_placebo}: q5 gross alpha {rows[-1]['q5 gross alpha %']:.2f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
        panel.drop(columns="__placebo", inplace=True)
        pl = pd.DataFrame(rows)
        placebo_tab = pd.DataFrame({
            "real": pd.Series(real),
            "placebo mean": pl.mean(), "placebo 5%": pl.quantile(0.05), "placebo 95%": pl.quantile(0.95),
            "p-value (share of placebos >= real)": pd.Series(
                {c: ((pl[c] >= real[c]).sum() + 1) / (len(pl) + 1) for c in pl.columns}),
        })
        say(placebo_tab.to_string(), lines)
        say(f"(smallest possible p-value with {n_placebo} placebos: {1 / (n_placebo + 1):.3f})", lines)

    split_tab = None
    if n_splits:
        say(f"\n===== B. Random half universes ({n_splits} splits x 2 halves, net of costs) =====", lines)
        permnos = panel["PERMNO"].unique()
        rows = []
        t0 = time.time()
        for k in range(n_splits):
            a = set(rng.choice(permnos, len(permnos) // 2, replace=False))
            in_a = panel["PERMNO"].isin(a)
            for half, mask in [("A", in_a), ("B", ~in_a)]:
                panel["__half"] = panel[signal].where(mask)
                rows.append({"split": k + 1, "half": half, **run("__half", True)})
                print(f"  split {k + 1} half {half}: q5 net alpha {rows[-1]['q5 net alpha %']:.2f} "
                      f"(t {rows[-1]['q5 net t']:.1f}) ({time.time() - t0:.0f} s)", flush=True)
        panel.drop(columns="__half", inplace=True)
        split_tab = pd.DataFrame(rows).set_index(["split", "half"])
        cols = ["gross %", "net %", "FF6 net alpha %", "FF6 net t", "q5 net alpha %", "q5 net t"]
        say(split_tab[cols].to_string(), lines)
        say(f"Halves with q5 net alpha > 0: {(split_tab['q5 net alpha %'] > 0).mean() * 100:.0f}%, "
            f"with t > 2: {(split_tab['q5 net t'] > 2).mean() * 100:.0f}%", lines)
    return placebo_tab, split_tab


# ------------------------------------------------------------------ C. neighbours
def neighbours(signal, lines):
    say("\n===== C. Neighbouring parameters (robustness.csv of the last run) =====", lines)
    f = BASE_DIR / "results" / "signals" / signal / "robustness.csv"
    if not f.exists():
        say("No robustness.csv: run the backtest first.", lines)
        return None
    rb = pd.read_csv(f, index_col=[0, 1, 2])
    rb = rb.xs("vw", level=2) if "vw" in rb.index.get_level_values(2) else rb
    keep = [c for c in rb.columns if c in ("Net avg %", "FF6 net alpha %", "FF6 net t",
                                           "q5 net alpha %", "q5 net t")]
    tab = rb[keep]
    say(tab.to_string(), lines)
    return tab


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signal", default="qvi_timely")
    ap.add_argument("--config", default="configs/final.yaml")
    ap.add_argument("--n-placebo", type=int, default=20)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="no backtests: only C")
    args = ap.parse_args()

    cfg = yaml.safe_load((BASE_DIR / args.config).read_text())
    out_dir = BASE_DIR / "results" / "overfit" / args.signal
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    final_file = BASE_DIR / "results" / "signals" / args.signal / "returns_vw.csv"
    if not final_file.exists():
        sys.exit(f"{final_file} not found: run the backtest first "
                 f"(python scripts/run_signal.py --signal {args.signal} --config {args.config})")
    say(f"Overfitting tests for {args.signal} ({args.config}), VW, {START} to {END}", lines)

    if not args.quick:
        pl, sp = backtests(args.signal, cfg, args.n_placebo, args.n_splits, lines)
        if pl is not None:
            pl.to_csv(out_dir / "A_placebo.csv")
        if sp is not None:
            sp.to_csv(out_dir / "B_half_universes.csv")
    nb = neighbours(args.signal, lines)
    if nb is not None:
        nb.to_csv(out_dir / "C_neighbours.csv")
    say("\nLast test: the holdout 2016-2025, ONCE, after freezing the rules (see the docstring).", lines)
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nSaved in {out_dir}")


if __name__ == "__main__":
    main()
