"""
Fast strategy search: backtest several signals in parallel with configs/fast.yaml and
write one comparison table.

    python scripts/run_fast.py                          # the qvi_* ideas + the 2 baselines
    python scripts/run_fast.py --signals qvi_intensity
    python scripts/run_fast.py --workers 3              # signals run at the same time
    python scripts/run_fast.py --summary-only           # only rebuild the table

Why it is faster than run_roberto.py / run_all.py, with the same results:
  - configs/fast.yaml: same data, universe, costs and timing as strategies.yaml, but
    5 factor models and 4 robustness variants (each variant is a full backtest)
  - several signals run at the same time (--workers, default 2; each one needs a few
    GB of memory, so do not go above 3-4 on a laptop)
Results: results/signals/<signal>/ as usual (overwritten by the next run of that
signal) and a copy in results/fast/<signal>/; logs in results/fast/logs/; the
comparison in results/fast/summary.txt (and .csv).

Summary, value-weighted and equal-weighted long-short spread (10-1):
  gross and net mean (% per month, t) for 1963-2015, 1963-1994, 1995-2015, 2005-2015
  and by decade; gross and net alphas (Newey-West t, 6 lags) against CAPM, FF5, FF6,
  q5 and q5+STR for the four main periods; net q5 alpha in the robustness universes.
The holdout (2016-2025) is never run here.
"""
import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.evaluation.alpha import load_factors, regress  # noqa: E402

CONFIG = "configs/fast.yaml"                  # annual signals
CONFIG_MONTHLY = "configs/fast_quarterly.yaml"  # monthly signals: held 3 months + variants
# October 15 batch: timely value (current price, held 3 months) + the two additions that
# worked on the annual strategy (payout yield, innovation ability); baselines: the final
# annual strategy and its two best annual variants
IDEAS = ["qvi_timely"]
BASELINES = ["qvi_intan_full"]


def config_for(signal: str) -> str:
    """configs/fast_monthly.yaml for monthly signals, configs/fast.yaml otherwise."""
    from src.signals import load_signal
    return CONFIG_MONTHLY if load_signal(signal).REBALANCE == "monthly" else CONFIG
OUT = BASE_DIR / "results" / "fast"
MAIN_PERIODS = {"1963-2015": ("1963-07", "2015-12"), "1963-1994": ("1963-07", "1994-12"),
                "1995-2015": ("1995-01", "2015-12"), "2005-2015": ("2005-01", "2015-12")}
DECADES = {"1960s": ("1963-07", "1969-12"), "1970s": ("1970-01", "1979-12"),
           "1980s": ("1980-01", "1989-12"), "1990s": ("1990-01", "1999-12"),
           "2000s": ("2000-01", "2009-12"), "2010-15": ("2010-01", "2015-12")}
MODELS = ["CAPM", "FF5", "FF6", "q5", "q5+STR"]


def run_one(signal: str) -> tuple:
    log = OUT / "logs" / f"{signal}.txt"
    with log.open("w") as fh:
        res = subprocess.run([sys.executable, "scripts/run_signal.py", "--signal", signal,
                              "--config", config_for(signal)], cwd=BASE_DIR, stdout=fh,
                             stderr=subprocess.STDOUT)
    if res.returncode == 0:
        src = BASE_DIR / "results" / "signals" / signal
        dst = OUT / signal
        dst.mkdir(parents=True, exist_ok=True)
        for f in ["returns_vw.csv", "returns_ew.csv", "robustness.csv", "spread.csv", "costs_summary.csv"]:
            if (src / f).exists():
                shutil.copy(src / f, dst / f)
    return signal, res.returncode


def t_stat(x: pd.Series) -> float:
    return x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 else np.nan


def summarise(signals: list) -> pd.DataFrame:
    cfg = yaml.safe_load((BASE_DIR / CONFIG).read_text())
    factors = load_factors(BASE_DIR / cfg["data"]["factors"])
    rows = []
    for s in signals:
        folder = OUT / s
        if not (folder / "returns_vw.csv").exists():
            print(f"  (no results for {s})")
            continue
        for w in ["vw", "ew"]:
            r = pd.read_csv(folder / f"returns_{w}.csv", index_col=0)
            r.index = pd.PeriodIndex(r.index, freq="M")
            gross = r[[c for c in r.columns if "-" in c and not c.endswith("net")][0]]
            net = r[[c for c in r.columns if c.endswith(" net")][0]]
            row = {"signal": s, "w": w}
            for p, (a, b) in {**MAIN_PERIODS, **DECADES}.items():
                for lab, x in [("gross", gross), ("net", net)]:
                    x = x.loc[a:b].dropna()
                    row[f"{p} {lab} %"] = x.mean() * 100
                    row[f"{p} {lab} t"] = t_stat(x)
            for p, (a, b) in MAIN_PERIODS.items():
                for lab, x in [("gross", gross), ("net", net)]:
                    for m in MODELS:
                        res = regress(x, factors, m, long_short=True, nw_lags=6, start=a, end=b)
                        row[f"{p} {lab} {m} alpha %"] = res["alpha %"]
                        row[f"{p} {lab} {m} t"] = res["t(alpha)"]
            rob = folder / "robustness.csv"
            if rob.exists():
                rb = pd.read_csv(rob, index_col=[0, 1, 2])
                for var in rb.index.get_level_values(0).unique():
                    key = (var, "full", w)
                    if key in rb.index:
                        row[f"[{var}] net %"] = rb.loc[key, "Net avg %"]
                        row[f"[{var}] q5 net alpha %"] = rb.loc[key, "q5 net alpha %"]
                        row[f"[{var}] q5 net t"] = rb.loc[key, "q5 net t"]
                        row[f"[{var}] turnover %"] = (rb.loc[key, "Turnover % long"]
                                                      + rb.loc[key, "Turnover % short"]) / 2
            rows.append(row)
    return pd.DataFrame(rows).set_index(["signal", "w"])


def pretty(table: pd.DataFrame) -> str:
    """Readable blocks: for each weighting, signals in columns."""
    out = []
    for w in ["vw", "ew"]:
        t = table.xs(w, level="w").T
        out.append(f"\n================ {w.upper()} long-short spread 10-1 (per month) ================")
        blocks = [
            ("Mean return, % (t)", [c for c in t.index if c.endswith(" %") and " alpha" not in c
                                    and not c.startswith("[")]),
            ("Alphas, % (Newey-West t)", [c for c in t.index if " alpha %" in c and not c.startswith("[")]),
            ("Robustness universes (full sample, net)", [c for c in t.index if c.startswith("[")]),
        ]
        for title, cols in blocks:
            out.append(f"\n--- {title} ---")
            lines = []
            for c in cols:
                if title.startswith("Robust"):
                    lines.append([c] + [f"{v:.2f}" for v in t.loc[c]])
                else:
                    tcol = c.replace(" alpha %", " t").replace(" %", " t")
                    lines.append([c.replace(" alpha %", "").replace(" %", "")]
                                 + [f"{a:.2f} ({b:.1f})" for a, b in zip(t.loc[c], t.loc[tcol])])
            frame = pd.DataFrame(lines, columns=["", *t.columns]).set_index("")
            out.append(frame.to_string())
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signals", nargs="+", default=IDEAS + BASELINES)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)

    if not args.summary_only:
        print(f"Backtesting {len(args.signals)} signals ({CONFIG} for annual, {CONFIG_MONTHLY} for "
              f"monthly ones), {args.workers} at a time "
              f"(logs in results/fast/logs/)...")
        failed = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for signal, code in pool.map(run_one, args.signals):
                print(f"  {signal:28s} {'ok' if code == 0 else 'FAILED (see results/fast/logs/' + signal + '.txt)'}",
                      flush=True)
                if code:
                    failed.append(signal)
        if failed:
            print(f"!! Failed: {failed}")

    table = summarise(args.signals)
    table.to_csv(OUT / "summary.csv")
    text = pretty(table)
    (OUT / "summary.txt").write_text(f"Fast comparison ({CONFIG}), 1963-2015 (holdout not used)\n{text}\n")
    print(text)
    print(f"\nSaved results/fast/summary.txt and summary.csv")


if __name__ == "__main__":
    main()
