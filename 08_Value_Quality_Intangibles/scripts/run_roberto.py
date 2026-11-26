"""
Reproduce every result of the final strategy (docs/STRATEGY.md) in one command.

Run from the project folder, after the team's data step (data/interim/ must exist:
python src/data/clean.py):
    python scripts/run_roberto.py                # everything, 2 backtests at a time
    python scripts/run_roberto.py --workers 3    # faster if the computer has the memory
    python scripts/run_roberto.py --summary-only # only rebuild the comparison table

Team standard: no microcaps (configs/strategies.yaml: stocks above the NYSE 20th size
percentile), value-weighted, after trading costs, 1963-07 to 2015-12. The result with
every stock is the "all_stocks" robustness row. The holdout (2016-2025) is never run here.

What it runs:
  1. Backtests, several at the same time:
       FINAL (qvi_intan_full) with configs/strategies.yaml (all 7 robustness variants)
       the comparison signals with configs/fast.yaml (same data, universe, costs and
       timing, fewer factor models and robustness variants, so faster):
         quality_value_industry        the first version (no intangibles)
         quality_value_industry_intan  intangibles in the book value only
         quality_value_industry_ff49   the first version with 49 industries
         quality_value                 quality + value without industries
         value_industry, cash_op       the two halves alone
  2. For FINAL: analyse_strategy.py (luck, one year, market, mix), run_capacity.py,
     yearly_performance.py and make_charts.py (results/charts/performance.html)
  3. A comparison table (results/roberto/summary.txt and .csv), including each signal's
     monthly net return minus the first version's (quality_value_industry) and its t:
     t above 2 = reliably better than the first version, not luck
Every script's output is saved in results/roberto/logs/.
"""
import argparse
import importlib
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

FINAL = "qvi_intan_full"
FINAL_CONFIG = "configs/strategies.yaml"
COMPARE = ["quality_value_industry", "quality_value_industry_intan", "quality_value_industry_ff49",
           "quality_value", "value_industry", "cash_op"]
COMPARE_CONFIG = "configs/fast.yaml"
AFTER = [["analyse_strategy.py", "--signal", FINAL, "--config", FINAL_CONFIG],
         ["run_capacity.py", "--signal", FINAL, "--config", FINAL_CONFIG],
         ["yearly_performance.py", "--signal", FINAL],
         ["make_charts.py", "--signal", FINAL]]
PERIODS = {"1963-1994": ("1963-07", "1994-12"), "1995-2015": ("1995-01", "2015-12"),
           "1963-2015": ("1963-07", "2015-12")}
MODELS = ["CAPM", "FF5", "FF6", "q5"]
OUT = BASE_DIR / "results" / "roberto"
NEEDED = [BASE_DIR / "data" / "interim" / f for f in ("crsp_clean.parquet", "compustat_clean.parquet",
                                                      "spreads_clean.parquet")]


def run(step: str, args: list) -> int:
    log = OUT / "logs" / f"{step}.txt"
    with log.open("w") as fh:
        res = subprocess.run([sys.executable, *args], cwd=BASE_DIR, stdout=fh, stderr=subprocess.STDOUT)
    status = "ok" if res.returncode == 0 else f"FAILED (see {log.relative_to(BASE_DIR)})"
    print(f"  {step:40s} {status}", flush=True)
    return res.returncode


def backtest(signal: str) -> int:
    cfg = FINAL_CONFIG if signal == FINAL else COMPARE_CONFIG
    return run(f"signal_{signal}", ["scripts/run_signal.py", "--signal", signal, "--config", cfg])


def t_stat(x: pd.Series) -> float:
    return x.mean() / x.std() * np.sqrt(len(x))


def summary(signals: list) -> pd.DataFrame:
    cfg = yaml.safe_load((BASE_DIR / FINAL_CONFIG).read_text())
    f = load_factors(BASE_DIR / cfg["data"]["factors"])
    rows, nets = [], {}
    for s in signals:
        folder = BASE_DIR / "results" / "signals" / s
        if not (folder / "returns_vw.csv").exists():
            print(f"  (no results for {s})")
            continue
        m = importlib.import_module(f"src.signals.{s}")
        r = pd.read_csv(folder / "returns_vw.csv", index_col=0)
        r.index = pd.PeriodIndex(r.index, freq="M")
        net = r[f"{m.LONG}-{m.SHORT} net"]
        nets[s] = net.loc["1963-07":"2015-12"]
        row = {"signal": s}
        for p, (a, b) in PERIODS.items():
            x = net.loc[a:b].dropna()
            row[f"{p} net %"], row[f"{p} t"] = x.mean() * 100, t_stat(x)
        x = net.loc["1963-07":"2015-12"].dropna()
        row["1963-2015 Sharpe"] = x.mean() / x.std() * np.sqrt(12)
        for model in MODELS:
            res = regress(x, f, model, long_short=True, nw_lags=6)
            row[f"{model} net alpha %"], row[f"{model} t"] = res["alpha %"], res["t(alpha)"]
        rob = folder / "robustness.csv"
        if rob.exists():
            rb = pd.read_csv(rob, index_col=[0, 1, 2])
            if ("all_stocks", "full", "vw") in rb.index:
                row["all stocks net %"] = rb.loc[("all_stocks", "full", "vw"), "Net avg %"]
                row["all stocks q5 net alpha %"] = rb.loc[("all_stocks", "full", "vw"), "q5 net alpha %"]
        rows.append(row)
    table = pd.DataFrame(rows).set_index("signal")
    base = "quality_value_industry"  # is a signal reliably better than the first version?
    if base in nets:
        for s in table.index:
            for p, (a, b) in [("1963-2015", PERIODS["1963-2015"]), ("1995-2015", PERIODS["1995-2015"])]:
                d = (nets[s] - nets[base]).loc[a:b].dropna()
                table.loc[s, f"minus {base} {p} %"] = d.mean() * 100 if s != base else np.nan
                table.loc[s, f"minus {base} {p} t"] = t_stat(d) if s != base else np.nan
    table.to_csv(OUT / "summary.csv")
    text = table.T.to_string(float_format=lambda v: f"{v:.2f}")
    (OUT / "summary.txt").write_text(
        "Value-weighted long-short spread 10-1 AFTER costs, no microcaps, per month "
        f"({FINAL} with {FINAL_CONFIG}, the others with {COMPARE_CONFIG})\n\n{text}\n")
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=2, help="backtests at the same time (each needs a few GB)")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    missing = [str(p.relative_to(BASE_DIR)) for p in NEEDED if not p.exists()]
    if missing:
        sys.exit(f"Missing {missing}: run the team's data step first (python src/data/clean.py).")
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    signals = [FINAL, *COMPARE]

    if not args.summary_only:
        print(f"1. Backtests, {args.workers} at a time (FINAL = {FINAL}, slowest, starts first)")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            codes = dict(zip(signals, pool.map(backtest, signals)))
        if codes[FINAL]:
            sys.exit(f"The final strategy failed: see results/roberto/logs/signal_{FINAL}.txt")
        print("2. Analysis of the final strategy")
        for cmd in AFTER:
            run(cmd[0][:-3], [f"scripts/{cmd[0]}", *cmd[1:]])

    pd.set_option("display.width", 220)
    print("\n3. Comparison (value-weighted, after costs, % per month, no microcaps):")
    print(summary(signals).T.to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"\nOutputs: results/roberto/ (summary, logs), results/analysis/{FINAL}_strategies/, "
          f"results/charts/performance.html")


if __name__ == "__main__":
    main()
