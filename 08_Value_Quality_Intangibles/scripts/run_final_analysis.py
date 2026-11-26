"""
Every result of the FINAL strategy (qvi_timely, sorted every 3 months) in one command.

Run from the project folder (data/interim/ must exist):
    python scripts/run_final_analysis.py                    # everything (about 1-2 hours)
    python scripts/run_final_analysis.py --skip-backtest    # reuse results/signals/qvi_timely/
    python scripts/run_final_analysis.py --only overfit     # one step (names below)
    python scripts/run_final_analysis.py --quick-overfit    # overfitting tests without backtests

Steps (each output also saved in results/final/logs/<step>.txt):
  backtest    run_signal.py with configs/fast_quarterly.yaml: EW and VW, all factor
              models, train / validation / last 30 / last 20 years, robustness
              (all stocks, large caps, hold 1 / 6 months, buffer)
  analyse     analyse_strategy.py: luck (bootstrap, t > 3), one year? (without the best years, 2008, recessions,
              rolling 10 years, decades), market (factor loadings, crises), overlay
  capacity    run_capacity.py: fund of $100m / $500m / $1bn / $2.5bn per leg
  costs       run_capacity.py --scenarios: spread only, sqrt impact, $10m, 5 days to trade
  yearly      yearly_performance.py: every year, long-short vs long-only vs market
  charts      make_charts.py: results/charts/performance.html
  overfit     overfit_tests.py: placebo, random half universes, neighbouring parameters
The holdout (2016-2025) is never run here.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SIGNAL = "qvi_timely"
CONFIG = "configs/fast_quarterly.yaml"
LOGS = BASE_DIR / "results" / "final" / "logs"


def steps(args) -> dict:
    s, c = ["--signal", SIGNAL], ["--config", CONFIG]
    overfit = ["overfit_tests.py", *s, *c, "--n-placebo", str(args.n_placebo), "--n-splits", str(args.n_splits)]
    if args.quick_overfit:
        overfit.append("--quick")
    return {
        "backtest": ["run_signal.py", *s, *c],
        "analyse": ["analyse_strategy.py", *s, *c],
        "capacity": ["run_capacity.py", *s, *c],
        "costs": ["run_capacity.py", *s, *c, "--scenarios"],
        "yearly": ["yearly_performance.py", *s],
        "charts": ["make_charts.py", *s],
        "overfit": overfit,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-backtest", action="store_true")
    ap.add_argument("--only", nargs="+", help="run only these steps")
    ap.add_argument("--n-placebo", type=int, default=20)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--quick-overfit", action="store_true")
    args = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    todo = steps(args)
    names = args.only or [n for n in todo if not (args.skip_backtest and n == "backtest")]
    failed = []
    for name in names:
        cmd = [sys.executable, f"scripts/{todo[name][0]}", *todo[name][1:]]
        log = LOGS / f"{name}.txt"
        print(f"[{time.strftime('%H:%M')}] {name}: {' '.join(cmd[1:])}  (log: results/final/logs/{name}.txt)",
              flush=True)
        t0 = time.time()
        with log.open("w") as fh:   # output on screen AND in the log
            proc = subprocess.Popen(cmd, cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                sys.stdout.write("    " + line)
                fh.write(line)
            code = proc.wait()
        print(f"    -> {'ok' if code == 0 else 'FAILED'} in {(time.time() - t0) / 60:.1f} min\n", flush=True)
        if code:
            failed.append(name)
            if name == "backtest":
                sys.exit("The backtest failed: the other steps need it. See the log.")
    print("Done." if not failed else f"Failed steps: {failed} (see results/final/logs/)")
    print("Results: results/signals/qvi_timely/, results/analysis/, results/capacity/, "
          "results/charts/performance.html, results/overfit/qvi_timely/summary.txt")


if __name__ == "__main__":
    main()
