"""
Run the whole pipeline in one go, in the order each step needs:
  0. check the inputs: the WRDS files in data/raw/ and the factor file (restored
     from git if it is missing; never rebuilt, so the team keeps one vintage)
  1. clean the WRDS data -> data/interim/ (everything below reads it)
  2. reproduce the professor's asset growth numbers (regression test)
  3. for every signal in src/signals/: compare with Ken French (if it has a
     FRENCH_BENCHMARK), then backtest on train + validation

Run from the project folder:
    python scripts/run_all.py
    python scripts/run_all.py --signals momentum book_to_market
    python scripts/run_all.py --skip-clean      # reuse data/interim/ (faster)
    python scripts/run_all.py --config configs/my_test.yaml
    python scripts/run_all.py --final --skip-clean   # only our final strategies
                                                     # (final_signals in configs/strategies.yaml)

Never touches the test period: there is no --holdout here on purpose. Run
    python scripts/run_signal.py --signal <name> --holdout
by hand, once, when the signal and the config are frozen.

Each step runs as its own process. If the cleaning fails, nothing else runs
(every later step needs its output). If a later step fails, the others still
run, and the summary at the end lists what failed. The full output of every
step is saved in results/logs/<step>.log.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))  # lets Python find the "src" folder

from src.signals import available, load_signal  # noqa: E402

RAW_DIR = BASE_DIR / "data" / "raw"
RAW_FILES = ["crsp_monthly.csv", "ccm_funda.csv", "crsp_daily.parquet"]
INTERIM_DIR = BASE_DIR / "data" / "interim"
CLEAN_FILES = ["crsp_clean.parquet", "compustat_clean.parquet", "spreads_clean.parquet"]
LOG_DIR = BASE_DIR / "results" / "logs"

# --final: our final strategies, with the project config, no validation/regression.
# The list of strategies is the key final_signals of that config (edit it there).
FINAL_CONFIG = "configs/strategies.yaml"


def final_signals() -> list:
    """The signals listed under final_signals in FINAL_CONFIG ([] if none)."""
    path = BASE_DIR / FINAL_CONFIG
    names = (yaml.safe_load(path.read_text()) or {}).get("final_signals") if path.exists() else None
    return [names] if isinstance(names, str) else list(names or [])


def check_inputs(skip_clean: bool, factors_file: Path) -> None:
    """Stop before running anything if an input is missing."""
    if skip_clean:
        missing = [INTERIM_DIR / f for f in CLEAN_FILES if not (INTERIM_DIR / f).exists()]
        hint = "Run without --skip-clean to build them from data/raw/."
    else:
        missing = [RAW_DIR / f for f in RAW_FILES if not (RAW_DIR / f).exists()]
        hint = ("Download them from WRDS and put them in data/raw/ with exactly these names.\n"
                "crsp_daily.parquet: put crsp_daily.csv in data/raw/ and run "
                "`python convert_to_parquet.py crsp_daily.csv`.")
    if missing:
        sys.exit("Missing input files:\n" + "\n".join(f"  {p}" for p in missing) + f"\n{hint}")

    if not factors_file.exists():
        # The factor file is in git (one vintage for the whole team): restore it
        # instead of downloading a new one with src/data/factors.py.
        rel = factors_file.relative_to(BASE_DIR).as_posix()
        print(f"{rel} is missing: restoring the team's copy from git...")
        subprocess.run(["git", "checkout", "HEAD", "--", rel], cwd=BASE_DIR)
        if not factors_file.exists():
            sys.exit(f"Could not restore {rel}. Run `git pull`, or get the file from a "
                     "teammate. Only rebuild it on purpose (src/data/factors.py).")


def run_step(label: str, args: list) -> tuple:
    """Run one script, show its output live and save it to a log file.
    Returns (ok, seconds)."""
    print(f"\n{'=' * 78}\n>> {label}\n{'=' * 78}", flush=True)
    log_file = LOG_DIR / f"{label.replace(' ', '_').replace(':', '')}.log"
    start = time.time()
    with log_file.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, "-u", *args], cwd=BASE_DIR, text=True, encoding="utf-8",
            errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
        proc.wait()
    seconds = time.time() - start
    ok = proc.returncode == 0
    if not ok:
        print(f"!! {label} failed (exit code {proc.returncode}), see {log_file}")
    return ok, seconds


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the whole pipeline.")
    parser.add_argument("--signals", nargs="+", default=None,
                        help=f"signals to run (default: all of {', '.join(available())})")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--skip-clean", action="store_true",
                        help="reuse the cleaned data already in data/interim/")
    parser.add_argument("--skip-validation", action="store_true", help="skip the Ken French comparisons")
    parser.add_argument("--skip-regression", action="store_true", help="skip run_asset_growth.py")
    parser.add_argument("--final", action="store_true",
                        help=f"only the final strategies (final_signals in {FINAL_CONFIG}: "
                             f"{', '.join(final_signals()) or 'none listed'}) with that config, "
                             "skipping validation and regression")
    args = parser.parse_args()
    if args.final:
        args.signals = args.signals or final_signals()
        if not args.signals:
            sys.exit(f"--final: no final_signals listed in {FINAL_CONFIG}.")
        args.config = FINAL_CONFIG
        args.skip_validation = args.skip_regression = True

    signals = args.signals or available()
    for name in signals:
        load_signal(name)  # fails now, not halfway through, if a name is wrong

    # 0. Inputs
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = BASE_DIR / config_path
    cfg = yaml.safe_load(config_path.read_text())
    check_inputs(args.skip_clean, BASE_DIR / cfg["data"]["factors"])

    # 1-3. Steps, in the order they depend on each other
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    steps = []  # (label, script arguments)
    if not args.skip_clean:
        steps.append(("clean data", ["src/data/clean.py"]))
    if not args.skip_regression:
        steps.append(("regression asset_growth", ["scripts/run_asset_growth.py"]))
    for name in signals:
        if not args.skip_validation and getattr(load_signal(name), "FRENCH_BENCHMARK", None):
            steps.append((f"validate {name}", ["scripts/validate_signal.py", "--signal", name]))
        steps.append((f"backtest {name}",
                      ["scripts/run_signal.py", "--signal", name, "--config", args.config]))

    print(f"{len(steps)} steps: " + ", ".join(label for label, _ in steps))
    results = []
    for label, script_args in steps:
        ok, seconds = run_step(label, script_args)
        results.append((label, ok, seconds))
        if label == "clean data" and not ok:
            print("!! Cleaning failed: everything after it needs the clean data, stopping.")
            break

    print(f"\n{'=' * 78}\nSummary\n{'=' * 78}")
    for label, ok, seconds in results:
        print(f"  {'OK    ' if ok else 'FAILED'}  {label:<35} {seconds / 60:5.1f} min")
    print(f"Logs in: {LOG_DIR}")
    if not all(ok for _, ok, _ in results) or len(results) < len(steps):
        sys.exit(1)


if __name__ == "__main__":
    main()