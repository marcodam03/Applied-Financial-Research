"""
Interactive menu: choose a script of the project, answer a few questions (signal,
config, options) and it runs. Before running, the menu prints the exact command,
so you can also copy it and run it yourself next time.

Run from the project folder:
    python scripts/menu.py

At every question: Enter takes the value in [brackets], b goes back to the menu,
q quits. Ctrl+C stops a running script and brings you back to the menu.

Nothing about the signals is written in this file: the lists are read from
src/signals/ (every file there, re-read each time a list is shown, so a signal
added or edited while the menu is open shows up at once), configs/*.yaml, and
final_signals in configs/strategies.yaml (the final strategies, marked *).
"""
import importlib
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))  # lets Python find the "src" folder

from run_all import FINAL_CONFIG, final_signals  # noqa: E402  (scripts/run_all.py)
from src.signals import available, load_signal  # noqa: E402

CONFIG_DIR = BASE_DIR / "configs"
CLEAN_FILES = [BASE_DIR / "data" / "interim" / f
               for f in ("crsp_clean.parquet", "compustat_clean.parquet", "spreads_clean.parquet")]
HOLDOUT_LOG = BASE_DIR / "holdout_log.txt"


class Back(Exception):
    """The user typed b: back to the main menu."""


class Quit(Exception):
    """The user typed q."""


# ------------------------------------------------------------------------------
# Questions
# ------------------------------------------------------------------------------
def ask(prompt: str, default: str = "") -> str:
    shown = f" [{default}]" if default else ""
    answer = input(f"{prompt}{shown}: ").strip()
    if answer.lower() == "b":
        raise Back
    if answer.lower() == "q":
        raise Quit
    return answer or default


def show_list(title: str, options: List[str]) -> None:
    print(f"\n{title}")
    for i, label in enumerate(options, 1):
        print(f"  {i:>2}. {label}")


def pick_one(title: str, options: List[str], default: Optional[int] = None) -> int:
    """Numbered list; returns the position (from 0) of the option chosen."""
    show_list(title, options)
    while True:
        answer = ask("Number", str(default + 1) if default is not None else "")
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer) - 1
        print(f"  Type a number from 1 to {len(options)} (b = back, q = quit).")


def pick_many(title: str, options: List[str]) -> List[int]:
    """Numbered list; several numbers separated by spaces, Enter = all."""
    show_list(title, options)
    while True:
        parts = ask("Numbers separated by spaces (Enter = all)").replace(",", " ").split()
        if not parts:
            return list(range(len(options)))
        if all(p.isdigit() and 1 <= int(p) <= len(options) for p in parts):
            return sorted({int(p) - 1 for p in parts})
        print(f"  Type numbers from 1 to {len(options)}, e.g. 1 3 4 (b = back, q = quit).")


def yes(prompt: str, default: bool) -> bool:
    return ask(f"{prompt} (y/n)", "y" if default else "n").lower().startswith("y")


def shorten(text: str, width: int = 58) -> str:
    return text if len(text) <= width else text[:width - 3] + "..."


def read_signals() -> Dict[str, ModuleType]:
    """Every signal of src/signals/ that loads, read again from its file (not the copy
    Python kept from the last list). A file with an error is left out with a message,
    instead of stopping the menu."""
    importlib.invalidate_caches()
    for module in [m for m in sys.modules if m.startswith("src.signals.")]:
        del sys.modules[module]
    specs = {}
    for name in available():
        try:
            specs[name] = load_signal(name)
        except Exception as e:  # noqa: BLE001 (any error in the signal's file)
            print(f"!! src/signals/{name}.py left out, it does not load: {e.__class__.__name__}: {e}")
    return specs


def signal_labels(specs: Dict[str, ModuleType]) -> List[str]:
    """'* hybrid_combo     monthly  Hybrid: half slow_combo (annual, low turnover), ...'"""
    finals = final_signals()
    width = max(map(len, specs), default=0)
    labels = []
    for name, spec in specs.items():
        lines = (spec.__doc__ or "").strip().splitlines()
        mark = "*" if name in finals else " "
        labels.append(f"{mark} {name:<{width}} {spec.REBALANCE:<8} {shorten(lines[0] if lines else '')}")
    return labels


def pick_signal(specs: Optional[Dict[str, ModuleType]] = None) -> str:
    specs = read_signals() if specs is None else specs
    if not specs:
        print("\n!! No signal to choose from.")
        raise Back
    title = f"Which signal? (* = final strategy, final_signals in {FINAL_CONFIG})"
    return list(specs)[pick_one(title, signal_labels(specs))]


def config_for(signal: str) -> str:
    """Config proposed by default: the final strategies' config for a final strategy."""
    return Path(FINAL_CONFIG).name if signal in final_signals() else "default.yaml"


def pick_config(default: str = "default.yaml") -> str:
    """configs/<file>.yaml, default.yaml first. Each one is shown with its first comment line."""
    files = sorted(CONFIG_DIR.glob("*.yaml"), key=lambda p: (p.name != "default.yaml", p.name))
    labels = []
    for path in files:
        comments = [l for l in path.read_text(encoding="utf-8").splitlines() if l.startswith("#")]
        about = comments[0].lstrip("# ").strip() if comments else ""
        labels.append(f"{path.name:<17} {shorten(about)}")
    names = [p.name for p in files]
    start = names.index(default) if default in names else 0
    return f"configs/{files[pick_one('Which config?', labels, default=start)].name}"


# ------------------------------------------------------------------------------
# Menu entries: each one asks its questions and returns the script and arguments
# ------------------------------------------------------------------------------
def clean_data() -> List[str]:
    print("\nRebuilds data/interim/ from the WRDS files in data/raw/ (a few minutes).")
    print("Afterwards, rerun the asset growth check (it must still match the professor).")
    return ["src/data/clean.py"]


def asset_growth_check() -> List[str]:
    return ["scripts/run_asset_growth.py"]


def asset_growth_alphas() -> List[str]:
    if not (BASE_DIR / "results" / "asset_growth" / "ew_returns.csv").exists():
        print("\n!! This needs results/asset_growth/: run the asset growth check first.")
        raise Back
    return ["scripts/run_alphas_asset_growth.py"]


def validate() -> List[str]:
    specs = {n: s for n, s in read_signals().items() if getattr(s, "FRENCH_BENCHMARK", None)}
    return ["scripts/validate_signal.py", "--signal", pick_signal(specs)]


def backtest() -> List[str]:
    signal = pick_signal()
    return ["scripts/run_signal.py", "--signal", signal, "--config", pick_config(config_for(signal))]


def pipeline() -> List[str]:
    specs = read_signals()
    finals = final_signals()
    mode = pick_one("Which signals?", [
        "All signals",
        f"Only the final strategies (--final, {FINAL_CONFIG}, no French or asset growth "
        f"check): {', '.join(finals) or 'none listed'}",
        "Choose the signals",
    ])
    args = ["scripts/run_all.py"]
    if mode == 1:
        wrong = [n for n in finals if n not in specs]
        if not finals or wrong:
            print(f"\n!! Fix final_signals in {FINAL_CONFIG} first: " + (
                f"no file in src/signals/ or a file that does not load: {', '.join(wrong)}"
                if wrong else "the list is empty."))
            raise Back
        args.append("--final")
    else:
        if mode == 2:
            chosen = pick_many("Which signals? (* = final strategy)", signal_labels(specs))
            args += ["--signals"] + [list(specs)[i] for i in chosen]
        elif len(specs) < len(available()):  # leave out the files that do not load (listed above)
            args += ["--signals", *specs]
        args += ["--config", pick_config()]
        if not yes("Compare with Ken French (signals that have a benchmark)?", True):
            args.append("--skip-validation")
        if not yes("Run the asset growth check against the professor's numbers?", True):
            args.append("--skip-regression")
    if all(f.exists() for f in CLEAN_FILES):
        if not yes("Clean the WRDS data again first? Slow, only needed after a change "
                   "to clean.py or data/raw", False):
            args.append("--skip-clean")
    else:
        print("\nNo cleaned data in data/interim/ yet: the pipeline starts by cleaning it.")
    return args


def holdout() -> List[str]:
    print("\n!! The holdout (2016-01 on) is used ONCE per signal, when the signal and the")
    print("!! config are frozen. Every run adds a line to holdout_log.txt: commit it.")
    print("!! Signals run with configs/strategies.yaml before 2026-10-04 already saw the")
    print("!! holdout years without a line in the log.")
    signal = pick_signal()
    lines = HOLDOUT_LOG.read_text().splitlines() if HOLDOUT_LOG.exists() else []
    previous = [line for line in lines if f"signal={signal}" in line.split()]
    if previous:
        print(f"\n!! The holdout of '{signal}' was already used {len(previous)} time(s):")
        for line in previous:
            print(f"!!   {line}")
    config = pick_config(config_for(signal))
    if ask(f"Type the signal name ({signal}) to confirm") != signal:
        print("Not confirmed: nothing was run.")
        raise Back
    return ["scripts/run_signal.py", "--signal", signal, "--config", config, "--holdout"]


def factors() -> List[str]:
    choice = pick_one("The factor file is the team's shared vintage (in git). What do you want?", [
        "Add or refresh only the short-term reversal column STREV (--add-reversal)",
        "Rebuild the whole file from Ken French and global-q (past values change!)",
    ])
    if choice == 0:
        return ["src/data/factors.py", "--add-reversal"]
    print("\n!! A rebuild downloads the latest vintages: past factor values change a little,")
    print("!! so alphas computed before are no longer comparable. Commit the new file for the team.")
    if ask("Type rebuild to confirm") != "rebuild":
        print("Not confirmed: nothing was run.")
        raise Back
    return ["src/data/factors.py"]


def quit_menu() -> List[str]:
    raise Quit


# In the order of the README: data, checks, signals, holdout, factor file
MENU = [
    ("Clean the WRDS data", "clean.py", clean_data),
    ("Check asset growth vs the professor", "run_asset_growth.py", asset_growth_check),
    ("Asset growth alphas", "run_alphas_asset_growth.py", asset_growth_alphas),
    ("Compare a signal with Ken French", "validate_signal.py", validate),
    ("Backtest a signal", "run_signal.py", backtest),
    ("Run the whole pipeline", "run_all.py", pipeline),
    ("Holdout run of a signal (ONCE)", "run_signal.py --holdout", holdout),
    ("Update the factor file", "factors.py", factors),
    ("Quit", "", quit_menu),
]


def run(args: List[str]) -> None:
    print(f"\nCommand:  python {' '.join(args)}")
    if not yes("Run it?", True):
        return
    try:
        code = subprocess.call([sys.executable, *args], cwd=BASE_DIR)
    except KeyboardInterrupt:
        print("\n!! Stopped (Ctrl+C).")
        return
    print("\nFinished." if code == 0 else f"\n!! Failed (exit code {code}): see the messages above.")


def main() -> None:
    print("Applied Investments: choose what to run (b = back, q = quit).")
    labels = [f"{label:<37} ({script})" if script else label for label, script, _ in MENU]
    while True:
        try:
            run(MENU[pick_one("What do you want to run?", labels)][2]())
        except Back:
            continue
        except (Quit, EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        except Exception as e:  # noqa: BLE001 (e.g. a config being edited): back to the menu
            print(f"\n!! {e.__class__.__name__}: {e}")


if __name__ == "__main__":
    main()
