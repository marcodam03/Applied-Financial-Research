"""
Signals: one file per signal in this folder.

A signal module src/signals/<name>.py defines:
    <name>(panel) -> pd.Series   the signal for every row of the monthly panel
    REBALANCE                    "annual" (accounting signal, sorted at the end of June)
                                 or "monthly" (market signal, sorted every month end)
    COMPUSTAT_COLS               Compustat columns the signal needs ([] if none)
    CRSP_COLS                    optional: CRSP columns needed on top of the usual ones
    LONG, SHORT                  portfolio bought and portfolio sold in the spread
    FRENCH_BENCHMARK             optional: Ken French file with the same portfolios,
                                 used by scripts/validate_signal.py

To add a signal: copy asset_growth.py (accounting data) or momentum.py (market
data), change the formula, then run
    python scripts/run_signal.py --signal <name>
"""
import importlib
from pathlib import Path
from types import ModuleType
from typing import List

import pandas as pd

from src.data.panel import load_panel
from src.portfolio.sorts import OPTIONAL_COLS, REQUIRED_COLS

REQUIRED_ATTRS = ["REBALANCE", "COMPUSTAT_COLS", "LONG", "SHORT"]


def available() -> List[str]:
    """Names of all signal modules in this folder."""
    return sorted(p.stem for p in Path(__file__).parent.glob("*.py") if not p.stem.startswith("_"))


def load_signal(name: str) -> ModuleType:
    if name not in available():
        raise ValueError(f"Unknown signal '{name}'. Available: {', '.join(available())}")
    module = importlib.import_module(f"{__name__}.{name}")
    missing = [a for a in [name] + REQUIRED_ATTRS if not hasattr(module, a)]
    if missing:
        raise AttributeError(f"src/signals/{name}.py must define: {', '.join(missing)}")
    return module


def build_panel(module: ModuleType) -> pd.DataFrame:
    """Load the columns the signal needs and add the signal as a column named
    after the module (e.g. panel["momentum"])."""
    name = module.__name__.rsplit(".", 1)[-1]
    crsp_cols = REQUIRED_COLS + OPTIONAL_COLS + list(getattr(module, "CRSP_COLS", []))
    panel = load_panel(crsp_cols=crsp_cols, comp_cols=list(module.COMPUSTAT_COLS) or None)
    panel[name] = getattr(module, name)(panel)
    return panel
