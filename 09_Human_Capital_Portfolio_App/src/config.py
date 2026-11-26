"""Shared helper: read the project parameters.

Usage:
    from src.config import load_params
    params = load_params()
    params["backtest"]["window_months"]   # -> 60
"""
from pathlib import Path

import yaml

PARAMS_PATH = Path(__file__).resolve().parent.parent / "config" / "params.yaml"


def load_params(path=PARAMS_PATH):
    """Return the content of config/params.yaml as a dictionary."""
    with open(path) as f:
        return yaml.safe_load(f)
