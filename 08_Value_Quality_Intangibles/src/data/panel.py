"""
Load the monthly panel: clean CRSP + only the Compustat columns you ask for.

Instead of one huge CSV with every column, the two clean tables are merged on
demand. Memory then depends only on the columns a strategy really uses.

Example:
    from src.data.panel import load_panel
    panel = load_panel(
        crsp_cols=["PERMNO", "MthCalDt", "formation_year", "PrimaryExch", "MthRet", "mkt_cap"],
        comp_cols=["at", "at_lag"],
    )
"""
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

# Project folder = first folder above this file that contains data/raw
BASE_DIR = next(
    (p for p in Path(__file__).resolve().parents if (p / "data" / "raw").is_dir()),
    Path(__file__).resolve().parents[2],
)
INTERIM_DIR = BASE_DIR / "data" / "interim"
CRSP_FILE = INTERIM_DIR / "crsp_clean.parquet"
COMP_FILE = INTERIM_DIR / "compustat_clean.parquet"
SPREAD_FILE = INTERIM_DIR / "spreads_clean.parquet"

KEYS = ["PERMNO", "MthCalDt", "formation_year"]


def load_spreads() -> pd.DataFrame:
    """Trading cost inputs of each stock-month (written by src/data/clean.py):
    PERMNO, month (monthly period, like the holdings of sort_portfolios),
    half_spread (at the month's last close; estimated before 1993 for NYSE/AMEX),
    spread_source, size_group (micro/small/large) and adv (daily dollar volume)."""
    cols = ["PERMNO", "month_end", "half_spread", "spread_source", "size_group", "adv"]
    spreads = pd.read_parquet(SPREAD_FILE, columns=cols)
    spreads["month"] = spreads["month_end"].dt.to_period("M")
    return spreads[["PERMNO", "month"] + cols[2:]]


def load_panel(
    crsp_cols: Optional[List[str]] = None,
    comp_cols: Union[None, str, List[str]] = None,
    check: bool = True,
) -> pd.DataFrame:
    """
    crsp_cols: CRSP columns to load (None = all).
    comp_cols: Compustat columns to add (None = none, "all" = all).
    Compustat data of fiscal year t-1 is attached to the months July t - June t+1.
    """
    if crsp_cols is not None:
        crsp_cols = list(dict.fromkeys(KEYS + list(crsp_cols)))
    crsp = pd.read_parquet(CRSP_FILE, columns=crsp_cols)

    if comp_cols is None:
        return crsp.sort_values(["PERMNO", "MthCalDt"]).reset_index(drop=True)

    if comp_cols == "all":
        comp = pd.read_parquet(COMP_FILE)
    else:
        cols = list(dict.fromkeys(["LPERMNO", "formation_year", "datadate"] + list(comp_cols)))
        comp = pd.read_parquet(COMP_FILE, columns=cols)

    panel = crsp.merge(
        comp,
        left_on=["PERMNO", "formation_year"],
        right_on=["LPERMNO", "formation_year"],
        how="left",
    ).drop(columns="LPERMNO")

    if check:
        lag = (
            (panel["MthCalDt"].dt.year - panel["datadate"].dt.year) * 12
            + (panel["MthCalDt"].dt.month - panel["datadate"].dt.month)
        )
        if (lag < 6).any():
            raise ValueError(f"{int((lag < 6).sum())} rows use accounting data less than 6 months old")
        if panel.duplicated(subset=["PERMNO", "MthCalDt"]).any():
            raise ValueError("Duplicate (PERMNO, MthCalDt) rows after the merge")

    return panel.sort_values(["PERMNO", "MthCalDt"]).reset_index(drop=True)