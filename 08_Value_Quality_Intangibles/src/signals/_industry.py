"""
Industry helpers: Fama-French 12 (main) or 49 (robustness) industries from the CRSP SIC code.

- ff12(sic):            FF12 industry number (1 to 12) for each SIC code, using the
                        official SIC ranges from Ken French's website (Siccodes12).
                        12 = "Other" (everything not listed, as on the website).
- ff49(sic):            FF49 industry number (1 to 49), read from Ken French's official
                        file Siccodes49 (downloaded once to data/external/raw/; if the
                        download fails, save Siccodes49.zip there by hand). SIC codes
                        not listed go to "Other", as on the website.
- industry_june(panel, scheme): FF12 ("ff12", default) or FF49 ("ff49") industry of each
                        stock at the end of June t, one value per (PERMNO, formation_year).
- is_financial(sic):    SIC 6000-6999 (banks, insurers, real estate, other finance).
- industry_rank(frame, industry, min_stocks): percentile rank of each column
                        within (formation year, industry), from -0.5 to +0.5.
                        Groups with fewer than min_stocks stocks get NaN.
"""
import io
import re
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.signals._yearly import KEY, june

NAMES = {1: "NoDur", 2: "Durbl", 3: "Manuf", 4: "Enrgy", 5: "Chems", 6: "BusEq",
         7: "Telcm", 8: "Utils", 9: "Shops", 10: "Hlth", 11: "Money", 12: "Other"}

_RANGES = {
    1: [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)],
    2: [(2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711), (3714, 3714),
        (3716, 3716), (3750, 3751), (3792, 3792), (3900, 3939), (3990, 3999)],
    3: [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569),
        (3580, 3629), (3700, 3709), (3712, 3713), (3715, 3715), (3717, 3749),
        (3752, 3791), (3793, 3799), (3830, 3839), (3860, 3899)],
    4: [(1200, 1399), (2900, 2999)],
    5: [(2800, 2829), (2840, 2899)],
    6: [(3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)],
    7: [(4800, 4899)],
    8: [(4900, 4949)],
    9: [(5000, 5999), (7200, 7299), (7600, 7699)],
    10: [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)],
    11: [(6000, 6999)],
}


def ff12(sic: pd.Series) -> pd.Series:
    sic = pd.to_numeric(sic, errors="coerce")
    out = pd.Series(np.where(sic.notna() & (sic > 0), 12, np.nan), index=sic.index)
    for ind, ranges in _RANGES.items():
        for low, high in ranges:
            out[sic.between(low, high)] = ind
    return out


_HERE = Path(__file__).resolve()
_BASE = next((p for p in _HERE.parents if (p / "data" / "raw").is_dir()), _HERE.parents[2])
FF49_ZIP = _BASE / "data" / "external" / "raw" / "Siccodes49.zip"
FF49_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes49.zip"


@lru_cache(maxsize=1)
def ff49_table():
    """(ranges, names): ranges = list of (low, high, industry number)."""
    if not FF49_ZIP.exists():
        FF49_ZIP.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(FF49_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            FF49_ZIP.write_bytes(r.read())
    with zipfile.ZipFile(FF49_ZIP) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    ranges, names, current = [], {}, None
    for line in io.StringIO(text):
        rng = re.match(r"\s*(\d{4})\s*-\s*(\d{4})", line)
        head = re.match(r"\s*(\d{1,2})\s+(\S+)", line)
        if rng and current is not None:
            ranges.append((int(rng.group(1)), int(rng.group(2)), current))
        elif head:
            current = int(head.group(1))
            names[current] = head.group(2)
    if len(names) != 49 or len(ranges) < 150:
        raise ValueError(f"Siccodes49 parsed badly: {len(names)} industries, {len(ranges)} ranges")
    return ranges, names


def ff49(sic: pd.Series) -> pd.Series:
    ranges, names = ff49_table()
    other = next(k for k, v in names.items() if v.lower() == "other")
    sic = pd.to_numeric(sic, errors="coerce")
    out = pd.Series(np.where(sic.notna() & (sic > 0), other, np.nan), index=sic.index)
    for low, high, ind in ranges:
        out[sic.between(low, high)] = ind
    return out


def is_financial(sic: pd.Series) -> pd.Series:
    return pd.to_numeric(sic, errors="coerce").between(6000, 6999)


def industry_june(panel: pd.DataFrame, scheme: str = "ff12") -> pd.Series:
    mapper = {"ff12": ff12, "ff49": ff49}[scheme]
    ind = june(panel, mapper(panel["SICCD"]))
    return ind[~ind.index.duplicated()].dropna()


def financial_june(panel: pd.DataFrame) -> pd.Series:
    fin = june(panel, is_financial(panel["SICCD"]).astype(float))
    return fin[~fin.index.duplicated()]


def industry_rank(frame: pd.DataFrame, industry: pd.Series, min_stocks: int = 10) -> pd.DataFrame:
    frame = frame.replace([np.inf, -np.inf], np.nan)
    ind = industry.reindex(frame.index)
    groups = [frame.index.get_level_values("formation_year"), ind.to_numpy()]
    out = {}
    for col in frame.columns:
        g = frame[col].groupby(groups)
        r = g.rank(pct=True) - 0.5
        out[col] = r.where(g.transform("count") >= min_stocks)
    return pd.DataFrame(out, index=frame.index)


__all__ = ["KEY", "NAMES", "ff12", "ff49", "is_financial", "industry_june", "financial_june", "industry_rank"]
