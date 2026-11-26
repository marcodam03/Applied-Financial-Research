"""
Build data/external/factors_monthly.csv from the original public sources.

Sources:
  - Kenneth French data library: Fama-French 5 factors (2x3), momentum and short-term reversal
    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
  - Hou, Xue and Zhang q-factor data library: q5 factors (monthly)
    https://global-q.org/factors.html

Run from the project folder:  python src/data/factors.py

The script downloads the files automatically and keeps a copy of the raw files in
data/external/raw/. If a download fails (no internet, site changed the file
name), download the file by hand from the pages above, put it in
data/external/raw/ and run the script again:
  - F-F_Research_Data_5_Factors_2x3_CSV.zip
  - F-F_Momentum_Factor_CSV.zip
  - q5_factors_monthly_<year>.csv    (the monthly q5 file, any year in the name)

Output: one row per month ("YYYY-MM"), all returns in DECIMALS (0.01 = 1%):
  MKT_RF, SMB, HML, RMW, CMA, RF        Fama-French 5 factors (from 1963-07)
  MOM                                   momentum (from 1927-01)
  Q_RF, Q_MKT, Q_ME, Q_IA, Q_ROE, Q_EG  q5 factors (from 1967-01)
  STREV                                 short-term reversal (from 1926)

To add only STREV to an existing file (all other columns unchanged):
  python src/data/factors.py --add-reversal
Months where a library has no data are left empty.
"""
import io
import re
import urllib.request
import zipfile
from datetime import date
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# Project folder = first folder above this file that contains data/raw
_HERE = Path(__file__).resolve()
BASE_DIR = next((p for p in _HERE.parents if (p / "data" / "raw").is_dir()), _HERE.parents[2])
EXT_DIR = BASE_DIR / "data" / "external"
RAW_DIR = EXT_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = EXT_DIR / "factors_monthly.csv"

FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FF5_FILE = "F-F_Research_Data_5_Factors_2x3_CSV.zip"
MOM_FILE = "F-F_Momentum_Factor_CSV.zip"
STREV_FILE = "F-F_ST_Reversal_Factor_CSV.zip"  # short-term reversal (prior month return)
# The q-factor file name contains the year of the last update, so we try recent years
Q_URL = "https://global-q.org/uploads/1/2/2/6/122679606/q5_factors_monthly_{year}.csv"


# ------------------------------------------------------------------------------
# Download (or reuse a file already in data/external/raw/)
# ------------------------------------------------------------------------------
def download(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            dest.write_bytes(r.read())
        print(f"  downloaded {url}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  could not download {url} ({e.__class__.__name__})")
        return False


def get_french(filename: str) -> Path:
    path = RAW_DIR / filename
    if not download(FRENCH_URL + filename, path) and not path.exists():
        raise FileNotFoundError(
            f"Download {filename} by hand from the Kenneth French data library "
            f"and put it in {RAW_DIR}"
        )
    return path


def get_q() -> Path:
    for year in range(date.today().year, 2019, -1):
        path = RAW_DIR / f"q5_factors_monthly_{year}.csv"
        if download(Q_URL.format(year=year), path):
            return path
    local = sorted(RAW_DIR.glob("q5_factors_monthly*.csv"))
    if not local:
        raise FileNotFoundError(
            "Download the monthly q5 factors CSV by hand from https://global-q.org/factors.html "
            f"and put it in {RAW_DIR}"
        )
    print(f"  using the local file {local[-1].name}")
    return local[-1]


# ------------------------------------------------------------------------------
# Parsing
# ------------------------------------------------------------------------------
def read_french_zip(path: Path) -> pd.DataFrame:
    """French CSV files have text lines, then a header, then monthly rows
    (YYYYMM, values in %), then the annual rows. We keep the monthly block only."""
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        lines = z.read(name).decode("latin-1").splitlines()

    header, rows = None, []
    for line in lines:
        fields = [f.strip() for f in line.split(",")]
        if re.fullmatch(r"\d{6}", fields[0]):
            rows.append(fields)
        elif rows:            # first non-monthly line after the monthly block
            break
        elif len(fields) > 1 and fields[0] == "" and any(fields[1:]):
            header = fields[1:]  # the line just before the data

    df = pd.DataFrame([r[1:] for r in rows], columns=header, dtype=float)
    df.index = pd.PeriodIndex([f"{r[0][:4]}-{r[0][4:]}" for r in rows], freq="M")
    df = df.replace([-99.99, -999], np.nan) / 100  # percent -> decimals
    return df


def read_french_blocks(path: Path) -> Dict[str, pd.DataFrame]:
    """Every monthly table of a French CSV file, keyed by its title line
    (e.g. 'Value Weight Returns -- Monthly'). Values are as in the file."""
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        lines = z.read(name).decode("latin-1").splitlines()

    blocks, title, header, dates, rows = {}, None, None, [], []
    for line in lines + [""]:
        fields = [f.strip() for f in line.split(",")]
        if re.fullmatch(r"\d{6}", fields[0]):
            dates.append(f"{fields[0][:4]}-{fields[0][4:]}")
            rows.append(fields[1:len(header) + 1])
            continue
        if rows:  # first line after a monthly table
            blocks[title] = pd.DataFrame(
                rows, columns=header, index=pd.PeriodIndex(dates, freq="M"), dtype=float
            )
            dates, rows = [], []
        if len(fields) > 1 and fields[0] == "" and any(fields[1:]):
            header = fields[1:]
        elif len(fields) == 1 and fields[0]:
            title = fields[0]
    return blocks


def read_french_portfolios(path: Path, n_portfolios: int = 10) -> Dict[str, pd.DataFrame]:
    """Monthly returns (decimals) of a French portfolio file: {"vw": ..., "ew": ...}.
    Keeps the last n_portfolios columns (the deciles, low to high) as columns 1..n."""
    out = {}
    for title, df in read_french_blocks(path).items():
        t = title.lower()
        if "returns" not in t or "monthly" not in t:
            continue
        key = "vw" if "value weight" in t else "ew" if "equal weight" in t else None
        if key and key not in out:
            part = df.iloc[:, -n_portfolios:].replace([-99.99, -999], np.nan) / 100
            out[key] = part.set_axis(range(1, n_portfolios + 1), axis=1)
    if set(out) != {"ew", "vw"}:
        raise ValueError(f"Could not find the monthly EW and VW returns in {path.name}")
    return out


def read_q(path: Path) -> pd.DataFrame:
    q = pd.read_csv(path)
    q.columns = [c.strip().upper() for c in q.columns]
    q.index = pd.PeriodIndex(
        [f"{int(y)}-{int(m):02d}" for y, m in zip(q["YEAR"], q["MONTH"])], freq="M"
    )
    q = q[["R_F", "R_MKT", "R_ME", "R_IA", "R_ROE", "R_EG"]].astype(float)
    # The q files are in percent; check it instead of assuming it
    if q["R_MKT"].abs().mean() > 0.2:
        q = q / 100
    q.columns = ["Q_RF", "Q_MKT", "Q_ME", "Q_IA", "Q_ROE", "Q_EG"]
    return q


# ------------------------------------------------------------------------------
# Build
# ------------------------------------------------------------------------------
def build() -> pd.DataFrame:
    print("Fama-French 5 factors")
    ff5 = read_french_zip(get_french(FF5_FILE))
    ff5 = ff5.rename(columns={"Mkt-RF": "MKT_RF"})[["MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"]]

    print("Momentum")
    mom = read_french_zip(get_french(MOM_FILE))
    mom = mom.iloc[:, [0]].set_axis(["MOM"], axis=1)  # column is called "Mom"

    print("q5 factors")
    q = read_q(get_q())

    print("Short-term reversal")
    strev = read_reversal()

    factors = pd.concat([ff5, mom, q, strev], axis=1).sort_index()
    factors.index.name = "month"

    # Sanity checks: returns in decimals, no gaps inside each library
    assert factors["MKT_RF"].abs().max() < 0.5, "MKT_RF too large: units are probably wrong"
    for col in factors.columns:
        s = factors[col].dropna()
        expected = (s.index[-1] - s.index[0]).n + 1
        if len(s) != expected:
            print(f"  !! WARNING: {col} has {expected - len(s)} missing months inside its sample")

    print("\nSample of each factor:")
    for col in factors.columns:
        s = factors[col].dropna()
        print(f"  {col:7s} {s.index.min()} to {s.index.max()}  mean {s.mean() * 100:6.3f}% per month")
    return factors


def read_reversal() -> pd.DataFrame:
    """Ken French's short-term reversal factor (decimals), column STREV."""
    strev = read_french_zip(get_french(STREV_FILE))
    return strev.iloc[:, [0]].set_axis(["STREV"], axis=1)  # column is called "ST_Rev"


def add_reversal_only() -> None:
    """Add (or refresh) only the STREV column of the existing factor file. Every
    other column stays exactly as it is, so the team's factor vintage is unchanged."""
    if not OUTPUT.exists():
        raise FileNotFoundError(f"{OUTPUT} not found: run without --add-reversal first")
    factors = pd.read_csv(OUTPUT, index_col=0)
    factors.index = pd.PeriodIndex(factors.index, freq="M")
    print("Short-term reversal")
    strev = read_reversal()["STREV"]
    factors["STREV"] = strev.reindex(factors.index)
    s = factors["STREV"].dropna()
    print(f"  STREV   {s.index.min()} to {s.index.max()}  mean {s.mean() * 100:6.3f}% per month")
    assert s.abs().max() < 0.5, "STREV too large: units are probably wrong"
    factors.index.name = "month"
    factors.to_csv(OUTPUT)
    print(f"Saved: {OUTPUT}  {factors.shape}")


def compare_with_old(new: pd.DataFrame) -> None:
    """If a factors file already exists, show how much the new one differs."""
    if not OUTPUT.exists():
        return
    old = pd.read_csv(OUTPUT, index_col=0)
    old.index = pd.PeriodIndex(old.index, freq="M")
    common_cols = [c for c in new.columns if c in old.columns]
    common_idx = new.index.intersection(old.index)
    diff = (new.loc[common_idx, common_cols] - old.loc[common_idx, common_cols]).abs().max()
    print("\nLargest difference with the existing factors file (in %):")
    print((diff * 100).round(4).to_string())
    print("(Small differences are normal: the libraries revise past data at every update.)")


if __name__ == "__main__":
    import sys

    if "--add-reversal" in sys.argv:  # only add the STREV column, keep everything else
        add_reversal_only()
        sys.exit()
    factors = build()
    compare_with_old(factors)
    factors.to_csv(OUTPUT)
    print(f"\nSaved: {OUTPUT}  {factors.shape}")