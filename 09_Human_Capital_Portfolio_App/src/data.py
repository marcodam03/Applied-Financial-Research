"""Data layer (step 1).

Downloads prices and rates, cleans them, converts all returns to CHF,
and saves copies in data/ so the app keeps working if a website is down.

Main functions:

    from src.data import load_returns, load_rf
    returns = load_returns("2000-01-01", "2026-12-31")   # DataFrame, monthly CHF returns, the project's column names
    rf = load_rf("2000-01-01", "2026-12-31")             # Series RF_CHF, monthly CHF risk-free rate

Both read the saved files in data/ (fast, works offline).
To download everything again and print a coverage report, run from the project folder:

    python -m src.data
"""
import io
import os
import zipfile
import warnings

import numpy as np
import pandas as pd
import requests

from src.config import load_params

# Folder of the project (one level above src/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{file}_CSV.zip"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
SNB_URL = "https://data.snb.ch/api/cube/{cube}/data/csv/en"

RETURNS_FILE = "returns_chf.csv"
SECTORS_FILE = "sectors_chf.csv"
UNIVERSE2_FILE = "universe2_chf.csv"   # version 2: 12 industries, corporate bonds, commodities

RF_FILE = "rf_chf.csv"
COVERAGE_FILE = "coverage.csv"


# =============================================================================
# 1. Small helpers
# =============================================================================

def _data_folder(params):
    folder = os.path.join(ROOT, params["data"].get("folder", "data"))
    os.makedirs(os.path.join(folder, "raw"), exist_ok=True)
    return folder


def _as_series(data):
    """Saved copies are read back as one-column tables: turn them into a Series."""
    return data.iloc[:, 0] if isinstance(data, pd.DataFrame) else data


def _to_monthly_last(series):
    """Daily, monthly or irregular series -> last available value of each month,
    indexed by the month-end date."""
    series = series.dropna()
    monthly = series.groupby(series.index.to_period("M")).last()
    monthly.index = monthly.index.to_timestamp(how="end").normalize()
    monthly.index.name = "date"
    return monthly


def last_complete_month(today=None):
    """Month-end date of the last month that is already finished.
    On 18 November 2026 this is 31 October 2026: November is not over yet."""
    today = pd.Timestamp.today() if today is None else pd.Timestamp(today)
    return (today.to_period("M") - 1).to_timestamp(how="end").normalize()


def _cached(params, name, download_function):
    """Download a raw series and save it in data/raw/.
    If the download fails (website down, no internet), use the saved copy."""
    path = os.path.join(_data_folder(params), "raw", f"{name}.csv")
    try:
        data = download_function()
        data.to_csv(path)
        return data
    except Exception as error:
        if os.path.exists(path):
            warnings.warn(f"Download of {name} failed ({error}). Using the saved copy {path}.")
            return pd.read_csv(path, index_col=0, parse_dates=True)   # always a DataFrame
        raise RuntimeError(f"Download of {name} failed and no saved copy exists: {error}")


# =============================================================================
# 2. Downloaders (one per website)
# =============================================================================

def download_yahoo(tickers, start):
    """Month-end adjusted close prices (dividends included) from Yahoo Finance."""
    import yfinance as yf
    start = (pd.Timestamp(start) - pd.DateOffset(months=2)).strftime("%Y-%m-%d")  # extra month for the first return
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    close = data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close = close[[t for t in tickers if t in close.columns]]
    if close.dropna(how="all").empty:
        raise ValueError(f"Yahoo returned no data for {tickers}")
    return close.apply(_to_monthly_last)


def download_fred(series_id):
    """One series from FRED (St. Louis Fed), original frequency."""
    data = pd.read_csv(FRED_URL.format(series=series_id), na_values=".")
    # the first column is the date ("DATE" or "observation_date" depending on the FRED version)
    series = pd.Series(data[series_id].astype(float).values,
                       index=pd.DatetimeIndex(pd.to_datetime(data.iloc[:, 0])), name=series_id)
    return series.dropna()


def _parse_kf_csv(text):
    """Read the FIRST table of a Kenneth French CSV file (the monthly one).
    The files have some text, then a header line starting with ',', then rows
    starting with a date YYYYMM, then a blank line before the next table."""
    header, rows, started = None, [], False
    for line in text.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if not started:
            if line.startswith(",") and len(cells) > 1:
                header, started = cells, True
            continue
        if len(cells[0]) == 6 and cells[0].isdigit():
            rows.append(cells)
        else:
            break                                   # end of the monthly table
    if header is None or not rows:
        raise ValueError("Could not find the monthly table in the Kenneth French file")
    table = pd.DataFrame(rows, columns=["date"] + header[1:]).set_index("date")
    table = table.apply(pd.to_numeric, errors="coerce")
    table = table.replace([-99.99, -999], np.nan) / 100.0   # percent -> decimal; -99.99 = missing
    table.index = pd.to_datetime(table.index, format="%Y%m").to_period("M").to_timestamp(how="end").normalize()
    table.index.name = "date"
    return table


def download_kenfrench(file):
    """Monthly table of a Kenneth French data library file, in decimals."""
    response = requests.get(KF_URL.format(file=file), timeout=60)
    response.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    name = [n for n in archive.namelist() if n.lower().endswith(".csv")][0]
    return _parse_kf_csv(archive.read(name).decode("latin-1"))


def _parse_snb_csv(text, code):
    """SNB data portal CSV: a few metadata lines, then 'Date;D0;...;Value'.
    Keep the rows whose dimension equals `code` (e.g. 'SARON' or '10J').
    Works for monthly ('2000-01') and daily ('2000-01-03') cubes."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.replace('"', "").startswith("Date;"))
    table = pd.read_csv(io.StringIO("\n".join(lines[start:])), sep=";")
    dims = [c for c in table.columns if c not in ("Date", "Value")]
    mask = pd.Series(False, index=table.index)
    for d in dims:
        mask |= table[d].astype(str) == code
    if not mask.any():
        available = {d: sorted(table[d].astype(str).unique().tolist())[:40] for d in dims}
        raise ValueError(f"Code '{code}' not found. Available codes: {available}")
    table = table[mask]
    series = pd.Series(pd.to_numeric(table["Value"], errors="coerce").values,
                       index=pd.DatetimeIndex(pd.to_datetime(table["Date"])))
    return series.dropna().sort_index()


def download_snb(cube, code):
    """One series from the Swiss National Bank data portal (annual %)."""
    response = requests.get(SNB_URL.format(cube=cube), timeout=120)
    response.raise_for_status()
    series = _parse_snb_csv(response.text, code)
    series.name = f"{cube}_{code}"
    return series


# =============================================================================
# 3. Rates, exchange rates and CHF conversion
# =============================================================================

def build_fx(params):
    """Month-end exchange rates: CHF per 1 unit of foreign currency."""
    ids = params["data"]["fx"]
    usdchf = _to_monthly_last(_as_series(_cached(params, "fred_" + ids["USDCHF"], lambda: download_fred(ids["USDCHF"]))))
    eurusd = _to_monthly_last(_as_series(_cached(params, "fred_" + ids["EURUSD"], lambda: download_fred(ids["EURUSD"]))))
    fx = pd.DataFrame({"USD": usdchf, "EUR": usdchf * eurusd, "CHF": 1.0})
    fx.index.name = "date"
    return fx


def build_rf(params):
    """CHF risk-free rate as a monthly return (column RF_CHF).
    The rate observed at the end of month t-1 is the return earned during month t.
    Negative rates are kept as they are. Also returns the annual % level (for the hedge carry)."""
    spec = params["data"]["rates"]["RF_CHF"]
    level = _cached(params, f"snb_{spec['cube']}_{spec['code']}", lambda: download_snb(spec["cube"], spec["code"]))
    level = _to_monthly_last(_as_series(level))     # annual %, month-end
    rf = (level / 100.0 / 12.0).shift(1)
    rf.name = "RF_CHF"
    return rf, level


def to_chf(local_returns, currency, fx):
    """Unhedged conversion: (1 + r_CHF) = (1 + r_local) x (1 + change of the exchange rate)."""
    if currency == "CHF":
        return local_returns
    fx_return = fx[currency].pct_change()
    return (1 + local_returns) * (1 + fx_return) - 1


def synthetic_bond_returns(yield_level_pct, duration):
    """Monthly return of a constant-duration bond index built from a yield:
    carry (last month's yield / 12) minus duration x change in yield."""
    y = yield_level_pct / 100.0
    return y.shift(1) / 12.0 - duration * y.diff()


# =============================================================================
# 4. One source -> monthly CHF returns
# =============================================================================

def clean_ticker_returns(returns, limit):
    """Data errors of single stocks/ETFs (wrong split adjustments on Yahoo) show up as
    impossible monthly returns. They are set to NaN (never 0), and listed."""
    bad = returns.abs() > limit
    removed = [(t, d.date(), round(returns.loc[d, t], 3)) for t in returns.columns for d in returns.index[bad[t]]]
    return returns.mask(bad), removed


def _hedge_carry(params, source, index, rf_level):
    """1-month currency hedge of a foreign asset: (CHF rate - foreign rate) / 12, both rates known at
    the start of the month. Negative when CHF rates are lower. Added to the local return."""
    foreign = _to_monthly_last(_as_series(_cached(params, "fred_" + source["foreign_rf"],
                                                  lambda: download_fred(source["foreign_rf"]))))
    carry = (rf_level.reindex(index) - foreign.reindex(index)) / 100.0 / 12.0
    return carry.shift(1)


def source_returns(params, source, fx, rf_level):
    """Monthly returns in CHF of one source described in params.yaml.
    Returns (series, list of removed data errors)."""
    kind = source["type"]

    if kind == "yahoo":
        tickers = source["tickers"]
        prices = _cached(params, "yahoo_" + "_".join(tickers),
                         lambda: download_yahoo(tickers, params["data"]["start_date"]))
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
        ticker_returns = prices.pct_change(fill_method=None)       # NaN when a price is missing
        ticker_returns, removed = clean_ticker_returns(ticker_returns, params["data"]["max_abs_monthly_return_ticker"])
        local = ticker_returns.mean(axis=1, skipna=True)            # equal weight of the tickers available that month
        if source.get("hedged", False):
            return local + _hedge_carry(params, source, local.index, rf_level), removed
        return to_chf(local, source["currency"], fx), removed

    if kind == "kf_market":
        table = _cached(params, "kf_" + source["file"], lambda: download_kenfrench(source["file"]))
        local = table["Mkt-RF"] + table["RF"]          # market return = excess return + US T-bill
        return to_chf(local, source["currency"], fx), []

    if kind == "kf_industry":
        table = _cached(params, "kf_" + source["file"], lambda: download_kenfrench(source["file"]))
        return to_chf(table[source["column"]], source["currency"], fx), []

    if kind == "snb_yield":
        level = _cached(params, f"snb_{source['cube']}_{source['code']}",
                        lambda: download_snb(source["cube"], source["code"]))
        return synthetic_bond_returns(_to_monthly_last(_as_series(level)), source["duration"]), []

    if kind == "fred_yield":
        level = _to_monthly_last(_as_series(_cached(params, "fred_" + source["series"],
                                                    lambda: download_fred(source["series"]))))
        local = synthetic_bond_returns(level, source["duration"])
        if source.get("hedged", False):
            return local + _hedge_carry(params, source, local.index, rf_level), []
        return to_chf(local, source["currency"], fx), []

    raise ValueError(f"Unknown source type: {kind}")


def describe_source(source):
    if source["type"] == "yahoo":
        return "Yahoo " + "+".join(source["tickers"])
    if source["type"] in ("kf_market", "kf_industry"):
        return "Ken French " + source["file"] + (f" [{source['column']}]" if "column" in source else "")
    if source["type"] == "snb_yield":
        return f"SNB {source['cube']} {source['code']} (synthetic, D={source['duration']})"
    if source["type"] == "fred_yield":
        return f"FRED {source['series']} (synthetic, D={source['duration']}" + (", hedged)" if source.get("hedged") else ")")
    return source["type"]


def splice(series_list):
    """Combine several sources of the same asset.
    The first series has priority wherever it has data; each next series only fills
    the months still missing (before the first one starts, after it ends, or a month
    removed by the cleaning). Returns the combined series and the source used each month."""
    index = series_list[0].index
    for s in series_list[1:]:
        index = index.union(s.index)
    combined = pd.Series(np.nan, index=index)
    used = pd.Series(np.nan, index=index)
    for i, s in enumerate(series_list):
        s = s.reindex(index)
        fill = combined.isna() & s.notna()
        combined[fill] = s[fill]
        used[fill] = i
    combined.index.name = "date"
    return combined, used


# =============================================================================
# 5. Build, save, load
# =============================================================================

def build_data(params=None, verbose=True):
    """Download everything and build the monthly CHF returns.
    Returns (returns, rf, coverage). `returns` has the core assets AND the sector menu.
    Only months where every core asset and the risk-free rate have data are kept."""
    if params is None:
        params = load_params()
    fx = build_fx(params)
    rf, rf_level = build_rf(params)

    specs = {**params["data"]["assets"], **params["data"].get("sectors", {}),      # core universe + sector menu
             **params["data"].get("universe2", {})}                               # + version 2 universe
    columns, used_by_asset, removed_all = {}, {}, []
    for asset, spec in specs.items():
        pieces = []
        for source in spec["sources"]:
            try:
                series, removed = source_returns(params, source, fx, rf_level)
                removed_all += [(asset,) + r for r in removed]
            except Exception as error:
                warnings.warn(f"{asset}: source {describe_source(source)} failed: {error}")
                series = pd.Series(dtype=float)
            pieces.append(series)
        if all(p.dropna().empty for p in pieces):
            warnings.warn(f"{asset}: no data at all, asset skipped")
            continue
        series, used = splice(pieces)
        columns[asset] = series
        used_by_asset[asset] = used

    returns = pd.DataFrame(columns)
    returns.index.name = "date"

    # --- Keep only finished months where every CORE asset and the risk-free rate have data
    # (the sector menu never shortens the sample of the core universe)
    start = pd.Timestamp(params["data"]["start_date"]) + pd.offsets.MonthEnd(0)
    core = [a for a in params["data"]["assets"] if a in returns.columns]
    complete = returns[core].join(rf, how="inner").dropna()
    first = max(start, complete.index.min())
    last = min(complete.index.max(), last_complete_month())   # the current month is never included
    returns, rf = returns.loc[first:last], rf.loc[first:last]
    holes = returns.isna().sum()
    if holes.any():
        warnings.warn(f"Missing months inside the sample (left as NaN): {holes[holes > 0].to_dict()}")

    # --- Which source was used for each asset, inside the final sample
    coverage = []
    for asset, used in used_by_asset.items():
        used = used.loc[first:last]
        for i, source in enumerate(specs[asset]["sources"]):
            months = used[used == i].index
            if len(months):
                coverage.append({"asset": asset, "source": describe_source(source),
                                 "from": months.min().date(), "to": months.max().date(), "months": len(months)})
    coverage = pd.DataFrame(coverage)
    if verbose:
        print_report(params, returns, rf, coverage, removed_all)
    return returns, rf, coverage


def print_report(params, returns, rf, coverage, removed):
    print("\nSources used for each asset:")
    print(coverage.to_string(index=False))
    print(f"\nFinal sample: {returns.index.min().date()} to {returns.index.max().date()}, "
          f"{len(returns)} months, {returns.shape[1]} assets + RF_CHF")
    last_dates = {a: c["to"] for a, c in coverage.groupby("asset").max().iterrows()}
    print("Last month available by asset:", {a: str(d) for a, d in last_dates.items()})
    if removed:
        print("\nData errors removed (single ticker, |monthly return| above "
              f"{params['data']['max_abs_monthly_return_ticker']:.0%}):")
        for asset, ticker, date, value in removed:
            print(f"  {asset:10s} {ticker:8s} {date}  {value:+.1%}")
    stats = pd.DataFrame({"mean (ann.)": returns.mean() * 12, "vol (ann.)": returns.std() * np.sqrt(12),
                          "min month": returns.min(), "max month": returns.max()})
    print("\nSummary statistics (CHF, annualised):")
    print(stats.round(4).to_string())
    warn = params["data"]["warn_abs_monthly_return"]
    big = returns.abs() > warn
    if big.any().any():
        print(f"\nMonths above {warn:.0%} to check by hand (may be real, e.g. banks in 2009):")
        for asset in big.columns[big.any()]:
            print(f"  {asset}: " + ", ".join(f"{d.date()} {returns.loc[d, asset]:+.1%}" for d in returns.index[big[asset]]))
    print(f"\nRF_CHF: mean {rf.mean() * 12:.2%} per year, {int((rf < 0).sum())} months negative")


def update_data(params=None, verbose=True):
    """Download everything again and overwrite the saved files in data/:
    core returns, sector menu, risk-free rate and coverage."""
    if params is None:
        params = load_params()
    returns, rf, coverage = build_data(params, verbose)
    core = [a for a in params["data"]["assets"] if a in returns.columns]
    sectors = [a for a in params["data"].get("sectors", {}) if a in returns.columns]
    universe2 = [a for a in params["data"].get("universe2", {}) if a in returns.columns]
    folder = _data_folder(params)
    returns[core].to_csv(os.path.join(folder, RETURNS_FILE))
    returns[sectors].to_csv(os.path.join(folder, SECTORS_FILE))
    if universe2:
        returns[universe2].to_csv(os.path.join(folder, UNIVERSE2_FILE))
    rf.to_frame().to_csv(os.path.join(folder, RF_FILE))
    coverage.to_csv(os.path.join(folder, COVERAGE_FILE), index=False)
    if verbose:
        print(f"\nSaved {RETURNS_FILE}, {SECTORS_FILE}, {UNIVERSE2_FILE}, {RF_FILE} and {COVERAGE_FILE} in {folder}")
    return returns[core], rf


def _read_saved(params, file):
    path = os.path.join(_data_folder(params), file)
    if not os.path.exists(path):
        update_data(params, verbose=False)
    data = pd.read_csv(path, index_col="date", parse_dates=["date"])
    return data


def load_returns(start=None, end=None, params=None):
    """Monthly CHF simple returns in decimals, one column per asset
    (EQ_WORLD, BANKS_EU, EQ_CH, BOND_CHF, BOND_GLB_H, REIT_CH, GOLD, EMPLOYER),
    month-end index named 'date'. start/end as 'YYYY-MM-DD' (None = full sample).
    These are total returns: excess returns = load_returns() minus load_rf()."""
    if params is None:
        params = load_params()
    return _read_saved(params, RETURNS_FILE).loc[start:end]


def load_sector_returns(start=None, end=None, params=None):
    """Monthly CHF returns of the sector menu (SEC_FIN, SEC_TECH, ...), same index as
    load_returns. Used when the client's career is tied to a sector other than BANKS_EU."""
    if params is None:
        params = load_params()
    return _read_saved(params, SECTORS_FILE).loc[start:end]


def load_universe2_returns(start=None, end=None, params=None):
    """Version 2 universe: monthly CHF returns of the 12 Fama-French industries (IND_...), corporate
    bonds (BOND_CORP) and commodities (COMMOD), same index as load_returns."""
    if params is None:
        params = load_params()
    return _read_saved(params, UNIVERSE2_FILE).loc[start:end]


def load_rf(start=None, end=None, params=None):
    """Monthly CHF risk-free rate (Series named RF_CHF), same index as load_returns."""
    if params is None:
        params = load_params()
    return _read_saved(params, RF_FILE)["RF_CHF"].loc[start:end]


if __name__ == "__main__":
    update_data()
