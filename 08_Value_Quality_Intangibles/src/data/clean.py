"""
Strategy-agnostic cleaning of CRSP monthly, Compustat annual and CRSP daily quotes.

Run ONCE (and again only if you fix a data problem, never for a new strategy):
    python <folder>/clean.py

Outputs (Parquet, needs `pip install pyarrow`):
    data/interim/crsp_clean.parquet       one row per stock-month (all eligible stocks)
    data/interim/compustat_clean.parquet  one row per stock and formation year,
                                          all accounting items + previous-year values
    data/interim/spreads_clean.parquet    one row per stock-month: half bid-ask spread
                                          at the month's last close (estimated before
                                          1993 for NYSE/AMEX), size group and daily
                                          dollar volume (trading costs)

No strategy signal is computed here. Signals live in src/signals/ and the two
tables are merged on demand by src/data/panel.py (load_panel).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:  # cost_inputs.py sits next to this file
    from cost_inputs import add_cost_inputs
except ImportError:
    from src.data.cost_inputs import add_cost_inputs

# ------------------------------------------------------------------------------
# Paths (same as before: the folder must be called "data" in lowercase)
# ------------------------------------------------------------------------------
# Project folder = first folder above this file that contains data/raw
_HERE = Path(__file__).resolve() if "__file__" in globals() else Path.cwd() / "x"
BASE_DIR = next((p for p in _HERE.parents if (p / "data" / "raw").is_dir()), _HERE.parent.parent)
RAW_DIR = BASE_DIR / "data" / "raw"
INTERIM_DIR = BASE_DIR / "data" / "interim"
INTERIM_DIR.mkdir(parents=True, exist_ok=True)

US_EXCHANGES = ["N", "A", "Q"]  # NYSE, AMEX, NASDAQ
FUND_REIT_SIC = [6720, 6722, 6723, 6724, 6725, 6726, 6798]

# ------------------------------------------------------------------------------
# Delisting returns (Shumway 1997).
# Leave SHUMWAY_FILL = False, run the script once and read the "delisting check"
# printout. If many delisting months have a missing return, set it to True.
# SHUMWAY_ACTION_TYPES: None = fill every missing delisting return; otherwise a
# list of DelActionType codes (the performance-related ones) to restrict it.
# Checked on our WRDS extract (CIZ format, to 2025-12): only 13 of 22,858
# delisting months (0.06%) have a missing return, because CIZ MthRet already
# includes the delisting return. So the fill stays off.
# ------------------------------------------------------------------------------
SHUMWAY_FILL = False
SHUMWAY_ACTION_TYPES = None
SHUMWAY_RET_NYSE_AMEX = -0.30
SHUMWAY_RET_NASDAQ = -0.55

# ------------------------------------------------------------------------------
# Compustat backfill bias (Banz and Breen 1986). When Compustat adds a firm, it
# also adds some earlier years of data, and only firms that survived get added.
# Fama and French (1993) use a fiscal year only if the firm has been on Compustat
# for two years: MIN_COMPUSTAT_YEARS = 2 does that (the first year is dropped, but
# still serves as the previous-year value of the second). 1 = no filter.
# Leave it at 1. Our CCM extract has no year before the CRSP link starts, so the
# "first year on Compustat" is almost always the first year after listing: on our
# data (checked 2026-10-04), 75% of the years the filter drops are new listings and
# only 4.5% look like backfill. It also moves the B/M deciles away from Ken French's
# (EW 10-1 spread 0.88 instead of 0.98, French 0.98), so his portfolios very
# likely do not apply it.
# Setting it to 2 is a robustness check, not a fix.
# ------------------------------------------------------------------------------
MIN_COMPUSTAT_YEARS = 1

# ------------------------------------------------------------------------------
# Bid-ask spreads (trading costs), from the closing quotes of CRSP daily
# (data/raw/crsp_daily.parquet: `python convert_to_parquet.py crsp_daily.csv`).
# CRSP has closing quotes for NASDAQ since 1973 but for NYSE/AMEX only since the
# end of December 1992; before, only on days a stock did not trade. Checked on our
# extract (2026-10-07): NYSE stocks with a quote on the last day of the month, 1% in
# 1992-11, 98% in 1992-12. So NYSE/AMEX quotes are used from QUOTES_FROM on, NASDAQ
# quotes always. Where there is no usable quote before 1993, the spread is estimated
# (src/data/cost_inputs.py): the net results cover the whole sample.
# A stock without a quote on the last day of the month gets its last earlier quote,
# at most MAX_QUOTE_AGE_DAYS old. That fills CRSP's gap in NYSE quotes from 2000-10
# to 2001-04 (only 10-60% of NYSE stocks have a month-end quote in those months).
# Quotes with (ask - bid) / midpoint above MAX_RELATIVE_SPREAD are data errors and
# are dropped (0.14% of month-end quotes 1993-2015, mostly penny stocks).
# ------------------------------------------------------------------------------
QUOTES_FROM = "1992-12-01"
MAX_QUOTE_AGE_DAYS = 365
MAX_RELATIVE_SPREAD = 0.5

# ------------------------------------------------------------------------------
# Compustat columns. Items not in the file are reported and created as missing.
# ------------------------------------------------------------------------------
ID_COLS = [
    "GVKEY", "LINKPRIM", "LIID", "LINKTYPE", "LPERMNO", "LPERMCO", "LINKDT",
    "LINKENDDT", "datadate", "fyear", "indfmt", "consol", "popsrc", "datafmt",
    "tic", "cusip", "conm", "sich",
]
ACCOUNTING_ITEMS = [
    # Balance sheet
    "at", "act", "che", "rect", "invt", "aco", "ppent", "ppegt", "intan", "gdwl",
    "ivao", "ao", "lct", "ap", "dlc", "txp", "lco", "dltt", "txditc", "txdb",
    "lo", "lt", "mib", "pstk", "pstkrv", "pstkl", "ceq", "seq", "re", "csho",
    # Income statement
    "sale", "revt", "cogs", "xsga", "xrd", "xad", "dp", "oibdp", "oiadp", "xint",
    "pi", "txt", "ib", "ibc", "ni", "spi", "xido", "epspx", "dvc", "dvt", "dvp", "dv",
    # Cash flow statement
    "oancf", "ivncf", "fincf", "capx", "aqc", "sppe", "sstk", "prstkc", "dltis",
    "dltr", "dpc",
    # Other
    "prcc_f", "emp",
]
# Every accounting item (plus book equity) also gets its previous fiscal year
# value, stored as "<item>_lag" (e.g. at_lag, sale_lag, be_lag).
LAG_ITEMS = ACCOUNTING_ITEMS + ["be"]


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def column_is_in(df: pd.DataFrame, col: str, allowed: list, label: str) -> pd.Series:
    """True/False per row: is df[col] one of the allowed values?
    If the column does not exist, print a warning and return all True."""
    if col not in df.columns:
        print(f"  !! WARNING: column '{col}' not found, filter '{label}' NOT applied")
        return pd.Series(True, index=df.index)
    return df[col].isin(allowed)


def filter_equal(df: pd.DataFrame, col: str, allowed, label: str) -> pd.DataFrame:
    """Keep only the rows where df[col] is one of the allowed values."""
    if not isinstance(allowed, (list, tuple, set)):
        allowed = [allowed]
    return df[column_is_in(df, col, list(allowed), label)]


def text_columns_to_string(df: pd.DataFrame) -> pd.DataFrame:
    """Parquet needs one type per column: store text columns as strings."""
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype("string")
    return df


# ==============================================================================
# 1. CRSP
# ==============================================================================
def clean_crsp(
    input_file: Path = RAW_DIR / "crsp_monthly.csv",
    output_file: Path = INTERIM_DIR / "crsp_clean.parquet",
) -> pd.DataFrame:
    print(f"\n[1/4] Loading CRSP Monthly Data from: {input_file}")
    if not input_file.exists():
        raise FileNotFoundError(f"Missing file: {input_file}")

    crsp = pd.read_csv(input_file, low_memory=False)
    print(f"  -> Raw records: {len(crsp):,}")

    # A. Date parsing and key integrity
    crsp["MthCalDt"] = pd.to_datetime(crsp["MthCalDt"], errors="coerce")
    crsp = crsp.dropna(subset=["MthCalDt", "PERMNO"])
    crsp["PERMNO"] = crsp["PERMNO"].astype(int)

    # B. Duplicate PERMNO-months: keep the record with the dividend
    if "DisDivAmt" in crsp.columns:
        crsp = crsp.sort_values(by=["PERMNO", "MthCalDt", "DisDivAmt"], na_position="first")
    else:
        crsp = crsp.sort_values(by=["PERMNO", "MthCalDt"])
    crsp = crsp.drop_duplicates(subset=["PERMNO", "MthCalDt"], keep="last")
    print(f"  -> After deduplication on (PERMNO, MthCalDt): {len(crsp):,} rows")

    # C. Common US equity on NYSE/AMEX/NASDAQ (no funds, no REITs)
    if "SICCD" in crsp.columns:
        crsp["SICCD"] = pd.to_numeric(crsp["SICCD"], errors="coerce")
    eligible = pd.Series(True, index=crsp.index)
    eligible &= column_is_in(crsp, "SecurityType", ["EQTY"], "equity only")
    eligible &= column_is_in(crsp, "SecuritySubType", ["COM"], "common stock only")
    eligible &= column_is_in(crsp, "ShareType", ["NS"], "no special share type")
    eligible &= column_is_in(crsp, "USIncFlg", ["Y"], "US incorporated")
    eligible &= column_is_in(crsp, "IssuerType", ["ACOR", "CORP"], "corporate issuer")
    eligible &= column_is_in(crsp, "PrimaryExch", US_EXCHANGES, "NYSE/AMEX/NASDAQ")
    if "SICCD" in crsp.columns:
        eligible &= ~crsp["SICCD"].isin(FUND_REIT_SIC)
    else:
        print("  !! WARNING: column 'SICCD' not found, fund/REIT filter NOT applied")

    # D. Keep the delisting month (eligible this month OR last month)
    eligible_last_month = eligible.groupby(crsp["PERMNO"]).shift(1, fill_value=False)
    crsp = crsp[eligible | eligible_last_month].copy()
    print(f"  -> After common equity and exchange filters (delisting months kept): {len(crsp):,} rows")

    # E. Prices, returns, shares, market cap
    for col in ["MthPrc", "MthRet", "MthRetx", "ShrOut", "MthVol", "DisDivAmt"]:
        if col in crsp.columns:
            crsp[col] = pd.to_numeric(crsp[col], errors="coerce")
    crsp["MthPrc"] = crsp["MthPrc"].abs()  # negative price = bid/ask average
    crsp["mkt_cap"] = crsp["MthPrc"] * crsp["ShrOut"]

    # F. NEW: delisting returns (diagnostic + optional Shumway fill)
    crsp = delisting_check(crsp)

    # G. NEW: firm-level market cap (all share classes of the same company)
    if "PERMCO" in crsp.columns:
        grp = crsp.groupby(["PERMCO", "MthCalDt"])["mkt_cap"]
        crsp["me_firm"] = grp.transform("sum")
        crsp.loc[grp.transform("count") == 0, "me_firm"] = np.nan  # all classes missing
    else:
        print("  !! WARNING: column 'PERMCO' not found, me_firm NOT computed")

    # H. Fama-French annual alignment (July t to June t+1 -> formation_year t)
    crsp["formation_year"] = np.where(
        crsp["MthCalDt"].dt.month >= 7,
        crsp["MthCalDt"].dt.year,
        crsp["MthCalDt"].dt.year - 1,
    )

    # I. NEW: market cap unit check on Apple (PERMNO 14593)
    apple = crsp.loc[crsp["PERMNO"] == 14593, ["MthCalDt", "mkt_cap"]].tail(1)
    if len(apple):
        d, m = apple.iloc[0]
        print(f"  -> UNIT CHECK: Apple mkt_cap on {d:%Y-%m} = {m:,.0f}")
        print("     Compare with Apple's real market cap: if mkt_cap is in thousands of")
        print("     dollars, the number above is the dollar value divided by 1,000.")

    keep_cols = [
        "PERMNO", "PERMCO", "Ticker", "PrimaryExch", "SICCD", "MthCalDt",
        "formation_year", "MthPrc", "MthRet", "MthRetx", "MthVol", "ShrOut",
        "mkt_cap", "me_firm", "DisDivAmt", "MthDelFlg", "DelActionType",
    ]
    crsp_clean = crsp[[c for c in keep_cols if c in crsp.columns]].reset_index(drop=True)
    crsp_clean = text_columns_to_string(crsp_clean)
    crsp_clean.to_parquet(output_file, index=False)
    print(f"  -> Saved clean CRSP to: {output_file} {crsp_clean.shape}")
    return crsp_clean


def delisting_check(crsp: pd.DataFrame) -> pd.DataFrame:
    """Count delisting months with a missing return and, if SHUMWAY_FILL is
    True, replace those missing returns with the Shumway (1997) values.

    Delisting month = the last month a stock appears in our cleaned data, if it
    is before the end of the sample (the stock stops trading or leaves
    NYSE/AMEX/NASDAQ). This does not rely on the format of the MthDelFlg column.
    crsp must be sorted by PERMNO and date (it is, after the deduplication)."""
    is_last = ~crsp.duplicated(subset="PERMNO", keep="last")
    delist = is_last & (crsp["MthCalDt"] < crsp["MthCalDt"].max())
    missing = delist & crsp["MthRet"].isna()
    print(f"  -> Delisting check: {delist.sum():,} delisting months, "
          f"{missing.sum():,} with a missing return ({missing.sum() / max(delist.sum(), 1):.1%})")
    if "MthDelFlg" in crsp.columns:
        print("     MthDelFlg values in those months (for information):")
        print(crsp.loc[delist, "MthDelFlg"].value_counts(dropna=False).head(5).to_string())
    if "DelActionType" in crsp.columns and missing.any():
        print("     DelActionType of the delisting months with a missing return:")
        print(crsp.loc[missing, "DelActionType"].value_counts(dropna=False).to_string())

    if SHUMWAY_FILL:
        to_fill = missing
        if SHUMWAY_ACTION_TYPES is not None and "DelActionType" in crsp.columns:
            to_fill &= crsp["DelActionType"].isin(SHUMWAY_ACTION_TYPES)
        # In the delisting month the exchange code changes, so use last month's
        prev_exch = crsp.groupby("PERMNO")["PrimaryExch"].shift(1)
        fill_value = np.where(prev_exch == "Q", SHUMWAY_RET_NASDAQ, SHUMWAY_RET_NYSE_AMEX)
        crsp.loc[to_fill, "MthRet"] = fill_value[to_fill.to_numpy()]
        print(f"     Shumway fill applied to {to_fill.sum():,} months")
    return crsp


# ==============================================================================
# 2. COMPUSTAT
# ==============================================================================
def clean_compustat(
    input_file: Path = RAW_DIR / "ccm_funda.csv",
    output_file: Path = INTERIM_DIR / "compustat_clean.parquet",
) -> pd.DataFrame:
    print(f"\n[2/4] Loading Compustat Fundamentals Annual from: {input_file}")
    if not input_file.exists():
        raise FileNotFoundError(f"Missing file: {input_file}")

    header_cols = pd.read_csv(input_file, nrows=0).columns.tolist()
    wanted = ID_COLS + ACCOUNTING_ITEMS
    missing = [c for c in wanted if c not in header_cols]
    if missing:
        print(f"  !! WARNING: not in the file (created as missing): {missing}")
    comp = pd.read_csv(input_file, usecols=[c for c in wanted if c in header_cols], low_memory=False)
    print(f"  -> Loaded {comp.shape[1]} columns across {len(comp):,} raw rows")

    # A. Link filters and standard Compustat formats
    comp = filter_equal(comp, "LINKPRIM", ["P", "C"], "primary links")
    comp = filter_equal(comp, "LINKTYPE", ["LU", "LC"], "reliable link types")
    comp = filter_equal(comp, "indfmt", "INDL", "industrial format")
    comp = filter_equal(comp, "datafmt", "STD", "standardized data")
    comp = filter_equal(comp, "popsrc", "D", "domestic")
    comp = filter_equal(comp, "consol", "C", "consolidated")

    # B. Accounting items as numbers (missing stays missing, no zero-filling).
    # First fiscal year of each firm on Compustat, before any other filter (for the
    # backfill filter in F2)
    comp = comp.copy()
    comp["fyear"] = pd.to_numeric(comp["fyear"], errors="coerce")
    comp["first_fyear"] = comp.groupby("GVKEY")["fyear"].transform("min")
    for var in ACCOUNTING_ITEMS + ["sich"]:
        comp[var] = pd.to_numeric(comp[var], errors="coerce") if var in comp.columns else np.nan

    comp = comp[comp["at"] > 0]
    print(f"  -> After link and positive-asset filters: {len(comp):,} rows")

    # C. Link valid on the balance sheet date
    comp["datadate"] = pd.to_datetime(comp["datadate"], errors="coerce")
    comp["LINKDT"] = pd.to_datetime(comp["LINKDT"], errors="coerce")
    comp["LINKENDDT"] = pd.to_datetime(
        comp["LINKENDDT"].replace("E", "2099-12-31"), errors="coerce"
    ).fillna(pd.Timestamp("2099-12-31"))
    comp = comp[(comp["datadate"] >= comp["LINKDT"]) & (comp["datadate"] <= comp["LINKENDDT"])]

    comp["LPERMNO"] = pd.to_numeric(comp["LPERMNO"], errors="coerce")
    comp = comp.dropna(subset=["LPERMNO", "datadate"])
    comp["LPERMNO"] = comp["LPERMNO"].astype(int)

    # D. One record per (GVKEY, datadate), primary link preferred
    comp = comp.sort_values(by=["GVKEY", "datadate", "LINKPRIM"])
    comp = comp.drop_duplicates(subset=["GVKEY", "datadate"], keep="last")

    # E. Fama-French book equity (universal convention, so it stays here)
    pref_stock = (
        comp["pstkrv"].combine_first(comp["pstkl"]).combine_first(comp["pstk"]).fillna(0.0)
    )
    stockholders_equity = (
        comp["seq"]
        .combine_first(comp["ceq"] + comp["pstk"])
        .combine_first(comp["at"] - comp["lt"])
    )
    comp["be"] = stockholders_equity + comp["txditc"].fillna(0.0) - pref_stock
    comp.loc[comp["be"] <= 0, "be"] = np.nan

    # F. NEW: previous-year value of EVERY item (replaces the at_lag-only code).
    # Only if the two fiscal years are consecutive (no gap in the data).
    comp = comp.sort_values(by=["GVKEY", "fyear", "datadate"])
    comp = comp.drop_duplicates(subset=["GVKEY", "fyear"], keep="last")
    lagged = comp.groupby("GVKEY")[LAG_ITEMS].shift(1)
    prev_fyear = comp.groupby("GVKEY")["fyear"].shift(1)
    lagged[(comp["fyear"] - prev_fyear) != 1] = np.nan
    lagged.columns = [f"{c}_lag" for c in LAG_ITEMS]
    comp = pd.concat([comp, lagged], axis=1).copy()

    # F2. Backfill bias: drop the firm's first year(s) on Compustat (after the lags,
    # so the second year still has its previous-year values)
    if MIN_COMPUSTAT_YEARS > 1:
        too_new = comp["fyear"] - comp["first_fyear"] < MIN_COMPUSTAT_YEARS - 1
        comp = comp[~too_new]
        print(f"  -> After dropping years with less than {MIN_COMPUSTAT_YEARS} years on Compustat: "
              f"{len(comp):,} rows ({too_new.sum():,} dropped)")

    # G. Formation year: fiscal year ending in calendar year t-1 is used from July t
    comp["formation_year"] = comp["datadate"].dt.year + 1

    # H. One record per stock and formation year (most recent report)
    comp = comp.sort_values(by=["LPERMNO", "formation_year", "datadate"])
    comp = comp.drop_duplicates(subset=["LPERMNO", "formation_year"], keep="last")

    out_cols = (
        ["GVKEY", "LPERMNO", "conm", "tic", "cusip", "sich", "datadate", "fyear", "formation_year"]
        + ACCOUNTING_ITEMS + ["be"] + list(lagged.columns)
    )
    comp_clean = comp[[c for c in out_cols if c in comp.columns]].reset_index(drop=True)
    comp_clean = text_columns_to_string(comp_clean)
    comp_clean.to_parquet(output_file, index=False)
    print(f"  -> Saved clean Compustat to: {output_file} {comp_clean.shape}")
    return comp_clean


# ==============================================================================
# 3. MERGE CHECK (the merge itself is done on demand by src/data/panel.py)
# ==============================================================================
def check_merge(crsp: pd.DataFrame, comp: pd.DataFrame) -> None:
    print("\n[3/4] Checking the CRSP-Compustat merge")
    panel = crsp[["PERMNO", "MthCalDt", "formation_year"]].merge(
        comp[["LPERMNO", "formation_year", "datadate"]],
        left_on=["PERMNO", "formation_year"],
        right_on=["LPERMNO", "formation_year"],
        how="left",
    )
    lag = (
        (panel["MthCalDt"].dt.year - panel["datadate"].dt.year) * 12
        + (panel["MthCalDt"].dt.month - panel["datadate"].dt.month)
    )
    print(f"  -> Rows using accounting data less than 6 months old (should be 0): {(lag < 6).sum()}")
    print(f"  -> Duplicate (PERMNO, MthCalDt) rows (should be 0): "
          f"{panel.duplicated(subset=['PERMNO', 'MthCalDt']).sum()}")
    print(f"  -> Share of stock-months with accounting data: {panel['datadate'].notna().mean():.1%}")


# ==============================================================================
# 4. BID-ASK SPREADS (CRSP daily)
# ==============================================================================
def clean_spreads(
    crsp: pd.DataFrame,
    input_file: Path = RAW_DIR / "crsp_daily.parquet",
    output_file: Path = INTERIM_DIR / "spreads_clean.parquet",
) -> pd.DataFrame:
    """Half bid-ask spread of every stock at the last close of every month, the
    cost of a trade per dollar: (ask - bid) / (ask + bid). One row per stock-month
    of crsp (the clean CRSP monthly table):
        month_end      last trading day of the month (whole market)
        quote_day      day of the quote used: month_end, or the stock's last earlier
                       quote (at most MAX_QUOTE_AGE_DAYS old)
        half_spread    the quote; 0 in the month a stock stops trading for good
                       (delisting): that exit is not a trade of ours; before 1993
                       without a usable quote, the estimate of cost_inputs.py;
                       missing if none of these (costs.py fills it)
        spread_source  quote / estimate / delisted / none
        size_group     micro / small / large (NYSE size percentiles 20 and 50)
        adv            average daily dollar volume (price impact)"""
    print(f"\n[4/4] Loading CRSP daily quotes from: {input_file}")
    if not input_file.exists():
        raise FileNotFoundError(f"Missing file: {input_file} "
                                "(put crsp_daily.csv in data/raw and run: "
                                "python convert_to_parquet.py crsp_daily.csv)")

    # Too big for memory (100M+ rows): read one row group at a time and keep, for
    # each stock-month, its last day in the file and its last valid quote
    source = pq.ParquetFile(input_file)
    last_days, quotes = [], []
    for i in range(source.num_row_groups):
        d = source.read_row_group(i, columns=["PERMNO", "DlyCalDt", "DlyBid", "DlyAsk"]).to_pandas()
        d["month"] = d["DlyCalDt"].dt.to_period("M")
        last_days.append(d.groupby(["PERMNO", "month"])["DlyCalDt"].max())
        d["half_spread"] = (d["DlyAsk"] - d["DlyBid"]) / (d["DlyAsk"] + d["DlyBid"])
        valid = (d["DlyBid"] > 0) & d["half_spread"].between(0, MAX_RELATIVE_SPREAD / 2)
        quotes.append(d[valid].sort_values(["PERMNO", "DlyCalDt"])
                      .drop_duplicates(["PERMNO", "month"], keep="last"))
    print(f"  -> Daily rows read: {source.metadata.num_rows:,}")

    # A stock-month can be split over two row groups
    spreads = pd.concat(last_days).groupby(level=["PERMNO", "month"]).max().rename("last_day").reset_index()
    quotes = (pd.concat(quotes).sort_values(["PERMNO", "DlyCalDt"])
              .drop_duplicates(["PERMNO", "month"], keep="last")
              .rename(columns={"DlyCalDt": "quote_day"})[["PERMNO", "month", "quote_day", "half_spread"]])
    # NYSE/AMEX quotes before QUOTES_FROM exist only on days without a trade: dropped
    exch = crsp[["PERMNO", "MthCalDt", "PrimaryExch"]].assign(month=crsp["MthCalDt"].dt.to_period("M"))
    quotes = quotes.merge(exch[["PERMNO", "month", "PrimaryExch"]], on=["PERMNO", "month"], how="left")
    early = quotes["month"] < pd.Period(QUOTES_FROM, freq="M")
    quotes = quotes[~early | (quotes["PrimaryExch"] == "Q")].drop(columns="PrimaryExch")
    spreads = spreads.merge(quotes, on=["PERMNO", "month"], how="left").sort_values(["PERMNO", "month"])
    spreads["month_end"] = spreads.groupby("month")["last_day"].transform("max")
    recent = spreads["month"] >= pd.Period(QUOTES_FROM, freq="M")
    on_last_day = (spreads.loc[recent, "quote_day"] == spreads.loc[recent, "month_end"]).mean()

    # No quote on the last day: the stock's last earlier quote, if recent enough
    spreads[["quote_day", "half_spread"]] = spreads.groupby("PERMNO")[["quote_day", "half_spread"]].ffill()
    too_old = (spreads["month_end"] - spreads["quote_day"]).dt.days > MAX_QUOTE_AGE_DAYS
    spreads.loc[too_old, "quote_day"] = pd.NaT
    spreads.loc[too_old, "half_spread"] = np.nan

    # Stops trading during its last month in the file (delisting): exit costs nothing
    stopped = ((spreads["month"] == spreads.groupby("PERMNO")["month"].transform("max"))
               & (spreads["last_day"] < spreads["month_end"]))
    spreads.loc[stopped, "quote_day"] = pd.NaT
    spreads.loc[stopped, "half_spread"] = 0.0

    print(f"  -> Daily file, from {QUOTES_FROM[:7]}: {recent.sum():,} stock-months. Quote on the "
          f"month's last day: {on_last_day:.1%}, earlier quote: "
          f"{spreads.loc[recent, 'quote_day'].notna().mean() - on_last_day:.1%}, "
          f"no quote: {spreads.loc[recent, 'half_spread'].isna().mean():.1%}")
    spreads["stopped"] = stopped

    # Every stock-month of the clean CRSP table, with the estimate before 1993, the
    # size group and the dollar volume (src/data/cost_inputs.py)
    spreads = add_cost_inputs(spreads[["PERMNO", "month", "month_end", "quote_day",
                                       "half_spread", "stopped"]], crsp)
    by_year = spreads.groupby(spreads["month_end"].dt.year)["half_spread"].median() * 10_000
    print("     Median half spread (bps, our stocks) in "
          + ", ".join(f"{y}: {by_year[y]:.0f}" for y in by_year.index[::5]))

    spreads_clean = text_columns_to_string(spreads.reset_index(drop=True))
    spreads_clean.to_parquet(output_file, index=False)
    print(f"  -> Saved clean spreads to: {output_file} {spreads_clean.shape}")
    return spreads_clean


if __name__ == "__main__":
    crsp_clean = clean_crsp()
    comp_clean = clean_compustat()
    check_merge(crsp_clean, comp_clean)
    clean_spreads(crsp_clean)