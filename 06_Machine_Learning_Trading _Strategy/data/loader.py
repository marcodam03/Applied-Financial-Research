# data/loader.py
import yfinance as yf
import pandas as pd
from pathlib import Path
from data.tickers import sp500_tickers


def download_prices(
    tickers=None,
    outfile="data/sp500_prices.csv",
    start="2020-11-01",
    end="2025-11-01",
    period=None,
):
    """
    Download daily Adjusted Close and Volume data from Yahoo Finance.
    """

    # If no tickers are given, it uses the S&P500 list
    if tickers is None:
        all_tickers = sp500_tickers()
        tickers = all_tickers
        print(f"Using {len(tickers)} S&P500 tickers")

    # Ensure the output directory exists
    OUTFILE = Path(outfile)
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading price data for {len(tickers)} tickers ({period}) ...")

    # Download data from Yahoo Finance
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    # Yahoo returns a MultiIndex only when downloading more than one ticker
    if isinstance(raw.columns, pd.MultiIndex):
        # Just take the Adj Close and Volume blocks
        adj = raw["Adj Close"]
        vol = raw["Volume"]
    else:
        # In case of a single ticker: fix column name to keep the same structure as above
        adj = raw[["Adj Close"]].rename(columns={"Adj Close": tickers[0]})
        vol = raw[["Volume"]].rename(columns={"Volume": tickers[0]})

    # Convert the DataFrame into (Date, Ticker, Value) format
    adj_long = adj.stack().rename("Adj Close")
    vol_long = vol.stack().rename("Volume")

    # Combine into a single DataFrame
    data = pd.concat([adj_long, vol_long], axis=1).reset_index()
    data = data.rename(columns={"level_0": "Date", "level_1": "Ticker"})

    # Make sure values are numeric and remove rows without prices
    data["Adj Close"] = pd.to_numeric(data["Adj Close"], errors="coerce")
    data["Volume"] = pd.to_numeric(data["Volume"], errors="coerce")
    data = data.dropna(subset=["Adj Close"])

    # Save dataset to CSV
    data.to_csv(OUTFILE, index=False)
    print(f"Saved {len(data):,} rows for {data['Ticker'].nunique()} tickers → {OUTFILE}")

    return data
