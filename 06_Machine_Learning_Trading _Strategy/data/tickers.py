# data/tickers.py
import pandas as pd
from pathlib import Path

def sp500_tickers():
    """
    It loads the S&P500 tickers from the CSV (data/sp500_tickers.csv).
    The file needs a column called 'Symbol'.
    """

    path = Path("data/sp500_tickers.csv")
    df = pd.read_csv(path)
    # Yahoo uses '-' instead of '.' in tickers
    tickers = (
        df["Symbol"]
        .astype(str)
        .str.replace(".", "-", regex=False)
        .tolist()
    )

    # remove duplicates and sort
    return sorted(set(tickers))
