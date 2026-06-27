# features/build_dataset.py
import pandas as pd
from pathlib import Path

def build_ml_dataset(prices_csv="data/sp500_prices.csv", out_csv="data/ml_dataset.csv"):
    """
    It builds the monthly ML dataset from daily S&P500 prices.

    It loads the daily Adjusted Close and Volume, builds daily and monthly returns,
    computes the required features (1M/3M/6M momentum, 3M volatility, 50D MA ratio,
    daily RSI(14) aggregated to month-end), and creates a binary result variable:
    1 if next-month return is > 0, 0 otherwise.
    """

    RAW = Path(prices_csv)
    OUT = Path(out_csv)

    # Load only the columns we need and make sure Date appears as datetime
    df = pd.read_csv(RAW, usecols=["Date","Adj Close","Volume","Ticker"], parse_dates=["Date"])
    df["Adj Close"] = pd.to_numeric(df["Adj Close"], errors="coerce")
    df = df.dropna(subset=["Date","Ticker","Adj Close"])

    # Daily price matrix: rows = dates, columns = tickers
    px_d = df.pivot_table(index="Date", columns="Ticker", values="Adj Close").sort_index()
    ret_d = px_d.pct_change()

    # Monthly prices (last trading day of each month) and monthly returns
    px_m = px_d.resample("ME").last()
    ret_m = px_m.pct_change()

    # --- FEATURES ---
    # 1M, 3M, 6M momentum
    f1m = ret_m.copy()
    f3m = (px_m / px_m.shift(3) - 1)
    f6m = (px_m / px_m.shift(6) - 1)

    # 3M volatility: daily rolling std over 63 trading days, then take month-end
    vol3 = ret_d.rolling(63).std().resample("ME").last()

    # 50D moving-average ratio at month-end
    ma50 = (px_d / px_d.rolling(50).mean()).resample("ME").last()

    # RSI(14D) on daily data, then sampled at month-end
    delta = px_d.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta.clip(upper=0))
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi_d = 100 - (100 / (1 + rs))
    rsi_m = rsi_d.resample("ME").last()

    # Helper to go from wide to long
    def to_long(w, name):
        return w.stack(future_stack=True).rename(name).reset_index()

    # Merge all features on (Date, Ticker)
    feats = (
        to_long(f1m, "ret_1m")
        .merge(to_long(f3m, "ret_3m"), on=["Date","Ticker"])
        .merge(to_long(f6m, "ret_6m"), on=["Date","Ticker"])
        .merge(to_long(vol3, "vol_3m"), on=["Date","Ticker"])
        .merge(to_long(ma50, "ma_ratio_50D"), on=["Date","Ticker"])
        .merge(to_long(rsi_m, "RSI_14D"), on=["Date","Ticker"])
    )

    # TARGET: next-month return and binary label
    future = to_long(ret_m.shift(-1), "ret_next1m")
    data = feats.merge(future, on=["Date","Ticker"])
    data["target"] = (data["ret_next1m"] > 0).astype(int)

    # Drop rows with missing values and sort
    data = data.dropna().sort_values(["Date","Ticker"]).reset_index(drop=True)

    # Save to CSV
    data.to_csv(OUT, index=False)
    print(f"Saved {len(data):,} rows with {data['Ticker'].nunique()} tickers → {OUT}")
    return data
