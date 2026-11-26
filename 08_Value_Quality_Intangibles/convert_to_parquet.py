"""
Convert the WRDS CSV files in data/raw/ to Parquet (same folder).

    python convert_to_parquet.py                    # every CSV in data/raw/
    python convert_to_parquet.py crsp_daily.csv     # only the files named
"""
import os
import sys
import time
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.csv as pv
    import pyarrow.parquet as pq
except ImportError:
    print("\n[ERROR] pyarrow is not installed. Please run: pip install pyarrow\n")
    sys.exit(1)

DATA_DIR = "data/raw"

# Files bigger than this are converted block by block: the whole CRSP daily file
# (6+ GB) does not fit in memory.
STREAM_ABOVE_MB = 1024

# Column types of the CRSP daily file. When streaming, Arrow fixes the types from
# the first block, so codes that look like numbers there (HdrCUSIP 68391610, later
# 88160R10) must be declared as text.
CRSP_DAILY_TYPES = {
    "PERMNO": pa.int64(),
    "HdrCUSIP": pa.string(),
    "CUSIP9": pa.string(),
    "Ticker": pa.string(),
    "PERMCO": pa.int64(),
    "DlyCalDt": pa.timestamp("s"),
    "DlyBid": pa.float64(),
    "DlyAsk": pa.float64(),
}


def stream_csv_to_parquet(file_path: str, target_path: str, column_types: dict = None) -> int:
    """Read the CSV block by block and append each block to the Parquet file, so
    memory holds one block at a time. Written to a .part file first: an interrupted
    run leaves no half file that the next run would skip."""
    reader = pv.open_csv(
        file_path,
        read_options=pv.ReadOptions(block_size=64 << 20),
        convert_options=pv.ConvertOptions(column_types=column_types or {}),
    )
    part_path = target_path + ".part"
    rows = 0
    with pq.ParquetWriter(part_path, reader.schema, compression="snappy") as writer:
        for batch in reader:
            writer.write_batch(batch)
            rows += batch.num_rows
    os.replace(part_path, target_path)
    return rows


def convert_file(file_path: str, target_name: str, column_types: dict = None):
    target_path = os.path.join(DATA_DIR, target_name)

    # 1. Skip files already successfully converted
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"  -> Skipping '{target_name}': already exists ({size_mb:.1f} MB).")
        return

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"\nProcessing {os.path.basename(file_path)} ({file_size_mb:.1f} MB) -> {target_name}...")
    start_time = time.time()

    # 2. Large file: block by block (no fallback, the Pandas one would not fit in memory)
    if file_size_mb > STREAM_ABOVE_MB:
        rows = stream_csv_to_parquet(file_path, target_path, column_types)
        out_size_mb = os.path.getsize(target_path) / (1024 * 1024)
        elapsed = time.time() - start_time
        print(f"  -> Saved {target_name} ({rows:,} rows, streamed)")
        print(f"  -> Size: {file_size_mb:.1f} MB -> {out_size_mb:.1f} MB (-{(1 - out_size_mb / file_size_mb) * 100:.1f}%) in {elapsed:.1f}s")
        return

    # 3. Fast C++ PyArrow engine (processes 1 GB in ~5-10 seconds without memory thrashing)
    try:
        table = pv.read_csv(file_path, convert_options=pv.ConvertOptions(column_types=column_types or {}))
        pq.write_table(table, target_path, compression="snappy")
        out_size_mb = os.path.getsize(target_path) / (1024 * 1024)
        elapsed = time.time() - start_time
        print(f"  -> Saved {target_name} ({table.num_rows:,} rows, {table.num_columns} cols)")
        print(f"  -> Size: {file_size_mb:.1f} MB -> {out_size_mb:.1f} MB (-{(1 - out_size_mb / file_size_mb) * 100:.1f}%) in {elapsed:.1f}s")
        return
    except Exception as err:
        print(f"  -> PyArrow direct read failed ({err}), falling back to direct Pandas write...")

    # 4. Fallback: Pandas without row-by-row date loops
    df = pd.read_csv(file_path, low_memory=False)
    df.to_parquet(target_path, engine="pyarrow", compression="snappy", index=False)
    out_size_mb = os.path.getsize(target_path) / (1024 * 1024)
    elapsed = time.time() - start_time
    print(f"  -> Saved {target_name} ({len(df):,} rows, {len(df.columns)} cols) in {elapsed:.1f}s")


def convert_csvs_to_parquet(only: list = None):
    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] Directory '{DATA_DIR}' not found.")
        return

    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if only:
        missing = [f for f in only if f not in csv_files]
        if missing:
            print(f"[ERROR] Not in '{DATA_DIR}/': {missing}")
            return
        csv_files = list(only)
    if not csv_files:
        print(f"No CSV files found in '{DATA_DIR}/'.")
        return

    for file_name in csv_files:
        file_path = os.path.join(DATA_DIR, file_name)

        # Inspect headers to identify dataset type
        header = pd.read_csv(file_path, nrows=5)
        cols = set(header.columns)

        if "PERMNO" in cols and ("MthCalDt" in cols or "MthRet" in cols):
            print(f"[CRSP Monthly] Detected in {file_name}")
            convert_file(file_path, "crsp_monthly.parquet")

        elif "PERMNO" in cols and "DlyCalDt" in cols:
            print(f"[CRSP Daily] Detected in {file_name}")
            types = {c: t for c, t in CRSP_DAILY_TYPES.items() if c in cols}
            convert_file(file_path, "crsp_daily.parquet", types)

        elif "GVKEY" in cols and ("datadate" in cols or "fyear" in cols):
            print(f"[Compustat Annual] Detected in {file_name}")
            convert_file(file_path, "compustat_annual.parquet")

        else:
            target_name = file_name.replace(".csv", ".parquet")
            print(f"[Generic CSV] Detected in {file_name}")
            convert_file(file_path, target_name)


if __name__ == "__main__":
    convert_csvs_to_parquet(sys.argv[1:])
