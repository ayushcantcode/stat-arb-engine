import yaml
import yfinance as yf
import pandas as pd
from pathlib import Path
import duckdb
import sys

# Estabilishing Dynamic Paths
# This ensures the script works no matter where you run it from
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
RAW_DATA_DIR = BASE_DIR / "data" / "raw"


def load_config(path: Path) -> dict:
    """Parses the YAML configuration file into a Python dictionary."""
    try:
        with open(path, "r") as file:
            # reads the raw text inside your.yaml and deserializes it into a python dictionary.
            return yaml.safe_load(file) 
    except Exception as e:
        print(f"FATAL ERROR: Failed to parse config.yaml. Check your syntax. Details: {e}")
        exit(1)

def ingest_data():
    """Pulls parameters, downloads market data, cleans it, and routes to DuckDB."""
    print(f"Loading Configuration from: {CONFIG_PATH}...")
    config = load_config(CONFIG_PATH)

    # Extracting parameters from the yaml file
    tickers = config['universe']['assets']
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    use_adj_close = config['data']['adjusted_close']
    interval = config['data']['interval']

    # Read the DB path from config instead of hardcoding it, so there's a
    # single source of truth, previously this was computed independently
    # of the yaml value, which happened to match by coincidence.
    db_path = BASE_DIR / config['database']['path']

    print(f"Target Universe: {tickers}")
    print(f"Timeframe: {start_date} to {end_date} (interval = {interval})")

    # Ensure the raw data directory exists
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    print("Initiating API connection to yfinance...")

    # Download the bulk data matrix
    # auto_adjust -> gives us adjusted values for all the columns (High, Low, Close, etc)
    raw_data = yf.download(
        tickers, start=start_date, end=end_date,
        interval=interval, auto_adjust=use_adj_close, progress=False,
    )

    if raw_data.empty:
        print("FATAL ERROR: yfinance returned no data at all. Check tickers/dates/network.")
        sys.exit(1)

    # DATA QUALITY AND CLEANING PROTOCOLS
    print("\nExecuting Data Quality & Cleaning Protocols...")

    # Because auto_adjust handles the math, the 'Close' column is now precisely what we need.
    # .copy() prevents Pandas SettingWithCopy warnings later
    price_matrix = raw_data['Close'].copy()

    if isinstance(price_matrix, pd.Series):
        price_matrix = price_matrix.to_frame(name=tickers[0])

    if use_adj_close:
        print("[+] Successfully isolated 'Adjusted Close' prices (Corporate Actions Neutralized).")
    else:
        print("[-] Warning: 'Adj Close' disabled in config or not found. Using raw 'Close'.")

    failed_tickers = price_matrix.columns[price_matrix.isna().all()].tolist()

    if failed_tickers:
        print(f"[-] WARNING: No data returned for {failed_tickers}, dropping from this run's universe.")
        price_matrix = price_matrix.drop(columns=failed_tickers)

    if price_matrix.shape[1] == 0:
        print("FATAL ERROR: No valid tickers remain after removing failed downloads.")
        sys.exit(1)

    # Missing Data fix (Matrix Alignment)
    initial_rows = len(price_matrix)
    # Drops all the rows with NA or NaN values.
    # inplace=True -> Makes the change in the pandas df price matrix itself rather than creating a new one.
    rows_with_na = price_matrix[price_matrix.isna().any(axis=1)]
    dropped_rows = len(rows_with_na)

    if dropped_rows > 0:
        sample_dates = [d.strftime('%Y-%m-%d') for d in rows_with_na.index[:5]]
        print(f"[+] Matrix Alignment: Dropping {dropped_rows} rows containing NaN values.")
        print(f"    Sample dropped dates: {sample_dates}")

        # A handful of dropped rows is normal (mismatched holiday calendars,
        # single-day halts). A LARGE fraction usually means one asset has a
        # much shorter history than the rest of the universe -- worth
        # surfacing loudly rather than silently truncating years of data.
        drop_fraction = dropped_rows / initial_rows
        if drop_fraction > 0.05:
            print(
                f"[!] WARNING: Dropped {drop_fraction:.1%} of rows -- unusually high for "
                f"holiday/halt gaps alone. Check for an asset with a shorter listing history "
                f"than the rest of the universe (e.g. a newer ETF or a stock with a later IPO)."
            )

    else:
        print("[+] Matrix Alignment Complete: No missing data detected.")   

    price_matrix = price_matrix.dropna()
    # Convert the index (Date) to a column so it writes cleanly to SQL
    price_matrix.reset_index(inplace=True)

    # ANALYTICAL BACKEND ROUTING (DUCKDB)
    print("\nRouting matrix to DuckDB Analytical Engine...")

    try:
        # Connect to DuckDB (creates the file if it doesn't exist)
        conn = duckdb.connect(str(db_path))

        # DuckDB can natively read the 'price_matrix' Pandas DataFrame from memory
        conn.execute("CREATE OR REPLACE TABLE historical_prices AS SELECT * FROM price_matrix")

        # Validate the write operation
        row_count = conn.execute("SELECT COUNT(*) FROM historical_prices").fetchone()[0]
        conn.close()

        # Guard against silently committing an empty table -- this is the
        # failure mode the all-NaN-ticker bug used to produce with no error.
        if row_count == 0:
            print(" FATAL ERROR: 0 rows written to database. Aborting.")
            sys.exit(1)
 
        print(f" SUCCESS: Matrix committed to DuckDB at {db_path}")
        print(f" Database Table 'historical_prices' initialized with {row_count} rows.")
    
    except Exception as e:
        print(f" FATAL ERROR: Database routing failed. Details: {e}")
        exit(1)

    # Save a CSV Backup just for visual inspection
    csv_output = RAW_DATA_DIR / 'historical_prices.csv'
    price_matrix.to_csv(csv_output, index=False)
    print(f"[+] Clean backup CSV saved to {csv_output}")
    print(f"Shape: {price_matrix.shape[0]} trading days, {price_matrix.shape[1] - 1} assets")

if __name__ == '__main__':
    ingest_data()