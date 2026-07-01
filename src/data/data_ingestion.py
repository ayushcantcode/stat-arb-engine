import yaml
import yfinance as yf
import pandas as pd
from pathlib import Path
import duckdb

# Estabilishing Dynamic Paths
# This ensures the script works no matter where you run it from
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
DB_DIR = BASE_DIR / "db"
DB_PATH = DB_DIR / "market_data.duckdb"

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

    print(f"Target Universe: {tickers}")
    print(f"Timeframe: {start_date} to {end_date}")

    # Ensure the raw data directory exists
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)   

    print("Initiating API connection to yfinance...")

    # Download the bulk data matrix
    # auto_adjust -> gives us adjusted values for all the columns (High, Low, Close, etc)
    raw_data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=use_adj_close)

    # DATA QUALITY AND CLEANING PROTOCOLS
    print("\nExecuting Data Quality & Cleaning Protocols...")

    # Because auto_adjust handles the math, the 'Close' column is now precisely what we need.
    # .copy() prevents Pandas SettingWithCopy warnings later
    price_matrix = raw_data['Close'].copy()

    if use_adj_close:
        print("[+] Successfully isolated 'Adjusted Close' prices (Corporate Actions Neutralized).")
    else:
        print("[-] Warning: 'Adj Close' disabled in config or not found. Using raw 'Close'.")

    # Missing Data fix (Matrix Alignment)
    initial_rows = len(price_matrix)
    # Drops all the rows with NA or NaN values.
    # inplace=True -> Makes the change in the pandas df price matrix itself rather than creating a new one.
    price_matrix.dropna(inplace=True)
    final_rows = len(price_matrix)
    dropped_rows = initial_rows - final_rows

    if dropped_rows > 0:
        print(f"[+] Matrix Alignment Complete: Dropped {dropped_rows} rows containing NaN values (Holidays/Halts).")
    else:
        print("[+] Matrix Alignment Complete: No missing data detected.")   

    # Convert the index (Date) to a column so it writes cleanly to SQL
    price_matrix.reset_index(inplace=True)

    # ANALYTICAL BACKEND ROUTING (DUCKDB)
    print("\nRouting matrix to DuckDB Analytical Engine...")

    try:
        # Connect to DuckDB (creates the file if it doesn't exist)
        conn = duckdb.connect(str(DB_PATH))

        # DuckDB can natively read the 'price_matrix' Pandas DataFrame from memory
        conn.execute("CREATE OR REPLACE TABLE historical_prices AS SELECT * FROM price_matrix")

        # Validate the write operation
        row_count = conn.execute("SELECT COUNT(*) FROM historical_prices").fetchone()[0]
        conn.close()

        print(f"[+] SUCCESS: Matrix committed to DuckDB at {DB_PATH}")
        print(f"[+] Database Table 'historical_prices' initialized with {row_count} rows.")
    
    except Exception as e:
        print(f"[-] FATAL ERROR: Database routing failed. Details: {e}")
        exit(1)

    # Save a CSV Backup just for visual inspection
    csv_output = RAW_DATA_DIR / 'historical_prices.csv'
    price_matrix.to_csv(csv_output, index=False)
    print(f"[+] Clean backup CSV saved to {csv_output}")
    print(f"Shape: {price_matrix.shape[0]} trading days, {price_matrix.shape[1] - 1} assets")

if __name__ == '__main__':
    ingest_data()