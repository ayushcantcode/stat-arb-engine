import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def engineer_features(ticker, period=5):
    """
    Loads raw price data, calculates log returns, and tests for staionarity.
    """

    print(f"--- Loading data for {ticker} ---")

    file_path = f"data/{ticker.lower()}_{period}yr_data.csv"

    try:
        df = pd.read_csv(file_path, index_col="Date", parse_dates=True)

    except FileNotFoundError:
        print(f"Error: Could not find {file_path}. Make sure you run data_ingestion.py")
        return
    
    df['Log_Return'] = np.log(df['Close']) - np.log(df['Close'].shift(1))
    df = df.dropna()

    print("Log returns calculated successfully.")

    print("\n--- Running Augmented Dickey-Fuller Test ---")

    adf_result = adfuller(df['Log_Return'])
    adf_statistic = adf_result[0]
    p_value = adf_result[1]

    print(f"ADF Statistic: {adf_statistic:.4f}")
    print(f"p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("Conclusion: p-value < 0.05. The data is STATIONARY. We are cleared to build the model.")
    else:
        print("Conclusion: p-value >= 0.05. The data is NON-STATIONARY. We cannot use this for ML.")

if __name__ == "__main__":
    # Execute the function for SPY
    engineer_features("SPY")