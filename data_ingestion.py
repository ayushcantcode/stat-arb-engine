import yfinance as yf
import pandas as pd
import os

def download_ticker_data(ticker="SPY", period=5):
    """
    Downloads X years of historical daily data for ANY given ticker 
    and saves it to a dynamically named CSV file.
    """
    
    print(f"Connecting to Yahoo Finance API for {ticker}...")

    asset = yf.Ticker(ticker)
    print(f"Downloading {period} years of daily data for {ticker}...")
    historical_data = asset.history(period=f"{period}y")


    if historical_data.empty:
        print(f"Error: No data pulled for {ticker}. Check your network or API limits.")
        return
    
    if not os.path.exists("data"):
        os.makedirs("data")

    file_path = f"data/{ticker.lower()}_{period}yr_data.csv"
    historical_data.to_csv(file_path)

    print(f"Success! Data saved to {file_path}")
    print(f"\n------First 5 Rows of {ticker} Data------")
    print(historical_data.head())

    if __name__ == "__main__":
        download_ticker_data("SPY")