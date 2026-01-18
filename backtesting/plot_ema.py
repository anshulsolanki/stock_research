import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def fetch_data(ticker, period="5y"):
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Calculate Indicators
    df['50EMA'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['200EMA'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    return df

def plot_ema(ticker):
    df = fetch_data(ticker)
    
    if df.empty:
        print("No data found.")
        return

    # Identify Crossovers
    df['Signal'] = np.where(df['50EMA'] > df['200EMA'], 1, 0)
    df['Crossover'] = df['Signal'].diff()
    
    golden_cross = df[df['Crossover'] == 1]
    death_cross = df[df['Crossover'] == -1]
    
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Close'], label='Close Price', alpha=0.5, color='gray')
    plt.plot(df.index, df['50EMA'], label='50 EMA', color='blue', alpha=0.8)
    plt.plot(df.index, df['200EMA'], label='200 EMA', color='orange', alpha=0.8)
    
    # Plot Crossovers
    plt.scatter(golden_cross.index, golden_cross['50EMA'], color='green', marker='^', s=100, label='Golden Cross (Buy)')
    plt.scatter(death_cross.index, death_cross['50EMA'], color='red', marker='v', s=100, label='Death Cross (Sell)')
    
    plt.title(f'{ticker} - 50 EMA vs 200 EMA Crossover')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = f"{ticker}_ema_crossover.png"
    plt.savefig(output_file)
    print(f"Chart saved to {output_file}")
    
    # Try to open the file (macOS specific)
    os.system(f"open {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", type=str, nargs='?', default="DABUR.NS", help="Stock Ticker (default: DABUR.NS)")
    args = parser.parse_args()
    
    plot_ema(args.ticker)
