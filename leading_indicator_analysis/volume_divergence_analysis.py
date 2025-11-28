import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats

def fetch_data(ticker, interval='1d', days_back=365):
    """
    Fetches historical data.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=False, multi_level_index=False)
    
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    return df

def calculate_trends(df, window=14):
    """
    Calculates the slope of Price and Volume over a rolling window to identify trends.
    Using a larger window (14) helps smoothen the noise.
    """
    # We'll use a rolling window to calculate the slope of the regression line
    # for both Close Price and Volume.
    
    def get_slope(series):
        y = series.values
        x = np.arange(len(y))
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope

    # Calculate slopes
    df['Price_Slope'] = df['Close'].rolling(window=window).apply(get_slope, raw=False)
    df['Volume_Slope'] = df['Volume'].rolling(window=window).apply(get_slope, raw=False)
    
    return df

def detect_volume_divergence(df):
    """
    Detects Volume Divergences based on Price and Volume slopes.
    
    1. Bullish Reversal (Weak Selling): Price Falls (Slope < 0), Volume Drops (Slope < 0)
    2. Bearish Reversal (Weak Buying): Price Rises (Slope > 0), Volume Drops (Slope < 0)
    """
    divergences = []
    
    # We look at the latest completed trends (e.g., last row)
    # But to find "points", we scan the dataframe
    
    # Iterate through the dataframe (starting from window size)
    # We'll report significant divergences where slopes are clearly defined
    
    # For reporting, we'll iterate over the last 60 days to capture more context
    subset = df.iloc[-60:].copy()
    
    last_signal_type = None
    
    for date, row in subset.iterrows():
        price_slope = row['Price_Slope']
        vol_slope = row['Volume_Slope']
        price = row['Close']
        
        if pd.isna(price_slope) or pd.isna(vol_slope):
            continue
            
        current_signal = None
        details = ""
        
        # Case A: Price Falls, Volume Drops -> Weak Selling -> Bullish Reversal
        if price_slope < 0 and vol_slope < 0:
            current_signal = 'Bullish Reversal (Weak Selling)'
            details = f"Price Falling (Slope: {price_slope:.2f}), Volume Dropping (Slope: {vol_slope:.2f})"
            
        # Case B: Price Rises, Volume Drops -> Weak Buying -> Bearish Reversal
        elif price_slope > 0 and vol_slope < 0:
            current_signal = 'Bearish Reversal (Weak Buying)'
            details = f"Price Rising (Slope: {price_slope:.2f}), Volume Dropping (Slope: {vol_slope:.2f})"
            
        # Filter consecutive signals to reduce noise
        # We only record the signal if it's different from the last one
        # OR if enough time has passed (e.g., 5 days) - simpler to just check type change for now
        if current_signal and current_signal != last_signal_type:
            divergences.append({
                'Type': current_signal,
                'Date': date,
                'Price': price,
                'Details': details
            })
            last_signal_type = current_signal
        elif not current_signal:
             # Reset if no signal conditions are met (optional, but helps if we want to catch re-entries)
             # For strict smoothing, we might keep last_signal_type until it flips
             pass
            
    return divergences

def plot_volume_divergence(df, ticker, divergences):
    """
    Plots Price and Volume with divergence markers.
    """
    plt.figure(figsize=(14, 10))
    
    # Price Plot
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df.index, df['Close'], label='Close Price', color='blue')
    plt.title(f'{ticker} Price & Volume Divergences')
    plt.ylabel('Price')
    plt.grid(True)
    
    # Plot Divergence Markers on Price
    for div in divergences:
        color = 'red' if 'Bearish' in div['Type'] else 'green'
        marker = 'v' if 'Bearish' in div['Type'] else '^'
        # Plot slightly above/below price for visibility
        y_pos = div['Price'] * (1.01 if 'Bearish' in div['Type'] else 0.99)
        ax1.scatter(div['Date'], y_pos, color=color, marker=marker, s=100, label=div['Type'], zorder=5)
        
    # Remove duplicate labels in legend
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())

    # Volume Plot
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    colors = np.where(df['Close'].diff() >= 0, 'green', 'red') # Color volume by price change
    ax2.bar(df.index, df['Volume'], color=colors, alpha=0.5, label='Volume')
    plt.ylabel('Volume')
    plt.xlabel('Date')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def analyze_batch(tickers_file='tickers.txt'):
    """
    Reads tickers from a file and performs Volume Divergence analysis.
    """
    import os
    
    if not os.path.exists(tickers_file):
        # Try looking in parent directory if not found in current
        if os.path.exists(os.path.join('..', '..', tickers_file)):
             tickers_file = os.path.join('..', '..', tickers_file)
        elif os.path.exists(os.path.join('..', tickers_file)):
             tickers_file = os.path.join('..', tickers_file)
        else:
            print(f"Error: {tickers_file} not found.")
            return
        
    with open(tickers_file, 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]
        
    print(f"Found {len(tickers)} tickers in {tickers_file}. Starting batch analysis...")
    
    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")
        try:
            df = fetch_data(ticker)
            df = calculate_trends(df, window=14) # 14-day trend window for smoothing
            divergences = detect_volume_divergence(df)
            
            if divergences:
                # Print only the last detected divergence to avoid clutter
                last_div = divergences[-1]
                print(f"  Latest Signal: {last_div['Type']} on {last_div['Date'].date()} at {last_div['Price']:.2f}")
                print(f"    {last_div['Details']}")
            else:
                print("  No recent volume divergences detected.")
                
        except Exception as e:
            print(f"  Error analyzing {ticker}: {e}")

if __name__ == "__main__":
    # Check if we should run batch or single
    import os
    
    run_batch = False

    # Check for tickers.txt in various locations (current, parent, root)
    if os.path.exists('tickers.txt') and run_batch:
        analyze_batch('tickers.txt')
    elif os.path.exists('../tickers.txt') and run_batch:
        analyze_batch('../tickers.txt')
    elif os.path.exists('../../tickers.txt') and run_batch:
        analyze_batch('../../tickers.txt')
    else:
        # Fallback to single ticker
        ticker = "LT.NS"
        print(f"Running Volume Divergence Analysis for {ticker}...")
        
        try:
            df = fetch_data(ticker)
            df = calculate_trends(df, window=14)
            divergences = detect_volume_divergence(df)
            
            if divergences:
                print("\n--- Volume Divergences Detected (Last 60 Days) ---")
                for div in divergences:
                    print(f"{div['Type']} on {div['Date'].date()} at Price {div['Price']:.2f}")
                    print(f"  {div['Details']}")
            else:
                print("\nNo recent volume divergences detected.")
                
            plot_volume_divergence(df, ticker, divergences)
                
        except Exception as e:
            print(f"Error: {e}")
