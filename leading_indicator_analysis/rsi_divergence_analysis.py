import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.signal import argrelextrema

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

def calculate_rsi(df, period=14):
    """
    Calculates the Relative Strength Index (RSI).
    """
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def detect_rsi_divergence(df, order=5):
    """
    Detects Bullish and Bearish RSI divergences.
    
    Bullish Divergence: Price Lower Low, RSI Higher Low
    Bearish Divergence: Price Higher High, RSI Lower High
    """
    # Find local peaks and troughs for Price
    df['price_peak'] = df.iloc[argrelextrema(df['Close'].values, np.greater_equal, order=order)[0]]['Close']
    df['price_trough'] = df.iloc[argrelextrema(df['Close'].values, np.less_equal, order=order)[0]]['Close']
    
    # Find local peaks and troughs for RSI
    df['rsi_peak'] = df.iloc[argrelextrema(df['RSI'].values, np.greater_equal, order=order)[0]]['RSI']
    df['rsi_trough'] = df.iloc[argrelextrema(df['RSI'].values, np.less_equal, order=order)[0]]['RSI']
    
    divergences = []
    
    # Check for Bearish Divergence (Price Higher High, RSI Lower High)
    # We look at points where both Price and RSI have peaks
    # Note: Exact alignment of peaks might not happen, so we look for recent peaks
    
    price_peaks = df[df['price_peak'].notna()]
    
    if len(price_peaks) >= 2:
        # Iterate through recent peaks to find divergence
        # We compare the last two significant peaks
        curr_peak = price_peaks.iloc[-1]
        prev_peak = price_peaks.iloc[-2]
        
        # Get RSI values at these specific dates
        curr_rsi = df.loc[curr_peak.name]['RSI']
        prev_rsi = df.loc[prev_peak.name]['RSI']
        
        # Check condition: Price Higher High AND RSI Lower High
        if curr_peak['Close'] > prev_peak['Close'] and curr_rsi < prev_rsi:
             # Additional check: Ensure RSI is in overbought territory (optional but common)
             # if prev_rsi > 70: 
            divergences.append({
                'Type': 'Bearish Divergence',
                'Date': curr_peak.name,
                'Price': curr_peak['Close'],
                'Details': f"Price HH ({prev_peak['Close']:.2f} -> {curr_peak['Close']:.2f}), RSI LH ({prev_rsi:.2f} -> {curr_rsi:.2f})"
            })

    # Check for Bullish Divergence (Price Lower Low, RSI Higher Low)
    price_troughs = df[df['price_trough'].notna()]
    
    if len(price_troughs) >= 2:
        curr_trough = price_troughs.iloc[-1]
        prev_trough = price_troughs.iloc[-2]
        
        curr_rsi = df.loc[curr_trough.name]['RSI']
        prev_rsi = df.loc[prev_trough.name]['RSI']
        
        # Check condition: Price Lower Low AND RSI Higher Low
        if curr_trough['Close'] < prev_trough['Close'] and curr_rsi > prev_rsi:
            # Additional check: Ensure RSI is in oversold territory (optional)
            # if prev_rsi < 30:
            divergences.append({
                'Type': 'Bullish Divergence',
                'Date': curr_trough.name,
                'Price': curr_trough['Close'],
                'Details': f"Price LL ({prev_trough['Close']:.2f} -> {curr_trough['Close']:.2f}), RSI HL ({prev_rsi:.2f} -> {curr_rsi:.2f})"
            })
            
    return divergences

def plot_rsi_divergence(df, ticker, divergences):
    """
    Plots Price and RSI with divergence markers.
    """
    plt.figure(figsize=(14, 8))
    
    # Price Plot
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['Close'], label='Close Price', color='blue')
    
    # Plot Divergence Markers on Price
    for div in divergences:
        color = 'red' if 'Bearish' in div['Type'] else 'green'
        marker = 'v' if 'Bearish' in div['Type'] else '^'
        plt.scatter(div['Date'], div['Price'], color=color, marker=marker, s=100, label=div['Type'], zorder=5)
        
    plt.title(f'{ticker} Price & Divergences')
    plt.legend()
    plt.grid(True)
    
    # RSI Plot
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df['RSI'], label='RSI (14)', color='purple')
    plt.axhline(70, linestyle='--', alpha=0.5, color='red')
    plt.axhline(30, linestyle='--', alpha=0.5, color='green')
    plt.title('Relative Strength Index (RSI)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def analyze_batch(tickers_file='tickers.txt'):
    """
    Reads tickers from a file and performs RSI Divergence analysis for each.
    """
    import os
    
    if not os.path.exists(tickers_file):
        print(f"Error: {tickers_file} not found.")
        return
        
    with open(tickers_file, 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]
        
    print(f"Found {len(tickers)} tickers in {tickers_file}. Starting batch analysis...")
    
    results = []
    
    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")
        try:
            df = fetch_data(ticker)
            df = calculate_rsi(df)
            divergences = detect_rsi_divergence(df)
            
            current_rsi = df['RSI'].iloc[-1]
            print(f"  Current RSI: {current_rsi:.2f}")
            
            if divergences:
                print("  --- Divergences Detected ---")
                for div in divergences:
                    print(f"  {div['Type']} on {div['Date'].date()} at Price {div['Price']:.2f}")
                    print(f"    {div['Details']}")
            else:
                print("  No recent divergences.")
                
        except Exception as e:
            print(f"  Error analyzing {ticker}: {e}")

if __name__ == "__main__":
    run_batch = False

    # Check if we should run batch or single
    import os
    if os.path.exists('tickers.txt') and run_batch:
        analyze_batch()
    else:
        ticker = "axisbank.NS"

        print(f"Running RSI Divergence Analysis for {ticker}...")
        
        try:
            df = fetch_data(ticker)
            df = calculate_rsi(df)
            divergences = detect_rsi_divergence(df)
            
            print(f"\nCurrent RSI: {df['RSI'].iloc[-1]:.2f}")
            
            if divergences:
                print("\n--- RSI Divergences Detected (Recent) ---")
                for div in divergences:
                    print(f"{div['Type']} on {div['Date'].date()} at Price {div['Price']:.2f}")
                    print(f"  {div['Details']}")
            else:
                print("\nNo recent RSI divergences detected.")
                
            #plot_rsi_divergence(df, ticker, divergences)
            
        except Exception as e:
            print(f"Error: {e}")
