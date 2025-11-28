import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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

def calculate_indicators(df):
    """
    Calculates Bollinger Bands and ATR.
    """
    # --- Bollinger Bands (20, 2) ---
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['StdDev_20'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (2 * df['StdDev_20'])
    df['BB_Lower'] = df['SMA_20'] - (2 * df['StdDev_20'])
    
    # Band Width: (Upper - Lower) / SMA
    # This normalizes the width relative to price
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['SMA_20']
    
    # --- ATR (14) ---
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    df['ATR'] = true_range.rolling(window=14).mean()
    
    return df

def detect_squeeze(df, lookback=126):
    """
    Detects Squeeze conditions and Breakouts.
    Lookback default 126 days (~6 months) for defining "low" volatility.
    """
    signals = []
    
    # We scan the last 60 days for recent signals
    subset = df.iloc[-60:].copy()
    
    # Pre-calculate rolling minimums for the whole dataframe to check "multi-month lows"
    # We want to know if current BB Width is the lowest in 'lookback' periods
    df['Min_BB_Width'] = df['BB_Width'].rolling(window=lookback).min()
    df['Min_ATR'] = df['ATR'].rolling(window=lookback).min()
    
    for date, row in subset.iterrows():
        # Get the full context row from original df to access rolling min values correctly
        # (subsetting might lose the rolling window context if not careful, 
        # but here we calculated rolling on full df first, so it's fine)
        full_row = df.loc[date]
        
        bb_width = full_row['BB_Width']
        min_bb_width = full_row['Min_BB_Width']
        
        atr = full_row['ATR']
        min_atr = full_row['Min_ATR']
        
        close = full_row['Close']
        bb_upper = full_row['BB_Upper']
        bb_lower = full_row['BB_Lower']
        
        # --- 1. Bollinger Band Squeeze ---
        # Condition: Band Width is at or near the 6-month low (within 5% tolerance)
        is_bb_squeeze = bb_width <= (min_bb_width * 1.05)
        
        # --- 2. ATR Contraction ---
        # Condition: ATR is at or near the 6-month low
        is_atr_contraction = atr <= (min_atr * 1.05)
        
        signal_type = []
        if is_bb_squeeze:
            signal_type.append("BB Squeeze")
        if is_atr_contraction:
            signal_type.append("ATR Contraction")
            
        if signal_type:
            # Check for Breakout if in Squeeze
            # Note: Breakout might happen *after* the squeeze, but we check if it's happening *now*
            # while still relatively tight or just expanding.
            
            breakout = ""
            if close > bb_upper:
                breakout = " (Bullish Breakout)"
            elif close < bb_lower:
                breakout = " (Bearish Breakout)"
                
            signals.append({
                'Date': date,
                'Type': " + ".join(signal_type) + breakout,
                'Price': close,
                'BB_Width': bb_width,
                'ATR': atr
            })
            
    return signals

def plot_squeeze(df, ticker, signals):
    """
    Plots Price with BB and Squeeze indicators.
    """
    plt.figure(figsize=(14, 10))
    
    # --- Subplot 1: Price & Bollinger Bands ---
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(df.index, df['Close'], label='Close', color='blue', alpha=0.6)
    ax1.plot(df.index, df['BB_Upper'], label='Upper BB', color='gray', linestyle='--', alpha=0.5)
    ax1.plot(df.index, df['BB_Lower'], label='Lower BB', color='gray', linestyle='--', alpha=0.5)
    ax1.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], color='gray', alpha=0.1)
    
    # Plot Signals
    for sig in signals:
        if "Breakout" in sig['Type']:
            color = 'green' if 'Bullish' in sig['Type'] else 'red'
            marker = '^' if 'Bullish' in sig['Type'] else 'v'
            ax1.scatter(sig['Date'], sig['Price'], color=color, marker=marker, s=100, zorder=5, label='Breakout' if 'Breakout' not in [l.get_label() for l in ax1.get_lines()] else "")
        elif "Squeeze" in sig['Type'] or "Contraction" in sig['Type']:
            # Mark squeeze points with yellow dots
            ax1.scatter(sig['Date'], sig['Price'], color='orange', marker='.', s=50, zorder=4, label='Squeeze' if 'Squeeze' not in [l.get_label() for l in ax1.get_lines()] else "")

    plt.title(f'{ticker} Volatility Squeeze Analysis')
    plt.ylabel('Price')
    plt.legend(loc='best')
    plt.grid(True)
    
    # --- Subplot 2: BB Width ---
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(df.index, df['BB_Width'], label='BB Width', color='purple')
    # Plot threshold line (rolling min) - visualizing it might be cluttered, so just the width
    # Mark Squeeze zones
    squeeze_mask = df['BB_Width'] <= (df['BB_Width'].rolling(window=126).min() * 1.05)
    ax2.fill_between(df.index, 0, df['BB_Width'], where=squeeze_mask, color='orange', alpha=0.3, label='Squeeze Zone')
    
    plt.ylabel('BB Width')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # --- Subplot 3: ATR ---
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(df.index, df['ATR'], label='ATR (14)', color='brown')
    
    contraction_mask = df['ATR'] <= (df['ATR'].rolling(window=126).min() * 1.05)
    ax3.fill_between(df.index, 0, df['ATR'], where=contraction_mask, color='yellow', alpha=0.3, label='Contraction Zone')
    
    plt.ylabel('ATR')
    plt.xlabel('Date')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def analyze_batch(tickers_file='tickers.txt'):
    import os
    if not os.path.exists(tickers_file):
        if os.path.exists(os.path.join('..', '..', tickers_file)): tickers_file = os.path.join('..', '..', tickers_file)
        elif os.path.exists(os.path.join('..', tickers_file)): tickers_file = os.path.join('..', tickers_file)
        else:
            print(f"Error: {tickers_file} not found.")
            return

    with open(tickers_file, 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]
        
    print(f"Found {len(tickers)} tickers. Starting Volatility Squeeze analysis...")
    
    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")
        try:
            df = fetch_data(ticker)
            df = calculate_indicators(df)
            signals = detect_squeeze(df)
            
            if signals:
                # Group by recent status
                latest = signals[-1]
                print(f"  Latest Status: {latest['Type']} on {latest['Date'].date()}")
                print(f"  BB Width: {latest['BB_Width']:.4f}, ATR: {latest['ATR']:.2f}")
            else:
                print("  No recent volatility squeeze detected.")
                
        except Exception as e:
            print(f"  Error analyzing {ticker}: {e}")

if __name__ == "__main__":
    import os
    run_batch = False # Default to single for testing
    
    if os.path.exists('tickers.txt') and run_batch:
        analyze_batch('tickers.txt')
    elif os.path.exists('../tickers.txt') and run_batch:
        analyze_batch('../tickers.txt')
    else:
        ticker = "tcs.NS"
        print(f"Running Volatility Squeeze Analysis for {ticker}...")
        
        try:
            df = fetch_data(ticker)
            df = calculate_indicators(df)
            signals = detect_squeeze(df)
            
            if signals:
                print("\n--- Volatility Squeeze Signals (Last 60 Days) ---")
                for sig in signals:
                    print(f"{sig['Date'].date()}: {sig['Type']} (Price: {sig['Price']:.2f})")
            else:
                print("\nNo recent volatility squeeze detected.")
                
            plot_squeeze(df, ticker, signals)
                
        except Exception as e:
            print(f"Error: {e}")
