"""
VOLUME DIVERGENCE ANALYSIS
==========================

PURPOSE:
--------
This module analyzes the relationship between Price peaks/troughs and their associated Volume
to identify potential reversals based on Volume-Price exhaustion. 

WHAT IT DOES:
-------------
1. **Peak & Trough Detection**:
   - Uses `scipy.signal.argrelextrema` to find significant Price Highs (Peaks) and Lows (Troughs).
   - "Order" parameter controls the sensitivity (higher = more significant peaks).

2. **Divergence Detection (Peak-to-Peak)**:
   - **Bullish Reversal (Selling Exhaustion)**:
     - Identification: Price makes a Lower Low (LL) BUT Volume is Lower than the previous low.
     - Interpretation: Selling pressure is drying up despite the price drop.
   - **Bearish Reversal (Buying Exhaustion)**:
     - Identification: Price makes a Higher High (HH) BUT Volume is Lower than the previous high.
     - Interpretation: Buying interest is fading despite the price rise.

3. **Visual Analysis**:
   - Generates a chart with Price peaks/troughs marked.
   - Overlays divergence signals at the exact peak/trough point.

METHODOLOGY:
------------
Unlike rolling slopes, this method looks at the *relative strength* of participants at major turning points.
- If the second peak happens on significantly less volume, the trend is considered "exhausted".

CONFIGURATION:
--------------
- **ORDER**: 5 (Sensitivity for peak detection)
- **LOOKBACK_PERIODS**: 365 (1 Year of data)
- **DEFAULT_TICKER**: ICICIBANK.NS

USAGE:
------
    from experiments_space import volume_divergence_analysis
    results = volume_divergence_analysis.run_analysis(ticker="ICICIBANK.NS")
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
import traceback

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'ORDER': 5,            # Sensitivity for peak detection
    'LOOKBACK_PERIODS': 365,
    'DEFAULT_TICKER': 'ICICIBANK.NS'
}

# ==========================================
# DATA FETCHING
# ==========================================
def fetch_data(ticker, interval='1d', lookback_periods=365):
    """
    Fetches historical data for the given ticker.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_periods)
    
    # 15m restriction logic 
    if interval == '15m':
        lookback_periods = min(lookback_periods, 59)
        start_date = end_date - timedelta(days=lookback_periods)
    
    print(f"Fetching data for {ticker} from {start_date.date()} to {end_date.date()}...")
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, 
                     progress=False, auto_adjust=False, multi_level_index=False)
    
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    return df

# ==========================================
# CALCULATION LOGIC
# ==========================================
def detect_divergences(df, order=5):
    """
    Identifies divergence conditions based on Peak-to-Peak Volume comparison.
    """
    # 1. Find local peaks and troughs for Price
    # We use Close price for peaks
    price_values = df['Close'].values
    peak_indices = argrelextrema(price_values, np.greater_equal, order=order)[0]
    trough_indices = argrelextrema(price_values, np.less_equal, order=order)[0]
    
    # Mark them in the dataframe for plotting
    df['price_peak'] = np.nan
    df['price_trough'] = np.nan
    df.iloc[peak_indices, df.columns.get_loc('price_peak')] = df.iloc[peak_indices]['Close']
    df.iloc[trough_indices, df.columns.get_loc('price_trough')] = df.iloc[trough_indices]['Close']
    
    divergences = []
    
    # 2. Check for Bearish Divergence (Price Higher High, Volume Lower)
    for i in range(1, len(peak_indices)):
        prev_idx = peak_indices[i-1]
        curr_idx = peak_indices[i]
        
        prev_price = df.iloc[prev_idx]['Close']
        curr_price = df.iloc[curr_idx]['Close']
        
        # We look at Volume at these specific peaks
        prev_vol = df.iloc[prev_idx]['Volume']
        curr_vol = df.iloc[curr_idx]['Volume']
        
        # Bearish: Higher High in Price, but Lower Volume
        if curr_price > prev_price and curr_vol < prev_vol:
            divergences.append({
                'Type': 'Bearish Reversal (Buying Exhaustion)',
                'Date': df.index[curr_idx],
                'Price': curr_price,
                'Details': f"Price HH ({prev_price:.2f} -> {curr_price:.2f}), Volume ↓ ({prev_vol:,.0f} -> {curr_vol:,.0f})"
            })

    # 3. Check for Bullish Divergence (Price Lower Low, Volume Lower)
    for i in range(1, len(trough_indices)):
        prev_idx = trough_indices[i-1]
        curr_idx = trough_indices[i]
        
        prev_price = df.iloc[prev_idx]['Close']
        curr_price = df.iloc[curr_idx]['Close']
        
        prev_vol = df.iloc[prev_idx]['Volume']
        curr_vol = df.iloc[curr_idx]['Volume']
        
        # Bullish: Lower Low in Price, but Lower Volume (Selling exhaustion)
        if curr_price < prev_price and curr_vol < prev_vol:
            divergences.append({
                'Type': 'Bullish Reversal (Selling Exhaustion)',
                'Date': df.index[curr_idx],
                'Price': curr_price,
                'Details': f"Price LL ({prev_price:.2f} -> {curr_price:.2f}), Volume ↓ ({prev_vol:,.0f} -> {curr_vol:,.0f})"
            })
            
    return divergences

# ==========================================
# PLOTTING
# ==========================================
def plot_results(df, ticker, divergences, show_plot=True):
    """
    Generates a chart showing price, volume, and divergence markers.
    """
    fig = plt.figure(figsize=(14, 10))
    
    # Subplot 1: Price
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.5)
    
    # Plot Peaks/Troughs
    peaks = df[df['price_peak'].notna()]
    troughs = df[df['price_trough'].notna()]
    ax1.scatter(peaks.index, peaks['Close'], color='orange', marker='v', s=40, alpha=0.5, label='Peaks')
    ax1.scatter(troughs.index, troughs['Close'], color='cyan', marker='^', s=40, alpha=0.5, label='Troughs')
    
    # Plot Divergence Markers
    for div in divergences:
        if 'Bearish' in div['Type']:
            color = 'red'
            marker = 'v'
            y_pos = div['Price'] * 1.02
        else: # Bullish
            color = 'green'
            marker = '^'
            y_pos = div['Price'] * 0.98
            
        ax1.scatter(div['Date'], y_pos, 
                    color=color, marker=marker, s=150, 
                    label=div['Type'], zorder=10, 
                    edgecolors='black', linewidths=1.5)
        
    ax1.set_title(f'{ticker} - Volume Divergence Analysis (Peak-to-Peak, Order: {CONFIG["ORDER"]})', fontweight='bold')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.2)
    
    # Subplot 2: Volume
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    colors = np.where(df['Close'].diff() >= 0, 'green', 'red')
    ax2.bar(df.index, df['Volume'], color=colors, alpha=0.4, label='Volume')
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.2)
    
    # Clean up legend
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='best', fontsize='small')
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
        
    return fig

# ==========================================
# MAIN API
# ==========================================
def run_analysis(ticker=None, show_plot=True, config=None, df=None):
    """
    Standard entry point for analysis.
    """
    current_config = CONFIG.copy()
    if config:
        current_config.update(config)
        
    if ticker is None:
        ticker = current_config['DEFAULT_TICKER']
        
    try:
        # 1. Fetch Data
        if df is None:
            df = fetch_data(ticker, lookback_periods=current_config['LOOKBACK_PERIODS'])
        else:
            df = df.copy()
            
        # 2. Calculate
        divergences = detect_divergences(df, order=current_config['ORDER'])
        
        # 3. Plot
        fig = plot_results(df, ticker, divergences, show_plot=show_plot)
        
        # 4. Result Structure
        latest_signal = divergences[-1] if divergences else None
        
        return {
            'success': True,
            'ticker': ticker,
            'divergences': divergences,
            'latest_signal': latest_signal,
            'figure': fig,
            'dataframe': df
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'ticker': ticker,
            'error': str(e)
        }

# ==========================================
# BATCH EXECUTION
# ==========================================
def analyze_batch(tickers_file):
    import os
    if not os.path.exists(tickers_file):
        print(f"Error: {tickers_file} not found.")
        return
        
    with open(tickers_file, 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]
        
    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")
        result = run_analysis(ticker, show_plot=False)
        if result['success'] and result['latest_signal']:
            print(f"  Signal: {result['latest_signal']['Type']} on {result['latest_signal']['Date'].date()}")
        elif not result['success']:
            print(f"  Error: {result.get('error')}")

if __name__ == "__main__":
    import sys
    
    target_ticker = CONFIG['DEFAULT_TICKER']
    if len(sys.argv) > 1:
        target_ticker = sys.argv[1]
        
    print(f"Running Refined Volume Divergence Analysis for {target_ticker}...")
    result = run_analysis(target_ticker, show_plot=True)
    
    if result['success']:
        divs = result['divergences']
        if divs:
            print(f"\n--- Detected {len(divs)} Peak-to-Peak Divergences ---")
            for div in divs[-10:]: # Show last 10
                 print(f"{div['Type']} on {div['Date'].date()} @ {div['Price']:.2f}")
                 print(f"  {div['Details']}")
        else:
            print("No divergences found.")
    else:
        print(f"Analysis Failed: {result['error']}")
