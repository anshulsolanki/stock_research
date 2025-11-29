import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.signal import argrelextrema

# Configuration
RSI_CONFIG = {
    # RSI Parameters
    'PERIOD': 14,
    'ORDER': 5,
    'RSI_OVERBOUGHT': 70,
    'RSI_OVERSOLD': 30,
    
    # Data Fetching
    'INTERVAL': '1d',
    'LOOKBACK_PERIODS': 365 * 2,
    
    # Execution Control
    'DEFAULT_TICKER': 'LT.NS',
    'BATCH_RELATIVE_PATH': '../data/tickers_list.json',
    'RUN_BATCH': False
}

def fetch_data(ticker, config=None):
    """
    Fetches historical data using config parameters.
    """
    # Use provided config or default to global RSI_CONFIG
    cfg = config if config else RSI_CONFIG
    
    interval = cfg.get('INTERVAL', RSI_CONFIG['INTERVAL'])
    lookback_periods = cfg.get('LOOKBACK_PERIODS', RSI_CONFIG['LOOKBACK_PERIODS'])
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_periods)
    
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=False, multi_level_index=False)
    
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    return df

def calculate_rsi(df, config=None):
    """
    Calculates the Relative Strength Index (RSI) using Wilder's Smoothing Method.
    
    This matches the standard RSI calculation used by TradingView, Yahoo Finance, and other platforms.
    Wilder's smoothing is essentially an EMA with alpha = 1/period.
    """
    cfg = config if config else RSI_CONFIG
    period = cfg.get('PERIOD', RSI_CONFIG['PERIOD'])
    
    # Calculate price changes
    delta = df['Close'].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Use Wilder's smoothing method (EMA with alpha = 1/period)
    # This is equivalent to ewm with alpha=1/period or adjust=False
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def detect_rsi_divergence(df, config=None):
    """
    Detects Bullish and Bearish RSI divergences.
    
    Bullish Divergence: Price Lower Low, RSI Higher Low (potential reversal up)
    Bearish Divergence: Price Higher High, RSI Lower High (potential reversal down)
    """
    cfg = config if config else RSI_CONFIG
    order = cfg.get('ORDER', RSI_CONFIG['ORDER'])
     
    # Find local peaks and troughs for Price
    price_peak_indices = argrelextrema(df['Close'].values, np.greater_equal, order=order)[0]
    price_trough_indices = argrelextrema(df['Close'].values, np.less_equal, order=order)[0]
    
    # Find local peaks and troughs for RSI
    rsi_peak_indices = argrelextrema(df['RSI'].values, np.greater_equal, order=order)[0]
    rsi_trough_indices = argrelextrema(df['RSI'].values, np.less_equal, order=order)[0]
    
    # Mark peaks and troughs in dataframe for visualization
    df['price_peak'] = np.nan
    df['price_trough'] = np.nan
    df['rsi_peak'] = np.nan
    df['rsi_trough'] = np.nan
    
    df.iloc[price_peak_indices, df.columns.get_loc('price_peak')] = df.iloc[price_peak_indices]['Close']
    df.iloc[price_trough_indices, df.columns.get_loc('price_trough')] = df.iloc[price_trough_indices]['Close']
    df.iloc[rsi_peak_indices, df.columns.get_loc('rsi_peak')] = df.iloc[rsi_peak_indices]['RSI']
    df.iloc[rsi_trough_indices, df.columns.get_loc('rsi_trough')] = df.iloc[rsi_trough_indices]['RSI']
    
    divergences = []
    
    # Check for Bearish Divergence (Price Higher High, RSI Lower High)
    # Iterate through all consecutive price peaks
    for i in range(1, len(price_peak_indices)):
        prev_idx = price_peak_indices[i-1]
        curr_idx = price_peak_indices[i]
        
        prev_price = df.iloc[prev_idx]['Close']
        curr_price = df.iloc[curr_idx]['Close']
        prev_rsi = df.iloc[prev_idx]['RSI']
        curr_rsi = df.iloc[curr_idx]['RSI']
        
        # Bearish Divergence: Price makes higher high BUT RSI makes lower high
        if curr_price > prev_price and curr_rsi < prev_rsi:
            divergences.append({
                'Type': 'Bearish Divergence',
                'Date': df.index[curr_idx],
                'Price': curr_price,
                'Details': f"Price HH ({prev_price:.2f} -> {curr_price:.2f}), RSI LH ({prev_rsi:.2f} -> {curr_rsi:.2f})"
            })

    # Check for Bullish Divergence (Price Lower Low, RSI Higher Low)
    # Iterate through all consecutive price troughs
    for i in range(1, len(price_trough_indices)):
        prev_idx = price_trough_indices[i-1]
        curr_idx = price_trough_indices[i]
        
        prev_price = df.iloc[prev_idx]['Close']
        curr_price = df.iloc[curr_idx]['Close']
        prev_rsi = df.iloc[prev_idx]['RSI']
        curr_rsi = df.iloc[curr_idx]['RSI']
        
        # Bullish Divergence: Price makes lower low BUT RSI makes higher low
        if curr_price < prev_price and curr_rsi > prev_rsi:
            divergences.append({
                'Type': 'Bullish Divergence',
                'Date': df.index[curr_idx],
                'Price': curr_price,
                'Details': f"Price LL ({prev_price:.2f} -> {curr_price:.2f}), RSI HL ({prev_rsi:.2f} -> {curr_rsi:.2f})"
            })
            
    return divergences

def plot_rsi_divergence(df, ticker, show_plot=True, config=None, divergences=None):
    """
    Plots Price and RSI with divergence markers and configuration details.
    """
    cfg = config if config else RSI_CONFIG
    overbought = cfg.get('RSI_OVERBOUGHT', RSI_CONFIG['RSI_OVERBOUGHT'])
    oversold = cfg.get('RSI_OVERSOLD', RSI_CONFIG['RSI_OVERSOLD'])
    period = cfg.get('PERIOD', RSI_CONFIG['PERIOD'])
    order = cfg.get('ORDER', RSI_CONFIG['ORDER'])
    interval = cfg.get('INTERVAL', RSI_CONFIG['INTERVAL'])
    lookback = cfg.get('LOOKBACK_PERIODS', RSI_CONFIG['LOOKBACK_PERIODS'])
    
    fig = plt.figure(figsize=(14, 10))
    
    # Price Plot
    ax1 = plt.subplot(2, 1, 1)
    plt.plot(df.index, df['Close'], label='Close Price', color='blue', linewidth=1.5)
    
    # Plot price peaks and troughs
    price_peaks = df[df['price_peak'].notna()]
    price_troughs = df[df['price_trough'].notna()]
    
    if not price_peaks.empty:
        plt.scatter(price_peaks.index, price_peaks['Close'], color='orange', marker='v', s=80, alpha=0.6, label='Price Peaks', zorder=3)
    if not price_troughs.empty:
        plt.scatter(price_troughs.index, price_troughs['Close'], color='cyan', marker='^', s=80, alpha=0.6, label='Price Troughs', zorder=3)
    
    # Plot Divergence Markers on Price
    if divergences:
        for div in divergences:
            color = 'red' if 'Bearish' in div['Type'] else 'green'
            marker = 'v' if 'Bearish' in div['Type'] else '^'
            plt.scatter(div['Date'], div['Price'], color=color, marker=marker, s=200, 
                       label=div['Type'], zorder=5, edgecolors='black', linewidths=2)
        
    plt.title(f'{ticker} Price & RSI Divergences', fontsize=14, fontweight='bold')
    plt.ylabel('Price', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # RSI Plot
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    plt.plot(df.index, df['RSI'], label=f'RSI ({period})', color='purple', linewidth=1.5)
    plt.axhline(overbought, linestyle='--', alpha=0.5, color='red', label=f'Overbought ({overbought})')
    plt.axhline(oversold, linestyle='--', alpha=0.5, color='green', label=f'Oversold ({oversold})')
    plt.fill_between(df.index, 0, df['RSI'], where=(df['RSI'] >= overbought), 
                     color='red', alpha=0.1, interpolate=True)
    plt.fill_between(df.index, 0, df['RSI'], where=(df['RSI'] <= oversold), 
                     color='green', alpha=0.1, interpolate=True)
    
    # Plot RSI peaks and troughs
    rsi_peaks = df[df['rsi_peak'].notna()]
    rsi_troughs = df[df['rsi_trough'].notna()]
    
    if not rsi_peaks.empty:
        plt.scatter(rsi_peaks.index, rsi_peaks['RSI'], color='orange', marker='v', s=80, alpha=0.6, label='RSI Peaks', zorder=3)
    if not rsi_troughs.empty:
        plt.scatter(rsi_troughs.index, rsi_troughs['RSI'], color='cyan', marker='^', s=80, alpha=0.6, label='RSI Troughs', zorder=3)
    
    # Plot divergence markers on RSI
    if divergences:
        for div in divergences:
            color = 'red' if 'Bearish' in div['Type'] else 'green'
            marker = 'v' if 'Bearish' in div['Type'] else '^'
            # Get RSI value at divergence date
            rsi_val = df.loc[div['Date']]['RSI']
            plt.scatter(div['Date'], rsi_val, color=color, marker=marker, s=200, 
                       zorder=5, edgecolors='black', linewidths=2)
    
    plt.title('Relative Strength Index (RSI)', fontsize=12)
    plt.ylabel('RSI', fontsize=11)
    plt.xlabel('Date', fontsize=11)
    plt.ylim(0, 100)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
        
    return fig

def run_analysis(ticker, show_plot=True, config=None):
    """
    Main analysis function that can be called from a GUI or other scripts.
    
    Args:
        ticker: Stock ticker symbol
        show_plot: If True, displays the plot. If False, just returns the figure.
        config: Optional dictionary to override default configuration
    
    Returns:
        dict: Analysis results
    """
    # Set non-interactive backend when not showing plot (e.g., for Flask/web)
    if not show_plot:
        matplotlib.use('Agg', force=True)
    
    try:
        # Merge provided config with defaults
        current_config = RSI_CONFIG.copy()
        if config:
            current_config.update(config)

        # Fetch and calculate
        df = fetch_data(ticker, config=current_config)
        df = calculate_rsi(df, config=current_config)
        divergences = detect_rsi_divergence(df, config=current_config)
        
        current_rsi = df['RSI'].iloc[-1]
        
        # Generate plot
        fig = plot_rsi_divergence(df, ticker, show_plot=show_plot, config=current_config, divergences=divergences)
        
        return {
            'success': True,
            'ticker': ticker,
            'current_rsi': current_rsi,
            'divergences': divergences,
            'figure': fig,
            'dataframe': df
        }
        
    except Exception as e:
        return {
            'success': False,
            'ticker': ticker,
            'error': str(e)
        }

def analyze_batch(json_file):
    """
    Reads tickers from a JSON file and performs RSI Divergence analysis for each.
    """
    import os
    import json
    
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found.")
        return
        
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    # tickers_list.json is a simple dictionary of "Name": "Ticker"
    tickers = list(data.values())
            
    # Remove duplicates
    tickers = list(set(tickers))
        
    print(f"Found {len(tickers)} unique tickers in {json_file}. Starting RSI Divergence analysis...")
    
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
    import os
    
    # Load execution parameters from config
    run_batch = RSI_CONFIG['RUN_BATCH']
    default_ticker = RSI_CONFIG['DEFAULT_TICKER']
    batch_relative_path = RSI_CONFIG['BATCH_RELATIVE_PATH']
    
    # Resolve batch file path
    batch_file = os.path.join(os.path.dirname(__file__), batch_relative_path)

    if run_batch and os.path.exists(batch_file):
        analyze_batch(batch_file)
    else:
        print(f"Running RSI Divergence Analysis for {default_ticker}...")
        
        result = run_analysis(default_ticker, show_plot=True)
        
        if result['success']:
            print(f"\nCurrent RSI: {result['current_rsi']:.2f}")
            
            if result['divergences']:
                print("\n--- RSI Divergences Detected (Recent) ---")
                for div in result['divergences']:
                    print(f"{div['Type']} on {div['Date'].date()} at Price {div['Price']:.2f}")
                    print(f"  {div['Details']}")
            else:
                print("\nNo recent RSI divergences detected.")
        else:
            print(f"Error: {result.get('error')}")

