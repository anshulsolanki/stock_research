"""
MACD (Moving Average Convergence Divergence) ANALYSIS TOOL
===========================================================

PURPOSE:
--------
This module performs comprehensive MACD (Moving Average Convergence Divergence) analysis, a popular
momentum and trend-following indicator used in technical analysis. MACD helps identify:
- Trend direction changes (bullish/bearish crossovers)
- Momentum strength (histogram expansion/contraction)
- Potential reversals through divergence detection

WHAT IT DOES:
-------------
1. **MACD Calculation**: Computes three key components
   - MACD Line = Fast EMA (12) - Slow EMA (26)
   - Signal Line = 9-period EMA of MACD Line
   - Histogram = MACD Line - Signal Line

2. **Trend Identification**:
   - Bullish: MACD Line > Signal Line (upward momentum)
   - Bearish: MACD Line < Signal Line (downward momentum)
   - Crossovers generate buy/sell signals

3. **Momentum Analysis**:
   - Strengthening: Histogram expanding (momentum increasing)
   - Weakening: Histogram contracting (momentum decreasing)

4. **Divergence Detection**: Identifies price-indicator divergences
   - Bullish Divergence: Price makes lower low BUT MACD makes higher low (reversal signal)
   - Bearish Divergence: Price makes higher high BUT MACD makes lower high (reversal signal)

5. **Visual Analysis**: Generates dual-panel charts showing:
   - Price action with identified peaks/troughs
   - MACD line, SignalLine, and color-coded histogram

METHODOLOGY:
------------
MACD Formula:
- Fast EMA: Exponential Moving Average with 12-period span
- Slow EMA: Exponential Moving Average with 26-period span
- MACD Line: Fast EMA - Slow EMA
- Signal Line: 9-period EMA of MACD Line
- Histogram: MACD Line - Signal Line

Signal Generation:
- Buy Signal (Bullish Crossover): MACD crosses above Signal Line
- Sell Signal (Bearish Crossover): MACD crosses below Signal Line

Divergence Detection:
- Uses scipy.signal to find local peaks (highs) and troughs (lows)
- Compares price action vs MACD action to identify divergences

KEY METRICS:
------------
- MACD Line: Primary indicator value
- Signal Line: Trigger line for crossovers
- Histogram: Visual representation of MACD-Signal difference
- Trend: Bullish or Bearish based on line positions
- Momentum: Strengthening or Weakening based on histogram
- Crossover Signal: Recent buy/sell crossover events
- Divergences: List of detected bullish/bearish divergences

CONFIGURATION:
--------------
Default parameters (customizable via MACD_CONFIG):
- FAST: 12 periods (fast EMA)
- SLOW: 26 periods (slow EMA)
- SIGNAL: 9 periods (signal line EMA)
- INTERVAL: '1d' (daily data, also supports '1wk', '1h', '15m', etc.)
- LOOKBACK_PERIODS: 730 days (2 years of history)

USAGE:
------
Run as standalone script:
    python macd_analysis.py

Or import and use programmatically:
    from macd_analysis import run_analysis
    results = run_analysis(ticker="AAPL", show_plot=True, config={'FAST': 8, 'SLOW': 21})

OUTPUT:
-------
Returns dictionary containing:
- MACD Line, Signal Line, Histogram values
- Trend and Momentum indicators
- Recent crossover signals
- Detected divergences with dates and prices
- Matplotlib figure for visualization
- Full DataFrame with all calculated indicators

TYPICAL USE CASES:
------------------
1. Trend confirmation: Validate price trends with MACD direction
2. Entry/exit timing: Use crossovers for trade signals
3. Momentum tracking: Monitor histogram for strength changes
4. Reversal detection: Identify divergences as early warning signs
5. Multi-timeframe analysis: Run on different intervals for comprehensive view
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.signal import argrelextrema

# Configuration
MACD_CONFIG = {
    # MACD Parameters
    'FAST': 12,
    'SLOW': 26,
    'SIGNAL': 9,
    
    # Data Fetching
    'INTERVAL': '1d',  # Data interval: '1d', '1wk', '1h', '15m', etc.
    'LOOKBACK_PERIODS': 730,  # 2 year
    
    # Execution Control
    'DEFAULT_TICKER': 'HDFCBANK.NS',
    'BATCH_RELATIVE_PATH': '../data/tickers_list.json',
    'RUN_BATCH': False
}

def fetch_data(ticker, config=None):
    """
    Fetches historical data using config parameters.
    """
    # Use provided config or default to global MACD_CONFIG
    cfg = config if config else MACD_CONFIG
    
    interval = cfg.get('INTERVAL', MACD_CONFIG['INTERVAL'])
    lookback_periods = cfg.get('LOOKBACK_PERIODS', MACD_CONFIG['LOOKBACK_PERIODS'])
    
    # Limit lookback for 15m interval (Yahoo Finance restriction: max 60 days)
    if interval == '15m':
        lookback_periods = min(lookback_periods, 59)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_periods)
    
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=False, multi_level_index=False)
    
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    return df

def calculate_macd(df, config=None):
    """
    Calculates MACD Line, Signal Line, and Histogram using config.
    """
    # Use provided config or default to global MACD_CONFIG
    cfg = config if config else MACD_CONFIG
    
    fast = cfg.get('FAST', MACD_CONFIG['FAST'])
    slow = cfg.get('SLOW', MACD_CONFIG['SLOW'])
    signal = cfg.get('SIGNAL', MACD_CONFIG['SIGNAL'])
    
    # Calculate Fast and Slow EMAs
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    
    # MACD Line
    df['MACD_Line'] = ema_fast - ema_slow
    
    # Signal Line
    df['Signal_Line'] = df['MACD_Line'].ewm(span=signal, adjust=False).mean()
    
    # Histogram
    df['MACD_Histogram'] = df['MACD_Line'] - df['Signal_Line']
    
    return df

def detect_divergence(df, order=5):
    """
    Detects Bullish and Bearish divergences.
    """
    # Find local peaks and troughs
    # 'order' determines the window size for local extrema
    
    # Highs (Peaks)
    df['is_peak'] = df.iloc[argrelextrema(df['Close'].values, np.greater_equal, order=order)[0]]['Close']
    
    # Lows (Troughs)
    df['is_trough'] = df.iloc[argrelextrema(df['Close'].values, np.less_equal, order=order)[0]]['Close']
    
    divergences = []
    
    # We need at least two peaks or two troughs to compare
    peaks = df[df['is_peak'].notna()]
    troughs = df[df['is_trough'].notna()]
    
    # Check for Bearish Divergence (Price Higher High, MACD Lower High)
    if len(peaks) >= 2:
        last_peak = peaks.iloc[-1]
        prev_peak = peaks.iloc[-2]
        
        if last_peak['Close'] > prev_peak['Close'] and last_peak['MACD_Line'] < prev_peak['MACD_Line']:
            divergences.append({
                'Type': 'Bearish Divergence',
                'Date': last_peak.name,
                'Price': last_peak['Close'],
                'Details': f"Price HH ({prev_peak['Close']:.2f} -> {last_peak['Close']:.2f}), MACD LH ({prev_peak['MACD_Line']:.2f} -> {last_peak['MACD_Line']:.2f})"
            })
            
    # Check for Bullish Divergence (Price Lower Low, MACD Higher Low)
    if len(troughs) >= 2:
        last_trough = troughs.iloc[-1]
        prev_trough = troughs.iloc[-2]
        
        if last_trough['Close'] < prev_trough['Close'] and last_trough['MACD_Line'] > prev_trough['MACD_Line']:
            divergences.append({
                'Type': 'Bullish Divergence',
                'Date': last_trough.name,
                'Price': last_trough['Close'],
                'Details': f"Price LL ({prev_trough['Close']:.2f} -> {last_trough['Close']:.2f}), MACD HL ({prev_trough['MACD_Line']:.2f} -> {last_trough['MACD_Line']:.2f})"
            })
            
    return divergences

def analyze_macd(df):
    """
    Analyzes Trend, Momentum, and Divergences.
    """
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    results = {
        'MACD_Line': latest['MACD_Line'],
        'Signal_Line': latest['Signal_Line'],
        'Histogram': latest['MACD_Histogram'],
        'Trend': 'Neutral',
        'Momentum': 'Neutral',
        'Crossover_Signal': None,
        'Divergences': []
    }
    
    # 1. Trend Identification (Crossover)
    if latest['MACD_Line'] > latest['Signal_Line']:
        results['Trend'] = 'Bullish'
        if prev['MACD_Line'] <= prev['Signal_Line']:
            results['Crossover_Signal'] = 'Bullish Crossover (Buy)'
    else:
        results['Trend'] = 'Bearish'
        if prev['MACD_Line'] >= prev['Signal_Line']:
            results['Crossover_Signal'] = 'Bearish Crossover (Sell)'
            
    # 2. Momentum Measurement (Histogram)
    # Compare absolute size of histogram to previous
    if abs(latest['MACD_Histogram']) > abs(prev['MACD_Histogram']):
        results['Momentum'] = 'Strengthening'
    else:
        results['Momentum'] = 'Weakening'
        
    # 3. Divergences
    results['Divergences'] = detect_divergence(df)
    
    return results

def plot_macd(df, ticker, show_plot=True, config=None):
    """
    Plots Price, MACD, Signal, and Histogram.
    
    Args:
        df: DataFrame with MACD data
        ticker: Stock ticker symbol
        show_plot: If True, displays plot. If False, just returns the figure.
        config: Configuration dictionary
    
    Returns:
        matplotlib.figure.Figure: The plot figure object
    """
    # Use provided config or default to global MACD_CONFIG
    cfg = config if config else MACD_CONFIG
    fast = cfg.get('FAST', MACD_CONFIG['FAST'])
    slow = cfg.get('SLOW', MACD_CONFIG['SLOW'])
    signal = cfg.get('SIGNAL', MACD_CONFIG['SIGNAL'])

    fig = plt.figure(figsize=(14, 10))

    # Price Plot
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['Close'], label='Close Price', color='blue')
    plt.title(f'{ticker} Price')
    plt.legend()
    plt.grid(True)
    
    # MACD Plot
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df['MACD_Line'], label='MACD Line', color='blue')
    plt.plot(df.index, df['Signal_Line'], label='Signal Line', color='orange')
    
    # Histogram
    colors = np.where(df['MACD_Histogram'] >= 0, 'green', 'red')
    plt.bar(df.index, df['MACD_Histogram'], color=colors, alpha=0.5, label='Histogram')
    
    plt.title(f'{ticker} MACD ({fast}, {slow}, {signal})')
    plt.legend()
    plt.grid(True)
    
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
        dict: Analysis results containing:
            - 'success': bool, whether analysis succeeded
            - 'ticker': str, analyzed ticker
            - 'macd_line': float, latest MACD line value
            - 'signal_line': float, latest Signal line value
            - 'histogram': float, latest histogram value
            - 'trend': str, 'Bullish' or 'Bearish'
            - 'momentum': str, 'Strengthening' or 'Weakening'
            - 'crossover_signal': str or None, recent crossover signal
            - 'divergences': list, detected divergences
            - 'figure': matplotlib.figure.Figure, plot figure object
            - 'dataframe': pd.DataFrame, full analysis data
            - 'error': str, error message if analysis failed
    """
    # Set non-interactive backend when not showing plot (e.g., for Flask/web)
    if not show_plot:
        matplotlib.use('Agg', force=True)
    
    try:
        # Merge provided config with defaults
        current_config = MACD_CONFIG.copy()
        if config:
            current_config.update(config)

        # Fetch and calculate
        df = fetch_data(ticker, config=current_config)
        df = calculate_macd(df, config=current_config)
        analysis = analyze_macd(df)
        
        # Generate plot
        fig = plot_macd(df, ticker, show_plot=show_plot, config=current_config)
        
        return {
            'success': True,
            'ticker': ticker,
            'macd_line': analysis['MACD_Line'],
            'signal_line': analysis['Signal_Line'],
            'histogram': analysis['Histogram'],
            'trend': analysis['Trend'],
            'momentum': analysis['Momentum'],
            'crossover_signal': analysis['Crossover_Signal'],
            'divergences': analysis['Divergences'],
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
        
    print(f"Found {len(tickers)} unique tickers in {json_file}. Starting MACD analysis...")
    
    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")
        try:
            df = fetch_data(ticker)
            df = calculate_macd(df)
            analysis = analyze_macd(df)
            
            print(f"  Trend: {analysis['Trend']}, Momentum: {analysis['Momentum']}")
            if analysis['Crossover_Signal']:
                print(f"  Signal: {analysis['Crossover_Signal']}")
            
            if analysis['Divergences']:
                last_div = analysis['Divergences'][-1]
                print(f"  Latest Divergence: {last_div['Type']} on {last_div['Date'].date()}")
                
        except Exception as e:
            print(f"  Error analyzing {ticker}: {e}")

if __name__ == "__main__":
    import os
    
    # Load execution parameters from config
    run_batch = MACD_CONFIG['RUN_BATCH']
    default_ticker = MACD_CONFIG['DEFAULT_TICKER']
    batch_relative_path = MACD_CONFIG['BATCH_RELATIVE_PATH']
    
    # Resolve batch file path
    batch_file = os.path.join(os.path.dirname(__file__), batch_relative_path)
    
    if run_batch and os.path.exists(batch_file):
        analyze_batch(batch_file)
    else:
        print(f"Running MACD Analysis for {default_ticker}...")
        
        try:
            df = fetch_data(default_ticker)
            df = calculate_macd(df)
            analysis = analyze_macd(df)
            
            print("\n--- MACD Analysis Results ---")
            print(f"MACD Line:      {analysis['MACD_Line']:.2f}")
            print(f"Signal Line:    {analysis['Signal_Line']:.2f}")
            print(f"Histogram:      {analysis['Histogram']:.2f}")
            print(f"Trend:          {analysis['Trend']}")
            print(f"Momentum:       {analysis['Momentum']}")
            
            if analysis['Crossover_Signal']:
                print(f"Signal:         {analysis['Crossover_Signal']}")
                
            if analysis['Divergences']:
                print("--- Divergences Detected (Recent) ---")
                for div in analysis['Divergences']:
                    print(f"{div['Type']} on {div['Date'].date()} at Price {div['Price']:.2f}")
                    print(f"  {div['Details']}")
            else:
                print("No recent divergences detected.")
                
            # Plot
            plot_macd(df, default_ticker)
            
        except Exception as e:
            print(f"Error: {e}")

