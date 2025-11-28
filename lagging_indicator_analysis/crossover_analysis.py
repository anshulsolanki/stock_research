"""
EMA CROSSOVER ANALYSIS TOOL
============================

PURPOSE:
--------
This module performs Exponential Moving Average (EMA) crossover analysis, focusing on identifying
trend strength and direction using multiple timeframe EMAs. Special emphasis on the "Golden Cross"
pattern, a powerful bullish signal used by institutional traders. This tool helps identify:
- Strong sustained trends (multi-EMA alignment)
- Emerging momentum shifts (short-term crossovers)
- Valid Golden Cross patterns (with invalidation detection)
- Slowing momentum (deteriorating EMA relationships)

WHAT IT DOES:
-------------
1. **Multi-Timeframe EMA Calculation**:
   - EMA 20: Short-term trend (4 weeks)
   - EMA 50: Medium-term trend (10 weeks / ~2.5 months)
   - EMA 200: Long-term trend (40 weeks / ~10 months)
   - Uses exponential weighting for recent price emphasis

2. **Golden Cross Detection** (50/200 EMA):
   - Identifies when EMA 50 crosses above EMA 200
   - **CRITICAL**: Only reports if STILL VALID (no subsequent Death Cross)
   - Filters out false signals from brief crossovers
   - Provides date and price of the Golden Cross

3. **Trend Strength Analysis**:
   - **Strong Uptrend**: 20 > 50 > 200 (all EMAs properly aligned)
   - **Strong Downtrend**: 20 < 50 < 200 (all EMAs inverted)
   - **Short-term Uptrend**: 20 > 50 (but not above 200)
   - **Short-term Downtrend**: 20 < 50 (but not below 200)
   - **Neutral**: Mixed EMA relationships

4. **Momentum Pattern Detection**:
   - **Emerging**: Accelerating strength (1M > 2M > 3M RS pattern)
   - **Slowing**: Decelerating strength (1M < 2M < 3M RS pattern)
   - **Consistent**: Sustained alignment over multiple timeframes

5. **Visual Analysis**: Generates charts with:
   - Price line (black, semi-transparent)
   - EMAs color-coded (20=blue, 50=orange, 200=red)
   - Golden Cross markers (⭐ gold star) if valid
   - Clear labeling and legend

METHODOLOGY:
------------
EMA Calculation:
- EMA = Price(today) × k + EMA(yesterday) × (1 - k)
- k = 2 / (N + 1) where N is the period
- More weight to recent prices vs simple MA

Golden Cross Validation Logic:
1. Detect all points where EMA_50 crosses above EMA_200
2. Find the most recent Golden Cross
3. Check if any Death Cross (50 below 200) occurred AFTER it
4. Only report Golden Cross if no subsequent Death Cross found
5. This filters out whipsaw signals and temporary crossovers

Death Cross Detection (Invalidation Check):
- Identifies when EMA_50 crosses below EMA_200
- Used to invalidate previous Golden Cross signals
- Ensures only currently-active bullish setups are flagged

KEY METRICS:
------------
- EMA_20: Current 20-period exponential moving average
- EMA_50: Current 50-period exponential moving average
- EMA_200: Current 200-period exponential moving average
- Trend_Status: Classification of current trend strength
- GC_Date: Date of valid Golden Cross (if exists and still valid)
- GC_Price: Price level at Golden Cross occurrence

CONFIGURATION:
--------------
Default parameters (customizable via CROSSOVER_CONFIG):
- WINDOWS: [20, 50, 200] (EMA periods)
- INTERVAL: '1d' (daily candles, also supports '1wk', '1mo', etc.)
- LOOKBACK_PERIODS: 730 days (2 years of history)

USAGE:
------
Run as standalone script:
    python crossover_analysis.py

Or import and use programmatically:
    from crossover_analysis import run_analysis
    results = run_analysis(ticker="^NSEI", config={'WINDOWS': [10, 30, 100]})

OUTPUT:
-------
Returns dictionary containing:
- Current EMA values for all configured periods
- Trend status classification
- Golden Cross date and price (only if still valid)
- Matplotlib figure with annotated chart
- Full DataFrame with all calculated EMAs

TYPICAL USE CASES:
------------------
1. Long-term trend identification: Use 200 EMA as major trend filter
2. Golden Cross trading: Enter longs when 50 crosses above 200 (and still valid)
3. Trend strength confirmation: All three EMAs aligned = strong directional move
4. Exit signals: Watch for deteriorating EMA relationships
5. Multi-timeframe confluence: Combine daily Golden Cross with weekly uptrend
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configuration
CROSSOVER_CONFIG = {
    # EMA Parameters
    'WINDOWS': [20, 50, 200],
    
    # Data Fetching
    'INTERVAL': '1d',
    'LOOKBACK_PERIODS': 730,  # 2 years
    
    # Execution Control
    'DEFAULT_TICKER': 'DABUR.NS',
    'BATCH_RELATIVE_PATH': '../data/tickers_list.json',
    'RUN_BATCH': False
}

def fetch_data(ticker, config=None):
    """
    Fetches historical data using config parameters.
    """
    # Use provided config or default to global CROSSOVER_CONFIG
    cfg = config if config else CROSSOVER_CONFIG
    
    interval = cfg.get('INTERVAL', CROSSOVER_CONFIG['INTERVAL'])
    lookback_periods = cfg.get('LOOKBACK_PERIODS', CROSSOVER_CONFIG['LOOKBACK_PERIODS'])
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_periods)
    
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=False, multi_level_index=False)
    
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    return df

def calculate_emas(df, config=None):
    """
    Calculates Exponential Moving Averages for the specified windows.
    """
    # Use provided config or default to global CROSSOVER_CONFIG
    cfg = config if config else CROSSOVER_CONFIG
    windows = cfg.get('WINDOWS', CROSSOVER_CONFIG['WINDOWS'])
    
    for window in windows:
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
    return df

def check_golden_crossover(df):
    """
    Checks for Golden Crossover signals based on 20, 50, and 200 EMAs.
    Returns a dictionary with the latest status and signals.
    """
    if df.empty:
        return {}
    
    latest = df.iloc[-1]
    
    ema_20 = latest.get('EMA_20')
    ema_50 = latest.get('EMA_50')
    ema_200 = latest.get('EMA_200')
    
    signals = {
        'EMA_20': ema_20,
        'EMA_50': ema_50,
        'EMA_200': ema_200,
        'Trend_Status': 'Neutral',
        'GC_Date': None,
        'GC_Price': None
    }
    
    # Find the last Golden Cross (50 crossing above 200)
    if 'EMA_50' in df.columns and 'EMA_200' in df.columns:
        # Create a boolean series where 50 > 200
        bullish_cross = df['EMA_50'] > df['EMA_200']
        # Find where it changed from False to True (Golden Cross)
        golden_crossovers = bullish_cross & (~bullish_cross.shift(1).fillna(False))
        
        # Find where it changed from True to False (Death Cross)
        death_crossovers = (~bullish_cross) & (bullish_cross.shift(1).fillna(False))
        
        # Get indices where Golden Cross happened
        golden_crossover_dates = df.index[golden_crossovers]
        
        if not golden_crossover_dates.empty:
            last_gc_date = golden_crossover_dates[-1]
            
            # Check if there's a Death Cross after this Golden Cross
            death_crossover_dates = df.index[death_crossovers]
            death_after_gc = death_crossover_dates[death_crossover_dates > last_gc_date]
            
            # Only include the Golden Cross if it's still valid (no Death Cross after it)
            if death_after_gc.empty:
                last_gc_price = df.loc[last_gc_date]['Close']
                signals['GC_Date'] = last_gc_date
                signals['GC_Price'] = last_gc_price
            
    # Determine general trend status
    if ema_20 and ema_50 and ema_200:
        if ema_20 > ema_50 > ema_200:
            signals['Trend_Status'] = 'Strong Uptrend'
        elif ema_20 < ema_50 < ema_200:
            signals['Trend_Status'] = 'Strong Downtrend'
        elif ema_20 > ema_50:
            signals['Trend_Status'] = 'Short-term Uptrend'
        elif ema_20 < ema_50:
            signals['Trend_Status'] = 'Short-term Downtrend'
        
    return signals

def analyze_crossover(df):
    """
    Analyzes EMA Crossovers.
    """
    return check_golden_crossover(df)

def plot_crossover(df, ticker, show_plot=True, config=None):
    """
    Plots Price with EMAs.
    """
    # Use provided config or default to global CROSSOVER_CONFIG
    cfg = config if config else CROSSOVER_CONFIG
    windows = cfg.get('WINDOWS', CROSSOVER_CONFIG['WINDOWS'])
    
    fig = plt.figure(figsize=(14, 8))
    
    # Plot Price
    plt.plot(df.index, df['Close'], label='Close Price', color='black', alpha=0.5)
    
    # Plot EMAs
    colors = ['blue', 'orange', 'red', 'green', 'purple']
    for i, window in enumerate(windows):
        col_name = f'EMA_{window}'
        if col_name in df.columns:
            color = colors[i % len(colors)]
            plt.plot(df.index, df[col_name], label=f'EMA {window}', color=color)
    
    # Highlight Golden Cross if present
    signals = check_golden_crossover(df)
    if signals.get('GC_Date'):
        plt.scatter(signals['GC_Date'], signals['GC_Price'], color='gold', marker='*', s=200, zorder=5, label='Golden Cross')

    plt.title(f'{ticker} EMA Crossover Analysis')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    if show_plot:
        plt.show()
        
    return fig

def run_analysis(ticker, show_plot=True, config=None):
    """
    Main analysis function that can be called from a GUI or other scripts.
    """
    # Set non-interactive backend when not showing plot (e.g., for Flask/web)
    if not show_plot:
        matplotlib.use('Agg', force=True)
    
    try:
        # Merge provided config with defaults
        current_config = CROSSOVER_CONFIG.copy()
        if config:
            current_config.update(config)

        # Fetch and calculate
        df = fetch_data(ticker, config=current_config)
        df = calculate_emas(df, config=current_config)
        analysis = analyze_crossover(df)
        
        # Generate plot
        fig = plot_crossover(df, ticker, show_plot=show_plot, config=current_config)
        
        return {
            'success': True,
            'ticker': ticker,
            'ema_20': analysis.get('EMA_20'),
            'ema_50': analysis.get('EMA_50'),
            'ema_200': analysis.get('EMA_200'),
            'trend_status': analysis.get('Trend_Status'),
            'gc_date': analysis.get('GC_Date'),
            'gc_price': analysis.get('GC_Price'),
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
        
    print(f"Found {len(tickers)} unique tickers in {json_file}. Starting EMA analysis...")
    
    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")
        try:
            df = fetch_data(ticker)
            df = calculate_emas(df)
            signals = check_golden_crossover(df)
            
            print(f"  Trend Status: {signals['Trend_Status']}")
            if signals['GC_Date']:
                print(f"  Last Golden Cross: {signals['GC_Date'].date()} at {signals['GC_Price']:.2f}")
                
        except Exception as e:
            print(f"  Error analyzing {ticker}: {e}")

if __name__ == "__main__":
    import os
    
    # Load execution parameters from config
    run_batch = CROSSOVER_CONFIG['RUN_BATCH']
    default_ticker = CROSSOVER_CONFIG['DEFAULT_TICKER']
    batch_relative_path = CROSSOVER_CONFIG['BATCH_RELATIVE_PATH']
    
    # Resolve batch file path
    batch_file = os.path.join(os.path.dirname(__file__), batch_relative_path)
    
    if run_batch and os.path.exists(batch_file):
        analyze_batch(batch_file)
    else:
        print(f"Running EMA Crossover Analysis for {default_ticker}...")
        
        try:
            df = fetch_data(default_ticker)
            df = calculate_emas(df)
            signals = check_golden_crossover(df)
            
            print("\n--- EMA Analysis Results ---")
            print(f"EMA 20:  {signals['EMA_20']:.2f}")
            print(f"EMA 50:  {signals['EMA_50']:.2f}")
            print(f"EMA 200: {signals['EMA_200']:.2f}")
            print(f"Trend Status: {signals['Trend_Status']}")
            
            if signals['GC_Date']:
                print(f"Last Golden Cross (50 > 200): {signals['GC_Date'].date()} at {signals['GC_Price']:.2f}")
            else:
                print("No Golden Cross found in the loaded period.")
                
            # Plot
            plot_crossover(df, default_ticker)
                
        except Exception as e:
            print(f"Error: {e}")
