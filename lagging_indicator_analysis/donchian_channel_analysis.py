"""
DONCHIAN CHANNELS ANALYSIS TOOL
================================

PURPOSE:
--------
This module implements Donchian Channels analysis, a breakout and trend-following system
developed by Richard Donchian (father of trend following). Donchian Channels identify:
- Strong breakout points (new highs/lows)
- Dynamic support and resistance levels
- Trend direction and strength
- Volatility cycles (channel width changes)

WHAT IT DOES:
-------------
1. **Donchian Channel Calculation**:
   - Upper Channel = Highest High over N periods (default: 20)
   - Lower Channel = Lowest Low over N periods (default: 20)
   - Middle Channel = (Upper Channel + Lower Channel) / 2
   - Uses rolling window to track price extremes

2. **Breakout Detection**:
   - **Bullish Breakout**: Close price exceeds previous day's upper channel
   - **Bearish Breakout**: Close price falls below previous day's lower channel
   - Signals new momentum and potential trend continuation
   - Tracks last 60 days of breakout signals

3. **Price Position Analysis**:
   - **Bullish**: Active bullish breakout occurring
   - **Bearish**: Active bearish breakout occurring
   - **Bullish Bias**: Price above middle channel (upper half)
   - **Bearish Bias**: Price below middle channel (lower half)

4. **Signal Tracking**:
   - Records date, price, and channel range for each breakout
   - Color-codes signals (bullish=green ↑, bearish=red ↓)
   - Provides historical context for pattern analysis

5. **Visual Analysis**: Generates charts showing:
   - Price line (blue)
   - Upper and Lower channels (green/red dashed lines)
   - Middle channel (gray dotted line)
   - Gray-shaded channel area
   - Breakout markers with color coding

METHODOLOGY:
------------
Donchian Channel Theory:
- Developed by Richard Donchian in the 1960s for commodity trading
- Forms dynamic support (lower band) and resistance (upper band)
- Breakouts signal potential trend changes or continuations
- Channel width indicates volatility:
  - Narrow channels = low volatility, breakout likely
  - Wide channels = high volatility, consolidation likely

Breakout Signal Logic:
- Uses PREVIOUS day's channel to avoid look-ahead bias
- Close > Previous Upper Channel = Bullish breakout (new high)
- Close < Previous Lower Channel = Bearish breakout (new low)
- Classic trend-following: buy breakouts up, sell breakouts down

Channel Interpretation:
- Price near upper channel = strong upward pressure
- Price near lower channel = strong downward pressure
- Price at middle channel = neutral/equilibrium
- Breakout failures (quick reversals) indicate false signals

KEY METRICS:
------------
- DC_Upper: Current upper channel value (resistance)
- DC_Lower: Current lower channel value (support)
- DC_Middle: Current middle channel value (equilibrium)
- Last_Price: Current closing price
- Status: Position classification (Bullish/Bearish/Bias)
- Breakout_Signal: Current active breakout (if any)
- Signals: Array of recent breakout events with details

CONFIGURATION:
--------------
Default parameters (customizable via DONCHIAN_CONFIG):
- WINDOW: 20 periods (Donchian channel lookback)
- INTERVAL: '1d' (daily candles, also supports '1wk', '1mo', '1h', '15m')
- LOOKBACK_PERIODS: 730 days (2 years of history)

Parameter Effects:
- **Shorter Window** (10-15): More sensitive, frequent breakouts, more whipsaws
- **Standard Window** (20): Balanced, classic Turtle Traders used 20/55 system
- **Longer Window** (40-55): Fewer signals, stronger trends, delayed entries

USAGE:
------
Run as standalone script:
    python donchian_channel_analysis.py

Or import and use programmatically:
    from donchian_channel_analysis import run_analysis
    results = run_analysis(ticker="SBIN.NS", config={'WINDOW': 20})

OUTPUT:
-------
Returns dictionary containing:
- Current channel values (upper, middle, lower)
- Last price
- Status classification
- Active breakout signal (if any)
- Recent breakout signals array with full details
- Matplotlib figure with channel visualization
- Full DataFrame with calculated channels

TYPICAL USE CASES:
------------------
1. Breakout trading: Enter on channel breakouts, exit on opposite breakout
2. Trend identification: Price position relative to channels shows trend
3. Support/resistance: Channels act as dynamic S/R zones
4. Volatility analysis: Channel width expansion/contraction cycles
5. Turtle Trading System: Classic 20/55 Donchian dual-timeframe strategy
6. Mean reversion: Trade bounces off extremes in ranging markets
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configuration
DONCHIAN_CONFIG = {
    # Donchian Channel Parameters
    'WINDOW': 20,
    
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
    # Use provided config or default to global DONCHIAN_CONFIG
    cfg = config if config else DONCHIAN_CONFIG
    
    interval = cfg.get('INTERVAL', DONCHIAN_CONFIG['INTERVAL'])
    lookback_periods = cfg.get('LOOKBACK_PERIODS', DONCHIAN_CONFIG['LOOKBACK_PERIODS'])
    
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

def calculate_donchian_channels(df, config=None):
    """
    Calculates Donchian Channels using config.
    Upper Channel = Max High of last N periods
    Lower Channel = Min Low of last N periods
    Middle Channel = (Upper + Lower) / 2
    """
    # Use provided config or default to global DONCHIAN_CONFIG
    cfg = config if config else DONCHIAN_CONFIG
    window = cfg.get('WINDOW', DONCHIAN_CONFIG['WINDOW'])
    
    # Note: The channel for 'today' is usually based on the *previous* N days to avoid look-ahead bias in backtesting,
    # but for live visualization, we often want to see the current levels.
    # Standard formula: Max(High, N)
    
    df['DC_Upper'] = df['High'].rolling(window=window).max()
    df['DC_Lower'] = df['Low'].rolling(window=window).min()
    df['DC_Middle'] = (df['DC_Upper'] + df['DC_Lower']) / 2
    
    return df

def detect_breakouts(df):
    """
    Detects Breakout signals.
    Buy: Close > Previous DC Upper
    Sell: Close < Previous DC Lower
    """
    signals = []
    
    df['Prev_DC_Upper'] = df['DC_Upper'].shift(1)
    df['Prev_DC_Lower'] = df['DC_Lower'].shift(1)
    
    # We scan the last 60 days
    subset = df.iloc[-60:].copy()
    
    for date, row in subset.iterrows():
        close = row['Close']
        prev_upper = row['Prev_DC_Upper']
        prev_lower = row['Prev_DC_Lower']
        
        if pd.isna(prev_upper) or pd.isna(prev_lower):
            continue
            
        signal_type = None
        
        if close > prev_upper:
            signal_type = "Bullish Breakout"
        elif close < prev_lower:
            signal_type = "Bearish Breakout"
            
        if signal_type:
            signals.append({
                'Date': date,
                'Type': signal_type,
                'Price': close,
                'Upper': prev_upper,
                'Lower': prev_lower
            })
            
    return signals

def analyze_donchian(df):
    """
    Analyzes current status relative to channels.
    """
    latest = df.iloc[-1]
    
    results = {
        'DC_Upper': latest['DC_Upper'],
        'DC_Lower': latest['DC_Lower'],
        'DC_Middle': latest['DC_Middle'],
        'Close': latest['Close'],
        'Status': 'Neutral',
        'Breakout_Signal': None,
        'Signals': []
    }
    
    # Check for current breakout
    prev = df.iloc[-2]
    if latest['Close'] > prev['DC_Upper']:
        results['Status'] = 'Bullish'
        results['Breakout_Signal'] = 'Bullish Breakout (Buy)'
    elif latest['Close'] < prev['DC_Lower']:
        results['Status'] = 'Bearish'
        results['Breakout_Signal'] = 'Bearish Breakout (Sell)'
    elif latest['Close'] > latest['DC_Middle']:
        results['Status'] = 'Bullish Bias'
    else:
        results['Status'] = 'Bearish Bias'
        
    # Get historical signals
    results['Signals'] = detect_breakouts(df)
    
    return results

def plot_donchian_channels(df, ticker, show_plot=True, config=None):
    """
    Plots Price with Donchian Channels and signals.
    """
    # Use provided config or default to global DONCHIAN_CONFIG
    cfg = config if config else DONCHIAN_CONFIG
    window = cfg.get('WINDOW', DONCHIAN_CONFIG['WINDOW'])
    
    signals = detect_breakouts(df)
    
    fig = plt.figure(figsize=(14, 8))
    
    # Plot Price
    plt.plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.6)
    
    # Plot Channels
    plt.plot(df.index, df['DC_Upper'], label=f'Upper Channel ({window})', color='green', linestyle='--', alpha=0.6)
    plt.plot(df.index, df['DC_Lower'], label=f'Lower Channel ({window})', color='red', linestyle='--', alpha=0.6)
    plt.plot(df.index, df['DC_Middle'], label='Middle Channel', color='gray', linestyle=':', alpha=0.4)
    
    # Fill the channel
    plt.fill_between(df.index, df['DC_Upper'], df['DC_Lower'], color='gray', alpha=0.1)
    
    # Plot Signals
    for sig in signals:
        color = 'green' if 'Bullish' in sig['Type'] else 'red'
        marker = '^' if 'Bullish' in sig['Type'] else 'v'
        plt.scatter(sig['Date'], sig['Price'], color=color, marker=marker, s=100, zorder=5, 
                    label=sig['Type'] if sig['Type'] not in [l.get_label() for l in plt.gca().get_lines()] else "")

    plt.title(f'{ticker} Donchian Channels ({window}) Breakout Analysis')
    plt.xlabel('Date')
    plt.ylabel('Price')
    
    # Remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')
    
    plt.grid(True)
    plt.tight_layout()
    
    if show_plot:
        plt.show()
        
    return fig

def run_analysis(ticker, show_plot=True, config=None, df=None):
    """
    Main analysis function that can be called from a GUI or other scripts.
    """
    # Set non-interactive backend when not showing plot (e.g., for Flask/web)
    if not show_plot:
        matplotlib.use('Agg', force=True)
    
    try:
        # Merge provided config with defaults
        current_config = DONCHIAN_CONFIG.copy()
        if config:
            current_config.update(config)

        # Fetch and calculate
        if df is None:
            df = fetch_data(ticker, config=current_config)
            
        df = calculate_donchian_channels(df, config=current_config)
        analysis = analyze_donchian(df)
        
        # Generate plot
        fig = plot_donchian_channels(df, ticker, show_plot=show_plot, config=current_config)
        
        return {
            'success': True,
            'ticker': ticker,
            'dc_upper': analysis['DC_Upper'],
            'dc_lower': analysis['DC_Lower'],
            'dc_middle': analysis['DC_Middle'],
            'last_price': analysis['Close'],
            'status': analysis['Status'],
            'breakout_signal': analysis['Breakout_Signal'],
            'signals': analysis['Signals'],
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
        
    print(f"Found {len(tickers)} unique tickers in {json_file}. Starting Donchian Channel analysis...")
    
    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")
        try:
            df = fetch_data(ticker)
            df = calculate_donchian_channels(df)
            analysis = analyze_donchian(df)
            
            print(f"  Status: {analysis['Status']}")
            if analysis['Breakout_Signal']:
                print(f"  Signal: {analysis['Breakout_Signal']}")
            
            if analysis['Signals']:
                last_sig = analysis['Signals'][-1]
                print(f"  Latest Breakout: {last_sig['Type']} on {last_sig['Date'].date()}")
                
        except Exception as e:
            print(f"  Error analyzing {ticker}: {e}")

if __name__ == "__main__":
    import os
    
    # Load execution parameters from config
    run_batch = DONCHIAN_CONFIG['RUN_BATCH']
    default_ticker = DONCHIAN_CONFIG['DEFAULT_TICKER']
    batch_relative_path = DONCHIAN_CONFIG['BATCH_RELATIVE_PATH']
    
    # Resolve batch file path
    batch_file = os.path.join(os.path.dirname(__file__), batch_relative_path)
    
    if run_batch and os.path.exists(batch_file):
        analyze_batch(batch_file)
    else:
        print(f"Running Donchian Channel Analysis for {default_ticker}...")
        
        try:
            df = fetch_data(default_ticker)
            df = calculate_donchian_channels(df)
            analysis = analyze_donchian(df)
            
            print("\n--- Donchian Channel Analysis Results ---")
            print(f"Upper Channel:  {analysis['DC_Upper']:.2f}")
            print(f"Lower Channel:  {analysis['DC_Lower']:.2f}")
            print(f"Middle Channel: {analysis['DC_Middle']:.2f}")
            print(f"Last Price:     {analysis['Close']:.2f}")
            print(f"Status:         {analysis['Status']}")
            
            if analysis['Breakout_Signal']:
                print(f"Signal:         {analysis['Breakout_Signal']}")
                
            if analysis['Signals']:
                print("\n--- Breakout Signals (Last 60 Days) ---")
                for sig in analysis['Signals']:
                    print(f"{sig['Date'].date()}: {sig['Type']} at {sig['Price']:.2f} (Channel: {sig['Lower']:.2f} - {sig['Upper']:.2f})")
            else:
                print("\nNo recent breakouts detected.")
                
            # Plot
            plot_donchian_channels(df, default_ticker)
                
        except Exception as e:
            print(f"Error: {e}")
