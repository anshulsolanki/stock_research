"""
BOLLINGER BANDS ANALYSIS TOOL
==============================

PURPOSE:
--------
This module performs comprehensive Bollinger Bands analysis with RSI and candlestick pattern
confirmation. Bollinger Bands are volatility-based bands that help identify:
- Overbought/oversold conditions
- Price volatility changes (squeeze/expansion)
- Potential reversal points
- High-probability entry/exit signals when combined with RSI and patterns

WHAT IT DOES:
-------------
1. **Bollinger Bands Calculation**:
   - Middle Band = 20-period Simple Moving Average (SMA)
   - Upper Band = Middle Band + (2 × Standard Deviation)
   - Lower Band = Middle Band - (2 × Standard Deviation)
   - %B = Position within bands: (Price - Lower) / (Upper - Lower)
   - BandWidth = (Upper - Lower) / Middle (volatility measure)

2. **RSI Integration** (14-period):
   - Identifies overbought (>70) and oversold (<30) conditions
   - Used as confirmation filter for signals

3. **Candlestick Pattern Detection**:
   - Hammer: Bullish reversal (long lower wick, small body)
   - Shooting Star: Bearish reversal (long upper wick, small body)
   - Bullish Engulfing: Strong bullish reversal
   - Bearish Engulfing: Strong bearish reversal

4. **Multi-Factor Signal Generation**:
   **BUY SIGNAL** requires:
   - Low ≤ Lower Band (price touches/pierces lower band)
   - AND (RSI < 30 OR Bullish Candlestick Pattern)
   
   **SELL SIGNAL** requires:
   - High ≥ Upper Band (price touches/pierces upper band)
   - AND (RSI > 70 AND Bearish Candlestick Pattern)

5. **Status Classification**:
   - Above Upper Band: Strong momentum, potential overbought
   - Below Lower Band: Weak momentum, potential oversold
   - Upper Half: Bullish bias
   - Lower Half: Bearish bias

6. **Visual Analysis**: Generates charts showing:
   - Price with gray-shaded band area
   - Buy signals (green ↑) at lower band
   - Sell signals (red ↓) at upper band

METHODOLOGY:
------------
Bollinger Bands Theory:
- Bands expand during high volatility, contract during low volatility
- ~95% of price action occurs within 2 standard deviations
- "Walking the bands" indicates strong trends
- Band squeezes often precede significant moves

Signal Logic:
- Lower band + oversold/bullish pattern = potential bounce
- Upper band + overbought/bearish pattern = potential reversal
- Combines mean reversion with momentum confirmation

%B Interpretation:
- %B > 1: Price above upper band
- %B = 0.5: Price at middle band
- %B < 0: Price below lower band

BandWidth Interpretation:
- Low BandWidth: Low volatility (squeeze), breakout likely
- High BandWidth: High volatility, potential for consolidation

KEY METRICS:
------------
- BB_Upper: Upper band value (resistance level)
- BB_Lower: Lower band value (support level)
- SMA_20: Middle band / 20-period average
- %B: Normalized position (0 to 1)
- BandWidth: Volatility measure
- RSI: 14-period Relative Strength Index
- Status: Current position relative to bands
- Signals: Recent buy/sell signals with reasons

CONFIGURATION:
--------------
Default parameters (customizable via BOLLINGER_CONFIG):
- WINDOW: 20 periods (SMA calculation)
- NUM_STD: 2 (standard deviations for bands)
- INTERVAL: '1d' (daily data, also supports '1wk', '1h', '15m')
- LOOKBACK_PERIODS: 730 days (2 years of history)

USAGE:
------
Run as standalone script:
    python bollinger_band_analysis.py

Or import and use programmatically:
    from bollinger_band_analysis import run_analysis
    results = run_analysis(ticker="TCS.NS", config={'WINDOW': 20, 'NUM_STD': 2.5})

OUTPUT:
-------
Returns dictionary containing:
- Current band values (upper, middle, lower)
- Position metrics (%B, BandWidth)
- Status classification
- Current RSI value
- Recent signals with detailed reasons (RSI + patterns)
- Matplotlib figure with annotated chart
- Full DataFrame with all indicators

TYPICAL USE CASES:
------------------
1. Mean reversion trading: Buy at lower band, sell at upper band in ranging markets
2. Breakout confirmation: Band expansion validates breakout moves
3. Squeeze plays: Identify low volatility setups before volatility expansion
4. Trend confirmation: "Walking the bands" indicates strong directional moves
5. Multi-indicator confluence: Combine with RSI + patterns for high-probability setups
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configuration
BOLLINGER_CONFIG = {
    # Bollinger Band Parameters
    'WINDOW': 20,
    'NUM_STD': 2,
    
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
    cfg = config if config else BOLLINGER_CONFIG
    
    interval = cfg.get('INTERVAL', BOLLINGER_CONFIG['INTERVAL'])
    lookback_periods = cfg.get('LOOKBACK_PERIODS', BOLLINGER_CONFIG['LOOKBACK_PERIODS'])
    
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

def calculate_bollinger_bands(df, config=None):
    """
    Calculates Bollinger Bands, RSI, and Candlestick Patterns.
    """
    # Use provided config or default to global BOLLINGER_CONFIG
    cfg = config if config else BOLLINGER_CONFIG
    window = cfg.get('WINDOW', BOLLINGER_CONFIG['WINDOW'])
    num_std = cfg.get('NUM_STD', BOLLINGER_CONFIG['NUM_STD'])
    
    # Bollinger Bands
    df['SMA_20'] = df['Close'].rolling(window=window).mean()
    df['StdDev'] = df['Close'].rolling(window=window).std()
    
    df['BB_Upper'] = df['SMA_20'] + (num_std * df['StdDev'])
    df['BB_Lower'] = df['SMA_20'] - (num_std * df['StdDev'])
    
    df['Pct_B'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    df['BandWidth'] = (df['BB_Upper'] - df['BB_Lower']) / df['SMA_20']
    
    # RSI Calculation (14 periods)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Candlestick Patterns
    # 1. Hammer / Shooting Star
    body = abs(df['Close'] - df['Open'])
    upper_wick = df['High'] - df[['Close', 'Open']].max(axis=1)
    lower_wick = df[['Close', 'Open']].min(axis=1) - df['Low']
    
    # Hammer: Small body, long lower wick (> 2x body), small upper wick
    df['Is_Hammer'] = (lower_wick > 2 * body) & (upper_wick < 0.5 * body)
    
    # Shooting Star: Small body, long upper wick (> 2x body), small lower wick
    df['Is_Shooting_Star'] = (upper_wick > 2 * body) & (lower_wick < 0.5 * body)
    
    # 2. Engulfing Patterns
    prev_body = body.shift(1)
    prev_close = df['Close'].shift(1)
    prev_open = df['Open'].shift(1)
    
    # Bullish Engulfing: Prev Red, Curr Green, Curr Body > Prev Body, Curr Open < Prev Close, Curr Close > Prev Open
    df['Is_Bullish_Engulfing'] = (prev_close < prev_open) & (df['Close'] > df['Open']) & \
                                 (df['Open'] < prev_close) & (df['Close'] > prev_open)
                                 
    # Bearish Engulfing: Prev Green, Curr Red, Curr Body > Prev Body, Curr Open > Prev Close, Curr Close < Prev Open
    df['Is_Bearish_Engulfing'] = (prev_close > prev_open) & (df['Close'] < df['Open']) & \
                                 (df['Open'] > prev_close) & (df['Close'] < prev_open)
    
    return df

def generate_signals(df):
    """
    Generates signals based on Bollinger Bands, RSI, and Candle Patterns.
    
    BUY SIGNAL:
    1. Low <= Lower Band (Touch/Pierce)
    2. AND (RSI < 30 OR Bullish Candle Pattern)
    
    SELL SIGNAL:
    1. High >= Upper Band (Touch/Pierce)
    2. AND (RSI > 70 AND Bearish Candle Pattern)
    """
    signals = []
    subset = df.iloc[-60:].copy() # Analyze last 60 days
    
    for date, row in subset.iterrows():
        close = row['Close']
        low = row['Low']
        high = row['High']
        upper = row['BB_Upper']
        lower = row['BB_Lower']
        rsi = row['RSI']
        
        if pd.isna(upper) or pd.isna(lower) or pd.isna(rsi):
            continue
            
        signal_type = None
        reason = []
        
        # Buy Logic
        # Condition 1: Touch Lower Band
        if low <= lower:
            # Condition 2: Oversold OR Bullish Pattern
            is_oversold = rsi < 30
            is_bullish_pattern = row['Is_Hammer'] or row['Is_Bullish_Engulfing']
            
            if is_oversold or is_bullish_pattern:
                signal_type = "Buy Signal"
                if is_oversold: reason.append(f"RSI Oversold ({rsi:.1f})")
                if row['Is_Hammer']: reason.append("Hammer")
                if row['Is_Bullish_Engulfing']: reason.append("Bullish Engulfing")

        # Sell Logic
        # Condition 1: Touch Upper Band
        elif high >= upper:
            # Condition 2: Overbought AND Bearish Pattern
            is_overbought = rsi > 70
            is_bearish_pattern = row['Is_Shooting_Star'] or row['Is_Bearish_Engulfing']
            
            if is_overbought and is_bearish_pattern:
                signal_type = "Sell Signal"
                reason.append(f"RSI Overbought ({rsi:.1f})")
                if row['Is_Shooting_Star']: reason.append("Shooting Star")
                if row['Is_Bearish_Engulfing']: reason.append("Bearish Engulfing")
            
        if signal_type:
            signals.append({
                'Date': date,
                'Type': signal_type,
                'Price': close,
                'Upper': upper,
                'Lower': lower,
                'RSI': rsi,
                'Reason': ", ".join(reason)
            })
            
    return signals

def analyze_bollinger(df):
    """
    Analyzes current status relative to Bollinger Bands.
    """
    latest = df.iloc[-1]
    
    results = {
        'BB_Upper': latest['BB_Upper'],
        'BB_Lower': latest['BB_Lower'],
        'SMA_20': latest['SMA_20'],
        'Close': latest['Close'],
        'Pct_B': latest['Pct_B'],
        'BandWidth': latest['BandWidth'],
        'RSI': latest['RSI'],
        'Status': 'Neutral',
        'Signal': None,
        'Signals': []
    }
    
    # Determine status based on latest close
    if latest['Close'] > latest['BB_Upper']:
        results['Status'] = 'Above Upper Band'
    elif latest['Close'] < latest['BB_Lower']:
        results['Status'] = 'Below Lower Band'
    elif latest['Pct_B'] > 0.5:
        results['Status'] = 'Upper Half'
    else:
        results['Status'] = 'Lower Half'
        
    # Get historical signals
    signals = generate_signals(df)
    results['Signals'] = signals
    
    # Check if the latest candle triggered a signal
    if signals and signals[-1]['Date'] == df.index[-1]:
        results['Signal'] = signals[-1]['Type']
    
    return results

def plot_bollinger_bands(df, ticker, show_plot=True, config=None):
    """
    Plots Price with Bollinger Bands and signals.
    """
    # Use provided config or default to global BOLLINGER_CONFIG
    cfg = config if config else BOLLINGER_CONFIG
    window = cfg.get('WINDOW', BOLLINGER_CONFIG['WINDOW'])
    num_std = cfg.get('NUM_STD', BOLLINGER_CONFIG['NUM_STD'])
    
    signals = generate_signals(df)
    
    fig = plt.figure(figsize=(14, 8))
    
    # Plot Price
    plt.plot(df.index, df['Close'], label='Close Price', color='black', alpha=0.7)
    
    # Plot Bands
    plt.plot(df.index, df['BB_Upper'], label='Upper Band', color='green', linestyle='--', alpha=0.5)
    plt.plot(df.index, df['BB_Lower'], label='Lower Band', color='red', linestyle='--', alpha=0.5)
    plt.plot(df.index, df['SMA_20'], label='Middle Band (SMA 20)', color='blue', linestyle='-', alpha=0.5)
    
    # Fill the bands
    plt.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], color='gray', alpha=0.1)
    
    # Plot Signals
    for sig in signals:
        color = 'green' if 'Buy' in sig['Type'] else 'red'
        marker = '^' if 'Buy' in sig['Type'] else 'v'
        # Only plot if it's a distinct event to avoid cluttering "walking the bands"
        plt.scatter(sig['Date'], sig['Price'], color=color, marker=marker, s=100, zorder=5)

    plt.title(f'{ticker} Bollinger Bands ({window}, {num_std}) Analysis')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='best')
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
        current_config = BOLLINGER_CONFIG.copy()
        if config:
            current_config.update(config)

        # Fetch and calculate
        if df is None:
            df = fetch_data(ticker, config=current_config)
            
        df = calculate_bollinger_bands(df, config=current_config)
        analysis = analyze_bollinger(df)
        
        # Generate plot
        fig = plot_bollinger_bands(df, ticker, show_plot=show_plot, config=current_config)
        
        return {
            'success': True,
            'ticker': ticker,
            'bb_upper': analysis['BB_Upper'],
            'bb_lower': analysis['BB_Lower'],
            'sma_20': analysis['SMA_20'],
            'last_price': analysis['Close'],
            'pct_b': analysis['Pct_B'],
            'bandwidth': analysis['BandWidth'],
            'status': analysis['Status'],
            'signal': analysis['Signal'],
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
        
    print(f"Found {len(tickers)} unique tickers in {json_file}. Starting Bollinger Band analysis...")
    
    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")
        try:
            df = fetch_data(ticker)
            df = calculate_bollinger_bands(df)
            analysis = analyze_bollinger(df)
            
            print(f"  Status: {analysis['Status']}")
            if analysis['Signal']:
                print(f"  Signal: {analysis['Signal']}")
            
            if analysis['Signals']:
                last_sig = analysis['Signals'][-1]
                print(f"  Latest Signal: {last_sig['Type']} on {last_sig['Date'].date()}")
                
        except Exception as e:
            print(f"  Error analyzing {ticker}: {e}")

if __name__ == "__main__":
    import os
    
    # Load execution parameters from config
    run_batch = BOLLINGER_CONFIG['RUN_BATCH']
    default_ticker = BOLLINGER_CONFIG['DEFAULT_TICKER']
    batch_relative_path = BOLLINGER_CONFIG['BATCH_RELATIVE_PATH']
    
    # Resolve batch file path
    batch_file = os.path.join(os.path.dirname(__file__), batch_relative_path)
    
    if run_batch and os.path.exists(batch_file):
        analyze_batch(batch_file)
    else:
        print(f"Running Bollinger Band Analysis for {default_ticker}...")
        
        try:
            df = fetch_data(default_ticker)
            df = calculate_bollinger_bands(df)
            analysis = analyze_bollinger(df)
            
            print("\n--- Bollinger Band Analysis Results ---")
            print(f"Upper Band:  {analysis['BB_Upper']:.2f}")
            print(f"Lower Band:  {analysis['BB_Lower']:.2f}")
            print(f"Middle Band: {analysis['SMA_20']:.2f}")
            print(f"Last Price:  {analysis['Close']:.2f}")
            print(f"%B:          {analysis['Pct_B']:.2f}")
            print(f"BandWidth:   {analysis['BandWidth']:.4f}")
            print(f"Status:      {analysis['Status']}")
            
            if analysis['Signal']:
                print(f"Signal:      {analysis['Signal']}")
                
            if analysis['Signals']:
                print("\n--- Recent Signals (Last 60 Days) ---")
                for sig in analysis['Signals']:
                    print(f"{sig['Date'].date()}: {sig['Type']} at {sig['Price']:.2f} (Band: {sig['Lower']:.2f} - {sig['Upper']:.2f})")
            else:
                print("\nNo recent signals detected.")
                
            # Plot
            plot_bollinger_bands(df, default_ticker)
                
        except Exception as e:
            print(f"Error: {e}")
