"""
SUPERTREND ANALYSIS TOOL
========================

PURPOSE:
--------
This module implements Supertrend analysis, a powerful trend-following indicator that combines
price action with volatility (ATR - Average True Range). Supertrend excels at:
- Identifying clear trend direction (uptrend/downtrend)
- Generating objective buy/sell signals at trend changes
- Providing dynamic support/resistance levels
- Filtering out market noise through ATR-based bands

WHAT IT DOES:
-------------
1. **ATR Calculation**: Measures market volatility
   - True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
   - ATR = Moving average of True Range over specified period

2. **Supertrend Bands Calculation**:
   - Basic Upper Band = (High + Low)/2 + (Multiplier × ATR)
   - Basic Lower Band = (High + Low)/2 - (Multiplier × ATR)
   - Final bands calculated recursively based on price action

3. **Trend Detection**:
   - Uptrend (1): Price > Supertrend line (Green zone)
   - Downtrend (-1): Price < Supertrend line (Red zone)
   - Provides clear, unambiguous trend signals

4. **Signal Generation**:
   - Buy Signal: Trend changes from -1 to 1 (price crosses above Supertrend)
   - Sell Signal: Trend changes from 1 to -1 (price crosses below Supertrend)

5. **Visual Analysis**: Generates charts with:
   - Price line with color-coded zones (green=bullish, red=bearish)
   - Supertrend line acting as dynamic support/resistance
   - Clear buy (↑) and sell (↓) markers at crossover points

METHODOLOGY:
------------
Supertrend Algorithm:
1. Calculate ATR using specified period (default: 14)
2. Compute basic bands using HL/2 ± (Multiplier × ATR)
3. Apply recursive logic for final bands:
   - Upper band: Maintain previous unless price closes above it
   - Lower band: Maintain previous unless price closes below it
4. Determine Supertrend value:
   - Follow upper band in downtrend
   - Follow lower band in uptrend
   - Switch when price crosses the band

Parameter Effects:
- **Period (ATR)**: Lower = more sensitive, Higher = smoother
  - 10: Aggressive, more signals, more whipsaws
  - 14: Balanced (recommended default)
  - 20: Conservative, fewer signals, misses early entries

- **Multiplier**: Lower = tighter bands, Higher = wider bands
  - 2.0: Tight bands, frequent signals
  - 3.0: Balanced (recommended default)
  - 4.0+: Wide bands, fewer false signals but delayed entries

KEY METRICS:
------------
- Trend: 1 (Uptrend/Buy) or -1 (Downtrend/Sell)
- Supertrend Value: Current band level (support/resistance)
- Last Price: Latest closing price
- Status: "UPTREND (Buy)" or "DOWNTREND (Sell)"
- ATR: Current volatility measure

CONFIGURATION:
--------------
Default parameters (customizable via SUPERTREND_CONFIG):
- PERIOD: 14 (ATR calculation period)
- MULTIPLIER: 3.0 (ATR multiplier for bands)
- INTERVAL: '1d' (daily candles, also supports '1wk', '1h', '15m')
- LOOKBACK_PERIODS: 730 days (2 years of history)

USAGE:
------
Run as standalone script:
    python supertrend_analysis.py

Or import and use programmatically:
    from supertrend_analysis import run_analysis
    results = run_analysis(ticker="RELIANCE.NS", config={'PERIOD': 10, 'MULTIPLIER': 2.5})

OUTPUT:
-------
Returns dictionary containing:
- Current trend direction (1 or -1)
- Supertrend value (support/resistance level)
- Last price and date
- Status string for display
- Matplotlib figure with annotated chart
- Full DataFrame with ATR, bands, and trend data

TYPICAL USE CASES:
------------------
1. Trend following: Stay long in uptrends, short in downtrends
2. Stop loss placement: Use Supertrend line as trailing stop
3. Entry confirmation: Enter on trend change signals
4. Multi-timeframe analysis: Combine daily + weekly for stronger signals
5. Portfolio filtering: Only consider stocks in uptrend for long positions
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime, timedelta

# Configuration
SUPERTREND_CONFIG = {
    # ATR Period: Number of periods for Average True Range calculation. Controls sensitivity to price changes.
    # Best: 10 (default, balanced), 7 (more signals, aggressive), 14 (fewer signals, conservative)
    'PERIOD': 14,
    
    # ATR Multiplier: Factor to multiply ATR for band width. Higher values reduce false signals but miss early entries.
    # Best: 3.0 (default, balanced), 2.0 (tighter bands, more signals), 4.0 (wider bands, fewer signals)
    'MULTIPLIER': 3.0,
    
    # Data Interval: Time interval for candles. Affects trend detection timeframe.
    # Best: '1d' (daily swing trading), '1wk' (weekly position trading), '15m' or '1h' (intraday)
    'INTERVAL': '1d',
    
    # Lookback Periods: Number of days of historical data to fetch for analysis.
    # Best: 365 (1 year for trends), 180 (6 months for recent analysis), 730 (2 years for long-term)
    'LOOKBACK_PERIODS': 730,  # 2 year
    
    'DEFAULT_TICKER': 'LT.NS',
    'BATCH_RELATIVE_PATH': '../data/tickers_list.json',
    'RUN_BATCH': False
}

def fetch_data(ticker, interval, lookback_periods):
    """
    Fetches historical data for the given ticker.
    """
    # Limit lookback for 15m interval (Yahoo Finance restriction: max 60 days)
    if interval == '15m':
        lookback_periods = min(lookback_periods, 59)
    
    end_date = datetime.now()
    # Assuming lookback_periods represents days for now, as yfinance works with dates
    # If interval is intraday, we might need to adjust logic, but keeping it simple as per request
    start_date = end_date - timedelta(days=lookback_periods)
    
    print(f"Fetching data for {ticker} from {start_date.date()} to {end_date.date()}...")
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=False, multi_level_index=False)
    
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
    
    # Handle MultiIndex columns if they exist (common in newer yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    return df

def calculate_atr(df, period):
    """
    Calculates the Average True Range (ATR).
    """
    high = df['High']
    low = df['Low']
    close = df['Close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

def calculate_supertrend(df, config=None):
    """
    Calculates the Supertrend indicator using config.
    """
    # Use provided config or default to global SUPERTREND_CONFIG
    cfg = config if config else SUPERTREND_CONFIG
    period = cfg.get('PERIOD', SUPERTREND_CONFIG['PERIOD'])
    multiplier = cfg.get('MULTIPLIER', SUPERTREND_CONFIG['MULTIPLIER'])
    
    # Ensure we have ATR
    df['ATR'] = calculate_atr(df, period)
    
    # Basic Upper and Lower Bands
    hl2 = (df['High'] + df['Low']) / 2
    df['Basic_Upper'] = hl2 + (multiplier * df['ATR'])
    df['Basic_Lower'] = hl2 - (multiplier * df['ATR'])
    
    # Final Upper and Lower Bands
    df['Final_Upper'] = 0.0
    df['Final_Lower'] = 0.0
    df['Supertrend'] = 0.0
    
    # Loop through to calculate bands and trend
    # Note: Iterating is slower but easier to implement for the recursive logic of Supertrend
    for i in range(period, len(df)):
        # Final Upper Band
        if df['Basic_Upper'].iloc[i] < df['Final_Upper'].iloc[i-1] or \
           df['Close'].iloc[i-1] > df['Final_Upper'].iloc[i-1]:
            df.iloc[i, df.columns.get_loc('Final_Upper')] = df['Basic_Upper'].iloc[i]
        else:
            df.iloc[i, df.columns.get_loc('Final_Upper')] = df['Final_Upper'].iloc[i-1]
            
        # Final Lower Band
        if df['Basic_Lower'].iloc[i] > df['Final_Lower'].iloc[i-1] or \
           df['Close'].iloc[i-1] < df['Final_Lower'].iloc[i-1]:
            df.iloc[i, df.columns.get_loc('Final_Lower')] = df['Basic_Lower'].iloc[i]
        else:
            df.iloc[i, df.columns.get_loc('Final_Lower')] = df['Final_Lower'].iloc[i-1]
            
        # Supertrend
        if df['Supertrend'].iloc[i-1] == df['Final_Upper'].iloc[i-1]:
            if df['Close'].iloc[i] <= df['Final_Upper'].iloc[i]:
                df.iloc[i, df.columns.get_loc('Supertrend')] = df['Final_Upper'].iloc[i]
            else:
                df.iloc[i, df.columns.get_loc('Supertrend')] = df['Final_Lower'].iloc[i]
        elif df['Supertrend'].iloc[i-1] == df['Final_Lower'].iloc[i-1]:
            if df['Close'].iloc[i] >= df['Final_Lower'].iloc[i]:
                df.iloc[i, df.columns.get_loc('Supertrend')] = df['Final_Lower'].iloc[i]
            else:
                df.iloc[i, df.columns.get_loc('Supertrend')] = df['Final_Upper'].iloc[i]
        else:
             # Initialize
            if df['Close'].iloc[i] <= df['Final_Upper'].iloc[i]:
                df.iloc[i, df.columns.get_loc('Supertrend')] = df['Final_Upper'].iloc[i]
            else:
                df.iloc[i, df.columns.get_loc('Supertrend')] = df['Final_Lower'].iloc[i]
                
    # Identify Trend direction
    df['Trend'] = np.where(df['Close'] > df['Supertrend'], 1, -1)
    
    return df

def plot_supertrend(df, ticker, show_plot=True, config=None):
    """
    Plots the Close price and Supertrend with Buy/Sell signals.
    
    Args:
        df: DataFrame with Supertrend data
        ticker: Stock ticker symbol
        show_plot: If True, displays plot. If False, just returns the figure.
        config: Configuration dictionary
    
    Returns:
        matplotlib.figure.Figure: The plot figure object
    """
    # Use provided config or default to global SUPERTREND_CONFIG
    cfg = config if config else SUPERTREND_CONFIG
    period = cfg.get('PERIOD', SUPERTREND_CONFIG['PERIOD'])
    multiplier = cfg.get('MULTIPLIER', SUPERTREND_CONFIG['MULTIPLIER'])
    interval = cfg.get('INTERVAL', SUPERTREND_CONFIG['INTERVAL'])
    
    fig = plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.5)
    plt.plot(df.index, df['Supertrend'], label='Supertrend', color='purple', linewidth=2)
    
    # Color the area between price and supertrend
    plt.fill_between(df.index, df['Close'], df['Supertrend'], where=df['Close'] > df['Supertrend'], color='green', alpha=0.1)
    plt.fill_between(df.index, df['Close'], df['Supertrend'], where=df['Close'] < df['Supertrend'], color='red', alpha=0.1)
    
    # Buy/Sell Signals
    # Buy when trend changes from -1 to 1
    buy_signals = df[(df['Trend'] == 1) & (df['Trend'].shift(1) == -1)]
    # Sell when trend changes from 1 to -1
    sell_signals = df[(df['Trend'] == -1) & (df['Trend'].shift(1) == 1)]
    
    plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', s=100, label='Buy Signal', zorder=5)
    plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', s=100, label='Sell Signal', zorder=5)
    
    # Print latest signal
    last_trend = df['Trend'].iloc[-1]
    last_signal_date = df.index[-1]
    if last_trend == 1:
        print(f"Current Status: UPTREND (Buy) as of {last_signal_date}")
    else:
        print(f"Current Status: DOWNTREND (Sell) as of {last_signal_date}")

    plt.title(f'Supertrend Analysis for {ticker} (Period: {period}, Multiplier: {multiplier}, Interval: {interval})')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    
    return fig

    return fig

def run_analysis(ticker=None, show_plot=True, config=None):
    """
    Main analysis function that can be called from a GUI or other scripts.
    
    Args:
        ticker: Stock ticker symbol (if None, uses DEFAULT_TICKER from config)
        show_plot: If True, displays the plot. If False, just returns the figure.
        config: Optional dictionary to override default configuration
    
    Returns:
        dict: Analysis results containing:
            - 'success': bool, whether analysis succeeded
            - 'ticker': str, analyzed ticker
            - 'last_trend': int, 1 for uptrend, -1 for downtrend
            - 'last_price': float, latest closing price
            - 'last_date': datetime, date of latest data
            - 'supertrend_value': float, latest supertrend value
            - 'status': str, "UPTREND (Buy)" or "DOWNTREND (Sell)"
            - 'figure': matplotlib.figure.Figure, plot figure object
            - 'dataframe': pd.DataFrame, full analysis data (optional)
            - 'error': str, error message if analysis failed
    """
    # Merge provided config with defaults
    current_config = SUPERTREND_CONFIG.copy()
    if config:
        current_config.update(config)

    if ticker is None:
        ticker = current_config['DEFAULT_TICKER']
    
    try:
        # Fetch and calculate
        df = fetch_data(ticker, 
                       interval=current_config['INTERVAL'], 
                       lookback_periods=current_config['LOOKBACK_PERIODS'])
        df = calculate_supertrend(df, config=current_config)
        
        # Extract results
        last_trend = int(df['Trend'].iloc[-1])
        last_price = float(df['Close'].iloc[-1])
        last_date = df.index[-1]
        supertrend_value = float(df['Supertrend'].iloc[-1])
        status = "UPTREND (Buy)" if last_trend == 1 else "DOWNTREND (Sell)"
        
        # Generate plot
        fig = plot_supertrend(df, ticker, show_plot=show_plot, config=current_config)
        
        return {
            'success': True,
            'ticker': ticker,
            'last_trend': last_trend,
            'last_price': last_price,
            'last_date': last_date,
            'supertrend_value': supertrend_value,
            'status': status,
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
        
    print(f"Found {len(tickers)} unique tickers in {json_file}. Starting Supertrend analysis...")
    
    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")
        try:
            # Use defaults for batch
            df = fetch_data(ticker, interval=SUPERTREND_CONFIG['INTERVAL'], lookback_periods=SUPERTREND_CONFIG['LOOKBACK_PERIODS'])
            df = calculate_supertrend(df)
            
            # Extract latest signal
            last_trend = df['Trend'].iloc[-1]
            last_signal_date = df.index[-1]
            status = "UPTREND (Buy)" if last_trend == 1 else "DOWNTREND (Sell)"
            print(f"  Current Status: {status} as of {last_signal_date.date()}")
                
        except Exception as e:
            print(f"  Error analyzing {ticker}: {e}")

if __name__ == "__main__":
    import os
    
    # Load execution parameters from config
    run_batch = SUPERTREND_CONFIG['RUN_BATCH']
    batch_relative_path = SUPERTREND_CONFIG['BATCH_RELATIVE_PATH']
    
    # Resolve batch file path
    batch_file = os.path.join(os.path.dirname(__file__), batch_relative_path)
    
    if run_batch and os.path.exists(batch_file):
        analyze_batch(batch_file)
    else:
        # Run single stock analysis
        results = run_analysis(show_plot=True)
        
        if results['success']:
            print(f"\n--- Analysis Complete ---")
            print(f"Ticker: {results['ticker']}")
            print(f"Status: {results['status']}")
            print(f"Last Price: {results['last_price']:.2f}")
            print(f"Supertrend: {results['supertrend_value']:.2f}")
            print(f"Date: {results['last_date'].date()}")
        else:
            print(f"Error: {results['error']}")
