"""
VOLATILITY SQUEEZE ANALYSIS TOOL
=================================

PURPOSE:
--------
This module performs comprehensive Volatility Squeeze analysis, a powerful leading indicator
used in technical analysis. The Volatility Squeeze helps identify:
- Periods of low volatility that often precede significant price moves
- Potential breakout opportunities before they occur
- Market consolidation phases where institutional accumulation/distribution may occur
- High-probability trade setups based on volatility contraction

WHAT IT DOES:
-------------
1. **Bollinger Band Squeeze Detection**: Identifies extreme narrowing of price bands
   - Calculates Bollinger Bands (20-period SMA ± 2 standard deviations)
   - Measures Band Width as a normalized indicator of volatility
   - Detects when Band Width reaches multi-month lows (squeeze condition)
   
2. **ATR Contraction Detection**: Identifies periods of low true range
   - Calculates ATR (Average True Range) over 14 periods
   - Detects when ATR reaches multi-month lows
   - Confirms low volatility using absolute price movement measure

3. **Breakout Detection**: Identifies when price breaks out of squeeze with volume confirmation
   - **Bullish Breakout**: Price closes above Upper Bollinger Band AND volume surge present
     → Suggests strong upward move initiated with institutional participation
   - **Bearish Breakout**: Price closes below Lower Bollinger Band AND volume surge present
     → Suggests strong downward move initiated with selling pressure

4. **Dual Confirmation**: Combines BB and ATR for higher conviction
   - Squeeze signals are strongest when both BB and ATR confirm low volatility
   - Single indicator signals also tracked for early warning

5. **Volume Confirmation**: Validates breakouts with volume surge
   - Calculates Volume Moving Average (default: 20 periods)
   - Requires volume surge (default: 1.5x Volume MA) for breakout confirmation
   - Filters false breakouts that occur on low volume

6. **Visual Analysis**: Generates three-panel charts showing:
   - Price action with Bollinger Bands and breakout markers
   - BB Width with squeeze zones highlighted
   - ATR with contraction zones highlighted

METHODOLOGY:
------------
Bollinger Bands Formula:
- Middle Band (SMA): 20-period simple moving average of close price
- Upper Band: SMA + (2 × Standard Deviation)
- Lower Band: SMA - (2 × Standard Deviation)
- Band Width: (Upper Band - Lower Band) / SMA (normalized measure)

ATR (Average True Range) Formula:
- True Range (TR) = max(High - Low, |High - Previous Close|, |Low - Previous Close|)
- ATR = 14-period simple moving average of TR

Squeeze Detection Algorithm:
1. Calculate rolling 20th percentile of BB Width over lookback period (default: 126 days ≈ 6 months)
2. Calculate rolling 20th percentile of ATR over same lookback period
3. BB Squeeze condition: Current BB Width ≤ 20th percentile of BB Width (bottom 20% of values)
4. ATR Contraction condition: Current ATR ≤ 20th percentile of ATR (bottom 20% of values)
5. Scan last 60 days for recent squeeze conditions
6. Calculate Volume Moving Average for breakout confirmation
7. Check if price has broken out (close > Upper BB or close < Lower BB)
8. Confirm breakout with volume surge (current volume > volume_surge_multiplier × volume MA)

KEY METRICS:
------------
- BB Width: Normalized measure of Bollinger Band width (volatility proxy)
- ATR: Average True Range (absolute volatility measure)
- Squeeze Signals: Periods where volatility is extremely low
- Breakout Signals: Price movement beyond bands during/after squeeze
- Signal Types:
  * "BB Squeeze": Bollinger Bands at multi-month low width
  * "ATR Contraction": ATR at multi-month low
  * "BB Squeeze + ATR Contraction": Both confirm (highest conviction)
  * "(Bullish Breakout)": Price breaking above upper band
  * "(Bearish Breakout)": Price breaking below lower band

CONFIGURATION:
--------------
Default parameters (customizable via SQUEEZE_CONFIG or function arguments):
- BB_PERIOD: 20 (Bollinger Band SMA period)
- BB_STD: 2 (Number of standard deviations for bands)
- ATR_PERIOD: 14 (ATR calculation period)
- VOLUME_MA_PERIOD: 20 (Volume moving average period for surge detection)
- VOLUME_SURGE_MULTIPLIER: 1.5 (Multiplier for volume surge - e.g., 1.5 = 150% of average)
- LOOKBACK: 126 (Days to define "low" volatility - approximately 6 months)
- SCAN_DAYS: 60 (Recent days to scan for signals)
- PERCENTILE_THRESHOLD: 20 (Percentile threshold for squeeze detection - values below this are considered "squeeze")
- INTERVAL: '1d' (daily data; also supports '1wk', '1mo', etc.)
- LOOKBACK_PERIODS: 365 days (1 year of history)

USAGE:
------
Run as standalone script:
    python volatility_squeeze_analysis.py

Or import and use programmatically:
    from volatility_squeeze_analysis import run_analysis
    results = run_analysis(ticker="AAPL", show_plot=True, config={'LOOKBACK': 126})

For batch analysis:
    from volatility_squeeze_analysis import analyze_batch
    analyze_batch('tickers.txt')

OUTPUT:
-------
Returns dictionary containing:
- success: Boolean indicating if analysis completed successfully
- ticker: Stock ticker symbol
- current_bb_width: Current Bollinger Band Width
- current_atr: Current ATR value
- signals: List of detected squeeze/breakout signals with detailed information
- figure: Matplotlib figure object for visualization
- dataframe: Full DataFrame with all calculated indicators

TYPICAL USE CASES:
------------------
1. **Breakout Trading**: Enter positions when squeeze is followed by breakout
2. **Volatility Anticipation**: Identify low-volatility periods before big moves
3. **Range Trading**: Avoid trading during squeeze (choppy, low-movement periods)
4. **Risk Management**: Size positions smaller during squeeze, larger after breakout
5. **Multi-timeframe Analysis**: Confirm squeeze on multiple timeframes for stronger signals
6. **Market Timing**: Use for timing entries in trending stocks

INTERPRETATION GUIDE:
---------------------
**BB Squeeze + ATR Contraction (Dual Confirmation):**
- Highest conviction setup - both volatility measures in bottom 20% of range
- Suggests impending large move (direction unknown until breakout)
- Often precedes 10%+ moves within 1-3 weeks
- Wait for breakout to determine direction

**BB Squeeze Only:**
- Bollinger Bands extremely tight
- Price consolidating within narrow range
- Moderate conviction - confirm with other indicators

**ATR Contraction Only:**
- True range declining significantly
- Lower daily price movement
- Moderate conviction - confirm with price action

**Bullish Breakout (After Squeeze):**
- Price closes above Upper Bollinger Band
- Volume surge confirms (>1.5x average)
- Suggests start of uptrend with strong momentum
- Entry signal for long positions

**Bearish Breakout (After Squeeze):**
- Price closes below Lower Bollinger Band
- Volume surge confirms (>1.5x average)
- Suggests start of downtrend with selling pressure
- Entry signal for short positions

**No Recent Squeeze:**
- Normal volatility levels
- No imminent breakout expected
- Continue monitoring or wait for setup

TECHNICAL NOTES:
----------------
- Uses normalized BB Width (divided by SMA) for price-independent comparison
- Percentile-based thresholds adapt to each stock's volatility characteristics
- Volume confirmation prevents false breakouts on low-volume price spikes
- Lookback period of 126 days (6 months) balances sensitivity and significance
- Scan window of 60 days captures recent signals without clutter
- Matplotlib backend set to 'Agg' when called from web app (Flask compatibility)
- All calculations handle missing data appropriately with pandas rolling functions

DEPENDENCIES:
-------------
- pandas: Data manipulation and analysis
- numpy: Numerical operations
- yfinance: Historical stock data fetching
- matplotlib: Chart visualization
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys
import base64
from io import BytesIO

# Configuration
SQUEEZE_CONFIG = {
    # Bollinger Band Parameters
    'BB_PERIOD': 20,
    'BB_STD': 2,
    
    # ATR Parameters
    'ATR_PERIOD': 14,
    
    # Volume Parameters
    'VOLUME_MA_PERIOD': 20,  # Period for volume moving average
    'VOLUME_SURGE_MULTIPLIER': 1.5,  # Volume surge threshold (1.5 = 150% of MA)
    
    # Squeeze Detection Parameters
    'LOOKBACK': 126,  # Days to define "low" volatility (6 months)
    'SCAN_DAYS': 60,  # Recent days to scan for signals
    'PERCENTILE_THRESHOLD': 20,  # Percentile threshold (values below this = squeeze)
    
    # Data Fetching
    'INTERVAL': '1d',
    'LOOKBACK_PERIODS': 365,  # Days of historical data to fetch
    
    # Execution Control
    'RUN_ON_INIT': False
}


def fetch_data(ticker, config=None):
    """
    Fetches historical OHLCV data for the given ticker.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'RELIANCE.NS')
    config : dict, optional
        Configuration dictionary with INTERVAL and LOOKBACK_PERIODS
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with Date index and OHLCV columns
    """
    if config is None:
        config = SQUEEZE_CONFIG
        
    interval = config.get('INTERVAL', '1d')
    lookback_periods = config.get('LOOKBACK_PERIODS', 365)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_periods)
    
    print(f"Fetching data for {ticker} from {start_date.date()} to {end_date.date()}...")
    
    df = yf.download(
        ticker, 
        start=start_date, 
        end=end_date, 
        interval=interval, 
        progress=False, 
        auto_adjust=False,
        multi_level_index=False
    )
    
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
    
    # Handle multi-level column index if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    return df


def calculate_bollinger_bands(df, config=None):
    """
    Calculates Bollinger Bands and Band Width.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with Close price column
    config : dict, optional
        Configuration dictionary with BB_PERIOD and BB_STD
        
    Returns:
    --------
    pd.DataFrame
        Input DataFrame with added columns: SMA_20, StdDev_20, BB_Upper, BB_Lower, BB_Width
    """
    if config is None:
        config = SQUEEZE_CONFIG
    
    bb_period = config.get('BB_PERIOD', 20)
    bb_std = config.get('BB_STD', 2)
    
    # Calculate Bollinger Bands
    df['SMA_20'] = df['Close'].rolling(window=bb_period).mean()
    df['StdDev_20'] = df['Close'].rolling(window=bb_period).std()
    df['BB_Upper'] = df['SMA_20'] + (bb_std * df['StdDev_20'])
    df['BB_Lower'] = df['SMA_20'] - (bb_std * df['StdDev_20'])
    
    # Band Width: Normalized by SMA for price-independent comparison
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['SMA_20']
    
    return df


def calculate_atr(df, config=None):
    """
    Calculates Average True Range (ATR).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with High, Low, Close columns
    config : dict, optional
        Configuration dictionary with ATR_PERIOD
        
    Returns:
    --------
    pd.DataFrame
        Input DataFrame with added ATR column
    """
    if config is None:
        config = SQUEEZE_CONFIG
        
    atr_period = config.get('ATR_PERIOD', 14)
    
    # Calculate True Range components
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    # True Range is the maximum of the three
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    # ATR is the moving average of True Range
    df['ATR'] = true_range.rolling(window=atr_period).mean()
    
    return df


def calculate_volume_ma(df, config=None):
    """
    Calculates Volume Moving Average for breakout confirmation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with Volume column
    config : dict, optional
        Configuration dictionary with VOLUME_MA_PERIOD
        
    Returns:
    --------
    pd.DataFrame
        Input DataFrame with added Volume_MA column
    """
    if config is None:
        config = SQUEEZE_CONFIG
        
    volume_ma_period = config.get('VOLUME_MA_PERIOD', 20)
    
    # Calculate Volume Moving Average
    df['Volume_MA'] = df['Volume'].rolling(window=volume_ma_period).mean()
    
    return df


def detect_squeeze(df, config=None):
    """
    Detects Volatility Squeeze conditions and Volume-Confirmed Breakouts.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with BB_Width, ATR, Close, BB_Upper, BB_Lower, Volume, Volume_MA columns
    config : dict, optional
        Configuration dictionary with LOOKBACK, SCAN_DAYS, PERCENTILE_THRESHOLD, VOLUME_SURGE_MULTIPLIER
        
    Returns:
    --------
    list of dict
        List of detected signals with Type, Date, Price, BB_Width, ATR
    """
    if config is None:
        config = SQUEEZE_CONFIG
    
    lookback = config.get('LOOKBACK', 126)
    scan_days = config.get('SCAN_DAYS', 60)
    percentile = config.get('PERCENTILE_THRESHOLD', 20)
    volume_surge_multiplier = config.get('VOLUME_SURGE_MULTIPLIER', 1.5)
    
    signals = []
    
    # Calculate rolling percentiles for squeeze detection
    df['Percentile_BB_Width'] = df['BB_Width'].rolling(window=lookback).quantile(percentile / 100)
    df['Percentile_ATR'] = df['ATR'].rolling(window=lookback).quantile(percentile / 100)
    
    # Scan recent period for signals
    subset = df.iloc[-scan_days:].copy()
    
    for date, row in subset.iterrows():
        # Get full context from original dataframe
        full_row = df.loc[date]
        
        bb_width = full_row['BB_Width']
        percentile_bb_width = full_row['Percentile_BB_Width']
        
        atr = full_row['ATR']
        percentile_atr = full_row['Percentile_ATR']
        
        close = full_row['Close']
        bb_upper = full_row['BB_Upper']
        bb_lower = full_row['BB_Lower']
        volume = full_row['Volume']
        volume_ma = full_row['Volume_MA']
        
        # Check for BB Squeeze (below 20th percentile)
        is_bb_squeeze = bb_width <= percentile_bb_width
        
        # Check for ATR Contraction (below 20th percentile)
        is_atr_contraction = atr <= percentile_atr
        
        signal_type = []
        if is_bb_squeeze:
            signal_type.append("BB Squeeze")
        if is_atr_contraction:
            signal_type.append("ATR Contraction")
        
        if signal_type:
            # Check for Volume-Confirmed Breakout
            breakout = ""
            volume_surge = volume > (volume_ma * volume_surge_multiplier) if pd.notna(volume_ma) else False
            
            if close > bb_upper and volume_surge:
                breakout = " (Bullish Breakout)"
            elif close < bb_lower and volume_surge:
                breakout = " (Bearish Breakout)"
            
            signals.append({
                'Type': " + ".join(signal_type) + breakout,
                'Date': date,
                'Price': close,
                'BB_Width': bb_width,
                'ATR': atr
            })
    
    return signals


def plot_squeeze(df, ticker, signals, config=None):
    """
    Generates three-panel chart showing Price, BB Width, and ATR with squeeze indicators.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with all calculated indicators
    ticker : str
        Stock ticker symbol for chart title
    signals : list
        List of detected squeeze/breakout signals
    config : dict, optional
        Configuration dictionary (not currently used in plotting)
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the three-panel plot
    """
    if config is None:
        config = SQUEEZE_CONFIG
    
    lookback = config.get('LOOKBACK', 126)
    
    fig = plt.figure(figsize=(14, 8))
    
    # --- Subplot 1: Price & Bollinger Bands ---
    ax1 = plt.subplot(3, 1, 1)
    
    # Plot candlesticks
    for idx in range(len(df)):
        date = df.index[idx]
        open_price = df['Open'].iloc[idx]
        close_price = df['Close'].iloc[idx]
        high_price = df['High'].iloc[idx]
        low_price = df['Low'].iloc[idx]
        
        # Determine color based on bullish/bearish candle (Zerodha-style bright colors)
        color = '#05fa32' if close_price >= open_price else '#fc0c08'  # Bright green for bullish, bright red for bearish
        
        # Draw the high-low line (wick)
        ax1.plot([date, date], [low_price, high_price], color='black', linewidth=0.5, alpha=0.7)
        
        # Draw the open-close rectangle (body)
        height = abs(close_price - open_price)
        bottom = min(open_price, close_price)
        ax1.bar(date, height, bottom=bottom, width=0.6, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Plot Bollinger Bands
    ax1.plot(df.index, df['BB_Upper'], label='Upper BB', color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax1.plot(df.index, df['BB_Lower'], label='Lower BB', color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax1.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], color='gray', alpha=0.1)
    
    # Plot Signals
    breakout_plotted = False
    squeeze_plotted = False
    
    for sig in signals:
        if "Breakout" in sig['Type']:
            color = 'green' if 'Bullish' in sig['Type'] else 'red'
            marker = '^' if 'Bullish' in sig['Type'] else 'v'
            label = 'Breakout' if not breakout_plotted else ""
            ax1.scatter(sig['Date'], sig['Price'], color=color, marker=marker, s=100, 
                       zorder=5, label=label, edgecolors='black', linewidths=1)
            breakout_plotted = True
        elif "Squeeze" in sig['Type'] or "Contraction" in sig['Type']:
            label = 'Squeeze' if not squeeze_plotted else ""
            ax1.scatter(sig['Date'], sig['Price'], color='orange', marker='o', s=50, 
                       zorder=4, label=label, alpha=0.7)
            squeeze_plotted = True
    
    ax1.set_title(f'{ticker} - Volatility Squeeze Analysis', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=11)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # --- Subplot 2: BB Width ---
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(df.index, df['BB_Width'], label='BB Width', color='purple', linewidth=1.5)
    
    # Highlight Squeeze zones (below 20th percentile)
    percentile = config.get('PERCENTILE_THRESHOLD', 20)
    squeeze_mask = df['BB_Width'] <= df['BB_Width'].rolling(window=lookback).quantile(percentile / 100)
    ax2.fill_between(df.index, 0, df['BB_Width'], where=squeeze_mask, 
                      color='orange', alpha=0.3, label='Squeeze Zone (≤20th percentile)')
    
    ax2.set_ylabel('BB Width', fontsize=11)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # --- Subplot 3: ATR ---
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(df.index, df['ATR'], label=f'ATR ({config.get("ATR_PERIOD", 14)})', 
             color='brown', linewidth=1.5)
    
    # Highlight Contraction zones (below 20th percentile)
    contraction_mask = df['ATR'] <= df['ATR'].rolling(window=lookback).quantile(percentile / 100)
    ax3.fill_between(df.index, 0, df['ATR'], where=contraction_mask, 
                      color='yellow', alpha=0.3, label='Contraction Zone (≤20th percentile)')
    
    ax3.set_ylabel('ATR', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Adjust spacing to prevent overflow
    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.95, hspace=0.3)
    
    return fig


def run_analysis(ticker, show_plot=True, config=None):
    """
    Main orchestration function for Volatility Squeeze analysis.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol to analyze
    show_plot : bool, optional
        If True, displays the plot interactively. If False, returns figure for web app use.
    config : dict, optional
        Configuration dictionary to override defaults
        
    Returns:
    --------
    dict
        Results dictionary containing:
        - success: bool
        - ticker: str
        - current_bb_width: float
        - current_atr: float
        - signals: list of dict
        - figure: matplotlib.figure.Figure
        - dataframe: pd.DataFrame
    """
    # Set matplotlib backend for web app compatibility
    if not show_plot:
        matplotlib.use('Agg')
    
    # Merge provided config with defaults
    if config is None:
        config = SQUEEZE_CONFIG.copy()
    else:
        merged_config = SQUEEZE_CONFIG.copy()
        merged_config.update(config)
        config = merged_config
    
    try:
        # Fetch data
        df = fetch_data(ticker, config)
        
        # Calculate indicators
        df = calculate_bollinger_bands(df, config)
        df = calculate_atr(df, config)
        df = calculate_volume_ma(df, config)
        
        # Detect squeeze conditions
        signals = detect_squeeze(df, config)
        
        # Generate plot
        fig = plot_squeeze(df, ticker, signals, config)
        
        if show_plot:
            plt.show()
        
        # Get current values
        current_bb_width = df['BB_Width'].iloc[-1] if 'BB_Width' in df.columns else None
        current_atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else None
        
        return {
            'success': True,
            'ticker': ticker,
            'current_bb_width': current_bb_width,
            'current_atr': current_atr,
            'signals': signals,
            'figure': fig,
            'dataframe': df
        }
        
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        return {
            'success': False,
            'ticker': ticker,
            'error': str(e)
        }


def analyze_batch(tickers_file='tickers.txt'):
    """
    Runs Volatility Squeeze analysis on multiple tickers from a file.
    
    Parameters:
    -----------
    tickers_file : str
        Path to text file containing ticker symbols (one per line)
    """
    # Check for file in multiple locations
    if not os.path.exists(tickers_file):
        if os.path.exists(os.path.join('..', '..', tickers_file)):
            tickers_file = os.path.join('..', '..', tickers_file)
        elif os.path.exists(os.path.join('..', tickers_file)):
            tickers_file = os.path.join('..', tickers_file)
        else:
            print(f"Error: {tickers_file} not found.")
            return
    
    with open(tickers_file, 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]
    
    print(f"\nFound {len(tickers)} tickers. Starting Volatility Squeeze analysis...\n")
    
    for ticker in tickers:
        print(f"Analyzing {ticker}...")
        
        try:
            results = run_analysis(ticker, show_plot=False)
            
            if results['success']:
                signals = results['signals']
                
                if signals:
                    latest = signals[-1]
                    print(f"  Latest Status: {latest['Type']} on {latest['Date'].date()}")
                    print(f"  BB Width: {latest['BB_Width']:.4f}, ATR: {latest['ATR']:.2f}")
                else:
                    print("  No recent volatility squeeze detected.")
            else:
                print(f"  Error: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"  Error analyzing {ticker}: {e}")
        
        print()  # Blank line between tickers


if __name__ == "__main__":
    import os
    
    # Default to single ticker for testing
    run_batch = False
    
    if os.path.exists('tickers.txt') and run_batch:
        analyze_batch('tickers.txt')
    elif os.path.exists('../tickers.txt') and run_batch:
        analyze_batch('../tickers.txt')
    else:
        ticker = "DCBBANK.NS"
        print(f"Running Volatility Squeeze Analysis for {ticker}...\n")
        
        try:
            results = run_analysis(ticker, show_plot=True)
            
            if results['success']:
                signals = results['signals']
                
                print(f"\nCurrent BB Width: {results['current_bb_width']:.4f}")
                print(f"Current ATR: {results['current_atr']:.2f}\n")
                
                if signals:
                    print("--- Volatility Squeeze Signals (Last 60 Days) ---")
                    for sig in signals:
                        print(f"{sig['Date'].date()}: {sig['Type']} (Price: {sig['Price']:.2f}, "
                              f"BB Width: {sig['BB_Width']:.4f}, ATR: {sig['ATR']:.2f})")
                else:
                    print("No recent volatility squeeze detected.")
            else:
                print(f"Analysis failed: {results.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"Error: {e}")
