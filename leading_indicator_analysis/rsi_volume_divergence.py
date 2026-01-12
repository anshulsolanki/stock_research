# -------------------------------------------------------------------------------
# Project: Stock Analysis (https://github.com/anshulsolanki/stock_analysis)
# Author:  Anshul Solanki
# License: MIT License
# 
# DISCLAIMER: 
# This software is for educational purposes only. It is not financial advice.
# Stock trading involves risks. The author is not responsible for any losses.
# -------------------------------------------------------------------------------

"""
Copyright (c) 2026 Anshul Solanki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
RSI-VOLUME DIVERGENCE ANALYSIS TOOL
====================================

PURPOSE:
--------
This module performs comprehensive RSI-Volume Divergence analysis, a powerful triple-indicator system
that combines price action, RSI momentum, and volume analysis to identify high-conviction reversal signals.

RSI-Volume Divergence occurs when price + RSI momentum move in one direction, but volume moves in the 
opposite direction, revealing hidden strength or weakness in the market.

WHAT IT DOES:
-------------
1. **Triple Divergence Detection**: Analyzes three indicators simultaneously
   - Price action (highs and lows)
   - RSI momentum (Wilder's Smoothing Method)
   - Volume trends (using Volume Moving Averages)

2. **Bullish RSI-Volume Divergence**:
   - Price makes lower lows (LL) - downtrend continuing
   - RSI makes higher lows (HL) - momentum recovering
   - Volume decreases during fall - sellers losing strength
   → **Signal**: Early bullish reversal (buyers accumulating quietly)
   → **Interpretation**: Selling pressure exhausting, potential bottom forming

3. **Bearish RSI-Volume Divergence**:
   - Price makes higher highs (HH) - uptrend continuing
   - RSI makes lower highs (LH) - momentum fading
   - Volume falls on rallies - buyers losing conviction
   → **Signal**: Potential topping/reversal (weak rally)
   → **Interpretation**: Buying pressure weakening, potential top forming

4. **Early Trend Reversal Signals**:
   - Identifies strongest confluence signals
   - Price at new extreme + RSI divergence + volume confirmation
   - Highest conviction turning points

5. **Volume Analysis**:
   - Calculates Volume SMA-20 (short-term volume trend)
   - Calculates Volume SMA-50 (long-term accumulation/distribution)
   - Compares current volume against both MAs for confirmation

6. **Visual Analysis**: Generates three-panel charts showing:
   - Price action with marked peaks, troughs, and divergence points
   - RSI indicator with overbought/oversold zones
   - Volume bars with moving averages and trend indicators

METHODOLOGY:
------------
RSI Calculation (Wilder's Smoothing Method):
- Calculate price changes: delta = Close(t) - Close(t-1)
- Separate gains and losses
- Apply Wilder's smoothing: EMA with alpha = 1/period
- Calculate RS = avg_gain / avg_loss
- Calculate RSI = 100 - (100 / (1 + RS))

Volume Moving Averages:
- Volume MA-20: Short-term volume trend (recent accumulation/distribution)
- Volume MA-50: Long-term volume baseline (institutional activity)

Bullish RSI-Volume Divergence Detection:
1. Find all consecutive price troughs (local lows)
2. For each consecutive pair of troughs:
   a. Check if price LL: current_price < previous_price
   b. Check if RSI HL: current_rsi > previous_rsi
   c. Check if volume decreasing: current_volume < previous_volume
3. If all three conditions met → Bullish RSI-Volume Divergence
4. Record divergence with full details

Bearish RSI-Volume Divergence Detection:
1. Find all consecutive price peaks (local highs)
2. For each consecutive pair of peaks:
   a. Check if price HH: current_price > previous_price
   b. Check if RSI LH: current_rsi < previous_rsi
   c. Check if volume decreasing: current_volume < previous_volume
3. If all three conditions met → Bearish RSI-Volume Divergence
4. Record divergence with full details

Early Reversal Identification:
- Look for divergences where volume shrinks significantly
- Prioritize divergences in overbought/oversold RSI zones
- Flag highest-conviction signals for user attention

WHY COMBINE RSI + VOLUME?
--------------------------
**Standard Divergence (Price vs RSI):**
- Shows momentum weakness/strength
- Indicates potential reversal
- BUT: Doesn't show conviction behind the move

**Price + Volume Divergence:**
- Shows supply-demand shift
- Reveals institutional activity
- BUT: Doesn't capture momentum dynamics

**Both Together (RSI-Volume Divergence):**
- Momentum reversal + Supply-demand shift
- Higher conviction turning point
- Reduces false signals
- Catches reversals before they're obvious

KEY METRICS:
------------
- Current RSI: Latest RSI value (0-100 scale)
- Current Volume: Latest volume bar
- Volume MA-20: 20-period volume moving average
- Volume MA-50: 50-period volume moving average
- Bullish Divergences: List of all detected bullish RSI-volume divergences
- Bearish Divergences: List of all detected bearish RSI-volume divergences
- Early Reversals: Highest-conviction reversal signals
- Peaks and Troughs: All identified local extrema

CONFIGURATION:
--------------
Default parameters (customizable via RSI_VOLUME_CONFIG or function arguments):
- RSI_PERIOD: 14 (standard RSI calculation period)
- VOLUME_MA_SHORT: 20 (short-term volume trend)
- VOLUME_MA_LONG: 50 (long-term volume baseline)
- ORDER: 5 (sensitivity for peak/trough detection)
- RSI_OVERBOUGHT: 70 (threshold for overbought zone)
- RSI_OVERSOLD: 30 (threshold for oversold zone)
- INTERVAL: '1d' (daily data; also supports '1wk', '1mo', '1h', '15m', etc.)
- LOOKBACK_PERIODS: 730 days (2 years of history)

USAGE:
------
Run as standalone script:
    python rsi_volume_divergence.py

Or import and use programmatically:
    from rsi_volume_divergence import run_analysis
    results = run_analysis(ticker="AAPL", show_plot=True, config={'RSI_PERIOD': 14})

OUTPUT:
-------
Returns dictionary containing:
- success: Boolean indicating if analysis completed successfully
- ticker: Stock ticker symbol
- current_rsi: Current RSI value
- current_volume: Latest volume
- volume_ma_20: 20-period volume MA
- volume_ma_50: 50-period volume MA
- bullish_divergences: List of bullish RSI-volume divergences
- bearish_divergences: List of bearish RSI-volume divergences
- early_reversals: High-conviction reversal signals
- figure: Matplotlib figure object for visualization
- dataframe: Full DataFrame with all calculated indicators

TYPICAL USE CASES:
------------------
1. **Early Reversal Detection**: Catch trend reversals before they're obvious
2. **Confirmation Tool**: Validate other technical signals with volume context
3. **Entry Timing**: Use bullish divergences for long entry with high conviction
4. **Exit Timing**: Use bearish divergences for exit or short signals
5. **Risk Management**: Avoid buying into rallies with falling volume
6. **Multi-timeframe Analysis**: Run on different intervals for comprehensive view

INTERPRETATION GUIDE:
---------------------
**Bullish RSI-Volume Divergence:**
- "Hidden accumulation" - Smart money buying quietly
- Price falling but momentum recovering + volume drying up
- Sellers exhausted, buyers stepping in
- **Best when**: RSI < 30 (oversold) and volume well below MA-50
- **Action**: Consider long positions, watch for confirmation

**Bearish RSI-Volume Divergence:**
- "Hidden distribution" - Smart money exiting quietly
- Price rising but momentum fading + volume decreasing
- Buyers losing conviction, sellers preparing
- **Best when**: RSI > 70 (overbought) and volume below MA-20
- **Action**: Consider taking profits, watch for breakdown

**Early Reversal Signal:**
- All three indicators aligning perfectly
- Volume shrinks significantly (< 50% of MA-50)
- RSI divergence clear and strong
- **Action**: Highest conviction signal - strong reversal likely

**No Divergence:**
- Price, RSI, and volume all trending together
- Trend likely to continue
- Wait for divergence or other signals

TECHNICAL NOTES:
----------------
- Uses Wilder's Smoothing (EMA) for accurate RSI calculation
- Volume comparison uses both current volume and volume MAs
- Matplotlib backend set to 'Agg' when called from web app
- Three-panel chart with synchronized x-axis for easy analysis
- All divergences across dataset detected, not just most recent
- Peak/trough detection uses scipy.signal.argrelextrema

DEPENDENCIES:
-------------
- pandas: Data manipulation and analysis
- numpy: Numerical operations
- yfinance: Historical stock data fetching
- matplotlib: Chart visualization
- scipy.signal: Peak and trough detection
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.signal import argrelextrema

# Configuration
RSI_VOLUME_CONFIG = {
    # RSI Parameters
    'RSI_PERIOD': 14,
    'ORDER': 5,
    'RSI_OVERBOUGHT': 70,
    'RSI_OVERSOLD': 30,
    
    # Volume Parameters
    'VOLUME_MA_SHORT': 20,
    'VOLUME_MA_LONG': 50,
    
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
    Fetches historical OHLCV data using config parameters.
    """
    cfg = config if config else RSI_VOLUME_CONFIG
    
    interval = cfg.get('INTERVAL', RSI_VOLUME_CONFIG['INTERVAL'])
    lookback_periods = cfg.get('LOOKBACK_PERIODS', RSI_VOLUME_CONFIG['LOOKBACK_PERIODS'])
    
    end_date = datetime.now()
    
    # Adjust lookback calculation based on interval to ensure sufficient data points
    if interval in ['1wk', '1w']:
        # For weekly data, need 7x more calendar days to get same number of data points
        start_date = end_date - timedelta(days=lookback_periods * 7)
    elif interval in ['1mo', '1M']:
        # For monthly data, need 30x more calendar days
        start_date = end_date - timedelta(days=lookback_periods * 30)
    elif 'm' in interval or 'h' in interval:
        # For intraday intervals (15m, 1h, etc.), use a fixed period of days
        # Intraday data is typically limited to 60 days max by yfinance
        # Use 59 days to be safe
        start_date = end_date - timedelta(days=min(59, lookback_periods))
    else:
        # For daily intervals ('1d'), use lookback_periods as days directly
        start_date = end_date - timedelta(days=lookback_periods)
    
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, 
                     progress=False, auto_adjust=False, multi_level_index=False)
    
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
    cfg = config if config else RSI_VOLUME_CONFIG
    period = cfg.get('RSI_PERIOD', RSI_VOLUME_CONFIG['RSI_PERIOD'])
    
    # Calculate price changes
    delta = df['Close'].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Use Wilder's smoothing method (EMA with alpha = 1/period)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def calculate_volume_ma(df, config=None):
    """
    Calculates Volume Moving Averages for trend analysis.
    
    Volume MA-20: Short-term volume trend
    Volume MA-50: Long-term accumulation/distribution baseline
    """
    cfg = config if config else RSI_VOLUME_CONFIG
    ma_short = cfg.get('VOLUME_MA_SHORT', RSI_VOLUME_CONFIG['VOLUME_MA_SHORT'])
    ma_long = cfg.get('VOLUME_MA_LONG', RSI_VOLUME_CONFIG['VOLUME_MA_LONG'])
    
    df['Volume_MA_20'] = df['Volume'].rolling(window=ma_short).mean()
    df['Volume_MA_50'] = df['Volume'].rolling(window=ma_long).mean()
    
    return df

def detect_rsi_volume_divergence(df, config=None):
    """
    Detects Bullish and Bearish RSI-Volume Divergences.
    
    Bullish: Price LL, RSI HL, Volume decreasing
    Bearish: Price HH, RSI LH, Volume decreasing
    """
    cfg = config if config else RSI_VOLUME_CONFIG
    order = cfg.get('ORDER', RSI_VOLUME_CONFIG['ORDER'])
     
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
    
    bullish_divergences = []
    bearish_divergences = []
    
    # ========== Bearish RSI-Volume Divergence (Price HH, RSI LH, Volume Decreasing) ==========
    for i in range(1, len(price_peak_indices)):
        prev_idx = price_peak_indices[i-1]
        curr_idx = price_peak_indices[i]
        
        prev_price = df.iloc[prev_idx]['Close']
        curr_price = df.iloc[curr_idx]['Close']
        prev_rsi = df.iloc[prev_idx]['RSI']
        curr_rsi = df.iloc[curr_idx]['RSI']
        prev_volume = df.iloc[prev_idx]['Volume']
        curr_volume = df.iloc[curr_idx]['Volume']
        
        # Check conditions: Price HH AND RSI LH AND Volume Decreasing
        if curr_price > prev_price and curr_rsi < prev_rsi and curr_volume < prev_volume:
            bearish_divergences.append({
                'Type': 'Bearish RSI-Volume Divergence',
                'Date': df.index[curr_idx],
                'Price': curr_price,
                'RSI': curr_rsi,
                'Volume': curr_volume,
                'Details': f"Price HH ({prev_price:.2f} → {curr_price:.2f}), RSI LH ({prev_rsi:.2f} → {curr_rsi:.2f}), Volume↓ ({prev_volume:,.0f} → {curr_volume:,.0f})"
            })

    # ========== Bullish RSI-Volume Divergence (Price LL, RSI HL, Volume Decreasing) ==========
    for i in range(1, len(price_trough_indices)):
        prev_idx = price_trough_indices[i-1]
        curr_idx = price_trough_indices[i]
        
        prev_price = df.iloc[prev_idx]['Close']
        curr_price = df.iloc[curr_idx]['Close']
        prev_rsi = df.iloc[prev_idx]['RSI']
        curr_rsi = df.iloc[curr_idx]['RSI']
        prev_volume = df.iloc[prev_idx]['Volume']
        curr_volume = df.iloc[curr_idx]['Volume']
        
        # Check conditions: Price LL AND RSI HL AND Volume Decreasing
        if curr_price < prev_price and curr_rsi > prev_rsi and curr_volume < prev_volume:
            bullish_divergences.append({
                'Type': 'Bullish RSI-Volume Divergence',
                'Date': df.index[curr_idx],
                'Price': curr_price,
                'RSI': curr_rsi,
                'Volume': curr_volume,
                'Details': f"Price LL ({prev_price:.2f} → {curr_price:.2f}), RSI HL ({prev_rsi:.2f} → {curr_rsi:.2f}), Volume↓ ({prev_volume:,.0f} → {curr_volume:,.0f})"
            })
            
    return bullish_divergences, bearish_divergences

def identify_early_reversals(df, bullish_divs, bearish_divs, config=None):
    """
    Identifies early reversal signals - highest conviction divergences.
    
    Criteria for early reversal:
    - Bullish: RSI < 30 (oversold) + volume significantly below MA-50
    - Bearish: RSI > 70 (overbought) + volume significantly below MA-20
    """
    cfg = config if config else RSI_VOLUME_CONFIG
    rsi_overbought = cfg.get('RSI_OVERBOUGHT', RSI_VOLUME_CONFIG['RSI_OVERBOUGHT'])
    rsi_oversold = cfg.get('RSI_OVERSOLD', RSI_VOLUME_CONFIG['RSI_OVERSOLD'])
    
    early_reversals = []
    
    # Check bullish divergences for early reversal signals
    for div in bullish_divs:
        div_date = div['Date']
        div_rsi = div['RSI']
        div_volume = div['Volume']
        
        # Get volume MA at that date
        volume_ma_50 = df.loc[div_date, 'Volume_MA_50']
        
        # Early bullish reversal: RSI oversold + volume well below MA-50
        if div_rsi < rsi_oversold and div_volume < (volume_ma_50 * 0.7):
            early_reversals.append({
                'Type': 'Early Bullish Reversal',
                'Date': div_date,
                'Price': div['Price'],
                'RSI': div_rsi,
                'Volume': div_volume,
                'Details': f"Strong bullish signal: RSI={div_rsi:.1f} (oversold), Volume={div_volume:,.0f} (<<MA50={volume_ma_50:,.0f})"
            })
    
    # Check bearish divergences for early reversal signals
    for div in bearish_divs:
        div_date = div['Date']
        div_rsi = div['RSI']
        div_volume = div['Volume']
        
        # Get volume MA at that date
        volume_ma_20 = df.loc[div_date, 'Volume_MA_20']
        
        # Early bearish reversal: RSI overbought + volume below MA-20
        if div_rsi > rsi_overbought and div_volume < volume_ma_20:
            early_reversals.append({
                'Type': 'Early Bearish Reversal',
                'Date': div_date,
                'Price': div['Price'],
                'RSI': div_rsi,
                'Volume': div_volume,
                'Details': f"Strong bearish signal: RSI={div_rsi:.1f} (overbought), Volume={div_volume:,.0f} (<MA20={volume_ma_20:,.0f})"
            })
    
    return early_reversals

def plot_rsi_volume_divergence(df, ticker, show_plot=True, config=None, bullish_divs=None, bearish_divs=None, early_reversals=None):
    """
    Plots Price, RSI, and Volume with divergence markers.
    
    Three-panel chart:
    1. Price with divergence markers
    2. RSI with overbought/oversold zones
    3. Volume with moving averages
    """
    cfg = config if config else RSI_VOLUME_CONFIG
    overbought = cfg.get('RSI_OVERBOUGHT', RSI_VOLUME_CONFIG['RSI_OVERBOUGHT'])
    oversold = cfg.get('RSI_OVERSOLD', RSI_VOLUME_CONFIG['RSI_OVERSOLD'])
    period = cfg.get('RSI_PERIOD', RSI_VOLUME_CONFIG['RSI_PERIOD'])
    
    fig = plt.figure(figsize=(14, 12))
    
    # ========== Price Plot ==========
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(df.index, df['Close'], label='Close Price', color='blue', linewidth=1.5)
    
    # Plot price peaks and troughs
    price_peaks = df[df['price_peak'].notna()]
    price_troughs = df[df['price_trough'].notna()]
    
    if not price_peaks.empty:
        plt.scatter(price_peaks.index, price_peaks['Close'], color='orange', marker='v', 
                   s=60, alpha=0.6, zorder=3)
    if not price_troughs.empty:
        plt.scatter(price_troughs.index, price_troughs['Close'], color='cyan', marker='^', 
                   s=60, alpha=0.6, zorder=3)
    
    # Plot Divergence Markers
    if bullish_divs:
        for div in bullish_divs:
            plt.scatter(div['Date'], div['Price'], color='green', marker='^', s=200, 
                       zorder=5, edgecolors='black', linewidths=2)
    
    if bearish_divs:
        for div in bearish_divs:
            plt.scatter(div['Date'], div['Price'], color='red', marker='v', s=200, 
                       zorder=5, edgecolors='black', linewidths=2)
    
    # Plot Early Reversal markers (with star)
    if early_reversals:
        for rev in early_reversals:
            color = 'darkgreen' if 'Bullish' in rev['Type'] else 'darkred'
            plt.scatter(rev['Date'], rev['Price'], color=color, marker='*', s=300, 
                       zorder=6, edgecolors='gold', linewidths=2)
    
    plt.title(f'{ticker} RSI-Volume Divergence Analysis', fontsize=14, fontweight='bold')
    plt.ylabel('Price', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # ========== RSI Plot ==========
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
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
        plt.scatter(rsi_peaks.index, rsi_peaks['RSI'], color='orange', marker='v', 
                   s=60, alpha=0.6, zorder=3)
    if not rsi_troughs.empty:
        plt.scatter(rsi_troughs.index, rsi_troughs['RSI'], color='cyan', marker='^', 
                   s=60, alpha=0.6, zorder=3)
    
    # Plot divergence markers on RSI
    if bullish_divs:
        for div in bullish_divs:
            plt.scatter(div['Date'], div['RSI'], color='green', marker='^', s=200, 
                       zorder=5, edgecolors='black', linewidths=2)
    
    if bearish_divs:
        for div in bearish_divs:
            plt.scatter(div['Date'], div['RSI'], color='red', marker='v', s=200, 
                       zorder=5, edgecolors='black', linewidths=2)
    
    plt.title('Relative Strength Index (RSI)', fontsize=12)
    plt.ylabel('RSI', fontsize=11)
    plt.ylim(0, 100)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # ========== Volume Plot ==========
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    
    # Color volume bars based on price direction
    colors = ['green' if df['Close'].iloc[i] >= df['Close'].iloc[i-1] else 'red' 
              for i in range(1, len(df))]
    colors = ['gray'] + colors  # First bar is gray
    
    plt.bar(df.index, df['Volume'], color=colors, alpha=0.5, width=0.8)
    
    # Plot Volume MAs
    plt.plot(df.index, df['Volume_MA_20'], label='Volume MA-20', color='blue', linewidth=1.5, alpha=0.7)
    plt.plot(df.index, df['Volume_MA_50'], label='Volume MA-50', color='orange', linewidth=1.5, alpha=0.7)
    
    plt.title('Volume with Moving Averages', fontsize=12)
    plt.ylabel('Volume', fontsize=11)
    plt.xlabel('Date', fontsize=11)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
        
    return fig

def run_analysis(ticker, show_plot=True, config=None, df=None):
    """
    Main analysis function that can be called from a GUI or other scripts.
    
    Args:
        ticker: Stock ticker symbol
        show_plot: If True, displays the plot. If False, just returns the figure.
        config: Optional dictionary to override default configuration
        df: Optional pre-fetched DataFrame. If None, fetches new data.
    
    Returns:
        dict: Analysis results including divergences, early reversals, and figure
    """
    # Set non-interactive backend when not showing plot (e.g., for Flask/web)
    if not show_plot:
        matplotlib.use('Agg', force=True)
    
    try:
        # Merge provided config with defaults
        current_config = RSI_VOLUME_CONFIG.copy()
        if config:
            current_config.update(config)

        # Fetch and calculate
        if df is not None:
             df = df.copy()
        else:
             df = fetch_data(ticker, config=current_config)
             
        df = calculate_rsi(df, config=current_config)
        df = calculate_volume_ma(df, config=current_config)
        
        bullish_divs, bearish_divs = detect_rsi_volume_divergence(df, config=current_config)
        early_reversals = identify_early_reversals(df, bullish_divs, bearish_divs, config=current_config)
        
        current_rsi = df['RSI'].iloc[-1]
        current_volume = df['Volume'].iloc[-1]
        volume_ma_20 = df['Volume_MA_20'].iloc[-1]
        volume_ma_50 = df['Volume_MA_50'].iloc[-1]
        
        # Generate plot
        fig = plot_rsi_volume_divergence(df, ticker, show_plot=show_plot, config=current_config,
                                         bullish_divs=bullish_divs, bearish_divs=bearish_divs,
                                         early_reversals=early_reversals)
        
        return {
            'success': True,
            'ticker': ticker,
            'current_rsi': current_rsi,
            'current_volume': current_volume,
            'volume_ma_20': volume_ma_20,
            'volume_ma_50': volume_ma_50,
            'bullish_divergences': bullish_divs,
            'bearish_divergences': bearish_divs,
            'early_reversals': early_reversals,
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
    Reads tickers from a JSON file and performs RSI-Volume Divergence analysis for each.
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
        
    print(f"Found {len(tickers)} unique tickers in {json_file}. Starting RSI-Volume Divergence analysis...")
    
    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")
        try:
            result = run_analysis(ticker, show_plot=False)
            
            if result['success']:
                print(f"  Current RSI: {result['current_rsi']:.2f}")
                print(f"  Current Volume: {result['current_volume']:,.0f}")
                print(f"  Volume MA-20: {result['volume_ma_20']:,.0f}")
                print(f"  Volume MA-50: {result['volume_ma_50']:,.0f}")
                
                if result['bullish_divergences']:
                    print(f"  --- Bullish RSI-Volume Divergences ---")
                    for div in result['bullish_divergences']:
                        print(f"  {div['Date'].date()}: {div['Details']}")
                
                if result['bearish_divergences']:
                    print(f"  --- Bearish RSI-Volume Divergences ---")
                    for div in result['bearish_divergences']:
                        print(f"  {div['Date'].date()}: {div['Details']}")
                
                if result['early_reversals']:
                    print(f"  ⭐ --- EARLY REVERSAL SIGNALS ---")
                    for rev in result['early_reversals']:
                        print(f"  {rev['Date'].date()}: {rev['Details']}")
                
                if not result['bullish_divergences'] and not result['bearish_divergences']:
                    print("  No RSI-Volume divergences detected.")
            else:
                print(f"  Error: {result.get('error')}")
                
        except Exception as e:
            print(f"  Error analyzing {ticker}: {e}")

if __name__ == "__main__":
    import os
    
    # Load execution parameters from config
    run_batch = RSI_VOLUME_CONFIG['RUN_BATCH']
    default_ticker = RSI_VOLUME_CONFIG['DEFAULT_TICKER']
    batch_relative_path = RSI_VOLUME_CONFIG['BATCH_RELATIVE_PATH']
    
    # Resolve batch file path
    batch_file = os.path.join(os.path.dirname(__file__), batch_relative_path)

    if run_batch and os.path.exists(batch_file):
        analyze_batch(batch_file)
    else:
        print(f"Running RSI-Volume Divergence Analysis for {default_ticker}...")
        
        result = run_analysis(default_ticker, show_plot=True)
        
        if result['success']:
            print(f"\nCurrent RSI: {result['current_rsi']:.2f}")
            print(f"Current Volume: {result['current_volume']:,.0f}")
            print(f"Volume MA-20: {result['volume_ma_20']:,.0f}")
            print(f"Volume MA-50: {result['volume_ma_50']:,.0f}")
            
            if result['bullish_divergences']:
                print("\n--- Bullish RSI-Volume Divergences Detected ---")
                for div in result['bullish_divergences']:
                    print(f"{div['Date'].date()}: {div['Details']}")
            
            if result['bearish_divergences']:
                print("\n--- Bearish RSI-Volume Divergences Detected ---")
                for div in result['bearish_divergences']:
                    print(f"{div['Date'].date()}: {div['Details']}")
            
            if result['early_reversals']:
                print("\n⭐ --- EARLY REVERSAL SIGNALS ---")
                for rev in result['early_reversals']:
                    print(f"{rev['Date'].date()}: {rev['Details']}")
            
            if not result['bullish_divergences'] and not result['bearish_divergences']:
                print("\nNo RSI-Volume divergences detected.")
        else:
            print(f"Error: {result.get('error')}")
