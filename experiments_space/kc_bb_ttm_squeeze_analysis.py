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
TTM SQUEEZE ANALYSIS TOOL
=========================

PURPOSE:
--------
This module implements the Combined Bollinger Band (BB) and Keltner Channel (KC) Squeeze Strategy,
commonly known as the TTM Squeeze. It identifies periods of extreme low volatility (squeeze)
that often precede strong price breakouts.

LOGIC:
------
1. Volatility Squeeze (Red Dot):
   - Occurs when Bollinger Bands are completely inside Keltner Channels.
   - Condition: BB_Upper < KC_Upper AND BB_Lower > KC_Lower.
   - Signals a consolidation phase.

2. Breakout (Squeeze Firing):
   - Occurs when the squeeze is released.
   - Buy Signal: Squeeze was On yesterday AND Close > BB_Upper.
   - Sell Signal: Squeeze was On yesterday AND Close < BB_Lower.

INDICATORS:
-----------
1. Keltner Channels (KC):
   - Middle: EMA 20
   - Upper/Lower: Middle +/- (1.5 * ATR 10)
   
2. Bollinger Bands (BB):
   - Middle: SMA 20
   - Upper/Lower: Middle +/- (2.0 * StdDev 20)

USAGE:
------
Run as standalone:
    python ttm_squeeze_analysis.py

Import:
    from ttm_squeeze_analysis import run_analysis
    results = run_analysis("RELIANCE.NS")
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Configuration
TTM_CONFIG = {
    # Bollinger Band Parameters
    'BB_LENGTH': 20,
    'BB_MULT': 2.0,
    
    # Keltner Channel Parameters
    'KC_LENGTH': 20,
    'KC_ATR_LENGTH': 10,
    'KC_MULT': 1.5,
    
    # Data Fetching
    'INTERVAL': '1d',
    'LOOKBACK_PERIODS': 365,
    
    # Plotting
    'SHOW_PLOT': True
}

def fetch_data(ticker, config=None):
    """Fetches historical data."""
    if config is None: config = TTM_CONFIG
    
    interval = config.get('INTERVAL', '1d')
    lookback = config.get('LOOKBACK_PERIODS', 365)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback)
    
    print(f"Fetching data for {ticker}...")
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
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    return df

def calculate_indicators(df, config=None):
    """Calculates BB, KC, and Squeeze indicators."""
    if config is None: config = TTM_CONFIG
    
    # Parameters
    bb_len = config.get('BB_LENGTH', 20)
    bb_mult = config.get('BB_MULT', 2.0)
    kc_len = config.get('KC_LENGTH', 20)
    kc_atr_len = config.get('KC_ATR_LENGTH', 10)
    kc_mult = config.get('KC_MULT', 1.5)
    
    # --- Bollinger Bands ---
    df['SMA_20'] = df['Close'].rolling(window=bb_len).mean()
    df['StdDev_20'] = df['Close'].rolling(window=bb_len).std()
    df['BB_Upper'] = df['SMA_20'] + (bb_mult * df['StdDev_20'])
    df['BB_Lower'] = df['SMA_20'] - (bb_mult * df['StdDev_20'])
    
    # --- Keltner Channels ---
    # EMA 20
    df['EMA_20'] = df['Close'].ewm(span=kc_len, adjust=False).mean()
    
    # ATR 10
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=kc_atr_len).mean()
    
    df['KC_Upper'] = df['EMA_20'] + (kc_mult * df['ATR'])
    df['KC_Lower'] = df['EMA_20'] - (kc_mult * df['ATR'])
    
    # --- Squeeze Detection ---
    # Squeeze On: BB inside KC
    df['Squeeze_On'] = (df['BB_Upper'] < df['KC_Upper']) & (df['BB_Lower'] > df['KC_Lower'])
    
    # --- Breakout Signals ---
    # Trigger: Squeeze was True yesterday AND Price breaks BB today
    df['Squeeze_Was_On'] = df['Squeeze_On'].shift(1)
    
    conditions = [
        (df['Squeeze_Was_On'] == True) & (df['Close'] > df['BB_Upper']),
        (df['Squeeze_Was_On'] == True) & (df['Close'] < df['BB_Lower'])
    ]
    choices = ['Buy Signal', 'Sell Signal']
    df['Signal'] = np.select(conditions, choices, default='')
    
    return df

def plot_analysis(df, ticker, config=None):
    """Plots Price, BB, KC, and Squeeze signals."""
    if config is None: config = TTM_CONFIG
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # --- Main Chart: Price, BB, KC ---
    ax1.plot(df.index, df['Close'], label='Close', color='black', alpha=0.6)
    
    # BB
    ax1.plot(df.index, df['BB_Upper'], label='BB Upper', color='blue', linestyle='--', alpha=0.7)
    ax1.plot(df.index, df['BB_Lower'], label='BB Lower', color='blue', linestyle='--', alpha=0.7)
    ax1.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], color='blue', alpha=0.05)
    
    # KC
    ax1.plot(df.index, df['KC_Upper'], label='KC Upper', color='orange', linestyle='-', alpha=0.7)
    ax1.plot(df.index, df['KC_Lower'], label='KC Lower', color='orange', linestyle='-', alpha=0.7)
    
    # Plot Buy/Sell Signals
    buy_signals = df[df['Signal'] == 'Buy Signal']
    sell_signals = df[df['Signal'] == 'Sell Signal']
    
    if not buy_signals.empty:
        ax1.scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', s=100, label='Buy Signal', zorder=5)
    if not sell_signals.empty:
        ax1.scatter(sell_signals.index, sell_signals['Close'], color='red', marker='v', s=100, label='Sell Signal', zorder=5)
        
    ax1.set_title(f'{ticker} - TTM Squeeze Analysis')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # --- Sub Chart: Squeeze Status ---
    # Plot "dots" on a line. Red = Squeeze On, Green = Squeeze Off
    
    # Create a series for colors
    colors = np.where(df['Squeeze_On'], 'red', 'green')
    
    # We plot a scatter of dots at y=0
    ax2.scatter(df.index, np.zeros(len(df)), c=colors, s=30, marker='o')
    
    # Add legend manually for dots
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='Squeeze On (Consolidation)', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Squeeze Off (Normal)', markersize=8)
    ]
    ax2.legend(handles=legend_elements, loc='upper left')
    
    ax2.set_yticks([])  # Hide y-axis ticks
    ax2.set_ylabel('Squeeze Status')
    ax2.set_title('Squeeze Indicator (Red = Squeeze On)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def run_analysis(ticker, show_plot=True, config=None):
    """Main execution function."""
    # Set backend if not showing plot
    if not show_plot:
        matplotlib.use('Agg')
        
    if config is None: config = TTM_CONFIG.copy()
    else:
        temp = TTM_CONFIG.copy()
        temp.update(config)
        config = temp
        
    try:
        df = fetch_data(ticker, config)
        df = calculate_indicators(df, config)
        
        fig = plot_analysis(df, ticker, config)
        
        if show_plot:
            plt.show()
            
        # Extract signals
        recent_signals = []
        # Check last 60 days
        subset = df.iloc[-60:]
        for date, row in subset.iterrows():
            if row['Signal']:
                recent_signals.append({
                    'Date': date,
                    'Type': row['Signal'],
                    'Price': row['Close']
                })
            elif row['Squeeze_On']:
                 # Optional: report active squeeze
                 pass
                 
        current_squeeze = df['Squeeze_On'].iloc[-1]
        
        return {
            'success': True,
            'ticker': ticker,
            'current_squeeze': bool(current_squeeze),
            'signals': recent_signals,
            'figure': fig,
            'dataframe': df
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    ticker = "AXISBANK.NS"
    print(f"Running TTM Squeeze Analysis for {ticker}...")
    results = run_analysis(ticker, show_plot=True)
    
    if results['success']:
        print(f"Current Squeeze Status: {'ON' if results['current_squeeze'] else 'OFF'}")
        if results['signals']:
            print("Recent Signals:")
            for sig in results['signals']:
                print(f"  {sig['Date'].date()}: {sig['Type']} @ {sig['Price']:.2f}")
        else:
            print("No recent buy/sell signals.")
