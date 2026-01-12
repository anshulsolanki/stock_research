"""
MULTI-TIMEFRAME ANALYSIS TOOL
=============================

PURPOSE:
--------
This module implements a comprehensive multi-timeframe analysis strategy that combines
Trend-Following (Supertrend) and Momentum (MACD) indicators across three distinct time horizons:
1. Weekly (Long-term Trend)
2. Daily (Medium-term Swing)
3. 15-Minute (Short-term Entry/Exit)

WHAT IT DOES:
-------------
1. **Centralized Data Fetching**:
   - Downloads historical data for all three timeframes in a single execution.
   - Optimizes API usage by fetching 10 years (Weekly), 2 years (Daily), and 60 days (15m).

2. **Composite Technical Analysis**:
   - **Supertrend**: Identifies the primary trend direction and support/resistance levels.
     - Weekly: Smoother trend (Period 7, Multiplier 2.0)
     - Daily: Standard swing (Period 14, Multiplier 3.0)
     - 15m: Intraday precision (Period 21, Multiplier 3.0)
   - **MACD**: Gauge momentum and potential reversals.
     - Analyzes Trend (Bullish/Bearish), Momentum (Strengthening/Weakening), and Crossover Signals.

3. **Visual Analysis**:
   - Generates a 3-pane candlestick chart (Weekly, Daily, 15m).
   - Overlays 20, 50, and 200 EMAs for dynamic support/resistance.
   - Uses Zerodha-style Red/Green candlesticks for clear readability.

METHODOLOGY:
------------
The "Triple Screen" inspired approach:
1. **Weekly Chart**: Defines the "Tide" (Long-term direction). Trade in this direction.
2. **Daily Chart**: Defines the "Wave" (Intermediate trend). Look for pullbacks here.
3. **15-Minute Chart**: Defines the "Ripple" (Entry timing). Fine-tune entry price.

CONFIGURATION:
--------------
Default parameters are tuned for Indian Stock Market (NSE) volatility:
- **Supertrend**:
  - Weekly: 7, 2.0 (Faster reaction to long-term shifts)
  - Daily: 14, 3.0 (Standard noise filtering)
  - 15m: 21, 3.0 (High noise filtering for intraday)
- **MACD**: Standard 12, 26, 9 settings for all timeframes.

USAGE:
------
Run as standalone script:
    python multi_timeframe_analysis.py

Or import and use programmatic API:
    from lagging_indicator_analysis import multi_timeframe_analysis
    results = multi_timeframe_analysis.run_analysis(ticker="RELIANCE.NS")

OUTPUT:
-------
Returns a dictionary containing:
- 'success': bool
- 'ticker': str
- 'supertrend_results': List of dicts (Timeframe, Status, Value, Signal Date)
- 'macd_results': List of dicts (Timeframe, Trend, Momentum, Signal)
- 'figure': Matplotlib figure object containing the 3-timeframe charts

DEPENDENCIES:
-------------
- supertrend_analysis.py
- macd_analysis.py
- pandas, numpy, matplotlib, yfinance
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
import yfinance as yf
import traceback

# Use Agg backend by default for non-interactive environments (web/batch)
# Interactive mode (main) can override this.
matplotlib.use('Agg')

# ==========================================
# PATH SETUP
# ==========================================
# Add project root to path to allow imports from other directories
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import Analysis Modules
try:
    import supertrend_analysis
    import macd_analysis
except ImportError:
    try:
        from lagging_indicator_analysis import supertrend_analysis
        from lagging_indicator_analysis import macd_analysis
    except ImportError:
        print("Error: Could not import analysis modules (supertrend_analysis, macd_analysis).")
        supertrend_analysis = None
        macd_analysis = None

# ==========================================
# CONFIGURATION
# ==========================================

# Base Configuration for Supertrend
SUPERTREND_CONFIG = {
    'PERIOD': 14,
    'MULTIPLIER': 3.0,
    'LOOKBACK_PERIODS': 730,  # 2 years default
}

# Timeframe-Specific Parameters
SUPERTREND_PARAMS = {
    '1wk': {'PERIOD': 7, 'MULTIPLIER': 2.0},
    '1d': {'PERIOD': 14, 'MULTIPLIER': 3.0},
    '15m': {'PERIOD': 21, 'MULTIPLIER': 3.0}
}

# Base Configuration for MACD
MACD_CONFIG = {
    'FAST': 12,
    'SLOW': 26,
    'SIGNAL': 9,
    'LOOKBACK_PERIODS': 730,
}

DEFAULT_TICKER = 'DABUR.NS'

# Timeframe Definitions
TIMEFRAMES = [
    # 1 Week: 10 years lookback to calculate 200 Weekly EMA correctly
    {'label': '1 Week', 'interval': '1wk', 'lookback': 3650}, 
    # 1 Day: 2 years lookback
    {'label': '1 Day', 'interval': '1d', 'lookback': 730},
    # 15 Min: max 60 days (Yahoo Finance limit)
    {'label': '15 Min', 'interval': '15m', 'lookback': 59}
]

# ==========================================
# DATA FETCHING
# ==========================================

def fetch_data_helper(ticker, interval, lookback_days):
    """
    Helper function to fetch data for a specific interval and lookback.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    # Enforce strict limit for 15m to avoid API errors
    if interval == '15m':
        actual_lookback = min(lookback_days, 59)
        start_date = end_date - timedelta(days=actual_lookback)
        
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, 
                     progress=False, auto_adjust=False, multi_level_index=False)
    
    if df.empty:
        return None
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    return df

def fetch_all_data(ticker):
    """
    Fetches data for all configured timeframes efficiently.
    Returns a dictionary: {interval: dataframe}
    """
    print(f"Fetching data for {ticker} across all timeframes...")
    data_map = {}
    
    for tf in TIMEFRAMES:
        interval = tf['interval']
        lookback = tf['lookback']
        label = tf['label']
        
        try:
            df = fetch_data_helper(ticker, interval, lookback)
            if df is not None and not df.empty:
                data_map[interval] = df
                print(f"  ✓ Fetched {label}: {len(df)} candles")
            else:
                print(f"  ✗ Failed to fetch {label}: Empty Data")
                data_map[interval] = None
        except Exception as e:
            print(f"  ✗ Error fetching {label}: {e}")
            data_map[interval] = None
            
    return data_map

# ==========================================
# CHARTING FUNCTIONS
# ==========================================

def plot_candlestick_on_ax(ax, df, title):
    """
    Plots Zerodha-style candlesticks and EMAs on a given matplotlib axes.
    """
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No Data Available", ha='center', va='center')
        ax.set_title(title, fontweight='bold')
        return

    # Colors
    col_up = 'green'
    col_down = 'red'
    width = 0.6
    
    # Calculate EMAs
    ema20 = df['Close'].ewm(span=20, adjust=False).mean()
    ema50 = df['Close'].ewm(span=50, adjust=False).mean()
    ema200 = df['Close'].ewm(span=200, adjust=False).mean()
    
    # Plot EMAs
    ax.plot(range(len(df)), ema20, color='green', linewidth=1.0, label='20 EMA', alpha=0.9)
    ax.plot(range(len(df)), ema50, color='blue', linewidth=1.0, label='50 EMA', alpha=0.9)
    ax.plot(range(len(df)), ema200, color='red', linewidth=1.0, label='200 EMA', alpha=0.9)
    
    # Plot Candlesticks
    # Using range(len(df)) for x-axis to avoid gaps for non-trading days
    for i in range(len(df)):
        open_p = df['Open'].iloc[i]
        close_p = df['Close'].iloc[i]
        high_p = df['High'].iloc[i]
        low_p = df['Low'].iloc[i]
        
        # Color Determination
        color = col_up if close_p >= open_p else col_down
        
        # High-Low Line (Wick)
        ax.plot([i, i], [low_p, high_p], color=color, linewidth=1)
        
        # Open-Close Rectangle (Body)
        body_height = abs(close_p - open_p)
        body_bottom = min(open_p, close_p)
        
        # Handle Doji (Open roughly equals Close)
        if body_height == 0:
            body_height = (high_p - low_p) * 0.01 
            if body_height == 0: body_height = 0.01
            
        rect = Rectangle((i - width/2, body_bottom), width, body_height,
                         facecolor=color, edgecolor=color)
        ax.add_patch(rect)
        
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize='small')
    
    # Format X-axis Dates
    # Show ~10 evenly spaced ticks
    step = max(1, len(df) // 10)
    tick_pos = list(range(0, len(df), step))
    tick_labels = [df.index[p].strftime('%Y-%m-%d') for p in tick_pos]
    
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=30, ha='right', fontsize=8)
    ax.set_xlim(-1, len(df))

def generate_candlestick_figure(ticker, data_map):
    """
    Generates the composite 3-pane candlestick chart figure.
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    fig.subplots_adjust(hspace=0.4)
    
    for i, tf in enumerate(TIMEFRAMES):
        interval = tf['interval']
        label = tf['label']
        original_df = data_map.get(interval)
        
        try:
            if original_df is not None and not original_df.empty:
                df = original_df.copy()
                plot_candlestick_on_ax(axes[i], df, f"{ticker} - {label} (Candlestick + EMAs)")
            else:
                axes[i].text(0.5, 0.5, f"No Data for {label}", ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f"{ticker} - {label}", fontweight='bold')
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error: {e}", ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f"{ticker} - {label}", fontweight='bold')
        
    plt.tight_layout()
    return fig

# ==========================================
# ANALYSIS FUNCTIONS
# ==========================================

def get_supertrend_results(ticker, data_map):
    """
    Runs Supertrend analysis on all timeframes returning structured results.
    """
    results_list = []
    
    for tf in TIMEFRAMES:
        interval = tf['interval']
        lookback = tf['lookback']
        original_df = data_map.get(interval)
        
        if original_df is None:
            results_list.append({
                'timeframe': tf['label'],
                'status': "Data Missing",
                'supertrend_value': None,
                'signal_date': None,
                'last_price': None
            })
            continue

        df_copy = original_df.copy()
        
        # Prepare Config
        run_config = SUPERTREND_CONFIG.copy()
        run_config['INTERVAL'] = interval
        run_config['LOOKBACK_PERIODS'] = lookback
        if interval in SUPERTREND_PARAMS:
            run_config.update(SUPERTREND_PARAMS[interval])
            
        try:
            if supertrend_analysis:
                result = supertrend_analysis.run_analysis(ticker=ticker, show_plot=False, config=run_config, df=df_copy)
                
                if result['success']:
                    # Important: Close figure to free memory and prevent display issues
                    plt.close(result['figure'])
                    
                    results_list.append({
                        'timeframe': tf['label'],
                        'status': result['status'],
                        'supertrend_value': round(result['supertrend_value'], 2),
                        'signal_date': result['signal_date'].strftime('%Y-%m-%d') if result['signal_date'] else "N/A",
                        'last_price': round(result['last_price'], 2)
                    })
                else:
                    results_list.append({
                        'timeframe': tf['label'],
                        'status': f"Error: {result.get('error')}",
                        'supertrend_value': None,
                        'signal_date': None,
                        'last_price': None
                    })
            else:
                results_list.append({'timeframe': tf['label'], 'status': "Module Missing", 'supertrend_value': None})
                
        except Exception as e:
            results_list.append({
                'timeframe': tf['label'],
                'status': f"Exception: {str(e)}",
                'supertrend_value': None,
                'signal_date': None,
                'last_price': None
            })
    
    return results_list

def get_macd_results(ticker, data_map):
    """
    Runs MACD analysis on all timeframes returning structured results.
    """
    results_list = []
    
    for tf in TIMEFRAMES:
        interval = tf['interval']
        original_df = data_map.get(interval)
        
        if original_df is None:
            results_list.append({
                'timeframe': tf['label'],
                'trend': "Data Missing",
                'momentum': None,
                'signal': None
            })
            continue
             
        df_copy = original_df.copy()
        
        # Prepare Config
        run_config = MACD_CONFIG.copy()
        run_config['INTERVAL'] = tf['interval']
        run_config['LOOKBACK_PERIODS'] = tf['lookback']
        
        try:
            if macd_analysis:
                result = macd_analysis.run_analysis(ticker=ticker, show_plot=False, config=run_config, df=df_copy)
                
                if result['success']:
                    plt.close(result['figure'])
                    signal = result['crossover_signal'] if result['crossover_signal'] else "No Signal"
                    
                    results_list.append({
                        'timeframe': tf['label'],
                        'trend': result['trend'],
                        'momentum': result['momentum'],
                        'signal': signal
                    })
                else:
                    results_list.append({
                        'timeframe': tf['label'],
                        'trend': f"Error: {result.get('error')}",
                        'momentum': None,
                        'signal': None
                    })
            else:
                results_list.append({'timeframe': tf['label'], 'trend': "Module Missing", 'momentum': None, 'signal': None})

        except Exception as e:
            results_list.append({
                'timeframe': tf['label'],
                'trend': f"Exception: {str(e)}",
                'momentum': None,
                'signal': None
            })

    return results_list

def run_analysis(ticker, show_plot=False, config=None):
    """
    Main entry point for programmatic execution (Web App / Report Generator).
    
    Args:
        ticker (str): Stock ticker symbol.
        show_plot (bool): Whether to display the plot (interactive mode).
        config (dict): Optional configuration override.
        
    Returns:
        dict: comprehensive analysis results.
    """
    try:
        # 1. Fetch data for all timeframes
        data_map = fetch_all_data(ticker)
        
        # 2. Get Analysis Results
        supertrend_results = get_supertrend_results(ticker, data_map)
        macd_results = get_macd_results(ticker, data_map)
        
        # 3. Generate Chart
        fig = generate_candlestick_figure(ticker, data_map)
        
        if show_plot:
            # Note: For web apps, show_plot is usually False
            # For local execution, we might want to switch backend or just show
            try:
                plt.show()
            except:
                pass
        
        return {
            'success': True,
            'ticker': ticker,
            'supertrend_results': supertrend_results,
            'macd_results': macd_results,
            'figure': fig
        }
        
    except Exception as e:
        print(f"Error in multi-timeframe analysis: {e}")
        traceback.print_exc()
        return {
            'success': False,
            'ticker': ticker,
            'error': str(e)
        }

# ==========================================
# MAIN EXECUTION (CLI)
# ==========================================

def print_cli_results(results):
    """
    Helper to pretty-print results to the console.
    """
    print("\n" + "="*80)
    print(f"MULTI-TIMEFRAME ANALYSIS REPORT: {results['ticker']}")
    print("="*80)
    
    # Supertrend Table
    print("\n[SUPERTREND ANALYSIS]")
    st_df = pd.DataFrame(results['supertrend_results'])
    if not st_df.empty:
        # Reorder columns for readability
        cols = ['timeframe', 'status', 'supertrend_value', 'signal_date']
        print(st_df[cols].to_string(index=False))
    else:
        print("No results.")
        
    # MACD Table
    print("\n[MACD ANALYSIS]")
    macd_df = pd.DataFrame(results['macd_results'])
    if not macd_df.empty:
        print(macd_df.to_string(index=False))
    else:
        print("No results.")
    
    print("\n" + "-"*80)
    print("Chart generated successfully.")

if __name__ == "__main__":
    # Force interactive backend for standalone run
    try:
        matplotlib.use('MacOSX')
    except ImportError:
        try:
            matplotlib.use('TkAgg')
        except:
            pass # Keep default if specific backends fail
            
    # Simple CLI argument parsing
    target_ticker = DEFAULT_TICKER
    if len(sys.argv) > 1:
        target_ticker = sys.argv[1]
        
    print(f"Starting Multi-Timeframe Analysis for {target_ticker}...")
    
    # Run Analysis
    results = run_analysis(target_ticker, show_plot=True)
    
    if results['success']:
        print_cli_results(results)
    else:
        print(f"Analysis Failed: {results.get('error')}")
