import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('MacOSX', force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
import yfinance as yf

# Add project root to path to allow imports from other directories
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from lagging_indicator_analysis import supertrend_analysis
    from lagging_indicator_analysis import macd_analysis
except ImportError:
    # If running from project root
    try:
        from lagging_indicator_analysis import supertrend_analysis
        from lagging_indicator_analysis import macd_analysis
    except ImportError:
        print("Error: Could not import analysis modules. Please run from project root or experiments_space.")
        sys.exit(1)

# ==========================================
# CONFIGURATION
# ==========================================

# Supertrend Configuration - Default/Base (will be overridden)
SUPERTREND_CONFIG = {
    'PERIOD': 14,
    'MULTIPLIER': 3.0,
    'LOOKBACK_PERIODS': 730,  # 2 year
}

# Specific configs per timeframe
SUPERTREND_PARAMS = {
    '1wk': {'PERIOD': 7, 'MULTIPLIER': 2.0},
    '1d': {'PERIOD': 14, 'MULTIPLIER': 3.0},
    '15m': {'PERIOD': 21, 'MULTIPLIER': 3.0}
}

# MACD Configuration
MACD_CONFIG = {
    'FAST': 12,
    'SLOW': 26,
    'SIGNAL': 9,
    'LOOKBACK_PERIODS': 730,  # 2 year
}

# Common Configuration
DEFAULT_TICKER = 'DABUR.NS'
TIMEFRAMES = [
    # 10 years for weekly to ensure enough data for 200 EMA (200 weeks ~ 4 years, taking 10y to be safe)
    {'label': '1 Week', 'interval': '1wk', 'lookback': 3650}, 
    # 2 years for daily is sufficient for 200 EMA (200 days)
    {'label': '1 Day', 'interval': '1d', 'lookback': 730},
    # 60 days max for 15m (Yahoo limit)
    {'label': '15 Min', 'interval': '15m', 'lookback': 59}
]

# ==========================================
# EXPERIMENT FUNCTIONS
# ==========================================

def fetch_all_data(ticker):
    """
    Fetches data for all configured timeframes once.
    Returns a dictionary: {interval: dataframe}
    """
    print(f"Fetching all data for {ticker}...")
    data_map = {}
    
    for tf in TIMEFRAMES:
        interval = tf['interval']
        lookback = tf['lookback']
        label = tf['label']
        
        try:
            df = fetch_data_helper(ticker, interval, lookback)
            if df is not None and not df.empty:
                data_map[interval] = df
                print(f"  ✓ Fetched {label} data ({len(df)} candles)")
            else:
                print(f"  ✗ Failed to fetch {label} data (Empty)")
                data_map[interval] = None
        except Exception as e:
            print(f"  ✗ Error fetching {label} data: {e}")
            data_map[interval] = None
            
    return data_map


def run_supertrend_experiment(ticker, data_map):
    print(f"\nrunning supertrend_analysis.py 3 times with 1 week , 1 day , 15 min setting for {ticker}")
    print("-" * 80)
    
    results_list = []
    
    for tf in TIMEFRAMES:
        interval = tf['interval']
        lookback = tf['lookback']
        
        # Get pre-fetched data
        original_df = data_map.get(interval)
        
        if original_df is None:
             results_list.append({
                'Time period': tf['label'],
                'Status': "Data Missing",
                'Supertrend value': None,
                'Signal date': None
            })
             continue

        # Create a copy because analysis modifies the dataframe
        df_copy = original_df.copy()
        
        # Update config for this run
        run_config = SUPERTREND_CONFIG.copy()
        run_config['INTERVAL'] = interval
        run_config['LOOKBACK_PERIODS'] = lookback
        
        # Apply specific params if available
        if interval in SUPERTREND_PARAMS:
            run_config.update(SUPERTREND_PARAMS[interval])
            
        print(f"Running for {tf['label']} with Period={run_config['PERIOD']}, Multiplier={run_config['MULTIPLIER']}")
        
        try:
            # Run analysis
            # Pass the dataframe directly
            result = supertrend_analysis.run_analysis(ticker=ticker, show_plot=False, config=run_config, df=df_copy)
            
            if result['success']:
                # Close the figure to prevent it from showing up later with plt.show()
                plt.close(result['figure'])
                
                results_list.append({
                    'Time period': tf['label'],
                    'Status': result['status'],
                    'Supertrend value': result['supertrend_value'],
                    'Signal date': result['signal_date'].date()
                })
            else:
                 results_list.append({
                    'Time period': tf['label'],
                    'Status': f"Error: {result.get('error')}",
                    'Supertrend value': None,
                    'Signal date': None
                })
                
        except Exception as e:
             results_list.append({
                    'Time period': tf['label'],
                    'Status': f"Exception: {str(e)}",
                    'Supertrend value': None,
                    'Signal date': None
                })

    # Create DataFrame for display
    df_results = pd.DataFrame(results_list)
    
    # Print formatted table
    # Using to_string for nice alignment
    if not df_results.empty:
        print(df_results.to_string(index=False))
    else:
        print("No results generated.")


def run_macd_experiment(ticker, data_map):
    print(f"\n\nrunning macd_analysis.py 3 times with 1 week , 1 day , 15 min setting for {ticker}")
    print("-" * 80)
    
    results_list = []
    
    for tf in TIMEFRAMES:
        # Update config for this run
        run_config = MACD_CONFIG.copy()
        run_config['INTERVAL'] = tf['interval']
        run_config['LOOKBACK_PERIODS'] = tf['lookback']
        
        interval = tf['interval']
        original_df = data_map.get(interval)
        
        if original_df is None:
             results_list.append({
                'Time period': tf['label'],
                'Trend': "Data Missing",
                'Momentum': None,
                'Signal': None
            })
             continue
             
        df_copy = original_df.copy()
        
        try:
            # Run analysis
            result = macd_analysis.run_analysis(ticker=ticker, show_plot=False, config=run_config, df=df_copy)
            
            if result['success']:
                # Close the figure to prevent it from showing up later with plt.show()
                plt.close(result['figure'])
                
                # Determine Signal column
                signal = result['crossover_signal'] if result['crossover_signal'] else "No Signal"
                
                results_list.append({
                    'Time period': tf['label'],
                    'Trend': result['trend'],
                    'Momentum': result['momentum'],
                    'Signal': signal
                })
            else:
                 results_list.append({
                    'Time period': tf['label'],
                    'Trend': f"Error: {result.get('error')}",
                    'Momentum': None,
                    'Signal': None
                })
                
        except Exception as e:
             results_list.append({
                    'Time period': tf['label'],
                    'Trend': f"Exception: {str(e)}",
                    'Momentum': None,
                    'Signal': None
                })

    # Create DataFrame for display
    df_results = pd.DataFrame(results_list)
    
    # Print formatted table
    if not df_results.empty:
        print(df_results.to_string(index=False))
    else:
        print("No results generated.")


def plot_candlestick_on_ax(ax, df, title):
    """
    Plots Zerodha-style candlesticks and EMAs on a given axes.
    """
    # Colors matching the request and Zerodha style
    col_up = 'green'
    col_down = 'red'
    width = 0.6
    width2 = 0.1
    
    # EMAs
    # 20 EMA - green, 50 EMA - blue , 200 - EMA red
    ema20 = df['Close'].ewm(span=20, adjust=False).mean()
    ema50 = df['Close'].ewm(span=50, adjust=False).mean()
    ema200 = df['Close'].ewm(span=200, adjust=False).mean()
    
    # Plot EMAs
    ax.plot(range(len(df)), ema20, color='green', linewidth=1.0, label='20 EMA', alpha=0.9)
    ax.plot(range(len(df)), ema50, color='blue', linewidth=1.0, label='50 EMA', alpha=0.9)
    ax.plot(range(len(df)), ema200, color='red', linewidth=1.0, label='200 EMA', alpha=0.9)
    
    # Plot Candlesticks
    for i in range(len(df)):
        open_p = df['Open'].iloc[i]
        close_p = df['Close'].iloc[i]
        high_p = df['High'].iloc[i]
        low_p = df['Low'].iloc[i]
        
        color = col_up if close_p >= open_p else col_down
        
        # High-Low Line
        ax.plot([i, i], [low_p, high_p], color=color, linewidth=1)
        
        # Open-Close Rectangle
        body_height = abs(close_p - open_p)
        body_bottom = min(open_p, close_p)
        
        # If open == close (doji), make it a thin line
        if body_height == 0:
            body_height = (high_p - low_p) * 0.01 # minimal height
            if body_height == 0: body_height = 0.01
            
        rect = Rectangle((i - width/2, body_bottom), width, body_height,
                         facecolor=color, edgecolor=color)
        ax.add_patch(rect)
        
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize='small')
    
    # Format X-axis
    # Show approx 10 dates
    step = max(1, len(df) // 10)
    tick_pos = list(range(0, len(df), step))
    # df.index is DatetimeIndex
    tick_labels = [df.index[p].strftime('%Y-%m-%d') for p in tick_pos]
    
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=30, ha='right', fontsize=8)
    ax.set_xlim(-1, len(df))


def fetch_data_helper(ticker, interval, lookback_days):
    """
    Simple helper to fetch data similar to analysis scripts.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    # 15m max 60 days restriction
    if interval == '15m':
        # Ensure we don't ask for more than 59 days for safe measure
        actual_lookback = min(lookback_days, 59)
        start_date = end_date - timedelta(days=actual_lookback)
        
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=False, multi_level_index=False)
    
    if df.empty:
        return None
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    return df


def run_candlestick_experiment(ticker, data_map):
    # Ensure interactive backend is active (recover from Agg if set by other modules)
    try:
        matplotlib.use('MacOSX', force=True)
    except:
        pass
        
    print(f"\n\nGenerating Candlestick Charts for {ticker} (1W, 1D, 15m)")
    print("-" * 80)
    
    # Prepare figure for 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    fig.subplots_adjust(hspace=0.4)
    
    # Iterate through timeframes
    # We use the same order as other experiments: 1 Week, 1 Day, 15 Min
    
    for i, tf in enumerate(TIMEFRAMES):
        interval = tf['interval']
        label = tf['label']
        
        # Data is already fetched
        original_df = data_map.get(interval)
        
        try:
            if original_df is not None and not original_df.empty:
                # Use a copy so we don't mess up if modifying for plotting
                df = original_df.copy()
                plot_candlestick_on_ax(axes[i], df, f"{ticker} - {label}")
            else:
                axes[i].text(0.5, 0.5, f"No Data for {label}", ha='center')
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error: {e}", ha='center')
            print(f"Error plotting {interval} data: {e}")
        
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ticker = DEFAULT_TICKER
    # You could add argparse here if needed to change ticker from command line
    
    # 1. Fetch data ONCE
    all_data = fetch_all_data(ticker)
    
    # 2. Run experiments with pre-fetched data
    run_supertrend_experiment(ticker, all_data)
    run_macd_experiment(ticker, all_data)
    run_candlestick_experiment(ticker, all_data)
