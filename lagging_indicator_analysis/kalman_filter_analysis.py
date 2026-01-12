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
KALMAN FILTER ANALYSIS TOOL
============================

PURPOSE:
--------
This module implements an Adaptive Kalman Filter for financial time series analysis.
The Kalman Filter is a recursive state-space model that excels at:
- Filtering market noise from price data
- Providing zero-lag trend analysis
- Generating clear trading signals (BUY/SELL/HOLD)
- Adapting to different market conditions (Breakout vs Trend Following)

WHAT IT DOES:
-------------
1. **Kalman Filter Application**: Uses pykalman library to apply recursive filtering
   - State Model: Treats "true price" as a hidden state
   - Prediction-Update Loop: Continuously refines price estimates
   
2. **Parameter Optimization**: Automatic grid search to find optimal Q and R values
   - Q (Process Noise): Controls filter adaptability/responsiveness
   - R (Measurement Noise): Controls filter smoothness
   
3. **Dual Operational Modes**:
   - Breakout Detection: High Q, Low R (responsive to trend changes)
   - Trend Following: Low Q, High R (smooth trend continuation)
   
4. **Trading Signals**: Generates actionable signals based on:
   - Current price vs Kalman filtered price
   - Recent trend direction
   - Mode-specific logic

5. **Visual Analysis**: Generates charts with:
   - Actual price (dotted black line)
   - Kalman filtered price (yellow line)
   - EMAs (20, 50, 200)
   - Trading signal annotation box

KEY METRICS:
------------
- Kalman Q: Process noise covariance (adaptability)
- Kalman R: Measurement noise covariance (smoothness)
- Trading Signal: BUY, SELL, or HOLD
- MSE: Mean squared error between actual and filtered prices
- Last Price: Latest closing price
- Kalman Value: Latest filtered price estimate

CONFIGURATION:
--------------
Default parameters (customizable via KALMAN_CONFIG):
- MODE: 'Trend_Following' or 'Breakout_Detection'
- Q_STEPS: 20 (grid search resolution for Q)
- R_STEPS: 20 (grid search resolution for R)
- Q_RANGE: (1e-6, 0.1) (search space for process noise)
- R_RANGE: (0.01, 100) (search space for measurement noise)
- OPTIMIZE: True (auto-optimize parameters)
- INTERVAL: '1d' (daily candles)
- LOOKBACK_PERIODS: 730 days (2 years history)

USAGE:
------
Run as standalone script:
    python kalman_filter_analysis.py

Or import and use programmatically:
    from kalman_filter_analysis import run_analysis
    results = run_analysis(ticker="MARICO.NS", config={'MODE': 'Breakout_Detection'})

OUTPUT:
-------
Returns dictionary containing:
- success: bool
- ticker: str
- mode: str (Breakout_Detection or Trend_Following)
- kalman_Q: float (optimal process noise)
- kalman_R: float (optimal measurement noise)
- trading_signal: str (BUY/SELL/HOLD)
- last_price: float
- kalman_value: float (filtered price)
- last_date: datetime
- mse: float
- figure: matplotlib.figure.Figure
- dataframe: pd.DataFrame (with all calculations)

TYPICAL USE CASES:
------------------
1. Trend following with noise reduction
2. Early breakout detection
3. Dynamic support/resistance identification
4. Multi-timeframe signal confirmation
5. Portfolio filtering based on signal direction
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuration
KALMAN_CONFIG = {
    # Operational Mode: Determines filtering behavior
    # 'Breakout_Detection': High Q, Low R - Responsive to trend changes
    # 'Trend_Following': Low Q, High R - Smooth trend continuation
    'MODE': 'Trend_Following',
    
    # Grid Search Parameters
    'Q_STEPS': 20,              # Number of grid points for Q parameter
    'R_STEPS': 20,              # Number of grid points for R parameter
    'Q_RANGE': (1e-6, 0.1),    # Process noise covariance search range
    'R_RANGE': (0.01, 100),     # Measurement noise covariance search range
    'OPTIMIZE': True,           # Auto-optimize Q and R via grid search
    
    # Signal Generation Parameters
    'SIGNAL_LOOKBACK': 10,      # Number of periods to look back for trend detection
    
    # Data Fetching Parameters
    'INTERVAL': '1d',
    'LOOKBACK_PERIODS': 730,    # 2 years
    
    # Execution Parameters
    'DEFAULT_TICKER': 'MARICO.NS',
    'BATCH_RELATIVE_PATH': '../data/tickers_list.json',
    'RUN_BATCH': False
}

def fetch_data(ticker, interval, lookback_periods):
    """
    Fetches historical data for the given ticker.
    
    Args:
        ticker: Stock ticker symbol
        interval: Data interval ('1d', '1wk', '1h', '15m')
        lookback_periods: Number of days to look back
    
    Returns:
        pd.DataFrame: Historical OHLCV data
    """
    # Limit lookback for 15m interval
    if interval == '15m':
        lookback_periods = min(lookback_periods, 59)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_periods)
    
    print(f"Fetching data for {ticker} from {start_date.date()} to {end_date.date()}...")
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, 
                     progress=False, auto_adjust=False, multi_level_index=False)
    
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
    
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    return df

def apply_kalman_filter(df, Q, R):
    """
    Apply Kalman Filter to price data.
    
    Args:
        df: DataFrame with price data
        Q: Process noise covariance (adaptability)
        R: Measurement noise covariance (smoothness)
    
    Returns:
        np.ndarray: Filtered state means (Kalman prices)
    """
    prices = df['Close'].values.reshape(-1, 1)
    
    # Initialize Kalman Filter
    kf = KalmanFilter(
        transition_matrices=np.array([[1]]),      # F: Random Walk model
        observation_matrices=np.array([[1]]),     # H: Direct price mapping
        initial_state_mean=prices[0],
        initial_state_covariance=1.0,
        transition_covariance=Q * np.eye(1),
        observation_covariance=R * np.eye(1)
    )
    
    # Apply filter
    filtered_state_means, _ = kf.filter(prices)
    
    return filtered_state_means.flatten()

def optimize_parameters(df, config):
    """
    Perform grid search to find optimal Q and R parameters.
    
    Args:
        df: DataFrame with price data
        config: Configuration dictionary
    
    Returns:
        tuple: (optimal_Q, optimal_R, best_mse)
    """
    prices = df['Close'].values.reshape(-1, 1)
    q_min, q_max = config.get('Q_RANGE', KALMAN_CONFIG['Q_RANGE'])
    r_min, r_max = config.get('R_RANGE', KALMAN_CONFIG['R_RANGE'])
    q_steps = config.get('Q_STEPS', KALMAN_CONFIG['Q_STEPS'])
    r_steps = config.get('R_STEPS', KALMAN_CONFIG['R_STEPS'])
    
    # Create logarithmic search space
    q_values = np.logspace(np.log10(q_min), np.log10(q_max), q_steps)
    r_values = np.logspace(np.log10(r_min), np.log10(r_max), r_steps)
    
    print(f"\\nOptimizing Kalman parameters via Grid Search...")
    print(f"  Q range: {q_min:.2e} to {q_max:.2e} ({q_steps} steps)")
    print(f"  R range: {r_min:.2e} to {r_max:.2e} ({r_steps} steps)")
    
    best_mse = float('inf')
    best_Q = None
    best_R = None
    
    total = len(q_values) * len(r_values)
    iteration = 0
    
    for Q in q_values:
        for R in r_values:
            iteration += 1
            try:
                filtered = apply_kalman_filter(df, Q, R)
                mse = mean_squared_error(prices.flatten(), filtered)
                
                if mse < best_mse:
                    best_mse = mse
                    best_Q = Q
                    best_R = R
                
                if iteration % 50 == 0:
                    progress = (iteration / total) * 100
                    print(f"  Progress: {progress:.1f}% | Best MSE: {best_mse:.6f}")
            
            except Exception:
                continue
    
    print(f"✓ Optimization complete: Q={best_Q:.6e}, R={best_R:.6e}, MSE={best_mse:.6f}")
    
    return best_Q, best_R, best_mse

def calculate_emas(df):
    """
    Calculate Exponential Moving Averages.
    
    Args:
        df: DataFrame with Close prices
    
    Returns:
        tuple: (ema_20, ema_50, ema_200)
    """
    close = df['Close']
    ema_20 = close.ewm(span=20, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()
    ema_200 = close.ewm(span=200, adjust=False).mean()
    
    return ema_20, ema_50, ema_200

def get_trading_signal(df, kalman_prices, mode, config=None):
    """
    Generate trading signal based on Kalman filter and mode.
    
    Args:
        df: DataFrame with price data
        kalman_prices: Kalman filtered prices
        mode: 'Breakout_Detection' or 'Trend_Following'
        config: Configuration dictionary
    
    Returns:
        str: 'BUY', 'SELL', or 'HOLD'
    """
    cfg = config if config else KALMAN_CONFIG
    signal_lookback = cfg.get('SIGNAL_LOOKBACK', 10)
    lookback = min(signal_lookback, len(kalman_prices))
    recent_trend = kalman_prices[-1] - kalman_prices[-lookback]
    current_price = df['Close'].iloc[-1]
    filtered_price = kalman_prices[-1]
    
    if mode == 'Breakout_Detection':
        # Breakout mode: price breaking above/below filter
        if current_price > filtered_price and recent_trend > 0:
            return 'BUY'
        elif current_price < filtered_price and recent_trend < 0:
            return 'SELL'
        else:
            return 'HOLD'
    else:
        # Trend following: stay with the trend
        if recent_trend > 0:
            return 'BUY'
        elif recent_trend < 0:
            return 'SELL'
        else:
            return 'HOLD'

def plot_kalman(df, kalman_prices, ticker, trading_signal, config, show_plot=True):
    """
    Plot Kalman Filter analysis with price, EMAs, and trading signal.
    
    Args:
        df: DataFrame with price data
        kalman_prices: Kalman filtered prices
        ticker: Stock ticker symbol
        trading_signal: Trading signal string
        config: Configuration dictionary
        show_plot: If True, display plot (not applicable with Agg backend)
    
    Returns:
        matplotlib.figure.Figure: Plot figure object
    """
    mode = config.get('MODE', KALMAN_CONFIG['MODE'])
    Q = df.attrs.get('kalman_Q', 0)
    R = df.attrs.get('kalman_R', 0)
    
    # Calculate EMAs
    ema_20, ema_50, ema_200 = calculate_emas(df)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))
    dates = df.index
    
    # Plot EMAs first (bottom layer)
    ax.plot(dates, ema_200, label='EMA 200', linewidth=2, color='red', alpha=0.7, zorder=1)
    ax.plot(dates, ema_50, label='EMA 50', linewidth=2, color='blue', alpha=0.7, zorder=2)
    ax.plot(dates, ema_20, label='EMA 20', linewidth=2, color='green', alpha=0.7, zorder=3)
    
    # Plot Kalman filter
    ax.plot(dates, kalman_prices, label='Kalman Filter', 
            linewidth=2.5, color='yellow', alpha=0.9, zorder=4)
    
    # Plot actual price on top
    ax.plot(dates, df['Close'], label='Actual Price', 
            alpha=0.9, linewidth=2, color='black', linestyle=':', zorder=5)
    
    # Add trading signal annotation
    signal_colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'orange'}
    signal_color = signal_colors.get(trading_signal, 'black')
    
    ax.text(0.02, 0.98, f'Trading Signal: {trading_signal}', 
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor=signal_color, alpha=0.3, edgecolor=signal_color, linewidth=2))
    
    # Add parameter info
    param_text = f'Mode: {mode}\\nQ={Q:.2e}, R={R:.2e}'
    ax.text(0.02, 0.88, param_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Formatting
    ax.set_title(f'{ticker} - Kalman Filter Analysis', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    return fig

def run_analysis(ticker=None, show_plot=True, config=None, df=None):
    """
    Main analysis function for Kalman Filter.
    
    Args:
        ticker: Stock ticker symbol (if None, uses DEFAULT_TICKER)
        show_plot: If True, displays plot (Agg backend doesn't show, but flag maintained for API compatibility)
        config: Optional dictionary to override default configuration
        df: Optional[pd.DataFrame], pre-loaded DataFrame to use instead of fetching data
    
    Returns:
        dict: Analysis results containing:
            - 'success': bool, whether analysis succeeded
            - 'ticker': str, analyzed ticker
            - 'mode': str, operational mode
            - 'kalman_Q': float, optimal Q parameter
            - 'kalman_R': float, optimal R parameter
            - 'trading_signal': str, BUY/SELL/HOLD
            - 'last_price': float, latest closing price
            - 'kalman_value': float, latest Kalman filtered price
            - 'last_date': datetime, date of latest data
            - 'mse': float, mean squared error
            - 'figure': matplotlib.figure.Figure, plot object
            - 'dataframe': pd.DataFrame, full analysis data
            - 'error': str, error message if failed
    """
    # Merge provided config with defaults
    current_config = KALMAN_CONFIG.copy()
    if config:
        current_config.update(config)
    
    if ticker is None:
        ticker = current_config['DEFAULT_TICKER']
    
    try:
        # Fetch data
        if df is None:
            df = fetch_data(ticker, 
                           interval=current_config['INTERVAL'],
                           lookback_periods=current_config['LOOKBACK_PERIODS'])
        # else: use provided df
        
        # Optimize or use default parameters
        if current_config.get('OPTIMIZE', True):
            Q, R, mse = optimize_parameters(df, current_config)
        else:
            Q = current_config.get('Q', 0.01)
            R = current_config.get('R', 1.0)
            kalman_prices = apply_kalman_filter(df, Q, R)
            mse = mean_squared_error(df['Close'].values, kalman_prices)
        
        # Apply Kalman filter with optimal parameters
        kalman_prices = apply_kalman_filter(df, Q, R)
        
        # Store parameters in dataframe attributes
        df.attrs['kalman_Q'] = Q
        df.attrs['kalman_R'] = R
        df['Kalman'] = kalman_prices
        
        # Generate trading signal
        mode = current_config['MODE']
        trading_signal = get_trading_signal(df, kalman_prices, mode, current_config)
        
        # Extract final values
        last_price = float(df['Close'].iloc[-1])
        kalman_value = float(kalman_prices[-1])
        last_date = df.index[-1]
        
        # Generate plot
        fig = plot_kalman(df, kalman_prices, ticker, trading_signal, 
                         current_config, show_plot=show_plot)
        
        return {
            'success': True,
            'ticker': ticker,
            'mode': mode,
            'kalman_Q': Q,
            'kalman_R': R,
            'trading_signal': trading_signal,
            'last_price': last_price,
            'kalman_value': kalman_value,
            'last_date': last_date,
            'mse': mse,
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
    Analyze multiple tickers from JSON file.
    
    Args:
        json_file: Path to JSON file containing ticker list
    """
    import os
    import json
    
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found.")
        return
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract unique tickers
    tickers = list(set(data.values()))
    
    print(f"Found {len(tickers)} unique tickers. Starting Kalman Filter analysis...\\n")
    
    for ticker in tickers:
        print(f"\\n{'='*70}")
        print(f"Analyzing {ticker}...")
        print('='*70)
        
        results = run_analysis(ticker=ticker, show_plot=False)
        
        if results['success']:
            print(f"  Mode: {results['mode']}")
            print(f"  Trading Signal: {results['trading_signal']}")
            print(f"  Last Price: {results['last_price']:.2f}")
            print(f"  Kalman Value: {results['kalman_value']:.2f}")
            print(f"  Q: {results['kalman_Q']:.6e}")
            print(f"  R: {results['kalman_R']:.6e}")
            print(f"  MSE: {results['mse']:.6f}")
            print(f"  Date: {results['last_date'].date()}")
        else:
            print(f"  Error: {results['error']}")

if __name__ == "__main__":
    import os
    
    # Load execution parameters from config
    run_batch = KALMAN_CONFIG['RUN_BATCH']
    batch_relative_path = KALMAN_CONFIG['BATCH_RELATIVE_PATH']
    ticker = KALMAN_CONFIG['DEFAULT_TICKER']
    
    # Resolve batch file path
    batch_file = os.path.join(os.path.dirname(__file__), batch_relative_path)
    
    if run_batch and os.path.exists(batch_file):
        analyze_batch(batch_file)
    else:
        # Run dual-mode analysis
        print(f"\n{'='*70}")
        print(f" KALMAN FILTER ANALYSIS - {ticker}")
        print('='*70)
        
        # ========== MODE 1: BREAKOUT DETECTION ==========
        print("\n📊 MODE 1: BREAKOUT DETECTION")
        print("-" * 70)
        
        results_breakout = run_analysis(
            ticker=ticker,
            show_plot=False,
            config={'MODE': 'Breakout_Detection'}
        )
        
        if results_breakout['success']:
            # Save plot
            plot_filename = f"kalman_{ticker.replace('.', '_')}_Breakout_Detection.png"
            results_breakout['figure'].savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close(results_breakout['figure'])
            print(f"✓ Plot saved: {plot_filename}")
            
            print(f"  Trading Signal: {results_breakout['trading_signal']}")
            print(f"  Last Price: {results_breakout['last_price']:.2f}")
            print(f"  Kalman Value: {results_breakout['kalman_value']:.2f}")
            print(f"  Q: {results_breakout['kalman_Q']:.6e}")
            print(f"  R: {results_breakout['kalman_R']:.6e}")
            print(f"  MSE: {results_breakout['mse']:.6f}")
        else:
            print(f"  Error: {results_breakout['error']}")
        
        # ========== MODE 2: TREND FOLLOWING ==========
        print("\n\n📊 MODE 2: TREND FOLLOWING")
        print("-" * 70)
        
        results_trend = run_analysis(
            ticker=ticker,
            show_plot=False,
            config={'MODE': 'Trend_Following'}
        )
        
        if results_trend['success']:
            # Save plot
            plot_filename = f"kalman_{ticker.replace('.', '_')}_Trend_Following.png"
            results_trend['figure'].savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close(results_trend['figure'])
            print(f"✓ Plot saved: {plot_filename}")
            
            print(f"  Trading Signal: {results_trend['trading_signal']}")
            print(f"  Last Price: {results_trend['last_price']:.2f}")
            print(f"  Kalman Value: {results_trend['kalman_value']:.2f}")
            print(f"  Q: {results_trend['kalman_Q']:.6e}")
            print(f"  R: {results_trend['kalman_R']:.6e}")
            print(f"  MSE: {results_trend['mse']:.6f}")
        else:
            print(f"  Error: {results_trend['error']}")
        
        # ========== COMPARISON ==========
        if results_breakout['success'] and results_trend['success']:
            print("\n\n📈 MODE COMPARISON")
            print("=" * 70)
            print(f"{'Metric':<30} {'Breakout Detection':<20} {'Trend Following':<20}")
            print("-" * 70)
            print(f"{'Trading Signal':<30} {results_breakout['trading_signal']:<20} {results_trend['trading_signal']:<20}")
            print(f"{'Kalman Q':<30} {results_breakout['kalman_Q']:<20.6e} {results_trend['kalman_Q']:<20.6e}")
            print(f"{'Kalman R':<30} {results_breakout['kalman_R']:<20.6e} {results_trend['kalman_R']:<20.6e}")
            print(f"{'MSE':<30} {results_breakout['mse']:<20.6f} {results_trend['mse']:<20.6f}")
            print(f"{'Kalman Value':<30} {results_breakout['kalman_value']:<20.2f} {results_trend['kalman_value']:<20.2f}")
            print("=" * 70)
            print(f"\nDate: {results_trend['last_date'].date()}")
            print('='*70)
