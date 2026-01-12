"""
VOLUME ANALYSIS TOOL
=====================

PURPOSE:
--------
This module analyzes Volume-Price relationships to identify:
- Buying/Selling Exhaustion (Divergences)
- Climax Volume (Churning)
- Distribution Days

It is designed to be used as a backend service for the stock research dashboard.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
import traceback
import io
import base64

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'ORDER': 5,            # Sensitivity for peak detection
    'LOOKBACK_PERIODS': 365,
    'DEFAULT_TICKER': 'ICICIBANK.NS'
}

# ==========================================
# DATA FETCHING
# ==========================================
def fetch_data(ticker, interval='1d', lookback_periods=365):
    """
    Fetches historical data for the given ticker.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'ICICIBANK.NS').
        interval (str): Data interval (default '1d'). Supports '1d', '1wk', '15m'.
        lookback_periods (int): Number of periods to fetch (default 365).
                                Note: Automatically adjusted for '15m' interval to max 59 days.

    Returns:
        pd.DataFrame: DataFrame containing 'Open', 'High', 'Low', 'Close', 'Volume'.
                      Columns are flattened if MultiIndex.
    
    Raises:
        ValueError: If no data is found for the ticker.
    """
    #i am hard coding the lookback_periods to 1 year for now. There is no point doing more than that.
    #because we are looking for peaks and troughs, and we need enough data to find them.
    lookback_periods=365

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_periods)
    
    # 15m restriction logic 
    if interval == '15m':
        lookback_periods = min(lookback_periods, 59)
        start_date = end_date - timedelta(days=lookback_periods)
    
    print(f"Fetching data for {ticker} from {start_date.date()} to {end_date.date()}...")
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, 
                     progress=False, auto_adjust=False, multi_level_index=False)
    
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    return df

# ==========================================
# CALCULATION LOGIC
# ==========================================
def detect_divergences(df, order=5):
    """
    Identifies volume-price divergences and institutional activity signals.

    Methodology:
    1. Buying/Selling Exhaustion:
       - Uses local peaks/troughs of 'Close' price (defined by 'order' parameter).
       - Compares consecutive peaks (for Bearish) or troughs (for Bullish).
       - Bearish Divergence (Buying Exhaustion): Price makes Higher High, but Volume makes Lower High.
       - Bullish Divergence (Selling Exhaustion): Price makes Lower Low, but Volume makes Lower Low.
       
    2. Climax Volume (Churning):
       - Volume > 2x 20-day Volume MA.
       - Small candle body (< 25% of daily range).
       - Indicates high activity with little price progress (often smart money selling into strength).
       
    3. Distribution Days:
       - Price closes lower than previous day.
       - Volume > 20-day Volume MA.
       - Price drop is controlled (>-1.5%), indicating systematic unloading rather than panic selling.

    Args:
        df (pd.DataFrame): DataFrame with 'Close', 'Open', 'High', 'Low', 'Volume'.
        order (int): The comparison window for peak/trough detection (default 5).

    Returns:
        list: A list of dictionaries, each representing a detected signal with:
              - 'Type': Signal description.
              - 'Date': Timestamp of the signal.
              - 'Price': Price level at signal.
              - 'Details': Textual explanation of the signal.
    """
    divergences = []
    
    # --- 1. Technical Calculations ---
    # Volume MA (20-day)
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
    
    # Candle metrics
    df['Body_Size'] = abs(df['Close'] - df['Open'])
    df['Range'] = df['High'] - df['Low']
    # Avoid division by zero
    df['Body_Pct_Range'] = np.where(df['Range'] > 0, df['Body_Size'] / df['Range'], 0)
    
    # --- 2. Advanced Volume Signals ---
    
    # A) Climax Volume (Churning)
    # Condition: Volume > 2.0x Volume MA AND Body < 25% of Range
    # We also check if it's near a high (optional, but churning usually implies top)
    # Using fillna(False) to handle initial period where MA is NaN
    df['Climax_Churn'] = ((df['Volume'] > 2.0 * df['Volume_MA_20']) & (df['Body_Pct_Range'] < 0.25)).fillna(False)
    
    # B) Distribution Days
    # Condition: Red Day, Volume > Volume MA, Falls < 1.5%
    prev_close = df['Close'].shift(1)
    df['Distribution_Day'] = ((df['Close'] < prev_close) & \
                             (df['Volume'] > 1.0 * df['Volume_MA_20']) & \
                             ((df['Close'] - prev_close) / prev_close > -0.015)).fillna(False)

    # Collect these signals for plotting/reporting
    for date, row in df.iterrows():
        if row['Climax_Churn']:
            divergences.append({
                'Type': 'Climax Volume (Churning)',
                'Date': date,
                'Price': row['High'], # Plot above marker
                'Details': f"Vol: {row['Volume']:,.0f} ( > 2x MA), Small Body"
            })
            
        if row['Distribution_Day']:
            pct_change = 0
            prior = df.loc[:date].iloc[-2]['Close'] if len(df.loc[:date]) > 1 else row['Close']
            if prior != 0:
                pct_change = (row['Close'] - prior) / prior
                
            divergences.append({
                'Type': 'Distribution Day',
                'Date': date,
                'Price': row['High'],
                'Details': f"Vol: {row['Volume']:,.0f} (> Avg), Price Change: {pct_change:.2%}" 
            })

    # --- 3. Peak/Trough Divergences ---
    price_values = df['Close'].values
    peak_indices = argrelextrema(price_values, np.greater_equal, order=order)[0]
    trough_indices = argrelextrema(price_values, np.less_equal, order=order)[0]
    
    # Mark them in the dataframe for plotting
    df['price_peak'] = np.nan
    df['price_trough'] = np.nan
    df.iloc[peak_indices, df.columns.get_loc('price_peak')] = df.iloc[peak_indices]['Close']
    df.iloc[trough_indices, df.columns.get_loc('price_trough')] = df.iloc[trough_indices]['Close']
    
    # Check for Bearish Divergence (Price Higher High, Volume Lower)
    for i in range(1, len(peak_indices)):
        prev_idx = peak_indices[i-1]
        curr_idx = peak_indices[i]
        
        prev_price = df.iloc[prev_idx]['Close']
        curr_price = df.iloc[curr_idx]['Close']
        
        prev_vol = df.iloc[prev_idx]['Volume']
        curr_vol = df.iloc[curr_idx]['Volume']
        
        if curr_price > prev_price and curr_vol < prev_vol:
            divergences.append({
                'Type': 'Buying Exhaustion (Bearish Div)',
                'Date': df.index[curr_idx],
                'Price': curr_price,
                'Details': f"Price HH ({prev_price:.2f} -> {curr_price:.2f}), Volume ↓ ({prev_vol:,.0f} -> {curr_vol:,.0f})"
            })

    # Check for Bullish Divergence (Price Lower Low, Volume Lower)
    for i in range(1, len(trough_indices)):
        prev_idx = trough_indices[i-1]
        curr_idx = trough_indices[i]
        
        prev_price = df.iloc[prev_idx]['Close']
        curr_price = df.iloc[curr_idx]['Close']
        
        prev_vol = df.iloc[prev_idx]['Volume']
        curr_vol = df.iloc[curr_idx]['Volume']
        
        if curr_price < prev_price and curr_vol < prev_vol:
            divergences.append({
                'Type': 'Selling Exhaustion (Bullish Div)',
                'Date': df.index[curr_idx],
                'Price': curr_price,
                'Details': f"Price LL ({prev_price:.2f} -> {curr_price:.2f}), Volume ↓ ({prev_vol:,.0f} -> {curr_vol:,.0f})"
            })
            
    return divergences

# ==========================================
# PLOTTING
# ==========================================
def plot_results(df, ticker, divergences, show_plot=True, return_figure=False):
    """
    Generates a matplotlib chart visualizing Price, Volume, and Signals.

    Features:
    - Subplot 1 (Price): 
      - Close Price line.
      - Peaks (Orange v) and Troughs (Cyan ^).
      - Signal Markers: 
        - Climax Churn: Purple 'X'
        - Distribution: Maroon '.'
        - Buying Exhaustion: Red 'v'
        - Selling Exhaustion: Green '^'
    - Subplot 2 (Volume):
      - Volume bars (Green for up days, Red for down days).
      - 20-period Volume Moving Average (Orange line).

    Args:
        df (pd.DataFrame): The data to plot.
        ticker (str): Ticker symbol for title.
        divergences (list): List of signal dictionaries to overlay.
        show_plot (bool): If True, calls plt.show().
        return_figure (bool): If True, returns the Figure object (for PDF reporting).

    Returns:
        str or matplotlib.figure.Figure: 
            - Base64 encoded PNG string (if return_figure=False).
            - Figure object (if return_figure=True or show_plot=True).
    """
    # Determine backend if not showing plot
    if not show_plot:
        matplotlib.use('Agg')
        
    fig = plt.figure(figsize=(14, 10))
    
    # Subplot 1: Price
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.5)
    
    # Plot Peaks/Troughs
    peaks = df[df['price_peak'].notna()]
    troughs = df[df['price_trough'].notna()]
    ax1.scatter(peaks.index, peaks['Close'], color='orange', marker='v', s=40, alpha=0.5, label='Peaks')
    ax1.scatter(troughs.index, troughs['Close'], color='cyan', marker='^', s=40, alpha=0.5, label='Troughs')
    
    # Plot Divergence Markers
    for div in divergences:
        marker_size = 150
        linewidth = 1.5
        
        if 'Climax Volume' in div['Type']:
            color = 'purple'
            marker = 'X'
            y_pos = div['Price'] * 1.01
            label = 'Climax Churn'
        elif 'Distribution Day' in div['Type']:
            color = 'maroon'
            marker = '.'
            y_pos = div['Price'] * 1.005
            marker_size = 50
            label = 'Distribution'
            linewidth = 0.5
        elif 'Buying Exhaustion' in div['Type']:
            color = 'red'
            marker = 'v'
            y_pos = div['Price'] * 1.005
            label = 'Buying Exhaustion'
        else: # Selling Exhaustion
            color = 'green'
            marker = '^'
            y_pos = div['Price'] * 0.995
            label = 'Selling Exhaustion'
            
        ax1.scatter(div['Date'], y_pos, 
                    color=color, marker=marker, s=marker_size, 
                    label=label, zorder=10, 
                    edgecolors='black', linewidths=linewidth)
        
    ax1.set_title(f'{ticker} - Volume Analysis', fontweight='bold')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.2)
    
    # Subplot 2: Volume
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    # Define colors for volume bars
    colors = np.where(df['Close'] >= df['Open'], 'green', 'red')
    ax2.bar(df.index, df['Volume'], color=colors, alpha=0.4, label='Volume')
    
    # Plot Volume MA if available
    if 'Volume_MA_20' in df.columns:
        ax2.plot(df.index, df['Volume_MA_20'], color='orange', linewidth=1, label='Vol MA (20)')
        
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.2)
    ax2.legend(loc='upper right')
    
    # Clean up legend on ax1
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='best', fontsize='small')
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
        return fig
        
    if return_figure:
        return fig
        
    # Return base64 string
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig) 
    return img_str

# ==========================================
# MAIN API
# ==========================================
def run_analysis(ticker=None, show_plot=True, config=None, df=None, return_figure=False):
    """
    Orchestrates the full Volume Analysis pipeline.

    Steps:
    1. Fetches data (if df not provided).
    2. Runs `detect_divergences` to find signals.
    3. Calls `plot_results` to generate visualization.
    4. Formats results for JSON response.

    Args:
        ticker (str): Ticker symbol.
        show_plot (bool): Whether to display the plot interactively.
        config (dict): Optional configuration overrides (e.g., {'ORDER': 10}).
        df (pd.DataFrame, optional): Pre-fetched DataFrame to reuse.
        return_figure (bool): If True, returns Figure object instead of base64 string.

    Returns:
        dict: Analysis results containing:
              - 'success': bool
              - 'divergences': List of detected signals (JSON serializable).
              - 'chart_image': Base64 PNG string (or None if return_figure=True).
              - 'figure': Matplotlib Figure object (if return_figure=True).
    """
    current_config = CONFIG.copy()
    if config:
        current_config.update(config)
        
    if ticker is None:
        ticker = current_config['DEFAULT_TICKER']
        
    try:
        # 1. Fetch Data
        if df is None:
            df = fetch_data(ticker, lookback_periods=current_config['LOOKBACK_PERIODS'])
        else:
            df = df.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            
        # 2. Calculate
        divergences = detect_divergences(df, order=current_config['ORDER'])
        
        # 3. Plot
        chart_output = plot_results(df, ticker, divergences, show_plot=show_plot, return_figure=return_figure)
        
        # Format divergences for JSON serialization (convert Timestamps to strings)
        serializable_divs = []
        latest_signal = "Neutral"

        if divergences:
             # Sort by date
            divergences.sort(key=lambda x: x['Date'])
            
            # Get latest signal if recent
            last_date = divergences[-1]['Date']
            if (df.index[-1] - last_date).days <= 5:
                latest_signal = divergences[-1]['Type']

            for div in divergences:
                serializable_divs.append({
                    'Date': div['Date'].strftime('%Y-%m-%d'),
                    'Price': float(div['Price']),
                    'Type': div['Type'],
                    'Details': div['Details']
                })
        
        result = {
            'success': True,
            'ticker': ticker,
            'divergences': serializable_divs,
            'latest_signal': latest_signal
        }
        
        if return_figure:
            result['figure'] = chart_output
            result['chart_image'] = None
        else:
             result['chart_image'] = chart_output if not show_plot else None
             
        return result
        
    except Exception as e:
        traceback.print_exc()
        return {
            'success': False,
            'ticker': ticker,
            'error': str(e)
        }

if __name__ == "__main__":
    result = run_analysis('ICICIBANK.NS', show_plot=True)
