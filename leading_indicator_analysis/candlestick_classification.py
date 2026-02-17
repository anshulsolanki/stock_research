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
Candlestick Classification System

This script classifies candlesticks (daily/weekly) into 5 categories:
1. Seller Strong Control (-2): Large bearish body, small wicks, close near low, high volume
2. Seller Control (-1): Moderate bearish candle or strong candle with low volume
3. No Control (0): Small body, long wicks, indecision patterns
4. Buyer Control (+1): Moderate bullish candle or strong candle with low volume
5. Buyer Strong Control (+2): Large bullish body, small wicks, close near high, high volume

Classification Algorithm:
- Body size normalized by ATR (volatility adjustment)
- Close position shows conviction
- Wicks show rejections
- Volume confirms/weakens the signal
"""

import sys
import os
import argparse
import io
import base64
import traceback
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from datetime import datetime, timedelta
import yfinance as yf

# ==========================================
# CONFIGURATION
# ==========================================

CONFIG = {
    'DEFAULT_TICKER': 'DABUR.NS',
    'DEFAULT_DAILY_LOOKBACK': 180,
    'DEFAULT_WEEKLY_LOOKBACK': 3 * 365,  # ~3 years
    'ATR_PERIOD': 14,
    'VOLUME_MA_PERIOD': 20,
    'STRONG_BODY_THRESHOLD': 1.5,
    'MODERATE_BODY_THRESHOLD': 0.8,
    'WEAK_BODY_THRESHOLD': 0.5,
    'SMALL_WICK_THRESHOLD': 0.2,
    'LARGE_WICK_THRESHOLD': 0.4,
    'CLOSE_NEAR_HIGH': 0.7,
    'CLOSE_NEAR_LOW': 0.3,
    'HIGH_VOLUME_THRESHOLD': 1.3,
    'LOW_VOLUME_THRESHOLD': 0.8,
    'CHART_CANDLES': 180
}

# ==========================================
# DATA FETCHING
# ==========================================

def fetch_data(ticker, interval='1d', lookback_days=180):
    """
    Fetch historical stock data.
    
    Args:
        ticker (str): Stock ticker symbol
        interval (str): '1d' for daily, '1wk' for weekly
        lookback_days (int): Days of history to fetch
        
    Returns:
        pd.DataFrame: OHLCV data with DatetimeIndex
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days + 30)  # Extra buffer
    
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
    
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
    
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    return df

# ==========================================
# INDICATOR CALCULATION
# ==========================================

def calculate_indicators(df, config=None):
    """
    Calculate ATR and Volume MA.
    
    Args:
        df (pd.DataFrame): OHLCV data
        config (dict): Configuration overrides
        
    Returns:
        pd.DataFrame: Data with ATR and Volume_MA columns
    """
    cfg = CONFIG.copy()
    if config:
        cfg.update(config)
    
    # ATR calculation
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=cfg['ATR_PERIOD']).mean()
    
    # Volume MA
    df['Volume_MA'] = df['Volume'].rolling(window=cfg['VOLUME_MA_PERIOD']).mean()
    
    # Drop NaN rows
    df_clean = df.dropna()
    
    return df_clean

def calculate_candlestick_metrics(df):
    """
    Calculate candlestick metrics for classification.
    
    Returns:
        pd.DataFrame: Data with metrics columns
    """
    df['Body_Size'] = np.abs(df['Close'] - df['Open'])
    df['Total_Range'] = df['High'] - df['Low']
    df['Upper_Wick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['Lower_Wick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    # Normalized metrics
    df['Body_Strength'] = df['Body_Size'] / df['ATR']
    df['Upper_Wick_Percent'] = df['Upper_Wick'] / df['Total_Range']
    df['Lower_Wick_Percent'] = df['Lower_Wick'] / df['Total_Range']
    df['Close_Position'] = (df['Close'] - df['Low']) / df['Total_Range']
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # Direction flags
    df['Is_Bullish'] = df['Close'] > df['Open']
    df['Is_Bearish'] = df['Close'] < df['Open']
    
    return df

# ==========================================
# CLASSIFICATION LOGIC
# ==========================================

def classify_candlestick(row, config=None):
    """
    Classify a single candlestick into 5 categories.
    
    Returns:
        tuple: (final_score: int, category_name: str)
    """
    cfg = CONFIG.copy()
    if config:
        cfg.update(config)
    
    body_strength = row['Body_Strength']
    close_pos = row['Close_Position']
    upper_wick = row['Upper_Wick_Percent']
    lower_wick = row['Lower_Wick_Percent']
    vol_ratio = row['Volume_Ratio']
    is_bullish = row['Is_Bullish']
    is_bearish = row['Is_Bearish']
    
    score = 0
    
    # Step 1: Base score from body strength
    if body_strength > cfg['STRONG_BODY_THRESHOLD']:
        score = 2 if is_bullish else -2 if is_bearish else 0
    elif body_strength > cfg['MODERATE_BODY_THRESHOLD']:
        score = 1 if is_bullish else -1 if is_bearish else 0
    elif body_strength < cfg['WEAK_BODY_THRESHOLD']:
        score = 0
    else:
        score = 0.5 if is_bullish else -0.5 if is_bearish else 0
    
    # Step 2: Close position adjustment
    if score > 0:
        if close_pos < cfg['CLOSE_NEAR_LOW']:
            score = max(score - 1, 0)
        elif close_pos < 0.5:
            score = max(score - 0.5, 0)
    elif score < 0:
        if close_pos > cfg['CLOSE_NEAR_HIGH']:
            score = min(score + 1, 0)
        elif close_pos > 0.5:
            score = min(score + 0.5, 0)
    
    # Step 3: Wick analysis
    if lower_wick > cfg['LARGE_WICK_THRESHOLD']:
        score += 0.5 if score >= 0 else 0.3
    if upper_wick > cfg['LARGE_WICK_THRESHOLD']:
        score -= 0.5 if score <= 0 else 0.3
    if upper_wick > cfg['LARGE_WICK_THRESHOLD'] and lower_wick > cfg['LARGE_WICK_THRESHOLD']:
        score = score * 0.3
    if upper_wick < cfg['SMALL_WICK_THRESHOLD'] and lower_wick < cfg['SMALL_WICK_THRESHOLD']:
        if abs(score) > 0.5:
            score = score * 1.2
    
    # Step 4: Volume multiplier
    if vol_ratio > cfg['HIGH_VOLUME_THRESHOLD']:
        score = score * 1.2
    elif vol_ratio < cfg['LOW_VOLUME_THRESHOLD']:
        score = score * 0.6
    
    # Step 5: Map to categories
    score = max(-2, min(2, score))
    
    if score >= 1.5:
        return 2, "Buyer Strong Control"
    elif score >= 0.5:
        return 1, "Buyer Control"
    elif score > -0.5:
        return 0, "No Control"
    elif score > -1.5:
        return -1, "Seller Control"
    else:
        return -2, "Seller Strong Control"

def apply_classification(df, config=None):
    """Apply classification to all candlesticks."""
    classifications = df.apply(lambda row: classify_candlestick(row, config), axis=1)
    df['Classification_Score'] = [c[0] for c in classifications]
    df['Classification'] = [c[1] for c in classifications]
    return df

# ==========================================
# PLOTTING
# ==========================================

def plot_results(df, ticker, show_plot=True, return_figure=False, config=None):
    """
    Create candlestick chart with classification-based coloring.
    
    Args:
        df (pd.DataFrame): Data with classifications
        ticker (str): Ticker symbol
        show_plot (bool): Whether to display interactively
        return_figure (bool): If True, return Figure object; else base64 string
        config (dict): Configuration overrides
        
    Returns:
        str or matplotlib.figure.Figure: Base64 PNG or Figure object
    """
    cfg = CONFIG.copy()
    if config:
        cfg.update(config)
    
    if not show_plot:
        matplotlib.use('Agg')
    
    plt.close('all')
    
    # Prepare chart data
    num_candles = min(cfg['CHART_CANDLES'], len(df))
    df_chart = df.tail(num_candles).copy().reset_index()
    
    # Color mapping
    color_map = {
        'Seller Strong Control': '#8B0000',
        'Seller Control': '#FF6B6B',
        'No Control': '#808080',
        'Buyer Control': '#90EE90',
        'Buyer Strong Control': '#006400'
    }
    df_chart['Color'] = df_chart['Classification'].map(color_map)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                     gridspec_kw={'height_ratios': [3, 1]},
                                     sharex=True)
    
    # Plot candlesticks
    candle_width = 0.6
    for idx, row in df_chart.iterrows():
        x = idx
        open_price = row['Open']
        high = row['High']
        low = row['Low']
        close = row['Close']
        color = row['Color']
        
        # Wick
        ax1.plot([x, x], [low, high], color='black', linewidth=0.5, zorder=1)
        
        # Body
        body_height = close - open_price
        body_bottom = min(open_price, close)
        rect = plt.Rectangle((x - candle_width/2, body_bottom), 
                             candle_width, abs(body_height),
                             facecolor=color, edgecolor='black', 
                             linewidth=0.5, zorder=2)
        ax1.add_patch(rect)
    
    # Format price chart
    ax1.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax1.set_title(f'{ticker} - Candlestick Classification', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(-0.5, len(df_chart) - 0.5)
    
    # Legend
    legend_elements = [
        Patch(facecolor='#8B0000', edgecolor='black', label='Seller Strong Control'),
        Patch(facecolor='#FF6B6B', edgecolor='black', label='Seller Control'),
        Patch(facecolor='#808080', edgecolor='black', label='No Control'),
        Patch(facecolor='#90EE90', edgecolor='black', label='Buyer Control'),
        Patch(facecolor='#006400', edgecolor='black', label='Buyer Strong Control')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)
    
    # Volume bars
    for idx, row in df_chart.iterrows():
        ax2.bar(idx, row['Volume'], width=0.8, color=row['Color'], 
                edgecolor='black', linewidth=0.3, alpha=0.7)
    
    ax2.set_ylabel('Volume', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(-0.5, len(df_chart) - 0.5)
    
    # X-axis labels
    if 'Date' in df_chart.columns:
        date_labels = df_chart['Date'].dt.strftime('%Y-%m-%d')
        tick_spacing = max(1, len(df_chart) // 10)
        tick_positions = range(0, len(df_chart), tick_spacing)
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels([date_labels.iloc[i] for i in tick_positions], 
                            rotation=45, ha='right')
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
        return fig
    
    if return_figure:
        return fig
    
    # Return base64 string
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

# ==========================================
# MAIN API
# ==========================================

def run_analysis(ticker=None, show_plot=True, config=None, df=None, 
                 return_figure=False, timeframe='daily', lookback=None):
    """
    Run complete candlestick classification analysis.
    
    Args:
        ticker (str): Stock ticker symbol
        show_plot (bool): Whether to display plot
        config (dict): Configuration overrides
        df (pd.DataFrame): Pre-fetched data (optional)
        return_figure (bool): If True, return Figure; else base64 PNG
        timeframe (str): 'daily' or 'weekly'
        lookback (int): Days of historical data (None = use defaults)
        
    Returns:
        dict: Analysis results with keys:
              - success (bool)
              - ticker (str)
              - chart_image (str or None)
              - figure (Figure or None)
              - classification_summary (dict)
              - latest_classification (str)
    """
    current_config = CONFIG.copy()
    if config:
        current_config.update(config)
    
    if ticker is None:
        ticker = current_config['DEFAULT_TICKER']
    
    try:
        # Fetch data
        # Fetch data
        if df is None:
            interval = '1wk' if timeframe == 'weekly' else '1d'
            if lookback is None:
                lookback = current_config['DEFAULT_WEEKLY_LOOKBACK'] if timeframe == 'weekly' else current_config['DEFAULT_DAILY_LOOKBACK']
            
            df = fetch_data(ticker, interval=interval, lookback_days=lookback)
        else:
            df = df.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Ensure we only have the required columns to avoid issues with dropna() 
            # if the passed df has extra columns from other analyses
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            # Check if all columns exist
            if all(col in df.columns for col in required_cols):
                 df = df[required_cols]
        
        # Calculate indicators and classify
        
        # DEBUG: Check input DF
        # print(f"DEBUG: Input DF shape: {df.shape}")
        # print(f"DEBUG: Input DF columns: {df.columns}")
        
        df = calculate_indicators(df, current_config)
        
        if df.empty:
            return {
                'success': False,
                'ticker': ticker,
                'error': f"Insufficient data. Input shape was {df.shape} (after indicators). Requires > {current_config['VOLUME_MA_PERIOD']} periods."
            }
            
        df = calculate_candlestick_metrics(df)
        df = apply_classification(df, current_config)
        
        # Generate chart
        chart_output = plot_results(df, ticker, show_plot=show_plot, 
                                    return_figure=return_figure, config=current_config)
        
        # Summary statistics
        if df.empty:
             return {
                'success': False,
                'ticker': ticker,
                'error': "No classification data available"
            }
            
        recent_scores = df['Classification_Score'].tail(10)
        latest_class = df['Classification'].iloc[-1]
        
        classification_counts = df['Classification'].value_counts().to_dict()
        
        result = {
            'success': True,
            'ticker': ticker,
            'latest_classification': latest_class,
            'recent_trend_score': float(recent_scores.mean()),
            'classification_distribution': classification_counts
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

# ==========================================
# CLI ENTRY POINT
# ==========================================

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description='Candlestick Classification System')
    parser.add_argument('ticker', type=str, nargs='?', default=CONFIG['DEFAULT_TICKER'],
                        help=f'Stock ticker (default: {CONFIG["DEFAULT_TICKER"]})')
    parser.add_argument('--timeframe', type=str, choices=['daily', 'weekly'], default='daily',
                        help='Timeframe: daily or weekly (default: daily)')
    parser.add_argument('--lookback', type=int, 
                        help='Lookback period in days (default: 180 for daily, 1095 for weekly)')
    
    args = parser.parse_args()
    
    ticker = args.ticker
    if '.' not in ticker and '^' not in ticker:
        ticker = f"{ticker.upper()}.NS"
    else:
        ticker = ticker.upper()
    
    print(f"\nRunning Candlestick Classification for {ticker} ({args.timeframe})\n")
    
    result = run_analysis(ticker=ticker, show_plot=True, timeframe=args.timeframe, 
                         lookback=args.lookback, return_figure=False)
    
    if result['success']:
        print(f"\n✓ Analysis complete!")
        print(f"  Latest Classification: {result['latest_classification']}")
        print(f"  Recent Trend Score (10-day avg): {result['recent_trend_score']:.2f}")
        print(f"\n  Classification Distribution:")
        for category, count in result['classification_distribution'].items():
            print(f"    {category}: {count}")
    else:
        print(f"\n✗ Analysis failed: {result['error']}")

if __name__ == "__main__":
    main()
