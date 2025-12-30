"""
Candlestick Chart Generator (Zerodha-style)
Creates candlestick charts with volume bars and volume profile
"""

import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ============================================================================
# CONFIGURATION
# ============================================================================

# Chart Colors
CONFIG_COLORS = {
    'bullish_candle': 'green',      # Color for bullish candles (close > open)
    'bearish_candle': 'red',        # Color for bearish candles (close < open)
    'ema_20': 'green',              # Color for 20-period EMA
    'ema_50': 'blue',               # Color for 50-period EMA
    'ema_200': 'red',               # Color for 200-period EMA
    'volume_profile': '#E8D5F2',    # Light purple for volume profile (use hex codes for lighter colors)
}

# EMA Settings
CONFIG_EMA = {
    'show_ema_20': True,            # Show 20-period EMA
    'show_ema_50': True,            # Show 50-period EMA
    'show_ema_200': True,           # Show 200-period EMA
    'ema_linewidth': 1.5,           # Line width for EMAs
    'ema_alpha': 0.8,               # Transparency (0.0 to 1.0)
}

# Volume Profile Settings
CONFIG_VOLUME_PROFILE = {
    'num_bins': 50,                 # Number of price levels for volume profile (20-100)
    'position': 'right',            # Position: 'left' or 'right'
    'alpha': 0.5,                   # Transparency (0.0 to 1.0, lower = lighter)
}

# Chart Size
CONFIG_CHART = {
    'figure_width': 16,             # Figure width in inches
    'figure_height': 10,            # Figure height in inches
    'candle_width': 0.8,            # Width of candlestick bodies (0.1 to 1.0)
    'volume_bar_width': 0.8,        # Width of volume bars (0.1 to 1.0)
}

# Grid Settings
CONFIG_GRID = {
    'show_grid': True,              # Show grid lines
    'grid_alpha': 0.3,              # Grid transparency (0.0 to 1.0)
}

# Volume Bar Settings
CONFIG_VOLUME_BAR = {
    'alpha': 0.7,                   # Transparency for volume bars (0.0 to 1.0)
}

# Psychological Pattern Settings
CONFIG_PATTERNS = {
    'show_patterns': True,
    
    # Thresholds - STRICT FILTERING
    'strength_body_multiplier': 2.5,     # Increased to 2.5x average body
    'strength_wick_tolerance': 0.2,      # Tigher wick tolerance (20%)
    'rejection_wick_ratio': 0.65,        # Stricter wick ratio (65% of range)
    'volume_multiplier': 1.5,            # New: Volume must be > 1.5x average
    'lookback_period': 10,               # New: Must be local high/low in last 10 candles
    
    'indecision_body_threshold': 0.6,    # Stricter: Body < 60% of average
    'indecision_consecutive': 4,         # Stricter: Need 4+ candles
    
    # Visuals
    'colors': {
        'strength_bullish': 'green',
        'strength_bearish': 'red',
        'control_shift_bullish': 'green',
        'control_shift_bearish': 'red',
        'indecision': 'gray',
        'text': 'black'
    },
    'style': {
        'arrow_props': dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
        'font_size_strength': 10,
        'font_size_rejection': 9,
        'font_size_indecision': 10,
        'ellipse_alpha': 0.2
    }
}

# ============================================================================
# END CONFIGURATION
# ============================================================================


def detect_psychological_patterns(df):
    """
    Identify psychological candlestick patterns:
    1. Strength Candles (Full Control)
    2. Control Shift Candles (Wick Rejection)
    3. Indecision Zones (Stalemate)
    
    Returns:
    - patterns: Dictionary of DataFrames/Indices for each pattern
    """
    df = df.copy()
    
    # Calculate basic metrics
    df['Body'] = abs(df['Close'] - df['Open'])
    df['Range'] = df['High'] - df['Low']
    df['UpperWick'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['LowerWick'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    # Average body size (smoothing over 20 periods)
    df['AvgBody'] = df['Body'].rolling(window=20).mean()
    # Average Volume (smoothing over 20 periods)
    df['AvgVol'] = df['Volume'].rolling(window=20).mean()
    
    # Calculate Local Highs and Lows for Context
    lookback = CONFIG_PATTERNS['lookback_period']
    # Rolling max of Highs (shift 1 to not include current candle yet, or include it)
    # We want to know if current candle is a local extreme relative to past N candles
    df['RollingMax'] = df['High'].rolling(window=lookback).max()
    df['RollingMin'] = df['Low'].rolling(window=lookback).min()
    
    patterns = {
        'strength': [],
        'control_shift': [],
        'indecision': []
    }
    
    # Iterate to find patterns
    indecision_streak = []
    
    # Keep track of last signal indices to prevent crowding
    last_strength_idx = -100
    last_rejection_idx = -100
    
    for i, (idx, row) in enumerate(df.iterrows()):
        # Skip if not enough history for averages
        if pd.isna(row['AvgBody']) or pd.isna(row['AvgVol']):
            continue
            
        current_vol = row['Volume']
        avg_vol = row['AvgVol']
        
        # 1. Strength Candle Detection
        # Criteria: 
        # - Large body (> 2.5x avg)
        # - Small wicks (< 20% of body)
        # - Volume Confirmation (> 1.5x avg volume)
        # - Deduplication: Don't mark if another strength candle was within last 5 days
        
        is_strength = False
        if (row['Body'] > row['AvgBody'] * CONFIG_PATTERNS['strength_body_multiplier']) and \
           (row['UpperWick'] < row['Body'] * CONFIG_PATTERNS['strength_wick_tolerance']) and \
           (row['LowerWick'] < row['Body'] * CONFIG_PATTERNS['strength_wick_tolerance']) and \
           (current_vol > avg_vol * CONFIG_PATTERNS['volume_multiplier']):
            
            # Check proximity
            if (i - last_strength_idx) > 5:
                patterns['strength'].append({
                    'index': i,
                    'type': 'bullish' if row['Close'] > row['Open'] else 'bearish',
                    'price': row['High'] if row['Close'] > row['Open'] else row['Low']
                })
                last_strength_idx = i
                is_strength = True

        # 2. Control Shift (Rejection) Detection
        # Criteria:
        # - Long wick (> 65% of range)
        # - Context: Must be at a Local High (for bearish rejection) or Local Low (for bullish)
        # - Deduplication: Don't mark if crowded
        
        if not is_strength and row['Range'] > 0:
            upper_ratio = row['UpperWick'] / row['Range']
            lower_ratio = row['LowerWick'] / row['Range']
            
            # Bearish Rejection (Upper Wick) -> Must be at Local High
            # We verify if Current High is close to the Rolling Max (within 1%)
            is_local_high = row['High'] >= row['RollingMax'] * 0.99
            
            if (upper_ratio > CONFIG_PATTERNS['rejection_wick_ratio']) and is_local_high:
                if (i - last_rejection_idx) > 5:
                    patterns['control_shift'].append({
                        'index': i,
                        'type': 'bearish_rejection',
                        'price': row['High']
                    })
                    last_rejection_idx = i
            
            # Bullish Rejection (Lower Wick) -> Must be at Local Low
            is_local_low = row['Low'] <= row['RollingMin'] * 1.01
            
            if (lower_ratio > CONFIG_PATTERNS['rejection_wick_ratio']) and is_local_low:
                if (i - last_rejection_idx) > 5:
                    patterns['control_shift'].append({
                        'index': i,
                        'type': 'bullish_rejection',
                        'price': row['Low']
                    })
                    last_rejection_idx = i
        
        # 3. Indecision Zone Detection
        # Series of small bodies (< 60% avg)
        if row['Body'] < row['AvgBody'] * CONFIG_PATTERNS['indecision_body_threshold']:
            indecision_streak.append(i)
        else:
            # End of streak, check if it qualifies (Need 4+ candles)
            if len(indecision_streak) >= CONFIG_PATTERNS['indecision_consecutive']:
                patterns['indecision'].append({
                    'start_idx': indecision_streak[0],
                    'end_idx': indecision_streak[-1],
                    'indices': list(indecision_streak)
                })
            indecision_streak = [] # Reset
            
    # Check for streak ending at the very last candle
    if len(indecision_streak) >= CONFIG_PATTERNS['indecision_consecutive']:
        patterns['indecision'].append({
            'start_idx': indecision_streak[0],
            'end_idx': indecision_streak[-1],
            'indices': list(indecision_streak)
        })
            
    return patterns


def fetch_stock_data(ticker, interval, period):
    """
    Fetch stock data from Yahoo Finance
    
    Parameters:
    - ticker: Stock symbol (e.g., 'DABUR.NS')
    - interval: '1d' for daily, '1wk' for weekly, '15m' for 15-minute
    - period: Time period (e.g., '1y', '6mo', '60d')
    
    Returns:
    - DataFrame with OHLCV data
    """
    stock = yf.Ticker(ticker)
    df = stock.history(interval=interval, period=period)
    
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
    
    return df


def calculate_volume_profile(df, num_bins=None):
    """
    Calculate volume profile (volume at different price levels)
    
    Parameters:
    - df: DataFrame with OHLC data
    - num_bins: Number of price bins (defaults to CONFIG value)
    
    Returns:
    - price_levels: Array of price levels
    - volumes: Array of volumes at each price level
    """
    if num_bins is None:
        num_bins = CONFIG_VOLUME_PROFILE['num_bins']
    
    # Get price range
    price_min = df['Low'].min()
    price_max = df['High'].max()
    
    # Create price bins
    price_bins = np.linspace(price_min, price_max, num_bins + 1)
    volume_profile = np.zeros(num_bins)
    
    # Calculate volume for each price level
    for idx, row in df.iterrows():
        # For each candle, distribute volume across price range
        low, high = row['Low'], row['High']
        volume = row['Volume']
        
        # Find which bins this candle covers
        for i in range(num_bins):
            bin_low = price_bins[i]
            bin_high = price_bins[i + 1]
            
            # Calculate overlap between candle range and bin
            overlap_low = max(low, bin_low)
            overlap_high = min(high, bin_high)
            
            if overlap_high > overlap_low:
                # Distribute volume proportionally
                overlap_ratio = (overlap_high - overlap_low) / (high - low) if high > low else 1.0
                volume_profile[i] += volume * overlap_ratio
    
    # Calculate midpoints of bins
    price_levels = (price_bins[:-1] + price_bins[1:]) / 2
    
    return price_levels, volume_profile


def plot_candlestick_chart(ticker, chart_type='daily', period='1y'):
    """
    Create a candlestick chart with volume and volume profile
    
    Parameters:
    - ticker: Stock symbol (e.g., 'DABUR.NS')
    - chart_type: 'daily', 'weekly', or '15min'
    - period: Time period (default '1y')
    """
    # Map chart type to yfinance interval
    interval_map = {
        'daily': '1d',
        'weekly': '1wk',
        '15min': '15m'
    }
    
    # Adjust period for 15min data (max 60 days for 15min)
    if chart_type == '15min' and period in ['1y', '6mo']:
        period = '60d'
        print(f"Note: Period adjusted to 60d for 15min interval")
    
    interval = interval_map.get(chart_type, '1d')
    
    # Fetch data
    print(f"Fetching {chart_type} data for {ticker}...")
    df = fetch_stock_data(ticker, interval, period)
    
    if len(df) == 0:
        raise ValueError("No data available for the specified period")
    
    print(f"Data fetched: {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    # Calculate volume profile
    price_levels, volume_profile = calculate_volume_profile(df)
    
    # Detect pyschological patterns
    patterns = None
    if CONFIG_PATTERNS['show_patterns']:
        print("Detecting psychological candlestick patterns...")
        patterns = detect_psychological_patterns(df)
        strength_count = len(patterns['strength'])
        rejection_count = len(patterns['control_shift'])
        indecision_count = len(patterns['indecision'])
        print(f"Patterns found: {strength_count} Strength, {rejection_count} Control Shift, {indecision_count} Indecision Zones")
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(CONFIG_CHART['figure_width'], CONFIG_CHART['figure_height']))
    
    # Determine volume profile position
    if CONFIG_VOLUME_PROFILE['position'] == 'right':
        # Create grid: [candlestick | volume_profile]
        #               [volume_bars |              ]
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[5, 1], 
                              hspace=0.05, wspace=0.05)
        ax_candle = fig.add_subplot(gs[0, 0])
        ax_volume = fig.add_subplot(gs[1, 0], sharex=ax_candle)
        ax_profile = fig.add_subplot(gs[0, 1], sharey=ax_candle)
        ax_empty = fig.add_subplot(gs[1, 1])
        ax_empty.axis('off')
    else:
        # Create grid: [volume_profile | candlestick]
        #               [              | volume_bars  ]
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1, 5], 
                              hspace=0.05, wspace=0.05)
        ax_candle = fig.add_subplot(gs[0, 1])
        ax_volume = fig.add_subplot(gs[1, 1], sharex=ax_candle)
        ax_profile = fig.add_subplot(gs[0, 0], sharey=ax_candle)
        ax_empty = fig.add_subplot(gs[1, 0])
        ax_empty.axis('off')
    
    # Plot candlestick
    for idx, (date, row) in enumerate(df.iterrows()):
        color = CONFIG_COLORS['bullish_candle'] if row['Close'] >= row['Open'] else CONFIG_COLORS['bearish_candle']
        
        # Draw high-low line
        ax_candle.plot([idx, idx], [row['Low'], row['High']], color=color, linewidth=1)
        
        # Draw candle body
        body_height = abs(row['Close'] - row['Open'])
        body_bottom = min(row['Open'], row['Close'])
        
        rect = Rectangle((idx - CONFIG_CHART['candle_width']/2, body_bottom), 
                        CONFIG_CHART['candle_width'], body_height, 
                        facecolor=color, edgecolor=color, alpha=0.8)
        ax_candle.add_patch(rect)
    
    # Plot Patterns
    if patterns:
        # 1. Strength Candles
        for p in patterns['strength']:
            idx = p['index']
            is_bullish = p['type'] == 'bullish'
            
            # Text annotation with arrow
            text = "Bullish Strength\n(Breakout)" if is_bullish else "Bearish Strength\n(Breakdown)"
            color = CONFIG_PATTERNS['colors']['strength_bullish'] if is_bullish else CONFIG_PATTERNS['colors']['strength_bearish']
            
            # Position arrow slightly above/below
            y_point = df.iloc[idx]['High'] if is_bullish else df.iloc[idx]['Low']
            y_text_offset = (df['High'].max() - df['Low'].min()) * 0.15
            y_text = y_point + y_text_offset if is_bullish else y_point - y_text_offset
            
            ax_candle.annotate(text, 
                              xy=(idx, y_point), 
                              xytext=(idx, y_text),
                              arrowprops=dict(facecolor=color, shrink=0.05, width=1, headwidth=6),
                              fontsize=CONFIG_PATTERNS['style']['font_size_strength'],
                              color=color,
                              fontweight='bold',
                              ha='center')
            
        # 2. Control Shift (Rejection)
        for p in patterns['control_shift']:
            idx = p['index']
            price = p['price']
            is_bearish_rejection = p['type'] == 'bearish_rejection'
            
            # Draw Ellipse
            # Height of ellipse is 5% of chart range, width is 1.5 candles
            h_ellipse = (df['High'].max() - df['Low'].min()) * 0.05
            w_ellipse = 1.5
            
            color = CONFIG_PATTERNS['colors']['control_shift_bearish'] if is_bearish_rejection else CONFIG_PATTERNS['colors']['control_shift_bullish']
            
            ellipse = Ellipse((idx, price), width=w_ellipse, height=h_ellipse,
                             edgecolor=color, facecolor='none', linewidth=1.5)
            ax_candle.add_patch(ellipse)
            
            # Text Label nearby
            label = "Control Shift\n(Bearish Reversal)" if is_bearish_rejection else "Control Shift\n(Bullish Reversal)"
            y_text_offset = h_ellipse * 1.5
            y_text = price + y_text_offset if is_bearish_rejection else price - y_text_offset
            
            ax_candle.text(idx, y_text, label,
                          ha='center', va='center',
                          color=color,
                          fontsize=CONFIG_PATTERNS['style']['font_size_rejection'],
                          fontweight='bold')

        # 3. Indecision Zones
        for p in patterns['indecision']:
            start_idx = p['start_idx']
            end_idx = p['end_idx']
            
            # Find high and low for the box
            zone_high = df.iloc[start_idx:end_idx+1]['High'].max()
            zone_low = df.iloc[start_idx:end_idx+1]['Low'].min()
            
            # Draw box
            rect = Rectangle((start_idx - 0.4, zone_low), 
                            (end_idx - start_idx + 0.8), 
                            (zone_high - zone_low),
                            facecolor=CONFIG_PATTERNS['colors']['indecision'], 
                            edgecolor='black', linestyle='--', alpha=0.2, linewidth=1)
            ax_candle.add_patch(rect)
            
            # Label
            mid_x = (start_idx + end_idx) / 2
            ax_candle.text(mid_x, zone_high + (zone_high - zone_low)*0.1, 
                          "Indecision Zone\n(Consolidation)",
                          ha='center', va='bottom',
                          color='black',
                          fontsize=CONFIG_PATTERNS['style']['font_size_indecision'],
                          fontweight='bold')

    # Calculate and plot EMAs (Exponential Moving Averages)
    if CONFIG_EMA['show_ema_20']:
        df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
        ax_candle.plot(range(len(df)), df['EMA20'], 
                      color=CONFIG_COLORS['ema_20'], 
                      linewidth=CONFIG_EMA['ema_linewidth'], 
                      label='EMA20', 
                      alpha=CONFIG_EMA['ema_alpha'])
    
    if CONFIG_EMA['show_ema_50']:
        df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
        ax_candle.plot(range(len(df)), df['EMA50'], 
                      color=CONFIG_COLORS['ema_50'], 
                      linewidth=CONFIG_EMA['ema_linewidth'], 
                      label='EMA50', 
                      alpha=CONFIG_EMA['ema_alpha'])
    
    if CONFIG_EMA['show_ema_200']:
        df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
        ax_candle.plot(range(len(df)), df['EMA200'], 
                      color=CONFIG_COLORS['ema_200'], 
                      linewidth=CONFIG_EMA['ema_linewidth'], 
                      label='EMA200', 
                      alpha=CONFIG_EMA['ema_alpha'])
    
    ax_candle.set_ylabel('Price', fontsize=12, fontweight='bold')
    ax_candle.set_title(f'{ticker} - {chart_type.upper()} Chart', fontsize=14, fontweight='bold')
    ax_candle.legend(loc='upper left')
    if CONFIG_GRID['show_grid']:
        ax_candle.grid(True, alpha=CONFIG_GRID['grid_alpha'])
    ax_candle.tick_params(labelbottom=False)
    
    # Plot volume bars
    colors = [CONFIG_COLORS['bullish_candle'] if row['Close'] >= row['Open'] 
              else CONFIG_COLORS['bearish_candle'] for _, row in df.iterrows()]
    ax_volume.bar(range(len(df)), df['Volume'], 
                  color=colors, 
                  alpha=CONFIG_VOLUME_BAR['alpha'], 
                  width=CONFIG_CHART['volume_bar_width'])
    ax_volume.set_ylabel('Volume', fontsize=12, fontweight='bold')
    ax_volume.set_xlabel('Date', fontsize=12, fontweight='bold')
    if CONFIG_GRID['show_grid']:
        ax_volume.grid(True, alpha=CONFIG_GRID['grid_alpha'])
    
    # Set x-axis labels (show dates at regular intervals)
    num_labels = 10
    step = max(1, len(df) // num_labels)
    tick_positions = range(0, len(df), step)
    tick_labels = [df.index[i].strftime('%Y-%m-%d') for i in tick_positions]
    ax_volume.set_xticks(tick_positions)
    ax_volume.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    # Plot volume profile (horizontal bars)
    if CONFIG_VOLUME_PROFILE['position'] == 'right':
        # Volume profile on right - bars grow to the left
        ax_profile.barh(price_levels, volume_profile, 
                       height=(price_levels[1] - price_levels[0]), 
                       color=CONFIG_COLORS['volume_profile'], 
                       alpha=CONFIG_VOLUME_PROFILE['alpha'])
        ax_profile.set_xlabel('Volume', fontsize=10, fontweight='bold')
        ax_profile.tick_params(labelright=False, labelleft=False)
        if CONFIG_GRID['show_grid']:
            ax_profile.grid(True, alpha=CONFIG_GRID['grid_alpha'], axis='y')
    else:
        # Volume profile on left - bars grow to the right
        ax_profile.barh(price_levels, volume_profile, 
                       height=(price_levels[1] - price_levels[0]), 
                       color=CONFIG_COLORS['volume_profile'], 
                       alpha=CONFIG_VOLUME_PROFILE['alpha'])
        ax_profile.set_xlabel('Volume', fontsize=10, fontweight='bold')
        ax_profile.set_ylabel('Price', fontsize=12, fontweight='bold')
        ax_profile.tick_params(labelleft=False)
        ax_profile.invert_xaxis()
        if CONFIG_GRID['show_grid']:
            ax_profile.grid(True, alpha=CONFIG_GRID['grid_alpha'], axis='y')
    
    plt.tight_layout()
    plt.show()
    
    return fig


def main():
    """
    Main function - example usage
    """
    # Example 1: Daily chart
    plot_candlestick_chart(
        ticker='MARICO.NS',
        chart_type='daily',
        period='2y'
    )
    
    # Example 2: Weekly chart
    # plot_candlestick_chart(
    #     ticker='DABUR.NS',
    #     chart_type='weekly',
    #     period='2y'
    # )
    
    # Example 3: 15-minute chart
    # plot_candlestick_chart(
    #     ticker='DABUR.NS',
    #     chart_type='15min',
    #     period='5d'
    # )


if __name__ == "__main__":
    # You can customize parameters here
    import sys
    
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
        chart_type = sys.argv[2] if len(sys.argv) > 2 else 'daily'
        period = sys.argv[3] if len(sys.argv) > 3 else '1y'
        
        plot_candlestick_chart(ticker, chart_type, period)
    else:
        main()
