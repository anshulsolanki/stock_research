# -------------------------------------------------------------------------------
# Project: Stock Analysis (https://github.com/anshulsolanki/stock_analysis)
# Author:  Anshul Solanki
# License: MIT License
# -------------------------------------------------------------------------------

"""
Mark Minervini VCP Strategy Screener

This script implements Mark Minervini's Volatility Contraction Pattern (VCP) strategy
to identify high-growth stocks in Stage 2 uptrends with tight price consolidation.

Minervini Strategy Criteria:
----------------------------

Here is a step-by-step breakdown of how the code implements the VCP logic.

1. Phase 1: The Macro Setup (Trend Template)
- Before looking for a VCP, the script ensures the stock is in a confirmed Stage 2 Uptrend. It does this by calculating moving averages and 52-week highs/lows on the last 200+ days of data.
- Moving Average Stacking: It checks that the current price is above the 50-day, 150-day, and 200-day Simple Moving Averages (SMAs). It also strictly enforces the order: SMA50 > SMA150 > SMA200.
- Rising 200 SMA: It compares the current 200-day SMA to its value 20 days ago (prev_sma_200) to ensure the long-term trend is pointing upward.
- Price Proximity Filters: The stock must be at least 30% above its 52-week low (MIN_PRICE_TO_LOW = 1.30) and within 25% of its 52-week high (MAX_PRICE_TO_HIGH = 0.75).
Note: If the --use-fundamentals flag is passed, it also enforces SEPA fundamental criteria: Relative Strength (RS) rating >= 70, Quarterly EPS growth >= 20%, and Quarterly Sales growth >= 15%.

2. Phase 2 & 3: Identifying the Contractions (The VCP Logic)
if the macro setup is valid, the script analyzes the last 90 days of price action to find the actual Volatility Contraction Pattern.
- Data Smoothing: Real stock data is noisy, which makes finding exact pivot highs and lows difficult programmatically. The script solves this by applying a 5-day Exponential Moving Average (df['Close'].ewm(span=5).mean()) to smooth the price curve.
- Peak and Trough Detection: It uses scipy.signal.find_peaks with a minimum distance of 15 days (distance=15) on the smoothed data to identify local highs (peaks) and local lows (troughs).
- Flat Top Resistance Guard: It takes the last 3 peaks and checks their variance (peak_variance). If the peaks vary by more than 10%, the script discards the stock because a valid VCP requires a relatively "flat top" resistance zone, not wild swings.
- Calculating Pullback Depths: The script iterates through the identified peaks, finds the immediate subsequent trough, and calculates the percentage drop (depth) of that specific swing: (peak_val - trough_val) / peak_val.
- Progressive Contraction Guard: This is the heart of Minervini's strategy. The script checks the last 2 or 3 pullbacks to ensure they are getting progressively smaller from left to right (e.g., recent_pullbacks[0] > recent_pullbacks[1] > recent_pullbacks[2]).
- Final Tightness Guard: The very last pullback (the right side of the base) must be incredibly tight—specifically, less than or equal to a 6% drop (recent_pullbacks[-1] <= 0.06).
Note: If the --use-volume-dryup flag is passed, it also checks that the average volume during these pullback periods is progressively decreasing, indicating a dry-up of selling pressure.

3. Phase 4: Breakout, Volume, and Risk Management
Finally, if a valid contraction pattern exists, the script checks if a breakout is currently occurring.
- The Pivot Point: The script defines the pivot_point as the price of the final (most recent) peak in the contraction.
- Entry Trigger: The current close must be greater than the pivot_point.
- Volume Confirmation: Professional buying must accompany the breakout. The script enforces that the current day's volume is at least 150% of the 50-day average volume (current_vol >= vol_sma_50 * 1.5).

Risk Management Calculations:
- Stop Loss: Placed 2% below the low of the final trough (closes_smoothed[final_troughs[0]] * 0.98).
- Target: Automatically set to a 25% gain from the pivot point (pivot_point * 1.25).
- Risk/Reward: Calculates the ratio of the expected reward against the risk to the stop loss.

If all these conditions evaluate to True, the stock is flagged as a match, its metrics are formatted into a dictionary, and the main thread eventually plots it onto a PDF report using matplotlib.

Usage:
------
python minervini_screener.py [--limit N] [--sample N] [--refresh] [--use-fundamentals] [--use-volume-dryup]
"""

# Imports
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.signal
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize Console (Dummy wrapper for now, similar to canslim_screener)
class Console:
    def print(self, text):
        clean_text = text.replace("[bold green]", "").replace("[/bold green]", "")
        clean_text = clean_text.replace("[green]", "").replace("[/green]", "")
        clean_text = clean_text.replace("[red]", "").replace("[/red]", "")
        clean_text = clean_text.replace("[yellow]", "").replace("[/yellow]", "")
        clean_text = clean_text.replace("[cyan]", "").replace("[/cyan]", "")
        clean_text = clean_text.replace("[magenta]", "").replace("[/magenta]", "")
        clean_text = clean_text.replace("[blue]", "").replace("[/blue]", "")
        clean_text = clean_text.replace("[bold]", "").replace("[/bold]", "")
        print(clean_text)
    
    def status(self, text):
        print(f"STATUS: {text}")
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

console = Console()

# Configuration
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = os.path.join(BASE_DIR, 'screener_results', 'minervini_breakouts', TIMESTAMP)
CHARTS_DIR = os.path.join(OUTPUT_DIR, 'charts')
CACHE_DIR = os.path.join(DATA_DIR, 'data_cache')
JSON_PATH = os.path.join(DATA_DIR, 'nifty_500.json')
PDF_PATH = os.path.join(OUTPUT_DIR, f"minervini_Screener_Results_{TIMESTAMP}.pdf")

# Minervini Parameters
MIN_EPS_GROWTH = 0.20
MIN_SALES_GROWTH = 0.15
MIN_RS_RATING = 70
MIN_PRICE_TO_LOW = 1.30  # At least 30% above low
MAX_PRICE_TO_HIGH = 0.75 # Within 25% of high (inverted logic in check)

def setup_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

def load_tickers(limit=None):
    try:
        with open(JSON_PATH, 'r') as f:
            data = json.load(f)
        tickers = list(data.values())
        if limit:
            return tickers[:limit]
        return tickers
    except FileNotFoundError:
        console.print(f"[red]Error: {JSON_PATH} not found.[/red]")
        return []

def calculate_rs_ratings(tickers, end_date=None, preloaded_data=None):
    """Calculate IBD-style weighted Relative Strength rank across our universe.
    
    Uses a weighted formula: 40% most recent quarter, 20% each for prior 3 quarters.
    This emphasizes recent momentum, similar to IBD's Relative Strength rating.
    """
    returns = {}
    with console.status(f"[blue]Calculating Relative Strength rankings for {len(tickers)} stocks...[/blue]"):
        for ticker in tickers:
            if preloaded_data and ticker in preloaded_data:
                df = preloaded_data[ticker]['history'].copy()
            else:
                cache_path = os.path.join(CACHE_DIR, f"{ticker}_1d.csv")
                if os.path.exists(cache_path):
                    df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                else:
                    continue
            
            # Truncate to end_date if specified
            if end_date is not None:
                end_ts = pd.Timestamp(end_date)
                if df.index.tz is not None:
                    end_ts = end_ts.tz_localize(df.index.tz)
                df = df[df.index <= end_ts]
            if len(df) > 252:
                try:
                    close = df['Close']
                    # IBD-style weighted RS: 40% Q1 (recent), 20% each for Q2-Q4
                    q1 = (close.iloc[-1] - close.iloc[-63]) / close.iloc[-63]
                    q2 = (close.iloc[-63] - close.iloc[-126]) / close.iloc[-126]
                    q3 = (close.iloc[-126] - close.iloc[-189]) / close.iloc[-189]
                    q4 = (close.iloc[-189] - close.iloc[-252]) / close.iloc[-252]
                    weighted_return = 0.4 * q1 + 0.2 * q2 + 0.2 * q3 + 0.2 * q4
                    returns[ticker] = weighted_return
                except (IndexError, ZeroDivisionError):
                    pass
    
    if not returns:
        return {}
        
    s = pd.Series(returns)
    ranks = s.rank(pct=True) * 99
    return ranks.apply(lambda x: int(round(x))).to_dict()

def fetch_data(ticker, refresh=False):
    hist_cache_path = os.path.join(CACHE_DIR, f"{ticker}_1d.csv")
    info_cache_path = os.path.join(CACHE_DIR, f"{ticker}_info.json")
    
    data = {}
    
    try:
        if not refresh and os.path.exists(info_cache_path):
             with open(info_cache_path, 'r') as f:
                  data['info'] = json.load(f)
        else:
             stock = yf.Ticker(ticker)
             info = stock.info
             data['info'] = info
             with open(info_cache_path, 'w') as f:
                  json.dump(info, f)
        
        if not refresh and os.path.exists(hist_cache_path):
            df = pd.read_csv(hist_cache_path, index_col=0, parse_dates=True)
            data['history'] = df
        else:
            stock = yf.Ticker(ticker)
            df = stock.history(period="2y") 
            if len(df) < 200: return None
            df.to_csv(hist_cache_path)
            data['history'] = df
            
        return data
    except Exception as e:
        return None

def check_criteria(data, ticker, rs_rating, use_fundamentals=False, use_volume_dryup=False, end_date=None):
    info = data.get('info', {})
    df = data.get('history', None)
    
    if df is None or df.empty:
        return None
    
    # Truncate data to end_date if specified (for historical backtesting)
    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
        # Match timezone of dataframe index (yfinance returns tz-aware timestamps)
        if df.index.tz is not None:
            end_ts = end_ts.tz_localize(df.index.tz)
        df = df[df.index <= end_ts].copy()
    
    if len(df) < 200:
        return None
        
    # --- 1. Macro Setup (Trend Template) ---
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA150'] = df['Close'].rolling(window=150).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['52W_High'] = df['High'].rolling(window=252).max()
    df['52W_Low'] = df['Low'].rolling(window=252).min()
    df['VolSMA50'] = df['Volume'].rolling(window=50).mean()
    
    curr = df.iloc[-1]
    
    current_close = curr['Close']
    current_vol = curr['Volume']
    sma_50 = curr['SMA50']
    sma_150 = curr['SMA150']
    sma_200 = curr['SMA200']
    high_52w = curr['52W_High']
    low_52w = curr['52W_Low']
    vol_sma_50 = curr['VolSMA50']
    
    # Check lines: Price > 50, 150, 200 SMAs
    if current_close < sma_50 or current_close < sma_150 or current_close < sma_200:
        return None
        
    # Check order: 50 > 150 > 200 SMAs
    if sma_50 < sma_150 or sma_150 < sma_200:
        return None
        
    # 200 SMA rising
    prev_sma_200 = df['SMA200'].iloc[-20] if len(df) >= 20 else df['SMA200'].iloc[0]
    if sma_200 <= prev_sma_200:
        return None
        
    # Proximity to highs/lows
    if current_close < low_52w * MIN_PRICE_TO_LOW:
        return None
    if current_close < high_52w * MAX_PRICE_TO_HIGH:
        return None
        
    eps_growth = info.get('earningsQuarterlyGrowth', 0) or 0
    sales_growth = info.get('revenueGrowth', 0) or 0
    
    if use_fundamentals:
        # Relative Strength
        if rs_rating < MIN_RS_RATING:
            return None

        # --- 2. Fundamentals (SEPA) ---
        if eps_growth < MIN_EPS_GROWTH: return None
        if sales_growth < MIN_SALES_GROWTH: return None

    # --- 3. VCP Logic (Contractions) ---
    # Smooth with EMA to find "clean" swings (used only for detection)
    df['SmoothClose'] = df['Close'].ewm(span=5).mean()
    
    # Adaptive window: scan up to 130 trading days (~26 weeks max base)
    vcp_window = min(130, len(df))
    closes_smoothed = df['SmoothClose'].tail(vcp_window).values
    raw_closes = df['Close'].tail(vcp_window).values
    raw_lows = df['Low'].tail(vcp_window).values
    if use_volume_dryup:
        volumes = df['Volume'].tail(vcp_window).values
    
    # Peak/trough detection with prominence to filter insignificant wiggles
    mean_price = np.mean(closes_smoothed)
    prominence_threshold = 0.02 * mean_price
    peaks, _ = scipy.signal.find_peaks(closes_smoothed, distance=10, prominence=prominence_threshold)
    troughs, _ = scipy.signal.find_peaks(-closes_smoothed, distance=10, prominence=prominence_threshold)
    
    if len(peaks) < 2 or len(troughs) < 1:
        return None # Not enough swings

    # Align peaks and troughs to calculate pullback depths
    # We want Peak -> subsequent Trough sequence
    pullbacks = []
    pullback_vols = []
    pullback_peak_indices = []  # Track which peaks produced valid pullbacks
    pullback_trough_indices = []
    for p in peaks:
        # Find the first trough after this peak
        valid_troughs = troughs[troughs > p]
        if len(valid_troughs) > 0:
            t = valid_troughs[0]
            peak_val = closes_smoothed[p]
            trough_val = closes_smoothed[t]
            depth = (peak_val - trough_val) / peak_val
            pullbacks.append(depth)
            pullback_peak_indices.append(p)
            pullback_trough_indices.append(t)
            
            if use_volume_dryup:
                # Calculate average volume during this pullback period
                avg_vol = np.mean(volumes[p:t+1])
                pullback_vols.append(avg_vol)
            
    if len(pullbacks) < 2:
        return None
        
    # Check for progressive contraction (getting smaller)
    # E.g., depth1 > depth2 > depth3
    recent_pullbacks = pullbacks[-3:] if len(pullbacks) >= 3 else pullbacks
    recent_peak_indices = pullback_peak_indices[-3:] if len(pullback_peak_indices) >= 3 else pullback_peak_indices
    if use_volume_dryup:
        recent_vols = pullback_vols[-3:] if len(pullback_vols) >= 3 else pullback_vols
        
    if len(recent_pullbacks) == 3:
        if recent_pullbacks[0] <= recent_pullbacks[1] or recent_pullbacks[1] <= recent_pullbacks[2]:
            return None
        if use_volume_dryup:
            # Check for volume dry-up (average volume decreasing)
            if recent_vols[0] <= recent_vols[1] or recent_vols[1] <= recent_vols[2]:
                return None
    elif len(recent_pullbacks) == 2:
        if recent_pullbacks[0] <= recent_pullbacks[1]:
            return None
        if use_volume_dryup:
            # Check for volume dry-up
            if recent_vols[0] <= recent_vols[1]:
                return None

    # Final Tightness Guard (Must be <= 6%)
    if recent_pullbacks[-1] > 0.06:
        return None

    # Flat Top Resistance Guard: aligned with pullback peaks + the pivot peak
    pivot_peak_idx = peaks[-1]
    resistance_peak_indices = sorted(set(list(recent_peak_indices) + [pivot_peak_idx]))
    peak_prices = closes_smoothed[resistance_peak_indices]
    resistance_level = np.mean(peak_prices)
    peak_variance = (np.max(peak_prices) - np.min(peak_prices)) / resistance_level

    if peak_variance > 0.1: # If peaks vary by more than 10%, it's not a "flat top"
        return None

    # Base Length Validation: pattern should span 3-26 weeks (15-130 trading days)
    base_start = recent_peak_indices[0]
    base_end = pivot_peak_idx
    base_length = base_end - base_start
    if base_length < 15 or base_length > 130:
        return None

    # --- 4. Breakout and Risk Management ---
    # Pivot Point: use RAW close price at detected peak (not smoothed)
    pivot_point = raw_closes[pivot_peak_idx]
    
    # Entry Trigger: Today's close > Pivot
    if current_close <= pivot_point:
        return None
        
    # Volume Confirmation: Breakout vol >= 130% of 50-day avg (30% above average)
    if current_vol < vol_sma_50 * 1.3:
        return None
        
    # Stop Loss: use RAW Low price at the final trough (not smoothed close)
    # The final contraction's low is the last trough before the pivot
    pre_pivot_troughs = troughs[troughs < pivot_peak_idx]
    if len(pre_pivot_troughs) > 0:
        stop_loss = raw_lows[pre_pivot_troughs[-1]] * 0.98 # 2% below the raw low
    else:
        stop_loss = current_close * 0.92 # Fallback: 8% below current price
        
    # Target (25% gain from pivot)
    target_price = pivot_point * 1.25
    
    # Risk/Reward
    risk = current_close - stop_loss
    reward = target_price - current_close
    risk_reward = reward / risk if risk > 0 else 0
        
    return {
        'Ticker': ticker,
        'Date': df.index[-1].strftime('%Y-%m-%d'),
        'Close': round(current_close, 1),
        'Pivot': round(pivot_point, 2),
        'Target': round(target_price, 2),
        'Stop_Loss': round(stop_loss, 2),
        'Risk_Reward': round(risk_reward, 2),
        'Qtr_EPS%': round(eps_growth * 100, 2),
        'Qtr_Sales%': round(sales_growth * 100, 2),
        'RS_Rating': rs_rating,
        'Vol_Ratio': round(current_vol / vol_sma_50, 2),
        'Pullbacks': [round(float(p) * 100, 1) for p in recent_pullbacks]
    }

def render_pdf_standard_header(fig, title_text="Minervini VCP Report"):
    PRIMARY_COLOR = '#1e293b'
    BORDER_COLOR = '#94a3b8'
    fig.text(0.5, 0.96, title_text, ha='center', va='center', fontsize=16, weight='bold', color=PRIMARY_COLOR)
    copyright_text = f"© {datetime.now().year} Stock Research. All Rights Reserved."
    fig.text(0.5, 0.935, copyright_text, ha='center', va='center', fontsize=9, style='italic', color=BORDER_COLOR)
    from matplotlib.lines import Line2D
    line = Line2D([0.08, 0.92], [0.91, 0.91], transform=fig.transFigure, color=BORDER_COLOR, linewidth=1.0, alpha=0.5)
    fig.add_artist(line)

def create_pdf_title_page(pdf, timestamp, market_status):
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    render_pdf_standard_header(fig, "Stock Analysis Report")
    PRIMARY_COLOR = '#1e293b'
    ACCENT_COLOR = '#2563eb'
    SECONDARY_COLOR = '#475569'
    
    fig.text(0.5, 0.65, "Minervini VCP", ha='center', va='center', fontsize=36, weight='bold', color=ACCENT_COLOR)
    fig.text(0.5, 0.58, "High-Growth Analysis Report", ha='center', va='center', fontsize=22, weight='bold', color=SECONDARY_COLOR)
    fig.text(0.5, 0.52, f"Strategic Market Scan: {timestamp}", ha='center', va='center', fontsize=14, color=SECONDARY_COLOR)
    
    status_color = '#16a34a' if "Bullish" in market_status else '#dc2626'
    fig.text(0.5, 0.46, f"Market Condition: {market_status}", ha='center', va='center', fontsize=16, weight='bold', color=status_color)

    fig.text(0.1, 0.35, "Strengths of Minervini VCP", fontsize=16, weight='bold', color=PRIMARY_COLOR)
    strengths = [
        "● Focus on Stage 2 Uptrends",
        "● Visual pattern of contraction",
        "● Volume dry-up for supply exhaustion",
        "● Breakout on high volume"
    ]
    for i, s in enumerate(strengths):
        fig.text(0.15, 0.31 - (i * 0.04), s, fontsize=12, color=SECONDARY_COLOR)

    fig.text(0.6, 0.35, "Limitations", fontsize=16, weight='bold', color='#dc2626')
    limitations = [
        "● Prone to fail in choppy/bear markets",
        "● Strict adherence to stop losses", 
        "   (40–55% failure rate)",
        "● Low-Volume Breakouts are tricky"
    ]
    for i, l in enumerate(limitations):
        fig.text(0.6, 0.31 - (i * 0.04), l, fontsize=12, color=SECONDARY_COLOR)

    pdf.savefig(fig)
    plt.close(fig)

def render_pdf_documentation_page(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    render_pdf_standard_header(fig, "VCP Methodology & Rules")
    PRIMARY_COLOR = '#1e293b'
    SECONDARY_COLOR = '#475569'
    LEFT_MARGIN = 0.08
    
    text = """
1. Phase 1: The Macro Setup (Trend Template)
- Before looking for a VCP, the script ensures the stock is in a confirmed Stage 2 Uptrend. It does this by calculating moving averages and 52-week highs/lows on the last 200+ days of data.
- Moving Average Stacking: It checks that the current price is above the 50-day, 150-day, and 200-day Simple Moving Averages (SMAs). It also strictly enforces the order: SMA50 > SMA150 > SMA200.
- Rising 200 SMA: It compares the current 200-day SMA to its value 20 days ago (prev_sma_200) to ensure the long-term trend is pointing upward.
- Price Proximity Filters: The stock must be at least 30% above its 52-week low (MIN_PRICE_TO_LOW = 1.30) and within 25% of its 52-week high (MAX_PRICE_TO_HIGH = 0.75).
Note: If the --use-fundamentals flag is passed, it also enforces SEPA fundamental criteria: Relative Strength (RS) rating >= 70, Quarterly EPS growth >= 20%, and Quarterly Sales growth >= 15%.

2. Phase 2 & 3: Identifying the Contractions (The VCP Logic)
if the macro setup is valid, the script analyzes the last 90 days of price action to find the actual Volatility Contraction Pattern.
- Data Smoothing: Real stock data is noisy, which makes finding exact pivot highs and lows difficult programmatically. The script solves this by applying a 5-day Exponential Moving Average (df['Close'].ewm(span=5).mean()) to smooth the price curve.
- Peak and Trough Detection: It uses scipy.signal.find_peaks with a minimum distance of 15 days (distance=15) on the smoothed data to identify local highs (peaks) and local lows (troughs).
- Flat Top Resistance Guard: It takes the last 3 peaks and checks their variance (peak_variance). If the peaks vary by more than 10%, the script discards the stock because a valid VCP requires a relatively "flat top" resistance zone, not wild swings.
- Calculating Pullback Depths: The script iterates through the identified peaks, finds the immediate subsequent trough, and calculates the percentage drop (depth) of that specific swing: (peak_val - trough_val) / peak_val.
- Progressive Contraction Guard: This is the heart of Minervini's strategy. The script checks the last 2 or 3 pullbacks to ensure they are getting progressively smaller from left to right (e.g., recent_pullbacks[0] > recent_pullbacks[1] > recent_pullbacks[2]).
- Final Tightness Guard: The very last pullback (the right side of the base) must be incredibly tight—specifically, less than or equal to a 6% drop (recent_pullbacks[-1] <= 0.06).
Note: If the --use-volume-dryup flag is passed, it also checks that the average volume during these pullback periods is progressively decreasing, indicating a dry-up of selling pressure.

3. Phase 4: Breakout, Volume, and Risk Management
Finally, if a valid contraction pattern exists, the script checks if a breakout is currently occurring.
- The Pivot Point: The script defines the pivot_point as the price of the final (most recent) peak in the contraction.
- Entry Trigger: The current close must be greater than the pivot_point.
- Volume Confirmation: Professional buying must accompany the breakout. The script enforces that the current day's volume is at least 150% of the 50-day average volume (current_vol >= vol_sma_50 * 1.5).

Risk Management Calculations:
- Stop Loss: Placed 2% below the low of the final trough (closes_smoothed[final_troughs[0]] * 0.98).
- Target: Automatically set to a 25% gain from the pivot point (pivot_point * 1.25).
- Risk/Reward: Calculates the ratio of the expected reward against the risk to the stop loss.

If all these conditions evaluate to True, the stock is flagged as a match. !!!.
"""
    import textwrap
    lines = text.strip().split('\n')
    y = 0.82
    wrapper = textwrap.TextWrapper(width=95, break_long_words=False, replace_whitespace=False)
    
    for line in lines:
        stripped_line = line.strip()
        if stripped_line == "":
            y -= 0.015
            continue

        is_header = stripped_line.startswith(("1.", "2.", "3.", "Risk Management", "If all these")) or stripped_line.endswith(":")
        
        if is_header:
            wrapped = wrapper.wrap(stripped_line)
            for w_line in wrapped:
                if y < 0.06:
                    pdf.savefig(fig)
                    plt.clf()
                    plt.axis('off')
                    render_pdf_standard_header(fig, "VCP Methodology & Rules")
                    fig.text(LEFT_MARGIN, 0.86, "Mark Minervini VCP Rules", fontsize=20, weight='bold', color=PRIMARY_COLOR)
                    y = 0.82
                fig.text(LEFT_MARGIN, y, w_line, fontsize=11, weight='bold', color=PRIMARY_COLOR)
                y -= 0.028
        else:
            is_bullet = stripped_line.startswith("-")
            content_to_wrap = stripped_line[1:].strip() if is_bullet else stripped_line
            
            wrapped = wrapper.wrap(content_to_wrap)
            for i, w_line in enumerate(wrapped):
                if y < 0.06:
                    pdf.savefig(fig)
                    plt.clf()
                    plt.axis('off')
                    render_pdf_standard_header(fig, "VCP Methodology & Rules")
                    fig.text(LEFT_MARGIN, 0.86, "Mark Minervini VCP Rules", fontsize=20, weight='bold', color=PRIMARY_COLOR)
                    y = 0.82
                    
                if is_bullet and i == 0:
                    fig.text(LEFT_MARGIN, y, "●", fontsize=8, color='#2563eb', va='center')
                    fig.text(LEFT_MARGIN + 0.02, y, w_line, fontsize=10, color=SECONDARY_COLOR)
                elif is_bullet and i > 0:
                    fig.text(LEFT_MARGIN + 0.02, y, w_line, fontsize=10, color=SECONDARY_COLOR)
                else:
                    fig.text(LEFT_MARGIN, y, w_line, fontsize=10, color=SECONDARY_COLOR)
                y -= 0.022
                
    pdf.savefig(fig)
    plt.close(fig)

def render_pdf_styled_table(pdf, df, title):
    if df.empty:
        return
    rows_per_page = 22
    num_pages = (len(df) // rows_per_page) + 1
    for i in range(num_pages):
        start_idx = i * rows_per_page
        end_idx = min((i + 1) * rows_per_page, len(df))
        chunk = df.iloc[start_idx:end_idx].copy()
        
        fig = plt.figure(figsize=(14, 8.5)) 
        ax = fig.add_axes([0, 0.05, 1, 0.85])
        ax.axis('off')
        render_pdf_standard_header(fig, title_text=f"{title} - Page {i+1}/{num_pages}")
        
        table = ax.table(cellText=chunk.values, colLabels=chunk.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8.5) 
        table.scale(1.0, 1.8) 
        
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white', fontsize=8.5)
                cell.set_facecolor('#1e293b')
            else:
                cell.set_linewidth(0.3)
                cell.set_facecolor('#f8fafc' if row % 2 == 0 else 'white')
                if col == 0:
                     cell.set_text_props(weight='bold', color='#2563eb')
        pdf.savefig(fig)
        plt.close(fig)

def generate_chart(df, ticker, result, pdf=None, end_date=None):
    # Truncate to end_date if specified
    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
        if df.index.tz is not None:
            end_ts = end_ts.tz_localize(df.index.tz)
        df = df[df.index <= end_ts].copy()
    # Ensure SMA columns exist (they may be missing if check_criteria worked on a copy)
    if 'SMA50' not in df.columns:
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA150'] = df['Close'].rolling(window=150).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        df['VolSMA50'] = df['Volume'].rolling(window=50).mean()
    plot_df = df.tail(150)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Candles
    width = 0.6
    width2 = 0.05
    up = plot_df[plot_df.Close >= plot_df.Open]
    down = plot_df[plot_df.Close < plot_df.Open]
    
    ax1.bar(up.index, up.Close - up.Open, width, bottom=up.Open, color='green', alpha=0.8)
    ax1.bar(up.index, up.High - up.Close, width2, bottom=up.Close, color='green')
    ax1.bar(up.index, up.Low - up.Open, width2, bottom=up.Open, color='green')
    
    ax1.bar(down.index, down.Close - down.Open, width, bottom=down.Open, color='red', alpha=0.8)
    ax1.bar(down.index, down.High - down.Open, width2, bottom=down.Open, color='red')
    ax1.bar(down.index, down.Low - down.Close, width2, bottom=down.Close, color='red')
    
    # MAs
    ax1.plot(plot_df.index, plot_df['SMA50'], color='blue', label='SMA 50')
    ax1.plot(plot_df.index, plot_df['SMA150'], color='purple', label='SMA 150')
    ax1.plot(plot_df.index, plot_df['SMA200'], color='black', label='SMA 200')
    
    # Levels
    ax1.axhline(result['Pivot'], color='magenta', linestyle='--', linewidth=1.5, label=f"Pivot ({result['Pivot']})")
    ax1.axhline(result['Target'], color='green', linestyle=':', linewidth=1.5, label=f"Target ({result['Target']})")
    ax1.axhline(result['Stop_Loss'], color='red', linestyle=':', linewidth=1.5, label=f"Stop ({result['Stop_Loss']})")
    
    title = (f"{ticker} - Mark Minervini VCP | Pullbacks: {str([float(v) for v in result['Pullbacks']])}%\n"
             f"EPS: {result['Qtr_EPS%']}%, Sales: {result['Qtr_Sales%']}%, RS: {result['RS_Rating']}, Vol R: {result['Vol_Ratio']}")
    ax1.set_title(title, fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Volume
    colors = ['green' if c >= o else 'red' for c, o in zip(plot_df.Close, plot_df.Open)]
    ax2.bar(plot_df.index, plot_df.Volume, color=colors, width=0.6, alpha=0.6)
    ax2.plot(plot_df.index, plot_df['VolSMA50'], color='orange', label='Vol SMA 50')
    ax2.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if pdf:
        pdf.savefig(fig)
    
    filename = f"{ticker.replace('.NS', '')}_minervini.png"
    save_path = os.path.join(CHARTS_DIR, filename)
    plt.savefig(save_path)
    plt.close(fig)
    return save_path

def main():
    parser = argparse.ArgumentParser(description="Mark Minervini VCP Screener")
    parser.add_argument('--limit', type=int, help="Limit number of stocks to scan")
    parser.add_argument('--sample', type=int, help="Run on a random sample")
    parser.add_argument('--refresh', action='store_true', help="Force refresh of cached data")
    parser.add_argument('--use-fundamentals', action='store_true', default=True, help="Enable fundamental and RS filters")
    parser.add_argument('--use-volume-dryup', action='store_true', default=False, help="Enable volume dry-up filter during contractions")
    parser.add_argument('--end-date', type=str, default=None, help="Run screener up to this date (YYYY-MM-DD). Uses all available data if not set.")
    args = parser.parse_args()
    
    if args.end_date:
        try:
            pd.Timestamp(args.end_date)
            console.print(f"[cyan]Running screener with end date: {args.end_date}[/cyan]")
        except ValueError:
            console.print(f"[red]Invalid date format: {args.end_date}. Use YYYY-MM-DD.[/red]")
            return
    
    setup_directories()
    tickers = load_tickers()
    
    if args.limit:
        tickers = tickers[:args.limit]
    if args.sample:
        import random
        tickers = random.sample(tickers, args.sample)
        
    rs_ratings = {}
    if args.use_fundamentals:
        rs_ratings = calculate_rs_ratings(tickers, end_date=args.end_date)
    # Get Market Condition
    market_status = "Unknown"
    try:
        with console.status("[blue]Fetching NIFTY 50 Market Condition...[/blue]"):
            nifty = yf.Ticker('^NSEI')
            nifty_df = nifty.history(period="2y")
            # Truncate to end_date if specified
            if args.end_date:
                end_ts = pd.Timestamp(args.end_date)
                if nifty_df.index.tz is not None:
                    end_ts = end_ts.tz_localize(nifty_df.index.tz)
                nifty_df = nifty_df[nifty_df.index <= end_ts]
            if len(nifty_df) >= 50:
                nifty_df['SMA50'] = nifty_df['Close'].rolling(window=50).mean()
                current_nifty = nifty_df['Close'].iloc[-1]
                sma50_nifty = nifty_df['SMA50'].iloc[-1]
                if current_nifty > sma50_nifty:
                    market_status = f"Bullish (Nifty: {current_nifty:.2f} > 50SMA: {sma50_nifty:.2f})"
                else:
                    market_status = f"Bearish (Nifty: {current_nifty:.2f} < 50SMA: {sma50_nifty:.2f})"
    except Exception as e:
        console.print(f"[yellow]Error fetching market condition: {e}[/yellow]")

    results = []
    
    def scan_ticker(ticker):
        """Scan a single ticker for VCP pattern (used by ThreadPoolExecutor)."""
        data = fetch_data(ticker, refresh=args.refresh)
        if data:
            rs = rs_ratings.get(ticker, 0) if args.use_fundamentals else 0
            match = check_criteria(data, ticker, rs, use_fundamentals=args.use_fundamentals, use_volume_dryup=args.use_volume_dryup, end_date=args.end_date)
            if match:
                return (match, data)
        return None
    
    console.print(f"[bold green]Scanning {len(tickers)} stocks against Minervini VCP (parallel)...[/bold green]")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(scan_ticker, t): t for t in tickers}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    match, data = result
                    console.print(f"[green]FOUND VCP: {match['Ticker']} (Pullbacks: {match['Pullbacks']}%) [/green]")
                    results.append((match, data))
            except Exception as e:
                ticker = futures[future]
                console.print(f"[yellow]Error scanning {ticker}: {e}[/yellow]")
    
    if results:
        with PdfPages(PDF_PATH) as pdf:
            create_pdf_title_page(pdf, TIMESTAMP, market_status)
            render_pdf_documentation_page(pdf)
            
            df_results = pd.DataFrame([r[0] for r in results])
            df_results['Pullbacks'] = df_results['Pullbacks'].apply(lambda x: str([float(v) for v in x]) if isinstance(x, list) else x)
            render_pdf_styled_table(pdf, df_results, "Minervini VCP Screener Results")
            
            for match, data in results:
                generate_chart(data['history'], match['Ticker'], match, pdf=pdf, end_date=args.end_date)
                
            df_results.to_csv(os.path.join(OUTPUT_DIR, 'results.csv'), index=False)
            console.print(f"\n[bold]Results saved to {OUTPUT_DIR}/results.csv[/bold]")
            console.print(f"[bold green]PDF Report saved to {PDF_PATH}[/bold green]")
    else:
        console.print("[yellow]No stocks found matching Minervini VCP criteria.[/yellow]")

if __name__ == "__main__":
    main()
