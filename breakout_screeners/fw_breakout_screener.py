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
Financial-Wisdom Breakout Stock Screener (Blueprint 2025)

This script implements an advanced breakout strategy combining technical precision, 
momentum signals, and fundamental 'Quality' overlays as defined in the Financial-Wisdom / Blueprint 2025 methodology.

Technical Criteria (The 'Setup'):
--------------------------------
- Timeframe: Weekly charts for high-conviction trend analysis.
- Trend: Price must be above the 20-week Simple Moving Average (SMA20).
- Momentum: MACD Line must be above the Signal Line.
- MACD Recency: Bullish crossover must have occurred within the last 3 weeks.
- Consolidation: Minimum 6 weeks of tight price action (NATR < 8.0).
- Breakout: Price must break above the 10-week high (closing prices).
- Conviction: > 30% volume increase compared to the previous week.
- Candle Structure: Upper wick must be < 50% of the total candle range (Low Selling Pressure).
- Breakout Size: 5% < Gain < 20% from previous week's close.

Fundamental Overlays (The 'Quality' Filters):
-------------------------------------------
The screener integrates CANSLIM fundamental rules to ensure high-quality vehicles:
- Return on Equity (ROE): >= 17%
- Return on Capital (ROC): >= 10%
- Operating Margin: >= 10%
- Current Qtr EPS Growth: >= 20%
- Current Qtr Sales Growth: >= 20%
- Annual EPS Growth: >= 25%
- Institutional Sponsorship: >= 10%

Risk Management & Selling Strategy:
----------------------------------
- Initial Hard Stop (Risk Control): Placed at the lower boundary of the 'Middle 
  Third' of the consolidation box. This ensures capital protection on breakout.
- Safety Cap: Maximum allowed risk per trade is 20%.
- Raised Stop (Profit Taking): The strategy uses Weekly MACD for trade management.
- Exit Trigger: If MACD crosses below the Signal Line at the end of the week, the
  stop loss is raised to the LOW (wick) of that weekly candle.
- Final Exit: The position is closed if the price breaches this raised stop in 
  subsequent weeks, ensuring you stay with the momentum.

Usage:
------
python3 fw_breakout_screener.py [--limit N] [--sample N] [--refresh]
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches

# Initialize Console (Dummy wrapper)
class Console:
    def print(self, text):
        # Remove rich tags for simple print
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
OUTPUT_DIR = os.path.join(BASE_DIR, 'screener_results', 'fw_breakouts', TIMESTAMP)
CHARTS_DIR = os.path.join(OUTPUT_DIR, 'charts')
CACHE_DIR = os.path.join(DATA_DIR, 'data_cache')
JSON_PATH = os.path.join(DATA_DIR, 'nifty_500.json')
PDF_PATH = os.path.join(OUTPUT_DIR, f"Financial_Wisdom_Report_{TIMESTAMP}.pdf")

# Strategy Parameters
MIN_HISTORY_YEARS = 3
LOOKBACK_WEEKS = 10  # For Breakout Level and Rolling Max
NATR_CONSOLIDATION_WEEKS = 6 # Minimum weeks of consolidation (NATR < 8)
NATR_PERIOD = 14
NATR_THRESHOLD = 8.0
SMA_PERIOD = 20
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MAX_RISK_PCT = 0.20  # Max risk 20% per trade (Blueprint Page 7)
# Breakout Criteria
WICK_RATIO_THRESHOLD = 0.50
GAIN_MIN = 0.05
GAIN_MAX = 0.20
VOLUME_MULTIPLIER = 1.30

# CANSLIM Fundamental Overlays (The 'Quality' filters)
MIN_EPS_GROWTH = 0.20        # Current Qtr EPS >= 20%
MIN_SALES_GROWTH = 0.20      # Current Qtr Sales >= 20%
MIN_ANNUAL_EPS_GROWTH = 0.25 # Annual EPS Growth >= 25%
MIN_ROE = 0.17               # Return on Equity >= 17%
MIN_ROC = 0.10               # Return on Capital >= 10% (Blueprint Page 5)
MIN_OPERATING_MARGIN = 0.10  # Operating Margin >= 10% (Blueprint Page 5)
MIN_INST_OWN = 0.10          # Institutional Sponsorship >= 10%

# Momentum Recency
MACD_RECENCY_WEEKS = 3       # MACD crossover should be recent (last 3 weeks)

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

def fetch_data(ticker, refresh=False):
    hist_cache_path = os.path.join(CACHE_DIR, f"{ticker}_1wk.csv")
    info_cache_path = os.path.join(CACHE_DIR, f"{ticker}_info.json")
    
    data = {}
    
    try:
        # 1. INFO (Fundamentals)
        if not refresh and os.path.exists(info_cache_path):
             with open(info_cache_path, 'r') as f:
                 data['info'] = json.load(f)
        else:
             stock = yf.Ticker(ticker)
             info = stock.info
             data['info'] = info
             with open(info_cache_path, 'w') as f:
                 json.dump(info, f)

        # 2. HISTORY (Weekly Technicals)
        if not refresh and os.path.exists(hist_cache_path):
            # console.print(f"[dim]Loading {ticker} from cache[/dim]")
            data['history'] = pd.read_csv(hist_cache_path, index_col=0, parse_dates=True)
        else:
            stock = yf.Ticker(ticker)
            df = stock.history(period=f"{MIN_HISTORY_YEARS}y", interval="1wk")
            if len(df) < 52:
                return None
            df.to_csv(hist_cache_path)
            data['history'] = df
            
        return data
    except Exception as e:
        # console.print(f"[yellow]Error fetching {ticker}: {e}[/yellow]")
        return None

def calculate_indicators(df):
    df = df.copy()
    
    # 1. SMA 20
    df['SMA20'] = df['Close'].rolling(window=SMA_PERIOD).mean()
    
    # 2. MACD
    ema_12 = df['Close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    # Note: MACD calculation in pandas varies slightly from some platforms due to initial values, but this is standard.
    df['Signal'] = df['MACD'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    
    # 3. ATR & NATR
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=NATR_PERIOD).mean()
    df['NATR'] = (df['ATR'] / df['Close']) * 100
    
    # NEW: Consolidated NATR Check
    # Ensure NATR has been < 8 for at least NATR_CONSOLIDATION_WEEKS
    # We check the MAX NATR over the consolidation window.
    df['MaxNATR'] = df['NATR'].rolling(window=NATR_CONSOLIDATION_WEEKS).max()

    # 4. Rolling Max High (Resistance) - Previous 10 weeks (shifted)
    # We want max high of [t-10 ... t-1], excluding t (current candle)
    df['RollingMaxHigh'] = df['High'].shift(1).rolling(window=LOOKBACK_WEEKS).max()
    df['RollingMaxClose'] = df['Close'].shift(1).rolling(window=LOOKBACK_WEEKS).max()
    # RollingMinLow for visual context (Consolidation Box Low)
    df['RollingMinLow'] = df['Low'].shift(1).rolling(window=LOOKBACK_WEEKS).min() 
    
    # 5. 52-Week High for momentum context
    df['Rolling52WeekHigh'] = df['High'].shift(1).rolling(window=52).max()
    
    return df

def check_criteria(data, ticker):
    df = data.get('history', None)
    info = data.get('info', {})
    
    if df is None or len(df) < SMA_PERIOD:
        return None
    
    # 1. FUNDAMENTALS (C, A, I Overlays)
    # ----------------------------------
    # C: Current Qtr EPS and Sales Growth must be >= 20%
    eps_growth = info.get('earningsQuarterlyGrowth', 0)
    sales_growth = info.get('revenueGrowth', 0)
    
    # A: Annual Earnings Growth >= 25% and ROE >= 17%
    annual_eps_growth = info.get('earningsGrowth', 0)
    roe = info.get('returnOnEquity', 0)
    roc = info.get('returnOnCapital', info.get('returnOnAssets', 0)) # Fallback to ROA if ROC missing
    op_margin = info.get('operatingMargins', 0)
    
    # I: Institutional Sponsorship
    inst_own = info.get('heldPercentInstitutions', 0)
    
    # Handle None safely
    eps_growth = eps_growth if eps_growth is not None else 0
    sales_growth = sales_growth if sales_growth is not None else 0
    annual_eps_growth = annual_eps_growth if annual_eps_growth is not None else 0
    roe = roe if roe is not None else 0
    roc = roc if roc is not None else 0
    op_margin = op_margin if op_margin is not None else 0
    inst_own = inst_own if inst_own is not None else 0
    
    # Fundamental Overlays (STRICT)
    if eps_growth < MIN_EPS_GROWTH: return None
    if sales_growth < MIN_SALES_GROWTH: return None
    if annual_eps_growth < MIN_ANNUAL_EPS_GROWTH: return None
    if roe < MIN_ROE: return None
    if roc < MIN_ROC: return None
    if op_margin < MIN_OPERATING_MARGIN: return None
    if inst_own < MIN_INST_OWN: return None

    # 2. TECHNICALS
    # -------------
    # Momentum Recency: MACD Crossover within last 3 weeks
    # Check if a bullish crossover (MACD > Signal) occurred recently
    if len(df) < MACD_RECENCY_WEEKS + 1:
        return None
        
    recent_df = df.tail(MACD_RECENCY_WEEKS + 1)
    had_crossover = False
    
    # Check for a 'cross-up' event in the recency window
    # We look for a week where MACD was below Signal and became above Signal
    for i in range(1, len(recent_df)):
        current_macdh = recent_df.iloc[i]['MACD_Hist']
        prev_macdh = recent_df.iloc[i-1]['MACD_Hist']
        if prev_macdh <= 0 and current_macdh > 0:
            had_crossover = True
            break
            
    if not had_crossover:
        return None

    # Get latest candle (Breakout Candle)
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Criteria 1: Trend (Price > 20 SMA)
    if not (curr['Close'] > curr['SMA20']):
        return None
        
    # Criteria 2: Momentum (MACD > Signal)
    if not (curr['MACD'] > curr['Signal']):
        return None
        
    # Criteria 3: Consolidation (NATR < 8 for consecutive weeks)
    # We check prev['MaxNATR'] because check_criteria is called on the LATEST candle (curr).
    # We want the consolidation to have happened BEFORE the breakout candle (curr).
    # So we check if the PREVIOUS candle (prev) had a MaxNATR (over last 6 weeks) < 8.
    if not (prev['MaxNATR'] < NATR_THRESHOLD):
        return None

    # Criteria 4: Breakout Execution
    if not (curr['Close'] > curr['RollingMaxHigh']):
        return None
    if not (curr['Close'] > curr['RollingMaxClose']):
        return None
        
    # Criteria 5: Breakout Candle Structure
    candle_range = curr['High'] - curr['Low']
    if candle_range == 0: return None
    
    upper_wick = curr['High'] - max(curr['Open'], curr['Close'])
    wick_ratio = upper_wick / candle_range
    
    if not (wick_ratio < WICK_RATIO_THRESHOLD):
        return None
        
    # Breakout Size: 5% < Gain < 20%
    prev_close = prev['Close']
    gain_pct = (curr['Close'] - prev_close) / prev_close
    
    if not (GAIN_MIN < gain_pct < GAIN_MAX):
        return None
        
    # Criteria 6: Volume
    if not (curr['Volume'] > prev['Volume'] * VOLUME_MULTIPLIER):
        return None
        
    # NEW: Advanced Risk Management (Blueprint 2025)
    # 1. Stop Loss: Middle Third of Consolidation Box (Blueprint Page 8, 12)
    box_high = curr['RollingMaxHigh']
    box_low = curr['RollingMinLow']
    box_height = box_high - box_low
    
    # Stop is "lower of the middle portion"
    stop_loss = box_low + (box_height / 3)
    
    # 2. Risk Check (Max 20%)
    risk_pct = (curr['Close'] - stop_loss) / curr['Close']
    if risk_pct > MAX_RISK_PCT:
        return None
        
    # Extra Info: 52-Week High (Momentum proxy)
    is_52wk_high = curr['Close'] > curr['Rolling52WeekHigh']
        
    return {
        'Ticker': ticker,
        'Date': df.index[-1].strftime('%Y-%m-%d'),
        'Close': curr['Close'],
        'Gain%': round(gain_pct * 100, 2),
        'Volume_Mult': round(curr['Volume'] / prev['Volume'], 2),
        'NATR': round(prev['NATR'], 2),
        'Breakout_Level': curr['RollingMaxHigh'],
        'SMA20': curr['SMA20'],
        'Entry_Price': curr['Close'],
        'Stop_Loss': round(stop_loss, 2),
        'Risk%': round(risk_pct * 100, 2),
        '52W_High': is_52wk_high,
        'Qtr_EPS%': round(eps_growth * 100, 2),
        'Ann_EPS%': round(annual_eps_growth * 100, 2),
        'ROE%': round(roe * 100, 2),
        'ROC%': round(roc * 100, 2),
        'Op_Margin%': round(op_margin * 100, 2),
        'Inst_Own%': round(inst_own * 100, 2)
    }

def generate_chart(df, ticker, result, pdf=None):
    # Slice last 52 weeks for visibility
    plot_df = df.tail(52)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [4, 1, 1.5]}, sharex=True)
    
    # Plot Candles
    width = 0.6
    width2 = 0.05
    
    up = plot_df[plot_df.Close >= plot_df.Open]
    down = plot_df[plot_df.Close < plot_df.Open]
    
    # Up candles
    ax1.bar(up.index, up.Close - up.Open, width, bottom=up.Open, color='green', alpha=0.8)
    ax1.bar(up.index, up.High - up.Close, width2, bottom=up.Close, color='green')
    ax1.bar(up.index, up.Low - up.Open, width2, bottom=up.Open, color='green')
    
    # Down candles
    ax1.bar(down.index, down.Close - down.Open, width, bottom=down.Open, color='red', alpha=0.8)
    ax1.bar(down.index, down.High - down.Open, width2, bottom=down.Open, color='red')
    ax1.bar(down.index, down.Low - down.Close, width2, bottom=down.Close, color='red')
    
    # Plot SMA 20
    ax1.plot(plot_df.index, plot_df['SMA20'], color='blue', label='SMA 20', linewidth=1.5)
    
    # Plot Resistance Line (Breakout Level)
    # Using the breakout level from the result (which is RollingMaxHigh of the breakout candle)
    breakout_level = result['Breakout_Level']
    ax1.axhline(y=breakout_level, color='purple', linestyle='--', label=f'Resistance ({breakout_level:.2f})')
    
    # Plot Stop Loss Line
    stop_loss = result['Stop_Loss']
    ax1.axhline(y=stop_loss, color='red', linestyle=':', label=f'Stop Loss ({stop_loss:.2f})')
    
    # Annotate Entry and Stop
    ax1.text(plot_df.index[-1], result['Entry_Price'], f"Entry: {result['Entry_Price']:.2f}", 
             color='green', fontsize=10, verticalalignment='bottom', fontweight='bold')
    ax1.text(plot_df.index[-1], stop_loss, f"Stop: {stop_loss:.2f}", 
             color='red', fontsize=10, verticalalignment='top', fontweight='bold')
    
    # Plot Consolidation Box
    # We'll draw it for the LOOKBACK_WEEKS period ending at the previous candle
    # Box High = RollingMaxHigh, Box Low = RollingMinLow
    last_idx = plot_df.index.get_loc(plot_df.index[-1])
    try:
        # Get bounds based on index
        start_date = plot_df.index[-LOOKBACK_WEEKS - 1]
        end_date = plot_df.index[-2] # Previous candle
        
        # Recalculate box limits for this specific window to be precise
        box_window = df.iloc[-LOOKBACK_WEEKS-1:-1] # -1 excludes current
        box_high = breakout_level
        box_low = box_window['Low'].min()
        
        # Create Rectangle Patch
        # Matplotlib dates handling can be tricky, using index numbers might be safer but dates work if formatted?
        # Let's use axvspan limits or simply plotting a rectangle requires numerical coordinates
        # Simplified: Just draw lines or using a span
        
        rect_start = up.index[-LOOKBACK_WEEKS-1] # Approx
        # Better: use ax1.add_patch with date2num if needed, but let's stick to simple Hlines/Vlines or Span
        
        # High/Low Box lines
        ax1.hlines(y=[box_high, box_low], xmin=start_date, xmax=end_date, colors='orange', linestyles='dotted', linewidth=2)
        ax1.vlines(x=[start_date, end_date], ymin=box_low, ymax=box_high, colors='orange', linestyles='dotted', linewidth=2)
        
    except Exception as e:
        print(f"Error drawing box: {e}")

    title = (f"{ticker} - Weekly Breakout Setup\n"
             f"ROE: {result['ROE%']}%, ROC: {result['ROC%']}%, Margin: {result['Op_Margin%']}%\n"
             f"Qtr EPS: {result['Qtr_EPS%']}%, Ann EPS: {result['Ann_EPS%']}%, Inst Own: {result['Inst_Own%']}%\n"
             f"Entry: {result['Entry_Price']:.2f}, Stop: {result['Stop_Loss']:.2f}, Risk: {result['Risk%']}%")
    ax1.set_title(title, fontsize=12, fontweight='bold')
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Volume
    colors = ['green' if c >= o else 'red' for c, o in zip(plot_df.Close, plot_df.Open)]
    ax2.bar(plot_df.index, plot_df.Volume, color=colors, width=0.6, alpha=0.6)
    
    # Volume Threshold Line
    breakout_vol = plot_df.iloc[-1]['Volume']
    prev_vol = plot_df.iloc[-2]['Volume']
    threshold = prev_vol * VOLUME_MULTIPLIER
    ax2.axhline(y=threshold, color='orange', linestyle='--', label='1.3x Vol Threshold')
    
    ax2.set_ylabel("Volume")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot MACD (Panel 3)
    ax3.plot(plot_df.index, plot_df['MACD'], label='MACD Line', color='blue', linewidth=1.5)
    ax3.plot(plot_df.index, plot_df['Signal'], label='Signal Line', color='orange', linewidth=1.5)
    
    # Histogram
    hist_colors = ['green' if h >= 0 else 'red' for h in plot_df['MACD_Hist']]
    ax3.bar(plot_df.index, plot_df['MACD_Hist'], color=hist_colors, alpha=0.5, width=0.6, label='Histogram')
    
    ax3.set_title("MACD Analysis (Momentum)", fontsize=10, fontweight='bold')
    ax3.set_ylabel("MACD")
    ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    # Zero line
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if pdf:
        pdf.savefig(fig)
    
    filename = f"{ticker.replace('.NS', '')}_breakout.png"
    save_path = os.path.join(CHARTS_DIR, filename)
    plt.savefig(save_path)
    plt.close(fig)
    return save_path

def render_pdf_standard_header(fig, title_text="Stock Analysis Report"):
    """Adds the professional header and separator line to a page."""
    # Colors
    PRIMARY_COLOR = '#1e293b'  # Slate 800
    ACCENT_COLOR = '#2563eb'   # Blue 600
    BORDER_COLOR = '#94a3b8'   # Slate 400
    
    # Header Title
    fig.text(0.5, 0.96, title_text, ha='center', va='center', 
             fontsize=16, weight='bold', color=PRIMARY_COLOR)
    
    # Copyright Subtitle
    copyright_text = f"© {datetime.now().year} Stock Research. All Rights Reserved."
    fig.text(0.5, 0.935, copyright_text, ha='center', va='center', 
             fontsize=9, style='italic', color=BORDER_COLOR)
    
    # Separator Line
    from matplotlib.lines import Line2D
    line = Line2D([0.08, 0.92], [0.91, 0.91], transform=fig.transFigure, 
                  color=BORDER_COLOR, linewidth=1.0, alpha=0.5)
    fig.add_artist(line)

def create_pdf_title_page(pdf, timestamp):
    """Creates a professional title page for the PDF report."""
    # Use PORTRAIT for title page
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    
    render_pdf_standard_header(fig)
    
    # Professional Colors
    PRIMARY_COLOR = '#1e293b'
    ACCENT_COLOR = '#2563eb'
    SECONDARY_COLOR = '#475569'
    
    # Main Title
    fig.text(0.5, 0.65, "Financial-Wisdom", 
             ha='center', va='center', fontsize=42, weight='bold', color=ACCENT_COLOR)
    fig.text(0.5, 0.58, "Breakout Screener Report", 
             ha='center', va='center', fontsize=24, weight='bold', color=SECONDARY_COLOR)
    
    # Analysis Subtitle
    fig.text(0.5, 0.50, f"Strategic Market Scan: {timestamp}", 
             ha='center', va='center', fontsize=16, color=SECONDARY_COLOR)
    
    # Strategy Highlights Box
    fig.text(0.15, 0.35, "Strategy Framework", fontsize=16, weight='bold', color=PRIMARY_COLOR)
    highlights = [
        "● Technical Breakout Precision",
        "● CANSLIM Quality Overlays",
        "● 3-Panel Momentum Confirmation",
        "● Middle-Third Risk Management"
    ]
    for i, s in enumerate(highlights):
        fig.text(0.15, 0.31 - (i * 0.04), s, fontsize=12, color=SECONDARY_COLOR)
        
    pdf.savefig(fig)
    plt.close(fig)

def render_pdf_documentation_page(pdf):
    """Adds a documentation page explaining the methodology and rules."""
    # Use PORTRAIT for text-heavy methodology
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    
    render_pdf_standard_header(fig)
    
    # Standard Professional Colors and Margins
    PRIMARY_COLOR = '#1e293b'
    SECONDARY_COLOR = '#475569'
    ACCENT_COLOR = '#dc2626' # Red for bullets
    LEFT_MARGIN = 0.08
    
    # Section Header
    fig.text(LEFT_MARGIN, 0.86, "FW Breakout Screener Methodology", 
             ha='left', va='top', fontsize=20, weight='bold', color=PRIMARY_COLOR)
    
    # Methodology Content
    methodology_text = """
This script implements an advanced breakout strategy combining technical precision, 
momentum signals, and fundamental 'Quality' overlays as defined in the 
Financial-Wisdom / Blueprint 2025 methodology.

Step 1: Technical Setup (The Screening)
----------------------------------------
- Timeframe: Weekly charts for high-conviction trend analysis.
- Trend: Price must be above the 20-week Simple Moving Average (SMA20).
- Momentum: MACD Line must be above the Signal Line.
- MACD Recency: Bullish crossover must have occurred within the last 3 weeks.
- Consolidation: Minimum 6 weeks of tight price action (NATR < 8.0).
- Breakout: Price must break above the 10-week high (closing prices).
- Conviction: > 30% volume increase compared to the previous week.
- Candle Structure: Upper wick < 50% of the total candle (Low Selling Pressure).
- Breakout Size: 5% < Gain < 20% from previous week's close.

Step 2: Quality Overlays (The Filters)
---------------------------------------
The screener integrates CANSLIM fundamental rules to ensure high-quality:
- Return on Equity (ROE) >= 17%
- Return on Capital (ROC) >= 10%
- Operating Margin >= 10%
- Current Qtr EPS Growth >= 20%
- Current Qtr Sales Growth >= 20%
- Annual EPS Growth >= 25%
- Institutional Sponsorship >= 10%

Step 3: Risk & Selling Strategy
---------------------------------
- Initial Hard Stop: Placed at the lower boundary of the 'Middle Third' of the 
  consolidation box. This ensures capital protection on breakout.
- Safety Cap: Maximum allowed risk per trade is 20%.
- Exit Trigger: If MACD crosses below the Signal Line at the end of the week, the
  stop loss is raised to the LOW (wick) of that weekly candle.
- Final Exit: The position is closed if the price breaches this raised stop.
"""
    
    # Split into lines and render
    lines = methodology_text.strip().split('\n')
    y = 0.81
    for line in lines:
        if y < 0.07: # Simple page break check
            pdf.savefig(fig)
            plt.clf()
            plt.axis('off')
            render_pdf_standard_header(fig)
            y = 0.85
            
        stripped_line = line.strip()
        is_header = stripped_line.endswith(':') or stripped_line.startswith('---')
        
        # Also check if the line *above* dashed line is a header
        if str(stripped_line).startswith('Step'):
             is_header = True
             
        if is_header:
            if not str(stripped_line).startswith('-'):
                 fig.text(LEFT_MARGIN, y, stripped_line, fontsize=13, weight='bold', color=PRIMARY_COLOR)
                 y -= 0.035
        elif stripped_line == "":
            y -= 0.015
        else:
            # Bullet point handling
            if stripped_line.startswith('-'):
                fig.text(LEFT_MARGIN, y, "●", fontsize=10, color=ACCENT_COLOR)
                fig.text(LEFT_MARGIN + 0.03, y, line[1:].strip(), fontsize=10.5, color=SECONDARY_COLOR)
            else:
                fig.text(LEFT_MARGIN, y, line, fontsize=10.5, color=SECONDARY_COLOR)
            y -= 0.028
            
    # Add Page Info to Footer
    fig.text(0.5, 0.03, f"Page 2/N", ha='center', fontsize=9, color='#94a3b8')
    fig.text(0.92, 0.03, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ha='right', fontsize=9, color='#94a3b8')
            
    pdf.savefig(fig)
    plt.close(fig)

def render_pdf_styled_table(pdf, df, title):
    """Renders a dataframe as a styled table in the PDF."""
    if df.empty:
        return

    rows_per_page = 22
    num_pages = (len(df) // rows_per_page) + 1
    
    for i in range(num_pages):
        start_idx = i * rows_per_page
        end_idx = min((i + 1) * rows_per_page, len(df))
        chunk = df.iloc[start_idx:end_idx].copy()
        
        # Format floats for table
        for col in chunk.columns:
            if chunk[col].dtype == 'float64':
                chunk[col] = chunk[col].map('{:.1f}'.format)
        
        # Abbreviate long headers to prevent overlap
        header_map = {
            'Entry_Price': 'Entry', 
            'Breakout_Level': 'Breakout', 
            'Stop_Loss': 'Stop',
            'Risk%': 'Risk%',
            'Gain%': 'Gain%',
            'Op_Margin%': 'Margin%',
            'Qtr_EPS%': 'Q_EPS%',
            'Ann_EPS%': 'A_EPS%',
            'Inst_Own%': 'Inst%',
            'Volume_Mult': 'Vol_Mult',
            '52W_High': '52W_H'
        }
        chunk = chunk.rename(columns=header_map)
        
        # INCREASE WIDTH to 14 inches to prevent overlapping
        fig = plt.figure(figsize=(14, 8.5)) 
        ax = fig.add_axes([0, 0.05, 1, 0.85])
        ax.axis('off')
        
        # Professional Table Header
        render_pdf_standard_header(fig, title_text=f"{title} - Page {i+1}/{num_pages}")
        
        # Recalculate column widths for 14-inch width
        num_cols = len(chunk.columns)
        ticker_width = 0.12 # Reduced from 0.15 since page is wider
        other_width = (1.0 - ticker_width) / (num_cols - 1)
        col_widths = [ticker_width] + [other_width] * (num_cols - 1)
        
        table = ax.table(cellText=chunk.values, colLabels=chunk.columns, 
                        loc='center', cellLoc='center', colWidths=col_widths)
        
        table.auto_set_font_size(False)
        table.set_fontsize(8.5) # Can afford slightly larger font now
        table.scale(1.0, 1.8) 
        
        # Colors (Slate 800)
        PRIMARY_COLOR = '#1e293b'
        
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white', fontsize=8.5)
                cell.set_facecolor(PRIMARY_COLOR)
                cell.set_linewidth(0.5)
            else:
                cell.set_text_props(color='#475569') # Secondary text color
                cell.set_linewidth(0.3)
                cell.set_facecolor('#f8fafc' if row % 2 == 0 else 'white') # Zebra striping
                if col == 0: # Ticker column accent
                     cell.set_text_props(weight='bold', color='#2563eb')
        
        pdf.savefig(fig)
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Stock Breakout Screener")
    parser.add_argument('--limit', type=int, help="Limit number of stocks to scan")
    parser.add_argument('--sample', type=int, help="Run on a random sample")
    parser.add_argument('--refresh', action='store_true', help="Force refresh of cached data")
    args = parser.parse_args()
    
    setup_directories()
    tickers = load_tickers()
    
    if args.limit:
        tickers = tickers[:args.limit]
    if args.sample:
        import random
        tickers = random.sample(tickers, args.sample)
        
    results = []
    
    with console.status(f"[bold green]Scanning {len(tickers)} stocks...[/bold green]") as status:
        for ticker in tickers:
            data = fetch_data(ticker, refresh=args.refresh)
            if data is not None and data.get('history') is not None:
                # Calculate indicators on the history df
                data['history'] = calculate_indicators(data['history'])
                
                # Check criteria on the full data dict
                match = check_criteria(data, ticker)
                if match:
                    console.print(f"[green]FOUND: {ticker} - Gain: {match['Gain%']}% - ROE: {match['ROE%']}%[/green]")
                    # Store (match, data) for PDF generation
                    results.append((match, data))
                    
    # PDF Output Generation
    with PdfPages(PDF_PATH) as pdf:
        # 1. Title Page
        console.print("[blue]Generating PDF Title Page...[/blue]")
        create_pdf_title_page(pdf, TIMESTAMP)
        
        # 2. Methodology Page
        console.print("[blue]Generating Methodology Page...[/blue]")
        render_pdf_documentation_page(pdf)
        
        if results:
            # 3. Aggregated Summary Table
            df_results = pd.DataFrame([r[0] for r in results])
            # Reorder columns
            cols = ['Ticker', 'Date', 'Entry_Price', 'Breakout_Level', 'Stop_Loss', 'Risk%', 'Gain%', 'ROE%', 'ROC%', 'Op_Margin%', 'Qtr_EPS%', 'Ann_EPS%', 'Inst_Own%', 'Volume_Mult', 'NATR', '52W_High']
            if all(c in df_results.columns for c in cols):
                 df_results = df_results[cols]
            
            # Round all floats to 1 decimal place
            df_results = df_results.round(1)
                 
            console.print("[blue]Generating Summary Table...[/blue]")
            render_pdf_styled_table(pdf, df_results, "Breakout Screener Summary Results")
            
            # 4. Individual Stock Detail Pages (1 per chart)
            console.print(f"[blue]Generating {len(results)} individual stock pages...[/blue]")
            for match, data in results:
                generate_chart(data['history'], match['Ticker'], match, pdf=pdf)
            
            # Finalize Output
            print("\nBreakout Screener Results Summary:")
            print(df_results.to_string(index=False))
            
            # CSV backup for data portability
            df_results.to_csv(os.path.join(OUTPUT_DIR, 'results.csv'), index=False)
            console.print(f"\n[bold green]Report saved to {PDF_PATH}[/bold green]")
            console.print(f"[bold]CSV Backup: {OUTPUT_DIR}/results.csv[/bold]")
        else:
            console.print("[yellow]No stocks found criteria matching criteria.[/yellow]")
            # Fallback Page
            fig = plt.figure(figsize=(11, 8.5))
            plt.axis('off')
            plt.text(0.5, 0.5, "No stocks found matching the breakout criteria.", 
                     ha='center', va='center', fontsize=20, color='#64748b')
            pdf.savefig(fig)
            plt.close(fig)

if __name__ == "__main__":
    main()
