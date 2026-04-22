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
Phase 1: The Macro Setup (Trend Template)
- Current Price > 50-day, 150-day, and 200-day SMAs.
- 50-day SMA > 150-day SMA > 200-day SMA.
- 200-day SMA is rising (compared to 20 days ago).
- Current Price is at least 30% above its 52-week low.
- Current Price is within 25% of its 52-week high.
- Relative Strength: Percentile rank of 1-year returns >= 70.

Integrate Fundamental Filters (SEPA):
- EPS Growth: Most recent quarter EPS > 20% YoY.
- Revenue Growth: Most recent quarter Sales > 15% YoY.

Phase 2: Identifying the Contractions (The Swings)
- Progressive pullbacks: Pullback_1 > Pullback_2 > Pullback_3 etc.
- Use scipy.signal.find_peaks to find local highs and lows.

Phase 3: Volume Dry-Up
- Volume shrinks as contractions get smaller.

Phase 4: The Pivot Point and Breakout
- Pivot Point = Peak of final contraction.
- Entry Trigger: Current Close > Pivot Point.
- Volume Confirmation: Breakout volume >= 150% of 50-day average.
- Stop Loss: Slightly below the low of the final contraction.

Usage:
------
python minervini_screener.py [--limit N] [--sample N] [--refresh]
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

def calculate_rs_ratings(tickers):
    """Calculate strong 1-99 True Relative Strength rank across our universe."""
    returns = {}
    with console.status(f"[blue]Calculating Relative Strength rankings for {len(tickers)} stocks...[/blue]"):
        for ticker in tickers:
            cache_path = os.path.join(CACHE_DIR, f"{ticker}_1d.csv")
            if os.path.exists(cache_path):
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                if len(df) > 200:
                    try:
                        ret_1y = (df['Close'].iloc[-1] - df['Close'].iloc[-252]) / df['Close'].iloc[-252]
                        returns[ticker] = ret_1y
                    except IndexError:
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

def check_criteria(data, ticker, rs_rating, use_fundamentals=False, use_volume_dryup=False):
    info = data.get('info', {})
    df = data.get('history', None)
    
    if df is None or df.empty or len(df) < 200:
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
    # Use closing prices for peak finding to be conservative
    closes = df['Close'].tail(90).values
    if use_volume_dryup:
        volumes = df['Volume'].tail(90).values
    peaks, _ = scipy.signal.find_peaks(closes, distance=10)
    troughs, _ = scipy.signal.find_peaks(-closes, distance=10)
    
    if len(peaks) < 2 or len(troughs) < 2:
        return None # Not enough swings
        
    # Align peaks and troughs to calculate depths
    # We want Peak -> Trough sequence
    pullbacks = []
    pullback_vols = []
    for p in peaks:
        # Find the first trough after this peak
        valid_troughs = troughs[troughs > p]
        if len(valid_troughs) > 0:
            t = valid_troughs[0]
            peak_val = closes[p]
            trough_val = closes[t]
            depth = (peak_val - trough_val) / peak_val
            pullbacks.append(depth)
            
            if use_volume_dryup:
                # Calculate average volume during this pullback period
                avg_vol = np.mean(volumes[p:t+1])
                pullback_vols.append(avg_vol)
            
    if len(pullbacks) < 2:
        return None
        
    # Check for progressive contraction (getting smaller)
    # E.g., depth1 > depth2 > depth3
    # We check the last few pullbacks
    recent_pullbacks = pullbacks[-3:] if len(pullbacks) >= 3 else pullbacks
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

    # --- 4. Volume Dry-Up and Breakout ---
    # Pivot Point is the peak of the final contraction
    final_peak_idx = peaks[-1]
    pivot_point = closes[final_peak_idx]
    
    # Entry Trigger: Today's close > Pivot
    if current_close <= pivot_point:
        return None
        
    # Volume Confirmation: Breakout vol >= 150% of 50-day avg
    if current_vol < vol_sma_50 * 1.5:
        return None
        
    # Stop Loss: Slightly below the low of the final contraction
    # Find trough after final peak or use the most recent trough
    final_troughs = troughs[troughs > final_peak_idx]
    if len(final_troughs) > 0:
        stop_loss = closes[final_troughs[0]] * 0.98 # 2% below the low
    else:
        stop_loss = closes[troughs[-1]] * 0.98 if len(troughs) > 0 else current_close * 0.92
        
    # Target (Default to 25% similar to CANSLIM or fixed multiplier)
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

def create_pdf_title_page(pdf, timestamp):
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    render_pdf_standard_header(fig, "Mark Minervini VCP Screener")
    PRIMARY_COLOR = '#1e293b'
    ACCENT_COLOR = '#2563eb'
    SECONDARY_COLOR = '#475569'
    
    fig.text(0.5, 0.65, "Mark Minervini VCP", ha='center', va='center', fontsize=36, weight='bold', color=ACCENT_COLOR)
    fig.text(0.5, 0.58, "Volatility Contraction Pattern Strategy", ha='center', va='center', fontsize=22, weight='bold', color=SECONDARY_COLOR)
    fig.text(0.5, 0.52, f"Analysis Report: {timestamp}", ha='center', va='center', fontsize=14, color=SECONDARY_COLOR)
    
    fig.text(0.15, 0.35, "VCP Core Principles", fontsize=16, weight='bold', color=PRIMARY_COLOR)
    strengths = [
        "● Focus on Stage 2 Uptrends",
        "● Visual pattern of price contraction",
        "● Volume dry-up indicating supply exhaustion",
        "● Breakout on high volume trigger"
    ]
    for i, s in enumerate(strengths):
        fig.text(0.15, 0.31 - (i * 0.04), s, fontsize=12, color=SECONDARY_COLOR)

    pdf.savefig(fig)
    plt.close(fig)

def render_pdf_documentation_page(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    render_pdf_standard_header(fig, "VCP Methodology & Rules")
    PRIMARY_COLOR = '#1e293b'
    SECONDARY_COLOR = '#475569'
    LEFT_MARGIN = 0.08
    
    fig.text(LEFT_MARGIN, 0.86, "Mark Minervini VCP Rules", fontsize=20, weight='bold', color=PRIMARY_COLOR)
    
    text = """
Phase 1: Macro Trend (Template)
- Price above 50, 150, 200 SMAs.
- 50 SMA > 150 SMA > 200 SMA.
- 200 SMA rising for at least 20 days.
- Price at least 30% above 52W low.
- Price within 25% of 52W high.
- RS Rating >= 70 (Top 30% of market).

Phase 2: Fundamental Filters
- Quarterly EPS growth >= 20%.
- Quarterly Sales growth >= 15%.

Phase 3: Price Contraction (The Swings)
- Progressive tightening of price swings.
- Pullback depths must decrease from left to right.

Phase 4: Breakout & Entry
- Breakout day volume >= 150% of 50-day average.
- Stop loss placed below low of final contraction.
"""
    fig.text(LEFT_MARGIN, 0.80, text, ha='left', va='top', fontsize=11, color=SECONDARY_COLOR, linespacing=1.6)
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

def generate_chart(df, ticker, result, pdf=None):
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
    parser.add_argument('--use-fundamentals', action='store_true', default=False, help="Enable fundamental and RS filters")
    parser.add_argument('--use-volume-dryup', action='store_true', default=False, help="Enable volume dry-up filter during contractions")
    args = parser.parse_args()
    
    setup_directories()
    tickers = load_tickers()
    
    if args.limit:
        tickers = tickers[:args.limit]
    if args.sample:
        import random
        tickers = random.sample(tickers, args.sample)
        
    rs_ratings = {}
    if args.use_fundamentals:
        rs_ratings = calculate_rs_ratings(tickers)
    results = []
    
    with console.status(f"[bold green]Scanning {len(tickers)} stocks against Minervini VCP...[/bold green]"):
        for ticker in tickers:
            data = fetch_data(ticker, refresh=args.refresh)
            if data:
                rs = rs_ratings.get(ticker, 0) if args.use_fundamentals else 0
                match = check_criteria(data, ticker, rs, use_fundamentals=args.use_fundamentals, use_volume_dryup=args.use_volume_dryup)
                if match:
                    console.print(f"[green]FOUND VCP: {ticker} (Pullbacks: {match['Pullbacks']}%) [/green]")
                    results.append((match, data))
    
    if results:
        with PdfPages(PDF_PATH) as pdf:
            create_pdf_title_page(pdf, TIMESTAMP)
            render_pdf_documentation_page(pdf)
            
            df_results = pd.DataFrame([r[0] for r in results])
            df_results['Pullbacks'] = df_results['Pullbacks'].apply(lambda x: str([float(v) for v in x]) if isinstance(x, list) else x)
            render_pdf_styled_table(pdf, df_results, "Minervini VCP Screener Results")
            
            for match, data in results:
                generate_chart(data['history'], match['Ticker'], match, pdf=pdf)
                
            df_results.to_csv(os.path.join(OUTPUT_DIR, 'results.csv'), index=False)
            console.print(f"\n[bold]Results saved to {OUTPUT_DIR}/results.csv[/bold]")
            console.print(f"[bold green]PDF Report saved to {PDF_PATH}[/bold green]")
    else:
        console.print("[yellow]No stocks found matching Minervini VCP criteria.[/yellow]")

if __name__ == "__main__":
    main()
