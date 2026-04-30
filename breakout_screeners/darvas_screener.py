# -------------------------------------------------------------------------------
# Project: Stock Analysis
# Author:  Anshul Solanki 
# License: MIT License
# -------------------------------------------------------------------------------

"""
Nicolas Darvas Box Strategy Screener

This script implements Nicolas Darvas's Box strategy including:
1. Darvas Box detection (Ceiling and Floor).
2. Volume and Proximity to 52-week high filters.
3. 200-day EMA trend filter.

Usage:
------
python darvas_screener.py [--limit N] [--sample N] [--refresh]
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Initialize Console (Dummy wrapper for clean output)
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
OUTPUT_DIR = os.path.join(BASE_DIR, 'screener_results', 'darvas_boxes', TIMESTAMP)
CHARTS_DIR = os.path.join(OUTPUT_DIR, 'charts')
CACHE_DIR = os.path.join(DATA_DIR, 'data_cache')
JSON_PATH = os.path.join(DATA_DIR, 'nifty_500.json')
PDF_PATH = os.path.join(OUTPUT_DIR, f"darvas_Screener_Results_{TIMESTAMP}.pdf")

def setup_directories():
    """Create necessary directories for output, charts, and data cache."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

def load_tickers(limit=None):
    """Load tickers from the Nifty 500 JSON file."""
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
    """Fetch historical data and info for a ticker, using local cache if available."""
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
            if len(df) < 250: return None # Need enough data for 52w high and 200 EMA
            df.to_csv(hist_cache_path)
            data['history'] = df
            
        return data
    except Exception as e:
        return None

def detect_darvas_box(df, fifty_two_week_high, lookback=40):
    """
    Detects the most recent Darvas Box.
    
    Rules:
    1. Establish Top: High holds for 3 days.
    2. Establish Bottom: Low holds for 3 days without breaking top.
    """
    prices = df.tail(lookback)
    
    for i in range(len(prices) - 4, 3, -1):
        # Check for ceiling
        is_ceiling = True
        ceiling_candidate = prices['High'].iloc[i]
        
        # Constraint: Ceiling must be within 2% of 52-week high
        if (fifty_two_week_high - ceiling_candidate) / fifty_two_week_high > 0.02:
            continue
            
        for j in range(1, 4):
            if prices['High'].iloc[i-j] > ceiling_candidate or prices['High'].iloc[i+j] > ceiling_candidate:
                is_ceiling = False
                break
        
        if is_ceiling:
            # Look for floor after ceiling is established
            for k in range(i + 1, len(prices) - 3):
                is_floor = True
                floor_candidate = prices['Low'].iloc[k]
                if floor_candidate > ceiling_candidate:
                    continue
                
                for l in range(1, 4):
                    if prices['Low'].iloc[k-l] < floor_candidate or prices['Low'].iloc[k+l] < floor_candidate:
                        is_floor = False
                        break
                    if prices['High'].iloc[k+l] > ceiling_candidate:
                        is_floor = False
                        break
                
                if is_floor:
                    return ceiling_candidate, floor_candidate, prices.index[i]
                    
    return None, None, None

def check_criteria(data, ticker):
    """Evaluate a stock against Nicolas Darvas strategy criteria."""
    info = data.get('info', {})
    df = data.get('history', None)
    
    if df is None or df.empty or len(df) < 250:
        return None
        
    curr = df.iloc[-1]
    
    # 1. Moving Average Alignment Filter (Price > 50 > 150 > 200 SMA)
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA150'] = df['Close'].rolling(window=150).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    
    if not (curr['Close'] > df['SMA50'].iloc[-1] and 
            df['SMA50'].iloc[-1] > df['SMA150'].iloc[-1] and 
            df['SMA150'].iloc[-1] > df['SMA200'].iloc[-1]):
        return None
        
    # 2. Proximity to 52-Week High Filter (Within 5-10%)
    fifty_two_week_high = df['High'].rolling(window=252, min_periods=1).max().iloc[-1]
    distance_from_high = (fifty_two_week_high - curr['Close']) / fifty_two_week_high
    
    # The PDF says "Within 5%-10% of High". I interpret this as distance <= 10%.
    if distance_from_high > 0.10:
        return None
        
    # 3. Volume Filter (> 150% of 30-day average)
    df['VolSMA30'] = df['Volume'].rolling(window=30).mean()
    if curr['Volume'] <= df['VolSMA30'].iloc[-1] * 1.5:
        return None
        
    # 4. Darvas Box Detection
    ceiling, floor, box_start_date = detect_darvas_box(df, fifty_two_week_high)
    if ceiling is None or floor is None:
        return None
        
    # Breakout trigger: Close above ceiling
    # Or we are in the box and looking for breakout.
    # The PDF says "Buy only when the price breaks above the top of the box on high volume."
    # So if we are currently breaking out:
    is_breakout = curr['Close'] > ceiling and df['Close'].iloc[-2] <= ceiling
    
    # If not breaking out today, but in a valid box:
    is_in_box = curr['Close'] <= ceiling and curr['Close'] >= floor
    
    if not (is_breakout or is_in_box):
        return None
        
    setup_type = 'Breakout' if is_breakout else 'In Box'
    
    stop_loss = floor # Or just below ceiling as per risk tolerance
    box_width = ceiling - floor
    target_price = ceiling + box_width
    
    risk = curr['Close'] - stop_loss
    reward = target_price - curr['Close']
    risk_reward = reward / risk if risk > 0 else 0
    
    box_age_days = (df.index[-1] - box_start_date).days if box_start_date else 0
    box_tightness_pct = (ceiling - floor) / ceiling * 100
    
    # VCP detection
    box_df = df.loc[box_start_date:]
    box_length = len(box_df)
    vcp = 'No'
    if box_length >= 10:
        half_len = box_length // 2
        first_half = box_df.iloc[:half_len]
        last_5 = box_df.tail(5)
        
        range_first_half = (first_half['High'].max() - first_half['Low'].min()) / first_half['High'].max()
        range_last_5 = (last_5['High'].max() - last_5['Low'].min()) / last_5['High'].max()
        
        if range_last_5 < 0.5 * range_first_half:
            vcp = 'Yes'
        
    return {
        'Ticker': ticker,
        'Date': df.index[-1].strftime('%Y-%m-%d'),
        'Close': round(curr['Close'], 1),
        'Setup': setup_type,
        'Ceiling': round(ceiling, 1),
        'Floor': round(floor, 1),
        'Box_Start': box_start_date.strftime('%Y-%m-%d') if box_start_date else None,
        'Breakout_Date': df.index[-1].strftime('%Y-%m-%d') if is_breakout else None,
        'Box_Age': box_age_days,
        'Box_Tightness%': round(box_tightness_pct, 2),
        'VCP': vcp,
        'Dist_52W_High%': round(distance_from_high * 100, 2),
        'Vol_Ratio': round(curr['Volume'] / df['VolSMA30'].iloc[-1], 2),
        'Target': round(target_price, 2),
        'Stop_Loss': round(stop_loss, 2),
        'RR': round(risk_reward, 2)
    }

def render_pdf_standard_header(fig, title_text='Darvas Strategy Report'):
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
    render_pdf_standard_header(fig, "Nicolas Darvas Strategy")
    PRIMARY_COLOR = '#1e293b'
    ACCENT_COLOR = '#2563eb'
    SECONDARY_COLOR = '#475569'
    
    fig.text(0.5, 0.65, "Nicolas Darvas", ha='center', va='center', fontsize=36, weight='bold', color=ACCENT_COLOR)
    fig.text(0.5, 0.58, "Box Strategy Screener Results", ha='center', va='center', fontsize=22, weight='bold', color=SECONDARY_COLOR)
    fig.text(0.5, 0.52, f"Analysis Report: {timestamp}", ha='center', va='center', fontsize=14, color=SECONDARY_COLOR)
    
    fig.text(0.15, 0.45, "Core Principles", fontsize=16, weight='bold', color=PRIMARY_COLOR)
    principles = [
        "● Box Theory (Consolidation Breakouts)",
        "● Trend Following (Price > 50 > 150 > 200 SMA)",
        "● Volume Confirmation (> 150% of Average)",
        "● Proximity to 52-Week High"
    ]
    for i, p in enumerate(principles):
        fig.text(0.15, 0.41 - (i * 0.04), p, fontsize=12, color=SECONDARY_COLOR)
        
    # Market Environment Notes
    fig.text(0.15, 0.18, "Best Environment", fontsize=14, weight='bold', color=PRIMARY_COLOR)
    fig.text(0.15, 0.15, "● Works well in Strong Bull Markets and Industry-Specific 'Booms'", fontsize=11, color=SECONDARY_COLOR)
    
    fig.text(0.15, 0.11, "Avoid In", fontsize=14, weight='bold', color=PRIMARY_COLOR)
    avoids = [
        "● Choppy/Sideways Markets",
        "● Late-Stage Bull Markets",
        "● Bear Markets"
    ]
    for i, a in enumerate(avoids):
        fig.text(0.15, 0.08 - (i * 0.03), a, fontsize=11, color=SECONDARY_COLOR)

    pdf.savefig(fig)
    plt.close(fig)

def render_pdf_documentation_page(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    render_pdf_standard_header(fig, "Darvas Methodology & Rules")
    PRIMARY_COLOR = '#1e293b'
    SECONDARY_COLOR = '#475569'
    LEFT_MARGIN = 0.08
    
    fig.text(LEFT_MARGIN, 0.86, "Strategy Rules", fontsize=20, weight='bold', color=PRIMARY_COLOR)
    
    text = """
1. The Darvas Box Setup
- Establishing the Top: High holds for 3 consecutive days.
- Establishing the Bottom: Low holds for 3 consecutive days without breaking ceiling.
- The Breakout: Buy when price breaks above the top of the box on high volume.

2. Screening Criteria
- Proximity to 52w High: Within 5%-10% of the high.
- Volume: > 150% of the 30-day average volume.
- Trend: Moving Average Alignment (Price > 50 SMA > 150 SMA > 200 SMA).

3. Key Metrics (The "Sweet Spot")
- Box Age: Ideal breakouts occur from boxes 20-40 days old.
- Box Tightness: Ideal boxes are less than 10% wide.
- VCP: Volatility Contraction Pattern detected inside the box.

Risk Management
- Stop Loss: Low of the box or just below the ceiling.
- Target: Measured move (Ceiling + Box Width).
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
        table.auto_set_column_width(col=list(range(len(chunk.columns))))
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
    plot_df = df.tail(252) # Show full 1 year data
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
    
    # Moving Averages
    if 'SMA50' in plot_df.columns:
        ax1.plot(plot_df.index, plot_df['SMA50'], color='green', label='SMA 50')
    if 'SMA150' in plot_df.columns:
        ax1.plot(plot_df.index, plot_df['SMA150'], color='blue', label='SMA 150')
    if 'SMA200' in plot_df.columns:
        ax1.plot(plot_df.index, plot_df['SMA200'], color='red', label='SMA 200')
        
    # Darvas Box Levels
    date_str_list = plot_df.index.strftime('%Y-%m-%d').tolist()
    box_start_date_str = result.get('Box_Start')
    
    if box_start_date_str and box_start_date_str in date_str_list:
        idx = date_str_list.index(box_start_date_str)
        box_start_dt = plot_df.index[idx]
        
        # Draw lines only for the box period
        box_period = plot_df.loc[box_start_dt:].index
        ax1.plot(box_period, [result['Ceiling']] * len(box_period), color='cyan', linewidth=2, label=f"Ceiling ({result['Ceiling']})")
        ax1.plot(box_period, [result['Floor']] * len(box_period), color='magenta', linewidth=2, label=f"Floor ({result['Floor']})")
        
        # Draw vertical lines at start and end to complete the box
        ax1.vlines(box_start_dt, result['Floor'], result['Ceiling'], color='cyan', linestyle='--', linewidth=1.5)
        ax1.vlines(plot_df.index[-1], result['Floor'], result['Ceiling'], color='cyan', linestyle='--', linewidth=1.5)
    else:
        # Fallback to horizontal lines if start date not found or outside plot range
        ax1.axhline(result['Ceiling'], color='cyan', linestyle='-', linewidth=1.5, label=f"Ceiling ({result['Ceiling']})")
        ax1.axhline(result['Floor'], color='magenta', linestyle='-', linewidth=1.5, label=f"Floor ({result['Floor']})")
        
    # Mark Breakout Trigger
    trigger_date_str = result.get('Breakout_Date')
    if trigger_date_str and trigger_date_str in date_str_list:
        idx = date_str_list.index(trigger_date_str)
        trigger_dt = plot_df.index[idx]
        ax1.plot(trigger_dt, result['Ceiling'], marker='^', color='green', markersize=12, label='Breakout Trigger')
    
    # Trading Levels
    ax1.axhline(result['Target'], color='green', linestyle=':', linewidth=1.5, label=f"Target ({result['Target']})")
    ax1.axhline(result['Stop_Loss'], color='red', linestyle=':', linewidth=1.5, label=f"Stop ({result['Stop_Loss']})")
    
    title = f"{ticker} - Darvas {result['Setup']} | Dist 52W High: {result['Dist_52W_High%']}%"
    ax1.set_title(title, fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Volume
    colors = ['green' if c >= o else 'red' for c, o in zip(plot_df.Close, plot_df.Open)]
    ax2.bar(plot_df.index, plot_df.Volume, color=colors, width=0.6, alpha=0.6)
    if 'VolSMA30' in plot_df.columns:
        ax2.plot(plot_df.index, plot_df['VolSMA30'], color='orange', label='Vol SMA 30')
    ax2.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if pdf:
        pdf.savefig(fig)
    
    filename = f"{ticker.replace('.NS', '')}_darvas.png"
    save_path = os.path.join(CHARTS_DIR, filename)
    plt.savefig(save_path)
    plt.close(fig)
    return save_path

def main():
    parser = argparse.ArgumentParser(description="Nicolas Darvas Strategy Screener")
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
        
    console.print(f"Loaded {len(tickers)} tickers.")
    
    results = []
    
    with console.status(f"[bold green]Scanning {len(tickers)} stocks against Darvas Strategy...[/bold green]"):
        for ticker in tickers:
            data = fetch_data(ticker, refresh=args.refresh)
            if data:
                match = check_criteria(data, ticker)
                if match:
                    console.print(f"[green]FOUND {match['Setup']}: {ticker} [/green]")
                    results.append((match, data))
                    
    if results:
        with PdfPages(PDF_PATH) as pdf:
            create_pdf_title_page(pdf, TIMESTAMP)
            render_pdf_documentation_page(pdf)
            
            df_results = pd.DataFrame([r[0] for r in results])
            render_pdf_styled_table(pdf, df_results, "Darvas Screener Results")
            
            for match, data in results:
                generate_chart(data['history'], match['Ticker'], match, pdf=pdf)
                
            df_results.to_csv(os.path.join(OUTPUT_DIR, 'results.csv'), index=False)
            console.print(f"\n[bold]Results saved to {OUTPUT_DIR}/results.csv[/bold]")
            console.print(f"[bold green]PDF Report saved to {PDF_PATH}[/bold green]")
    else:
        console.print("[yellow]No stocks found matching Darvas criteria.[/yellow]")

if __name__ == "__main__":
    main()

# (I will implement them in the next step to keep the file write manageable or write them now if preferred)
