# -------------------------------------------------------------------------------
# Project: Stock Analysis (https://github.com/anshulsolanki/stock_analysis)
# Author:  Anshul Solanki
# License: MIT License
# -------------------------------------------------------------------------------

"""
Kristjan Qullamaggie Strategy Screener

This script implements Kristjan Qullamaggie's strategy including:
1. The Breakout (High Tight Flag)
2. The Episodic Pivot (EP)

Usage:
------
python qullamaggie_screener.py [--limit N] [--sample N] [--refresh]
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
import scipy.signal

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
OUTPUT_DIR = os.path.join(BASE_DIR, 'screener_results', 'qullamaggie_breakouts', TIMESTAMP)
CHARTS_DIR = os.path.join(OUTPUT_DIR, 'charts')
CACHE_DIR = os.path.join(DATA_DIR, 'data_cache')
JSON_PATH = os.path.join(DATA_DIR, 'nifty_500.json')
PDF_PATH = os.path.join(OUTPUT_DIR, f"qullamaggie_Screener_Results_{TIMESTAMP}.pdf")

def setup_directories():
    """
    Create necessary directories for output, charts, and data cache if they do not exist.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

def load_tickers(limit=None):
    """
    Load tickers from the Nifty 500 JSON file.

    Parameters:
    -----------
    limit : int, optional
        Maximum number of tickers to load.

    Returns:
    --------
    list
        List of ticker symbols.
    """
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
    """
    Fetch historical data and info for a ticker, using local cache if available.

    Parameters:
    -----------
    ticker : str
        Ticker symbol.
    refresh : bool, optional
        Force refresh of cached data.

    Returns:
    --------
    dict or None
        Dictionary containing 'info' and 'history' (DataFrame), or None if failed.
    """
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

def is_earnings_catalyst(ticker_obj):
    """
    Checks if there was an earnings event in the last 48 hours for the given ticker.

    Parameters:
    -----------
    ticker_obj : yfinance.Ticker
        The yfinance Ticker object.

    Returns:
    --------
    bool
        True if an earnings event occurred within the last 2 days, False otherwise.
    """
    try:
        # Get earnings dates
        earnings = ticker_obj.earnings_dates
        if earnings is None or earnings.empty:
            return False
            
        # Check if any earnings date is within the last 2 days
        today = datetime.now().date()
        for date in earnings.index:
            if abs((date.date() - today).days) <= 2:
                return True
        return False
    except:
        return False 

def check_news_catalyst(ticker_obj):
    """
    Scans recent news headlines for the given ticker for specific catalyst keywords.

    Parameters:
    -----------
    ticker_obj : yfinance.Ticker
        The yfinance Ticker object.

    Returns:
    --------
    str or None
        The matching headline if a keyword is found, None otherwise.
    """
    news = ticker_obj.news
    catalyst_keywords = ['earnings', 'beats', 'contract', 'fda', 'approval', 'guidance', 'partnership']
    
    for article in news:
        headline = article['title'].lower()
        if any(key in headline for key in catalyst_keywords):
            return headline # Return the headline to include in your report
    return None

def check_criteria(data, ticker):
    """
    Evaluate a stock against Kristjan Qullamaggie's strategy criteria.
    Checks for both Episodic Pivot (EP) and Breakout (High Tight Flag) setups.

    Parameters:
    -----------
    data : dict
        Dictionary containing 'info' and 'history' for the ticker.
    ticker : str
        Ticker symbol.

    Returns:
    --------
    dict or None
        Dictionary with match details if criteria are met, None otherwise.
    """
    info = data.get('info', {})
    df = data.get('history', None)
    
    if df is None or df.empty or len(df) < 200:
        return None
        
    curr = df.iloc[-1]
    prev_close = df['Close'].iloc[-2]
    curr_open = curr['Open']
    curr_vol = curr['Volume']
    
    df['VolSMA50'] = df['Volume'].rolling(window=50).mean()
    vol_sma_50 = df['VolSMA50'].iloc[-1]
    
    # --- 1. Episodic Pivot (EP) Setup ---
    gap_pct = (curr_open - prev_close) / prev_close
    vol_ratio = curr_vol / vol_sma_50 if vol_sma_50 > 0 else 0
    
    if gap_pct >= 0.10 and vol_ratio >= 3.0 and curr['Close'] > curr_open:
        # It's an EP
        ticker_obj = yf.Ticker(ticker)
        headline = check_news_catalyst(ticker_obj)
        is_earnings = is_earnings_catalyst(ticker_obj)
        
        catalyst_desc = "Unknown"
        if is_earnings:
            catalyst_desc = "Earnings"
        elif headline:
            catalyst_desc = f"News: {headline[:50]}..."
            
        stop_loss = curr['Low'] # Low of breakout day
        target_price = curr['Close'] * 1.25 # Default target
        
        risk = curr['Close'] - stop_loss
        reward = target_price - curr['Close']
        risk_reward = reward / risk if risk > 0 else 0
        
        return {
            'Ticker': ticker,
            'Date': df.index[-1].strftime('%Y-%m-%d'),
            'Close': round(curr['Close'], 1),
            'Setup': 'EP',
            'Catalyst': catalyst_desc,
            'Move%': round(gap_pct * 100, 2),
            'Vol_Ratio': round(vol_ratio, 2),
            'Target': round(target_price, 2),
            'Stop_Loss': round(stop_loss, 2),
            'Risk_Reward': round(risk_reward, 2)
        }
        
    # --- 2. Breakout (High Tight Flag) Setup ---
    # Check return over last 1, 2, 3 months (approx 21, 42, 63 trading days)
    df['Pct_Change_21'] = df['Close'].pct_change(periods=21, fill_method=None)
    df['Pct_Change_42'] = df['Close'].pct_change(periods=42, fill_method=None)
    df['Pct_Change_63'] = df['Close'].pct_change(periods=63, fill_method=None)
    
    curr = df.iloc[-1] # Update curr to include new columns
    
    pct_21 = curr['Pct_Change_21'] if not np.isnan(curr['Pct_Change_21']) else 0
    pct_42 = curr['Pct_Change_42'] if not np.isnan(curr['Pct_Change_42']) else 0
    pct_63 = curr['Pct_Change_63'] if not np.isnan(curr['Pct_Change_63']) else 0
    
    max_move = max(pct_21, pct_42, pct_63)
    
    if max_move < 0.30: # Must be at least 30%
        return None
        
    # --- Distance from 52-Week High Filter ---
    fifty_two_week_high = df['High'].rolling(window=252, min_periods=1).max().iloc[-1]
    distance_from_high = (fifty_two_week_high - curr['Close']) / fifty_two_week_high
    
    if distance_from_high > 0.10: # Must be within 10% of 52-week high
        return None
        
    # --- 3. The ADR (Average Daily Range) Filter ---
    df['ADR'] = ((df['High'] - df['Low']) / df['Close']).rolling(20).mean() * 100
    curr = df.iloc[-1] # Update curr to include ADR
    
    if curr['ADR'] < 3.0: # Must be at least 3.0% (Updated for Indian context)
        return None
        
    # Consolidation check
    consolidation_period = 20
    recent_closes = df['Close'].tail(consolidation_period)
    high_consol = recent_closes.max()
    low_consol = recent_closes.min()
    range_pct = (high_consol - low_consol) / high_consol
    
    if range_pct > 0.15:
        return None
        
    # Volatility Contraction (Shrinking Range)
    range_recent = df['Close'].iloc[-10:].max() - df['Close'].iloc[-10:].min()
    range_prior = df['Close'].iloc[-20:-10].max() - df['Close'].iloc[-20:-10].min()
    if range_recent >= range_prior:
        return None
        
    # Calculate MAs
    df['SMA10'] = df['Close'].rolling(window=10).mean()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    
    # Extension Limit
    curr_sma10 = df['SMA10'].iloc[-1]
    if (curr['Close'] - curr_sma10) / curr_sma10 > 0.02:
        return None
        
    # Surfing MAs
    recent_sma10 = df['SMA10'].tail(10)
    recent_sma20 = df['SMA20'].tail(10)
    recent_close_tail = df['Close'].tail(10)
    
    above_10 = (recent_close_tail > recent_sma10).sum()
    above_20 = (recent_close_tail > recent_sma20).sum()
    
    if above_10 < 8 and above_20 < 8:
        return None
        
    # Breakout trigger
    prev_high = df['Close'].iloc[-consolidation_period:-1].max()
    if curr['Close'] <= prev_high:
        return None
        
    # Volume confirmation for breakout
    if curr_vol < vol_sma_50 * 1.2:
        return None
        
    stop_loss = min(curr['Low'], df['Low'].iloc[-5:].min())
    target_price = curr['Close'] * 1.25
    
    risk = curr['Close'] - stop_loss
    reward = target_price - curr['Close']
    risk_reward = reward / risk if risk > 0 else 0
        
    return {
        'Ticker': ticker,
        'Date': df.index[-1].strftime('%Y-%m-%d'),
        'Close': round(curr['Close'], 1),
        'Setup': 'Breakout',
        'Move%': round(max_move * 100, 2),
        'Vol_Ratio': round(curr_vol / vol_sma_50, 2),
        'Dist_52W_High%': round(distance_from_high * 100, 2),
        'Target': round(target_price, 2),
        'Stop_Loss': round(stop_loss, 2),
        'Risk_Reward': round(risk_reward, 2)
    }

def render_pdf_standard_header(fig, title_text='Qullamaggie Strategy Report'):
    """
    Render a standard header with title and copyright on a matplotlib figure.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure object to draw on.
    title_text : str, optional
        The title text to display.
    """
    PRIMARY_COLOR = '#1e293b'
    BORDER_COLOR = '#94a3b8'
    fig.text(0.5, 0.96, title_text, ha='center', va='center', fontsize=16, weight='bold', color=PRIMARY_COLOR)
    copyright_text = f"© {datetime.now().year} Stock Research. All Rights Reserved."
    fig.text(0.5, 0.935, copyright_text, ha='center', va='center', fontsize=9, style='italic', color=BORDER_COLOR)
    from matplotlib.lines import Line2D
    line = Line2D([0.08, 0.92], [0.91, 0.91], transform=fig.transFigure, color=BORDER_COLOR, linewidth=1.0, alpha=0.5)
    fig.add_artist(line)

def create_pdf_title_page(pdf, timestamp):
    """
    Create a title page for the PDF report.

    Parameters:
    -----------
    pdf : PdfPages
        The PDF file object.
    timestamp : str
        The timestamp of the analysis.
    """
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    render_pdf_standard_header(fig, "Kristjan Qullamaggie Strategy")
    PRIMARY_COLOR = '#1e293b'
    ACCENT_COLOR = '#2563eb'
    SECONDARY_COLOR = '#475569'
    
    fig.text(0.5, 0.65, "Kristjan Qullamaggie", ha='center', va='center', fontsize=36, weight='bold', color=ACCENT_COLOR)
    fig.text(0.5, 0.58, "Strategy Screener Results", ha='center', va='center', fontsize=22, weight='bold', color=SECONDARY_COLOR)
    fig.text(0.5, 0.52, f"Analysis Report: {timestamp}", ha='center', va='center', fontsize=14, color=SECONDARY_COLOR)
    
    fig.text(0.15, 0.35, "Core Setups", fontsize=16, weight='bold', color=PRIMARY_COLOR)
    strengths = [
        "● The Breakout (High Tight Flag)",
        "● The Episodic Pivot (EP)",
        "● Momentum and Volatility Contraction"
    ]
    for i, s in enumerate(strengths):
        fig.text(0.15, 0.31 - (i * 0.04), s, fontsize=12, color=SECONDARY_COLOR)

    pdf.savefig(fig)
    plt.close(fig)

def render_pdf_documentation_page(pdf):
    """
    Create a documentation page explaining the strategy rules in the PDF report.

    Parameters:
    -----------
    pdf : PdfPages
        The PDF file object.
    """
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    render_pdf_standard_header(fig, "Qullamaggie Methodology & Rules")
    PRIMARY_COLOR = '#1e293b'
    SECONDARY_COLOR = '#475569'
    LEFT_MARGIN = 0.08
    
    fig.text(LEFT_MARGIN, 0.86, "Strategy Rules", fontsize=20, weight='bold', color=PRIMARY_COLOR)
    
    text = """
1. The Breakout (High Tight Flag)
- Momentum: Gained at least 30% in any rolling 1, 2, or 3 month period within the last 3 months.
- Distance from High: Within 10% of 52-week high.
- ADR Filter: Average Daily Range (last 20 days) >= 3.0%.
- Consolidation: Price stays within a 15% range for the last 20 days.
- Tightness: Price range of the last 10 days must be strictly less than the range of the preceding 10 days.
- Moving Averages: Price above 10 SMA or 20 SMA at least 80% of the time in last 10 days.
- Extension: Price within 2% of the 10-day SMA on the breakout day.
- Trigger: Breakout above the high of the consolidation range.

2. The Episodic Pivot (EP)
- The Gap: Open price >= 10% above the previous day's close.
- The Volume: Volume >= 3x the 50-day average volume.
- The Close: Close price must be greater than the open price.
- Catalyst: Automated detection of earnings or relevant news headlines.

Risk Management
- Stop Loss: Low of breakout day or recent swing low.
- Target: Default to 25% profit target.
"""
    fig.text(LEFT_MARGIN, 0.80, text, ha='left', va='top', fontsize=11, color=SECONDARY_COLOR, linespacing=1.6)
    pdf.savefig(fig)
    plt.close(fig)

def render_pdf_styled_table(pdf, df, title):
    """
    Render a styled table in the PDF report for the given DataFrame.

    Parameters:
    -----------
    pdf : PdfPages
        The PDF file object.
    df : pandas.DataFrame
        The DataFrame containing the results.
    title : str
        The title of the table/page.
    """
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
    """
    Generate a candlestick chart for the stock with entry, stop loss, and target levels.
    Saves the chart to a file and optionally appends it to a PDF report.

    Parameters:
    -----------
    df : pandas.DataFrame
        Historical price data.
    ticker : str
        Ticker symbol.
    result : dict
        Match details containing levels.
    pdf : PdfPages, optional
        The PDF file object to append the chart to.

    Returns:
    --------
    str
        Path to the saved chart image file.
    """
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
    if 'SMA10' in plot_df.columns:
        ax1.plot(plot_df.index, plot_df['SMA10'], color='blue', label='SMA 10')
    if 'SMA20' in plot_df.columns:
        ax1.plot(plot_df.index, plot_df['SMA20'], color='orange', label='SMA 20')
        
    # Levels
    ax1.axhline(result['Target'], color='green', linestyle=':', linewidth=1.5, label=f"Target ({result['Target']})")
    ax1.axhline(result['Stop_Loss'], color='red', linestyle=':', linewidth=1.5, label=f"Stop ({result['Stop_Loss']})")
    ax1.axhline(result['Close'], color='magenta', linestyle='--', linewidth=1.5, label=f"Entry ({result['Close']})")
    
    title = f"{ticker} - Qullamaggie {result['Setup']} | Move: {result['Move%']}% | Vol Ratio: {result['Vol_Ratio']}"
    ax1.set_title(title, fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Volume
    colors = ['green' if c >= o else 'red' for c, o in zip(plot_df.Close, plot_df.Open)]
    ax2.bar(plot_df.index, plot_df.Volume, color=colors, width=0.6, alpha=0.6)
    if 'VolSMA50' in plot_df.columns:
        ax2.plot(plot_df.index, plot_df['VolSMA50'], color='orange', label='Vol SMA 50')
    ax2.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if pdf:
        pdf.savefig(fig)
    
    filename = f"{ticker.replace('.NS', '')}_qullamaggie.png"
    save_path = os.path.join(CHARTS_DIR, filename)
    plt.savefig(save_path)
    plt.close(fig)
    return save_path

def main():
    """
    Main function to execute the Qullamaggie strategy screener.
    Parses arguments, loads tickers, scans stocks, and generates reports.
    """
    parser = argparse.ArgumentParser(description="Kristjan Qullamaggie Strategy Screener")
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
    
    with console.status(f"[bold green]Scanning {len(tickers)} stocks against Qullamaggie Strategy...[/bold green]"):
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
            render_pdf_styled_table(pdf, df_results, "Qullamaggie Screener Results")
            
            for match, data in results:
                generate_chart(data['history'], match['Ticker'], match, pdf=pdf)
                
            df_results.to_csv(os.path.join(OUTPUT_DIR, 'results.csv'), index=False)
            console.print(f"\n[bold]Results saved to {OUTPUT_DIR}/results.csv[/bold]")
            console.print(f"[bold green]PDF Report saved to {PDF_PATH}[/bold green]")
    else:
        console.print("[yellow]No stocks found matching Qullamaggie criteria.[/yellow]")

if __name__ == "__main__":
    main()
