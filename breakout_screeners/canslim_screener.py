"""
CANSLIM Stock Screener

This script implements William O'Neil's CANSLIM methodology to identify high-growth stocks 
with strong technical and fundamental characteristics.

CANSLIM Criteria Implemented:
-----------------------------
C - Current Quarterly Earnings: EPS and Sales growth >= 20%.
A - Annual Earnings Increases: Annual EPS growth >= 25% and ROE >= 17%.
N - New Products, Management, or Highs: Price within 20% of its 52-week high.
S - Supply and Demand: Trading volume must be above its 50-day average (Accumulation).
L - Leader or Laggard: Relative Strength (RS) Rating >= 80 (top 20% of the market).
I - Institutional Sponsorship: Institutional ownership >= 10%.
M - Market Direction: Checks if NIFTY 50 is above its 50-day moving average.

Technical Filters & Trade Setup:
--------------------------------
- Price >= 50-day and 200-day Simple Moving Averages (SMA).
- 50-day SMA > 200-day SMA (Upward Trend).
- Pivot Point: 20-day high (excluding today).
- Buy Zone: Price must be within 5% of the pivot level (avoiding extended stocks).
- Risk Management: 7.5% Stop Loss and 25% Profit Target.

Calculation Logic:
------------------
- RS Rating: Percentile rank (0-99) of 1-year returns across the screened universe.
- Market Status: 'Bullish' if NIFTY 50 > 50-day SMA, else 'Bearish'.
- Volume Ratio: Today's Volume / 50-day Average Volume.

Risk Management (Execution Rules):
---------------------
The strategy relies heavily on cutting losses quickly to maintain positive expectancy.
- Technical Stop Loss: You must exit immediately if the stock severely undercuts the 50-day or 10-week moving average.
- Hard Stop Limit: Never tolerate a pullback greater than a rigid 7% to 8% drop from your initial entry point.
- Risk/Reward Goal: Position sizing and profit-taking should be engineered so that your average winning trade is at least 2.5 times larger than your average losing trade.

Usage:
------
python canslim_screener.py [--limit N] [--sample N] [--refresh]
"""

# Imports
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
# Initialize Console (Dummy wrapper for now, similar to fw_breakout_screener)
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
OUTPUT_DIR = os.path.join(BASE_DIR, 'screener_results', 'CAMSLIM_breakouts', TIMESTAMP)
CHARTS_DIR = os.path.join(OUTPUT_DIR, 'charts')
CACHE_DIR = os.path.join(BASE_DIR, 'data_cache')
JSON_PATH = os.path.join(DATA_DIR, 'nifty_500.json')
PDF_PATH = os.path.join(OUTPUT_DIR, f"CANSLIM_Screener_Results_{TIMESTAMP}.pdf")

# CANSLIM Parameters
MIN_EPS_GROWTH = 0.20  # Current Qtr EPS >= 20%
MIN_SALES_GROWTH = 0.20 # Current Qtr Sales >= 20%
MIN_ANNUAL_EPS_GROWTH = 0.25 # Annual EPS Growth >= 25% (The 'A')
MIN_ROE = 0.17 # Return on Equity >= 17% (The 'A')
MIN_INST_OWN = 0.10 # Institutional Sponsorship >= 10% (The 'I')
MIN_RS_RATING = 80     # Relative Strength >= 80 (The 'L')
MIN_PRICE_TO_HIGH = 0.80 # Price within 20% of 52W High (The 'N')

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
    """
    Calculate strong 1-99 True Relative Strength rank across our universe.
    Returns: dict mapping ticker -> RS Rating (0-99 int)
    """
    returns = {}
    
    with console.status(f"[blue]Calculating Relative Strength rankings for {len(tickers)} stocks...[/blue]"):
        for ticker in tickers:
            cache_path = os.path.join(CACHE_DIR, f"{ticker}_1d.csv")
            if os.path.exists(cache_path):
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                if len(df) > 200: # Ensure enough history
                    # 1 Year return roughly
                    try:
                        ret_1y = (df['Close'].iloc[-1] - df['Close'].iloc[-252]) / df['Close'].iloc[-252]
                        returns[ticker] = ret_1y
                    except IndexError:
                        pass # Not enough data
    
    if not returns:
        return {}
        
    s = pd.Series(returns)
    # Calculate percentile rank (0 to 1), multiply by 99 to get 0-99 rating
    ranks = s.rank(pct=True) * 99
    
    # Return as dict {ticker: int(rank)}
    return ranks.apply(lambda x: int(round(x))).to_dict()

def fetch_data(ticker, refresh=False):
    # Separate cache for History and Info
    hist_cache_path = os.path.join(CACHE_DIR, f"{ticker}_1d.csv") # Daily data for CANSLIM
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
        
        # 2. HISTORY (Technicals)
        if not refresh and os.path.exists(hist_cache_path):
            df = pd.read_csv(hist_cache_path, index_col=0, parse_dates=True)
            data['history'] = df
        else:
            stock = yf.Ticker(ticker)
            # Need at least 1 year for RS and SMA 200, but fetch 2y to align with VCP cache
            df = stock.history(period="2y") 
            if len(df) < 200: return None
            df.to_csv(hist_cache_path)
            data['history'] = df
            
        return data
    except Exception as e:
        # console.print(f"[yellow]Error fetching {ticker}: {e}[/yellow]")
        return None

def check_criteria(data, ticker, rs_rating, market_status="Unknown"):
    info = data.get('info', {})
    df = data.get('history', None)
    
    if df is None or df.empty or len(df) < 200:
        return None
        
    # --- 1. FUNDAMENTALS (C, A, I) ---
    # C: Current Qtr EPS and Sales Growth must be >= 20%
    eps_growth = info.get('earningsQuarterlyGrowth', 0)
    sales_growth = info.get('revenueGrowth', 0)
    
    # A: Annual Earnings Growth >= 25% and ROE >= 17%
    annual_eps_growth = info.get('earningsGrowth', 0)
    roe = info.get('returnOnEquity', 0)
    
    # I: Institutional Sponsorship
    inst_own = info.get('heldPercentInstitutions', 0)
    
    # General Float info (not strictly filtering by size right now, but good to collect)
    shares_float = info.get('floatShares', 0)
    
    # Handle None safely
    if eps_growth is None: eps_growth = 0
    if sales_growth is None: sales_growth = 0
    if annual_eps_growth is None: annual_eps_growth = 0
    if roe is None: roe = 0
    if inst_own is None: inst_own = 0
    
    # STRICT Rules Evaluation:
    # 'C'
    if eps_growth < MIN_EPS_GROWTH: return None
    if sales_growth < MIN_SALES_GROWTH: return None
    
    # 'A'
    if annual_eps_growth < MIN_ANNUAL_EPS_GROWTH: return None
    if roe < MIN_ROE: return None
    
    # 'I'
    if inst_own < MIN_INST_OWN: return None

    # --- 2. TECHNICALS (N, L, M) ---
    # Calculate Indicators
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['52W_High'] = df['High'].rolling(window=252).max()
    df['VolSMA50'] = df['Volume'].rolling(window=50).mean()
    
    curr = df.iloc[-1]
    
    current_close = curr['Close']
    current_vol = curr['Volume']
    sma_50 = curr['SMA50']
    sma_200 = curr['SMA200']
    high_52w = curr['52W_High']
    vol_sma_50 = curr['VolSMA50']
    
    # Stock 1-year return
    # Ensure df has enough data, fetch_data guarantees 1y but let's be safe
    if len(df) < 2: return None
    stock_1y_return = (current_close - df['Close'].iloc[0]) / df['Close'].iloc[0]
    
    # Trend: Price must be above 50-day and 200-day moving averages
    if current_close < sma_50 or current_close < sma_200:
        return None
        
    # Trend: 50-day SMA must be above 200-day SMA
    if sma_50 < sma_200:
        return None
        
    # Base/New Highs: Price should be within 20% of its 52-week high
    if current_close < (high_52w * MIN_PRICE_TO_HIGH):
        return None
        
    # Relative Strength: Stock must be in the top 20% of the market (RS >= 80)
    if rs_rating < MIN_RS_RATING:
        return None
        
    # --- 3. VOLUME (S & I) ---
    # Rule: Is today's volume above the 50-day average?
    if current_vol < vol_sma_50:
        return None
        
    # --- 4. IMPROVED TRADE SETUP / RISK MANAGEMENT ---
    # Find a multi-week consolidation high (Pivot) excluding today
    pivot_level = df['High'].iloc[-21:-1].max() # High of the previous 20 trading days
    
    # Breakout Buy Zone: Only trigger if the current price is within 5% of that pivot
    if current_close > pivot_level * 1.05:
        return None # Already too extended ("out of the buy zone")
        
    breakout_level = pivot_level
    
    # CANSLIM strict stop-loss: 7% to 8% below purchase price. 
    # We use 7.5% here based on current close.
    stop_loss = current_close * 0.925
    
    # Typical CANSLIM profit taking objective is +20% to +25% from the pivot
    target_price = breakout_level * 1.25 
    
    # Risk/Reward Ratio
    risk_amount = current_close - stop_loss
    reward_amount = target_price - current_close
    risk_reward = reward_amount / risk_amount if risk_amount > 0 else 0
        
    return {
        'Ticker': ticker,
        'Date': df.index[-1].strftime('%Y-%m-%d'),
        'Close': round(current_close, 1),
        'Breakout': round(breakout_level, 2),
        'Target': round(target_price, 2),
        'Stop_Loss': round(stop_loss, 2),
        'Risk_Reward': round(risk_reward, 2),
        'Qtr_EPS%': round(eps_growth * 100, 2),
        'Qtr_Sales%': round(sales_growth * 100, 2),
        'Ann_EPS%': round(annual_eps_growth * 100, 2),
        'ROE%': round(roe * 100, 2),
        'Inst_Own%': round(inst_own * 100, 2),
        'RS_Rating': rs_rating,
        'Vol_Ratio': round(current_vol / vol_sma_50, 2),
        'SMA50': sma_50,
        'SMA200': sma_200,
        '52W_High': high_52w,
        'Market_Status': market_status
    }

def create_pdf_title_page(pdf, timestamp, market_status):
    """Creates a professional title page for the PDF report."""
    fig = plt.figure(figsize=(11, 8.5))
    plt.axis('off')
    
    plt.text(0.5, 0.7, "CANSLIM Stock Screener Report", 
             ha='center', va='center', fontsize=32, weight='bold', color='#1e293b')
    
    plt.text(0.5, 0.55, f"Analysis Date: {timestamp}", 
             ha='center', va='center', fontsize=18, color='#475569')
    
    # Market Status highlighting
    status_color = '#16a34a' if "Bullish" in market_status else '#dc2626'
    plt.text(0.5, 0.45, f"Market Condition: {market_status}", 
             ha='center', va='center', fontsize=20, weight='bold', color=status_color)
    
    # Strengths and Limitations
    # Left Column: Strengths
    plt.text(0.1, 0.35, "Strengths of CANSLIM", fontsize=14, weight='bold', color='#2563eb')
    strengths = [
        "✔ Works very well in bull markets",
        "✔ Combines fundamentals + technicals",
        "✔ Strong risk management focus"
    ]
    for i, s in enumerate(strengths):
        plt.text(0.1, 0.31 - (i * 0.035), s, fontsize=11, color='#4b5563')
        
    # Right Column: Limitations
    plt.text(0.55, 0.35, "Limitations", fontsize=14, weight='bold', color='#dc2626')
    limitations = [
        "⚠ Underperforms in sideways or bear markets",
        "⚠ Requires discipline and fast decision-making",
        "⚠ Frequent churn if market conditions are choppy"
    ]
    for i, l in enumerate(limitations):
        plt.text(0.55, 0.31 - (i * 0.035), l, fontsize=11, color='#4b5563')
    pdf.savefig(fig)
    plt.close(fig)

def render_pdf_documentation_page(pdf):
    """Adds a documentation page explaining the methodology and rules."""
    fig = plt.figure(figsize=(11, 8.5))
    plt.axis('off')
    
    # Title
    plt.text(0.5, 0.97, "CANSLIM Methodology & Execution Rules", 
             ha='center', va='top', fontsize=20, weight='bold', color='#1e293b')
    
    sections = [
        ("CANSLIM Criteria", [
            "C - Current Quarterly Earnings: EPS and Sales growth >= 20%.",
            "A - Annual Earnings Increases: Annual EPS growth >= 25% and ROE >= 17%.",
            "N - New Products, Management, or Highs: Price within 20% of 52-week high.",
            "S - Supply and Demand: Trading volume above 50-day average (Accumulation).",
            "L - Leader or Laggard: RS Rating >= 80 (top 20% of the market).",
            "I - Institutional Sponsorship: Institutional ownership >= 10%.",
            "M - Market Direction: NIFTY 50 above 50-day SMA."
        ]),
        ("Technical Filters & Trade Setup", [
            "Price >= 50-day and 200-day Simple Moving Averages (SMA).",
            "50-day SMA > 200-day SMA (Upward Trend).",
            "Pivot Point: 20-day high (excluding today).",
            "Buy Zone: Price must be within 5% of the pivot level.",
            "Trade Objective: 7.5% Stop Loss and 25% Profit Target."
        ]),
        ("Calculation Logic", [
            "RS Rating: Percentile rank (0-99) of 1-year returns in the universe.",
            "Market Status: Bullish if Nifty > 50SMA, else Bearish.",
            "Volume Ratio: Today's Volume / 50-day average volume."
        ]),
        ("Risk Management (Execution Rules)", [
            "The strategy relies heavily on cutting losses quickly to maintain positive expectancy.",
            "Technical SL: Exit immediately if stock undercuts 50-day or 10-week SMA.",
            "Hard Stop: Never tolerate a pullback greater than a rigid 7% to 8% drop from your initial entry point.",
            "Risk/Reward Goal: Position sizing and profit-taking should be engineered so that your average winning", 
            "trade is at least 2.5 times larger than your average losing trade."
        ])
    ]
    
    y = 0.88 # Moved down from 0.92 to avoid overlap with header
    for title, points in sections:
        # Heading
        plt.text(0.05, y, title, fontsize=14, weight='bold', color='#1e293b') # Margin 0.1 -> 0.05
        y -= 0.03
        
        # Bullet points
        for point in points:
            plt.text(0.07, y, f"• {point}", fontsize=11, color='#475569') # Margin 0.12 -> 0.07
            y -= 0.035
        
        y -= 0.02 # Space between sections
        
    pdf.savefig(fig)
    plt.close(fig)

def render_pdf_styled_table(pdf, df, title):
    """Renders a dataframe as a styled table in the PDF."""
    if df.empty:
        return

    rows_per_page = 20
    num_pages = (len(df) // rows_per_page) + 1
    
    for i in range(num_pages):
        start_idx = i * rows_per_page
        end_idx = min((i + 1) * rows_per_page, len(df))
        chunk = df.iloc[start_idx:end_idx]
        
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('tight')
        ax.axis('off')
        
        ax.set_title(f"{title} (Page {i+1}/{num_pages})", 
                     fontsize=16, weight='bold', pad=20, color='#1e293b')
        
        # Increase Ticker column width (0) and distribute others to fill the page
        col_widths = [0.14] + [0.065] * (len(chunk.columns) - 1) # Sum = 0.985
        
        table = ax.table(cellText=chunk.values, colLabels=chunk.columns, 
                        loc='center', cellLoc='center', colWidths=col_widths)
        
        table.auto_set_font_size(False)
        table.set_fontsize(6) 
        table.scale(1.0, 1.2) # Minimal row height
        
        # Style header and rows
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#1e293b')
            else:
                cell.set_facecolor('#f8fafc' if row % 2 == 0 else 'white')
                # Color code Ticker
                if col == 0: 
                     cell.set_text_props(weight='bold', color='#2563eb')
        
        pdf.savefig(fig)
        plt.close(fig)

def generate_chart(df, ticker, result, pdf=None):
    # Plot last 1 year or 6 months
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
    ax1.plot(plot_df.index, plot_df['SMA200'], color='black', label='SMA 200')
    
    # Trade Setup Levels
    ax1.axhline(result['Breakout'], color='magenta', linestyle='--', linewidth=1.5, alpha=0.8, label=f"Breakout ({result['Breakout']})")
    ax1.axhline(result['Target'], color='green', linestyle=':', linewidth=1.5, alpha=0.8, label=f"Target 25% ({result['Target']})")
    ax1.axhline(result['Stop_Loss'], color='red', linestyle=':', linewidth=1.5, alpha=0.8, label=f"Stop 7.5% ({result['Stop_Loss']})")
    
    # Title
    title = (f"{ticker} - CANSLIM Breakout Setup | Market: {result.get('Market_Status', 'Unknown')}\n"
             f"Qtr EPS: {result['Qtr_EPS%']}%, Ann EPS: {result['Ann_EPS%']}%, ROE: {result['ROE%']}%\n"
             f"RS Rating: {result['RS_Rating']}, Inst Own: {result['Inst_Own%']}% | R:R: {result['Risk_Reward']}")
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
    
    filename = f"{ticker.replace('.NS', '')}_canslim.png"
    save_path = os.path.join(CHARTS_DIR, filename)
    plt.savefig(save_path)
    plt.close(fig)
    return save_path

def main():
    parser = argparse.ArgumentParser(description="CANSLIM Stock Screener")
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
        
    # Calculate True RS Ratings prior to scanning
    rs_ratings = calculate_rs_ratings(tickers)
    
    # Get Market Condition
    market_status = "Unknown"
    try:
        with console.status("[blue]Fetching NIFTY 50 Market Condition...[/blue]"):
            nifty = yf.Ticker('^NSEI')
            nifty_df = nifty.history(period="6mo")
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
    
    with console.status(f"[bold green]Scanning {len(tickers)} stocks against CANSLIM...[/bold green]") as status:
        for ticker in tickers:
            data = fetch_data(ticker, refresh=args.refresh)
            if data:
                rs = rs_ratings.get(ticker, 0) # default to 0 if not calculated
                match = check_criteria(data, ticker, rs, market_status)
                if match:
                    console.print(f"[green]FOUND: {ticker} (Qtr EPS: {match['Qtr_EPS%']}%, Ann EPS: {match['Ann_EPS%']}%, ROE: {match['ROE%']}%, RS: {match['RS_Rating']}) [/green]")
                    # Store data to regenerate chart in PDF later
                    results.append((match, data))
    
    with PdfPages(PDF_PATH) as pdf:
        # 1. Title Page
        create_pdf_title_page(pdf, TIMESTAMP, market_status)
        
        # 2. Documentation Page (New)
        render_pdf_documentation_page(pdf)
        
        if results:
            # 2. Add Summary Table to PDF (Moved to top)
            df_results = pd.DataFrame([r[0] for r in results])
            cols = ['Ticker', 'Date', 'Close', 'Breakout', 'Target', 'Stop_Loss', 'Risk_Reward', 'Qtr_EPS%', 'Qtr_Sales%', 'Ann_EPS%', 'ROE%', 'Inst_Own%', 'RS_Rating', 'Vol_Ratio']
            if all(c in df_results.columns for c in cols):
                df_results = df_results[cols]
                
            print("\nCANSLIM Screener Results:")
            print(df_results.to_string(index=False))
            
            render_pdf_styled_table(pdf, df_results, "CANSLIM Screener Summary Results")
            
            # 3. Add individual Stock Pages
            for match, data in results:
                generate_chart(data['history'], match['Ticker'], match, pdf=pdf)
            
            df_results.to_csv(os.path.join(OUTPUT_DIR, 'results.csv'), index=False)
            console.print(f"\n[bold]Results saved to {OUTPUT_DIR}/results.csv[/bold]")
            console.print(f"[bold green]PDF Report saved to {PDF_PATH}[/bold green]")
        else:
            console.print("[yellow]No stocks found matching CANSLIM criteria.[/yellow]")
            # Add "No Results" page to PDF
            fig = plt.figure(figsize=(11, 8.5))
            plt.axis('off')
            plt.text(0.5, 0.5, "No stocks found matching CANSLIM criteria.", 
                     ha='center', va='center', fontsize=20, color='#64748b')
            pdf.savefig(fig)
            plt.close(fig)

if __name__ == "__main__":
    main()
