# -------------------------------------------------------------------------------
# Project: Stock Analysis (https://github.com/anshulsolanki/stock_analysis)
# Author:  Anshul Solanki
# License: MIT License
# -------------------------------------------------------------------------------

"""
Famous Trader Bear Market Screeners

This script implements strategies based on famous investors for identifying strong stocks 
or deep value opportunities during a bear market.

Strategies implemented:
1. Paul Tudor Jones "200-Day Defense": Technical reversal (Stock crosses above 200 SMA on high volume while Index < 200 SMA).
2. David Dreman "Contrarian": Fundamental value (Low PE, High Yield, Low Debt, Safe Payout).
3. Michael Burry "Deep Value": Asset-based value (Low EV/EBITDA, Low PB, Near 52W Low).

The Paul Tudor Jones "200-Day Defense"
Paul Tudor Jones famously uses the 200-day Moving Average as his "line in the sand." He stays out of the market entirely when it's below this line, but uses it to identify the exact turn of a bear market.
The Logic: "The very best money is made at the market turns." He doesn't buy a dip; he buys the reversal.
Screener Recipe (Technical Reversal):
Condition A: The index (Nifty 50) is still below its 200-day SMA (confirms a bear market).
Condition B (The Signal): The individual stock has just crossed above its 200-day SMA on high volume.
Volume: Current day's volume is > 2x the 10-day average.
Why it works: These are the "First Movers." If a stock can break its 200-day average while the Nifty is still crashing, it has massive institutional support.

The David Dreman "Contrarian" Screener
David Dreman is the king of contrarians. He famously argued that the "unpopular" stocks (those the market hates during a war or crisis) almost always outperform the "glamour" stocks over time.
The Logic: During a bear market, high-PE growth stocks crash the hardest. Dreman looks for the "unloved" giants that have stable businesses but "trashy" valuations.
Screener Recipe (Fundamental):
P/E Ratio: Bottom 40% of the market (usually < 15 in India).
Market Cap: Top 500 companies (to ensure they have the "staying power" to survive a war/recession).
Dividend Yield: > 1.5 * Market Average (He loves being paid to wait).
Payout Ratio: < 50% (Ensures the dividend is safe).
Debt to Equity: < 0.5.

The Michael Burry "Deep Value" Screener
Michael Burry is famous for his "Big Short" against the housing bubble, but his core philosophy is finding "Rare Birds"—stocks trading for less than their liquidation value.
The Logic: In a panic, the market often sells off stocks to a point where the company is worth more dead than alive. Burry looks for companies where you are essentially getting the business operations for free because the stock price is covered by cash or hard assets.
Screener Recipe (Fundamental + Technical):
Enterprise Value / EBITDA: < 5 (Extremely cheap).
Price to Book Value: < 0.8.
Proximity to Low: Price is within 10% to 15% of its 52-week Low.
Institutional Selling: FII/DII holding has already dropped (indicating "weak hands" have exited).
Strategy Type: Deep Value
Best Market Phase: Panic / "Absolute Bottom"

Usage:
------
python3 famous_trader_screeners.py [--limit N] [--ticker TICKER]
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Initialize Console (Simple wrapper for clean output)
class Console:
    def print(self, text):
        clean_text = str(text).replace("[bold green]", "").replace("[/bold green]", "")
        clean_text = clean_text.replace("[green]", "").replace("[/green]", "")
        clean_text = clean_text.replace("[red]", "").replace("[/red]", "")
        clean_text = clean_text.replace("[yellow]", "").replace("[/yellow]", "")
        clean_text = clean_text.replace("[cyan]", "").replace("[/cyan]", "")
        clean_text = clean_text.replace("[magenta]", "").replace("[/magenta]", "")
        clean_text = clean_text.replace("[blue]", "").replace("[/blue]", "")
        clean_text = clean_text.replace("[bold]", "").replace("[/bold]", "")
        print(clean_text)

console = Console()

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'data_cache')
JSON_PATH = os.path.join(DATA_DIR, 'nifty_500.json')
NSE_BASELINE_PATH = os.path.join(CACHE_DIR, "NSEI_baseline.csv")
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR_BASE = os.path.join(os.path.dirname(BASE_DIR), 'screener_results', 'bear_market')
OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, TIMESTAMP)

# ==========================================
# Data Loading Helpers
# ==========================================

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

def load_data(ticker, refresh=False):
    """Loads all available data for a ticker from cache."""
    data = {}
    paths = {
        'info': f"{ticker}_info.json",
        'daily': f"{ticker}_1d.csv",
        'weekly': f"{ticker}_1wk.csv",
        'financials': f"{ticker}_financials.csv",
        'quarterly_financials': f"{ticker}_quarterly_financials.csv",
        'balance_sheet': f"{ticker}_balance_sheet.csv",
        'cashflow': f"{ticker}_cashflow.csv"
    }
    
    info_path = os.path.join(CACHE_DIR, paths['info'])
    
    # Refresh logic omitted for now as we assume data is pre-downloaded
    # or we can use the existing download_data.py if needed.
    
    # Load Info
    try:
        with open(info_path, 'r') as f:
            data['info'] = json.load(f)
    except FileNotFoundError:
        data['info'] = {}

    # Load CSVS
    for key in ['daily', 'weekly', 'financials', 'quarterly_financials', 'balance_sheet', 'cashflow']:
        path = os.path.join(CACHE_DIR, paths[key])
        try:
            if key in ['daily', 'weekly']:
                 df = pd.read_csv(path, index_col=0, parse_dates=True)
            else:
                 df = pd.read_csv(path, index_col=0)
            data[key] = df
        except FileNotFoundError:
            data[key] = pd.DataFrame()

    # Load Baseline
    try:
        data['baseline'] = pd.read_csv(NSE_BASELINE_PATH, index_col=0, parse_dates=True)
    except FileNotFoundError:
        data['baseline'] = pd.DataFrame()

    return data

def get_latest_financial_value(df, metric_name):
    """Safely extracts the most recent value for a metric from a financials DataFrame."""
    if df is None or df.empty or metric_name not in df.index:
        return None
    try:
        row = df.loc[metric_name]
        return row.iloc[0] if hasattr(row, 'iloc') else row
    except (IndexError, KeyError):
        return None

def load_market_metrics(tickers):
    """Calculates average dividend yield and bottom 40th percentile P/E from all downloaded info.json files."""
    div_yields = []
    pes = []
    
    for ticker in tickers:
        df_info_path = os.path.join(CACHE_DIR, f"{ticker}_info.json")
        try:
            with open(df_info_path, 'r') as f:
                info = json.load(f)
                
                # Dividend Yield
                yld = None
                try:
                     yld_val = info.get('trailingAnnualDividendYield')
                     if yld_val is not None:
                          yld = float(yld_val)
                     else:
                          yld_val = info.get('dividendYield')
                          if yld_val is not None:
                               yld_val = float(yld_val)
                               if yld_val > 1:
                                    yld = yld_val / 100.0
                               elif yld_val > 0.1: # Ambiguous, assume percent
                                    yld = yld_val / 100.0
                               else:
                                    yld = yld_val # Assume ratio
                except (TypeError, ValueError):
                     yld = None
                
                if yld is not None and yld > 0 and yld < 0.5: # Sanity check for ratio
                     div_yields.append(yld)
                
                # PE
                try:
                     pe_val = info.get('trailingPE')
                     if pe_val is not None:
                          pe = float(pe_val)
                          if pe > 0 and pe < 200: # Sanity check
                               pes.append(pe)
                except (TypeError, ValueError):
                     pass
                     
        except (FileNotFoundError, json.JSONDecodeError):
            continue
            
    avg_div_yield = np.mean(div_yields) if div_yields else 0.015 # Fallback 1.5%
    pe_threshold = np.percentile(pes, 40) if pes else 15 # Fallback 15
    
    console.print(f"[cyan]Market Metrics calculated from {len(div_yields)} valid yields and {len(pes)} valid PEs.[/cyan]")
    console.print(f"[cyan]Average Dividend Yield: {avg_div_yield*100:.2f}%[/cyan]")
    console.print(f"[cyan]P/E Bottom 40% Threshold: {pe_threshold:.2f}[/cyan]")
    
    return {
        'avg_dividend_yield': avg_div_yield,
        'pe_threshold_bottom_40': pe_threshold
    }

GLOBAL_MARKET_METRICS = {}

def get_market_metrics():
    """Returns the globally calculated market metrics."""
    return GLOBAL_MARKET_METRICS

# ==========================================
# Strategy Implementations
# ==========================================

def strategy_paul_tudor_jones(data):
    """
    Paul Tudor Jones "200-Day Defense"
    - Index (Nifty 50) < 200 SMA (Confirms bear market)
    - Stock crossed above 200 SMA on high volume
    - Volume > 2x 10-day average
    """
    df = data.get('daily')
    baseline = data.get('baseline')

    if df is None or df.empty or len(df) < 200:
        return False, "Insufficient Stock Data (<200 days)"

    if baseline is None or baseline.empty or len(baseline) < 200:
        return False, "Insufficient Baseline Data (<200 days)"

    # Handle duplicates in index for safety
    df = df[~df.index.duplicated(keep='last')].copy()
    baseline = baseline[~baseline.index.duplicated(keep='last')].copy()

    # 1. Index < 200 SMA
    baseline['SMA200'] = baseline['Close'].rolling(window=200).mean()
    curr_base = baseline['Close'].iloc[-1]
    curr_base_sma = baseline['SMA200'].iloc[-1]
    
    if pd.isna(curr_base_sma) or curr_base >= curr_base_sma:
         return False, f"Index not in bear market (Index: {curr_base:.1f} >= SMA200: {curr_base_sma:.1f})"

    # 2. Stock crossed above 200 SMA
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    curr_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    curr_sma = df['SMA200'].iloc[-1]
    prev_sma = df['SMA200'].iloc[-2]

    if pd.isna(curr_sma) or pd.isna(prev_sma):
         return False, "Insufficient data for SMA calculation"

    if curr_price <= curr_sma or prev_price > prev_sma:
         return False, f"No Crossover (Close: {curr_price:.1f} <= SMA200: {curr_sma:.1f} or Prev Close: {prev_price:.1f} was already > SMA200)"

    # 3. Volume > 2x 10-day average
    # Use slicing to get previous 10 days excluding current
    if len(df) < 11:
         return False, "Insufficient volume data"
         
    avg_vol_10 = df['Volume'].iloc[-11:-1].mean()
    curr_vol = df['Volume'].iloc[-1]

    if pd.isna(avg_vol_10) or curr_vol <= 2 * avg_vol_10:
         return False, f"Volume climax failing (Vol: {curr_vol:.0f} <= 2 * Avg: {avg_vol_10:.0f})"

    return True, f"Match (Base < 200SMA, Stock Crossover, Vol: {curr_vol/avg_vol_10:.1f}x)"

def strategy_david_dreman(data):
    """
    David Dreman "Contrarian" Screening
    - P/E Ratio: Bottom 40% (calculated dynamically)
    - Dividend Yield: > 1.5 * Market Average (calculated dynamically)
    - Payout Ratio: < 50%
    - Debt to Equity: < 0.5
    """
    info = data.get('info', {})
    if not info:
        return False, "Missing Info"

    metrics = get_market_metrics()
    
    # 1. PE Ratio
    pe_val = info.get('trailingPE')
    try:
         if pe_val is not None:
              pe = float(pe_val)
         else:
              return False, "Missing PE"
    except (TypeError, ValueError):
         return False, f"Invalid PE: {pe_val}"
    
    pe_threshold = metrics.get('pe_threshold_bottom_40', 15)
    
    if pe <= 0:
        return False, f"Invalid PE: {pe}"
    
    if pe >= pe_threshold:
        return False, f"PE too high: {pe:.2f} >= Threshold: {pe_threshold:.2f}"

    # 2. Dividend Yield
    yld = None
    try:
         yld_val = info.get('trailingAnnualDividendYield')
         if yld_val is not None:
              yld = float(yld_val)
         else:
              yld_val = info.get('dividendYield')
              if yld_val is not None:
                   yld_val = float(yld_val)
                   if yld_val > 1:
                        yld = yld_val / 100.0
                   elif yld_val > 0.1:
                        yld = yld_val / 100.0
                   else:
                        yld = yld_val
    except (TypeError, ValueError):
         yld = None
         
    avg_yld = metrics.get('avg_dividend_yield', 0.015)
    
    if yld is None or yld <= avg_yld * 1.5:
        threshold_yld = avg_yld * 1.5
        return False, f"Yield too low: {yld*100 if yld else 0:.2f}% <= Threshold: {threshold_yld*100:.2f}%"

    # 3. Payout Ratio
    payout_val = info.get('payoutRatio')
    try:
         if payout_val is not None:
              payout = float(payout_val)
         else:
              return False, "Missing Payout Ratio"
    except (TypeError, ValueError):
         return False, f"Invalid Payout Ratio: {payout_val}"
         
    if payout <= 0:
         return False, f"Invalid Payout Ratio: {payout}"
         
    if payout >= 0.50:
         return False, f"Payout Ratio too high: {payout*100:.1f}% >= 50%"

    # 4. Debt to Equity
    de_val = info.get('debtToEquity')
    try:
         if de_val is not None:
              de = float(de_val)
         else:
              return False, "Missing Debt to Equity"
    except (TypeError, ValueError):
         return False, f"Invalid Debt to Equity: {de_val}"
         
    if de < 0:
         return False, f"Invalid Debt to Equity: {de}"
         
    # Normalize D/E (If > 5, assume it's a percentage)
    normalized_de = de
    if de > 5:
         normalized_de = de / 100.0
         
    if normalized_de >= 0.5:
         return False, f"Debt/Equity too high: {normalized_de:.2f} >= 0.5"

    return True, f"Match (PE: {pe:.1f}, Yield: {yld*100:.2f}%)"

def strategy_michael_burry(data):
    """
    Michael Burry "Deep Value"
    - Enterprise Value / EBITDA < 5
    - Price to Book < 0.8
    - Price within 15% of 52-week Low
    - Proxy: Price < 200 SMA for a significant period (e.g. 30 days)
    - Institutional Selling: FII/DII holding has dropped (Weak hands exited)
    """
    info = data.get('info', {})
    df = data.get('daily')

    if not info and (df is None or df.empty):
        return False, "Missing Info and daily data"

    if df is None or df.empty or len(df) < 200:
         return False, "Insufficient Stock Data (<200 days)"

    # 1. EV / EBITDA
    ev_ebitda = None
    try:
         ev_ebitda_val = info.get('enterpriseToEbitda') or info.get('evToEbitda')
         if ev_ebitda_val:
              ev_ebitda = float(ev_ebitda_val)
    except (TypeError, ValueError):
         pass
         
    if ev_ebitda is None or ev_ebitda <= 0 or ev_ebitda >= 5:
         return False, f"EV/EBITDA too high: {ev_ebitda if ev_ebitda else 'N/A'}"

    # 2. Price to Book
    pb = None
    try:
         pb_val = info.get('priceToBook')
         if pb_val:
              pb = float(pb_val)
    except (TypeError, ValueError):
         pass
         
    if pb is None or pb <= 0 or pb >= 0.8:
         return False, f"P/B too high: {pb if pb else 'N/A'}"

    # 3. Price within 15% of 52-week Low
    curr_price = float(df['Close'].iloc[-1])
    low_52w = info.get('fiftyTwoWeekLow') or df['Low'].tail(252).min()
         
    if pd.isna(low_52w) or curr_price > float(low_52w) * 1.15:
         return False, f"Not near 52W Low (Price: {curr_price:.2f}, Low: {low_52w:.2f})"

    # 4. Proxy: Price < 200 SMA (30-day persistence)
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    recent_30 = df.tail(30)
    if (recent_30['Close'] >= recent_30['SMA200']).any():
         return False, "Price not depressed (crossed above 200 SMA recently)"

    # 5. NEW: Institutional Selling (FII/DII)
    # Note: If your data source doesn't provide historical holding, 
    # we check if current institutional holding is significantly low (< 15%)
    # or use specific keys if available (e.g., from a custom scraper).
    inst_percent = info.get('heldPercentInstitutions') # Common yfinance key
    
    # If using Indian data scrapers, you might have 'fiiHolding' and 'diiHolding'
    fii = info.get('fiiHolding', 0)
    dii = info.get('diiHolding', 0)
    total_inst = (fii + dii) if (fii or dii) else (inst_percent * 100 if inst_percent else None)

    # Logic: Burry looks for cases where institutions have ALREADY sold.
    # We flag it if total institutional holding is lower than its 1-year average
    # OR if it's currently at a multi-year low (e.g., < 10%).
    if total_inst is not None:
        if total_inst > 40: # High institutional ownership often means 'crowded'
            return False, f"Too much institutional ownership ({total_inst:.1f}%)"
    else:
        # If no institutional data, we skip this check rather than failing, 
        # but log it for transparency.
        print (f"No institutional data for {ticker}")
        pass 

    return True, f"Match (EV/EBITDA: {ev_ebitda:.1f}, PB: {pb:.1f}, Inst: {total_inst if total_inst else 'N/A'}%)"

# ==========================================
# PDF Rendering Helpers
# ==========================================

def render_pdf_standard_header(fig, title_text="Stock Analysis Report"):
    """Adds professional header and separator line to a page."""
    PRIMARY_COLOR = '#1e293b'
    BORDER_COLOR = '#94a3b8'
    
    fig.text(0.5, 0.96, title_text, ha='center', va='center', fontsize=16, weight='bold', color=PRIMARY_COLOR)
    copyright_text = f"© {datetime.now().year} Stock Research. All Rights Reserved."
    fig.text(0.5, 0.935, copyright_text, ha='center', va='center', fontsize=9, style='italic', color=BORDER_COLOR)
    
    from matplotlib.lines import Line2D
    line = Line2D([0.08, 0.92], [0.91, 0.91], transform=fig.transFigure, color=BORDER_COLOR, linewidth=1.0, alpha=0.5)
    fig.add_artist(line)

def create_pdf_title_page(pdf, timestamp):
    """Creates a professional title page for the PDF report."""
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    
    render_pdf_standard_header(fig, title_text="Famous Trader Screeners Report")
    
    PRIMARY_COLOR = '#1e293b'
    ACCENT_COLOR = '#2563eb'
    SECONDARY_COLOR = '#475569'
    
    fig.text(0.5, 0.65, "Famous Trader Screeners", ha='center', va='center', fontsize=32, weight='bold', color=ACCENT_COLOR)
    fig.text(0.5, 0.58, "Paul Tudor Jones | David Dreman | Michael Burry", ha='center', va='center', fontsize=16, weight='bold', color=SECONDARY_COLOR)
    fig.text(0.5, 0.52, f"Strategic Market Scan: {timestamp}", ha='center', va='center', fontsize=14, color=SECONDARY_COLOR)
    
    fig.text(0.15, 0.35, "Screening Pillars", fontsize=16, weight='bold', color=PRIMARY_COLOR)
    pillars = [
        "● Paul Tudor Jones: Technical Reversal",
        "● David Dreman: Contrarian Fundamental",
        "● Michael Burry: Deep Value"
    ]
    for i, p in enumerate(pillars):
        fig.text(0.15, 0.31 - (i * 0.04), p, fontsize=12, color=SECONDARY_COLOR)

    pdf.savefig(fig)
    plt.close(fig)

def create_pdf_strategies_description_page(pdf):
    """Creates a page describing the strategies and their logic."""
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    
    render_pdf_standard_header(fig, title_text="Famous Trader Strategic Foundations")
    
    PRIMARY_COLOR = '#1e293b'
    SECONDARY_COLOR = '#475569'
    ACCENT_COLOR = '#2563eb'
    
    y = 0.88
    

    
    pdf.savefig(fig)
    plt.close(fig)


def render_pdf_styled_table(pdf, df, title, description=None):
    """Renders a dataframe as a styled table in the PDF."""
    rows_per_page = 22
    if df.empty:
         num_pages = 1
    else:
         num_pages = (len(df) - 1) // rows_per_page + 1
    
    for i in range(num_pages):
        fig = plt.figure(figsize=(14, 8.5)) 
        ax = fig.add_axes([0, 0.05, 1, 0.80]) # Slightly smaller setup to fit description
        ax.axis('off')
        
        page_suffix = f" - Page {i+1}/{num_pages}" if not df.empty else ""
        render_pdf_standard_header(fig, title_text=f"{title}{page_suffix}")
        
        # Add description on first page above table
        if description and i == 0:
             fig.text(0.08, 0.88, description, fontsize=10, color='#334155', linespacing=1.4, style='italic', ha='left', va='top')
        
        if df.empty:
             # Empty state message
             fig.text(0.5, 0.45, "No stocks met the criteria for this strategy group.", 
                      fontsize=12, color='#dc2626', weight='bold', ha='center') # Reddish for visibility
        else:
             start_idx = i * rows_per_page
             end_idx = min((i + 1) * rows_per_page, len(df))
             chunk = df.iloc[start_idx:end_idx].copy()
             
             # Add table
             table = ax.table(cellText=chunk.values, colLabels=chunk.columns, loc='center', cellLoc='center')
             table.auto_set_font_size(False)
             table.set_fontsize(8)
             table.scale(1.0, 1.4) 
             
             for k, cell in table.get_celld().items():
                  if k[0] == 0:  # Header
                       cell.set_text_props(weight='bold', color='white')
                       cell.set_facecolor('#1e3a8a') # Navy
                  else:
                       cell_text = cell.get_text().get_text().strip()
                       if cell_text == "PASSED":
                            cell.set_facecolor('#dcfce7') # Light green
                            cell.get_text().set_color('#166534') # Dark green text
                       elif cell_text == "FAILED":
                            cell.set_facecolor('#fee2e2') # Light red
                            cell.get_text().set_color('#991b1b') # Dark red text
        
        pdf.savefig(fig)
        plt.close(fig)

# ==========================================
# Main Execution Flow
# ==========================================

def run_screener(tickers, refresh=False, output_dir=None):
    """
    Main driver to run the famous trader screeners on a list of tickers.
    
    Args:
        tickers (list): List of ticker symbols to scan.
        refresh (bool): Whether to force refresh of cached data.
        output_dir (str, optional): Directory to save results. Defaults to timestamped folder.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    # Calculate Market Metrics once
    metrics = load_market_metrics(tickers)
    global GLOBAL_MARKET_METRICS
    GLOBAL_MARKET_METRICS = metrics

    strategies = [
        ("Paul Tudor Jones", strategy_paul_tudor_jones),
        ("David Dreman", strategy_david_dreman),
        ("Michael Burry", strategy_michael_burry),
    ]

    results = []

    console.print(f"[bold green]Scanning {len(tickers)} tickers for famous trader strategies...[/bold green]")
    
    for ticker in tickers:
        data = load_data(ticker, refresh=refresh)
        if not data.get('info') and (data.get('daily') is None or data.get('daily').empty):
            continue

        stock_matches = {'Ticker': ticker}
        has_match = False

        for name, func in strategies:
            try:
                is_match, reason = func(data)
                if is_match:
                    stock_matches[name] = "PASSED"
                    has_match = True
                else:
                    stock_matches[name] = "FAILED"
            except Exception as e:
                stock_matches[name] = "ERROR"
                console.print(f"[red]Error processing {ticker} for {name}: {e}[/red]")

        if has_match:
            results.append(stock_matches)
            console.print(f"[green]+ {ticker} passed one or more strategies.[/green]")

    if results:
        df = pd.DataFrame(results)
        # Reorder to list Ticker first
        cols = ['Ticker'] + [s[0] for s in strategies]
        df = df[cols]
        
        # --- High Level Analytics: Score Calculation ---
        strategy_names = [s[0] for s in strategies]
        df['Score'] = (df[strategy_names] == 'PASSED').sum(axis=1)
        
        # Sort by Score descending
        df_all = df.sort_values(by='Score', ascending=False).copy()
        df_all = df_all[['Ticker', 'Score'] + strategy_names]
        
        # Create Top 50
        df_top50 = df_all.head(50).copy()
        
        console.print("\n[bold]--- SCREEEN RESULTS SUMMARY (Top 50) ---[/bold]")
        print(df_top50.to_string(index=False))
        
        # Save Top 50 to CSV
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "famous_trader_screen_results.csv")
        df_all.to_csv(out_path, index=False)
        console.print(f"\n[bold]Results saved to {out_path}[/bold]")

        # --- Generate PDF Report ---
        temp_title = os.path.join(output_dir, f"temp_title.pdf")
        temp_desc = os.path.join(output_dir, f"temp_desc.pdf")
        temp_table = os.path.join(output_dir, f"temp_table.pdf")
        pdf_path = os.path.join(output_dir, f"Famous_Trader_Screeners_Report.pdf")
        
        try:
            from pypdf import PdfWriter
            
            # 1. Create Title Page
            with PdfPages(temp_title) as pdf_t:
                create_pdf_title_page(pdf_t, TIMESTAMP)



            # 3. Create Summary Table pages
            with PdfPages(temp_table) as pdf_tab:
                render_pdf_styled_table(pdf_tab, df_top50, "Top 50 Ranked Stocks")
                
                # Section B: PTJ
                df_ptj = df_all[df_all['Paul Tudor Jones'] == 'PASSED']
                render_pdf_styled_table(pdf_tab, df_ptj.head(50), 
                                        "Paul Tudor Jones '200-Day Defense'",
                                        description="Signal: Stock just crossed above its 200-day SMA on high volume while Index is below its 200-day SMA.")

                # Section C: Dreman
                df_dreman = df_all[df_all['David Dreman'] == 'PASSED']
                render_pdf_styled_table(pdf_tab, df_dreman.head(50), 
                                        "David Dreman 'Contrarian'",
                                        description="Signal: Low PE, high yield unloved giants with safe payout and low debt.")

                # Section D: Burry
                df_burry = df_all[df_all['Michael Burry'] == 'PASSED']
                render_pdf_styled_table(pdf_tab, df_burry.head(50), 
                                        "Michael Burry 'Deep Value'",
                                        description="Signal: Extremely cheap (EV/EBITDA < 5, PB < 0.8) near 52W low.")

            # 4. Merge PDFs
            merger = PdfWriter()
            merger.append(temp_title)
            
            famous_trader_pdf = "/Users/solankianshul/Documents/projects/stock_research/breakout_screeners/bear_market_screener/famous_trader_screener.pdf"
            if os.path.exists(famous_trader_pdf):
                 merger.append(famous_trader_pdf)
            else:
                 console.print(f"[yellow]Warning: {famous_trader_pdf} not found to merge.[/yellow]")
                 
            merger.append(temp_table)

            with open(pdf_path, 'wb') as f:
                 merger.write(f)

            console.print(f"\n[bold green]Report generated: {pdf_path}[/bold green]")
            
            # Cleanup
            try:
                os.remove(temp_title)
                os.remove(temp_desc)
                os.remove(temp_table)
            except OSError:
                 pass
        except Exception as e:
            console.print(f"[red]Error generating PDF: {e}[/red]")
    else:
        console.print("[yellow]No stocks passed any strategy.[/yellow]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Famous Trader Bear Market Screeners')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of tickers to process')
    parser.add_argument('--ticker', type=str, default=None, help='Process a single ticker')
    args = parser.parse_args()

    # Load tickers
    if args.ticker:
        tickers = [args.ticker]
    else:
        tickers = load_tickers(limit=args.limit)

    if not tickers:
        console.print("[red]No tickers found or error loading tickers.[/red]")
        exit(1)

    # Run screener
    run_screener(tickers, refresh=False)
