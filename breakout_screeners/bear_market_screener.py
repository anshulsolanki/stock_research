# -------------------------------------------------------------------------------
# Project: Stock Analysis (https://github.com/anshulsolanki/stock_analysis)
# Author:  Anshul Solanki
# License: MIT License
# -------------------------------------------------------------------------------

"""
Bear Market Stock Screener

This script implements 10 strategies for identifying strong stocks in a bear market,
combining technical resilience and fundamental strength.
Based on the methodologies described in Strategies.pdf.

Usage:
------
python3 bear_market_screener.py [--limit N] [--ticker TICKER] [--test]
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
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')
CACHE_DIR = os.path.join(BASE_DIR, 'data_cache')
JSON_PATH = os.path.join(DATA_DIR, 'nifty_500.json')
NSE_BASELINE = "NSEI_baseline.csv"
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = os.path.join(BASE_DIR, 'screener_results', 'bear_market', TIMESTAMP)

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
    
    render_pdf_standard_header(fig, title_text="Bear Market Screener Report")
    
    PRIMARY_COLOR = '#1e293b'
    ACCENT_COLOR = '#2563eb'
    SECONDARY_COLOR = '#475569'
    
    fig.text(0.5, 0.65, "Bear Market Screener", ha='center', va='center', fontsize=32, weight='bold', color=ACCENT_COLOR)
    fig.text(0.5, 0.58, "Defensive & Value Analysis", ha='center', va='center', fontsize=18, weight='bold', color=SECONDARY_COLOR)
    fig.text(0.5, 0.52, f"Strategic Market Scan: {timestamp}", ha='center', va='center', fontsize=14, color=SECONDARY_COLOR)
    
    fig.text(0.15, 0.35, "Screening Pillars", fontsize=16, weight='bold', color=PRIMARY_COLOR)
    pillars = [
        "● Defensive Prices & Margins",
        "● Strong Cashflow & Dividends",
        "● Low Leverage & Sound Valuation"
    ]
    for i, p in enumerate(pillars):
        fig.text(0.15, 0.31 - (i * 0.04), p, fontsize=12, color=SECONDARY_COLOR)

    pdf.savefig(fig)
    plt.close(fig)

def render_pdf_styled_table(pdf, df, title, description=None):
    """Renders a dataframe as a styled table in the PDF."""
    rows_per_page = 22
    if df.empty:
         num_pages = 1
    else:
         num_pages = (len(df) // rows_per_page) + 1
    
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
        
        pdf.savefig(fig)
        plt.close(fig)

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
    """Loads all available data for a ticker from cache, or refreshes it via yfinance."""
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
    
    if refresh:
        try:
             import yfinance as yf
             stock = yf.Ticker(ticker)
             
             info = stock.info
             if info:
                  with open(info_path, 'w') as f:
                       json.dump(info, f)
                       
             df_1d = stock.history(period="2y")
             if not df_1d.empty:
                  df_1d.to_csv(os.path.join(CACHE_DIR, paths['daily']))
                  
             df_wk = stock.history(period="5y") 
             if not df_wk.empty:
                  df_wk.to_csv(os.path.join(CACHE_DIR, paths['weekly']))
                  
             for attr in ['financials', 'quarterly_financials', 'balance_sheet', 'cashflow']:
                  val = getattr(stock, attr, None)
                  if val is not None and not val.empty:
                       val.to_csv(os.path.join(CACHE_DIR, paths[attr]))
                       
        except Exception as e:
             console.print(f"[yellow]Error refreshing {ticker}: {e}[/yellow]")

    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            data['info'] = json.load(f)
    else:
        data['info'] = {}

    # 2. Dataframes
    for key, filename in paths.items():
        if key == 'info': continue
        path = os.path.join(CACHE_DIR, filename)
        if os.path.exists(path):
            try:
                # Parse dates for technical history
                if key in ['daily', 'weekly']:
                    data[key] = pd.read_csv(path, index_col=0, parse_dates=True)
                else:
                    data[key] = pd.read_csv(path, index_col=0)
            except Exception:
                data[key] = pd.DataFrame()
        else:
            data[key] = pd.DataFrame()

    # Load Baseline
    baseline_path = os.path.join(CACHE_DIR, NSE_BASELINE)
    if os.path.exists(baseline_path):
        try:
            data['baseline'] = pd.read_csv(baseline_path, index_col=0, parse_dates=True)
        except Exception:
            data['baseline'] = pd.DataFrame()
    else:
        data['baseline'] = pd.DataFrame()

    return data

def get_latest_financial_value(df, row_name):
    """Safely extracts the latest value from a financial dataframe row."""
    if df.empty or row_name not in df.index:
        return None
    try:
        # Columns are usually dates, sorted descending or ascending
        # We assume columns are sorted descending (latest first) or we sort them
        cols = df.columns.tolist()
        # Ensure they are dates or sorted
        # Usually yfinance saves with date headers. 
        # Let's just grab the first column assuming it's the latest
        val = df.loc[row_name].iloc[0]
        if pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None

def get_row_average(df, row_name):
    """Calculates average of a row in a financial dataframe."""
    if df.empty or row_name not in df.index:
        return None
    try:
        vals = pd.to_numeric(df.loc[row_name], errors='coerce').dropna()
        if vals.empty:
            return None
        return float(vals.mean())
    except Exception:
        return None

# ==========================================
# Strategy Implementations
# ==========================================

def strategy_1_relative_strength(data):
    """
    1. Relative Strength Screeners
    - Price > 200 DMA
    - Relative Strength rising (Outperforming Index)
    - Price near 52-week high
    - High volume accumulation
    - Strong earnings growth
    """
    df = data.get('daily')
    baseline = data.get('baseline')
    info = data.get('info', {})

    if df is None or df.empty or len(df) < 200:
        return False, "Insufficient Price Data (<200 days)"

    if baseline is None or baseline.empty:
        return False, "Missing Baseline Index Data"

    curr_price = df['Close'].iloc[-1]
    
    # 1. Price > 200 DMA
    sma200 = df['Close'].rolling(window=200).mean().iloc[-1]
    if curr_price <= sma200:
        return False, "Price below 200 DMA"

    # 2. Relative Strength Rising (20-day ratio trend slope)
    if len(df) >= 20 and len(baseline) >= 20:
        combined = pd.DataFrame({'Stock': df['Close'], 'Index': baseline['Close']}).dropna()
        if len(combined) >= 20:
             combined['RS'] = combined['Stock'] / combined['Index']
             rs_recent = combined['RS'].tail(20)
             x = np.arange(len(rs_recent))
             y = rs_recent.values
             slope = np.polyfit(x, y, 1)[0]
             if slope <= 0:
                  return False, f"Relative Strength failing slope check ({slope:.6f})"

    # 3. Price near 52-week high (Within 15%)
    high_52w = df['High'].rolling(window=252).max().iloc[-1]
    if curr_price < high_52w * 0.85:
         return False, "Not near 52W High"

    # 4. High Volume Accumulation (Up-Days Volume > Down-Days Volume over 20 days)
    recent = df.tail(20).copy()
    recent['Price_Chg'] = recent['Close'].diff()
    # Handle Up Days vs Down Days Volume Means
    up_days = recent[recent['Price_Chg'] > 0]
    down_days = recent[recent['Price_Chg'] < 0]
    
    up_vol = up_days['Volume'].mean() if not up_days.empty else 0
    down_vol = down_days['Volume'].mean() if not down_days.empty else 0

    if up_vol <= down_vol:
         return False, f"Distribution zone: Up-Vol ({up_vol:.0f}) <= Down-Vol ({down_vol:.0f})"

    # 5. Earnings Growth (Fundamental)
    eps_growth = info.get('earningsQuarterlyGrowth', 0)
    eps_growth = eps_growth if eps_growth is not None else 0
    if eps_growth < 0.15: # 15% growth thresh
         return False, "Weak Earnings Growth (<15%)"

    return True, f"Match (SMA200: {sma200:.1f}, 52WH: {high_52w:.1f}, UpVol: {up_vol:.0f}, EPS_g: {eps_growth*100:.1f}%)"

def strategy_2_deep_value(data):
    """
    2. Deep Value Screeners
    - PE < 15 (Proxy for historical median PE)
    - Price / Book < 2
    - ROCE > 15%
    - Debt/Equity < 0.5
    - Sales growth > 10%
    """
    info = data.get('info', {})
    fin = data.get('financials')
    bs = data.get('balance_sheet')

    # 1. Info Data
    pe = info.get('trailingPE')
    pb = info.get('priceToBook')
    de = info.get('debtToEquity') # often in percentage or ratio

    # Safe checks
    if pe is None or pe > 15: return False, f"PE too high: {pe}"
    if pb is None or pb > 2.0: return False, f"PB too high: {pb}"
    
    # D/E handling (yfinance sometimes gives 100 for 1:1)
    if de is not None:
         de_val = de / 100.0 if de > 5 else de
         if de_val > 0.5: return False, f"D/E too high: {de_val:.2f}"

    # 2. ROCE = EBIT / (Total Assets - Current Liabilities)
    ebit = get_latest_financial_value(fin, 'EBIT')
    total_assets = get_latest_financial_value(bs, 'Total Assets')
    curr_liab = get_latest_financial_value(bs, 'Current Liabilities')

    if ebit and total_assets and curr_liab:
        capital_employed = total_assets - curr_liab
        if capital_employed > 0:
            roce = ebit / capital_employed
            if roce < 0.15: return False, f"ROCE too low: {roce*100:.1f}%"
        else: return False, "Invalid Capital Employed"
    else:
        return False, "Missing ROCE data"

    # 3. Sales Growth > 10%
    if fin is not None and not fin.empty and 'Total Revenue' in fin.index:
        try:
            revs = fin.loc['Total Revenue'].iloc[:2] # Latest 2 years
            if len(revs) >= 2:
                 growth = (revs.iloc[0] - revs.iloc[1]) / revs.iloc[1]
                 if growth < 0.10: return False, f"Sales Growth too low: {growth*100:.1f}%"
        except Exception: pass

    return True, f"Match (PE: {pe:.1f}, PB: {pb:.1f}, ROCE: {roce*100:.1f}%)"

def strategy_3_cash_flow_yield(data):
    """
    3. Cash Flow Yield Screeners
    - Free Cash Flow Yield > 5%
    - Debt/Equity < 0.5
    - ROE > 15%
    """
    info = data.get('info', {})
    cf = data.get('cashflow')

    # 1. FCF Yield = FCF / Market Cap
    fcf = get_latest_financial_value(cf, 'Free Cash Flow')
    mcap = info.get('marketCap')

    if fcf and mcap and mcap > 0:
        fcf_yield = fcf / mcap
        if fcf_yield < 0.05: return False, f"FCF Yield too low: {fcf_yield*100:.1f}%"
    else:
        return False, "Missing FCF or Market Cap"

    # 2. D/E and ROE
    de = info.get('debtToEquity')
    roe = info.get('returnOnEquity')

    if de is not None:
         de_val = de / 100.0 if de > 5 else de
         if de_val > 0.5: return False, f"D/E too high: {de_val:.2f}"
         
    if roe is None or roe < 0.15: return False, f"ROE too low: {roe}"

    return True, f"Match (FCF Yield: {fcf_yield*100:.1f}%, ROE: {roe*100:.1f}%)"

def strategy_4_dollar_earner(data):
    """
    4. "Dollar Earner" screener (Modified)
    - Debt to Equity < 0.2
    - Operating Profit Margin > 20%
    - Return on Capital (ROCE) > 25%
    *Note: Export Revenue % criteria omitted due to lack of standard data.*
    """
    info = data.get('info', {})
    fin = data.get('financials')
    bs = data.get('balance_sheet')

    de = info.get('debtToEquity')
    op_margin = info.get('operatingMargins')

    if de is not None:
         de_val = de / 100.0 if de > 5 else de
         if de_val > 0.2: return False, f"D/E too high: {de_val:.2f}"

    if op_margin is None or op_margin < 0.20: return False, f"Op Margin too low: {op_margin}"

    # ROCE
    ebit = get_latest_financial_value(fin, 'EBIT')
    total_assets = get_latest_financial_value(bs, 'Total Assets')
    curr_liab = get_latest_financial_value(bs, 'Current Liabilities')

    if ebit and total_assets and curr_liab:
            capital_employed = total_assets - curr_liab
            if capital_employed > 0:
                roce = ebit / capital_employed
                if roce < 0.25: return False, f"ROCE too low: {roce*100:.1f}%"
            else: return False, "Invalid Capital Employed"
    else:
         return False, "Missing ROCE data"

    return True, f"Match (D/E: {de_val:.2f}, OpMargin: {op_margin*100:.1f}%, ROCE: {roce*100:.1f}%)"

def strategy_5_acquirers_multiple(data):
    """
    5. The Acquirer's Multiple
    - EV / EBIT < 8
    - Debt to Equity < 0.5
    """
    info = data.get('info', {})
    fin = data.get('financials')

    ev = info.get('enterpriseValue')
    ebit = get_latest_financial_value(fin, 'EBIT')

    if ev and ebit and ebit > 0:
        mult = ev / ebit
        if mult > 8: return False, f"EV/EBIT too high: {mult:.1f}"
    else:
        return False, "Missing EV or EBIT (or EBIT <= 0)"

    de = info.get('debtToEquity')
    if de is not None:
         de_val = de / 100.0 if de > 5 else de
         if de_val > 0.5: return False, f"D/E too high: {de_val:.2f}"

    return True, f"Match (EV/EBIT: {mult:.1f}, D/E: {de_val:.2f})"

def strategy_6_net_nets(data):
    """
    6. Benjamin Graham’s "Net-Nets"
    - Current Assets - Total Liabilities > Market Cap (NCAV)
    - Price to Book Value < 0.7
    - Current Ratio > 2
    """
    info = data.get('info', {})
    bs = data.get('balance_sheet')

    curr_assets = get_latest_financial_value(bs, 'Current Assets')
    total_liab = get_latest_financial_value(bs, 'Total Liabilities Net Minority Interest')
    mcap = info.get('marketCap')

    if curr_assets and total_liab and mcap:
        ncav = curr_assets - total_liab
        if ncav <= mcap: return False, f"NCAV ({ncav/1e7:.1f}Cr) <= Market Cap ({mcap/1e7:.1f}Cr)"
    else:
        return False, "Missing Balance Sheet (NCAV) data"

    pb = info.get('priceToBook')
    curr_ratio = info.get('currentRatio')

    if pb is None or pb >= 0.7: return False, f"PB too high: {pb}"
    if curr_ratio is None or curr_ratio <= 2.0: return False, f"Current Ratio too low: {curr_ratio}"

    return True, f"Match (NCAV/Cap: {ncav/mcap:.1f}x, PB: {pb:.1f})"

def strategy_7_magic_formula(data):
    """
    7. Joel Greenblatt’s Magic Formula
    - EBIT / Enterprise Value (Earnings Yield) > 8% (Proxy)
    - ROCE > 20%
    """
    info = data.get('info', {})
    fin = data.get('financials')
    bs = data.get('balance_sheet')

    ev = info.get('enterpriseValue')
    ebit = get_latest_financial_value(fin, 'EBIT')

    if ev and ebit and ev > 0:
        yield_pct = ebit / ev
        if yield_pct < 0.08: return False, f"Earnings Yield too low: {yield_pct*100:.1f}%"
    else:
        return False, "Missing EV or EBIT"

    # ROCE
    total_assets = get_latest_financial_value(bs, 'Total Assets')
    curr_liab = get_latest_financial_value(bs, 'Current Liabilities')

    if total_assets and curr_liab:
        capital_employed = total_assets - curr_liab
        if capital_employed > 0:
            roce = ebit / capital_employed
            if roce < 0.20: return False, f"ROCE too low: {roce*100:.1f}%"
        else: return False, "Invalid Capital Employed"
    else:
        return False, "Missing ROCE data"

    return True, f"Match (Yield: {yield_pct*100:.1f}%, ROCE: {roce*100:.1f}%)"

def strategy_8_hedge_fund(data):
    """
    8. Hedge Fund Screener (Relative Resilience)
    - Index making new 20-day low
    - Stock NOT making new 20-day low
    - Volume declining on declines
    """
    df = data.get('daily')
    baseline = data.get('baseline')

    if df is None or df.empty or len(df) < 20 or baseline is None or baseline.empty or len(baseline) < 20:
        return False, "Insufficient Price Data"

    # Slice last 20 days
    recent_stock = df.tail(20)
    recent_base = baseline.tail(20)

    # 1. Index making new 20-day low (Current close is minimum of last 20)
    current_base = recent_base['Close'].iloc[-1]
    min_base_20 = recent_base['Close'].min()
    if current_base > min_base_20 * 1.005: # within 0.5% of min
         return False, "Index not at new low"

    # 2. Stock NOT making new 20-day low
    current_stock = recent_stock['Close'].iloc[-1]
    min_stock_20 = recent_stock['Close'].min()
    # Check if current stock price is safely above the 20-day low
    if current_stock <= min_stock_20 * 1.02: # Less than 2% above low
         return False, "Stock near 20-day low"

    # 3. Volume declining on declines
    if current_stock < recent_stock['Open'].iloc[-1]: # Today is red
         recent_vol = recent_stock['Volume'].iloc[-1]
         avg_vol = df['Volume'].rolling(window=20).mean().iloc[-1]
         if recent_vol >= avg_vol:
              return False, f"Heavy volume on decline ({recent_vol} >= {avg_vol:.0f})"

    return True, f"Resilient (Stock/Index Diff)"

def strategy_9_margin_resilience(data):
    """
    9. Margin Resilience
    - Gross Margin (Current Quarter) >= Gross Margin (5-Year Average)
    - Interest Coverage Ratio > 5
    """
    fin = data.get('financials')
    q_fin = data.get('quarterly_financials')

    if q_fin.empty or 'Gross Profit' not in q_fin.index or 'Total Revenue' not in q_fin.index:
         return False, "Missing Quarterly Margin data"

    # 1. Gross Margin Current Quarter
    try:
        q_gp = q_fin.loc['Gross Profit'].iloc[0]
        q_rev = q_fin.loc['Total Revenue'].iloc[0]
        if q_rev > 0:
             q_margin = q_gp / q_rev
        else: return False, "Zero Revenue"
    except Exception: return False, "Error calc Q Margin"

    # 2. 5-Year Average Gross Margin (using annuals as proxy)
    avg_margin = None
    if fin is not None and not fin.empty and 'Gross Profit' in fin.index:
        try:
             gps = fin.loc['Gross Profit']
             revs = fin.loc['Total Revenue']
             margins = gps / revs
             avg_margin = margins.mean()
        except Exception: pass

    if avg_margin is not None:
         if q_margin < avg_margin: return False, f"Q_Margin ({q_margin*100:.1f}%) < Avg ({avg_margin*100:.1f}%)"
    
    # 3. Interest Coverage = EBIT / Interest Expense
    ebit = get_latest_financial_value(fin, 'EBIT')
    interest = get_latest_financial_value(fin, 'Interest Expense')

    if ebit and interest and interest > 0:
        ic = ebit / interest
        if ic <= 5: return False, f"Interest Coverage too low: {ic:.1f}"
    else:
        return False, "Missing EBIT or Interest"

    return True, f"Match (Q_Margin: {q_margin*100:.1f}%, IntCover: {ic:.1f})"

def strategy_10_dividend_fortress(data):
    """
    10. The Dividend Fortress
    - Dividend Yield > 4%
    - Dividend Payout Ratio < 60%
    - 5-Year Dividend Growth: Positive
    """
    info = data.get('info', {})
    daily = data.get('daily')

    yld = info.get('dividendYield')
    payout = info.get('payoutRatio')

    # Float handling
    yld = yld if yld is not None else 0
    payout = payout if payout is not None else 2.0 # Force filter out if missing

    if yld < 0.04: return False, f"Yield too low: {yld*100:.1f}%"
    if payout > 0.60: return False, f"Payout too high: {payout*100:.1f}%"

    # 5-Year Dividend Growth Verification
    if daily is not None and not daily.empty and 'Dividends' in daily.columns:
        try:
            ann_div = daily['Dividends'].groupby(daily.index.year).sum()
            if len(ann_div) >= 2:
                 curr_div = ann_div.iloc[-1]
                 start_div = ann_div.iloc[-5] if len(ann_div) >= 5 else ann_div.iloc[0]
                 
                 if curr_div <= start_div:
                      return False, f"No 5-Year Growth: {start_div:.2f} to {curr_div:.2f}"
        except Exception:
             pass # Fallback to Yield filters if math fails

    return True, f"Match (Yield: {yld*100:.1f}%, Payout: {payout*100:.1f}%)"

# ==========================================
# Main Execution Flow
# ==========================================

def run_screener(tickers, refresh=False):
    strategies = [
        ("Relative Strength", strategy_1_relative_strength),
        ("Deep Value", strategy_2_deep_value),
        ("Cash Flow Yield", strategy_3_cash_flow_yield),
        ("Dollar Earner (Mod)", strategy_4_dollar_earner),
        ("Acquirer Multiple", strategy_5_acquirers_multiple),
        ("Net-Nets", strategy_6_net_nets),
        ("Magic Formula", strategy_7_magic_formula),
        ("Hedge Fund Screener", strategy_8_hedge_fund),
        ("Margin Resilience", strategy_9_margin_resilience),
        ("Dividend Fortress", strategy_10_dividend_fortress)
    ]

    results = []

    console.print(f"[bold green]Scanning {len(tickers)} tickers...[/bold green]")
    
    for ticker in tickers:
        data = load_data(ticker, refresh=refresh)
        if not data.get('info') and data.get('daily').empty:
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
        # Sum 1 for every 'PASSED' strategy
        df['Score'] = (df[strategy_names] == 'PASSED').sum(axis=1)
        
        # Sort by Score descending
        df_all = df.sort_values(by='Score', ascending=False).copy()
        df_all = df_all[['Ticker', 'Score'] + strategy_names]
        
        # Create Top 50
        df_top50 = df_all.head(50).copy()
        
        console.print("\n[bold]--- SCREEEN RESULTS SUMMARY (Top 50) ---[/bold]")
        print(df_top50.to_string(index=False))
        
        # Save Top 50 to CSV (or All to CSV, usually All is better for analysis)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "bear_market_screen_results.csv")
        df_all.to_csv(out_path, index=False)
        console.print(f"\n[bold]Results saved to {out_path}[/bold]")

        # --- Generate PDF Report ---
        temp_title = os.path.join(OUTPUT_DIR, f"temp_title.pdf")
        temp_table = os.path.join(OUTPUT_DIR, f"temp_table.pdf")
        pdf_path = os.path.join(OUTPUT_DIR, f"Bear_Market_Screener_Results.pdf")
        
        try:
            from pypdf import PdfWriter
            
            # 1. Create Title Page Page (Page 1)
            with PdfPages(temp_title) as pdf_t:
                create_pdf_title_page(pdf_t, TIMESTAMP)

            # 2. Create Summary Table pages
            with PdfPages(temp_table) as pdf_tab:
                # Section A: Top 50 
                render_pdf_styled_table(pdf_tab, df_top50, "Top 50 Ranked Stocks")
                # Section B: New Leaders (Momentum & Resilience)
                df_new_leaders = df_all[(df_all['Relative Strength'] == 'PASSED') & (df_all['Hedge Fund Screener'] == 'PASSED')]
                if not df_new_leaders.empty:
                    # Print to console too
                    console.print("\n[bold]--- NEW LEADERS (Momentum & Resilience) ---[/bold]")
                    print(df_new_leaders.head(50)[['Ticker', 'Score']].to_string(index=False))
                else:
                    console.print("\n[yellow]No stocks met the 'New Leaders' criteria (S1 + S8) - Still generating page.[/yellow]")
                    
                render_pdf_styled_table(pdf_tab, df_new_leaders.head(50), 
                                        "New Leaders (Momentum & Resilience)", 
                                        description="Strategies 1 (Relative Strength) + 8 (Hedge Fund Resilience)\n"
                                                    "Goal: Find stocks that the market refuses to sell.\n"
                                                    "Logic: These stocks have already \"bottomed\" and are being accumulated by institutions before the rest of the market.\n"
                                                    "Compatibility: High. S1 and S8 both look for price strength relative to a falling index.")

                # Section C: Quality Survivors (Moat & Cash)
                df_quality = df_all[
                    (df_all['Cash Flow Yield'] == 'PASSED') & 
                    (df_all['Magic Formula'] == 'PASSED') & 
                    (df_all['Margin Resilience'] == 'PASSED') & 
                    (df_all['Dividend Fortress'] == 'PASSED')
                ]
                if not df_quality.empty:
                    # Print to console too
                    console.print("\n[bold]--- QUALITY SURVIVORS (Moat & Cash) ---[/bold]")
                    print(df_quality.head(50)[['Ticker', 'Score']].to_string(index=False))
                else:
                    console.print("\n[yellow]No stocks met the 'Quality Survivors' criteria (S3 + S7 + S9 + S10) - Still generating page.[/yellow]")
                    
                render_pdf_styled_table(pdf_tab, df_quality.head(50), 
                                        "Quality Survivors (Moat & Cash)",
                                        description="Strategies: 3 (Cash Flow Yield) + 7 (Magic Formula) + 9 (Margin Resilience) + 10 (Dividend Fortress)\n"
                                                    "Goal: Buy robust businesses that generate cold, hard cash.\n"
                                                    "Logic: In a bear market, \"Profit is an opinion, but Cash is a fact.\"\n"
                                                    "Compatibility: Excellent. High-yield, high-margin, high-ROCE companies.")

                # Section D: Deep Value (Mean Reversion)
                df_deep_value = df_all[
                    (df_all['Deep Value'] == 'PASSED') & 
                    (df_all['Acquirer Multiple'] == 'PASSED') & 
                    (df_all['Net-Nets'] == 'PASSED')
                ]
                if not df_deep_value.empty:
                    # Print to console too
                    console.print("\n[bold]--- DEEP VALUE (Mean Reversion) ---[/bold]")
                    print(df_deep_value.head(50)[['Ticker', 'Score']].to_string(index=False))
                else:
                    console.print("\n[yellow]No stocks met the 'Deep Value' criteria (S2 + S5 + S6) - Still generating page.[/yellow]")
                    
                render_pdf_styled_table(pdf_tab, df_deep_value.head(50), 
                                        "Deep Value (Mean Reversion)",
                                        description="Strategies: 2 (Deep Value), 5 (Acquirer's Multiple), 6 (Net-Nets)\n"
                                                    "Goal: Buy stocks so cheap that the \"downside is capped\" by the balance sheet.\n"
                                                    "Logic: Buying $1.00 of assets for $0.50.\n"
                                                    "Compatibility: High within the tribe, extremely contradictory to Team Momentum.")

                # Section E: Institutional Favorite Combo (Conservative Growth)
                df_inst = df_all[(df_all['Relative Strength'] == 'PASSED') & (df_all['Margin Resilience'] == 'PASSED')]
                if not df_inst.empty:
                    console.print("\n[bold]--- INSTITUTIONAL FAVORITE COMBO ---[/bold]")
                    print(df_inst.head(50)[['Ticker', 'Score']].to_string(index=False))
                else:
                    console.print("\n[yellow]No stocks met the 'Institutional Favorite' criteria (S1 + S9) - Still generating page.[/yellow]")
                    
                render_pdf_styled_table(pdf_tab, df_inst.head(50), 
                                        "Institutional Favorite Combo (Conservative Growth)",
                                        description="Strategies: 1 (Relative Strength) + 9 (Margin Resilience)\n"
                                                    "Goal: Finds stocks that are showing price strength because their business margins are expanding during the crisis.\n")

                # Section F: War Chest Combo (Safe Income)
                df_war = df_all[(df_all['Cash Flow Yield'] == 'PASSED') & (df_all['Dividend Fortress'] == 'PASSED')]
                if not df_war.empty:
                    console.print("\n[bold]--- WAR CHEST COMBO (Safe Income) ---[/bold]")
                    print(df_war.head(50)[['Ticker', 'Score']].to_string(index=False))
                else:
                    console.print("\n[yellow]No stocks met the 'War Chest' criteria (S3 + S10) - Still generating page.[/yellow]")
                    
                render_pdf_styled_table(pdf_tab, df_war.head(50), 
                                        "War Chest Combo (Safe Income)",
                                        description="Strategies: 3 (Cash Flow Yield) + 10 (Dividend Fortress)\n"
                                                    "Goal: Finds stocks where the dividend is backed by massive free cash flow, providing a 'hard floor' for the price.\n")

                # Section G: Ugly Duckling Combo (Deep Value)
                df_ugly = df_all[(df_all['Acquirer Multiple'] == 'PASSED') & (df_all['Net-Nets'] == 'PASSED')]
                if not df_ugly.empty:
                    console.print("\n[bold]--- UGLY DUCKLING COMBO (Deep Value) ---[/bold]")
                    print(df_ugly.head(50)[['Ticker', 'Score']].to_string(index=False))
                else:
                    console.print("\n[yellow]No stocks met the 'Ugly Duckling' criteria (S5 + S6) - Still generating page.[/yellow]")
                    
                render_pdf_styled_table(pdf_tab, df_ugly.head(50), 
                                        "Ugly Duckling Combo (Deep Value)",
                                        description="Strategies: 5 (Acquirer's Multiple) + 6 (Net-Nets)\n"
                                                    "Goal: Finds absolute cheapest stocks in market that are ignored, suitable only for very long-term investors (3-5 years).\n")

                # Section H: The "Future Leader"
                df_future = df_all[
                    (df_all['Relative Strength'] == 'PASSED') & 
                    (df_all['Hedge Fund Screener'] == 'PASSED') & 
                    (df_all['Magic Formula'] == 'PASSED') & 
                    (df_all['Net-Nets'] == 'FAILED')
                ]
                if not df_future.empty:
                    console.print("\n[bold]--- THE FUTURE LEADER ---[/bold]")
                    print(df_future.head(50)[['Ticker', 'Score']].to_string(index=False))
                else:
                    console.print("\n[yellow]No stocks met the 'Future Leader' criteria (S1 + S8 + S7 and NOT S6) - Still generating page.[/yellow]")
                    
                render_pdf_styled_table(pdf_tab, df_future.head(50), 
                                        "The Future Leader",
                                        description="Goal: To identify high-quality businesses that institutions are actively defending and accumulating while market crashes.\n\n"
                                                    "Fundamental Validation (S7): Ensures the company is a \"cash machine\" with high returns on capital.\n"
                                                    "Institutional Footprints (S8): Spots stocks that refuse to make new lows, signaling big players quietly buying.\n"
                                                    "Momentum Launchpad (S1): Confirms the stock has already broken away from the bear market trend.")

                # Section I: All Strategy Matches
                render_pdf_styled_table(pdf_tab, df_all, "All Strategy Matches")
            
            # 3. Merge sequentially
            merger = PdfWriter()
            merger.append(temp_title) # Front Cover
            
            strategies_pdf = os.path.join(BASE_DIR, "Strategies.pdf")
            if os.path.exists(strategies_pdf):
                 merger.append(strategies_pdf) # Intermediate Pages
            else:
                 console.print(f"[yellow]Warning: {strategies_pdf} not found to merge.[/yellow]")
                 
            merger.append(temp_table) # Back summary grids
            
            merger.write(pdf_path)
            merger.close()
            
            # Clean up temps
            if os.path.exists(temp_title): os.remove(temp_title)
            if os.path.exists(temp_table): os.remove(temp_table)
                 
            console.print(f"[bold green]PDF Report saved to {pdf_path}[/bold green]")
        except Exception as e:
            console.print(f"[red]Error saving PDF Report: {e}[/red]")
    else:
        console.print("\n[yellow]No strategy matches found.[/yellow]")

def main():

    parser = argparse.ArgumentParser(description="Bear Market Stock Screener")
    parser.add_argument('--limit', type=int, help="Limit number of stocks to scan")
    parser.add_argument('--ticker', type=str, help="Scan a single specific ticker")
    parser.add_argument('--test', action='store_true', help="Run verification test cases")
    parser.add_argument('--refresh', action='store_true', help="Force refresh of cached data")
    args = parser.parse_args()

    if args.test:
        run_tests()
        return

    if args.ticker:
        tickers = [args.ticker]
    else:
        tickers = load_tickers(args.limit)

    if not tickers:
        return

    run_screener(tickers, refresh=args.refresh)

# ==========================================
# Test Cases & Verification
# ==========================================

def run_tests():
    """
    Runs component tests with sample mock data to verify strategy calculations.
    Includes both Pass and Fail cases for all 10 strategies.
    """
    console.print("[bold cyan]Running 20 Verification Test Cases...[/bold cyan]\n")

    def assert_test(strat_name, is_pass_test, expected, actual_tuple):
        match, reason = actual_tuple
        state = "PASS_CASE" if is_pass_test else "FAIL_CASE"
        if match == expected:
            console.print(f"[green][OK][/green] {strat_name} ({state})")
            return True
        else:
            console.print(f"[red][FAIL][/red] {strat_name} ({state}): Expected {expected}, got {match} ({reason})")
            return False

    def get_base_mock():
        # Setup standard passing 200+ day history DataFrame
        dates = pd.date_range(end='2026-03-19', periods=210)
        df = pd.DataFrame({
            'Open': [100.0] * 210, 'High': [102.0] * 210, 'Low': [98.0] * 210,
            'Close': [100.0] * 210, 'Volume': [1000] * 210
        }, index=dates)
        baseline = df.copy()
        baseline['Close'] = [100.0 - i * 0.05 for i in range(210)] # strictly falling index
        return {'daily': df, 'baseline': baseline, 'info': {}}

    all_passed = True

    # ----------------------------------------
    # 1. Strategy 1: Relative Strength
    # ----------------------------------------
    m1_pass = get_base_mock()
    m1_pass['info'] = {'earningsQuarterlyGrowth': 0.20}
    m1_pass['daily']['High'] = [110.0] * 210 # Near 110.0 max
    
    # Introduce PRICE UP-DAYS and HIGH VOLUME UP-DAYS to pass Volume Accumulation
    idx = m1_pass['daily'].index
    # Alternate step steps to create explicit Price_Chg triggers
    for i in range(190, 210):
        if i % 2 == 0:
             m1_pass['daily'].loc[idx[i], 'Close'] = 106.0  # UP Day
             m1_pass['daily'].loc[idx[i], 'Volume'] = 2500 # High Volume
        else:
             m1_pass['daily'].loc[idx[i], 'Close'] = 104.0  # DOWN Day
             m1_pass['daily'].loc[idx[i], 'Volume'] = 500  # Low Volume

    m1_pass['daily'].loc[idx[-1], 'Close'] = 107.0 # final high beat DMA
    all_passed &= assert_test("Strategy 1", True, True, strategy_1_relative_strength(m1_pass))

    m1_fail = get_base_mock()
    m1_fail['info'] = {'earningsQuarterlyGrowth': 0.05} # Weak EPS growth
    all_passed &= assert_test("Strategy 1", False, False, strategy_1_relative_strength(m1_fail))

    # ----------------------------------------
    # 2. Strategy 2: Deep Value
    # ----------------------------------------
    m2 = {'info': {'trailingPE': 12, 'priceToBook': 1.5, 'debtToEquity': 0.3}}
    m2['financials'] = pd.DataFrame({'2025': [200, 500, 100]}, index=['EBIT', 'Total Revenue', 'Cost Of Revenue'])
    m2['balance_sheet'] = pd.DataFrame({'2025': [1000, 200]}, index=['Total Assets', 'Current Liabilities'])
    all_passed &= assert_test("Strategy 2", True, True, strategy_2_deep_value(m2))

    m2_fail = m2.copy(); m2_fail['info'] = {'trailingPE': 20} # High PE
    all_passed &= assert_test("Strategy 2", False, False, strategy_2_deep_value(m2_fail))

    # ----------------------------------------
    # 3. Strategy 3: Cash Flow Yield
    # ----------------------------------------
    m3 = {'info': {'marketCap': 800, 'debtToEquity': 0.3, 'returnOnEquity': 0.18}}
    m3['cashflow'] = pd.DataFrame({'2025': [60]}, index=['Free Cash Flow']) # 60/800 = 7.5% Yield
    all_passed &= assert_test("Strategy 3", True, True, strategy_3_cash_flow_yield(m3))

    m3_fail = m3.copy(); m3_fail['info'] = {'marketCap': 2000} # FCF Yield will fail (3%)
    all_passed &= assert_test("Strategy 3", False, False, strategy_3_cash_flow_yield(m3_fail))

    # ----------------------------------------
    # 4. Strategy 4: Dollar Earner (Mod)
    # ----------------------------------------
    m4 = {'info': {'debtToEquity': 0.1, 'operatingMargins': 0.25}}
    m4['financials'] = m2['financials'] # reuse EBIT
    m4['balance_sheet'] = m2['balance_sheet'] # reuse Total Assets
    all_passed &= assert_test("Strategy 4", True, True, strategy_4_dollar_earner(m4))

    m4_fail = m4.copy(); m4_fail['info'] = {'debtToEquity': 0.5} # Fail high D/E
    all_passed &= assert_test("Strategy 4", False, False, strategy_4_dollar_earner(m4_fail))

    # ----------------------------------------
    # 5. Strategy 5: Acquirer's Multiple
    # ----------------------------------------
    m5 = {'info': {'enterpriseValue': 700, 'debtToEquity': 0.3}}
    m5['financials'] = pd.DataFrame({'2025': [100]}, index=['EBIT']) # EV/EBIT = 7
    all_passed &= assert_test("Strategy 5", True, True, strategy_5_acquirers_multiple(m5))

    m5_fail = m5.copy(); m5_fail['info'] = {'enterpriseValue': 1000} # Mult = 10
    all_passed &= assert_test("Strategy 5", False, False, strategy_5_acquirers_multiple(m5_fail))

    # ----------------------------------------
    # 6. Strategy 6: Graham Net-Nets
    # ----------------------------------------
    m6 = {'info': {'marketCap': 400, 'priceToBook': 0.5, 'currentRatio': 2.5}}
    m6['balance_sheet'] = pd.DataFrame({'2025': [1000, 500]}, index=['Current Assets', 'Total Liabilities Net Minority Interest'])
    all_passed &= assert_test("Strategy 6", True, True, strategy_6_net_nets(m6))

    m6_fail = m6.copy(); m6_fail['info'] = {'marketCap': 800} # NCAV 500 <= Mcap 800
    all_passed &= assert_test("Strategy 6", False, False, strategy_6_net_nets(m6_fail))

    # ----------------------------------------
    # 7. Strategy 7: Magic Formula
    # ----------------------------------------
    m7 = {'info': {'enterpriseValue': 500}}
    m7['financials'] = pd.DataFrame({'2025': [100]}, index=['EBIT']) # Yield = 20%
    m7['balance_sheet'] = pd.DataFrame({'2025': [600, 100]}, index=['Total Assets', 'Current Liabilities']) # ROCE = 100 / 500 = 20%
    all_passed &= assert_test("Strategy 7", True, True, strategy_7_magic_formula(m7))

    m7_fail = m7.copy(); m7_fail['financials'] = pd.DataFrame({'2025': [30]}, index=['EBIT']) # Yield Fail
    all_passed &= assert_test("Strategy 7", False, False, strategy_7_magic_formula(m7_fail))

    # ----------------------------------------
    # 8. Strategy 8: Hedge Fund Screener
    # ----------------------------------------
    m8 = get_base_mock() # Baseline is dropping in mock base setup
    m8['daily'].loc[m8['daily'].index[-1], 'Close'] = 105.0 # Boost current to be safely above low
    all_passed &= assert_test("Strategy 8", True, True, strategy_8_hedge_fund(m8))

    m8_fail = get_base_mock()
    m8_fail['daily']['Close'] = m8_fail['baseline']['Close'] # Dropping together
    all_passed &= assert_test("Strategy 8", False, False, strategy_8_hedge_fund(m8_fail))

    # ----------------------------------------
    # 9. Strategy 9: Margin Resilience
    # ----------------------------------------
    m9 = {'info': {}}
    m9['quarterly_financials'] = pd.DataFrame({'Q1': [40, 100]}, index=['Gross Profit', 'Total Revenue'])
    m9['financials'] = pd.DataFrame({'2024': [35, 100, 50, 5]}, index=['Gross Profit', 'Total Revenue', 'EBIT', 'Interest Expense'])
    all_passed &= assert_test("Strategy 9", True, True, strategy_9_margin_resilience(m9))

    m9_fail = m9.copy(); m9_fail['quarterly_financials'] = pd.DataFrame({'Q1': [20, 100]}, index=['Gross Profit', 'Total Revenue']) # Drop Margin
    all_passed &= assert_test("Strategy 9", False, False, strategy_9_margin_resilience(m9_fail))

    # ----------------------------------------
    # 10. Strategy 10: Dividend Fortress
    # ----------------------------------------
    m10_dates = pd.date_range(end='2026-03-19', periods=1500)
    m10_df = pd.DataFrame({'Dividends': [0.0]*1500}, index=m10_dates)
    m10_df.loc[m10_df.index[10], 'Dividends'] = 2.0  # past year
    m10_df.loc[m10_df.index[-10], 'Dividends'] = 5.0 # current year
    
    m10 = {
        'info': {'dividendYield': 0.05, 'payoutRatio': 0.50},
        'daily': m10_df
    }
    all_passed &= assert_test("Strategy 10", True, True, strategy_10_dividend_fortress(m10))

    m10_fail = {'info': {'dividendYield': 0.03}}
    all_passed &= assert_test("Strategy 10", False, False, strategy_10_dividend_fortress(m10_fail))

    if all_passed:
        console.print("\n[bold green]All algorithmic Pass/Fail unit tests passed![/bold green]")
    else:
        console.print("\n[bold red]Some unit tests failed. Please review assertions above.[/bold red]")

if __name__ == "__main__":
    main()
