# -------------------------------------------------------------------------------
# Project: Stock Analysis (https://github.com/anshulsolanki/stock_analysis)
# Author:  Anshul Solanki
# License: MIT License
# -------------------------------------------------------------------------------

"""
Technical Bear Market Stock Screener

This script implements 6 strategies for identifying strong stocks in a bear market,
combining technical resilience and momentum.
Based on the methodologies described in technical_screening_bearmarket.pdf.

Usage:
------
python3 bear_market_technical_screener.py [--limit N] [--ticker TICKER] [--test]
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
CACHE_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data_cache')
JSON_PATH = os.path.join(DATA_DIR, 'nifty_500.json')
NSE_BASELINE = "NSEI_baseline.csv"
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR_BASE = os.path.join(os.path.dirname(BASE_DIR), 'screener_results', 'bear_market')
OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, TIMESTAMP)

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
    
    render_pdf_standard_header(fig, title_text="Technical Bear Market Screener Report")
    
    PRIMARY_COLOR = '#1e293b'
    ACCENT_COLOR = '#2563eb'
    SECONDARY_COLOR = '#475569'
    
    fig.text(0.5, 0.65, "Technical Bear Market Screener", ha='center', va='center', fontsize=32, weight='bold', color=ACCENT_COLOR)
    fig.text(0.5, 0.58, "Price & Volume Analysis", ha='center', va='center', fontsize=18, weight='bold', color=SECONDARY_COLOR)
    fig.text(0.5, 0.52, f"Strategic Market Scan: {timestamp}", ha='center', va='center', fontsize=14, color=SECONDARY_COLOR)
    
    fig.text(0.15, 0.35, "Screening Pillars", fontsize=16, weight='bold', color=PRIMARY_COLOR)
    pillars = [
        "● Price & Volume Resilience",
        "● Momentum Reversals",
        "● Extreme Exhaustion"
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
# Technical Indicators Helpers
# ==========================================

def calculate_rsi(series, period=14):
    """
    Calculate Relative Strength Index (RSI) using Wilders smoothing.
    
    Args:
        series (pd.Series): Daily Close prices.
        period (int): Period for RSI calculation (default 14).
        
    Returns:
        pd.Series: RSI values.
    """
    if len(series) < period + 1:
        return pd.Series([np.nan] * len(series), index=series.index)
    
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(series, period=20, std=2):
    """
    Calculate Bollinger Bands (Upper, Lower).
    
    Args:
        series (pd.Series): Daily Close prices.
        period (int): Period for SMA (default 20).
        std (int): Standard deviation multiplier (default 2).
        
    Returns:
        tuple: (Upper Band, Lower Band) pd.Series.
    """
    if len(series) < period:
        return pd.Series([np.nan] * len(series), index=series.index), pd.Series([np.nan] * len(series), index=series.index)
    
    sma = series.rolling(window=period).mean()
    rstd = series.rolling(window=period).std()
    
    upper_band = sma + std * rstd
    lower_band = sma - std * rstd
    
    return upper_band, lower_band

def calculate_mansfield_rs(stock_series, index_series, period=50):
    """
    Calculate Mansfield Relative Strength.
    
    Args:
        stock_series (pd.Series): Daily Close prices of the stock.
        index_series (pd.Series): Daily Close prices of the index (baseline).
        period (int): Period for SMA of RS (default 50).
        
    Returns:
        pd.Series: Mansfield RS values.
    """
    # Align series
    common_index = stock_series.index.intersection(index_series.index)
    stock = stock_series.loc[common_index]
    index = index_series.loc[common_index]
    
    if len(stock) < period:
        return pd.Series([np.nan] * len(stock), index=common_index)
    
    rs = stock / index
    sma_rs = rs.rolling(window=period).mean()
    
    mansfield_rs = (rs / sma_rs) - 1
    return mansfield_rs

# ==========================================
# Strategy Implementations
# ==========================================

def strategy_11_oversold_reversal(data):
    """
    11. Oversold Reversal
    
    Identifies stocks that are extremely oversold and might be due for a bounce.
    
    Criteria:
    - RSI (14) < 30 (Traditional oversold level)
    - Price < Lower Bollinger Band (Price extended below volatility bands)
    - Distance from 20 DMA > 10% (Significant deviation from short-term mean)
    
    Args:
        data (dict): Dictionary containing 'daily' data DataFrame.
        
    Returns:
        tuple: (bool, str) Match status and reason.
    """
    df = data.get('daily')
    if df is None or df.empty or 'Close' not in df.columns:
        return False, "Missing daily data"
    
    close = df['Close']
    if len(close) < 20:
        return False, "Not enough data"
    
    rsi = calculate_rsi(close).iloc[-1]
    upper, lower = calculate_bollinger_bands(close)
    lower_bb = lower.iloc[-1]
    
    sma20 = close.rolling(window=20).mean().iloc[-1]
    
    curr_price = close.iloc[-1]
    
    if rsi is None or np.isnan(rsi) or lower_bb is None or np.isnan(lower_bb) or sma20 is None or np.isnan(sma20):
        return False, "Incomplete data"
        
    if rsi >= 30: return False, f"RSI too high: {rsi:.1f}"
    if curr_price >= lower_bb: return False, f"Price above lower BB"
    
    if curr_price >= 0.9 * sma20: return False, f"Too close to 20DMA: {(curr_price/sma20-1)*100:.1f}%"
    
    return True, f"Match (RSI: {rsi:.1f}, Dist 20DMA: {(curr_price/sma20-1)*100:.1f}%)"

def strategy_12_bb_exhaustion(data):
    """
    12. Bollinger Band Exhaustion
    
    Identifies stocks extended below volatility bands but still above a long-term trendline (200DMA),
    suggesting a pullback in an uptrend rather than a structural breakdown.
    
    Criteria:
    - Price < Lower Bollinger Band
    - RSI < 35
    - Price > 200 DMA
    
    Args:
        data (dict): Dictionary containing 'daily' data DataFrame.
        
    Returns:
        tuple: (bool, str) Match status and reason.
    """
    df = data.get('daily')
    if df is None or df.empty or 'Close' not in df.columns:
        return False, "Missing daily data"
    
    close = df['Close']
    if len(close) < 200:
         return False, "Not enough data for 200DMA"
    
    rsi = calculate_rsi(close).iloc[-1]
    upper, lower = calculate_bollinger_bands(close)
    lower_bb = lower.iloc[-1]
    
    sma200 = close.rolling(window=200).mean().iloc[-1]
    
    curr_price = close.iloc[-1]
    
    if rsi is None or np.isnan(rsi) or lower_bb is None or np.isnan(lower_bb) or sma200 is None or np.isnan(sma200):
        return False, "Incomplete data"
        
    if curr_price >= lower_bb: return False, f"Price above lower BB"
    if rsi >= 35: return False, f"RSI too high: {rsi:.1f}"
    if curr_price <= sma200: return False, "Price below 200DMA"
    
    return True, f"Match (RSI: {rsi:.1f}, Above 200DMA: {(curr_price/sma200-1)*100:.1f}%)"

def strategy_13_beaten_down_quality(data):
    """
    13. Beaten Down Quality
    
    Identifies high-quality stocks (strong ROCE, consistent growth) that are trading at a discount
    to their historical valuation and are significantly off their highs.
    
    Criteria:
    - Current P/E < 0.7 * 5Yr Average P/E (Historical discount)
    - ROCE > 22% (High capital efficiency)
    - Profit growth 5Years > 15% (Consistent earnings growth)
    - Distance from 52-week high > 30% (Significant drawdown)
    
    Args:
        data (dict): Dictionary containing 'info', 'financials', 'daily', and 'weekly' data.
        
    Returns:
        tuple: (bool, str) Match status and reason.
    """
    info = data.get('info', {})
    fin = data.get('financials')
    df = data.get('daily')
    
    if df is None or df.empty or 'Close' not in df.columns:
        return False, "Missing daily data"
    
    curr_price = df['Close'].iloc[-1]
    
    # 1. Distance from 52-week high > 30%
    high_52w = info.get('fiftyTwoWeekHigh')
    if high_52w is None:
        if len(df) >= 252:
            high_52w = df['High'].tail(252).max()
        else:
            return False, "Incomplete 52W High data"
    
    if high_52w > 0 and curr_price >= 0.7 * high_52w:
        return False, f"Too close to 52W High: {(curr_price/high_52w-1)*100:.1f}%"

    # 2. Current PE
    pe = info.get('trailingPE')
    if pe is None:
         try:
              if fin is not None and not fin.empty and 'Diluted EPS' in fin.index:
                   eps = fin.loc['Diluted EPS'].iloc[0]
                   if eps > 0:
                        pe = curr_price / eps
         except Exception: pass
         
    if pe is None: return False, "Missing Current PE"

    # 3. 5-Year Average PE
    avg_pe = None
    if fin is not None and not fin.empty and 'Diluted EPS' in fin.index:
         eps_series = fin.loc['Diluted EPS']
         eps_series.index = pd.to_datetime(eps_series.index)
         pes = []
         for date, eps in eps_series.items():
              try:
                   year = date.year
                   wk_df = data.get('weekly')
                   if wk_df is not None and not wk_df.empty:
                        year_df = wk_df[wk_df.index.year == year]
                        if not year_df.empty:
                             avg_price = year_df['Close'].mean()
                             if eps > 0:
                                  pes.append(avg_price / eps)
              except Exception: pass
         
         if len(pes) >= 2:
              avg_pe = sum(pes) / len(pes)
    
    if avg_pe is None:
         return False, "Missing Historical PE"
    
    if pe >= 0.7 * avg_pe:
         return False, f"PE not cheap enough: {pe:.1f} vs 0.7*Avg({avg_pe:.1f})"

    # 4. ROCE > 22%
    bs = data.get('balance_sheet')
    roce_val = None
    if fin is not None and not fin.empty and bs is not None and not bs.empty:
         try:
              ebit = get_latest_financial_value(fin, 'EBIT')
              assets = get_latest_financial_value(bs, 'Total Assets')
              cur_liab = get_latest_financial_value(bs, 'Total Current Liabilities')
              if assets is not None and cur_liab is not None and assets - cur_liab > 0:
                   roce_val = ebit / (assets - cur_liab)
         except Exception: pass
    
    if roce_val is None:
         return False, "Missing ROCE data"
    
    if roce_val < 0.22:
         return False, f"ROCE too low: {roce_val*100:.1f}%"

    # 5. Profit growth 5Years > 15%
    profit_growth = None
    if fin is not None and not fin.empty and 'Net Income' in fin.index:
         try:
              net_income = fin.loc['Net Income'].dropna()
              if len(net_income) >= 2:
                   start_val = net_income.iloc[-1]
                   curr_val = net_income.iloc[0]
                   years = len(net_income) - 1
                   if start_val > 0 and curr_val > 0:
                        profit_growth = (curr_val / start_val) ** (1/years) - 1
         except Exception: pass

    if profit_growth is None:
         return False, "Missing Profit Growth data"
    
    if profit_growth < 0.15:
         return False, f"Profit Growth too low: {profit_growth*100:.1f}%"

    return True, f"Match (PE: {pe:.1f}, ROCE: {roce_val*100:.1f}%, Growth: {profit_growth*100:.1f}%)"

def strategy_14_mean_reversion_rubber_band(data):
    """
    14. Mean Reversion (Rubber Band)
    
    Identifies extreme panic selling where the stock is extended far below its long-term average,
    creating a potential "rubber band" snap-back effect.
    
    Criteria:
    - RSI (14) < 25 (Extreme panic)
    - Price < Lower Bollinger Band (20, 2)
    - Distance from 200-day SMA < -20% (Far below long-term mean)
    
    Args:
        data (dict): Dictionary containing 'daily' data DataFrame.
        
    Returns:
        tuple: (bool, str) Match status and reason.
    """
    df = data.get('daily')
    if df is None or df.empty or 'Close' not in df.columns:
        return False, "Missing daily data"
    
    close = df['Close']
    if len(close) < 200:
         return False, "Not enough data for 200DMA"
    
    rsi = calculate_rsi(close).iloc[-1]
    upper, lower = calculate_bollinger_bands(close)
    lower_bb = lower.iloc[-1]
    
    sma200 = close.rolling(window=200).mean().iloc[-1]
    
    curr_price = close.iloc[-1]
    
    if rsi is None or np.isnan(rsi) or lower_bb is None or np.isnan(lower_bb) or sma200 is None or np.isnan(sma200):
        return False, "Incomplete data"
        
    if rsi >= 25: return False, f"RSI too high: {rsi:.1f}"
    if curr_price >= lower_bb: return False, f"Price above lower BB"
    
    if curr_price >= 0.8 * sma200: return False, f"Too close to 200DMA: {(curr_price/sma200-1)*100:.1f}%"
    
    return True, f"Match (RSI: {rsi:.1f}, Dist 200DMA: {(curr_price/sma200-1)*100:.1f}%)"

def strategy_15_relative_strength_new_leaders(data):
    """
    15. Relative Strength (New Leaders)
    
    Identifies stocks that are showing positive absolute and relative momentum even in a bear market,
    signaling potential future leadership.
    
    Criteria:
    - Stock 3-Month Return > 0% (Positive absolute momentum)
    - Mansfield Relative Strength > 0 (Outperforming the index)
    - Price > 200-day SMA (Long-term uptrend intact)
    
    Args:
        data (dict): Dictionary containing 'daily' and 'baseline' data DataFrames.
        
    Returns:
        tuple: (bool, str) Match status and reason.
    """
    df = data.get('daily')
    base_df = data.get('baseline')
    
    if df is None or df.empty or 'Close' not in df.columns:
        return False, "Missing daily data"
    if base_df is None or base_df.empty or 'Close' not in base_df.columns:
        return False, "Missing baseline data"
        
    close = df['Close']
    if len(close) < 200:
         return False, "Not enough data for 200DMA"
    
    common_dates = df.index.intersection(base_df.index)
    aligned_stock = df.loc[common_dates, 'Close']
    aligned_base = base_df.loc[common_dates, 'Close']
    
    if len(aligned_stock) < 200:
         return False, "Not enough common data points"
         
    sma200 = aligned_stock.rolling(window=200).mean().iloc[-1]
    curr_price = aligned_stock.iloc[-1]
    if curr_price <= sma200: return False, "Price below 200DMA"

    if len(aligned_stock) >= 63:
         ret_3m = aligned_stock.iloc[-1] / aligned_stock.iloc[-63] - 1
         if ret_3m <= 0: return False, f"Stock 3M return negative: {ret_3m*100:.1f}%"
    else:
         return False, "Not enough data for 3M return"

    mansfield_rs = calculate_mansfield_rs(aligned_stock, aligned_base).iloc[-1]
    if mansfield_rs is None or np.isnan(mansfield_rs):
         return False, "Error calculating Mansfield RS"
         
    if mansfield_rs <= 0: return False, f"Mansfield RS negative: {mansfield_rs:.3f}"

    return True, f"Match (3M Ret: {ret_3m*100:.1f}%, Mansfield: {mansfield_rs:.3f})"

def strategy_16_capitulation_volume_washout(data):
    """
    16. Capitulation (Volume Washout)
    
    Identifies a potential bottom where selling pressure reaches a climax (high volume at 52W low)
    followed by a reversal (bullish candle).
    
    Criteria:
    - Current Price ≈ 52-Week Low (Within 2%)
    - Volume > 3 * Average Volume (10-day) (Volume climax)
    - Closing Price > Opening Price (Bullish reversal candle)
    
    Args:
        data (dict): Dictionary containing 'daily' data DataFrame.
        
    Returns:
        tuple: (bool, str) Match status and reason.
    """
    df = data.get('daily')
    if df is None or df.empty or 'Close' not in df.columns or 'Volume' not in df.columns or 'Open' not in df.columns:
        return False, "Missing daily data"
    
    close = df['Close']
    open_price = df['Open']
    volume = df['Volume']
    
    if len(close) < 252:
         return False, "Not enough data for 52W Low"
         
    curr_price = close.iloc[-1]
    curr_open = open_price.iloc[-1]
    curr_vol = volume.iloc[-1]
    
    low_52w = close.tail(252).min()
    if curr_price > low_52w * 1.02: return False, f"Not at 52W Low: {curr_price:.1f} vs {low_52w:.1f}"

    avg_vol_10d = volume.tail(11).head(10).mean()
    if curr_vol <= 3 * avg_vol_10d: return False, f"Volume spike not big enough: {curr_vol/avg_vol_10d:.1f}x"

    if curr_price <= curr_open: return False, "Not a bullish candle (Close <= Open)"

    return True, f"Match (at 52W Low: {curr_price:.1f}, Vol Spike: {curr_vol/avg_vol_10d:.1f}x)"

# ==========================================
# Main Execution Flow
# ==========================================

def run_screener(tickers, refresh=False, output_dir=None):
    if output_dir is None:
        output_dir = OUTPUT_DIR

    strategies = [
        ("Oversold Reversal", strategy_11_oversold_reversal),
        ("BB Exhaustion", strategy_12_bb_exhaustion),
        ("Beaten Down Quality", strategy_13_beaten_down_quality),
        ("Mean Reversion (Rubber Band)", strategy_14_mean_reversion_rubber_band),
        ("Relative Strength (New)", strategy_15_relative_strength_new_leaders),
        ("Capitulation", strategy_16_capitulation_volume_washout)
    ]

    results = []
    analyzed_count = 0

    console.print(f"[bold green]Scanning {len(tickers)} tickers...[/bold green]")
    
    for ticker in tickers:
        data = load_data(ticker, refresh=refresh)
        if not data.get('info') and data.get('daily').empty:
            continue

        analyzed_count += 1

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

    console.print(f"\n[bold]Total stocks analyzed:[/bold] {analyzed_count}")
    console.print(f"[bold]Stocks passed one or more strategies:[/bold] {len(results)}")

    if results:
        df = pd.DataFrame(results)
        cols = ['Ticker'] + [s[0] for s in strategies]
        df = df[cols]
        
        strategy_names = [s[0] for s in strategies]
        df['Score'] = (df[strategy_names] == 'PASSED').sum(axis=1)
        
        df_all = df.sort_values(by='Score', ascending=False).copy()
        df_all = df_all[['Ticker', 'Score'] + strategy_names]
        
        df_top50 = df_all.head(50).copy()
        
        console.print("\n[bold]--- SCREEEN RESULTS SUMMARY (Top 50) ---[/bold]")
        print(df_top50.to_string(index=False))
        
        # Save Top 50 to CSV (or All to CSV, usually All is better for analysis)
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "bear_market_technical_screen_results.csv")
        df_all.to_csv(out_path, index=False)
        console.print(f"\n[bold]Results saved to {out_path}[/bold]")

        # --- Generate PDF Report ---
        temp_title = os.path.join(output_dir, f"temp_title.pdf")
        temp_table = os.path.join(output_dir, f"temp_table.pdf")
        pdf_path = os.path.join(output_dir, f"Bear_Market_Technical_Screener_Results.pdf")
        
        try:
            from pypdf import PdfWriter
            
            with PdfPages(temp_title) as pdf_t:
                create_pdf_title_page(pdf_t, TIMESTAMP)

            with PdfPages(temp_table) as pdf_tab:
                render_pdf_styled_table(pdf_tab, df_all, "Technical Strategy Matches")
            
            merger = PdfWriter()
            merger.append(temp_title)
            
            # Use technical_screening_bearmarket.pdf for intermediate pages if exists
            strategies_pdf = os.path.join(BASE_DIR, "technical_screening_bearmarket.pdf")
            if os.path.exists(strategies_pdf):
                 merger.append(strategies_pdf)
            else:
                 console.print(f"[yellow]Warning: {strategies_pdf} not found to merge.[/yellow]")
                 
            merger.append(temp_table)
            
            merger.write(pdf_path)
            merger.close()
            
            if os.path.exists(temp_title): os.remove(temp_title)
            if os.path.exists(temp_table): os.remove(temp_table)
                 
            console.print(f"[bold green]PDF Report saved to {pdf_path}[/bold green]")
        except Exception as e:
            console.print(f"[red]Error saving PDF Report: {e}[/red]")
    else:
        console.print("\n[yellow]No strategy matches found.[/yellow]")

def main():

    parser = argparse.ArgumentParser(description="Technical Bear Market Stock Screener")
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
    """
    console.print("[bold cyan]Running Technical Verification Test Cases...[/bold cyan]\n")

    def assert_test(strat_name, is_pass_test, expected, actual_tuple):
        match, reason = actual_tuple
        state = "PASS_CASE" if is_pass_test else "FAIL_CASE"
        if match == expected:
            console.print(f"[green][OK][/green] {strat_name} ({state})")
            return True
        else:
            console.print(f"[red][FAIL][/red] {strat_name} ({state}): Expected {expected}, got {match} ({reason})")
            return False

    def get_tech_base_mock(periods=300):
        dates = pd.date_range(end='2026-03-19', periods=periods)
        df = pd.DataFrame({
            'Open': [100.0] * periods, 'High': [102.0] * periods, 'Low': [98.0] * periods,
            'Close': [100.0] * periods, 'Volume': [1000] * periods
        }, index=dates)
        baseline = df.copy()
        baseline['Close'] = [100.0 - i * 0.1 for i in range(periods)] # strictly falling index
        return {'daily': df, 'baseline': baseline, 'info': {}}

    all_passed = True

    # ----------------------------------------
    # 11. Strategy 11: Oversold Reversal
    # ----------------------------------------
    m11_pass = get_tech_base_mock(252)
    close = [100.0] * 251 + [50.0]
    m11_pass['daily']['Close'] = close
    m11_pass['daily']['Low'] = [c - 2 for c in close]
    m11_pass['daily']['High'] = [c + 2 for c in close]
    all_passed &= assert_test("Strategy 11", True, True, strategy_11_oversold_reversal(m11_pass))

    m11_fail = get_tech_base_mock(252)
    m11_fail['daily']['Close'] = [100.0] * 252
    all_passed &= assert_test("Strategy 11", False, False, strategy_11_oversold_reversal(m11_fail))

    # ----------------------------------------
    # 12. Strategy 12: Bollinger Band Exhaustion
    # ----------------------------------------
    m12_pass = get_tech_base_mock(252)
    close = [50.0] * 200 + [150.0] * 51 + [80.0]
    m12_pass['daily']['Close'] = close
    m12_pass['daily']['Low'] = [c - 2 for c in close]
    m12_pass['daily']['High'] = [c + 2 for c in close]
    all_passed &= assert_test("Strategy 12", True, True, strategy_12_bb_exhaustion(m12_pass))

    m12_fail = get_tech_base_mock(252)
    m12_fail['daily']['Close'] = [100.0] * 252
    all_passed &= assert_test("Strategy 12", False, False, strategy_12_bb_exhaustion(m12_fail))

    # ----------------------------------------
    # 13. Strategy 13: Beaten Down Quality
    # ----------------------------------------
    m13_pass = get_tech_base_mock(252)
    m13_pass['daily']['High'] = [200.0] * 100 + [100.0] * 152
    m13_pass['daily']['Close'] = [100.0] * 252
    m13_pass['daily']['Low'] = [98.0] * 252
    m13_pass['info']['fiftyTwoWeekHigh'] = 200.0
    m13_pass['info']['trailingPE'] = 10.0
    
    m13_pass['financials'] = pd.DataFrame({
        '2025-03-31': [10.0, 1000.0, 2000.0, 100.0, 100.0],
        '2024-03-31': [8.0, 800.0, 1800.0, 100.0, 80.0],
        '2023-03-31': [6.0, 600.0, 1600.0, 100.0, 60.0],
        '2022-03-31': [5.0, 500.0, 1500.0, 100.0, 50.0],
        '2021-03-31': [4.0, 400.0, 1400.0, 100.0, 40.0]
    }, index=['Diluted EPS', 'EBIT', 'Total Assets', 'Total Current Liabilities', 'Net Income'])
    m13_pass['balance_sheet'] = m13_pass['financials'].copy()
    dates_wk = pd.date_range(end='2026-03-19', periods=260, freq='W')
    m13_pass['weekly'] = pd.DataFrame({
        'Close': [200.0] * 260
    }, index=dates_wk)
    
    all_passed &= assert_test("Strategy 13", True, True, strategy_13_beaten_down_quality(m13_pass))

    m13_fail = get_tech_base_mock(252)
    m13_fail['info']['trailingPE'] = 20.0
    all_passed &= assert_test("Strategy 13", False, False, strategy_13_beaten_down_quality(m13_fail))

    # ----------------------------------------
    # 14. Strategy 14: Mean Reversion (Rubber Band)
    # ----------------------------------------
    m14_pass = get_tech_base_mock(252)
    close = [100.0] * 251 + [70.0]
    m14_pass['daily']['Close'] = close
    m14_pass['daily']['Low'] = [c - 2 for c in close]
    m14_pass['daily']['High'] = [c + 2 for c in close]
    all_passed &= assert_test("Strategy 14", True, True, strategy_14_mean_reversion_rubber_band(m14_pass))

    m14_fail = get_tech_base_mock(252)
    m14_fail['daily']['Close'] = [100.0] * 252
    all_passed &= assert_test("Strategy 14", False, False, strategy_14_mean_reversion_rubber_band(m14_fail))

    # ----------------------------------------
    # 15. Strategy 15: Relative Strength (New Leaders)
    # ----------------------------------------
    m15_pass = get_tech_base_mock(252)
    close = [50.0] * 200 + [100.0] * 52
    m15_pass['daily']['Close'] = close
    m15_pass['daily']['Low'] = [c - 2 for c in close]
    m15_pass['daily']['High'] = [c + 2 for c in close]
    m15_pass['baseline'] = get_tech_base_mock(252)['baseline']
    m15_pass['baseline']['Close'] = [200.0 - i * 0.5 for i in range(252)]
    all_passed &= assert_test("Strategy 15", True, True, strategy_15_relative_strength_new_leaders(m15_pass))

    m15_fail = get_tech_base_mock(252)
    m15_fail['daily']['Close'] = [100.0 - i * 0.1 for i in range(252)]
    all_passed &= assert_test("Strategy 15", False, False, strategy_15_relative_strength_new_leaders(m15_fail))

    # ----------------------------------------
    # 16. Strategy 16: Capitulation (Volume Washout)
    # ----------------------------------------
    m16_pass = get_tech_base_mock(252)
    close = [100.0] * 251 + [50.0]
    m16_pass['daily']['Close'] = close
    m16_pass['daily']['Open'] = [100.0] * 251 + [48.0]
    m16_pass['daily']['Low'] = [98.0] * 251 + [47.0]
    m16_pass['daily']['High'] = [102.0] * 251 + [52.0]
    m16_pass['daily']['Volume'] = [1000] * 251 + [4000]
    all_passed &= assert_test("Strategy 16", True, True, strategy_16_capitulation_volume_washout(m16_pass))

    m16_fail = get_tech_base_mock(252)
    m16_fail['daily']['Volume'] = [1000] * 252
    all_passed &= assert_test("Strategy 16", False, False, strategy_16_capitulation_volume_washout(m16_fail))

    if all_passed:
        console.print("\n[bold green]All technical unit tests passed![/bold green]")
    else:
        console.print("\n[bold red]Some unit tests failed. Please review assertions above.[/bold red]")

if __name__ == "__main__":
    main()
