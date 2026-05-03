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
Mark Minervini Long-Term Breakout Screener

Identifies stocks forming wide, multi-month VCP bases suitable for position trades
(holding period: 2-6 months). This screener builds on the same Minervini Stage 2
framework as the ST screener but relaxes key parameters and adds institutional-quality
filters to favour stocks with durable, fundamental-backed breakouts.

Key Differences vs ST Screener (minervini_st_breakouts.py):
----------------------------------------------------------
  Parameter              | ST Value   | LT Value   | Rationale
  -----------------------|------------|------------|-----------------------------------
  VCP window             | 130 days   | 200 days   | Deeper, wider bases
  Peak detection dist.   | 10 days    | 15 days    | Fewer false peaks on longer bases
  Final tightness        | <= 6%      | <= 15%     | Allow wider consolidations
  Base length            | 15-130 d   | 30-200 d   | Multi-month bases
  Peak variance (flat)   | 10%        | 15%        | More tolerance for wider patterns
  Volume at breakout     | 1.3x       | 1.0x       | Early accumulation often quiet
  Target                 | 25%        | 50%        | Multi-month holding horizon
  Min pullbacks          | 2          | 1          | First-stage bases may be simpler
  Min price > 52w low    | 30%        | 20%        | Catch stocks earlier in Stage 2
  Max dist. from 52w high| 25%        | 35%        | Allow deeper bases
  Min RS Rating          | 70         | 60         | Slightly relaxed RS threshold

Additional LT-Only Features:
----------------------------
  1. Weekly Timeframe Confirmation: Weekly close > 10w & 40w SMAs, plus weekly
     RS line (stock/Nifty 50) near 10-week highs.
  2. Sector/Industry RS: Ranks sectors by 6-month return, rejects bottom 40%.
  3. OBV Accumulation Score: Measures On-Balance Volume slope during the base
     period to detect institutional accumulation.
  4. Fundamental Quality Score (0-100): Weighted composite of EPS growth, sales
     growth, RS rating, and profit margin. Used to rank candidates.

Usage:
------
python minervini_lt_breakouts.py [--limit N] [--sample N] [--refresh] [--end-date DD-MM-YYYY]
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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import shared utilities from ST screener
from minervini_st_breakouts import (
    Console, console, load_tickers, fetch_data, calculate_rs_ratings,
    render_pdf_standard_header,
    generate_chart as _st_generate_chart
)

# Configuration
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')
TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = os.path.join(BASE_DIR, 'screener_results', 'minervini_lt_breakouts', TIMESTAMP)
CHARTS_DIR = os.path.join(OUTPUT_DIR, 'charts')
CACHE_DIR = os.path.join(DATA_DIR, 'data_cache')
JSON_PATH = os.path.join(DATA_DIR, 'nifty_500.json')
PDF_PATH = os.path.join(OUTPUT_DIR, f"minervini_LT_Results_{TIMESTAMP}.pdf")

# ============================================================================
# LT Parameters (relaxed vs ST — see module docstring for comparison table)
# ============================================================================
MIN_EPS_GROWTH = 0.15       # Min quarterly EPS growth (ST: 0.20)
MIN_SALES_GROWTH = 0.10     # Min quarterly sales growth (ST: 0.15)
MIN_RS_RATING = 60          # Min IBD-style RS rating (ST: 70)
MIN_PRICE_TO_LOW = 1.20     # Must be ≥20% above 52-week low (ST: 30%)
MAX_PRICE_TO_HIGH = 0.65    # Must be within 35% of 52-week high (ST: 25%)
VCP_WINDOW = 200            # Scan last 200 trading days for VCP (ST: 130)
PEAK_DISTANCE = 15          # Min distance between detected peaks (ST: 10)
MAX_FINAL_TIGHTNESS = 0.15  # Last pullback must be ≤15% (ST: 6%)
MIN_BASE_LENGTH = 30        # Min base length in trading days (ST: 15)
MAX_BASE_LENGTH = 200       # Max base length in trading days (ST: 130)
PEAK_VARIANCE_LIMIT = 0.15  # Max peak-to-peak variance for flat top (ST: 0.10)
VOLUME_BREAKOUT_MULT = 1.0  # Breakout vol ≥ 1.0× avg vol (ST: 1.3×)
TARGET_MULT = 1.50          # Target = pivot × 1.50 i.e. 50% gain (ST: 25%)

def setup_directories():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

def generate_chart(df, ticker, result, pdf=None, end_date=None):
    """Generate candlestick + volume chart for an LT breakout candidate.
    
    Unlike the ST version, this shows 200 days of history (vs 150) and includes
    LT-specific metadata (Quality Score, OBV, Sector) in the chart title.
    Charts are saved to the LT output directory (screener_results/minervini_lt_breakouts/).
    """
    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
        if df.index.tz is not None:
            end_ts = end_ts.tz_localize(df.index.tz)
        df = df[df.index <= end_ts].copy()
    if 'SMA50' not in df.columns:
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        df['SMA150'] = df['Close'].rolling(window=150).mean()
        df['SMA200'] = df['Close'].rolling(window=200).mean()
        df['VolSMA50'] = df['Volume'].rolling(window=50).mean()
    plot_df = df.tail(200)  # Show more history for LT
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    width, width2 = 0.6, 0.05
    up = plot_df[plot_df.Close >= plot_df.Open]
    down = plot_df[plot_df.Close < plot_df.Open]
    ax1.bar(up.index, up.Close - up.Open, width, bottom=up.Open, color='green', alpha=0.8)
    ax1.bar(up.index, up.High - up.Close, width2, bottom=up.Close, color='green')
    ax1.bar(up.index, up.Low - up.Open, width2, bottom=up.Open, color='green')
    ax1.bar(down.index, down.Close - down.Open, width, bottom=down.Open, color='red', alpha=0.8)
    ax1.bar(down.index, down.High - down.Open, width2, bottom=down.Open, color='red')
    ax1.bar(down.index, down.Low - down.Close, width2, bottom=down.Close, color='red')
    ax1.plot(plot_df.index, plot_df['SMA50'], color='blue', label='SMA 50')
    ax1.plot(plot_df.index, plot_df['SMA150'], color='purple', label='SMA 150')
    ax1.plot(plot_df.index, plot_df['SMA200'], color='black', label='SMA 200')
    ax1.axhline(result['Pivot'], color='magenta', linestyle='--', linewidth=1.5, label=f"Pivot ({result['Pivot']})")
    ax1.axhline(result['Target'], color='green', linestyle=':', linewidth=1.5, label=f"Target ({result['Target']})")
    ax1.axhline(result['Stop_Loss'], color='red', linestyle=':', linewidth=1.5, label=f"Stop ({result['Stop_Loss']})")
    title = (f"{ticker} - Minervini LT | Pullbacks: {str([float(v) for v in result['Pullbacks']])}%\n"
             f"Q:{result.get('Quality_Score','')}, OBV:{result.get('OBV_Accum','')}, "
             f"Sector:{result.get('Sector','')}, RS:{result['RS_Rating']}")
    ax1.set_title(title, fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    colors = ['green' if c >= o else 'red' for c, o in zip(plot_df.Close, plot_df.Open)]
    ax2.bar(plot_df.index, plot_df.Volume, color=colors, width=0.6, alpha=0.6)
    ax2.plot(plot_df.index, plot_df['VolSMA50'], color='orange', label='Vol SMA 50')
    ax2.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    if pdf:
        pdf.savefig(fig)
    filename = f"{ticker.replace('.NS', '')}_lt_breakout.png"
    save_path = os.path.join(CHARTS_DIR, filename)
    plt.savefig(save_path)
    plt.close(fig)
    return save_path

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
        
        # Use a smaller font size for LT tables since they have more columns (16 cols)
        font_size = 6.5
        table.set_fontsize(font_size) 
        table.scale(1.0, 1.8) 
        
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white', fontsize=font_size)
                cell.set_facecolor('#1e293b')
            else:
                cell.set_linewidth(0.3)
                cell.set_facecolor('#f8fafc' if row % 2 == 0 else 'white')
                if col == 0:
                     cell.set_text_props(weight='bold', color='#2563eb')
        pdf.savefig(fig)
        plt.close(fig)

def calculate_sector_rs(tickers, end_date=None, preloaded_data=None):
    """Calculate sector-level relative strength rankings.
    
    Groups all tickers by their sector (from cached yfinance info), computes the
    average 6-month (126-day) return for each sector, then ranks sectors by
    percentile (0-100). Stocks in the bottom 40% sectors are rejected by check_criteria.
    
    Returns:
        tuple: (ticker_sectors, sector_ranks)
            - ticker_sectors: dict mapping ticker → sector name
            - sector_ranks: dict mapping sector name → percentile rank (0-100)
    """
    sector_map = {}
    ticker_sectors = {}
    for ticker in tickers:
        info_path = os.path.join(CACHE_DIR, f"{ticker}_info.json")
        if os.path.exists(info_path):
            try:
                with open(info_path, 'r') as f:
                    info = json.load(f)
                sector = info.get('sector', 'Unknown')
                if sector:
                    ticker_sectors[ticker] = sector
                    if sector not in sector_map:
                        sector_map[sector] = []
                    sector_map[sector].append(ticker)
            except Exception:
                pass

    sector_returns = {}
    for sector, stickers in sector_map.items():
        returns = []
        for ticker in stickers:
            try:
                if preloaded_data and ticker in preloaded_data:
                    df = preloaded_data[ticker]['history'].copy()
                else:
                    cache_path = os.path.join(CACHE_DIR, f"{ticker}_1d.csv")
                    if not os.path.exists(cache_path):
                        continue
                    df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                if end_date:
                    end_ts = pd.Timestamp(end_date)
                    if df.index.tz is not None:
                        end_ts = end_ts.tz_localize(df.index.tz)
                    df = df[df.index <= end_ts]
                if len(df) >= 126:
                    ret = (df['Close'].iloc[-1] - df['Close'].iloc[-126]) / df['Close'].iloc[-126]
                    returns.append(ret)
            except Exception:
                pass
        if returns:
            sector_returns[sector] = np.mean(returns)

    if not sector_returns:
        return ticker_sectors, {}

    s = pd.Series(sector_returns)
    ranks = s.rank(pct=True) * 100
    sector_ranks = ranks.to_dict()
    return ticker_sectors, sector_ranks

def check_weekly_confirmation(df, end_date=None):
    """Multi-timeframe weekly confirmation filter.
    
    Checks two conditions on weekly-resampled data:
    1. Weekly close must be above both the 10-week and 40-week simple moving averages
       (equivalent to 50-day and 200-day on weekly charts).
    2. Weekly RS line (stock / Nifty 50) must be at or near its 10-week high,
       confirming the stock is outperforming the broad market on a weekly basis.
    
    Returns True if both conditions are met, False otherwise.
    """
    if end_date:
        end_ts = pd.Timestamp(end_date)
        if df.index.tz is not None:
            end_ts = end_ts.tz_localize(df.index.tz)
        df = df[df.index <= end_ts]

    weekly = df['Close'].resample('W').last().dropna()
    if len(weekly) < 40:
        return False

    sma10w = weekly.rolling(10).mean()
    sma40w = weekly.rolling(40).mean()

    # Condition 1: Weekly close above 10w and 40w SMAs
    if weekly.iloc[-1] < sma10w.iloc[-1] or weekly.iloc[-1] < sma40w.iloc[-1]:
        return False

    # Condition 2: Weekly RS line making new highs vs Nifty 50
    try:
        nifty_cache = os.path.join(CACHE_DIR, "^NSEI_1d.csv")
        if os.path.exists(nifty_cache):
            nifty_df = pd.read_csv(nifty_cache, index_col=0, parse_dates=True)
        else:
            nifty_df = yf.Ticker('^NSEI').history(period="2y")
        if end_date:
            n_end = pd.Timestamp(end_date)
            if nifty_df.index.tz is not None:
                n_end = n_end.tz_localize(nifty_df.index.tz)
            nifty_df = nifty_df[nifty_df.index <= n_end]
        nifty_weekly = nifty_df['Close'].resample('W').last().dropna()
        # Align the two series on matching weekly dates
        common_idx = weekly.index.intersection(nifty_weekly.index)
        if len(common_idx) >= 10:
            rs_line = weekly.loc[common_idx] / nifty_weekly.loc[common_idx]
            rs_10w_high = rs_line.rolling(10).max()
            # RS line must be within 3% of its 10-week high (i.e. near new highs)
            if rs_line.iloc[-1] < rs_10w_high.iloc[-1] * 0.97:
                return False
    except Exception:
        pass  # If Nifty data unavailable, skip RS line check gracefully

    return True

def calculate_obv_score(df, base_start_idx, base_end_idx):
    """Measure institutional accumulation during the VCP base via On-Balance Volume.
    
    Computes OBV (cumulative sum of volume × sign of daily close change) over the
    base period, fits a linear regression, and normalises the slope by average volume.
    
    A positive slope indicates institutions are accumulating shares while price
    consolidates — a bullish signal for long-term breakouts.
    
    Args:
        df: DataFrame (reset_index) covering the VCP window.
        base_start_idx: Index of the first peak in the base.
        base_end_idx: Index of the pivot peak (end of base).
    
    Returns:
        tuple: (label, slope) where label is 'bullish'/'neutral'/'bearish'
               and slope is the normalised OBV slope.
    """
    base_slice = df.iloc[base_start_idx:base_end_idx + 1]
    if len(base_slice) < 10:
        return 'neutral', 0.0

    close_diff = base_slice['Close'].diff()
    obv = (np.sign(close_diff) * base_slice['Volume']).cumsum().values
    x = np.arange(len(obv))
    slope = np.polyfit(x, obv, 1)[0]

    mean_vol = base_slice['Volume'].mean()
    norm_slope = slope / mean_vol if mean_vol > 0 else 0

    if norm_slope > 0.05:
        return 'bullish', round(norm_slope, 4)
    elif norm_slope < -0.05:
        return 'bearish', round(norm_slope, 4)
    return 'neutral', round(norm_slope, 4)

def calculate_quality_score(eps_growth, sales_growth, rs_rating, profit_margin):
    """Compute a weighted fundamental quality score (0-100) for ranking candidates.
    
    Unlike the ST screener which uses pass/fail gating, this scores stocks on a
    continuum so the best fundamental setups float to the top of the results.
    
    Weights:
        - 30%: EPS growth (scaled relative to 15% threshold, capped at 3×)
        - 20%: Sales growth (scaled relative to 10% threshold, capped at 3×)
        - 30%: RS rating (scaled relative to 60 threshold, capped at 1.5×)
        - 20%: Profit margin (>15% = full, >5% = half, else zero)
    
    Returns:
        float: Quality score between 0 and 100.
    """
    eps_score = min(eps_growth / 0.15, 3.0) * 30
    sales_score = min(sales_growth / 0.10, 3.0) * 20
    rs_score = min(rs_rating / 60, 1.5) * 30
    margin_score = (20 if profit_margin > 0.15 else 10 if profit_margin > 0.05 else 0)
    return min(round(eps_score + sales_score + rs_score + margin_score, 1), 100)

def check_criteria(data, ticker, rs_rating, end_date=None,
                   sector_info=None, sector_ranks=None):
    """Main screening function — checks if a stock passes all LT breakout criteria.
    
    Pipeline (in order — early rejection for efficiency):
        1. Trend Template: Price > SMA50 > SMA150 > SMA200, rising SMA200,
           price within proximity bounds of 52-week high/low.
        2. Sector RS Filter: Reject stocks from bottom 40% sectors.
        3. Fundamentals: EPS ≥ 15%, Sales ≥ 10%, RS ≥ 60 (relaxed vs ST).
           Computes Quality Score for ranking.
        4. Weekly Confirmation: Weekly close > 10w & 40w SMAs, RS line vs Nifty.
        5. VCP Detection: Smoothed peak/trough detection on 200-day window.
           Progressive contraction required if ≥2 pullbacks. Final tightness ≤ 15%.
           Flat-top resistance variance ≤ 15%. Base length 30-200 days.
        6. OBV Accumulation: Measures institutional buying during the base.
        7. Breakout + Risk: Close > pivot, volume ≥ 1.0× avg.
           Stop loss 2% below last trough low, target 50% above pivot.
    
    Returns:
        dict with match details if all criteria pass, None otherwise.
    """
    info = data.get('info', {})
    df = data.get('history', None)
    if df is None or df.empty:
        return None

    if end_date:
        end_ts = pd.Timestamp(end_date)
        if df.index.tz is not None:
            end_ts = end_ts.tz_localize(df.index.tz)
        df = df[df.index <= end_ts].copy()

    if len(df) < 200:
        return None

    # --- 1. Trend Template (relaxed) ---
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA150'] = df['Close'].rolling(150).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['52W_High'] = df['High'].rolling(252).max()
    df['52W_Low'] = df['Low'].rolling(252).min()
    df['VolSMA50'] = df['Volume'].rolling(50).mean()

    curr = df.iloc[-1]
    current_close = curr['Close']
    current_vol = curr['Volume']
    sma_50, sma_150, sma_200 = curr['SMA50'], curr['SMA150'], curr['SMA200']
    high_52w, low_52w = curr['52W_High'], curr['52W_Low']
    vol_sma_50 = curr['VolSMA50']

    if current_close < sma_50 or current_close < sma_150 or current_close < sma_200:
        return None
    if sma_50 < sma_150 or sma_150 < sma_200:
        return None

    prev_sma_200 = df['SMA200'].iloc[-20] if len(df) >= 20 else df['SMA200'].iloc[0]
    if sma_200 <= prev_sma_200:
        return None

    if current_close < low_52w * MIN_PRICE_TO_LOW:
        return None
    if current_close < high_52w * MAX_PRICE_TO_HIGH:
        return None

    # --- 2. Sector RS Filter ---
    ticker_sector = (sector_info or {}).get(ticker, 'Unknown')
    sector_rank = (sector_ranks or {}).get(ticker_sector, 50)
    if sector_rank < 40:  # Bottom 40% sectors rejected
        return None

    # --- 3. Fundamentals (scoring, not gating) ---
    eps_growth = info.get('earningsQuarterlyGrowth', 0) or 0
    sales_growth = info.get('revenueGrowth', 0) or 0
    profit_margin = info.get('profitMargins', 0) or 0

    if rs_rating < MIN_RS_RATING:
        return None
    if eps_growth < MIN_EPS_GROWTH:
        return None
    if sales_growth < MIN_SALES_GROWTH:
        return None

    quality_score = calculate_quality_score(eps_growth, sales_growth, rs_rating, profit_margin)

    # --- 4. Weekly Confirmation ---
    if not check_weekly_confirmation(df, end_date):
        return None

    # --- 5. VCP Logic (relaxed contractions) ---
    df['SmoothClose'] = df['Close'].ewm(span=5).mean()
    vcp_window = min(VCP_WINDOW, len(df))
    closes_smoothed = df['SmoothClose'].tail(vcp_window).values
    raw_closes = df['Close'].tail(vcp_window).values
    raw_lows = df['Low'].tail(vcp_window).values

    mean_price = np.mean(closes_smoothed)
    prominence_threshold = 0.02 * mean_price
    peaks, _ = scipy.signal.find_peaks(closes_smoothed, distance=PEAK_DISTANCE, prominence=prominence_threshold)
    troughs, _ = scipy.signal.find_peaks(-closes_smoothed, distance=PEAK_DISTANCE, prominence=prominence_threshold)

    if len(peaks) < 2 or len(troughs) < 1:
        return None

    pullbacks = []
    pullback_peak_indices = []
    for p in peaks:
        valid_troughs = troughs[troughs > p]
        if len(valid_troughs) > 0:
            t = valid_troughs[0]
            depth = (closes_smoothed[p] - closes_smoothed[t]) / closes_smoothed[p]
            pullbacks.append(depth)
            pullback_peak_indices.append(p)

    if len(pullbacks) < 1:  # LT needs at least 1 pullback (vs 2 for ST)
        return None

    recent_pullbacks = pullbacks[-3:] if len(pullbacks) >= 3 else pullbacks
    recent_peak_indices = pullback_peak_indices[-3:] if len(pullback_peak_indices) >= 3 else pullback_peak_indices

    # Progressive contraction (if multiple pullbacks exist)
    if len(recent_pullbacks) >= 2:
        if recent_pullbacks[-2] <= recent_pullbacks[-1]:
            return None

    if recent_pullbacks[-1] > MAX_FINAL_TIGHTNESS:
        return None

    # Flat top resistance
    pivot_peak_idx = peaks[-1]
    resistance_peak_indices = sorted(set(list(recent_peak_indices) + [pivot_peak_idx]))
    peak_prices = closes_smoothed[resistance_peak_indices]
    resistance_level = np.mean(peak_prices)
    peak_variance = (np.max(peak_prices) - np.min(peak_prices)) / resistance_level
    if peak_variance > PEAK_VARIANCE_LIMIT:
        return None

    # Base length
    base_start = recent_peak_indices[0]
    base_end = pivot_peak_idx
    base_length = base_end - base_start
    if base_length < MIN_BASE_LENGTH or base_length > MAX_BASE_LENGTH:
        return None

    # --- 6. OBV Accumulation ---
    df_tail = df.tail(vcp_window).reset_index(drop=True)
    obv_label, obv_slope = calculate_obv_score(df_tail, base_start, base_end)

    # --- 7. Breakout and Risk Management ---
    pivot_point = raw_closes[pivot_peak_idx]
    if current_close <= pivot_point:
        return None

    if current_vol < vol_sma_50 * VOLUME_BREAKOUT_MULT:
        return None

    pre_pivot_troughs = troughs[troughs < pivot_peak_idx]
    if len(pre_pivot_troughs) > 0:
        stop_loss = raw_lows[pre_pivot_troughs[-1]] * 0.98
    else:
        stop_loss = current_close * 0.92

    target_price = pivot_point * TARGET_MULT
    if current_close >= target_price:
        return None

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
        'Pullbacks': [round(float(p) * 100, 1) for p in recent_pullbacks],
        'Quality_Score': quality_score,
        'OBV_Accum': obv_label,
        'Sector': ticker_sector,
        'Sector_Rank': round(sector_rank, 1),
    }

def create_pdf_title_page(pdf, timestamp, market_status):
    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    render_pdf_standard_header(fig, "Stock Analysis Report")
    ACCENT = '#16a34a'
    SECONDARY = '#475569'
    fig.text(0.5, 0.65, "Minervini LT Breakout", ha='center', va='center', fontsize=36, weight='bold', color=ACCENT)
    fig.text(0.5, 0.58, "Long-Term Position Trade Scanner", ha='center', va='center', fontsize=22, weight='bold', color=SECONDARY)
    fig.text(0.5, 0.52, f"Strategic Market Scan: {timestamp}", ha='center', va='center', fontsize=14, color=SECONDARY)
    status_color = '#16a34a' if "Bullish" in market_status else '#dc2626'
    fig.text(0.5, 0.46, f"Market Condition: {market_status}", ha='center', va='center', fontsize=16, weight='bold', color=status_color)

    PRIMARY = '#1e293b'
    fig.text(0.1, 0.35, "LT Screener Additions", fontsize=16, weight='bold', color=PRIMARY)
    features = [
        "● Relaxed VCP (15% tightness, 200d window)",
        "● Weekly timeframe confirmation",
        "● Sector/Industry RS ranking",
        "● OBV accumulation detection",
        "● Fundamental quality scoring (0-100)",
        "● 50% target from pivot",
    ]
    for i, s in enumerate(features):
        fig.text(0.15, 0.31 - (i * 0.04), s, fontsize=12, color=SECONDARY)
    pdf.savefig(fig)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Mark Minervini Long-Term Breakout Screener")
    parser.add_argument('--limit', type=int, help="Limit number of stocks to scan")
    parser.add_argument('--sample', type=int, help="Run on a random sample")
    parser.add_argument('--refresh', action='store_true', help="Force refresh of cached data")
    parser.add_argument('--end-date', type=str, default=None, help="Run screener up to this date (DD-MM-YYYY)")
    args = parser.parse_args()

    # If date is provided as DD-MM-YYYY, parse it into YYYY-MM-DD internally so pandas handles it
    parsed_end_date = None
    if args.end_date:
        try:
            dt = datetime.strptime(args.end_date, "%d-%m-%Y")
            parsed_end_date = dt.strftime("%Y-%m-%d")
            console.print(f"[cyan]Running LT screener with end date: {args.end_date} (parsed as {parsed_end_date})[/cyan]")
        except ValueError:
            console.print(f"[red]Invalid date format: {args.end_date}. Use DD-MM-YYYY.[/red]")
            return

    setup_directories()
    tickers = load_tickers()
    if args.limit:
        tickers = tickers[:args.limit]
    if args.sample:
        import random
        tickers = random.sample(tickers, args.sample)

    # RS ratings
    rs_ratings = calculate_rs_ratings(tickers, end_date=parsed_end_date)

    # Sector RS
    console.print("[blue]Calculating Sector RS rankings...[/blue]")
    sector_info, sector_ranks = calculate_sector_rs(tickers, end_date=parsed_end_date)

    # Market condition
    market_status = "Unknown"
    try:
        nifty = yf.Ticker('^NSEI')
        nifty_df = nifty.history(period="2y")
        if parsed_end_date:
            end_ts = pd.Timestamp(parsed_end_date)
            if nifty_df.index.tz is not None:
                end_ts = end_ts.tz_localize(nifty_df.index.tz)
            nifty_df = nifty_df[nifty_df.index <= end_ts]
        if len(nifty_df) >= 50:
            nifty_df['SMA50'] = nifty_df['Close'].rolling(50).mean()
            curr_n = nifty_df['Close'].iloc[-1]
            sma50_n = nifty_df['SMA50'].iloc[-1]
            if curr_n > sma50_n:
                market_status = f"Bullish (Nifty: {curr_n:.2f} > 50SMA: {sma50_n:.2f})"
            else:
                market_status = f"Bearish (Nifty: {curr_n:.2f} < 50SMA: {sma50_n:.2f})"
    except Exception as e:
        console.print(f"[yellow]Error fetching market condition: {e}[/yellow]")

    results = []

    def scan_ticker(ticker):
        data = fetch_data(ticker, refresh=args.refresh)
        if data:
            rs = rs_ratings.get(ticker, 0)
            match = check_criteria(data, ticker, rs, end_date=parsed_end_date,
                                   sector_info=sector_info, sector_ranks=sector_ranks)
            if match:
                return (match, data)
        return None

    console.print(f"[bold green]Scanning {len(tickers)} stocks (Minervini LT, parallel)...[/bold green]")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(scan_ticker, t): t for t in tickers}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    match, data = result
                    console.print(f"[green]FOUND LT: {match['Ticker']} (Q:{match['Quality_Score']}, OBV:{match['OBV_Accum']}, Sector:{match['Sector']})[/green]")
                    results.append((match, data))
            except Exception as e:
                console.print(f"[yellow]Error scanning {futures[future]}: {e}[/yellow]")

    if results:
        with PdfPages(PDF_PATH) as pdf:
            create_pdf_title_page(pdf, TIMESTAMP, market_status)
            df_results = pd.DataFrame([r[0] for r in results])
            df_results['Pullbacks'] = df_results['Pullbacks'].apply(lambda x: str([float(v) for v in x]) if isinstance(x, list) else x)
            # Sort by Quality Score descending
            df_results = df_results.sort_values('Quality_Score', ascending=False)
            render_pdf_styled_table(pdf, df_results, "Minervini LT Breakout Results")
            for match, data in results:
                generate_chart(data['history'], match['Ticker'], match, pdf=pdf, end_date=parsed_end_date)
            df_results.to_csv(os.path.join(OUTPUT_DIR, 'results.csv'), index=False)
            console.print(f"\n[bold]Results saved to {OUTPUT_DIR}/results.csv[/bold]")
            console.print(f"[bold green]PDF Report saved to {PDF_PATH}[/bold green]")
    else:
        console.print("[yellow]No stocks found matching Minervini LT criteria.[/yellow]")

if __name__ == "__main__":
    main()
