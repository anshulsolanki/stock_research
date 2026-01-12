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
BATCH ANALYSIS MODULE
=====================

PURPOSE:
--------
Process a list of stocks through multiple technical indicators to generate
a unified "Composite Strength Score" (CSS) for ranking and filtering.

Uses multiprocessing to handle multiple tickers efficiently.
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import multiprocessing
from functools import partial
import yfinance as yf

# Add parent directories to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lagging_indicator_analysis'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'leading_indicator_analysis'))

try:
    from macd_analysis import run_analysis as run_macd
    from supertrend_analysis import run_analysis as run_supertrend
    from rsi_volume_divergence import run_analysis as run_rsi_volume
    from volatility_squeeze_analysis import run_analysis as run_squeeze
    from rs_analysis import run_analysis as run_rs
    from crossover_analysis import run_analysis as run_crossover
except ImportError:
    # If running from app.py, paths might already be set, but if independent, we need them
    pass


# ========== HELPER FUNCTIONS FOR COLUMN EXTRACTION ==========

def get_trend_direction(macd_trend, supertrend_status):
    """
    Combines MACD and Supertrend to determine overall trend.
    Returns: 🟢 Bullish / 🔴 Bearish / 🟡 Neutral
    
    Logic:
    - Bullish: MACD Line > Signal Line (Bullish) AND Price > Supertrend (UPTREND)
    - Bearish: MACD Line < Signal Line (Bearish) AND Price < Supertrend (DOWNTREND)
    - Neutral: Mixed signals
    """
    macd_bullish = (macd_trend == 'Bullish')
    # Check if UPTREND is in the status (handles "UPTREND (Buy)" format)
    st_bullish = 'UPTREND' in str(supertrend_status)
    
    if macd_bullish and st_bullish:
        return '🟢 Bullish'
    elif not macd_bullish and not st_bullish:
        return '🔴 Bearish'
    else:
        return '🟡 Neutral'


def get_rsi_state(rsi_value):
    """
    Returns RSI state based on value.
    Returns: 🔥 Overbought / ❄️ Oversold / ➖ Neutral
    """
    if rsi_value > 70:
        return '🔥 Overbought'
    elif rsi_value < 30:
        return '❄️ Oversold'
    else:
        return '➖ Neutral'


def get_divergence_status(bullish_divs, bearish_divs):
    """
    Extract latest divergence type from RSI Volume divergences.
    Returns: 🐂 Bullish / 🐻 Bearish / ➖ None
    """
    # Check if we have any divergences
    has_bullish = bullish_divs and len(bullish_divs) > 0
    has_bearish = bearish_divs and len(bearish_divs) > 0
    
    if not has_bullish and not has_bearish:
        return '➖ None'
    
    # Get the most recent divergence by comparing dates
    latest_bullish_date = None
    latest_bearish_date = None
    
    if has_bullish:
        latest_bullish = bullish_divs[-1]
        latest_bullish_date = latest_bullish.get('Date')
    
    if has_bearish:
        latest_bearish = bearish_divs[-1]
        latest_bearish_date = latest_bearish.get('Date')
    
    # Return the most recent one
    if latest_bullish_date and latest_bearish_date:
        # Both exist, return the more recent one
        if latest_bullish_date > latest_bearish_date:
            return '🐂 Bullish'
        else:
            return '🐻 Bearish'
    elif latest_bullish_date:
        return '🐂 Bullish'
    elif latest_bearish_date:
        return '🐻 Bearish'
    else:
        return '➖ None'


def get_squeeze_status(squeeze_signals):
    """
    Extract current squeeze status.
    Returns: ⚠️ Squeeze / 💥 Breakout / ➖ Normal
    """
    if not squeeze_signals:
        return '➖ Normal'
    
    latest = squeeze_signals[-1]
    sig_type = latest.get('Type', '')
    
    if 'Squeeze' in sig_type:
        return '⚠️ Squeeze'
    elif 'Breakout' in sig_type:
        return '💥 Breakout'
    else:
        return '➖ Normal'


def get_trend_signal(crossover_signal, supertrend_status):
    """
    Determines buy/sell/hold signal from crossover and supertrend.
    Returns: ⬆️ Buy / ⬇️ Sell / ➖ Hold
    """
    if crossover_signal and 'Bullish' in str(crossover_signal):
        return '⬆️ Buy'
    elif crossover_signal and 'Bearish' in str(crossover_signal):
        return '⬇️ Sell'
    elif supertrend_status == 'UPTREND':
        return '⬆️ Buy'
    elif supertrend_status == 'DOWNTREND':
        return '⬇️ Sell'
    else:
        return '➖ Hold'


def check_recent_crossover(crossover_signal, days=3):
    """
    Check if crossover signal is recent (within last N days).
    Note: This is simplified - would need timestamp from signal to be precise.
    For now, we assume any signal returned is recent.
    """
    # In a full implementation, we'd check the signal timestamp
    # For now, treat any signal as recent
    return crossover_signal is not None and crossover_signal != 'No Signal'


def calculate_200_ema(ticker, interval='1d'):
    """
    Fetch data and calculate 200 EMA, compare with current price.
    Returns: (price, ema_200, above_ema) tuple
    """
    try:
        # Fetch enough data for 200 EMA (need 250+ days to be safe)
        df = yf.download(ticker, period='1y', interval=interval, progress=False)
        
        if df is None or len(df) < 200:
            return None, None, False
        
        # Calculate 200 EMA - need to call .mean() on the window object
        ema_200_series = df['Close'].ewm(span=200, adjust=False).mean()
        ema_200_val = float(ema_200_series.iloc[-1])
        current_price_val = float(df['Close'].iloc[-1])
        
        return current_price_val, ema_200_val, current_price_val > ema_200_val
    except Exception as e:
        print(f"Error calculating 200 EMA for {ticker}: {e}")
        return None, None, False


# ========== SCORING ALGORITHM ==========

def calculate_composite_score(metrics):
    """
    Calculate 0-100 Composite Strength Score (CSS) based on metrics.
    
    Formula: (Trend * 0.3) + (Momentum * 0.2) + (Signals * 0.2) + (RS * 0.3)
    
    ALIGNED WITH STRATEGY:
    - Trend (30): +15 Price>200EMA, +10 MACD>Signal, +5 Histogram↑
    - Momentum (20): +10 RSI 40-70, +10 No Bearish Div, +10 Bullish Div
    - Signals (20): +20 Recent Crossover, +15 Breakout, +10 Squeeze
    - RS (30): 8-10=30pts, 5-7=20pts, 3-4=5pts, 0-2=0pts
    """
    trend_score = 0
    momentum_score = 0
    signal_score = 0
    rs_raw_score = 0
    
    # --- 1. Trend Component (Max 30 pts) ---
    # +15 pts: Price > 200 EMA (Long-term trend)
    if metrics.get('above_200_ema'):
        trend_score += 15
    
    # +10 pts: MACD Line > Signal Line (Bullish MACD)
    if metrics.get('macd_trend') == 'Bullish':
        trend_score += 10
    
    # +5 pts: MACD Histogram Increasing (Momentum accelerating)
    if metrics.get('macd_momentum') == 'Strengthening':
        trend_score += 5
        
    # --- 2. Momentum Component (Max 20 pts) ---
    rsi = metrics.get('rsi_value', 50)
    
    # +10 pts: RSI between 40 and 70 (Healthy bullish zone)
    if 40 <= rsi <= 70:
        momentum_score += 10
    
    # Divergences - now check both bullish and bearish lists
    bullish_divs = metrics.get('bullish_divergences', [])
    bearish_divs = metrics.get('bearish_divergences', [])
    has_bullish_div = bullish_divs is not None and len(bullish_divs) > 0
    has_bearish_div = bearish_divs is not None and len(bearish_divs) > 0
    
    # +10 pts: No Bearish Divergence
    if not has_bearish_div:
        momentum_score += 10
    else:
        # Penalty: -10 pts for Bearish Divergence
        momentum_score -= 10
    
    # Bonus: +10 pts for Bullish Divergence (especially in oversold)
    if has_bullish_div:
        momentum_score += 10
        
    # Clamp component score
    momentum_score = max(0, min(20, momentum_score))
        
    # --- 3. Signal Component (Max 20 pts) ---
    # +20 pts: Recent Bullish Crossover (within last 3 days)
    crossover_signal = metrics.get('crossover_signal')
    if check_recent_crossover(crossover_signal) and 'Bullish' in str(crossover_signal):
        signal_score += 20
    elif check_recent_crossover(crossover_signal) and 'Bearish' in str(crossover_signal):
        # Penalty: -20 pts for Recent Sell Signal
        signal_score -= 20
        
    # Volatility Squeeze/Breakout
    squeeze_signals = metrics.get('squeeze_signals', [])
    if squeeze_signals:
        latest = squeeze_signals[-1]
        sig_type = latest.get('Type', '')
        
        # +15 pts: Volatility Breakout (Bullish)
        if 'Bullish Breakout' in sig_type:
            signal_score += 15
        elif 'Bearish Breakout' in sig_type:
            signal_score -= 15
        # +10 pts: Active Squeeze (Potential for move)
        elif 'Squeeze' in sig_type:
            signal_score += 10
            
    # Clamp component score
    signal_score = max(0, min(20, signal_score))
             
    # --- 4. RS Component (Max 30 pts) ---
    # IMPORTANT: RS Score is on 0-10 scale (decile rank)
    rs_score = metrics.get('rs_score', 0)
    
    # Convert 0-100 to 0-10 if needed (rs_analysis might return 0-100)
    if rs_score > 10:
        rs_score = rs_score / 10  # Convert to 0-10 scale
    
    # Map to points according to strategy
    if rs_score >= 8:  # Leader (8-10)
        rs_raw_score = 30
    elif rs_score >= 5:  # Performer (5-7)
        rs_raw_score = 20
    elif rs_score >= 3:  # Lagging (3-4)
        rs_raw_score = 5
    else:  # Laggard (0-2)
        rs_raw_score = 0
        
    # Final Calculation
    total_score = trend_score + momentum_score + signal_score + rs_raw_score
    
    return max(0, min(100, total_score))


# ========== MAIN ANALYSIS FUNCTION ==========

def sanitize_for_json(data):
    """
    Convert pandas Timestamps and other non-serializable objects to strings.
    """
    if isinstance(data, list):
        return [sanitize_for_json(item) for item in data]
    elif isinstance(data, dict):
        return {key: sanitize_for_json(value) for key, value in data.items()}
    elif isinstance(data, (pd.Timestamp, datetime)):
        return str(data)
    else:
        return data


def analyze_single_stock(ticker, interval='1d'):
    """
    Run all indicators for a single stock and aggregate results.
    Returns structured data matching strategy columns.
    
    Args:
        ticker: Stock symbol to analyze
        interval: Timeframe for analysis ('1d', '1wk', '1mo', '15m')
    """
    try:
        metrics = {}
        
        # 0. Calculate 200 EMA first (need raw data)
        price, ema_200, above_ema = calculate_200_ema(ticker, interval)
        metrics['price'] = price
        metrics['ema_200'] = ema_200
        metrics['above_200_ema'] = above_ema
        
        # 1. MACD
        macd_res = run_macd(ticker, show_plot=False, config={'interval': interval})
        if macd_res.get('success'):
            metrics['macd_trend'] = macd_res.get('trend')
            metrics['macd_momentum'] = macd_res.get('momentum')
            
        # 2. Supertrend
        st_res = run_supertrend(ticker, show_plot=False, config={'interval': interval})
        if st_res.get('success'):
            metrics['supertrend_status'] = st_res.get('status')  # UPTREND/DOWNTREND
            # Use supertrend price if we didn't get it from 200 EMA calc
            if metrics['price'] is None:
                metrics['price'] = st_res.get('last_price')
            
        # 3. RSI & Volume Divergence
        rsi_res = run_rsi_volume(ticker, show_plot=False, config={'interval': interval})
        if rsi_res.get('success'):
            metrics['rsi_value'] = rsi_res.get('current_rsi', 50)
            # RSI Volume Divergence returns separate lists for bullish and bearish
            bullish_divs = rsi_res.get('bullish_divergences', [])
            bearish_divs = rsi_res.get('bearish_divergences', [])
            # Sanitize for JSON
            metrics['bullish_divergences'] = sanitize_for_json(bullish_divs)
            metrics['bearish_divergences'] = sanitize_for_json(bearish_divs)
            
        # 4. Volatility Squeeze
        sq_res = run_squeeze(ticker, show_plot=False, config={'interval': interval})
        if sq_res.get('success'):
            squeeze_signals = sq_res.get('signals', [])
            metrics['squeeze_signals'] = sanitize_for_json(squeeze_signals)
            
        # 5. Crossover
        cross_res = run_crossover(ticker, show_plot=False, config={'interval': interval})
        if cross_res.get('success'):
            metrics['crossover_signal'] = cross_res.get('signal')
            
        # 6. Relative Strength
        rs_res = run_rs(ticker, show_plot=False, config={'interval': interval})
        if rs_res.get('success'):
            rs_score_raw = rs_res.get('rs_score', 0)
            # Convert to 0-10 scale if it's 0-100
            metrics['rs_score'] = rs_score_raw / 10 if rs_score_raw > 10 else rs_score_raw
            # Extract RS classification (e.g., "Emerging Leader", "Strong Leader", etc.)
            metrics['rs_classification'] = rs_res.get('classification', 'Neutral')
            
        # Calculate Composite Score
        score = calculate_composite_score(metrics)
        
        # Extract detailed divergence info
        bullish_divs = metrics.get('bullish_divergences', [])
        bearish_divs = metrics.get('bearish_divergences', [])
        divergence_status = get_divergence_status(bullish_divs, bearish_divs)
        
        # Get date from the most recent divergence
        divergence_date = None
        if bullish_divs and bearish_divs:
            # Get most recent between both
            latest_bull_date = bullish_divs[-1].get('Date') if bullish_divs else None
            latest_bear_date = bearish_divs[-1].get('Date') if bearish_divs else None
            if latest_bull_date and latest_bear_date:
                divergence_date = max(latest_bull_date, latest_bear_date)
            else:
                divergence_date = latest_bull_date or latest_bear_date
        elif bullish_divs:
            divergence_date = bullish_divs[-1].get('Date')
        elif bearish_divs:
            divergence_date = bearish_divs[-1].get('Date')
        
        # Build structured column output matching strategy with enhanced details
        columns = {
            'trend_direction': get_trend_direction(
                metrics.get('macd_trend', 'Neutral'),
                metrics.get('supertrend_status', 'NEUTRAL')
            ),
            'trend_direction_detail': {
                'macd': metrics.get('macd_trend', 'Neutral'),
                'supertrend': metrics.get('supertrend_status', 'NEUTRAL')
            },
            'macd_momentum': f"{'🟢' if metrics.get('macd_momentum') == 'Strengthening' else '🔴'} {metrics.get('macd_momentum', 'Neutral')}",
            'trend_signal': get_trend_signal(
                metrics.get('crossover_signal'),
                metrics.get('supertrend_status')
            ),
            'trend_signal_detail': {
                'crossover': metrics.get('crossover_signal') or 'No Signal',
                'supertrend': metrics.get('supertrend_status', 'NEUTRAL')
            },
            'rsi_value': round(metrics.get('rsi_value', 0), 1),
            'rsi_state': get_rsi_state(metrics.get('rsi_value', 50)),
            'divergence': divergence_status,
            'divergence_date': divergence_date,
            'squeeze': get_squeeze_status(metrics.get('squeeze_signals', [])),
            'rs_score': round(metrics.get('rs_score', 0), 1),  # 0-10 scale
            'rs_classification': metrics.get('rs_classification', 'Neutral')  # Classification added
        }
        
        return {
            'ticker': ticker,
            'success': True,
            'price': round(metrics.get('price', 0), 2) if metrics.get('price') else 0,
            'score': score,
            'columns': columns,
            'raw_metrics': sanitize_for_json(metrics)  # Sanitize for JSON
        }
        
    except Exception as e:
        return {
            'ticker': ticker,
            'success': False,
            'error': str(e)
        }


def run_batch_analysis(tickers, interval='1d'):
    """
    Run analysis on a list of tickers in parallel.
    
    Args:
        tickers: List of ticker symbols
        interval: Timeframe for analysis ('1d', '1wk', '1mo', '15m')
    """
    print(f"Starting batch analysis for {len(tickers)} stocks with interval={interval}...")
    start_time = time.time()
    
    # Use multiprocessing with partial to pass interval
    cpu_count = min(multiprocessing.cpu_count(), 4)
    
    # Create partial function with interval parameter
    analyze_func = partial(analyze_single_stock, interval=interval)
    
    with multiprocessing.Pool(processes=cpu_count) as pool:
        results = pool.map(analyze_func, tickers)
        
    duration = time.time() - start_time
    print(f"Batch analysis completed in {duration:.2f} seconds")
    
    # Sort by Score descending
    results.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    return results


if __name__ == "__main__":
    # Test
    test_tickers =['TCS.NS', 'INFY.NS']
    results = run_batch_analysis(test_tickers)
    import json
    print(json.dumps(results, indent=2))
