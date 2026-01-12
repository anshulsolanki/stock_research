"""
MEAN REVERSION SELL STRATEGY
============================

OVERVIEW:
---------
This module implements a quantitative "Mean Reversion Sell Strategy" designed to identify potential 
short selling opportunities in Large Cap stocks. It looks for stocks that are statistically overextended 
to the upside and showing signs of momentum exhaustion.

PHILOSOPHY:
-----------
The strategy is based on the principle that asset prices tend to revert to their historical mean 
after extreme deviations. By combining "3-Day Rule" parabolic moves, statistical extremes (Bollinger Bands), 
and momentum divergence (MACD Mismatch), we aim to time entries when the probability of a reversal is highest.

CORE INDICATORS:
----------------
1. RSI (Relative Strength Index):
   - Threshold: > 75 (Indicates overbought conditions)
   
2. ATR (Average True Range):
   - Condition: Expanding (Current ATR > 20-period SMA of ATR)
   - Rationale: High volatility often precedes a reversal.

3. Bollinger Bands (20, 2):
   - Condition: Price > Upper Band (2SD or 3SD)
   - Rationale: Prices at statistical extremes (2-3 standard deviations) are unsustainable.

4. MACD Momentum Mismatch:
   - Condition: Daily MACD Trend = UP vs. 15-min MACD Trend = DOWN
   - Rationale: Multi-timeframe conflict suggests short-term momentum is fading within a larger uptrend.

ADVANCED SIGNALS:
-----------------
1. Divergence:
   - Detection of Bearish Divergence (Price making Higher Highs while RSI/Volume makes Lower Highs).
   - Uses `rsi_volume_divergence` module.

2. Volume Analysis:
   - Identifies "Smart Money" selling patterns:
     * Climax Volume (Churning at highs)
     * Buying Exhaustion (High volume, small price progress)
     * Distribution Days (Selling on high volume)
   - Uses `volume_analysis` module.

3. Parabolic Move ("3-Day Rule"):
   - Checks for 3 consecutive days of strong green candles with large bodies.
   - Often signals a blow-off top.

OUTPUT:
-------
The script provides a tabular summary of the analysis, indicating "SELL_WATCH" if:
   - RSI > 75 AND (Bollinger Extreme OR MACD Mismatch OR Volume Signal OR Divergence detected)
Otherwise, the status is "NEUTRAL".

USAGE:
------
1. Single Ticker:
   python mean_reversion_strategy.py RELIANCE.NS

2. Batch Analysis (from file):
   python mean_reversion_strategy.py data/watchlist.json
"""

import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# Add project root to path to import analysis modules
# From batch_reports/scripts, we need to go up two levels to reach stock_research/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

try:
    from leading_indicator_analysis import volume_analysis
    from leading_indicator_analysis import rsi_volume_divergence
except ImportError:
    # Fallback: try different depths if moved again
    try:
         sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
         from leading_indicator_analysis import volume_analysis
         from leading_indicator_analysis import rsi_volume_divergence
    except ImportError:
         print(f"Warning: Could not import analysis modules. Current path: {sys.path}")
         # Attempt to suppress subsequent NameErrors by defining dummies if needed, 
         # but better to let it fail or print specific warning as done in analyze_ticker


def fetch_data(ticker, interval='1d', period='1y'):
    """
    Fetches historical stock data from Yahoo Finance.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'RELIANCE.NS').
        interval (str): Data interval (default: '1d').
        period (str): Lookback period (default: '1y').
        
    Returns:
        pd.DataFrame or None: DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume' columns, 
                              or None if data fetch fails.
    """
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False, multi_level_index=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker} ({interval}): {e}")
        return None

def calculate_rsi(series, period=14):
    """
    Calculates the Relative Strength Index (RSI).
    
    Args:
        series (pd.Series): Series of prices (usually Close).
        period (int): Lookback period (default: 14).
        
    Returns:
        pd.Series: RSI values (0-100).
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period=14):
    """
    Calculates the Average True Range (ATR).
    
    Args:
        df (pd.DataFrame): DataFrame containing 'High', 'Low', 'Close'.
        period (int): Lookback period (default: 14).
        
    Returns:
        pd.Series: ATR values.
    """
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=period).mean()

def calculate_bollinger_bands(series, period=20, std_dev=2):
    """
    Calculates Bollinger Bands (Middle, Upper, Lower).
    
    Args:
        series (pd.Series): Series of prices.
        period (int): SMA period (default: 20).
        std_dev (float): Number of standard deviations (default: 2).
        
    Returns:
        tuple(pd.Series, pd.Series, pd.Series): (Middle Band, Upper Band, Lower Band).
    """
    ma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = ma + (std * std_dev)
    lower = ma - (std * std_dev)
    return ma, upper, lower

def calculate_macd(series, fast=12, slow=26, signal=9):
    """
    Calculates Moving Average Convergence Divergence (MACD).
    
    Args:
        series (pd.Series): Series of prices.
        fast (int): Fast EMA period (default: 12).
        slow (int): Slow EMA period (default: 26).
        signal (int): Signal line EMA period (default: 9).
        
    Returns:
        tuple(pd.Series, pd.Series, pd.Series): (MACD Line, Signal Line, Histogram).
    """
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def check_parabolic_move(df, lookback=3):
    """
    Checks for a 'Parabolic Move' or '3-Day Rule', defined as:
    1. 3 consecutive days of higher closes (all green candles).
    2. Strong closes (Body is > 50% of the total daily range).
    3. Significant move size (Optional: >1% per day).
    
    Args:
        df (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close'.
        lookback (int): Number of days to check (default: 3).
        
    Returns:
        tuple(bool, str): (Detected status, Reason/Details).
    """
    if len(df) < lookback:
        return False, "Insufficient data"
        
    recent = df.iloc[-lookback:]
    
    # Condition 1: All Green Candles (Close > Open)
    if not all(recent['Close'] > recent['Open']):
        return False, "Not all 3 days are green"
    
    # Condition 2: Strong Closes (Body > 50% of Range)
    # This filters out 'weak' green candles with long upper wicks
    bodies = (recent['Close'] - recent['Open']).abs()
    ranges = (recent['High'] - recent['Low'])
    body_pct = bodies / ranges
    
    if not all(body_pct > 0.5):
        return False, "Weak closes (Body < 50% of Range)"
        
    # Condition 3: Minimum Move Size (Optional)
    # Ensuring these aren't tiny doji-like green candles
    pct_moves = (recent['Close'] - recent['Open']) / recent['Open']
    if not all(pct_moves > 0.01): 
        # For now, we are lenient on this condition but it's good to note
        pass

    return True, "3 consecutive strong green candles detected"

def analyze_ticker(ticker):
    """
    Performs the full Mean Reversion Sell Analysis for a single ticker.
    
    Steps:
    1. Fetches Daily (1y) and 15-minute (5d) data.
    2. Calculates Indicators: RSI, ATR (and SMA), Bollinger Bands, MACD.
    3. Evaluates Signal Conditions:
       - RSI > 75
       - Expanding Volatility (ATR)
       - Price vs Bollinger Upper Bands
       - Macd Mismatch (Daily Up / 15m Down)
    4. Runs Advanced Modules:
       - Divergence Analysis
       - Volume Analysis
    5. Aggregates results into a status dictionary.
    
    Args:
        ticker (str): Stock Symbol.
        
    Returns:
        dict: Analysis results dictionary.
    """
    #print(f"Analyzing {ticker}...")
    
    # 1. Fetch Data
    df_daily = fetch_data(ticker, interval='1d', period='1y')
    df_15m = fetch_data(ticker, interval='15m', period='5d') # 15m for checking momentum mismatch
    
    if df_daily is None or df_daily.empty:
        return None

    # 2. Calculate Indicators (Daily)
    # RSI
    df_daily['RSI'] = calculate_rsi(df_daily['Close'])
    
    # ATR
    df_daily['ATR'] = calculate_atr(df_daily)
    df_daily['ATR_SMA_20'] = df_daily['ATR'].rolling(window=20).mean()
    
    # Bollinger Bands (2SD and 3SD)
    ma, upper_2sd, lower_2sd = calculate_bollinger_bands(df_daily['Close'], std_dev=2)
    _, upper_3sd, _ = calculate_bollinger_bands(df_daily['Close'], std_dev=3)
    df_daily['BB_Upper_2SD'] = upper_2sd
    df_daily['BB_Upper_3SD'] = upper_3sd
    
    # MACD (Daily)
    macd_daily, signal_daily, _ = calculate_macd(df_daily['Close'])
    df_daily['MACD'] = macd_daily
    df_daily['MACD_Signal'] = signal_daily

    # MACD (15m)
    if df_15m is not None and not df_15m.empty:
        macd_15m, signal_15m, _ = calculate_macd(df_15m['Close'])
        df_15m['MACD'] = macd_15m
        df_15m['MACD_Signal'] = signal_15m
        current_macd_15m = macd_15m.iloc[-1]
        current_signal_15m = signal_15m.iloc[-1]
        trend_15m = "Up" if current_macd_15m > current_signal_15m else "Down"
    else:
        trend_15m = "Unknown"

    # --- EVALUATE CONDITIONS ---
    current_idx = -1
    current_rsi = df_daily['RSI'].iloc[current_idx]
    current_atr = df_daily['ATR'].iloc[current_idx]
    current_atr_sma = df_daily['ATR_SMA_20'].iloc[current_idx]
    current_close = df_daily['Close'].iloc[current_idx]
    current_upper_2sd = df_daily['BB_Upper_2SD'].iloc[current_idx]
    current_upper_3sd = df_daily['BB_Upper_3SD'].iloc[current_idx]
    current_macd_daily = df_daily['MACD'].iloc[current_idx]
    current_signal_daily = df_daily['MACD_Signal'].iloc[current_idx]
    
    # 1. RSI > 75
    rsi_check = current_rsi > 75
    
    # 2. ATR Expanding
    atr_check = current_atr > current_atr_sma
    
    # 3. Bollinger Band Position
    bb_position = "Inside"
    if current_close > current_upper_3sd:
        bb_position = "Above 3-SD (Extreme)"
    elif current_close > current_upper_2sd:
        bb_position = "Above 2-SD"
    elif current_close > ma.iloc[current_idx]:
        bb_position = "Upper Half"
        
    # 4. Momentum Mismatch
    trend_daily = "Up" if current_macd_daily > current_signal_daily else "Down"
    mismatch_check = (trend_daily == "Up" and trend_15m == "Down")

    # 5. Divergences (Using rsi_volume_divergence)
    divergence_check = False
    divergence_details = []
    try:
        # We need to adapt rsi_volume_divergence to accept DF or run it
        # Assuming run_analysis can take a 'df' or we let it fetch
        div_res = rsi_volume_divergence.run_analysis(ticker, df=df_daily, show_plot=False)
        if div_res.get('divergences'):
            # Check for Bearish Divergences in recent days
            for div in div_res['divergences']:
                # Pass if 'Bearish' is in type and date is very recent (last 3-5 days)
                # Note: div['Date'] might be timestamp or string depending on module
                # rsi_volume_divergence usually returns timestamps in 'divergences' list before json conversion?
                # Actually run_analysis returns serialized date often.
                pass 
            # For now, just logging if any found for simplification, refine later
            if len(div_res['divergences']) > 0:
                divergence_check = True
                divergence_details = div_res['divergences'][-1] # details of latest
    except Exception as e:
        print(f"Divergence check failed: {e}")

    # 6. Volume Analysis
    volume_signals = []
    try:
        vol_res = volume_analysis.run_analysis(ticker, df=df_daily, show_plot=False)
        if vol_res.get('divergences'):
             # Volume analysis uses 'divergences' key for signals too
             # Filter for Climax, Distribution, Exhaustion
             for sig in vol_res['divergences'][-3:]: # Check last few signals
                 # Check date recency if needed, for now just taking latest
                 volume_signals.append(sig['Type'])
    except Exception as e:
        print(f"Volume check failed: {e}")
        
    # 7. Parabolic Move
    parabolic_check, parabolic_msg = check_parabolic_move(df_daily)
    
    # RESULT CONSTRUCTION
    result = {
        "ticker": ticker,
        "date": df_daily.index[-1].strftime('%Y-%m-%d'),
        "status": "NEUTRAL", # Will update
        "indicators": {
            "RSI": { "value": round(float(current_rsi), 2), "pass": bool(rsi_check) },
            "ATR": { "value": round(float(current_atr), 2), "sma": round(float(current_atr_sma), 2), "status": "Expanding" if atr_check else "Contracting", "pass": bool(atr_check) },
            "Bollinger": { "position": bb_position, "close": round(float(current_close), 2), "upper_2sd": round(float(current_upper_2sd), 2), "pass": bb_position.startswith("Above") },
            "MACD_Mismatch": { "pass": bool(mismatch_check), "daily": trend_daily, "15m": trend_15m }
        },
        "signals": {
            "Divergence": { "detected": bool(divergence_check), "details": divergence_details },
            "Parabolic_Move": { "detected": bool(parabolic_check), "details": parabolic_msg },
            "Volume_Analysis": { "detected": len(volume_signals) > 0, "signals": volume_signals }
        }
    }
    
    # DETERMINE FINAL STATUS
    # "Sell Watch" if: RSI high + (BB Extreme OR Mismatch OR Volume Signal OR Div)
    if rsi_check and (bb_position.startswith("Above") or mismatch_check or len(volume_signals) > 0 or divergence_check):
         result['status'] = "SELL_WATCH"
         
    return result

def print_results_table(result):
    """
    Prints a formatted ASCII table of the analysis results.
    
    Args:
        result (dict): The result dictionary returned by analyze_ticker.
    """
    print(f"\n{'='*50}")
    print(f"MEAN REVERSION ANALYSIS: {result['ticker']} ({result['date']})")
    print(f"STATUS: {result['status']}")
    print(f"{'='*50}")
    
    # Indicators Table
    print("\n--- CORE INDICATORS ---")
    data_ind = []
    for name, details in result['indicators'].items():
        is_pass = details.get('pass', False)
        pass_status = "✅ PASS" if is_pass else "❌ FAIL"
        val = details.get('value', '')
        if name == 'Bollinger':
            val = details.get('position', '')
        elif name == 'MACD_Mismatch':
            val = f"Daily {details['daily']} / 15m {details['15m']}"
        elif name == 'ATR':
             val = f"{details['value']} ({details['status']})"
             
        data_ind.append({"Indicator": name, "Value/Status": val, "Check": pass_status})
        
    df_ind = pd.DataFrame(data_ind)
    # Use to_string to avoid requiring tabulate dependency
    print(df_ind.to_string(index=False))
    
    # Signals Table
    print("\n--- ADVANCED SIGNALS ---")
    data_sig = []
    for name, details in result['signals'].items():
        detected = "🚨 DETECTED" if details['detected'] else "⚪ None"
        info = ""
        if isinstance(details.get('details'), list) and details.get('details'):
            info = ", ".join(details['details']) if isinstance(details['details'][0], str) else str(details['details'])
        elif isinstance(details.get('signals'), list) and details.get('signals'):
             info = ", ".join(details['signals'])
        else:
            info = str(details.get('details', ''))
            
        data_sig.append({"Signal": name, "Status": detected, "Details": info[:50] + "..." if len(info) > 50 else info})
        
    df_sig = pd.DataFrame(data_sig)
    print(df_sig.to_string(index=False))
    print(f"\n{'='*50}\n")

def analyze_batch(tickers_file='tickers.txt'):
    """
    Runs Mean Reversion analysis on multiple tickers from a file.
    
    Parameters:
    -----------
    tickers_file : str
        Path to text file containing ticker symbols (one per line)
    """
    # Check for file in multiple locations
    if not os.path.exists(tickers_file):
        if os.path.exists(os.path.join('..', '..', tickers_file)):
            tickers_file = os.path.join('..', '..', tickers_file)
        elif os.path.exists(os.path.join('..', tickers_file)):
            tickers_file = os.path.join('..', tickers_file)
        else:
            print(f"Error: {tickers_file} not found.")
            return

    with open(tickers_file, 'r') as f:
        if tickers_file.endswith('.json'):
            try:
                import json
                data = json.load(f)
                if isinstance(data, dict):
                    # Handle specific formats like {"watchlist": [...]} or {"IT": {"stocks": {...}}}
                    if 'watchlist' in data:
                        tickers = data['watchlist']
                    elif 'tickers' in data:
                         tickers = data['tickers']
                    else:
                        # Check if it's a simple key-value pair where values are tickers
                        # e.g. {"Stock Name": "TICKER"}
                        all_values_are_strings = all(isinstance(v, str) for v in data.values())
                        if all_values_are_strings:
                            tickers = list(data.values())
                        else:
                             print(f"Warning: JSON structure unknown. Keys found: {list(data.keys())}")
                             tickers = []
                elif isinstance(data, list):
                    tickers = data
                else:
                    print("Error: JSON content must be a list or a dict with 'watchlist' key.")
                    return
            except json.JSONDecodeError:
                print(f"Error: Failed to parse JSON file {tickers_file}")
                return
        else:
            tickers = [line.strip() for line in f if line.strip()]

    print(f"\nFound {len(tickers)} tickers. Starting Mean Reversion Analysis...\n")
    print(f"{'='*80}")
    print(f"{'TICKER':<15} | {'STATUS':<12} | {'RSI':<8} | {'BB':<15} | {'MISMATCH':<10} | {'SIGNALS'}")
    print(f"{'='*80}")

    for ticker in tickers:
        try:
            res = analyze_ticker(ticker)
            if res:
                # Format row for summary table
                status_icon = "🔴" if res['status'] == "SELL_WATCH" else "⚪"
                
                # Indicators
                rsi_val = res['indicators']['RSI']['value']
                bb_pos = "Extreme" if "3-SD" in res['indicators']['Bollinger']['position'] else \
                         "Upper" if "Above" in res['indicators']['Bollinger']['position'] else "Inside"
                
                mismatch = "YES" if res['indicators']['MACD_Mismatch']['pass'] else "NO"
                
                # Signals count
                sig_list = []
                if res['signals']['Divergence']['detected']: sig_list.append("Div")
                if res['signals']['Volume_Analysis']['detected']: sig_list.append("Vol")
                if res['signals']['Parabolic_Move']['detected']: sig_list.append("Para")
                signals_str = ",".join(sig_list) if sig_list else "-"
                
                print(f"{ticker:<15} | {status_icon} {res['status'][:4]:<9} | {rsi_val:<8} | {bb_pos:<15} | {mismatch:<10} | {signals_str}")
                
                # Print full detail only if Sell Watch
                if res['status'] == "SELL_WATCH":
                     # print(f"\n   -> {ticker} is a potential SELL candidate!")
                     pass

        except Exception as e:
            print(f"{ticker:<15} | ERROR: {str(e)[:40]}")

    print(f"{'='*80}\n")

if __name__ == "__main__":
    import json
    
    # Check if first argument is a file (ends with .txt or .json) or looks like a file path
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    
    if arg and (arg.endswith('.txt') or arg.endswith('.json') or os.path.exists(arg)):
        # Run batch analysis
        analyze_batch(arg)
    else:
        # Run single ticker analysis
        ticker = arg if arg else "RELIANCE.NS"
        
        # Suppress individual analyze_ticker print if we want clean output, 
        # but analyze_ticker currently prints "Analyzing...".
        # We might want to keep it or modify analyze_ticker to be silent via param?
        # For now, just run it.
        res = analyze_ticker(ticker)
        if res:
            print_results_table(res)
        else:
            print("Analysis failed.")
