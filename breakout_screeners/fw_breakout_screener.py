import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import argparse
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import argparse
from datetime import datetime, timedelta
# from rich.console import Console
# from rich.table import Table
# from rich.progress import match_styles
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Initialize Console (Dummy wrapper)
class Console:
    def print(self, text):
        # Remove rich tags for simple print
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
OUTPUT_DIR = os.path.join(BASE_DIR, 'screener_results', 'fw_breakouts', TIMESTAMP)
CHARTS_DIR = os.path.join(OUTPUT_DIR, 'charts')
CACHE_DIR = os.path.join(BASE_DIR, 'data_cache')
JSON_PATH = os.path.join(DATA_DIR, 'nifty_500.json')

# Strategy Parameters
MIN_HISTORY_YEARS = 3
LOOKBACK_WEEKS = 10  # For Breakout Level and Rolling Max
NATR_CONSOLIDATION_WEEKS = 6 # Minimum weeks of consolidation (NATR < 8)
NATR_PERIOD = 14
NATR_THRESHOLD = 8.0
SMA_PERIOD = 20
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Breakout Criteria
WICK_RATIO_THRESHOLD = 0.50
GAIN_MIN = 0.05
GAIN_MAX = 0.20
VOLUME_MULTIPLIER = 1.30

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

def fetch_data(ticker, refresh=False):
    cache_path = os.path.join(CACHE_DIR, f"{ticker}_1wk.csv")
    
    try:
        # Check cache first
        if not refresh and os.path.exists(cache_path):
            # console.print(f"[dim]Loading {ticker} from cache[/dim]")
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return df

        # Fetch weekly data
        stock = yf.Ticker(ticker)
        df = stock.history(period=f"{MIN_HISTORY_YEARS}y", interval="1wk")
        
        # Ensure enough data
        if len(df) < 52: # Need at least a year for valid indicator calculation context
            return None
            
        # Save to cache
        df.to_csv(cache_path)
        
        return df
    except Exception as e:
        # console.print(f"[yellow]Error fetching {ticker}: {e}[/yellow]")
        return None

def calculate_indicators(df):
    df = df.copy()
    
    # 1. SMA 20
    df['SMA20'] = df['Close'].rolling(window=SMA_PERIOD).mean()
    
    # 2. MACD
    ema_12 = df['Close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    # Note: MACD calculation in pandas varies slightly from some platforms due to initial values, but this is standard.
    df['Signal'] = df['MACD'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    
    # 3. ATR & NATR
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=NATR_PERIOD).mean()
    df['NATR'] = (df['ATR'] / df['Close']) * 100
    
    # NEW: Consolidated NATR Check
    # Ensure NATR has been < 8 for at least NATR_CONSOLIDATION_WEEKS
    # We check the MAX NATR over the consolidation window.
    df['MaxNATR'] = df['NATR'].rolling(window=NATR_CONSOLIDATION_WEEKS).max()

    # 4. Rolling Max High (Resistance) - Previous 10 weeks (shifted)
    # We want max high of [t-10 ... t-1], excluding t (current candle)
    df['RollingMaxHigh'] = df['High'].shift(1).rolling(window=LOOKBACK_WEEKS).max()
    df['RollingMaxClose'] = df['Close'].shift(1).rolling(window=LOOKBACK_WEEKS).max()
    # RollingMinLow for visual context (Consolidation Box Low)
    df['RollingMinLow'] = df['Low'].shift(1).rolling(window=LOOKBACK_WEEKS).min() 
    
    return df

def check_criteria(df, ticker):
    if df is None or len(df) < SMA_PERIOD:
        return None
    
    # Get latest candle (Breakout Candle)
    # Note: If running on weekend, -1 is the last full week. 
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Criteria 1: Trend (Price > 20 SMA)
    if not (curr['Close'] > curr['SMA20']):
        return None
        
    # Criteria 2: Momentum (MACD > Signal)
    if not (curr['MACD'] > curr['Signal']):
        return None
        
    # Criteria 3: Consolidation (NATR < 8 for consecutive weeks)
    # We check prev['MaxNATR'] because check_criteria is called on the LATEST candle (curr).
    # We want the consolidation to have happened BEFORE the breakout candle (curr).
    # So we check if the PREVIOUS candle (prev) had a MaxNATR (over last 6 weeks) < 8.
    if not (prev['MaxNATR'] < NATR_THRESHOLD):
        return None

    # Criteria 4: Breakout Execution
    if not (curr['Close'] > curr['RollingMaxHigh']):
        return None
    if not (curr['Close'] > curr['RollingMaxClose']):
        return None
        
    # Criteria 5: Breakout Candle Structure
    candle_range = curr['High'] - curr['Low']
    if candle_range == 0: return None
    
    upper_wick = curr['High'] - max(curr['Open'], curr['Close'])
    wick_ratio = upper_wick / candle_range
    
    if not (wick_ratio < WICK_RATIO_THRESHOLD):
        return None
        
    # Breakout Size: 5% < Gain < 20%
    prev_close = prev['Close']
    gain_pct = (curr['Close'] - prev_close) / prev_close
    
    if not (GAIN_MIN < gain_pct < GAIN_MAX):
        return None
        
    # Criteria 6: Volume
    if not (curr['Volume'] > prev['Volume'] * VOLUME_MULTIPLIER):
        return None
        
    return {
        'Ticker': ticker,
        'Date': df.index[-1].strftime('%Y-%m-%d'),
        'Close': curr['Close'],
        'Gain%': round(gain_pct * 100, 2),
        'Volume_Mult': round(curr['Volume'] / prev['Volume'], 2),
        'NATR': round(prev['NATR'], 2),
        'Breakout_Level': curr['RollingMaxHigh'],
        'SMA20': curr['SMA20'],
        'Entry_Price': curr['Close'],
        'Stop_Loss': curr['Low'],
        'Risk%': round((curr['Close'] - curr['Low']) / curr['Close'] * 100, 2)
    }

def generate_chart(df, ticker, result):
    # Slice last 52 weeks for visibility
    plot_df = df.tail(52)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Plot Candles
    width = 0.6
    width2 = 0.05
    
    up = plot_df[plot_df.Close >= plot_df.Open]
    down = plot_df[plot_df.Close < plot_df.Open]
    
    # Up candles
    ax1.bar(up.index, up.Close - up.Open, width, bottom=up.Open, color='green', alpha=0.8)
    ax1.bar(up.index, up.High - up.Close, width2, bottom=up.Close, color='green')
    ax1.bar(up.index, up.Low - up.Open, width2, bottom=up.Open, color='green')
    
    # Down candles
    ax1.bar(down.index, down.Close - down.Open, width, bottom=down.Open, color='red', alpha=0.8)
    ax1.bar(down.index, down.High - down.Open, width2, bottom=down.Open, color='red')
    ax1.bar(down.index, down.Low - down.Close, width2, bottom=down.Close, color='red')
    
    # Plot SMA 20
    ax1.plot(plot_df.index, plot_df['SMA20'], color='blue', label='SMA 20', linewidth=1.5)
    
    # Plot Resistance Line (Breakout Level)
    # Using the breakout level from the result (which is RollingMaxHigh of the breakout candle)
    breakout_level = result['Breakout_Level']
    ax1.axhline(y=breakout_level, color='purple', linestyle='--', label=f'Resistance ({breakout_level:.2f})')
    
    # Plot Stop Loss Line
    stop_loss = result['Stop_Loss']
    ax1.axhline(y=stop_loss, color='red', linestyle=':', label=f'Stop Loss ({stop_loss:.2f})')
    
    # Annotate Entry and Stop
    ax1.text(plot_df.index[-1], result['Entry_Price'], f"Entry: {result['Entry_Price']:.2f}", 
             color='green', fontsize=10, verticalalignment='bottom', fontweight='bold')
    ax1.text(plot_df.index[-1], stop_loss, f"Stop: {stop_loss:.2f}", 
             color='red', fontsize=10, verticalalignment='top', fontweight='bold')
    
    # Plot Consolidation Box
    # We'll draw it for the LOOKBACK_WEEKS period ending at the previous candle
    # Box High = RollingMaxHigh, Box Low = RollingMinLow
    last_idx = plot_df.index.get_loc(plot_df.index[-1])
    try:
        # Get bounds based on index
        start_date = plot_df.index[-LOOKBACK_WEEKS - 1]
        end_date = plot_df.index[-2] # Previous candle
        
        # Recalculate box limits for this specific window to be precise
        box_window = df.iloc[-LOOKBACK_WEEKS-1:-1] # -1 excludes current
        box_high = breakout_level
        box_low = box_window['Low'].min()
        
        # Create Rectangle Patch
        # Matplotlib dates handling can be tricky, using index numbers might be safer but dates work if formatted?
        # Let's use axvspan limits or simply plotting a rectangle requires numerical coordinates
        # Simplified: Just draw lines or using a span
        
        rect_start = up.index[-LOOKBACK_WEEKS-1] # Approx
        # Better: use ax1.add_patch with date2num if needed, but let's stick to simple Hlines/Vlines or Span
        
        # High/Low Box lines
        ax1.hlines(y=[box_high, box_low], xmin=start_date, xmax=end_date, colors='orange', linestyles='dotted', linewidth=2)
        ax1.vlines(x=[start_date, end_date], ymin=box_low, ymax=box_high, colors='orange', linestyles='dotted', linewidth=2)
        
    except Exception as e:
        print(f"Error drawing box: {e}")

    ax1.set_title(f"{ticker} - Weekly Breakout Setup\nEntry: {result['Entry_Price']:.2f}, Stop: {result['Stop_Loss']:.2f}, Risk: {result['Risk%']}%", fontsize=14)
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Volume
    colors = ['green' if c >= o else 'red' for c, o in zip(plot_df.Close, plot_df.Open)]
    ax2.bar(plot_df.index, plot_df.Volume, color=colors, width=0.6, alpha=0.6)
    
    # Volume Threshold Line (1.3x Prev Volume - dynamic? No, just show recent volume)
    # We can highlight the breakout volume
    breakout_vol = plot_df.iloc[-1]['Volume']
    prev_vol = plot_df.iloc[-2]['Volume']
    threshold = prev_vol * VOLUME_MULTIPLIER
    ax2.axhline(y=threshold, color='orange', linestyle='--', label='1.3x Vol Threshold')
    
    ax2.set_ylabel("Volume")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filename = f"{ticker.replace('.NS', '')}_breakout.png"
    save_path = os.path.join(CHARTS_DIR, filename)
    plt.savefig(save_path)
    plt.close(fig)
    return save_path

def main():
    parser = argparse.ArgumentParser(description="Stock Breakout Screener")
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
        
    results = []
    
    with console.status(f"[bold green]Scanning {len(tickers)} stocks...[/bold green]") as status:
        for ticker in tickers:
            df = fetch_data(ticker, refresh=args.refresh)
            if df is not None:
                # Calculate
                df = calculate_indicators(df)
                
                # Check
                match = check_criteria(df, ticker)
                if match:
                    console.print(f"[green]FOUND: {ticker} - Gain: {match['Gain%']}%[/green]")
                    generate_chart(df, ticker, match)
                    results.append(match)
                    
    # Output
    if results:
        # Table
        df_results = pd.DataFrame(results)
        # Reorder columns
        cols = ['Ticker', 'Date', 'Close', 'Entry_Price', 'Stop_Loss', 'Risk%', 'Gain%', 'Volume_Mult', 'NATR']
        if all(c in df_results.columns for c in cols):
             df_results = df_results[cols]
             
        print("\nBreakout Screener Results:")
        print(df_results.to_string(index=False))
        
        # CSV
        df_results.to_csv(os.path.join(OUTPUT_DIR, 'results.csv'), index=False)
        console.print(f"\n[bold]Results saved to {OUTPUT_DIR}/results.csv[/bold]")
        console.print(f"[bold]Charts saved to {CHARTS_DIR}/[/bold]")
    else:
        console.print("[yellow]No stocks found criteria matching criteria.[/yellow]")

if __name__ == "__main__":
    main()
