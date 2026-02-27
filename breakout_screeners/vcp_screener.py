# Imports
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import find_peaks

# Initialize Console
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
OUTPUT_DIR = os.path.join(BASE_DIR, 'screener_results', 'mark_minervini_vcp_breakouts', TIMESTAMP)
CHARTS_DIR = os.path.join(OUTPUT_DIR, 'charts')
CACHE_DIR = os.path.join(BASE_DIR, 'data_cache')
JSON_PATH = os.path.join(DATA_DIR, 'nifty_500.json')

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
    hist_cache_path = os.path.join(CACHE_DIR, f"{ticker}_1d.csv")
    try:
        if not refresh and os.path.exists(hist_cache_path):
            df = pd.read_csv(hist_cache_path, index_col=0, parse_dates=True)
            return df
        
        stock = yf.Ticker(ticker)
        # Need enough history for 200 SMA and Trend Template
        df = stock.history(period="2y") 
        if len(df) < 200: return None
        df.to_csv(hist_cache_path)
        return df
    except Exception as e:
        # console.print(f"[yellow]Error fetching {ticker}: {e}[/yellow]")
        return None

def check_trend_template(df):
    """
    Phase 1: The Macro Setup (Minervini's Trend Template)
    """
    # Calculate MAs
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['SMA150'] = df['Close'].rolling(window=150).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    df['52W_Low'] = df['Low'].rolling(window=252).min()
    df['52W_High'] = df['High'].rolling(window=252).max()
    
    curr = df.iloc[-1]
    
    # 1. Price > SMA50 > SMA150 > SMA200
    if not (curr['Close'] > curr['SMA50'] and curr['SMA50'] > curr['SMA150'] and curr['SMA150'] > curr['SMA200']):
        return False
        
    # 2. SMA200 Rising (vs 20 days ago)
    # We check if current SMA200 > SMA200 1 month ago (approx 20 trading days)
    if len(df) < 220: return False # Safety
    try:
        sma200_prev = df['SMA200'].iloc[-20]
        if curr['SMA200'] <= sma200_prev:
            return False
    except:
        return False
        
    # 3. Price > 1.3 * 52W Low (30% above low)
    if curr['Close'] < (curr['52W_Low'] * 1.30):
        return False
        
    # 4. Price within 25% of 52W High (Price >= 0.75 * High)
    if curr['Close'] < (curr['52W_High'] * 0.75):
        return False
        
    return True

def find_contractions(df):
    """
    Phase 2: Identifying Contractions (The Swings)
    Returns a list of (Peak, Trough, Depth%) tuples for the identified contractions.
    Focus on the last 3-6 months (~90-120 days).
    """
    # Analyze last 130 days to catch base formation
    window_days = 130
    if len(df) < window_days: return []
    
    subset = df.iloc[-window_days:].copy()
    prices = subset['High'].values
    
    # Find Peaks
    # prominence=0.02 means peak must be distinguishable by 2% drop on either side (tunable)
    # distance=5 means peaks at least 5 days apart
    peaks, _ = find_peaks(prices, distance=5, prominence=prices.max()*0.02) 
    
    contractions = []
    
    # Iterate through peaks to find subsequent troughs
    for p_idx in peaks:
        peak_price = prices[p_idx]
        peak_date = subset.index[p_idx]
        
        # Look for trough AFTER peak
        # Slice from peak to end (or next peak?? VCP usually Peak -> Trough -> Peak...)
        # We define a contraction as Peak -> Lowest Low before next Peak or Current Price
        
        # Simple approach: Identify significant peaks. For each peak, find the lowest low between that peak and the next peak (or end).
        
        # We need to pair peaks. But VCP is Peak1 -> Trough1 -> Peak2 -> Trough2 ...
        pass
    
    # Alternative robust approach for VCP specifically:
    # 1. Macro High (Left of base)
    # 2. Current Price (Right edge)
    # 3. Intermediate Highs
    
    # Let's simple "Swing Highs" approach
    # We want to see decreasing depths.
    # Pattern: Peak A (Depth 20%) -> Peak B (Depth 10%) -> Peak C (Depth 5%) -> Breakout
    
    # Implementation:
    # Get all potential peaks. Filter for "Base Highs".
    # Assume 3-4 contractions max usually.
    
    # Let's iterate right-to-left?
    # Right-most peak is the Pivot.
    # Previous peak is Left Side of contraction.
    
    return [] # Placeholder

def check_vcp_pattern(df, ticker):
    """
    Phase 2 & 3 & 4
    """
    if not check_trend_template(df):
        return None
        
    # VCP Identification
    # Let's try a simplified robust logic:
    # 1. Find recent Pivot (Local High in last ~20 days)
    # 2. Check if we are near it (Breakout trigger)
    # 3. Look backward for previous larger contraction?
    
    # Explicit Rules from Prompt:
    # "find local highs and lows... 60 to 90 day window"
    subset = df.iloc[-90:].copy()
    highs = subset['High'].values
    lows = subset['Low'].values
    closes = subset['Close'].values
    
    # Find Peaks (Local Maxima)
    peak_idxs, _ = find_peaks(highs, distance=10, width=3)
    if len(peak_idxs) < 2: return None # Need at least 2 peaks for a contraction sequence?
    
    # We need contractions.
    # Contraction = (Peak - Subsequent Trough) / Peak
    
    # Let's analyze the sequence of contractions
    # We need to pair each Peak with the minimum Low found BEFORE the next Peak.
    
    contractions = [] # List of depths
    
    for i in range(len(peak_idxs)):
        start_idx = peak_idxs[i]
        end_idx = peak_idxs[i+1] if i+1 < len(peak_idxs) else len(subset)-1
        
        # Find Trough in this segment
        segment_lows = lows[start_idx:end_idx]
        if len(segment_lows) == 0: continue
            
        trough_val = segment_lows.min()
        peak_val = highs[start_idx]
        
        depth = (peak_val - trough_val) / peak_val
        contractions.append(depth)
    
    # VCP Rule: Contractions must get smaller (from left to right)
    # Example: [0.25, 0.15, 0.08]
    # We allow some tolerance, but generally decreasing.
    
    # We need at least 2 contractions typically? Or 2-6.
    if len(contractions) < 2: return None
    
    # Check decreasing trend
    # e.g. Depth(i) < Depth(i-1)
    is_vcp = True
    for i in range(1, len(contractions)):
        if contractions[i] >= contractions[i-1]:
            # Allow slight violation? Strict VCP says "progressively smaller".
            # Let's reject if it GROWS significantly. Equal or slightly more might be noise?
            # Strict rule: C_current < C_prev
            is_vcp = False
            break
            
    if not is_vcp: return None
    
    # Phase 3: Volume Dry-Up
    # "Average daily volume during final contraction significantly below SMA50"
    # Final contraction segment:
    last_peak_idx = peak_idxs[-1]
    final_segment_vol = subset['Volume'].iloc[last_peak_idx:].mean()
    
    # Calculate Volume SMA50 (from full df)
    vol_sma50 = df['Volume'].iloc[-50:].mean()
    
    # "Significantly below" -> let's say < 75% or 80% of SMA50?
    if final_segment_vol > (vol_sma50 * 0.9): 
        return None # Volume not dry enough? Tune this. Prompt says "significantly below".
        
    # Phase 4: Pivot Breakout
    # Pivot Point = Peak of Final Contraction
    pivot_price = highs[peak_idxs[-1]]
    
    curr = df.iloc[-1]
    close_price = curr['Close']
    curr_vol = curr['Volume']
    
    # Entry Trigger: Close > Pivot
    if close_price <= pivot_price:
        return None # Not triggered yet?
        # Or maybe it triggered today?
    
    # Volume Confirmation: Breakout volume > 1.5 * SMA50
    if curr_vol <= (vol_sma50 * 1.5):
        return None
        
    return {
        'Ticker': ticker,
        'Date': df.index[-1].strftime('%Y-%m-%d'),
        'Close': close_price,
        'Pivot': pivot_price,
        'Contractions': [round(c*100, 1) for c in contractions],
        'Vol_Bit': round(curr_vol/vol_sma50, 2),
        'Dry_Up': round(final_segment_vol/vol_sma50, 2)
    }

def generate_chart(df, ticker, result):
    # Plot last 6 months
    plot_df = df.tail(130)
    
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
    
    # Pivot Line
    ax1.axhline(y=result['Pivot'], color='purple', linestyle='--', label=f"Pivot: {result['Pivot']:.2f}")
    
    # Title
    cts = ", ".join([f"{c}%" for c in result['Contractions']])
    ax1.set_title(f"{ticker} - VCP Breakout\nContractions: {cts} | Vol Surge: {result['Vol_Bit']}x", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Volume
    colors = ['green' if c >= o else 'red' for c, o in zip(plot_df.Close, plot_df.Open)]
    ax2.bar(plot_df.index, plot_df.Volume, color=colors, width=0.6, alpha=0.6)
    
    # Vol SMA
    vol_sma = df['Volume'].rolling(window=50).mean().tail(130)
    ax2.plot(vol_sma.index, vol_sma, color='orange', label='Vol SMA 50')
    
    ax2.legend()
    plt.tight_layout()
    
    filename = f"{ticker.replace('.NS', '')}_vcp.png"
    save_path = os.path.join(CHARTS_DIR, filename)
    plt.savefig(save_path)
    plt.close(fig)
    return save_path

def main():
    parser = argparse.ArgumentParser(description="Mark Minervini VCP Screener")
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
    
    with console.status(f"[bold green]Scanning {len(tickers)} stocks for VCP...[/bold green]") as status:
        for ticker in tickers:
            df = fetch_data(ticker, refresh=args.refresh)
            if df is not None:
                match = check_vcp_pattern(df, ticker)
                if match:
                    console.print(f"[green]FOUND: {ticker} - Pivot: {match['Pivot']}[/green]")
                    generate_chart(df, ticker, match)
                    results.append(match)
    
    if results:
        df_results = pd.DataFrame(results)
        print("\nVCP Screener Results:")
        print(df_results.to_string(index=False))
        df_results.to_csv(os.path.join(OUTPUT_DIR, 'results.csv'), index=False)
        console.print(f"\n[bold]Results saved to {OUTPUT_DIR}/results.csv[/bold]")
    else:
        console.print("[yellow]No VCP candidates found.[/yellow]")

if __name__ == "__main__":
    main()
