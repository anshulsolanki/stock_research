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

def check_criteria(data, ticker, rs_rating):
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
        'Close': current_close,
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
        '52W_High': high_52w
    }

def generate_chart(df, ticker, result):
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
    title = (f"{ticker} - CANSLIM Breakout Setup\n"
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
    
    results = []
    
    with console.status(f"[bold green]Scanning {len(tickers)} stocks against CANSLIM...[/bold green]") as status:
        for ticker in tickers:
            data = fetch_data(ticker, refresh=args.refresh)
            if data:
                rs = rs_ratings.get(ticker, 0) # default to 0 if not calculated
                match = check_criteria(data, ticker, rs)
                if match:
                    console.print(f"[green]FOUND: {ticker} (Qtr EPS: {match['Qtr_EPS%']}%, Ann EPS: {match['Ann_EPS%']}%, ROE: {match['ROE%']}%, RS: {match['RS_Rating']}) [/green]")
                    generate_chart(data['history'], ticker, match)
                    results.append(match)
    
    if results:
        df_results = pd.DataFrame(results)
        cols = ['Ticker', 'Date', 'Close', 'Breakout', 'Target', 'Stop_Loss', 'Risk_Reward', 'Qtr_EPS%', 'Qtr_Sales%', 'Ann_EPS%', 'ROE%', 'Inst_Own%', 'RS_Rating', 'Vol_Ratio']
        if all(c in df_results.columns for c in cols):
            df_results = df_results[cols]
            
        print("\nCANSLIM Screener Results:")
        print(df_results.to_string(index=False))
        
        df_results.to_csv(os.path.join(OUTPUT_DIR, 'results.csv'), index=False)
        console.print(f"\n[bold]Results saved to {OUTPUT_DIR}/results.csv[/bold]")
    else:
        console.print("[yellow]No stocks found matching CANSLIM criteria.[/yellow]")

if __name__ == "__main__":
    main()
