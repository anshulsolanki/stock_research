import yfinance as yf
import pandas as pd
import json
import os
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data')
CACHE_DIR = os.path.join(BASE_DIR, 'data_cache')
JSON_PATH = os.path.join(DATA_DIR, 'nifty_500.json')
NSE_BASELINE = "^NSEI"

# Simple Console Class to replace Rich
class Console:
    def print(self, text, end='\n'):
        # Strip rich tags for clean output
        clean_text = text.replace("[bold green]", "").replace("[/bold green]", "")
        clean_text = clean_text.replace("[green]", "").replace("[/green]", "")
        clean_text = clean_text.replace("[red]", "").replace("[/red]", "")
        clean_text = clean_text.replace("[yellow]", "").replace("[/yellow]", "")
        clean_text = clean_text.replace("[cyan]", "").replace("[/cyan]", "")
        clean_text = clean_text.replace("[magenta]", "").replace("[/magenta]", "")
        clean_text = clean_text.replace("[blue]", "").replace("[/blue]", "")
        clean_text = clean_text.replace("[bold]", "").replace("[/bold]", "")
        clean_text = clean_text.replace("[dim]", "").replace("[/dim]", "")
        print(clean_text, end=end)

console = Console()

def setup_directories():
    os.makedirs(CACHE_DIR, exist_ok=True)

def load_tickers(limit=None):
    if not os.path.exists(JSON_PATH):
        console.print(f"[red]Error: {JSON_PATH} not found.[/red]")
        return []
    try:
        with open(JSON_PATH, 'r') as f:
            data = json.load(f)
        tickers = list(data.values())
        if limit:
            return tickers[:limit]
        return tickers
    except Exception as e:
        console.print(f"[red]Error loading tickers: {e}[/red]")
        return []

def process_ticker(ticker):
    """
    Downloads:
    1. Info (metadata) -> {ticker}_info.json
    2. Daily Data (2y) -> {ticker}_1d.csv (For VCP & CANSLIM)
    3. Weekly Data (3y) -> {ticker}_1wk.csv (For Breakout Screener)
    """
    try:
        daily_path = os.path.join(CACHE_DIR, f"{ticker}_1d.csv")
        weekly_path = os.path.join(CACHE_DIR, f"{ticker}_1wk.csv")
        info_path = os.path.join(CACHE_DIR, f"{ticker}_info.json")
        
        stock = yf.Ticker(ticker)
        
        # 1. Info
        try:
            info = stock.info
            with open(info_path, 'w') as f:
                json.dump(info, f)
        except Exception:
            pass

        # 2. Daily Data (2y)
        df_daily = stock.history(period="2y")
        if not df_daily.empty:
            df_daily.to_csv(daily_path)
        
        # 3. Weekly Data (3y)
        df_weekly = stock.history(period="3y", interval="1wk")
        if not df_weekly.empty:
            df_weekly.to_csv(weekly_path)
            
        return True, ticker
    except Exception as e:
        return False, f"{ticker}: {str(e)}"

def process_baseline():
    """Download baseline data for RS calculation"""
    try:
        console.print(f"[blue]Fetching baseline {NSE_BASELINE}...[/blue]")
        cache_path = os.path.join(CACHE_DIR, "NSEI_baseline.csv")
        stock = yf.Ticker(NSE_BASELINE)
        df = stock.history(period="1y")
        if not df.empty:
            df.to_csv(cache_path)
            console.print(f"[green]Baseline downloaded.[/green]")
        else:
            console.print(f"[yellow]Baseline fetch returned empty.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error fetching baseline: {e}[/red]")

def main():
    parser = argparse.ArgumentParser(description="Unified Data Downloader for Stock Screeners")
    parser.add_argument('--limit', type=int, help="Limit number of stocks to download")
    parser.add_argument('--workers', type=int, default=10, help="Number of concurrent download threads")
    args = parser.parse_args()
    
    setup_directories()
    
    tickers = load_tickers(args.limit)
    if not tickers:
        console.print("[yellow]No tickers to process.[/yellow]")
        return
        
    process_baseline()
    
    console.print(f"[bold cyan]Starting download for {len(tickers)} tickers with {args.workers} workers...[/bold cyan]")
    
    success_count = 0
    failure_count = 0
    failures = []
    
    # Simple progress tracking without rich
    start_time = time.time()
    total = len(tickers)
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_ticker = {executor.submit(process_ticker, t): t for t in tickers}
        
        for i, future in enumerate(as_completed(future_to_ticker)):
            success, msg = future.result()
            if success:
                success_count += 1
            else:
                failure_count += 1
                failures.append(msg)
            
            # Print progress every 10 items or on last item
            if (i + 1) % 10 == 0 or (i + 1) == total:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (total - (i + 1)) / rate if rate > 0 else 0
                print(f"\rProgress: {i + 1}/{total} ({((i+1)/total)*100:.1f}%) - Success: {success_count} - Failures: {failure_count} - ETA: {remaining:.0f}s", end="", flush=True)

    print() # Newline after progress
    
    console.print("\n[bold]Download Summary:[/bold]")
    console.print(f"[green]Success: {success_count}[/green]")
    if failure_count > 0:
        console.print(f"[red]Failures: {failure_count}[/red]")
        for fail in failures[:5]:
             console.print(f"[dim]{fail}[/dim]")
    
    console.print(f"\n[bold]Data saved to {CACHE_DIR}[/bold]")

if __name__ == "__main__":
    main()
