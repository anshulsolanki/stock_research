# -------------------------------------------------------------------------------
# Project: Stock Analysis (https://github.com/anshulsolanki/stock_analysis)
# Author:  Anshul Solanki
# License: MIT License
# -------------------------------------------------------------------------------

"""
Master Bear Market Stock Screener

This script runs both fundamental and technical bear market screeners sequentially
and saves their reports to a shared, timestamped folder.

Usage:
------
python3 bear_market_combined_screener.py [--limit N] [--ticker TICKER] [--refresh]
"""

import os
import argparse
from datetime import datetime
import json

# Import the screeners
import bear_market_fundamental_screener as bm_fundamental
import bear_market_technical_screener as bm_technical

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), 'data')
JSON_PATH = os.path.join(DATA_DIR, 'nifty_500.json')

def load_tickers(limit=None):
    """
    Loads tickers from the Nifty 500 JSON file.
    
    Args:
        limit (int, optional): Maximum number of tickers to return.
        
    Returns:
        list: List of ticker symbols.
    """
    try:
        with open(JSON_PATH, 'r') as f:
            data = json.load(f)
        tickers = list(data.values())
        if limit:
            return tickers[:limit]
        return tickers
    except FileNotFoundError:
        print(f"Error: {JSON_PATH} not found.")
        return []

def main():
    """
    Main execution function for the master screener.
    
    Parses command-line arguments, generates a common timestamp for the run,
    creates the shared output directory, and sequence-calls both the 
    fundamental and technical screeners.
    """
    parser = argparse.ArgumentParser(description="Master Bear Market Stock Screener")
    parser.add_argument('--limit', type=int, help="Limit number of stocks to scan")
    parser.add_argument('--ticker', type=str, help="Scan a single specific ticker")
    parser.add_argument('--refresh', action='store_true', help="Force refresh of cached data")
    args = parser.parse_args()

    # Generate a common timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Define the shared output directory
    output_dir = os.path.join(os.path.dirname(BASE_DIR), 'screener_results', 'bear_market', timestamp)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting Master Screener...")
    print(f"Results will be saved to: {output_dir}\n")

    if args.ticker:
        tickers = [args.ticker]
    else:
        tickers = load_tickers(args.limit)

    if not tickers:
        print("No tickers to scan.")
        return

    print("==================================================")
    print("Running Fundamental Screener...")
    print("==================================================")
    try:
        bm_fundamental.run_screener(tickers, refresh=args.refresh, output_dir=output_dir)
    except Exception as e:
        print(f"Error running fundamental screener: {e}")

    print("\n==================================================")
    print("Running Technical Screener...")
    print("==================================================")
    try:
        bm_technical.run_screener(tickers, refresh=args.refresh, output_dir=output_dir)
    except Exception as e:
        print(f"Error running technical screener: {e}")

    print("\n==================================================")
    print("Running Famous Trader Screeners...")
    print("==================================================")
    try:
        import famous_trader_screeners
        famous_trader_screeners.run_screener(tickers, refresh=args.refresh, output_dir=output_dir)
    except Exception as e:
        print(f"Error running famous trader screeners: {e}")

    print(f"\nMaster Screener execution complete. Reports are in {output_dir}")

if __name__ == "__main__":
    main()
