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
Minervini VCP In-Memory Batch Screener Orchestrator

This script optimizes backtesting workflows by orchestrating the Mark Minervini VCP strategy
iteratively across a chronological range of business days.

Execution Optimization & Strategy:
----------------------------------
- In-Memory Processing: Pre-loads all historical stock datasets into RAM once, entirely eliminating
  redundant disk IO operations during date iterations.
- Parallel Processing: Utilizes ThreadPoolExecutor to concurrently execute up to 4 independent daily 
  runs to maximize scanning throughput.
- Precise Date Simulation: Generates business day index ranges (skipping weekends) and slices data
  accurately to mimic end-of-day market states.
- Output Consolidation: Gathers successful individual daily PDF reports and csv recommendations into
  a single timestamped unified output folder while cleaning up temporary artifacts.

Usage:
------
python batch_run_minervini.py --start-date DD-MM-YYYY --end-date DD-MM-YYYY [--limit N] [--use-fundamentals] [--use-volume-dryup]
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import os
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import shutil

# Ensure parent scripts are importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PARENT_DIR)

import minervini_screener
from matplotlib.backends.backend_pdf import PdfPages

def run_batch():
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="Minervini VCP Screener In-Memory Batch Orchestrator")
    parser.add_argument('--start-date', type=str, required=True, help="Start date (DD-MM-YYYY or YYYY-MM-DD)")
    parser.add_argument('--end-date', type=str, required=True, help="End date (DD-MM-YYYY or YYYY-MM-DD)")
    parser.add_argument('--limit', type=int, help="Limit number of stocks to scan")
    parser.add_argument('--use-fundamentals', action='store_true', default=True, help="Enable fundamental and RS filters")
    parser.add_argument('--use-volume-dryup', action='store_true', default=False, help="Enable volume dry-up filter during contractions")
    
    args = parser.parse_args()

    def parse_date(date_str):
        for fmt in ("%d-%m-%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"Time data '{date_str}' does not match formats DD-MM-YYYY or YYYY-MM-DD")

    try:
        start_dt = parse_date(args.start_date)
        end_dt = parse_date(args.end_date)
    except ValueError as e:
        print(f"Error: {e}")
        return

    if start_dt > end_dt:
        print("Error: --start-date cannot be after --end-date")
        return

    minervini_screener.setup_directories()
    tickers = minervini_screener.load_tickers(args.limit)

    # --- 2. In-Memory Data Loading ---
    preloaded_data = {}
    minervini_screener.console.print(f"[bold blue]Pre-loading data for {len(tickers)} stocks into memory...[/bold blue]")
    for ticker in tickers:
        data = minervini_screener.fetch_data(ticker, refresh=False)
        if data:
            preloaded_data[ticker] = data

    dates = pd.date_range(start=start_dt, end=end_dt, freq='B')
    
    print(f"============================================================")
    print(f"Starting In-Memory Minervini VCP Batch Screener")
    print(f"Period: {start_dt.strftime('%d-%m-%Y')} to {end_dt.strftime('%d-%m-%Y')}")
    print(f"Total Trading Days to Scan: {len(dates)}")
    print(f"============================================================")

    BATCH_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    FINAL_BATCH_DIR = os.path.join(minervini_screener.BASE_DIR, 'screener_results', 'minervini_breakouts', f"batch_run_{BATCH_TIMESTAMP}")
    os.makedirs(FINAL_BATCH_DIR, exist_ok=True)

    def process_date(dt):
        date_str = dt.strftime('%Y-%m-%d')
        rs_ratings = {}
        if args.use_fundamentals:
            rs_ratings = minervini_screener.calculate_rs_ratings(tickers, end_date=date_str, preloaded_data=preloaded_data)
        
        results = []
        for ticker in tickers:
            data = preloaded_data.get(ticker)
            if data:
                rs = rs_ratings.get(ticker, 0) if args.use_fundamentals else 0
                match = minervini_screener.check_criteria(data, ticker, rs, use_fundamentals=args.use_fundamentals, use_volume_dryup=args.use_volume_dryup, end_date=date_str)
                if match:
                    results.append((match, data))
        
        if results:
            TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            import time
            time.sleep(1.1) 
            
            OUTPUT_DIR = os.path.join(minervini_screener.BASE_DIR, 'screener_results', 'minervini_breakouts', TIMESTAMP)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            PDF_PATH = os.path.join(OUTPUT_DIR, f"minervini_Screener_Results_{date_str}.pdf")
            
            with PdfPages(PDF_PATH) as pdf:
                minervini_screener.create_pdf_title_page(pdf, date_str, "Unknown")
                minervini_screener.render_pdf_documentation_page(pdf)
                df_results = pd.DataFrame([r[0] for r in results])
                minervini_screener.render_pdf_styled_table(pdf, df_results, f"Minervini Results {date_str}")
                
                for match, data in results:
                    minervini_screener.generate_chart(data['history'], match['Ticker'], match, pdf=pdf, end_date=date_str)
                    
                df_results.to_csv(os.path.join(OUTPUT_DIR, 'results.csv'), index=False)
            return date_str, len(results), PDF_PATH, OUTPUT_DIR, [r[0] for r in results]
        return date_str, 0, None, None, []

    pdf_moves = []
    dirs_to_remove = set()
    all_recommendations = []

    # --- 3. Parallel Execution Loop ---
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_date, dt): dt for dt in dates}
        for future in as_completed(futures):
            date_str, match_count, pdf_path, output_dir, daily_matches = future.result()
            if match_count > 0:
                print(f"[SUCCESS] {date_str} -> Found {match_count} matches. PDF: {os.path.basename(pdf_path)}")
                pdf_moves.append((pdf_path, os.path.join(FINAL_BATCH_DIR, os.path.basename(pdf_path))))
                dirs_to_remove.add(output_dir)
                all_recommendations.extend(daily_matches)
            else:
                print(f"[SUCCESS] {date_str} -> 0 matches.")

    # --- 4. File & Directory Consolidation ---
    if pdf_moves:
        print(f"\nConsolidating all PDFs into: {FINAL_BATCH_DIR}...")
        for src, dst in pdf_moves:
            shutil.move(src, dst)
            
        print("Cleaning up temporary daily run folders...")
        for folder in dirs_to_remove:
            try:
                shutil.rmtree(folder)
            except Exception as e:
                print(f"Warning: Failed to remove {folder}. {e}")

    if all_recommendations:
        df_consolidated = pd.DataFrame(all_recommendations)
        cols = ['Ticker', 'Date', 'Close', 'Pivot', 'Target', 'Stop_Loss', 'Risk_Reward', 'Qtr_EPS%', 'Qtr_Sales%', 'RS_Rating', 'Vol_Ratio', 'Pullbacks']
        valid_cols = [c for c in cols if c in df_consolidated.columns]
        df_consolidated = df_consolidated[valid_cols]
        
        df_consolidated['Pullbacks'] = df_consolidated['Pullbacks'].apply(lambda x: str(x) if isinstance(x, list) else x)
        df_consolidated = df_consolidated.drop_duplicates().sort_values(by=['Date', 'Ticker'])
        
        csv_path = os.path.join(FINAL_BATCH_DIR, "consolidated_results.csv")
        df_consolidated.to_csv(csv_path, index=False)
        print(f"Saved consolidated recommendations list to: {csv_path}")

    print(f"\n============================================================")
    print(f"In-Memory Batch Run Completed")
    print(f"Final Output Folder: {FINAL_BATCH_DIR}")
    print(f"============================================================")

if __name__ == "__main__":
    run_batch()
