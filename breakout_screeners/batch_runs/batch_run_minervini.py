import matplotlib
matplotlib.use('Agg')

import argparse
import os
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# Ensure parent scripts are importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PARENT_DIR)

import minervini_screener
from matplotlib.backends.backend_pdf import PdfPages
import shutil

def run_batch():
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

    # Step 1: Load ALL data into memory once
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
            return date_str, len(results), PDF_PATH, OUTPUT_DIR
        return date_str, 0, None, None

    pdf_moves = []
    dirs_to_remove = set()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_date, dt): dt for dt in dates}
        for future in as_completed(futures):
            date_str, match_count, pdf_path, output_dir = future.result()
            if match_count > 0:
                print(f"[SUCCESS] {date_str} -> Found {match_count} matches. PDF: {os.path.basename(pdf_path)}")
                pdf_moves.append((pdf_path, os.path.join(FINAL_BATCH_DIR, os.path.basename(pdf_path))))
                dirs_to_remove.add(output_dir)
            else:
                print(f"[SUCCESS] {date_str} -> 0 matches.")

    # Execute File/Folder Consolidation
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

    print(f"\n============================================================")
    print(f"In-Memory Batch Run Completed")
    print(f"Final Output Folder: {FINAL_BATCH_DIR}")
    print(f"============================================================")

if __name__ == "__main__":
    run_batch()
