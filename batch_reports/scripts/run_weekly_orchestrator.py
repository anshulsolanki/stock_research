"""
WEEKLY ANALYSIS ORCHESTRATOR
============================

Orchestrates the weekly analysis process:
1. Creates a new dated folder for the current week.
2. Runs the weekly analysis report.
3. Generates detailed reports for stocks picked by weekly analysis.
4. Generates detailed reports for stocks in the batch analysis list.
"""

import os
import sys
import datetime
import json
import logging

# Add parent directories to path to find other scripts
sys.path.append(os.path.dirname(__file__))

try:
    import weekly_analysis
    import stock_detailed_report
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def load_batch_tickers():
    """Load tickers from tickers_batch_analysis.json"""
    # Assuming the json is in ../../data/ relative to this script
    json_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'tickers_batch_analysis.json')
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            # JSON format is Name: Ticker
            return list(data.values())
    except Exception as e:
        print(f"Error loading batch tickers: {e}")
        return []

def run_orchestrator():
    print("Starting Weekly Analysis Orchestrator...")
    
    # 1. Create Dated Folder
    today = datetime.date.today().strftime('%Y-%m-%d')
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'reports', today)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Output directory already exists: {output_dir}")
        
    # 2. Run Weekly Analysis
    print("\n[Step 2] Running Weekly Analysis...")
    weekly_res = weekly_analysis.run_weekly_analysis(include_deepdive=False, output_dir=output_dir)
    
    weekly_picks = weekly_res.get('final_stock_list', [])
    print(f"Weekly Analysis completed. Report: {weekly_res['report_path']}")
    print(f"Stocks identified in Weekly Analysis: {weekly_picks}")
    
    # 3. Deepdive for Weekly Picks
    print("\n[Step 3] Generating Deepdive Reports for Weekly Picks...")
    for ticker in weekly_picks:
        try:
            stock_detailed_report.generate_stock_report(ticker, output_dir=output_dir)
        except Exception as e:
            print(f"Failed to generate report for {ticker}: {e}")
            
    # 4. Deepdive for Batch List
    print("\n[Step 4] Generating Deepdive Reports for Batch List...")
    batch_tickers = load_batch_tickers()
    print(f"Found {len(batch_tickers)} tickers in batch list.")
    
    # Avoid duplicate work if ticker was already processed in step 3
    processed_tickers = set(weekly_picks)
    
    for ticker in batch_tickers:
        if ticker in processed_tickers:
            print(f"Skipping {ticker} (already processed in Step 3)")
            continue
            
        try:
            stock_detailed_report.generate_stock_report(ticker, output_dir=output_dir)
            processed_tickers.add(ticker)
        except Exception as e:
            print(f"Failed to generate report for {ticker}: {e}")
            
    print(f"\nOrchestrator completed successfully. All reports saved in: {output_dir}")

if __name__ == "__main__":
    run_orchestrator()
