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
STOCKS IN SECTOR ANALYSIS MODULE
=================================

PURPOSE:
--------
Performs Relative Strength (RS) analysis for all stocks within a selected sector,
comparing each stock's performance against the sector's benchmark index.

This module reuses the core analysis logic from sector_analysis.py but applies it
at the stock level rather than the sector level.

USAGE:
------
From web app:
    from stock_in_sector_analysis import run_analysis
    result = run_analysis(sector_name='FMCG', show_plot=False)

Returns:
    dict: {
        'success': bool,
        'results': pd.DataFrame with stock metrics,
        'figure': matplotlib figure,
        'sector_name': str,
        'sector_index': str,
        'error': str (if failed)
    }
"""

import os
import json
import pandas as pd
from sector_analysis import perform_rs_analysis, plot_rs_trends


def run_analysis(sector_name, show_plot=False):
    """
    Runs RS Analysis for all stocks in a given sector.
    
    Args:
        sector_name (str): Name of the sector (e.g., 'FMCG', 'IT', 'Bank')
        show_plot (bool): If True, displays the plot. If False, returns the figure object.
        
    Returns:
        dict: Analysis results containing:
            - 'success': bool
            - 'results': pd.DataFrame (the RS table for stocks)
            - 'figure': matplotlib.figure.Figure
            - 'sector_name': str
            - 'sector_index': str
            - 'error': str (if failed)
    """
    try:
        # Load configurations from JSON file
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '..', 'data', 'tickers_grouped.json'),
            os.path.join(os.path.dirname(__file__), 'data', 'tickers_grouped.json'),
            '/Users/solankianshul/Documents/projects/stock_research/data/tickers_grouped.json'
        ]
        
        config_file = None
        for path in possible_paths:
            if os.path.exists(path):
                config_file = path
                break
                
        if not config_file:
            return {'success': False, 'error': 'Configuration file tickers_grouped.json not found.'}
            
        with open(config_file, 'r') as f:
            configs = json.load(f)
        
        # Validate sector name
        if sector_name not in configs:
            available_sectors = [k for k in configs.keys() if k != 'Sector']
            return {
                'success': False, 
                'error': f"Sector '{sector_name}' not found. Available sectors: {', '.join(available_sectors)}"
            }
        
        # Get sector configuration
        sector_config = configs[sector_name]
        index_symbol = sector_config['index_symbol']
        stocks = sector_config['stocks']
        
        if not stocks:
            return {'success': False, 'error': f'No stocks found for sector {sector_name}'}
        
        print(f"Analyzing {len(stocks)} stocks in {sector_name} sector against {index_symbol}...")
        
        # Run RS Analysis using the existing function from sector_analysis
        results, data = perform_rs_analysis(index_symbol, stocks, include_technical=True)
        
        # Generate Plot
        fig = plot_rs_trends(data, index_symbol, stocks, lookback_days=365, show_plot=show_plot)
        
        return {
            'success': True,
            'results': results,
            'figure': fig,
            'sector_name': sector_name,
            'sector_index': index_symbol
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    # Test the module
    print("Testing Stocks in Sector Analysis...")
    result = run_analysis('FMCG', show_plot=True)
    
    if result['success']:
        print("\n✅ Analysis successful!")
        print(f"Sector: {result['sector_name']}")
        print(f"Index: {result['sector_index']}")
        print(f"\nTop 5 stocks by score:")
        print(result['results'][['1M', '3M', '6M', '1Y', 'Score']].head())
    else:
        print(f"\n❌ Analysis failed: {result['error']}")
