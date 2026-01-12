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
RELATIVE STRENGTH (RS) ANALYSIS TOOL
=====================================

PURPOSE:
--------
This module performs comprehensive Relative Strength (RS) analysis to identify stocks that are 
outperforming or underperforming relative to a benchmark index (e.g., NIFTY 50, Nifty IT). 
RS analysis is a powerful technique used in technical analysis to identify:
- Strong momentum stocks consistently outperforming the market
- Emerging winners showing early signs of strength
- Early turnaround candidates showing improving relative performance

WHAT IT DOES:
-------------
1. **Multi-Timeframe RS Calculation**: Computes RS ratios across 7 timeframes (1M, 2M, 3M, 6M, 1Y, 3Y, 5Y)
   - RS Ratio = (Stock Performance) / (Index Performance)
   - RS > 1.0 indicates stock is outperforming the index
   - RS < 1.0 indicates stock is underperforming the index

2. **Momentum Pattern Analysis**:
   - **Consistent**: Stocks with sustained outperformance (3M, 6M, 1Y RS all > 1)
   - **Emerging**: Stocks with accelerating RS (1M > 2M > 3M, showing improving strength)
   - **Slowing**: Stocks with decelerating RS (1M < 2M < 3M, showing weakening strength)

3. **Early Turnaround Detection**:
   Identifies stocks that may be reversing from underperformance, based on:
   - Short-term RS rising (Emerging momentum)
   - Medium-term RS still below benchmark (historically lagging)
   - Absolute performance improving (3M returns > 6M returns)
   
4. **Technical Confirmation Filters**:
   - **MA Breakout**: Price above both 50-day and 200-day moving averages
   - **Volume Surge**: Current volume > 1.5x the 20-day average (indicates strong interest)

5. **Scoring & Ranking**: 
   Assigns scores based on multiple factors to rank stocks by overall strength:
   - Consistent outperformance: +1 point
   - Emerging momentum: +1 point
   - Early turnaround signal: +2 points
   - Plus the 1-month RS ratio value
   
6. **Visualization**: 
   Plots RS trend charts showing how each stock's relative strength has evolved over time

METHODOLOGY:
------------
RS Ratio Calculation:
- Normalize stock and index prices to the start of each period
- RS = (Normalized Stock Price) / (Normalized Index Price)
- A rising RS line indicates the stock is gaining relative strength
- A falling RS line indicates the stock is losing relative strength

KEY METRICS:
------------
- RS Ratio: The relative performance multiplier (>1 is outperformance)
- Consistent: Boolean flag for sustained multi-timeframe outperformance
- Emerging: Boolean flag for accelerating momentum
- Early_Turnaround: Boolean flag for potential reversal opportunities
- MA_Breakout: Boolean flag for bullish technical setup
- Volume_Surge: Boolean flag for strong buying interest
- Score: Composite score for overall ranking

CONFIGURATION:
--------------
Reads stock lists and index symbols from 'data/tickers_grouped.json' with structure:
{
    "IT": {
        "index_symbol": "^CNXIT",
        "stocks": {
            "TCS": "TCS.NS",
            "Infosys": "INFY.NS",
            ...
        }
    },
    ...
}

USAGE:
------
Run as standalone script:
    python rs_analysis.py
    
Or import and use programmatically:
    from rs_analysis import perform_rs_analysis
    results, data = perform_rs_analysis(index_symbol="^NSEI", stocks=stock_dict)

OUTPUT:
-------
- Pandas DataFrame with RS ratios, momentum flags, technical filters, and scores
- Console output showing sorted results by score
- Optional: Visual chart showing RS trend evolution over time
- Optional: Excel export of results

TYPICAL USE CASES:
------------------
1. Sector rotation analysis: Which IT stocks are outperforming the Nifty IT index?
2. Stock selection: Find the strongest momentum stocks for potential investment
3. Early opportunity detection: Identify underperformers showing early reversal signs
4. Portfolio monitoring: Track how holdings are performing relative to the benchmark
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
from datetime import datetime, timedelta

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def fetch_data(symbols, period="5y"):
    """
    Fetches historical data for a list of symbols.
    """
    # yfinance download returns a MultiIndex DataFrame if multiple symbols are passed
    # We need to handle this correctly
    data = yf.download(symbols, period=period, progress=False, auto_adjust=False, multi_level_index=False)["Close"].ffill()
    
    # If only one symbol, it returns a Series or single-column DF, ensure it's a DF
    if isinstance(data, pd.Series):
        data = data.to_frame()
        
    return data

def fetch_volume_data(symbols, period="5y"):
    """
    Fetches volume data for technical confirmation.
    """
    data = yf.download(symbols, period=period, progress=False, auto_adjust=False, multi_level_index=False)["Volume"].ffill()
    return data

def calculate_rs_table(data, index_symbol, stocks, time_frames):
    """
    Calculates the Relative Strength table.
    """
    rs_table = pd.DataFrame(index=stocks.keys())
    today = data.index.max()

    for label, offset in time_frames.items():
        start_date = today - offset
        # Ensure start_date is within data range
        if start_date < data.index.min():
            continue
            
        df_slice = data[data.index >= start_date]
        if df_slice.empty:
            continue
            
        # Normalize to start of period
        df_norm = df_slice / df_slice.iloc[0]
        
        # Calculate RS Ratio: Stock / Index
        rs_values = {}
        for stock_name, symbol in stocks.items():
            if symbol in df_norm.columns and index_symbol in df_norm.columns:
                rs_values[stock_name] = df_norm[symbol].iloc[-1] / df_norm[index_symbol].iloc[-1]
            else:
                rs_values[stock_name] = np.nan
                
        rs_table[label] = pd.Series(rs_values)

    return rs_table.round(3)

def analyze_momentum(rs_table):
    """
    Adds momentum flags (Consistent, Emerging, Slowing).
    """
    # Consistent long-term outperformers (6M & 1Y RS > 1)
    if '3M' in rs_table.columns and '6M' in rs_table.columns and '1Y' in rs_table.columns:
        rs_table['Consistent'] = (rs_table['3M'] > 1) & (rs_table['6M'] > 1) & (rs_table['1Y'] > 1)
    else:
        rs_table['Consistent'] = False

    # Emerging short-term RS (1M > 2M > 3M)
    if '1M' in rs_table.columns and '2M' in rs_table.columns and '3M' in rs_table.columns:
        rs_table['Emerging'] = (rs_table['1M'] > rs_table['2M']) & (rs_table['2M'] > rs_table['3M'])
        rs_table['Slowing'] = (rs_table['1M'] < rs_table['2M']) & (rs_table['2M'] < rs_table['3M'])
    else:
        rs_table['Emerging'] = False
        rs_table['Slowing'] = False
        
    return rs_table

def analyze_turnaround(rs_table, data, stocks):
    """
    Identifies early turnaround signals.
    """
    # 1. Short-term RS rising (Emerging)
    short_term_rs_rising = rs_table['Emerging']

    # 2. Medium-term RS <= 1 (sector lagging)
    if '6M' in rs_table.columns and '1Y' in rs_table.columns:
        medium_term_lagging = (rs_table['6M'] <= 1) | (rs_table['1Y'] <= 1)
    else:
        medium_term_lagging = False

    # 3. Absolute performance improving (normalized return 3M > 6M)
    absolute_perf = pd.Series(index=stocks.keys(), dtype=bool)
    
    for stock, symbol in stocks.items():
        if symbol not in data.columns:
            absolute_perf[stock] = False
            continue
            
        # Approx trading days: 3M ~ 63, 6M ~ 126
        if len(data) > 126:
            perf_3M = data[symbol].iloc[-1] / data[symbol].iloc[-63] - 1
            perf_6M = data[symbol].iloc[-1] / data[symbol].iloc[-126] - 1
            absolute_perf[stock] = perf_3M > perf_6M
        else:
            absolute_perf[stock] = False

    rs_table['Absolute_Perf_Improving'] = absolute_perf
    rs_table['Early_Turnaround'] = short_term_rs_rising & medium_term_lagging & absolute_perf
    
    return rs_table

def add_technical_filters(rs_table, data, volume_data, stocks):
    """
    Adds MA Breakout and Volume Surge checks.
    """
    ma_breakout = pd.Series(index=stocks.keys(), dtype=bool)
    vol_surge = pd.Series(index=stocks.keys(), dtype=bool)
    
    for stock, symbol in stocks.items():
        if symbol not in data.columns:
            continue
            
        close = data[symbol]
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()
        
        if len(close) > 200:
            ma_breakout[stock] = (close.iloc[-1] > ma50.iloc[-1]) & (close.iloc[-1] > ma200.iloc[-1])
        else:
            ma_breakout[stock] = False

        if volume_data is not None and symbol in volume_data.columns:
            volume = volume_data[symbol]
            vol_avg = volume.rolling(20).mean()
            if len(volume) > 20:
                vol_surge[stock] = volume.iloc[-1] > 1.5 * vol_avg.iloc[-1]
            else:
                vol_surge[stock] = False
        else:
            vol_surge[stock] = False
            
    rs_table['MA_Breakout'] = ma_breakout
    rs_table['Volume_Surge'] = vol_surge
    
    return rs_table

def calculate_score(rs_table):
    """
    Calculates an overall score for ranking.
    """
    rs_table['Score'] = 0
    rs_table.loc[rs_table['Consistent'], 'Score'] += 1
    rs_table.loc[rs_table['Emerging'], 'Score'] += 1
    rs_table.loc[rs_table['Early_Turnaround'], 'Score'] += 2
    
    if '1M' in rs_table.columns:
        rs_table['Score'] += rs_table['1M']
        
    return rs_table.sort_values(by='Score', ascending=False)

def perform_rs_analysis(index_symbol, stocks, include_technical=True):
    """
    Main orchestration function for RS Analysis.
    """
    print("Fetching data...")
    all_symbols = [index_symbol] + list(stocks.values())
    
    # Remove duplicates if any
    all_symbols = list(set(all_symbols))
    
    data = fetch_data(all_symbols)
    
    volume_data = None
    if include_technical:
        volume_data = fetch_volume_data(all_symbols)

    time_frames = {
        "1M": pd.DateOffset(months=1),
        "2M": pd.DateOffset(months=2),
        "3M": pd.DateOffset(months=3),
        "6M": pd.DateOffset(months=6),
        "1Y": pd.DateOffset(years=1),
        "3Y": pd.DateOffset(years=3),
        "5Y": pd.DateOffset(years=5)
    }

    print("Calculating Relative Strength...")
    rs_table = calculate_rs_table(data, index_symbol, stocks, time_frames)
    
    print("Analyzing Momentum...")
    rs_table = analyze_momentum(rs_table)
    
    print("Identifying Turnarounds...")
    rs_table = analyze_turnaround(rs_table, data, stocks)
    
    if include_technical:
        print("Adding Technical Filters...")
        rs_table = add_technical_filters(rs_table, data, volume_data, stocks)
        
    rs_table = calculate_score(rs_table)
    
    return rs_table, data

def plot_rs_trends(data, index_symbol, stocks, lookback_days=365, show_plot=True):
    """
    Plots the Relative Strength trends over time.
    """
    fig = plt.figure(figsize=(14, 8))
    
    # Filter data for lookback period
    start_date = data.index.max() - timedelta(days=lookback_days)
    subset = data[data.index >= start_date].copy()
    
    # Normalize Index
    if index_symbol not in subset.columns:
        print(f"Index {index_symbol} not found in data for plotting.")
        return None

    # Avoid division by zero or empty slice
    if subset.empty:
        print("No data to plot.")
        return None

    index_series = subset[index_symbol] / subset[index_symbol].iloc[0]
    
    for stock_name, symbol in stocks.items():
        if symbol == index_symbol:
            continue # Don't plot index against itself
            
        if symbol in subset.columns:
            # Normalize Stock
            stock_series = subset[symbol] / subset[symbol].iloc[0]
            
            # Calculate RS Ratio
            rs_ratio = stock_series / index_series
            
            # Highlight strong/weak lines
            linewidth = 1.5
            if rs_ratio.iloc[-1] > 1.1: linewidth = 2.5
            
            plt.plot(subset.index, rs_ratio, label=stock_name, linewidth=linewidth)
            
            # Add label at the end of the line
            if not rs_ratio.empty:
                plt.text(subset.index[-1], rs_ratio.iloc[-1], f" {stock_name}", fontsize=9, verticalalignment='center')
            
    plt.axhline(1.0, color='black', linestyle='--', linewidth=2, label='Benchmark')
    plt.title(f'Relative Strength vs {index_symbol} (Last {lookback_days} Days)')
    plt.xlabel('Date')
    plt.ylabel('RS Ratio (>1 = Outperforming)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    if show_plot:
        plt.show()
        
    return fig


def run_analysis(show_plot=False):
    """
    Runs the Sector Analysis for the web app.
    
    Args:
        show_plot (bool): If True, displays the plot. If False, returns the figure object.
        
    Returns:
        dict: Analysis results containing:
            - 'success': bool
            - 'results': pd.DataFrame (the RS table)
            - 'figure': matplotlib.figure.Figure
            - 'error': str (if failed)
    """
    import os
    import json
    
    try:
        # Load configurations from JSON file
        # Try multiple paths to find the file
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
        
        # Select "Sector" Configuration
        selected_config = "Sector" 
        
        if selected_config not in configs:
            return {'success': False, 'error': f"Configuration '{selected_config}' not found in {config_file}"}
            
        config = configs[selected_config]
        index_symbol = config['index_symbol']
        stocks = config['stocks']

        # Run Analysis
        results, data = perform_rs_analysis(index_symbol, stocks, include_technical=True)
        
        # Generate Plot
        fig = plot_rs_trends(data, index_symbol, stocks, show_plot=show_plot)
        
        return {
            'success': True,
            'results': results,
            'figure': fig
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    import os
    import json
    
    # Load configurations from JSON file
    config_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'tickers_grouped.json')
    
    if not os.path.exists(config_file):
        # Try local path
        config_file = os.path.join(os.path.dirname(__file__), 'data', 'tickers_grouped.json')
        
    if not os.path.exists(config_file):
        print(f"Error: Configuration file {config_file} not found.")
        exit(1)
        
    with open(config_file, 'r') as f:
        configs = json.load(f)
    
    # Select Configuration to run
    selected_config = "Sector" 
    
    if selected_config not in configs:
        print(f"Error: Configuration '{selected_config}' not found in {config_file}")
        exit(1)
        
    config = configs[selected_config]
    index_symbol = config['index_symbol']
    stocks = config['stocks']

    print(f"Running RS Analysis for '{selected_config}' against {index_symbol}...")
    try:
        results, data = perform_rs_analysis(index_symbol, stocks, include_technical=True)
        
        print("\n📊 Relative Strength Analysis Results (Sorted by Score):")
        
        # Select columns to display
        display_cols = ['1M', '3M', '6M', '1Y', 'Consistent', 'Emerging', 'Early_Turnaround', 'MA_Breakout', 'Volume_Surge', 'Score']
        # Filter columns that exist
        display_cols = [c for c in display_cols if c in results.columns]
        
        print(results[display_cols])
        
        # Plot Trends
        print("\nPlotting RS Trends...")
        plot_rs_trends(data, index_symbol, stocks)
        
    except Exception as e:
        print(f"Error: {e}")