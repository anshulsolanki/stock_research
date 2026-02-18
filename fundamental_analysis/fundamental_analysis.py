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
FUNDAMENTAL ANALYSIS MODULE
============================

PURPOSE:
--------
This module provides comprehensive fundamental analysis for stocks, answering key questions
about financial health and growth trends. Each function is designed to be called independently
via API for modular analysis.

FEATURES:
---------
1. Long-term Analysis (4 years):
   - Revenue growth trends and 3Y growth rate
   - Profit growth trends and 3Y growth rate
   - ROE (Return on Equity) growth trends and 3Y growth rate
   - EPS (Earnings Per Share) growth trends and 3Y growth rate
   - PE ratio comparison vs industry

2. Short-term Analysis (6 quarters):
   - Quarterly revenue growth trends
   - Quarterly profit growth trends
   - Quarterly ROE growth trends
   - Quarterly EPS growth trends

USAGE:
------
Each function can be called independently:

    from fundamental_analysis import analyze_revenue_growth_4y
    
    result = analyze_revenue_growth_4y(ticker="AAPL")
    print(result)

API INTEGRATION:
----------------
All functions return standardized dictionary responses suitable for API serialization.

DEPENDENCIES:
-------------
- yfinance: For fetching fundamental data
- pandas: For data manipulation
- numpy: For numerical calculations
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration
FUNDAMENTAL_CONFIG = {
    # Execution Control
    'DEFAULT_TICKER': 'DABUR.NS',
    'BATCH_RELATIVE_PATH': '../data/tickers_list.json',
    'RUN_BATCH': False
}


import time

# ============================================================================
# HELPER FUNCTIONS FOR DATA FETCHING
# ============================================================================

def fetch_financials(ticker, retries=3, delay=1):
    """
    Fetch annual financial statements for a ticker.
    Includes retry logic to handle network issues or rate limits.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    retries : int
        Number of retries
    delay : int
        Delay between retries in seconds
        
    Returns:
    --------
    dict
        Dictionary containing income_statement, balance_sheet, cashflow, and info
    """
    attempt = 0
    last_error = None
    
    while attempt <= retries:
        try:
            stock = yf.Ticker(ticker)
            # Accessing properties triggers the download
            return {
                'income_statement': stock.financials,
                'balance_sheet': stock.balance_sheet,
                'cashflow': stock.cashflow,
                'info': stock.info,
                'quarterly_income': stock.quarterly_financials,
                'quarterly_balance': stock.quarterly_balance_sheet
            }
        except Exception as e:
            last_error = e
            attempt += 1
            if attempt <= retries:
                time.sleep(delay * attempt) # Exponential backoffish
            
    return {'error': str(last_error)}


def calculate_growth_rate(values_dict, periods=['1Y', '3Y']):
    """
    Calculate growth rates for different time periods.
    
    Parameters:
    -----------
    values_dict : dict
        Dictionary with year keys and values
    periods : list
        List of periods to calculate growth for (1Y, 3Y)
        
    Returns:
    --------
    dict
        Growth rates for each period
    """
    growth_rates = {}
    sorted_years = sorted(values_dict.keys(), reverse=True)
    
    # Filter out None values
    sorted_years = [year for year in sorted_years if values_dict[year] is not None]
    
    if len(sorted_years) < 2:
        return {'error': 'Insufficient data for growth calculation'}
    
    current_value = values_dict[sorted_years[0]]
    
    for period in periods:
        if period == '1Y' and len(sorted_years) >= 2:
            past_value = values_dict[sorted_years[1]]
            if past_value and past_value != 0:
                growth_rates['1Y'] = ((current_value - past_value) / abs(past_value)) * 100
                
        elif period == '3Y':
            # Use index 3 if available (4 years of data = 3 years difference)
            # Otherwise use the oldest available data
            if len(sorted_years) >= 4:
                idx = 3
            elif len(sorted_years) >= 3:
                idx = 2
            else:
                idx = len(sorted_years) - 1
            
            past_value = values_dict[sorted_years[idx]]
            if past_value and past_value != 0:
                years_diff = sorted_years[0] - sorted_years[idx]
                if years_diff > 0:
                    growth_rates['3Y'] = ((current_value / past_value) ** (1/years_diff) - 1) * 100
                    

    
    # Set to 0 if not calculated
    for period in periods:
        if period not in growth_rates:
            growth_rates[period] = 0
    
    return growth_rates


def check_growth_trend(growth_rates):
    """
    Check if growth is accelerating (1Y > 3Y > 4Y).
    
    Parameters:
    -----------
    growth_rates : dict
        Dictionary with growth rates for different periods
        
    Returns:
    --------
    bool
        True if trend is accelerating
    """
    if '1Y' in growth_rates and '3Y' in growth_rates:
        return growth_rates['1Y'] > growth_rates['3Y']
    return False


# ============================================================================
# LONG-TERM ANALYSIS FUNCTIONS (4 YEARS)
# ============================================================================

def analyze_revenue_growth_4y(ticker, data=None):
    """
    Analyze revenue growth over the last 4 years.
    
    Checks:
    - Is revenue growing? (1Y > 3Y trend)
    - What is the 3-year CAGR?
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    data : dict, optional
        Pre-fetched financial data to avoid redundant API calls
        
    Returns:
    --------
    dict
        {
            'success': bool,
            'ticker': str,
            'metric': 'Revenue',
            'period': '4Y',
            'is_growing': bool,
            'has_accelerating_trend': bool,
            'growth_1y': float,
            'growth_3y_cagr': float,
            
            'revenue_history': dict,
            'analysis_date': str
        }
    """
    try:
        if data is None:
            data = fetch_financials(ticker)
        
        if 'error' in data:
            return {'success': False, 'error': data['error']}
        
        income_stmt = data['income_statement']
        
        # Get Total Revenue
        if 'Total Revenue' in income_stmt.index:
            revenue_row = income_stmt.loc['Total Revenue']
        else:
            return {'success': False, 'error': 'Revenue data not found'}
        
        # Build revenue history
        revenue_history = {}
        for date in revenue_row.index:
            year = date.year
            revenue_history[year] = float(revenue_row[date]) if pd.notna(revenue_row[date]) else None
        
        # Calculate growth rates
        growth_rates = calculate_growth_rate(revenue_history)
        
        if 'error' in growth_rates:
            return {'success': False, 'error': growth_rates['error']}
        
        # Check if accelerating
        is_accelerating = check_growth_trend(growth_rates)
        
        # Check if growing overall
        is_growing = growth_rates.get('3Y', 0) > 0
        
        return {
            'success': True,
            'ticker': ticker,
            'metric': 'Revenue',
            'period': '4Y',
            'is_growing': is_growing,
            'has_accelerating_trend': is_accelerating,
            'growth_1y': round(growth_rates.get('1Y', 0), 2),
            'growth_3y_cagr': round(growth_rates.get('3Y', 0), 2),
            'revenue_history': revenue_history,
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


def analyze_profit_growth_4y(ticker, data=None):
    """
    Analyze profit growth over the last 4 years.
    
    Checks:
    - Is profit growing? (1Y > 3Y trend)
    - What is the 3-year CAGR?
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    data : dict, optional
        Pre-fetched financial data to avoid redundant API calls
        
    Returns:
    --------
    dict
        {
            'success': bool,
            'ticker': str,
            'metric': 'Net Income',
            'period': '4Y',
            'is_growing': bool,
            'has_accelerating_trend': bool,
            'growth_1y': float,
            'growth_3y_cagr': float,
            
            'profit_history': dict,
            'analysis_date': str
        }
    """
    try:
        if data is None:
            data = fetch_financials(ticker)
        
        if 'error' in data:
            return {'success': False, 'error': data['error']}
        
        income_stmt = data['income_statement']
        
        # Get Net Income
        if 'Net Income' in income_stmt.index:
            profit_row = income_stmt.loc['Net Income']
        else:
            return {'success': False, 'error': 'Net Income data not found'}
        
        # Build profit history
        profit_history = {}
        for date in profit_row.index:
            year = date.year
            profit_history[year] = float(profit_row[date]) if pd.notna(profit_row[date]) else None
        
        # Calculate growth rates
        growth_rates = calculate_growth_rate(profit_history)
        
        if 'error' in growth_rates:
            return {'success': False, 'error': growth_rates['error']}
        
        # Check if accelerating
        is_accelerating = check_growth_trend(growth_rates)
        
        # Check if growing overall
        is_growing = growth_rates.get('3Y', 0) > 0
        
        return {
            'success': True,
            'ticker': ticker,
            'metric': 'Net Income',
            'period': '4Y',
            'is_growing': is_growing,
            'has_accelerating_trend': is_accelerating,
            'growth_1y': round(growth_rates.get('1Y', 0), 2),
            'growth_3y_cagr': round(growth_rates.get('3Y', 0), 2),
            'profit_history': profit_history,
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


def analyze_roe_growth_4y(ticker, data=None):
    """
    Analyze ROE (Return on Equity) growth over the last 4 years.
    
    Checks:
    - Is ROE growing? (1Y > 3Y trend)
    - What is the 3-year CAGR?
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    data : dict, optional
        Pre-fetched financial data to avoid redundant API calls
        
    Returns:
    --------
    dict
        {
            'success': bool,
            'ticker': str,
            'metric': 'ROE',
            'period': '4Y',
            'is_growing': bool,
            'has_accelerating_trend': bool,
            'growth_1y': float,
            'growth_3y_cagr': float,
            
            'roe_history': dict,
            'analysis_date': str
        }
    """
    try:
        if data is None:
            data = fetch_financials(ticker)
        
        if 'error' in data:
            return {'success': False, 'error': data['error']}
        
        income_stmt = data['income_statement']
        balance_sheet = data['balance_sheet']
        
        # Get Net Income and Stockholders Equity
        if 'Net Income' not in income_stmt.index or 'Stockholders Equity' not in balance_sheet.index:
            return {'success': False, 'error': 'Required data for ROE calculation not found'}
        
        net_income = income_stmt.loc['Net Income']
        equity = balance_sheet.loc['Stockholders Equity']
        
        # Calculate ROE for each year
        roe_history = {}
        for date in net_income.index:
            year = date.year
            if date in equity.index:
                ni_val = float(net_income[date]) if pd.notna(net_income[date]) else None
                eq_val = float(equity[date]) if pd.notna(equity[date]) else None
                
                if ni_val is not None and eq_val is not None and eq_val != 0:
                    roe_history[year] = (ni_val / eq_val) * 100
        
        if len(roe_history) < 2:
            return {'success': False, 'error': 'Insufficient ROE data'}
        
        # Calculate growth rates
        growth_rates = calculate_growth_rate(roe_history)
        
        if 'error' in growth_rates:
            return {'success': False, 'error': growth_rates['error']}
        
        # Check if accelerating
        is_accelerating = check_growth_trend(growth_rates)
        
        # Check if growing overall
        is_growing = growth_rates.get('3Y', 0) > 0
        
        return {
            'success': True,
            'ticker': ticker,
            'metric': 'ROE',
            'period': '4Y',
            'is_growing': is_growing,
            'has_accelerating_trend': is_accelerating,
            'growth_1y': round(growth_rates.get('1Y', 0), 2),
            'growth_3y_cagr': round(growth_rates.get('3Y', 0), 2),
            'roe_history': {k: round(v, 2) for k, v in roe_history.items()},
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


def analyze_eps_growth_4y(ticker, data=None):
    """
    Analyze EPS (Earnings Per Share) growth over the last 4 years.
    
    Checks:
    - Is EPS growing? (1Y > 3Y trend)
    - What is the 3-year CAGR?
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    data : dict, optional
        Pre-fetched financial data to avoid redundant API calls
        
    Returns:
    --------
    dict
        {
            'success': bool,
            'ticker': str,
            'metric': 'EPS',
            'period': '4Y',
            'is_growing': bool,
            'has_accelerating_trend': bool,
            'growth_1y': float,
            'growth_3y_cagr': float,
            
            'eps_history': dict,
            'analysis_date': str
        }
    """
    try:
        if data is None:
            data = fetch_financials(ticker)
        
        if 'error' in data:
            return {'success': False, 'error': data['error']}
        
        income_stmt = data['income_statement']
        
        # Get Basic EPS
        if 'Basic EPS' in income_stmt.index:
            eps_row = income_stmt.loc['Basic EPS']
        elif 'Diluted EPS' in income_stmt.index:
            eps_row = income_stmt.loc['Diluted EPS']
        else:
            return {'success': False, 'error': 'EPS data not found'}
        
        # Build EPS history (filter out None values)
        eps_history = {}
        for date in eps_row.index:
            year = date.year
            if pd.notna(eps_row[date]):
                eps_history[year] = float(eps_row[date])
        
        # Calculate growth rates
        growth_rates = calculate_growth_rate(eps_history)
        
        if 'error' in growth_rates:
            return {'success': False, 'error': growth_rates['error']}
        
        # Check if accelerating
        is_accelerating = check_growth_trend(growth_rates)
        
        # Check if growing overall
        is_growing = growth_rates.get('3Y', 0) > 0
        
        return {
            'success': True,
            'ticker': ticker,
            'metric': 'EPS',
            'period': '4Y',
            'is_growing': is_growing,
            'has_accelerating_trend': is_accelerating,
            'growth_1y': round(growth_rates.get('1Y', 0), 2),
            'growth_3y_cagr': round(growth_rates.get('3Y', 0), 2),
            'eps_history': {k: round(v, 2) for k, v in eps_history.items()},
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


def analyze_pe_vs_industry(ticker, data=None):
    """
    Analyze PE ratio compared to industry average.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    data : dict, optional
        Pre-fetched financial data to avoid redundant API calls
        
    Returns:
    --------
    dict
        {
            'success': bool,
            'ticker': str,
            'metric': 'PE Ratio',
            'current_pe': float,
            'industry': str,
            'industry_pe': float,
            'vs_industry': str (e.g., '+15.2% above industry'),
            'is_overvalued': bool,
            'is_undervalued': bool,
            'analysis_date': str
        }
    """
    try:
        if data and 'info' in data:
            info = data['info']
        else:
            stock = yf.Ticker(ticker)
            info = stock.info
        
        current_pe = info.get('trailingPE', None) or info.get('forwardPE', None)
        industry = info.get('industry', 'Unknown')
        sector = info.get('sector', 'Unknown')
        
        # Try to get industry PE (not always available)
        industry_pe = None
        
        # Some stocks have industry PE in info
        if 'industryPE' in info:
            industry_pe = info['industryPE']
        
        if current_pe is None:
            return {'success': False, 'error': 'PE ratio not available'}
        
        result = {
            'success': True,
            'ticker': ticker,
            'metric': 'PE Ratio',
            'current_pe': round(current_pe, 2),
            'industry': industry,
            'sector': sector,
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        if industry_pe:
            result['industry_pe'] = round(industry_pe, 2)
            diff_pct = ((current_pe - industry_pe) / industry_pe) * 100
            
            if diff_pct > 0:
                result['vs_industry'] = f'+{abs(diff_pct):.1f}% above industry'
                result['is_overvalued'] = diff_pct > 20
                result['is_undervalued'] = False
            else:
                result['vs_industry'] = f'{diff_pct:.1f}% below industry'
                result['is_overvalued'] = False
                result['is_undervalued'] = abs(diff_pct) > 20
        else:
            result['industry_pe'] = None
            result['vs_industry'] = 'Industry PE not available'
            result['is_overvalued'] = None
            result['is_undervalued'] = None
        
        return result
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


# ============================================================================
# SHORT-TERM ANALYSIS FUNCTIONS (6 QUARTERS)
# ============================================================================

def analyze_revenue_growth_6q(ticker, data=None):
    """
    Analyze revenue growth over the last 6 quarters.
    
    Checks:
    - Is revenue growing quarter-over-quarter?
    - Growth trend analysis
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    data : dict, optional
        Pre-fetched financial data to avoid redundant API calls
        
    Returns:
    --------
    dict
        {
            'success': bool,
            'ticker': str,
            'metric': 'Revenue',
            'period': '6Q',
            'is_growing': bool,
            'recent_quarter_growth': float (QoQ %),
            'average_qoq_growth': float,
            'revenue_by_quarter': dict,
            'analysis_date': str
        }
    """
    try:
        if data is None:
            data = fetch_financials(ticker)
        
        if 'error' in data:
            return {'success': False, 'error': data['error']}
        
        quarterly_income = data['quarterly_income']
        
        # Get Total Revenue
        if 'Total Revenue' in quarterly_income.index:
            revenue_row = quarterly_income.loc['Total Revenue']
        else:
            return {'success': False, 'error': 'Quarterly revenue data not found'}
        
        # Get last 6 quarters
        quarters = revenue_row.head(6)
        
        if len(quarters) < 2:
            return {'success': False, 'error': 'Insufficient quarterly data'}
        
        # Build quarterly revenue
        revenue_by_quarter = {}
        for date in quarters.index:
            quarter_label = f"{date.year}Q{(date.month-1)//3 + 1}"
            revenue_by_quarter[quarter_label] = float(quarters[date]) if pd.notna(quarters[date]) else None
        
        # Calculate QoQ growth rates
        qoq_growth = []
        quarters_list = list(quarters)
        
        for i in range(len(quarters_list) - 1):
            current = quarters_list[i]
            previous = quarters_list[i + 1]
            
            if pd.notna(current) and pd.notna(previous) and previous != 0:
                growth = ((current - previous) / abs(previous)) * 100
                qoq_growth.append(growth)
        
        # Recent quarter growth
        recent_growth = qoq_growth[0] if qoq_growth else None
        
        # Average QoQ growth
        avg_growth = np.mean(qoq_growth) if qoq_growth else None
        
        # Is growing?
        is_growing = avg_growth > 0 if avg_growth is not None else False
        
        return {
            'success': True,
            'ticker': ticker,
            'metric': 'Revenue',
            'period': '6Q',
            'is_growing': is_growing,
            'recent_quarter_growth': round(recent_growth, 2) if recent_growth is not None else None,
            'average_qoq_growth': round(avg_growth, 2) if avg_growth is not None else None,
            'revenue_by_quarter': revenue_by_quarter,
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


def analyze_profit_growth_6q(ticker, data=None):
    """
    Analyze profit growth over the last 6 quarters.
    
    Checks:
    - Is profit growing quarter-over-quarter?
    - Growth trend analysis
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    data : dict, optional
        Pre-fetched financial data to avoid redundant API calls
        
    Returns:
    --------
    dict
        {
            'success': bool,
            'ticker': str,
            'metric': 'Net Income',
            'period': '6Q',
            'is_growing': bool,
            'recent_quarter_growth': float (QoQ %),
            'average_qoq_growth': float,
            'profit_by_quarter': dict,
            'analysis_date': str
        }
    """
    try:
        if data is None:
            data = fetch_financials(ticker)
        
        if 'error' in data:
            return {'success': False, 'error': data['error']}
        
        quarterly_income = data['quarterly_income']
        
        # Get Net Income
        if 'Net Income' in quarterly_income.index:
            profit_row = quarterly_income.loc['Net Income']
        else:
            return {'success': False, 'error': 'Quarterly profit data not found'}
        
        # Get last 6 quarters
        quarters = profit_row.head(6)
        
        if len(quarters) < 2:
            return {'success': False, 'error': 'Insufficient quarterly data'}
        
        # Build quarterly profit
        profit_by_quarter = {}
        for date in quarters.index:
            quarter_label = f"{date.year}Q{(date.month-1)//3 + 1}"
            profit_by_quarter[quarter_label] = float(quarters[date]) if pd.notna(quarters[date]) else None
        
        # Calculate QoQ growth rates
        qoq_growth = []
        quarters_list = list(quarters)
        
        for i in range(len(quarters_list) - 1):
            current = quarters_list[i]
            previous = quarters_list[i + 1]
            
            if pd.notna(current) and pd.notna(previous) and previous != 0:
                growth = ((current - previous) / abs(previous)) * 100
                qoq_growth.append(growth)
        
        # Recent quarter growth
        recent_growth = qoq_growth[0] if qoq_growth else None
        
        # Average QoQ growth
        avg_growth = np.mean(qoq_growth) if qoq_growth else None
        
        # Is growing?
        is_growing = avg_growth > 0 if avg_growth is not None else False
        
        return {
            'success': True,
            'ticker': ticker,
            'metric': 'Net Income',
            'period': '6Q',
            'is_growing': is_growing,
            'recent_quarter_growth': round(recent_growth, 2) if recent_growth is not None else None,
            'average_qoq_growth': round(avg_growth, 2) if avg_growth is not None else None,
            'profit_by_quarter': profit_by_quarter,
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


def analyze_roe_growth_6q(ticker, data=None):
    """
    Analyze ROE growth over the last 6 quarters.
    
    Checks:
    - Is ROE growing quarter-over-quarter?
    - Growth trend analysis
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    data : dict, optional
        Pre-fetched financial data to avoid redundant API calls
        
    Returns:
    --------
    dict
        {
            'success': bool,
            'ticker': str,
            'metric': 'ROE',
            'period': '6Q',
            'is_growing': bool,
            'recent_quarter_growth': float (QoQ %),
            'average_qoq_growth': float,
            'roe_by_quarter': dict,
            'analysis_date': str
        }
    """
    try:
        if data is None:
            data = fetch_financials(ticker)
        
        if 'error' in data:
            return {'success': False, 'error': data['error']}
        
        quarterly_income = data['quarterly_income']
        quarterly_balance = data['quarterly_balance']
        
        # Get Net Income and Stockholders Equity
        if 'Net Income' not in quarterly_income.index or 'Stockholders Equity' not in quarterly_balance.index:
            return {'success': False, 'error': 'Required quarterly data for ROE not found'}
        
        net_income = quarterly_income.loc['Net Income'].head(6)
        equity = quarterly_balance.loc['Stockholders Equity'].head(6)
        
        # Calculate ROE for each quarter
        roe_by_quarter = {}
        roe_values = []
        
        for date in net_income.index:
            if date in equity.index:
                ni_val = float(net_income[date]) if pd.notna(net_income[date]) else None
                eq_val = float(equity[date]) if pd.notna(equity[date]) else None
                
                if ni_val is not None and eq_val is not None and eq_val != 0:
                    roe = (ni_val / eq_val) * 100
                    quarter_label = f"{date.year}Q{(date.month-1)//3 + 1}"
                    roe_by_quarter[quarter_label] = round(roe, 2)
                    roe_values.append(roe)
        
        if len(roe_values) < 2:
            return {'success': False, 'error': 'Insufficient quarterly ROE data'}
        
        # Calculate QoQ growth rates
        qoq_growth = []
        for i in range(len(roe_values) - 1):
            current = roe_values[i]
            previous = roe_values[i + 1]
            
            if previous != 0:
                growth = ((current - previous) / abs(previous)) * 100
                qoq_growth.append(growth)
        
        # Recent quarter growth
        recent_growth = qoq_growth[0] if qoq_growth else None
        
        # Average QoQ growth
        avg_growth = np.mean(qoq_growth) if qoq_growth else None
        
        # Is growing?
        is_growing = avg_growth > 0 if avg_growth is not None else False
        
        return {
            'success': True,
            'ticker': ticker,
            'metric': 'ROE',
            'period': '6Q',
            'is_growing': is_growing,
            'recent_quarter_growth': round(recent_growth, 2) if recent_growth is not None else None,
            'average_qoq_growth': round(avg_growth, 2) if avg_growth is not None else None,
            'roe_by_quarter': roe_by_quarter,
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


def analyze_eps_growth_6q(ticker, data=None):
    """
    Analyze EPS growth over the last 6 quarters.
    
    Checks:
    - Is EPS growing quarter-over-quarter?
    - Growth trend analysis
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    data : dict, optional
        Pre-fetched financial data to avoid redundant API calls
        
    Returns:
    --------
    dict
        {
            'success': bool,
            'ticker': str,
            'metric': 'EPS',
            'period': '6Q',
            'is_growing': bool,
            'recent_quarter_growth': float (QoQ %),
            'average_qoq_growth': float,
            'eps_by_quarter': dict,
            'analysis_date': str
        }
    """
    try:
        if data is None:
            data = fetch_financials(ticker)
        
        if 'error' in data:
            return {'success': False, 'error': data['error']}
        
        quarterly_income = data['quarterly_income']
        
        # Get Basic EPS
        if 'Basic EPS' in quarterly_income.index:
            eps_row = quarterly_income.loc['Basic EPS']
        elif 'Diluted EPS' in quarterly_income.index:
            eps_row = quarterly_income.loc['Diluted EPS']
        else:
            return {'success': False, 'error': 'Quarterly EPS data not found'}
        
        # Get last 6 quarters
        quarters = eps_row.head(6)
        
        if len(quarters) < 2:
            return {'success': False, 'error': 'Insufficient quarterly EPS data'}
        
        # Build quarterly EPS
        eps_by_quarter = {}
        for date in quarters.index:
            quarter_label = f"{date.year}Q{(date.month-1)//3 + 1}"
            eps_by_quarter[quarter_label] = round(float(quarters[date]), 2) if pd.notna(quarters[date]) else None
        
        # Calculate QoQ growth rates
        qoq_growth = []
        quarters_list = list(quarters)
        
        for i in range(len(quarters_list) - 1):
            current = quarters_list[i]
            previous = quarters_list[i + 1]
            
            if pd.notna(current) and pd.notna(previous) and previous != 0:
                growth = ((current - previous) / abs(previous)) * 100
                qoq_growth.append(growth)
        
        # Recent quarter growth
        recent_growth = qoq_growth[0] if qoq_growth else None
        
        # Average QoQ growth
        avg_growth = np.mean(qoq_growth) if qoq_growth else None
        
        # Is growing?
        is_growing = avg_growth > 0 if avg_growth is not None else False
        
        return {
            'success': True,
            'ticker': ticker,
            'metric': 'EPS',
            'period': '6Q',
            'is_growing': is_growing,
            'recent_quarter_growth': round(recent_growth, 2) if recent_growth is not None else None,
            'average_qoq_growth': round(avg_growth, 2) if avg_growth is not None else None,
            'eps_by_quarter': eps_by_quarter,
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


# ============================================================================
# UNIFIED ANALYSIS FUNCTIONS (FOR API/UI INTEGRATION)
# ============================================================================

def run_analysis(ticker):
    """
    Run all fundamental analysis and return aggregated results.
    
    This is the main function for API/UI integration. It calls all 9 individual
    analysis functions and aggregates the results into a single response.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
        
    Returns:
    --------
    dict
        {
            'success': bool,
            'ticker': str,
            'long_term': {
                'revenue_4y': dict,
                'profit_4y': dict,
                'roe_4y': dict,
                'eps_4y': dict,
                'pe_ratio': dict
            },
            'short_term': {
                'revenue_6q': dict,
                'profit_6q': dict,
                'roe_6q': dict,
                'eps_6q': dict
            },
            'analysis_date': str,
            'error': str (if failed)
        }
    """
    try:
        # Pre-fetch data once for all analysis functions
        data = fetch_financials(ticker)
        
        # Check for error in data fetching
        if 'error' in data:
            return {
                'success': False,
                'ticker': ticker,
                'error': data['error'],
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        results = {
            'success': True,
            'ticker': ticker,
            'long_term': {},
            'short_term': {},
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Long-term analysis (4Y) - Pass data
        results['long_term']['revenue_4y'] = analyze_revenue_growth_4y(ticker, data=data)
        results['long_term']['profit_4y'] = analyze_profit_growth_4y(ticker, data=data)
        results['long_term']['roe_4y'] = analyze_roe_growth_4y(ticker, data=data)
        results['long_term']['eps_4y'] = analyze_eps_growth_4y(ticker, data=data)
        results['long_term']['pe_ratio'] = analyze_pe_vs_industry(ticker, data=data)
        
        # Short-term analysis (6Q) - Pass data
        results['short_term']['revenue_6q'] = analyze_revenue_growth_6q(ticker, data=data)
        results['short_term']['profit_6q'] = analyze_profit_growth_6q(ticker, data=data)
        results['short_term']['roe_6q'] = analyze_roe_growth_6q(ticker, data=data)
        results['short_term']['eps_6q'] = analyze_eps_growth_6q(ticker, data=data)
        
        return results
        
    except Exception as e:
        return {
            'success': False,
            'ticker': ticker,
            'error': str(e),
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


def analyze_multiple_tickers(tickers_list):
    """
    Run fundamental analysis on multiple tickers in parallel.
    
    Parameters:
    -----------
    tickers_list : list
        List of ticker symbols to analyze
        
    Returns:
    --------
    dict
        {
            'success': bool,
            'analyzed_count': int,
            'failed_count': int,
            'results': list of dict (one per ticker),
            'failed_tickers': list of str
        }
    """
    results = []
    failed_tickers = []
    
    print(f"Starting parallel analysis for {len(tickers_list)} tickers using ThreadPoolExecutor (max_workers=3)...")
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_ticker = {executor.submit(run_analysis, ticker): ticker for ticker in tickers_list}
        
        if tqdm:
            iterator = tqdm(as_completed(future_to_ticker), total=len(tickers_list), desc="Analyzing", unit="ticker")
        else:
            iterator = as_completed(future_to_ticker)
            
        for future in iterator:
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                if result['success']:
                    results.append(result)
                    if not tqdm:
                        print(f"  [SUCCESS] {ticker}")
                else:
                    failed_tickers.append(ticker)
                    if not tqdm:
                        print(f"  [FAILED] {ticker}: {result.get('error', 'Unknown error')}")
                    # Capture failures in results too for logging
                    results.append(result) 
            except Exception as e:
                failed_tickers.append(ticker)
                # create a dummy failure result
                results.append({'success': False, 'ticker': ticker, 'error': str(e)})
                if not tqdm:
                    print(f"  [ERROR] {ticker}: {str(e)}")
    
    # Save error log to file
    if failed_tickers:
        error_details = {}
        for res in results:
            if not res['success']:
                error_details[res['ticker']] = res.get('error', 'Unknown')
        
        try:
            with open('analysis_errors.json', 'w') as f:
                json.dump(error_details, f, indent=4)
            print(f"Error details saved to {os.path.abspath('analysis_errors.json')}")
        except Exception as e:
            print(f"Failed to save error log: {e}")

    return {
        'success': True,
        'analyzed_count': len(results),
        'failed_count': len(failed_tickers),
        'results': results,
        'failed_tickers': failed_tickers,
        'errors': failed_tickers # This is just a list of strings currently
    }


def render_fundamentals_page(pdf, ticker, result):
    """
    Renders fundamental analysis tables to a PDF page.
    Matches the style of stock_detailed_report.py
    """
    if not result['success']:
        return

    lt = result['long_term']
    st = result['short_term']
    
    # Prepare 4Y Data
    fund_4y_data = []
    metrics_4y = [
        ('Revenue', lt['revenue_4y']),
        ('Net Income', lt['profit_4y']),
        ('ROE', lt['roe_4y']),
        ('EPS', lt['eps_4y'])
    ]
    
    # Helper for safe float formatting
    def safe_fmt(val, suffix="%"):
        if val is None:
            return "N/A"
        try:
            return f"{float(val):.2f}{suffix}"
        except (ValueError, TypeError):
            return str(val)

    for name, res in metrics_4y:
        fund_4y_data.append({
            'Metric': name,
            'Is Growing?': 'Yes' if res.get('is_growing') else 'No',
            'Accelerating?': 'Yes' if res.get('has_accelerating_trend') else 'No',
            '1Y Growth': safe_fmt(res.get('growth_1y', 0)),
            '3Y CAGR': safe_fmt(res.get('growth_3y_cagr', 0))
        })
        
    # Prepare 6Q Data
    fund_6q_data = []
    metrics_6q = [
        ('Revenue', st['revenue_6q']),
        ('Net Income', st['profit_6q']),
        ('ROE', st['roe_6q']),
        ('EPS', st['eps_6q'])
    ]
    
    for name, res in metrics_6q:
        fund_6q_data.append({
            'Metric': name,
            'Is Growing?': 'Yes' if res.get('is_growing') else 'No',
            'Recent QoQ': safe_fmt(res.get('recent_quarter_growth', 0)),
            'Avg QoQ': safe_fmt(res.get('average_qoq_growth', 0))
        })

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.3})
    
    # Helper to render table
    def render_table_on_ax(ax, df, title):
        ax.axis('tight')
        ax.axis('off')
        ax.set_title(title, fontsize=16, weight='bold', pad=10, color='#1e293b')
        
        # Pre-process data (replace icons if any, though we don't use them here yet)
        table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style cells
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='#475569')
                cell.set_facecolor('#f8fafc')
                cell.set_edgecolor('#e2e8f0')
                cell.set_linewidth(1)
            else:
                cell.set_edgecolor('#e2e8f0')
                cell.set_linewidth(0.5)
                col_name = df.columns[col]
                val = df.iloc[row-1][col_name]
                
                # Apply color coding
                if isinstance(val, str):
                    val_lower = val.lower()
                    if 'yes' in val_lower or 'growing' in val_lower:
                        cell.set_text_props(color='#16a34a', weight='bold')
                    elif 'no' in val_lower or 'declining' in val_lower:
                        cell.set_text_props(color='#ef4444', weight='bold')

    # Render Tables
    if fund_4y_data:
        render_table_on_ax(ax1, pd.DataFrame(fund_4y_data), f"{ticker} - Long-term Analysis (4 Years)")
    else:
        ax1.axis('off')
        ax1.text(0.5, 0.5, "No Long-term Data Available", ha='center', va='center')
        
    if fund_6q_data:
        render_table_on_ax(ax2, pd.DataFrame(fund_6q_data), f"{ticker} - Short-term Analysis (6 Quarters)")
    else:
        ax2.axis('off')
        ax2.text(0.5, 0.5, "No Short-term Data Available", ha='center', va='center')
        
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def render_batch_summary_page(pdf, batch_results):
    """
    Renders a summary page with all stocks ranked by performance.
    Includes all Long-Term (4Y) and Short-Term (6Q) metrics.
    Consolidates Growth and Acceleration into one cell.
    """
    if not batch_results.get('results'):
        return

    # Extract and score data
    summary_data = []
    
    for res in batch_results['results']:
        if not res['success']:
            continue
            
        ticker = res['ticker']
        lt = res['long_term']
        st = res['short_term']
        
        # Calculate Score
        lt_score = 0
        lt_score += 1 if lt.get('revenue_4y', {}).get('is_growing') else 0
        lt_score += 1 if lt.get('profit_4y', {}).get('is_growing') else 0
        lt_score += 1 if lt.get('roe_4y', {}).get('is_growing') else 0
        lt_score += 1 if lt.get('eps_4y', {}).get('is_growing') else 0
        
        st_score = 0
        st_score += 1 if st.get('revenue_6q', {}).get('is_growing') else 0
        st_score += 1 if st.get('profit_6q', {}).get('is_growing') else 0
        st_score += 1 if st.get('roe_6q', {}).get('is_growing') else 0
        st_score += 1 if st.get('eps_6q', {}).get('is_growing') else 0
        
        total_score = lt_score + st_score
        
        # Helper to format cell content
        def get_status(metric_res, is_short_term=False):
            if not metric_res:
                return "N/A"
            
            is_growing = metric_res.get('is_growing', False)
            
            # Check for acceleration (ONLY for Long Term)
            is_accelerating = False
            if not is_short_term:
                # For Long Term, use the explicit flag
                is_accelerating = metric_res.get('has_accelerating_trend', False)
                acc_str = " (Acc-Yes)" if is_accelerating else " (Acc-No)"
            else:
                acc_str = ""
            
            if is_growing:
                return f"Grow{acc_str}"
            else:
                return f"Drop{acc_str}"

        # Prepare Row Data
        row = {
            'Ticker': ticker,
            'LT': lt_score,
            'ST': st_score,
            'Total': total_score, # Used for sorting/coloring but hidden later
            'Rev 4Y': get_status(lt.get('revenue_4y')),
            'Net Income 4Y': get_status(lt.get('profit_4y')),
            'ROE 4Y': get_status(lt.get('roe_4y')),
            'EPS 4Y': get_status(lt.get('eps_4y')),
            'Rev 6Q': get_status(st.get('revenue_6q'), is_short_term=True),
            'Net Income 6Q': get_status(st.get('profit_6q'), is_short_term=True),
            'ROE 6Q': get_status(st.get('roe_6q'), is_short_term=True),
            'EPS 6Q': get_status(st.get('eps_6q'), is_short_term=True),
        }
        
        summary_data.append(row)
    
    if not summary_data:
        return

    # Create DataFrame and Sort
    df = pd.DataFrame(summary_data)
    
    # Sort by Total Score (desc), then LT Score (desc), then ST Score (desc)
    df = df.sort_values(by=['Total', 'LT', 'ST'], ascending=[False, False, False])

    # Drop 'Total' column from display as requested
    display_df = df.drop(columns=['Total'])
    
    # Ensure column order for display
    cols_order = ['Ticker', 'LT', 'ST', 
                  'Rev 4Y', 'Net Income 4Y', 'ROE 4Y', 'EPS 4Y', 
                  'Rev 6Q', 'Net Income 6Q', 'ROE 6Q', 'EPS 6Q']
    
    # Filter only existing columns
    cols_final = [c for c in cols_order if c in display_df.columns]
    display_df = display_df[cols_final]
    
    # Render Table
    rows_per_page = 18 # User requested exactly 18 rows per page
    num_pages = (len(display_df) // rows_per_page) + 1
    
    for i in range(num_pages):
        start_idx = i * rows_per_page
        end_idx = min((i + 1) * rows_per_page, len(display_df))
        chunk = display_df.iloc[start_idx:end_idx]
        
        if chunk.empty:
            continue
            
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('tight')
        ax.axis('off')
        
        title = "Batch Analysis Summary - Ranked by Performance"
        if num_pages > 1:
            title += f" (Page {i+1}/{num_pages})"
            
        ax.set_title(title, fontsize=16, weight='bold', pad=20, color='#1e293b')
        
        # Define column widths
        # Ticker: 0.12, LT/ST: 0.04 each, Metrics: ~0.10 each
        col_widths = [0.12, 0.04, 0.04] + [0.1] * 8
        
        # Create table
        # Removed bbox as per user request to revert to "normal" positioning but with less rows
        table = ax.table(cellText=chunk.values, colLabels=chunk.columns, 
                        loc='center', cellLoc='center', colWidths=col_widths)
        
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.2, 2.0) # Reduced from 2.5 to 2.0 as per user request
        
        # Style cells
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                # Header
                cell.set_text_props(weight='bold', color='#475569')
                cell.set_facecolor('#f8fafc')
                cell.set_edgecolor('#e2e8f0')
                cell.set_linewidth(1)
            else:
                # Body
                cell.set_edgecolor('#e2e8f0')
                cell.set_linewidth(0.5)
                
                col_name = chunk.columns[col]
                val = chunk.iloc[row-1][col_name]
                
                # Color Coding
                if col_name in ['LT', 'ST']:
                    cell.set_text_props(weight='bold')
                
                if isinstance(val, str):
                    cell.set_text_props(color='black') # Force black text
                    if 'Grow' in val:
                        if 'Acc-Yes' in val:
                            # Grow (Acc-Yes): Dark Green (Best)
                            cell.set_facecolor('#4ade80') # Light Green for background (readability)
                            # User said "earlier decided font colors" which were dark.
                            # Black text on #15803d is unreadable.
                            # I will use the REQUESTED colors but maybe I should warn?
                            # "change the color of cell instead to the earlier decided font colors"
                            # I'll use the exact colors. If it's ugly, I'll fix in next round.
                            # Actually, let's use slightly lighter versions of "Dark Green" etc for background if we can,
                            # BUT the user was specific.
                            # Let's try to map them:
                            # Dark Green #15803d -> Background
                            cell.set_facecolor('#15803d')
                            cell.set_text_props(weight='bold')
                        else:
                            # Grow (Acc-No): Light Green
                            cell.set_facecolor('#86efac') # Green-300
                            cell.set_text_props(weight='normal')
                    elif 'Drop' in val:
                        if 'Acc-Yes' in val:
                            # Drop (Acc-Yes): Light Blue
                            cell.set_facecolor('#93c5fd') # Blue-300
                            cell.set_text_props(weight='normal')
                        else:
                            # Drop (Acc-No): Red (Worst)
                            cell.set_facecolor('#dc2626')
                            cell.set_text_props(weight='bold')

        pdf.savefig(fig)
        plt.close(fig)


# ============================================================================
# MAIN EXECUTION (FOR TESTING)
# ============================================================================

def analyze_batch(json_file):
    # import json # Already imported globally
    
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found.")
        return

    with open(json_file, 'r') as f:
        data = json.load(f)
        
    # tickers_list.json is a simple dictionary of "Name": "Ticker"
    # Or strict list if it was a list, but assuming dict values like bollinger
    if isinstance(data, dict):
        tickers = list(data.values())
    elif isinstance(data, list):
        tickers = data
    else:
        print("Error: JSON format not recognized (expected dict or list)")
        return
            
    # Remove duplicates
    tickers = list(set(tickers))
        
    print(f"Found {len(tickers)} unique tickers in {json_file}. Starting Fundamental Analysis...")
    
    # Use the existing function to process the list
    batch_results = analyze_multiple_tickers(tickers)
    
    # Save errors to a file for debugging (Redundant if analyze_multiple_tickers does it, 
    # but good to be sure or move it here if we want absolute control)
    # The analyze_multiple_tickers function handles it now.
        
    # Generate PDF Report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"fundamental_batch_report_{timestamp}.pdf"
    print(f"\nGenerating PDF Report: {pdf_filename}...")
    
    try:
        with PdfPages(pdf_filename) as pdf:
            # Title Page
            plt.figure(figsize=(11, 8.5))
            plt.axis('off')
            plt.text(0.5, 0.6, "Fundamental Analysis Batch Report", ha='center', va='center', fontsize=24, weight='bold')
            plt.text(0.5, 0.4, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ha='center', va='center', fontsize=14)
            plt.text(0.5, 0.3, f"Tickers Analyzed: {len(batch_results['results'])}", ha='center', va='center', fontsize=12)
            pdf.savefig()
            plt.close()
            
            # Summary Page (Ranked)
            render_batch_summary_page(pdf, batch_results)
            
            # Individual Ticker Pages
            for res in batch_results['results']:
                if res['success']:
                    render_fundamentals_page(pdf, res['ticker'], res)
                    print(f"  Added page for {res['ticker']}")
                    
        print(f"PDF Report saved successfully: {os.path.abspath(pdf_filename)}")
        
    except Exception as e:
        print(f"Error generating PDF report: {e}")
    
    print("\n" + "="*80)
    print("BATCH ANALYSIS COMPLETE")
    print(f"Successfully Analyzed: {batch_results['analyzed_count']}")
    print(f"Failed: {batch_results['failed_count']}")
    print("="*80)
    
    # Print a summary table of the results
    if batch_results['results']:
        print("\nSUMMARY REPORT:")
        print("-" * 110)
        headers = f"{'TICKER':<15} | {'REV (4Y)':<10} | {'PROF (4Y)':<10} | {'ROE (4Y)':<10} | {'EPS (4Y)':<10} | {'VALUATION':<15}"
        print(headers)
        print("-" * 110)
        
        for res in batch_results['results']:
            if not res.get('success', False):
                continue
                
            ticker = res['ticker']
            lt = res['long_term']
            
            # Helper to format boolean to string
            def fmt_bool(b): return "GROWING" if b else "DECLINE"
            
            # Helper to safely get metric growth status
            def get_growth(metric_dict):
                return metric_dict.get('is_growing', False) if metric_dict else False
            
            rev_g = fmt_bool(get_growth(lt.get('revenue_4y')))
            prof_g = fmt_bool(get_growth(lt.get('profit_4y')))
            roe_g = fmt_bool(get_growth(lt.get('roe_4y')))
            eps_g = fmt_bool(get_growth(lt.get('eps_4y')))
            
            # Valuation string
            pe_data = lt['pe_ratio']
            if pe_data.get('is_overvalued'):
                pe_val = "OVERVALUED"
            elif pe_data.get('is_undervalued'):
                pe_val = "UNDERVALUED"
            else:
                pe_val = "FAIR/NEUTRAL"
                
            print(f"{ticker:<15} | {rev_g:<10} | {prof_g:<10} | {roe_g:<10} | {eps_g:<10} | {pe_val:<15}")
        print("-" * 110)


# ============================================================================
# MAIN EXECUTION (FOR TESTING)
# ============================================================================

if __name__ == "__main__":
    import os
    import argparse
    import sys
    
    # Parse Command Line Arguments
    parser = argparse.ArgumentParser(description='Run Fundamental Analysis')
    parser.add_argument('batch_file', nargs='?', help='Path to JSON file containing tickers for batch analysis')
    args = parser.parse_args()
    
    # Load execution parameters from config
    run_batch = FUNDAMENTAL_CONFIG['RUN_BATCH']
    default_ticker = FUNDAMENTAL_CONFIG['DEFAULT_TICKER']
    batch_relative_path = FUNDAMENTAL_CONFIG['BATCH_RELATIVE_PATH']
    
    # Determine mode based on arguments
    if args.batch_file:
        # Batch Mode via CLI
        batch_file = args.batch_file
        if os.path.exists(batch_file):
            print(f"Running Batch Analysis from: {batch_file}")
            analyze_batch(batch_file)
        else:
            print(f"Error: Batch file not found at {batch_file}")
    elif run_batch:
        # Batch Mode via Config
        batch_file = os.path.join(os.path.dirname(__file__), batch_relative_path)
        if os.path.exists(batch_file):
            print(f"Running Configured Batch Analysis from: {batch_file}")
            analyze_batch(batch_file)
        else:
            print(f"Error: Configured batch file not found at {batch_file}")
    else:
        # 1. Single Ticker Analysis
        ticker = default_ticker
        
        print("="*80)
        print(f"FUNDAMENTAL ANALYSIS FOR {ticker}")
        print("="*80)
        
        # Run full analysis wrapper for single ticker
        result = run_analysis(ticker)
        
        if result['success']:
            lt = result['long_term']
            st = result['short_term']
            
            print("\n" + "="*80)
            print("LONG-TERM ANALYSIS (4 YEARS)")
            print("="*80)
            
            print(f"\n1. Revenue Growth (4Y):")
            print("-" * 40)
            print(f"  Is Growing: {lt['revenue_4y']['is_growing']}")
            print(f"  CAGR (3Y):  {lt['revenue_4y']['growth_3y_cagr']}%")
            print(f"  History:    {lt['revenue_4y']['revenue_history']}")
            
            print(f"\n2. Profit Growth (4Y):")
            print("-" * 40)
            print(f"  Is Growing: {lt['profit_4y']['is_growing']}")
            print(f"  CAGR (3Y):  {lt['profit_4y']['growth_3y_cagr']}%")
            print(f"  History:    {lt['profit_4y']['profit_history']}")
            
            print(f"\n3. ROE Growth (4Y):")
            print("-" * 40)
            print(f"  Is Growing: {lt['roe_4y']['is_growing']}")
            print(f"  History:    {lt['roe_4y']['roe_history']}")
            
            print(f"\n4. EPS Growth (4Y):")
            print("-" * 40)
            print(f"  Is Growing: {lt['eps_4y']['is_growing']}")
            print(f"  History:    {lt['eps_4y']['eps_history']}")
            
            print(f"\n5. PE vs Industry:")
            print("-" * 40)
            pe = lt['pe_ratio']
            print(f"  Current PE:  {pe['current_pe']}")
            print(f"  Industry PE: {pe.get('industry_pe', 'N/A')}")
            print(f"  Status:      {pe.get('vs_industry', 'N/A')}")
            
            print("\n" + "="*80)
            print("SHORT-TERM ANALYSIS (6 QUARTERS)")
            print("="*80)
            
            print(f"\n6. Revenue Growth (6Q):")
            print("-" * 40)
            print(f"  Is Growing: {st['revenue_6q']['is_growing']}")
            print(f"  Recent QoQ: {st['revenue_6q'].get('recent_quarter_growth', 'N/A')}%")
            
            print(f"\n7. Profit Growth (6Q):")
            print("-" * 40)
            print(f"  Is Growing: {st['profit_6q']['is_growing']}")
            print(f"  Recent QoQ: {st['profit_6q'].get('recent_quarter_growth', 'N/A')}%")
            
            print("\n" + "="*80)
            
        else:
            print(f"Analysis Failed: {result.get('error')}")
            
        # Optional: Uncomment to force run batch demo locally if needed
        # run_batch_analysis_demo(['RELIANCE.NS', 'TCS.NS'])
