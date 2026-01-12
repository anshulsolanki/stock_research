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
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# HELPER FUNCTIONS FOR DATA FETCHING
# ============================================================================

def fetch_financials(ticker):
    """
    Fetch annual financial statements for a ticker.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
        
    Returns:
    --------
    dict
        Dictionary containing income_statement, balance_sheet, cashflow, and info
    """
    try:
        stock = yf.Ticker(ticker)
        return {
            'income_statement': stock.financials,
            'balance_sheet': stock.balance_sheet,
            'cashflow': stock.cashflow,
            'info': stock.info,
            'quarterly_income': stock.quarterly_financials,
            'quarterly_balance': stock.quarterly_balance_sheet
        }
    except Exception as e:
        return {'error': str(e)}


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

def analyze_revenue_growth_4y(ticker):
    """
    Analyze revenue growth over the last 4 years.
    
    Checks:
    - Is revenue growing? (1Y > 3Y trend)
    - What is the 3-year CAGR?
    
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


def analyze_profit_growth_4y(ticker):
    """
    Analyze profit growth over the last 4 years.
    
    Checks:
    - Is profit growing? (1Y > 3Y trend)
    - What is the 3-year CAGR?
    
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


def analyze_roe_growth_4y(ticker):
    """
    Analyze ROE (Return on Equity) growth over the last 4 years.
    
    Checks:
    - Is ROE growing? (1Y > 3Y trend)
    - What is the 3-year CAGR?
    
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


def analyze_eps_growth_4y(ticker):
    """
    Analyze EPS (Earnings Per Share) growth over the last 4 years.
    
    Checks:
    - Is EPS growing? (1Y > 3Y trend)
    - What is the 3-year CAGR?
    
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


def analyze_pe_vs_industry(ticker):
    """
    Analyze PE ratio compared to industry average.
    
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

def analyze_revenue_growth_6q(ticker):
    """
    Analyze revenue growth over the last 6 quarters.
    
    Checks:
    - Is revenue growing quarter-over-quarter?
    - Growth trend analysis
    
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


def analyze_profit_growth_6q(ticker):
    """
    Analyze profit growth over the last 6 quarters.
    
    Checks:
    - Is profit growing quarter-over-quarter?
    - Growth trend analysis
    
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


def analyze_roe_growth_6q(ticker):
    """
    Analyze ROE growth over the last 6 quarters.
    
    Checks:
    - Is ROE growing quarter-over-quarter?
    - Growth trend analysis
    
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


def analyze_eps_growth_6q(ticker):
    """
    Analyze EPS growth over the last 6 quarters.
    
    Checks:
    - Is EPS growing quarter-over-quarter?
    - Growth trend analysis
    
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
        results = {
            'success': True,
            'ticker': ticker,
            'long_term': {},
            'short_term': {},
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Long-term analysis (4Y)
        results['long_term']['revenue_4y'] = analyze_revenue_growth_4y(ticker)
        results['long_term']['profit_4y'] = analyze_profit_growth_4y(ticker)
        results['long_term']['roe_4y'] = analyze_roe_growth_4y(ticker)
        results['long_term']['eps_4y'] = analyze_eps_growth_4y(ticker)
        results['long_term']['pe_ratio'] = analyze_pe_vs_industry(ticker)
        
        # Short-term analysis (6Q)
        results['short_term']['revenue_6q'] = analyze_revenue_growth_6q(ticker)
        results['short_term']['profit_6q'] = analyze_profit_growth_6q(ticker)
        results['short_term']['roe_6q'] = analyze_roe_growth_6q(ticker)
        results['short_term']['eps_6q'] = analyze_eps_growth_6q(ticker)
        
        return results
        
    except Exception as e:
        return {
            'success': False,
            'ticker': ticker,
            'error': str(e),
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


def analyze_batch(tickers_list):
    """
    Run fundamental analysis on multiple tickers.
    
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
    
    for ticker in tickers_list:
        print(f"Analyzing {ticker}...")
        result = run_analysis(ticker)
        
        if result['success']:
            results.append(result)
        else:
            failed_tickers.append(ticker)
            print(f"  Failed: {result.get('error', 'Unknown error')}")
    
    return {
        'success': True,
        'analyzed_count': len(results),
        'failed_count': len(failed_tickers),
        'results': results,
        'failed_tickers': failed_tickers
    }


# ============================================================================
# MAIN EXECUTION (FOR TESTING)
# ============================================================================

if __name__ == "__main__":
    # Test all functions with a sample ticker
    ticker = "BRITANNIA.NS"
    
    print("="*80)
    print(f"FUNDAMENTAL ANALYSIS FOR {ticker}")
    print("="*80)
    
    print("\n" + "="*80)
    print("LONG-TERM ANALYSIS (4 YEARS)")
    print("="*80)
    
    print("\n1. Revenue Growth (4Y):")
    print("-" * 40)
    result = analyze_revenue_growth_4y(ticker)
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    print("\n2. Profit Growth (4Y):")
    print("-" * 40)
    result = analyze_profit_growth_4y(ticker)
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    print("\n3. ROE Growth (4Y):")
    print("-" * 40)
    result = analyze_roe_growth_4y(ticker)
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    print("\n4. EPS Growth (4Y):")
    print("-" * 40)
    result = analyze_eps_growth_4y(ticker)
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    print("\n5. PE vs Industry:")
    print("-" * 40)
    result = analyze_pe_vs_industry(ticker)
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*80)
    print("SHORT-TERM ANALYSIS (6 QUARTERS)")
    print("="*80)
    
    print("\n6. Revenue Growth (6Q):")
    print("-" * 40)
    result = analyze_revenue_growth_6q(ticker)
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    print("\n7. Profit Growth (6Q):")
    print("-" * 40)
    result = analyze_profit_growth_6q(ticker)
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    print("\n8. ROE Growth (6Q):")
    print("-" * 40)
    result = analyze_roe_growth_6q(ticker)
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    print("\n9. EPS Growth (6Q):")
    print("-" * 40)
    result = analyze_eps_growth_6q(ticker)
    for key, value in result.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*80)
