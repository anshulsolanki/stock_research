"""
RELATIVE STRENGTH (RS) ANALYSIS TOOL
=====================================

PURPOSE:
--------
This module performs comprehensive Relative Strength (RS) analysis, a powerful leading indicator
used to identify market leaders vs. laggards. The RS analysis helps in positional trading by:

1. **Identifying Strong Performers**: Stocks outperforming the market (RS > 1.0)
2. **Detecting Emerging Leaders**: Stocks with accelerating momentum (1M RS > 3M RS > 6M RS)
3. **Avoiding Laggards**: Stocks underperforming the market (RS < 1.0)
4. **Timing Entries**: Entering positions when relative strength is improving

METHODOLOGY:
------------
The analysis compares a stock's performance against a benchmark index across multiple timeframes:

1. **Benchmark Selection**: Automatically selects appropriate benchmark
   - Indian stocks (.NS, .BO): ^NSEI (Nifty 50)
   - US stocks: ^GSPC (S&P 500)
   - Custom benchmark can be specified

2. **Multi-Timeframe RS Calculation**: Calculates RS ratio for 4 periods
   - 1 Month (21 trading days): Captures short-term momentum
   - 3 Months (63 trading days): Identifies intermediate trends
   - 6 Months (126 trading days): Confirms sustained strength
   - 1 Year (252 trading days): Validates long-term leadership

3. **RS Ratio Formula**:
   RS Ratio = (Stock Return / Benchmark Return)
   
   Where:
   - Stock Return = (Current Price - Price N periods ago) / Price N periods ago
   - Benchmark Return = Same calculation for benchmark index

4. **Signal Classification**:
   - **Strong Leader**: RS > 1.2 across ALL timeframes
   - **Emerging Leader**: 1M RS > 3M RS (accelerating momentum)
   - **Leader**: RS > 1.0 across most timeframes
   - **Weakening Leader**: 1M RS < 3M RS (decelerating momentum)
   - **Laggard**: RS < 0.8 across most timeframes

5. **RS Score**: Composite score (0-100) aggregating performance across timeframes
   - Weighted average: 1M (30%), 3M (30%), 6M (25%), 1Y (15%)
   - Higher score = stronger relative performance

TRADING SIGNALS:
----------------
**Strong Leader (RS > 1.2 consistently):**
- High conviction buy candidate
- Stock significantly outperforming market
- Momentum likely to continue
- Action: Consider buying on pullbacks

**Emerging Leader (1M > 3M > 6M):**
- Accelerating relative strength
- Early momentum building
- Potential breakout candidate
- Action: Watch closely, consider entry

**Weakening Leader (1M < 3M):**
- Momentum decelerating
- May be losing leadership status
- Action: Consider taking profits or tightening stops

**Laggard (RS < 0.8):**
- Underperforming market significantly
- Avoid or exit positions
- Action: Skip or sell

USAGE:
------
Run as standalone script:
    python rs_analysis.py

Or import and use programmatically:
    from rs_analysis import run_analysis
    results = run_analysis(ticker="RELIANCE.NS", show_plot=True)
    
With custom benchmark:
    results = run_analysis(ticker="AAPL", benchmark="^GSPC", show_plot=True)

With custom configuration:
    custom_config = {'LOOKBACK_PERIODS': 630}
    results = run_analysis(ticker="TCS.NS", config=custom_config)

OUTPUT:
-------
Returns dictionary containing:
- success: Boolean indicating if analysis completed successfully
- ticker: Stock ticker symbol
- benchmark: Benchmark index used
- rs_ratios: Dictionary with RS ratios for each timeframe (1M, 3M, 6M, 1Y)
- rs_score: Composite RS score (0-100)
- classification: Overall classification (Strong Leader, Emerging Leader, etc.)
- signals: List of detected RS signals with dates and descriptions
- figure: Matplotlib figure object (three-panel chart)
- stock_data: DataFrame with stock price data
- benchmark_data: DataFrame with benchmark price data

VISUALIZATION:
--------------
Three-panel chart showing:
1. **Price Comparison**: Normalized stock vs. benchmark performance
2. **RS Ratio Trends**: RS ratios over time for each timeframe
3. **RS Score Timeline**: Composite score with signal annotations

IMPLEMENTATION NOTES:
---------------------
- Compatible with both standalone execution and web app integration
- Matplotlib backend set to 'Agg' when called from web app (Flask compatibility)
- All calculations handle missing data appropriately with pandas rolling functions
- Benchmark auto-detection based on ticker suffix

DEPENDENCIES:
-------------
- pandas: Data manipulation and analysis
- numpy: Numerical operations
- yfinance: Historical stock data fetching
- matplotlib: Chart visualization
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import json
import os

# Default configuration for RS Analysis
RS_CONFIG = {
    # Timeframes for RS calculation (in trading days)
    'TIMEFRAMES': {
        '1M': 21,
        '3M': 63,
        '6M': 126,
        '1Y': 252
    },
    
    # RS Ratio Thresholds
    'THRESHOLDS': {
        'STRONG_LEADER': 1.2,
        'LEADER': 1.0,
        'LAGGARD': 0.8
    },
    
    # Scoring Weights (must sum to 1.0)
    'WEIGHTS': {
        '1M': 0.30,
        '3M': 0.30,
        '6M': 0.25,
        '1Y': 0.15
    },
    
    # Data Fetching
    'INTERVAL': '1d',
    'LOOKBACK_PERIODS': 504,  # ~2 years of trading days
    
    # Benchmark Auto-Detection
    'BENCHMARKS': {
        'INDIAN': '^NSEI',  # Nifty 50
        'US': '^GSPC'       # S&P 500
    },
    
    # Execution Control
    'RUN_ON_INIT': False
}


def load_tickers_grouped():
    """
    Load the tickers_grouped.json file.
    
    Returns:
    --------
    dict or None
        Dictionary with sector data or None if file not found
    """
    # Try to find the tickers_grouped.json file
    possible_paths = [
        # Relative to current file location
        os.path.join(os.path.dirname(__file__), '..', 'data', 'tickers_grouped.json'),
        # Absolute path
        '/Users/solankianshul/Documents/projects/stock_research/data/tickers_grouped.json',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
                continue
    
    return None


def find_ticker_sector(ticker, tickers_data=None):
    """
    Find which sector a ticker belongs to.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    tickers_data : dict, optional
        Pre-loaded tickers data. If None, will load from file.
        
    Returns:
    --------
    tuple (sector_name, sector_index) or (None, None)
        Sector name and its index symbol, or (None, None) if not found
    """
    if tickers_data is None:
        tickers_data = load_tickers_grouped()
    
    if tickers_data is None:
        return None, None
    
    ticker_upper = ticker.upper()
    
    # Search through all sectors (skip the "Sector" entry which contains the indices list)
    for sector_name, sector_data in tickers_data.items():
        if sector_name == "Sector":  # Skip the indices list
            continue
            
        if 'stocks' in sector_data and 'index_symbol' in sector_data:
            # Check if ticker is in this sector's stocks
            for stock_name, stock_ticker in sector_data['stocks'].items():
                if stock_ticker.upper() == ticker_upper:
                    return sector_name, sector_data['index_symbol']
    
    return None, None


def detect_benchmark(ticker, use_sector_index=False):
    """
    Auto-detect appropriate benchmark based on ticker suffix or market.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol
    use_sector_index : bool, optional
        If True, attempts to find sector-specific index from tickers_grouped.json
        If False or sector not found, uses broad market index
        
    Returns:
    --------
    str
        Benchmark ticker symbol
    """
    # Try to use sector index if requested
    if use_sector_index:
        sector_name, sector_index = find_ticker_sector(ticker)
        if sector_index:
            print(f"Found ticker in sector '{sector_name}', using sector index: {sector_index}")
            return sector_index
        else:
            print(f"Ticker not found in tickers_grouped.json, falling back to broad market index")
    
    # Fall back to broad market detection
    ticker_upper = ticker.upper()
    
    # Indian stock exchanges
    if ticker_upper.endswith('.NS') or ticker_upper.endswith('.BO'):
        return RS_CONFIG['BENCHMARKS']['INDIAN']
    
    # Default to US benchmark for all other stocks
    return RS_CONFIG['BENCHMARKS']['US']


def fetch_data(ticker, config=None):
    """
    Fetches historical OHLCV data for the given ticker.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'RELIANCE.NS')
    config : dict, optional
        Configuration dictionary with INTERVAL and LOOKBACK_PERIODS
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with Date index and OHLCV columns
    """
    if config is None:
        config = RS_CONFIG
    
    interval = config.get('INTERVAL', '1d')
    lookback = config.get('LOOKBACK_PERIODS', 504)
    
    end_date = datetime.now()
    
    # Adjust lookback calculation based on interval to ensure sufficient data points
    if interval in ['1wk', '1w']:
        # For weekly data, need 7x more calendar days to get same number of data points
        # Add extra buffer (1.5x) for weekends/holidays
        start_date = end_date - timedelta(days=int(lookback * 7 * 1.5))
    elif interval in ['1mo', '1M']:
        # For monthly data, need 30x more calendar days
        start_date = end_date - timedelta(days=int(lookback * 30 * 1.5))
    elif 'm' in interval or 'h' in interval:
        # For intraday intervals (15m, 1h, etc.), use a fixed period of days
        # Intraday data is typically limited to 60 days max by yfinance
        start_date = end_date - timedelta(days=min(60, int(lookback * 1.5)))
    else:
        # For daily intervals ('1d'), use lookback with buffer for weekends/holidays
        start_date = end_date - timedelta(days=int(lookback * 1.5))
    
    # Fetch data using yfinance
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
    
    if data.empty:
        raise ValueError(f"No data retrieved for ticker {ticker}")
    
    # Ensure we have the Close column
    if 'Close' not in data.columns:
        raise ValueError(f"Close price data not available for {ticker}")
    
    return data


def calculate_returns(df, periods, config=None):
    """
    Calculate percentage returns for multiple timeframes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with Close price column
    periods : dict
        Dictionary mapping timeframe names to trading days (e.g., {'1M': 21, '3M': 63})
    config : dict, optional
        Configuration dictionary (not currently used)
        
    Returns:
    --------
    pd.DataFrame
        Input DataFrame with added return columns for each timeframe
    """
    df = df.copy()
    
    for name, days in periods.items():
        # Calculate return: (Current - Past) / Past
        df[f'Return_{name}'] = df['Close'].pct_change(periods=days) * 100
    
    return df


def calculate_rs_ratio(df_stock, df_benchmark, periods):
    """
    Calculate RS ratio by comparing stock returns to benchmark returns.
    
    Parameters:
    -----------
    df_stock : pd.DataFrame
        Stock data with return columns
    df_benchmark : pd.DataFrame
        Benchmark data with return columns
    periods : dict
        Dictionary mapping timeframe names to trading days
        
    Returns:
    --------
    pd.DataFrame, dict
        - DataFrame with RS ratio columns
        - Dictionary with latest RS ratios for each timeframe
    """
    df_stock = df_stock.copy()
    latest_rs = {}
    
    # Align dataframes by common dates
    common_dates = df_stock.index.intersection(df_benchmark.index)
    df_stock = df_stock.loc[common_dates]
    df_benchmark_aligned = df_benchmark.loc[common_dates]
    
    for name in periods.keys():
        stock_return = df_stock[f'Return_{name}']
        benchmark_return = df_benchmark_aligned[f'Return_{name}']
        
        # Calculate RS Ratio: Stock Return / Benchmark Return
        # Handle division by zero and negative returns appropriately
        rs_ratio = np.where(
            benchmark_return != 0,
            (1 + stock_return / 100) / (1 + benchmark_return / 100),
            np.nan
        )
        
        df_stock[f'RS_{name}'] = rs_ratio
        
        # Get latest value
        latest_value = df_stock[f'RS_{name}'].iloc[-1]
        latest_rs[name] = latest_value if not np.isnan(latest_value) else None
    
    return df_stock, latest_rs


def calculate_rs_score(rs_ratios, config=None):
    """
    Calculate composite RS score (0-100) based on weighted average of RS ratios.
    
    Parameters:
    -----------
    rs_ratios : dict
        Dictionary with RS ratios for each timeframe
    config : dict, optional
        Configuration dictionary with WEIGHTS
        
    Returns:
    --------
    float
        RS score between 0 and 100
    """
    if config is None:
        config = RS_CONFIG
    
    weights = config.get('WEIGHTS', RS_CONFIG['WEIGHTS'])
    
    # Calculate weighted score
    score = 0
    for name, weight in weights.items():
        rs_value = rs_ratios.get(name)
        if rs_value is not None and not np.isnan(rs_value):
            # Convert RS ratio to score contribution
            # RS of 1.0 = 50 points, RS of 1.2 = 60 points, RS of 0.8 = 40 points
            contribution = ((rs_value - 1.0) * 50 + 50) * weight
            score += contribution
    
    # Clamp between 0 and 100
    return max(0, min(100, score))


def classify_rs(rs_ratios, config=None):
    """
    Classify stock based on RS ratios across timeframes.
    
    Parameters:
    -----------
    rs_ratios : dict
        Dictionary with RS ratios for each timeframe
    config : dict, optional
        Configuration dictionary with THRESHOLDS
        
    Returns:
    --------
    str
        Classification label
    """
    if config is None:
        config = RS_CONFIG
    
    thresholds = config.get('THRESHOLDS', RS_CONFIG['THRESHOLDS'])
    
    # Get valid RS values
    valid_rs = {k: v for k, v in rs_ratios.items() if v is not None and not np.isnan(v)}
    
    if not valid_rs:
        return "Insufficient Data"
    
    # Check for Strong Leader (all RS > 1.2)
    if all(v >= thresholds['STRONG_LEADER'] for v in valid_rs.values()):
        return "Strong Leader"
    
    # Check for Emerging Leader (1M > 3M RS, showing acceleration)
    if '1M' in valid_rs and '3M' in valid_rs:
        if valid_rs['1M'] > valid_rs['3M'] and valid_rs['1M'] >= thresholds['LEADER']:
            return "Emerging Leader"
    
    # Check for Weakening Leader (1M < 3M, showing deceleration)
    if '1M' in valid_rs and '3M' in valid_rs:
        if valid_rs['1M'] < valid_rs['3M'] and valid_rs['3M'] >= thresholds['LEADER']:
            return "Weakening Leader"
    
    # Check for Leader (most RS > 1.0)
    leader_count = sum(1 for v in valid_rs.values() if v >= thresholds['LEADER'])
    if leader_count >= len(valid_rs) * 0.6:  # 60% or more
        return "Leader"
    
    # Check for Laggard (most RS < 0.8)
    laggard_count = sum(1 for v in valid_rs.values() if v <= thresholds['LAGGARD'])
    if laggard_count >= len(valid_rs) * 0.6:  # 60% or more
        return "Laggard"
    
    # Default to Neutral
    return "Neutral"


def detect_rs_signals(df_stock, rs_ratios, classification, config=None):
    """
    Detect significant RS signals and patterns.
    
    Parameters:
    -----------
    df_stock : pd.DataFrame
        Stock data with RS columns
    rs_ratios : dict
        Latest RS ratios
    classification : str
        Overall classification
    config : dict, optional
        Configuration dictionary
        
    Returns:
    --------
    list of dict
        List of detected signals
    """
    signals = []
    latest_date = df_stock.index[-1].strftime('%Y-%m-%d')
    
    # Add classification signal
    signals.append({
        'date': latest_date,
        'type': classification,
        'description': f"Current classification: {classification}",
        'rs_ratios': rs_ratios.copy()
    })
    
    # Check for emerging momentum
    if '1M' in rs_ratios and '3M' in rs_ratios and '6M' in rs_ratios:
        if (rs_ratios['1M'] > rs_ratios['3M'] > rs_ratios['6M'] and 
            rs_ratios['1M'] > 1.0):
            signals.append({
                'date': latest_date,
                'type': 'Accelerating Momentum',
                'description': f"1M RS ({rs_ratios['1M']:.2f}) > 3M RS ({rs_ratios['3M']:.2f}) > 6M RS ({rs_ratios['6M']:.2f})",
                'rs_ratios': rs_ratios.copy()
            })
    
    # Check for strong outperformance
    if all(v and v >= 1.2 for v in rs_ratios.values()):
        avg_rs = np.mean([v for v in rs_ratios.values() if v])
        signals.append({
            'date': latest_date,
            'type': 'Strong Outperformance',
            'description': f"All timeframes show RS > 1.2 (Avg: {avg_rs:.2f})",
            'rs_ratios': rs_ratios.copy()
        })
    
    # Check for underperformance
    if all(v and v <= 0.8 for v in rs_ratios.values()):
        avg_rs = np.mean([v for v in rs_ratios.values() if v])
        signals.append({
            'date': latest_date,
            'type': 'Weak Underperformance',
            'description': f"All timeframes show RS < 0.8 (Avg: {avg_rs:.2f})",
            'rs_ratios': rs_ratios.copy()
        })
    
    return signals


def get_trading_signal_summary(classification, rs_ratios):
    """
    Get actionable trading signal summary based on classification.
    
    Parameters:
    -----------
    classification : str
        Overall classification (Strong Leader, Emerging Leader, etc.)
    rs_ratios : dict
        Dictionary with RS ratios for each timeframe
        
    Returns:
    --------
    str
        Formatted trading signal summary with interpretation and action guidance
    """
    summaries = {
        "Strong Leader": {
            "title": "Strong Leader (RS > 1.2 consistently)",
            "points": [
                "High conviction buy candidate",
                "Stock significantly outperforming market",
                "Momentum likely to continue",
                "Action: Consider buying on pullbacks"
            ]
        },
        "Emerging Leader": {
            "title": "Emerging Leader (1M > 3M > 6M)",
            "points": [
                "Accelerating relative strength",
                "Early momentum building",
                "Potential breakout candidate",
                "Action: Watch closely, consider entry"
            ]
        },
        "Leader": {
            "title": "Leader (RS > 1.0 across most timeframes)",
            "points": [
                "Outperforming the market",
                "Sustained relative strength",
                "Good candidate for positional trades",
                "Action: Consider buying on minor dips"
            ]
        },
        "Weakening Leader": {
            "title": "Weakening Leader (1M < 3M)",
            "points": [
                "Momentum decelerating",
                "May be losing leadership status",
                "Still outperforming but slowing",
                "Action: Consider taking profits or tightening stops"
            ]
        },
        "Neutral": {
            "title": "Neutral (Mixed RS signals)",
            "points": [
                "No clear relative strength trend",
                "Moving in line with the market",
                "Requires further analysis",
                "Action: Wait for clearer signals"
            ]
        },
        "Laggard": {
            "title": "Laggard (RS < 0.8)",
            "points": [
                "Underperforming market significantly",
                "Weak relative strength",
                "Avoid or exit positions",
                "Action: Skip or sell"
            ]
        },
        "Insufficient Data": {
            "title": "Insufficient Data",
            "points": [
                "Not enough data to classify",
                "Cannot determine relative strength",
                "Action: Check ticker or try different timeframe"
            ]
        }
    }
    
    summary_data = summaries.get(classification, summaries["Neutral"])
    
    # Build formatted output
    output = f"\n{'─'*60}\n"
    output += f"📊 TRADING SIGNAL SUMMARY\n"
    output += f"{'─'*60}\n"
    output += f"\n{summary_data['title']}\n\n"
    
    for point in summary_data['points']:
        # Highlight action items
        if point.startswith("Action:"):
            output += f"  ⚡ {point}\n"
        else:
            output += f"  • {point}\n"
    
    return output


def plot_rs_analysis(df_stock, df_benchmark, ticker, benchmark, rs_ratios, classification, signals, config=None):
    """
    Generate three-panel chart showing RS analysis.
    
    Parameters:
    -----------
    df_stock : pd.DataFrame
        Stock data with RS columns
    df_benchmark : pd.DataFrame
        Benchmark data
    ticker : str
        Stock ticker symbol
    benchmark : str
        Benchmark ticker symbol
    rs_ratios : dict
        Latest RS ratios
    classification : str
        Overall classification
    signals : list of dict
        Detected signals
    config : dict, optional
        Configuration dictionary
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the three-panel plot
    """
    if config is None:
        config = RS_CONFIG
    
    periods = config.get('TIMEFRAMES', RS_CONFIG['TIMEFRAMES'])
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'Relative Strength Analysis: {ticker} vs {benchmark}\nClassification: {classification}', 
                 fontsize=14, fontweight='bold')
    
    # Align dataframes
    common_dates = df_stock.index.intersection(df_benchmark.index)
    df_stock_plot = df_stock.loc[common_dates]
    df_benchmark_plot = df_benchmark.loc[common_dates]
    
    # Panel 1: Normalized Price Comparison
    stock_normalized = (df_stock_plot['Close'] / df_stock_plot['Close'].iloc[0]) * 100
    benchmark_normalized = (df_benchmark_plot['Close'] / df_benchmark_plot['Close'].iloc[0]) * 100
    
    ax1.plot(df_stock_plot.index, stock_normalized, label=ticker, color='blue', linewidth=2)
    ax1.plot(df_benchmark_plot.index, benchmark_normalized, label=benchmark, color='gray', linewidth=1.5, linestyle='--')
    ax1.set_ylabel('Normalized Price (Base=100)', fontsize=10)
    ax1.set_title('Price Performance Comparison (Normalized)', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: RS Ratio Trends
    colors = {'1M': 'red', '3M': 'orange', '6M': 'green', '1Y': 'blue'}
    for name in periods.keys():
        col = f'RS_{name}'
        if col in df_stock_plot.columns:
            ax2.plot(df_stock_plot.index, df_stock_plot[col], 
                    label=f'{name} RS', color=colors.get(name, 'black'), linewidth=1.5)
    
    # Add RS = 1.0 reference line
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='RS = 1.0 (Neutral)')
    ax2.axhline(y=1.2, color='green', linestyle=':', linewidth=1, alpha=0.5, label='RS = 1.2 (Strong)')
    ax2.axhline(y=0.8, color='red', linestyle=':', linewidth=1, alpha=0.5, label='RS = 0.8 (Weak)')
    
    ax2.set_ylabel('RS Ratio', fontsize=10)
    ax2.set_title('RS Ratio Trends (Multiple Timeframes)', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper left', ncol=2, fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: RS Score Timeline (calculated over time)
    rs_scores = []
    for idx in range(len(df_stock_plot)):
        row_rs = {}
        for name in periods.keys():
            col = f'RS_{name}'
            if col in df_stock_plot.columns:
                val = df_stock_plot[col].iloc[idx]
                if not np.isnan(val):
                    row_rs[name] = val
        if row_rs:
            score = calculate_rs_score(row_rs, config)
            rs_scores.append(score)
        else:
            rs_scores.append(np.nan)
    
    ax3.plot(df_stock_plot.index, rs_scores, color='purple', linewidth=2, label='RS Score')
    ax3.axhline(y=50, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Neutral (50)')
    ax3.fill_between(df_stock_plot.index, 50, rs_scores, 
                     where=[s >= 50 for s in rs_scores], 
                     color='green', alpha=0.2, label='Outperforming')
    ax3.fill_between(df_stock_plot.index, 50, rs_scores, 
                     where=[s < 50 for s in rs_scores], 
                     color='red', alpha=0.2, label='Underperforming')
    
    ax3.set_ylabel('RS Score (0-100)', fontsize=10)
    ax3.set_xlabel('Date', fontsize=10)
    ax3.set_title('RS Composite Score Timeline', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Format x-axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig


def run_analysis(ticker, benchmark=None, show_plot=True, config=None, use_sector_index=False):
    """
    Main orchestration function for RS analysis.
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol to analyze
    benchmark : str, optional
        Benchmark ticker. If None, auto-detected based on stock ticker
    show_plot : bool, optional
        If True, displays the plot interactively. If False, returns figure for web app use.
    config : dict, optional
        Configuration dictionary to override defaults
    use_sector_index : bool, optional
        If True, attempts to use sector-specific index from tickers_grouped.json
        If False or not found, uses broad market index (default: False)
        
    Returns:
    --------
    dict
        Results dictionary containing:
        - success: bool
        - ticker: str
        - benchmark: str
        - sector: str (if found)
        - rs_ratios: dict
        - rs_score: float
        - classification: str
        - signals: list of dict
        - trading_summary: str
        - figure: matplotlib.figure.Figure
        - stock_data: pd.DataFrame
        - benchmark_data: pd.DataFrame
    """
    # Set matplotlib backend for web app compatibility
    if not show_plot:
        matplotlib.use('Agg')
    
    # Merge provided config with defaults
    if config is None:
        config = RS_CONFIG.copy()
    else:
        merged_config = RS_CONFIG.copy()
        merged_config.update(config)
        config = merged_config
    
    # Auto-detect benchmark if not provided
    sector_name = None
    if benchmark is None:
        benchmark = detect_benchmark(ticker, use_sector_index=use_sector_index)
        if use_sector_index:
            sector_name, _ = find_ticker_sector(ticker)
        print(f"Auto-detected benchmark: {benchmark}")
    
    try:
        # Fetch data for stock and benchmark
        print(f"Fetching data for {ticker}...")
        df_stock = fetch_data(ticker, config)
        
        print(f"Fetching data for {benchmark}...")
        df_benchmark = fetch_data(benchmark, config)
        
        # Calculate returns for both
        periods = config.get('TIMEFRAMES', RS_CONFIG['TIMEFRAMES'])
        df_stock = calculate_returns(df_stock, periods, config)
        df_benchmark = calculate_returns(df_benchmark, periods, config)
        
        # Calculate RS ratios
        df_stock, rs_ratios = calculate_rs_ratio(df_stock, df_benchmark, periods)
        
        # Calculate RS score
        rs_score = calculate_rs_score(rs_ratios, config)
        
        # Classify
        classification = classify_rs(rs_ratios, config)
        
        # Detect signals
        signals = detect_rs_signals(df_stock, rs_ratios, classification, config)
        
        # Generate plot
        fig = plot_rs_analysis(df_stock, df_benchmark, ticker, benchmark, 
                              rs_ratios, classification, signals, config)
        
        if show_plot:
            plt.show()
        
        print(f"\n{'='*60}")
        print(f"RS Analysis Results for {ticker}")
        print(f"{'='*60}")
        print(f"Benchmark: {benchmark}")
        print(f"Classification: {classification}")
        print(f"RS Score: {rs_score:.1f}/100")
        print(f"\nRS Ratios:")
        for name, value in rs_ratios.items():
            if value is not None:
                print(f"  {name}: {value:.3f}")
        print(f"\nSignals Detected: {len(signals)}")
        for signal in signals:
            print(f"  - {signal['type']}: {signal['description']}")
        print(f"{'='*60}")
        
        # Print trading signal summary
        trading_summary = get_trading_signal_summary(classification, rs_ratios)
        print(trading_summary)
        
        result = {
            'success': True,
            'ticker': ticker,
            'benchmark': benchmark,
            'rs_ratios': rs_ratios,
            'rs_score': rs_score,
            'classification': classification,
            'signals': signals,
            'trading_summary': get_trading_signal_summary(classification, rs_ratios),
            'figure': fig,
            'stock_data': df_stock,
            'benchmark_data': df_benchmark
        }
        
        # Add sector info if available
        if sector_name:
            result['sector'] = sector_name
        
        return result
        
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'ticker': ticker,
            'benchmark': benchmark if benchmark else 'Unknown',
            'error': str(e)
        }


# Main execution when run as standalone script
if __name__ == "__main__":
    # Example: Analyze an Indian stock against sector index
    print("Running RS Analysis on LT.NS (Indian Stock)...")
    print("Using SECTOR INDEX for comparison\n")
    result = run_analysis("DABUR.NS", show_plot=True, use_sector_index=True)
    
    # Example: Analyze using broad market index
    # print("\nRunning RS Analysis on LT.NS (Indian Stock)...")
    # print("Using BROAD MARKET INDEX for comparison\n")
    # result = run_analysis("LT.NS", show_plot=True, use_sector_index=False)
    
    # Example: Analyze a US stock (Apple)
    # print("\nRunning RS Analysis on AAPL (US Stock)...")
    # result = run_analysis("AAPL", show_plot=True)
