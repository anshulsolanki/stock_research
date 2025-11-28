import argparse
from stock_research.lagging_indicator_analysis.supertrend_analysis import fetch_data, calculate_supertrend, plot_supertrend
from stock_research.lagging_indicator_analysis.crossover_analysis import calculate_emas, check_golden_crossover

def get_trend_info(df):
    """
    Extracts the latest trend information from the DataFrame.
    Returns: (Trend Type, Last Signal Date, Signal Price)
    """
    if df.empty:
        return "Unknown", None, None
        
    current_trend = df['Trend'].iloc[-1]
    trend_type = "UPTREND" if current_trend == 1 else "DOWNTREND"
    
    # Find the last change
    # Create a mask where trend changes
    df['Trend_Change'] = df['Trend'].diff().fillna(0)
    
    # Filter for changes
    changes = df[df['Trend_Change'] != 0]
    
    if not changes.empty:
        last_signal = changes.iloc[-1]
        last_date = last_signal.name
        last_price = last_signal['Close']
    else:
        # If no change in the loaded data, take the first date
        last_date = df.index[0]
        last_price = df['Close'].iloc[0]
        
    return trend_type, last_date, last_price

def analyze_multi_timeframe(ticker):
    """
    Analyzes trends for Weekly, Daily, and 15min timeframes.
    Also checks for Golden Crossover (Daily).
    """
    # Import pandas here to ensure it's available for DataFrame creation
    import pandas as pd
    
    print(f"\n--- Multi-Timeframe Trend Analysis for {ticker} ---\n")
    
    configs = [
        ("Weekly", 7, 2.0, '1wk', 365*2),
        ("Daily", 14, 2.0, '1d', 365),
        ("15min", 21, 3.0, '15m', 59)
    ]
    
    # Dictionary to store flattened results for this ticker
    ticker_result = {'Ticker': ticker}
    
    # --- Supertrend Analysis ---
    for name, period, multiplier, interval, days_back in configs:
        try:
            df = fetch_data(ticker, interval, days_back)
            df = calculate_supertrend(df, period, multiplier)
            trend, date, price = get_trend_info(df)
            
            print(f"{name} ({interval}): {trend}")
            print(f"  Last Signal Date: {date}")
            print(f"  Signal Price: {price:.2f}")
            print("-" * 30)
            
            # Add to flattened dictionary with prefix
            ticker_result[f'{name} Trend'] = trend
            ticker_result[f'{name} Date'] = date
            ticker_result[f'{name} Price'] = price
            
        except Exception as e:
            print(f"{name}: Error - {e}")
            ticker_result[f'{name} Trend'] = "Error"
            ticker_result[f'{name} Date'] = None
            ticker_result[f'{name} Price'] = None
            
    # --- Golden Crossover Analysis (Daily) ---
    try:
        print("Checking Golden Crossover (Daily)...")
        # Fetch enough data for 200 EMA
        df_gc = fetch_data(ticker, '1d', 365*2)
        df_gc = calculate_emas(df_gc)
        gc_signals = check_golden_crossover(df_gc)
        
        ticker_result['GC Trend Status'] = gc_signals.get('Trend_Status', 'N/A')
        ticker_result['GC Date'] = gc_signals.get('GC_Date', None)
        ticker_result['GC Price'] = gc_signals.get('GC_Price', None)
        
        if gc_signals.get('GC_Date'):
            print(f"  Last Golden Cross: {gc_signals['GC_Date'].date()} at {gc_signals['GC_Price']:.2f}")
        else:
            print("  No Golden Cross found in period.")
            
    except Exception as e:
        print(f"Golden Crossover Error: {e}")
        ticker_result['GC Trend Status'] = "Error"
            
    return ticker_result

if __name__ == "__main__":
    import pandas as pd
    import os
    
    tickers_file = 'tickers.txt'
    all_results = []
    
    if os.path.exists(tickers_file):
        with open(tickers_file, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
            
        print(f"Found {len(tickers)} tickers in {tickers_file}. Starting batch analysis...")
        
        for ticker in tickers:
            try:
                result = analyze_multi_timeframe(ticker)
                all_results.append(result)
            except Exception as e:
                print(f"Failed to analyze {ticker}: {e}")
    else:
        print(f"Error: {tickers_file} not found. Please create it with one ticker per line.")
        # Fallback to single ticker for testing
        tickers = ['LT.NS']
        for ticker in tickers:
             result = analyze_multi_timeframe(ticker)
             all_results.append(result)

    
    if all_results:
        # Create DataFrame
        df_results = pd.DataFrame(all_results)
        
        # Remove timezone from any Date columns for Excel compatibility
        date_cols = [col for col in df_results.columns if 'Date' in col]
        for col in date_cols:
            df_results[col] = df_results[col].apply(lambda x: x.replace(tzinfo=None) if pd.notnull(x) and hasattr(x, 'tzinfo') else x)
        
        # Define filename
        filename = "stock_analysis_results.xlsx"
        
        # Save to Excel
        try:
            df_results.to_excel(filename, index=False)
            print(f"\nBatch analysis complete. Results successfully saved to {filename}")
        except Exception as e:
            print(f"\nError saving to Excel: {e}")


