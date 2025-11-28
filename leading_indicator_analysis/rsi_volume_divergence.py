import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
from scipy.signal import argrelextrema

# --- Data Fetching ---
def fetch_data(ticker, interval='1d', days_back=365):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=False, multi_level_index=False)
    
    if df.empty:
        raise ValueError(f"No data found for {ticker}")
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    return df

# --- RSI Logic ---
def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def detect_rsi_divergence(df, order=5):
    # Find local peaks and troughs for Price
    df['price_peak'] = df.iloc[argrelextrema(df['Close'].values, np.greater_equal, order=order)[0]]['Close']
    df['price_trough'] = df.iloc[argrelextrema(df['Close'].values, np.less_equal, order=order)[0]]['Close']
    
    # Find local peaks and troughs for RSI
    df['rsi_peak'] = df.iloc[argrelextrema(df['RSI'].values, np.greater_equal, order=order)[0]]['RSI']
    df['rsi_trough'] = df.iloc[argrelextrema(df['RSI'].values, np.less_equal, order=order)[0]]['RSI']
    
    divergences = []
    
    # Bearish Divergence (Price HH, RSI LH)
    price_peaks = df[df['price_peak'].notna()]
    if len(price_peaks) >= 2:
        curr_peak = price_peaks.iloc[-1]
        prev_peak = price_peaks.iloc[-2]
        curr_rsi = df.loc[curr_peak.name]['RSI']
        prev_rsi = df.loc[prev_peak.name]['RSI']
        
        if curr_peak['Close'] > prev_peak['Close'] and curr_rsi < prev_rsi:
            divergences.append({
                'Type': 'Bearish RSI Divergence',
                'Date': curr_peak.name,
                'Price': curr_peak['Close'],
                'Details': f"Price HH, RSI LH"
            })

    # Bullish Divergence (Price LL, RSI HL)
    price_troughs = df[df['price_trough'].notna()]
    if len(price_troughs) >= 2:
        curr_trough = price_troughs.iloc[-1]
        prev_trough = price_troughs.iloc[-2]
        curr_rsi = df.loc[curr_trough.name]['RSI']
        prev_rsi = df.loc[prev_trough.name]['RSI']
        
        if curr_trough['Close'] < prev_trough['Close'] and curr_rsi > prev_rsi:
            divergences.append({
                'Type': 'Bullish RSI Divergence',
                'Date': curr_trough.name,
                'Price': curr_trough['Close'],
                'Details': f"Price LL, RSI HL"
            })
            
    return divergences

# --- Volume Logic ---
def calculate_trends(df, window=14):
    def get_slope(series):
        y = series.values
        x = np.arange(len(y))
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope

    df['Price_Slope'] = df['Close'].rolling(window=window).apply(get_slope, raw=False)
    df['Volume_Slope'] = df['Volume'].rolling(window=window).apply(get_slope, raw=False)
    return df

def detect_volume_divergence(df):
    divergences = []
    subset = df.iloc[-60:].copy() # Scan last 60 days
    last_signal_type = None
    
    for date, row in subset.iterrows():
        price_slope = row['Price_Slope']
        vol_slope = row['Volume_Slope']
        price = row['Close']
        
        if pd.isna(price_slope) or pd.isna(vol_slope):
            continue
            
        current_signal = None
        details = ""
        
        if price_slope < 0 and vol_slope < 0:
            current_signal = 'Bullish Volume Divergence'
            details = "Price Falling, Volume Dropping (Weak Selling)"
        elif price_slope > 0 and vol_slope < 0:
            current_signal = 'Bearish Volume Divergence'
            details = "Price Rising, Volume Dropping (Weak Buying)"
            
        if current_signal and current_signal != last_signal_type:
            divergences.append({
                'Type': current_signal,
                'Date': date,
                'Price': price,
                'Details': details
            })
            last_signal_type = current_signal
            
    return divergences

# --- Combined Analysis ---
def analyze_combined(ticker, plot=False):
    print(f"\nAnalyzing {ticker} for Combined Signals...")
    try:
        df = fetch_data(ticker)
        
        # Calculate Indicators
        df = calculate_rsi(df)
        df = calculate_trends(df)
        
        # Detect Signals
        rsi_divs = detect_rsi_divergence(df)
        vol_divs = detect_volume_divergence(df)
        
        # Combine and Check for Confluence
        # We look for signals that happened recently (e.g., within the last 10 days)
        # and if both indicators agree, we issue a strong signal.
        
        recent_rsi = [d for d in rsi_divs if (datetime.now() - d['Date']).days <= 30]
        recent_vol = [d for d in vol_divs if (datetime.now() - d['Date']).days <= 30]
        
        print(f"  Recent RSI Signals: {len(recent_rsi)}")
        print(f"  Recent Volume Signals: {len(recent_vol)}")
        
        # Check for Strong Buy
        bullish_rsi = any('Bullish' in d['Type'] for d in recent_rsi)
        bullish_vol = any('Bullish' in d['Type'] for d in recent_vol)
        
        if bullish_rsi and bullish_vol:
            print(f"  [STRONG BUY SIGNAL] Confluence of Bullish RSI and Volume Divergence detected!")
            print(f"  - RSI: {[d['Date'].date() for d in recent_rsi if 'Bullish' in d['Type']]}")
            print(f"  - Vol: {[d['Date'].date() for d in recent_vol if 'Bullish' in d['Type']]}")
            
        # Check for Strong Sell
        bearish_rsi = any('Bearish' in d['Type'] for d in recent_rsi)
        bearish_vol = any('Bearish' in d['Type'] for d in recent_vol)
        
        if bearish_rsi and bearish_vol:
            print(f"  [STRONG SELL SIGNAL] Confluence of Bearish RSI and Volume Divergence detected!")
            print(f"  - RSI: {[d['Date'].date() for d in recent_rsi if 'Bearish' in d['Type']]}")
            print(f"  - Vol: {[d['Date'].date() for d in recent_vol if 'Bearish' in d['Type']]}")
            
        if not (bullish_rsi and bullish_vol) and not (bearish_rsi and bearish_vol):
             print("  No strong combined signals detected.")
             
        if plot:
            plot_combined_divergence(df, ticker, rsi_divs, vol_divs)

    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")

def plot_combined_divergence(df, ticker, rsi_divs, vol_divs):
    """
    Plots Price, RSI, and Volume with divergence markers.
    """
    plt.figure(figsize=(14, 12))
    
    # --- Subplot 1: Price ---
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(df.index, df['Close'], label='Close Price', color='blue', linewidth=1)
    plt.title(f'{ticker} Combined Divergence Analysis')
    plt.ylabel('Price')
    plt.grid(True)
    
    # Plot RSI Divergences on Price
    for div in rsi_divs:
        color = 'red' if 'Bearish' in div['Type'] else 'green'
        marker = 'v' if 'Bearish' in div['Type'] else '^'
        # Plot slightly offset
        y_pos = div['Price'] * (1.02 if 'Bearish' in div['Type'] else 0.98)
        ax1.scatter(div['Date'], y_pos, color=color, marker=marker, s=100, label=f"RSI {div['Type'].split()[0]}", zorder=5)

    # Plot Volume Divergences on Price
    for div in vol_divs:
        color = 'orange' if 'Bearish' in div['Type'] else 'cyan' # Different colors for Volume Divs
        marker = 'D' # Diamond marker for Volume Divs
        # Plot slightly offset (further than RSI)
        y_pos = div['Price'] * (1.04 if 'Bearish' in div['Type'] else 0.96)
        ax1.scatter(div['Date'], y_pos, color=color, marker=marker, s=80, label=f"Vol {div['Type'].split()[0]}", zorder=5)

    # Unique Legend
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='best')

    # --- Subplot 2: RSI ---
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(df.index, df['RSI'], label='RSI (14)', color='purple')
    ax2.axhline(70, linestyle='--', alpha=0.5, color='red')
    ax2.axhline(30, linestyle='--', alpha=0.5, color='green')
    plt.ylabel('RSI')
    plt.legend(loc='upper right')
    plt.grid(True)

    # --- Subplot 3: Volume ---
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    colors = np.where(df['Close'].diff() >= 0, 'green', 'red')
    ax3.bar(df.index, df['Volume'], color=colors, alpha=0.5, label='Volume')
    plt.ylabel('Volume')
    plt.xlabel('Date')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def analyze_batch(tickers_file='tickers.txt'):
    import os
    if not os.path.exists(tickers_file):
        if os.path.exists(os.path.join('..', '..', tickers_file)): tickers_file = os.path.join('..', '..', tickers_file)
        elif os.path.exists(os.path.join('..', tickers_file)): tickers_file = os.path.join('..', tickers_file)
        else:
            print(f"Error: {tickers_file} not found.")
            return

    with open(tickers_file, 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]
        
    print(f"Found {len(tickers)} tickers. Starting combined analysis...")
    for ticker in tickers:
        analyze_combined(ticker)

if __name__ == "__main__":
    import os
    run_batch = False # Default to batch for this powerful script
    
    if os.path.exists('tickers.txt') and run_batch:
        analyze_batch('tickers.txt')
    elif os.path.exists('../tickers.txt') and run_batch:
        analyze_batch('../tickers.txt')
    else:
        analyze_combined("reliance.NS", plot=True)
