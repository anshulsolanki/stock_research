import yfinance as yf
import pandas as pd
import numpy as np
import calendar
from datetime import datetime, timedelta
import argparse

def get_last_thursday(year, month):
    """Returns the date of the last Thursday of the given year and month."""
    cal = calendar.monthcalendar(year, month)
    # The last week might not have a Thursday if it ends early, check the last two weeks
    last_week = cal[-1]
    last_thursday = last_week[calendar.THURSDAY]
    
    if last_thursday == 0:
        # If the last week doesn't have a Thursday, check the previous week
        last_week = cal[-2]
        last_thursday = last_week[calendar.THURSDAY]
        
    return datetime(year, month, last_thursday).date()

def get_next_valid_trading_day(date, valid_dates_set, lookback=5):
    """
    Finds the date in valid_dates_set that is equal to or just before the rolled date
    to simulate selling on market close of expiry.
    If exact expiry is not a trading day, use the previous available trading day.
    """
    current_date = date
    for _ in range(lookback):
        if current_date in valid_dates_set:
            return current_date
        current_date -= timedelta(days=1)
    return None

def fetch_data(ticker, period="5y"):
    print(f"Fetching {period} data for {ticker}...")
    df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    
    # Flatten MultiIndex columns if present (common in recent yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Calculate indicators
    df['50EMA'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['200EMA'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    return df

def run_backtest(ticker, lot_size=1):
    try:
        df = fetch_data(ticker)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Clean data: drop NaN for initial 200 days
    # EMA might have values earlier, but we want valid crossover checks
    df = df.dropna(subset=['50EMA', '200EMA'])
    
    if df.empty:
        print("Not enough data after calculating indicators.")
        return


    # Valid trading dates set for quick lookup
    valid_dates_set = set(df.index.date)
    
    trades = []
    in_position = False
    entry_price = 0.0
    entry_date = None
    trend_start_date = None
    
    # Logic:
    # Golden Cross: 50EMA crosses above 200EMA
    # Death Cross: 50EMA crosses below 200EMA
    # Rollover: On Last Thursday of month (or prev trading day), if in position:
    #   Sell current, Buy next (record P/L)
    
    # We iterate day by day
    # To identify crossover, we need previous day values
    
    # Create a 'Signal' column: 1 if 50 > 200, 0 otherwise
    df['Signal'] = np.where(df['50EMA'] > df['200EMA'], 1, 0)
    df['Crossover'] = df['Signal'].diff() # 1 = Golden Cross, -1 = Death Cross
    
    last_expiry_month_processed = None
    
    # Iterate through the DataFrame
    # Using itertuples for speed, but index is Timestamp
    
    for row in df.itertuples():
        current_date = row.Index.date()
        current_close = row.Close
        crossover = row.Crossover
        signal = row.Signal
        
        # Check for Expiry Rollover FIRST if we are in a position
        if in_position:
            # Check if today is the expiry day for this month
            expiry_date = get_last_thursday(current_date.year, current_date.month)
            
            # Adjust expiry date to valid trading day
            # If current_date is >= expiry_date, we might have passed it or be on it
            # But we want to execute EXACTLY on the effective expiry day
            
            # Effective expiry is the date we actually trade
            effective_expiry = get_next_valid_trading_day(expiry_date, valid_dates_set)
            
            if effective_expiry and current_date == effective_expiry:
                # Need to ensure we haven't already processed this month's expiry
                # This check prevents double processing if logic is weird, though dates are unique
                current_month_key = (current_date.year, current_date.month)
                
                if last_expiry_month_processed != current_month_key:
                    # Time to ROLL
                    # Logic: Sell current position at Close, Record P/L, Buy Again (Keep Entry Price = Close)
                    
                    # BUT wait, the crossover check is usually for ENTRY/EXIT.
                    # If Death Cross happens ON expiry day, we just Exit. We don't roll.
                    # Let's check Death Cross status below.
                    
                    # If signal is still 1 (Golden) or 0 (Death Cross JUST happening)
                    # If Death Cross happens today (Crossover == -1), we will handle it in the Exit block.
                    # So we only roll if NO Death Cross today.
                    
                    if crossover != -1:
                        # ROLL
                        sell_price = current_close
                        pnl = (sell_price - entry_price) * lot_size
                        
                        trades.append({
                            "Type": "Rollover",
                            "Entry Date": entry_date,
                            "Entry Price": entry_price,
                            "Exit Date": current_date,
                            "Exit Price": sell_price,
                            "Expiry Month": current_date.strftime("%Y-%m"),
                            "P/L": pnl,
                            "Trend Start": trend_start_date
                        })
                        
                        # Re-enter for next month
                        entry_price = current_close
                        entry_date = current_date # Technically next day open?? But user said "same day"
                        last_expiry_month_processed = current_month_key

        # Check Signals
        if crossover == 1 and not in_position:
            # Golden Cross -> Buy
            entry_price = current_close
            entry_date = current_date
            trend_start_date = current_date
            in_position = True
            # Reset expiry tracking
            last_expiry_month_processed = None
            
        elif crossover == -1 and in_position:
            # Death Cross -> Sell (Exit completely)
            sell_price = current_close
            pnl = (sell_price - entry_price) * lot_size
            
            trades.append({
                "Type": "Exit (Death Cross)",
                "Entry Date": entry_date,
                "Entry Price": entry_price,
                "Exit Date": current_date,
                "Exit Price": sell_price,
                "Expiry Month": current_date.strftime("%Y-%m"),
                "P/L": pnl,
                "Trend Start": trend_start_date
            })
            
            in_position = False
            last_expiry_month_processed = None
            trend_start_date = None
            
    # End of data check
    if in_position:
        trades.append({
            "Type": "Open Position",
            "Entry Date": entry_date,
            "Entry Price": entry_price,
            "Exit Date": "Holding",
            "Exit Price": df.iloc[-1]['Close'],
            "Expiry Month": "Current",
            "P/L": (df.iloc[-1]['Close'] - entry_price) * lot_size,
            "Trend Start": trend_start_date
        })

    display_results(trades, lot_size)

def display_results(trades, lot_size):
    if not trades:
        print("No trades found in the last 5 years.")
        return

    total_pnl = sum(t['P/L'] for t in trades)
    
    # Format for table
    print(f"\n{'-'*140}")
    print(f"{'Buy Date':<12} {'Buy Price':<12} {'Sell Date':<12} {'Sell Price':<12} {'Month':<8} {'P/L':<10} {'Leg P/L':<10} {'Note':<40}")
    print(f"{'-'*140}")
    
    current_leg_pnl = 0.0
    
    for t in trades:
        current_leg_pnl += t['P/L']
        exit_price_str = f"{t['Exit Price']:.2f}" if isinstance(t['Exit Price'], (int, float)) else str(t['Exit Price'])
        entry_str = f"(Entry: {t['Trend Start']})" if t['Trend Start'] else ""
        note = f"{t['Type']} {entry_str}"
        
        # Determine if this is the end of a leg
        is_exit = "Exit" in t['Type'] or "Open Position" in t['Type']
        
        leg_pnl_str = f"{current_leg_pnl:.2f}" if is_exit else "-"
        
        print(f"{str(t['Entry Date']):<12} {t['Entry Price']:<12.2f} {str(t['Exit Date']):<12} {exit_price_str:<12} {t['Expiry Month']:<8} {t['P/L']:<10.2f} {leg_pnl_str:<10} {note:<40}")
        
        if is_exit:
            print(f"{'-'*140}")
            current_leg_pnl = 0.0
    
    print(f"Total P/L for Lot Size {lot_size}: {total_pnl:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Golden Cross Backtest with Futures Rolling")
    parser.add_argument("ticker", type=str, help="Stock Ticker (e.g., RELIANCE.NS, AAPL)")
    parser.add_argument("--lot_size", type=int, default=1, help="Lot Size (default: 1)")
    
    args = parser.parse_args()
    
    run_backtest(args.ticker, args.lot_size)

