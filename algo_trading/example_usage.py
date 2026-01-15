from zerodha_manager import KiteManager
import os
import pandas as pd
from dotenv import load_dotenv

def print_positions_table(positions):
    """Prints Zerodha positions in a clean table format."""
    net_positions = positions.get('net', [])
    if not net_positions:
        print("No net positions found.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(net_positions)
    
    # Select columns
    cols = ['tradingsymbol', 'exchange', 'quantity', 'average_price', 'last_price', 'pnl']
    df = df[cols].copy()
    
    # Rename for professional look
    df.columns = ['Symbol', 'Exch', 'Qty', 'Avg Price', 'LTP', 'PnL']
    
    # Format numbers
    df['Avg Price'] = df['Avg Price'].map('{:,.2f}'.format)
    df['LTP'] = df['LTP'].map('{:,.2f}'.format)
    df['PnL'] = df['PnL'].map('{:,.2f}'.format)
    
    print("\n" + "="*85)
    print(f"{'NET POSITIONS':^85}")
    print("="*85)
    print(df.to_string(index=False, justify='center'))
    print("="*85 + "\n")

def main():
    # Load environment variables
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

    # Initialize the manager
    try:
        manager = KiteManager()
        print("Kite Manager initialized successfully.")
    except ValueError as e:
        print(f"Initialization failed: {e}")
        return

    # 1. Fetch Positions
    try:
        positions = manager.get_positions()
        print_positions_table(positions)
    except Exception as e:
        print(f"Error fetching positions: {e}")

    # 2. Fetch LTP (Market Data)
    try:
        symbols = ["NSE:SBIN"]
        ltp_data = manager.get_ltp(symbols)
        if ltp_data:
            print(f"LTP for SBIN: {ltp_data.get('NSE:SBIN', {}).get('last_price')}")
    except Exception as e:
        print(f"Note: Could not fetch LTP ({e}).")
        print("This usually happens if your Kite Connect API subscription doesn't have Market Data permissions enabled.")

if __name__ == "__main__":
    main()
