from zerodha_manager import KiteManager
import os
import pandas as pd
from dotenv import load_dotenv

# To get access token , type this in browser
# 1. https://kite.zerodha.com/connect/login?v=3&api_key=YOUR_API_KEY
# 2. Something like this will appear - http://127.0.0.1/?action=login&type=login&status=success&request_token=IgNibLTGF3lq7O6rywHwXmxjbgYtKfuA
# 3. Copy request token and place it in generate_access_token.py line 14.
# 4. Run generate_access_token.py , you will get access token in console print
# 5. Copy access token and replace it in .env file

REQUEST_TOKEN = "zj3VoUCPP0X3QECUgv6dtRL8dFbMXiRF" #change it daily following aboev process

def main():
    # Load environment variables
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

    # Initialize the manager
    try:
        manager = KiteManager()
        print("Kite Manager initialized successfully.")
        manager.generate_session(REQUEST_TOKEN)
    except ValueError as e:
        print(f"Initialization failed: {e}")
        return

if __name__ == "__main__":
    main()