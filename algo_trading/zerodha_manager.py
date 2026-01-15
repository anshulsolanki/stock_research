import os
import logging
from kiteconnect import KiteConnect
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KiteManager:
    """
    A manager class to handle Zerodha Kite Connect API interactions.
    """
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("ZERODHA_API_KEY")
        self.api_secret = os.getenv("ZERODHA_API_SECRET")
        self.access_token = os.getenv("ZERODHA_ACCESS_TOKEN")
        
        if not self.api_key or not self.api_secret:
            logger.error("ZERODHA_API_KEY and ZERODHA_API_SECRET must be set in environment variables.")
            raise ValueError("Missing Zerodha API credentials.")
            
        self.kite = KiteConnect(api_key=self.api_key)
        
        if self.access_token:
            self.kite.set_access_token(self.access_token)
            logger.info("Kite session initialized with provided access token.")

    def get_login_url(self):
        """
        Generates the login URL to get the request token.
        """
        return self.kite.login_url()

    def generate_session(self, request_token):
        """
        Generates a session by exchanging the request token for an access token.
        """
        try:
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            self.access_token = data["access_token"]
            print(self.access_token)
            self.kite.set_access_token(self.access_token)
            logger.info("Session generated successfully.")
            return data
        except Exception as e:
            logger.error(f"Error generating session: {e}")
            raise

    def place_order(self, symbol, exchange, transaction_type, quantity, order_type, product, price=None, trigger_price=None):
        """
        Places an order on Zerodha Kite.
        """
        try:
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=exchange,
                tradingsymbol=symbol,
                transaction_type=transaction_type,
                quantity=quantity,
                product=product,
                order_type=order_type,
                price=price,
                trigger_price=trigger_price
            )
            logger.info(f"Order placed successfully. Order ID: {order_id}")
            return order_id
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise

    def get_positions(self):
        """
        Retrieves current portfolio positions.
        """
        try:
            return self.kite.positions()
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            raise

    def get_holdings(self):
        """
        Retrieves current portfolio holdings.
        """
        try:
            return self.kite.holdings()
        except Exception as e:
            logger.error(f"Error fetching holdings: {e}")
            raise

    def get_ltp(self, instruments):
        """
        Retrieves Last Traded Price (LTP) for given instruments.
        Example instruments: ["NSE:RELIANCE", "NSE:INFY"]
        """
        try:
            return self.kite.ltp(instruments)
        except Exception as e:
            logger.error(f"Error fetching LTP: {e}")
            raise
