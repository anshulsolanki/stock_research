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

import numpy as np
import pandas as pd
import yfinance as yf
from pykalman import KalmanFilter
from sklearn.metrics import mean_squared_error
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib

class KalmanStrategy:
    """
    Refined Kalman Filter implementation using EM Algorithm for parameter estimation.
    Designed to prevent price-hugging and provide stable trading signals.
    """
    
    BREAKOUT_DETECTION = 'Breakout_Detection'
    TREND_FOLLOWING = 'Trend_Following'
    
    def __init__(self, ticker: str, mode: str = TREND_FOLLOWING, period: str = '2y'):
        self.ticker = ticker
        self.mode = mode
        self.period = period
        self.data = None
        self.prices = None
        self.filtered_state = None
        self.optimal_Q = None
        self.optimal_R = None

    def fetch_data(self):
        print(f"Fetching data for {self.ticker}...")
        stock = yf.Ticker(self.ticker)
        self.data = stock.history(period=self.period)
        self.prices = self.data['Close'].values.reshape(-1, 1)
        return self.data

    def run_analysis(self, em_iterations: int = 5):
        """
        Uses EM Algorithm to self-calibrate Q and R, then filters price.
        """
        if self.prices is None:
            self.fetch_data()

        print(f"Running EM Algorithm ({em_iterations} iterations) for self-calibration...")

        # 1. Initialize KF with conservative 'seed' values
        # Low transition_covariance (Q) forces the EM to look for stable trends
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=self.prices[0],
            initial_state_covariance=1.0,
            observation_covariance=1.0,
            transition_covariance=0.0001 
        )

        # 2. Perform EM optimization
        # This solves the 'overfitting' problem by fitting parameters to the distribution
        kf = kf.em(self.prices, n_iter=em_iterations)
        
        # Store the learned parameters for readout
        # pykalman can return these as scalars or small arrays; ensure we store scalars
        self.optimal_Q = np.atleast_1d(kf.transition_covariance).flatten()[0]
        self.optimal_R = np.atleast_1d(kf.observation_covariance).flatten()[0]

        # 3. Apply the final filter
        self.filtered_state, _ = kf.filter(self.prices)
        
        print(f"✓ Calibration Complete. Learned Q: {self.optimal_Q:.2e}, R: {self.optimal_R:.2e}")
        return self.filtered_state

    def calculate_atr(self, window: int = 14) -> float:
        """
        Calculate the Average True Range (ATR) to measure current volatility.
        """
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean().iloc[-1]
        return atr

    def get_trading_signal(self, atr_multiplier: float = 1.5) -> str:
        """
        Generate signals using a Volatility-Adjusted Buffer (ATR-based).
        
        Parameters:
        -----------
        atr_multiplier : float
            How many ATRs away from the filter the price must be to trigger a signal.
            1.5 to 2.0 is standard for swing trading.
        """
        if self.filtered_state is None:
            raise ValueError("Run run_analysis() first.")
            
        current_price = self.prices[-1, 0]
        filtered_price = self.filtered_state[-1, 0]
        
        # Calculate Volatility-Adjusted Buffer
        atr = self.calculate_atr()
        dynamic_buffer = atr * atr_multiplier
        
        upper_threshold = filtered_price + dynamic_buffer
        lower_threshold = filtered_price - dynamic_buffer
        
        # Momentum check (Direction of the Kalman line)
        lookback = 10
        kalman_trend = self.filtered_state[-1] - self.filtered_state[-lookback]
        
        if self.mode == self.BREAKOUT_DETECTION:
            # Price must close outside the ATR-band to confirm a breakout
            if current_price > upper_threshold and kalman_trend > 0:
                return 'BUY'
            elif current_price < lower_threshold and kalman_trend < 0:
                return 'SELL'
        else:
            # Trend Following: Exit only if price aggressively violates the ATR-band
            if kalman_trend > 0 and current_price > lower_threshold:
                return 'BUY'
            elif kalman_trend < 0 and current_price < upper_threshold:
                return 'SELL'
                
        return 'HOLD'

    def plot_results(self, save_path: Optional[str] = None):
        """
        Visualizes the Kalman Filter price vs Actual price.
        """
        if self.filtered_state is None:
            raise ValueError("No analysis results. Run run_analysis() first.")

        plt.figure(figsize=(14, 7))
        dates = self.data.index
        
        # Plot Actual Price
        plt.plot(dates, self.prices.flatten(), label='Actual Price', 
                 color='black', linestyle=':', alpha=0.6, linewidth=1.5, zorder=1)
        
        # Plot Kalman Filter
        plt.plot(dates, self.filtered_state.flatten(), label='Kalman Filter (EM)', 
                 color='yellow', linewidth=2.5, zorder=2)
        
        # Calculate EMAs for context
        close_series = pd.Series(self.prices.flatten(), index=dates)
        ema_20 = close_series.ewm(span=20).mean()
        ema_50 = close_series.ewm(span=50).mean()
        ema_200 = close_series.ewm(span=200).mean()
        
        plt.plot(dates, ema_20, label='EMA 20', color='green', alpha=0.4, zorder=0)
        plt.plot(dates, ema_50, label='EMA 50', color='blue', alpha=0.4, zorder=0)
        plt.plot(dates, ema_200, label='EMA 200', color='red', alpha=0.4, zorder=0)

        # Signal annotation
        signal = self.get_trading_signal()
        signal_colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'orange'}
        plt.text(0.02, 0.95, f'Signal: {signal}', transform=plt.gca().transAxes,
                 fontsize=14, fontweight='bold', bbox=dict(facecolor=signal_colors.get(signal, 'white'), alpha=0.3))

        plt.title(f'Kalman Filter Analysis (EM Optimized) - {self.ticker}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.show()

# ==================== EXAMPLE USAGE ====================

def main():
    """
    Example usage demonstrating both operational modes.
    """
    # Define ticker
    TICKER = 'HDFCBANK.NS'
    
    print("\n" + "="*70)
    print(" ADAPTIVE KALMAN FILTER FOR TRADING - DEMONSTRATION")
    print("="*70 + "\n")
    
    # ========== MODE 1: BREAKOUT DETECTION ==========
    print("\n📊 MODE 1: BREAKOUT DETECTION")
    print("-" * 70)
    
    strategy_breakout = KalmanStrategy(
        ticker=TICKER,
        mode=KalmanStrategy.BREAKOUT_DETECTION,
        period='3y'
    )
    
    # Run analysis with EM calibration
    results_breakout = strategy_breakout.run_analysis(em_iterations=5)
    
    # Get trading signal
    signal = strategy_breakout.get_trading_signal()
    print(f"\n🎯 Trading Signal: {signal}")
    
    # Plot results
    strategy_breakout.plot_results()
    
    # ========== MODE 2: TREND FOLLOWING ==========
    print("\n\n📊 MODE 2: TREND FOLLOWING")
    print("-" * 70)
    
    strategy_trend = KalmanStrategy(
        ticker=TICKER,
        mode=KalmanStrategy.TREND_FOLLOWING,
        period='3y'
    )
    
    # Run analysis with EM calibration
    results_trend = strategy_trend.run_analysis(em_iterations=5)
    
    # Get trading signal
    signal = strategy_trend.get_trading_signal()
    print(f"\n🎯 Trading Signal: {signal}")
    
    # Plot results
    strategy_trend.plot_results()
    
    # ========== COMPARISON ==========
    print("\n\n📈 MODE COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<30} {'Breakout Detection':<20} {'Trend Following':<20}")
    print("-" * 70)
    print(f"{'Optimal Q':<30} {strategy_breakout.optimal_Q:<20.2e} {strategy_trend.optimal_Q:<20.2e}")
    print(f"{'Optimal R':<30} {strategy_breakout.optimal_R:<20.2e} {strategy_trend.optimal_R:<20.2e}")
    # Calculate simple price change for comparison
    start_price = strategy_breakout.prices[0,0]
    end_price = strategy_breakout.prices[-1,0]
    change_pct = (end_price - start_price) / start_price * 100
    print(f"{'Price Change %':<30} {change_pct:<20.2f} {change_pct:<20.2f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
