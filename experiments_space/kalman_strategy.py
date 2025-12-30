"""
Adaptive Kalman Filter for Financial Time Series
Version: 1.0

A recursive state-space model designed to filter market noise and provide 
zero-lag trend analysis for stock prices.

Dependencies: pykalman, yfinance, pandas, numpy, matplotlib, sklearn
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from sklearn.metrics import mean_squared_error
from typing import Tuple, Dict, Optional
import warnings
import os
warnings.filterwarnings('ignore')


class KalmanStrategy:
    """
    Production-ready Kalman Filter implementation for stock market analysis.
    
    Supports two operational modes:
    - Breakout Detection: High Q, Low R (responsive to trend changes)
    - Trend Following: Low Q, High R (smooth trend continuation)
    """
    
    # ==================== CONFIGURATION ====================
    
    # Operational Modes
    BREAKOUT_DETECTION = 'Breakout_Detection'
    TREND_FOLLOWING = 'Trend_Following'
    
    # Default parameter ranges for grid search
    Q_RANGE = (1e-6, 0.1)  # Process noise covariance
    R_RANGE = (0.01, 100)   # Measurement noise covariance
    
    # Default Kalman Filter matrices
    DEFAULT_TRANSITION_MATRIX = np.array([[1]])  # F - Random Walk model
    DEFAULT_OBSERVATION_MATRIX = np.array([[1]]) # H - Direct price mapping
    
    def __init__(
        self,
        ticker: str,
        mode: str = TREND_FOLLOWING,
        period: str = '1y',
        interval: str = '1d'
    ):
        """
        Initialize the Kalman Strategy.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol (e.g., 'DABUR.NS', 'AAPL')
        mode : str
            Operational mode: 'Breakout_Detection' or 'Trend_Following'
        period : str
            Data period for yfinance (e.g., '1y', '6mo', '2y')
        interval : str
            Data interval (e.g., '1d', '1h', '15m')
        """
        self.ticker = ticker
        self.mode = mode
        self.period = period
        self.interval = interval
        
        # Data storage
        self.data = None
        self.prices = None
        self.filtered_state = None
        
        # Optimal parameters from grid search
        self.optimal_Q = None
        self.optimal_R = None
        self.optimization_results = None
        
        # Validate mode
        if mode not in [self.BREAKOUT_DETECTION, self.TREND_FOLLOWING]:
            raise ValueError(
                f"Mode must be '{self.BREAKOUT_DETECTION}' or '{self.TREND_FOLLOWING}'"
            )
    
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch stock data using yfinance.
        
        Returns:
        --------
        pd.DataFrame
            Stock data with OHLCV columns
        """
        print(f"Fetching data for {self.ticker}...")
        
        try:
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(period=self.period, interval=self.interval)
            
            if self.data.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")
            
            self.prices = self.data['Close'].values.reshape(-1, 1)
            
            print(f"✓ Fetched {len(self.data)} data points")
            print(f"  Date range: {self.data.index[0]} to {self.data.index[-1]}")
            
            return self.data
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch data for {self.ticker}: {str(e)}")
    
    def apply_kalman_filter(
        self,
        Q: float,
        R: float,
        prices: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Kalman Filter with specified Q and R parameters.
        
        Parameters:
        -----------
        Q : float
            Process noise covariance (Adaptability Knob)
        R : float
            Measurement noise covariance (Smoothness Knob)
        prices : np.ndarray, optional
            Price data to filter. If None, uses self.prices
        
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            (filtered_state_means, filtered_state_covariances)
        """
        if prices is None:
            if self.prices is None:
                raise ValueError("No price data available. Call fetch_data() first.")
            prices = self.prices
        
        # Initialize Kalman Filter with specified parameters
        kf = KalmanFilter(
            transition_matrices=self.DEFAULT_TRANSITION_MATRIX,
            observation_matrices=self.DEFAULT_OBSERVATION_MATRIX,
            initial_state_mean=prices[0],
            initial_state_covariance=1.0,
            transition_covariance=Q * np.eye(1),
            observation_covariance=R * np.eye(1)
        )
        
        # Apply the filter using the Prediction-Update loop
        filtered_state_means, filtered_state_covariances = kf.filter(prices)
        
        return filtered_state_means, filtered_state_covariances
    
    # def calculate_mse(
    #     self,
    #     Q: float,
    #     R: float,
    #     prices: Optional[np.ndarray] = None
    # ) -> float:
    #     """
    #     Calculate Mean Squared Error for given Q and R parameters.
        
    #     Parameters:
    #     -----------
    #     Q : float
    #         Process noise covariance
    #     R : float
    #         Measurement noise covariance
    #     prices : np.ndarray, optional
    #         Price data. If None, uses self.prices
        
    #     Returns:
    #     --------
    #     float
    #         Mean Squared Error between actual and filtered prices
    #     """
    #     if prices is None:
    #         prices = self.prices
        
    #     filtered_state_means, _ = self.apply_kalman_filter(Q, R, prices)
    #     mse = mean_squared_error(prices, filtered_state_means)
        
    #     return mse
    def calculate_optimized_cost(
        self,
        Q: float,
        R: float,
        prices: Optional[np.ndarray] = None,
        lambda_param: float = 0.5  # Adjust this: higher = smoother, lower = tracks price more
    ) -> float:
        """
        Calculate a penalized cost function that balances accuracy (MSE) 
        and smoothness (Total Variation).
        """
        if prices is None:
            prices = self.prices
        
        # 1. Get filtered prices
        filtered_state_means, _ = self.apply_kalman_filter(Q, R, prices)
        filtered_flat = filtered_state_means.flatten()
        
        # 2. Calculate Standard MSE (Accuracy)
        mse = mean_squared_error(prices, filtered_state_means)
        
        # 3. Calculate Smoothness Penalty (Total Variation)
        # This measures how much the filtered line wiggles
        smoothness_penalty = np.sum(np.abs(np.diff(filtered_flat)))
        
        # 4. Normalize the penalty to be on a similar scale to price
        normalized_penalty = smoothness_penalty / len(filtered_flat)
        
        # 5. Combined Cost
        total_cost = mse + (lambda_param * normalized_penalty)
        
        return total_cost
    
    def grid_search_optimize(
        self,
        q_steps: int = 20,
        r_steps: int = 20,
        q_range: Optional[Tuple[float, float]] = None,
        r_range: Optional[Tuple[float, float]] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Perform Grid Search to find optimal Q and R parameters.
        
        Parameters:
        -----------
        q_steps : int
            Number of grid points for Q (default: 20)
        r_steps : int
            Number of grid points for R (default: 20)
        q_range : Tuple[float, float], optional
            Custom Q range (min, max). Uses default if None.
        r_range : Tuple[float, float], optional
            Custom R range (min, max). Uses default if None.
        verbose : bool
            Print progress information
        
        Returns:
        --------
        Dict
            Optimization results containing optimal Q, R, MSE, and full grid
        """
        if self.prices is None:
            raise ValueError("No price data available. Call fetch_data() first.")
        
        if verbose:
            print("\n" + "="*60)
            print(f"GRID SEARCH OPTIMIZATION - {self.mode}")
            print("="*60)
        
        # Use custom ranges or defaults
        q_min, q_max = q_range if q_range else self.Q_RANGE
        r_min, r_max = r_range if r_range else self.R_RANGE
        
        # Create logarithmic search space
        q_values = np.logspace(np.log10(q_min), np.log10(q_max), q_steps)
        r_values = np.logspace(np.log10(r_min), np.log10(r_max), r_steps)
        
        if verbose:
            print(f"\nSearch Space:")
            print(f"  Q range: {q_min:.2e} to {q_max:.2e} ({q_steps} steps)")
            print(f"  R range: {r_min:.2e} to {r_max:.2e} ({r_steps} steps)")
            print(f"  Total combinations: {q_steps * r_steps}")
            print("\nSearching for optimal parameters...")
        
        # Grid search
        best_mse = float('inf')
        best_Q = None
        best_R = None
        mse_grid = np.zeros((len(q_values), len(r_values)))
        
        total_iterations = len(q_values) * len(r_values)
        iteration = 0
        
        for i, Q in enumerate(q_values):
            for j, R in enumerate(r_values):
                iteration += 1
                
                try:
                    mse = self.calculate_optimized_cost(Q, R)
                    mse_grid[i, j] = mse
                    
                    if mse < best_mse:
                        best_mse = mse
                        best_Q = Q
                        best_R = R
                    
                    # Progress indication
                    if verbose and iteration % 50 == 0:
                        progress = (iteration / total_iterations) * 100
                        print(f"  Progress: {progress:.1f}% | Best MSE: {best_mse:.6f}")
                
                except Exception as e:
                    # Handle numerical instability
                    mse_grid[i, j] = np.nan
                    continue
        
        # Store results
        self.optimal_Q = best_Q
        self.optimal_R = best_R
        
        self.optimization_results = {
            'optimal_Q': best_Q,
            'optimal_R': best_R,
            'best_mse': best_mse,
            'q_values': q_values,
            'r_values': r_values,
            'mse_grid': mse_grid,
            'mode': self.mode
        }
        
        if verbose:
            print("\n" + "="*60)
            print("OPTIMIZATION COMPLETE")
            print("="*60)
            print(f"\n✓ Optimal Parameters Found:")
            print(f"  Q (Process Noise):      {best_Q:.6e}")
            print(f"  R (Measurement Noise):  {best_R:.6e}")
            print(f"  MSE:                    {best_mse:.6f}")
            print(f"\n  Mode Interpretation:")
            
            if self.mode == self.BREAKOUT_DETECTION:
                print(f"    → High Adaptability: {'Yes' if best_Q > 1e-3 else 'Moderate'}")
                print(f"    → Low Smoothness:    {'Yes' if best_R < 1 else 'Moderate'}")
            else:
                print(f"    → Low Adaptability:  {'Yes' if best_Q < 1e-3 else 'Moderate'}")
                print(f"    → High Smoothness:   {'Yes' if best_R > 1 else 'Moderate'}")
            
            print("="*60 + "\n")
        
        return self.optimization_results
    
    def run_analysis(
        self,
        optimize: bool = True,
        Q: Optional[float] = None,
        R: Optional[float] = None
    ) -> Dict:
        """
        Run complete Kalman Filter analysis.
        
        Parameters:
        -----------
        optimize : bool
            If True, run grid search to find optimal Q and R
        Q : float, optional
            Manual Q value (used if optimize=False)
        R : float, optional
            Manual R value (used if optimize=False)
        
        Returns:
        --------
        Dict
            Analysis results including filtered prices and parameters
        """
        # Fetch data if not already done
        if self.data is None:
            self.fetch_data()
        
        # Optimize or use provided parameters
        if optimize:
            self.grid_search_optimize()
            Q_use = self.optimal_Q
            R_use = self.optimal_R
        else:
            if Q is None or R is None:
                raise ValueError("Must provide Q and R values if optimize=False")
            Q_use = Q
            R_use = R
        
        # Apply Kalman Filter with optimal/specified parameters
        print(f"\nApplying Kalman Filter...")
        print(f"  Q = {Q_use:.6e}")
        print(f"  R = {R_use:.6e}")
        
        self.filtered_state, filtered_covariance = self.apply_kalman_filter(Q_use, R_use)
        
        # Calculate performance metrics
        mse = mean_squared_error(self.prices, self.filtered_state)
        price_change = ((self.prices[-1] - self.prices[0]) / self.prices[0] * 100)[0]
        filtered_change = ((self.filtered_state[-1] - self.filtered_state[0]) / self.filtered_state[0] * 100)[0]
        
        results = {
            'ticker': self.ticker,
            'mode': self.mode,
            'data': self.data,
            'prices': self.prices,
            'filtered_prices': self.filtered_state,
            'Q': Q_use,
            'R': R_use,
            'mse': mse,
            'price_change_pct': price_change,
            'filtered_change_pct': filtered_change,
            'optimization_results': self.optimization_results
        }
        
        print(f"✓ Analysis Complete")
        print(f"  MSE: {mse:.6f}")
        print(f"  Actual Price Change: {price_change:.2f}%")
        print(f"  Filtered Trend Change: {filtered_change:.2f}%")
        
        return results
    
    def plot_results(
        self,
        figsize: Tuple[int, int] = (16, 8),
        save_path: Optional[str] = None
    ):
        """
        Visualize Kalman Filter with price, EMAs, and trading signal.
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Figure size (width, height)
        save_path : str, optional
            Path to save the figure. If None, auto-generates filename.
        """
        if self.filtered_state is None:
            raise ValueError("No analysis results. Run run_analysis() first.")
        
        # Calculate EMAs
        close_prices = pd.Series(self.prices.flatten(), index=self.data.index)
        ema_20 = close_prices.ewm(span=20, adjust=False).mean()
        ema_50 = close_prices.ewm(span=50, adjust=False).mean()
        ema_200 = close_prices.ewm(span=200, adjust=False).mean()
        
        # Get trading signal
        signal = self.get_trading_signal()
        
        # Create single plot
        fig, ax = plt.subplots(figsize=figsize)
        dates = self.data.index
        
        # Plot EMAs first (bottom layer)
        ax.plot(dates, ema_200, label='EMA 200', linewidth=2, color='red', alpha=0.7, zorder=1)
        ax.plot(dates, ema_50, label='EMA 50', linewidth=2, color='blue', alpha=0.7, zorder=2)
        ax.plot(dates, ema_20, label='EMA 20', linewidth=2, color='green', alpha=0.7, zorder=3)
        
        # Plot Kalman filter
        ax.plot(dates, self.filtered_state.flatten(), label='Kalman Filter', 
                linewidth=2.5, color='yellow', alpha=0.9, zorder=4)
        
        # Plot actual price on top (most visible)
        ax.plot(dates, self.prices.flatten(), label='Actual Price', 
                alpha=0.9, linewidth=2, color='black', linestyle=':', zorder=5)
        
        # Add trading signal annotation
        signal_colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'orange'}
        signal_color = signal_colors.get(signal, 'black')
        
        ax.text(0.02, 0.98, f'Trading Signal: {signal}', 
                transform=ax.transAxes, fontsize=14, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor=signal_color, alpha=0.3, edgecolor=signal_color, linewidth=2))
        
        # Add parameter info
        param_text = f'Mode: {self.mode}\nQ={self.optimal_Q:.2e}, R={self.optimal_R:.2e}'
        ax.text(0.02, 0.88, param_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Formatting
        ax.set_title(f'{self.ticker} - Kalman Filter Analysis', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Auto-generate filename if not provided
        if save_path is None:
            save_path = f'kalman_filter_{self.ticker}_{self.mode}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n✓ Plot saved to: {save_path}")
    
    def get_trading_signal(self, buffer_pct: float = 0.005) -> str:
        """
        Generate trading signal based on filtered state with a volatility buffer.
        
        Parameters:
        -----------
        buffer_pct : float
            The percentage 'threshold' price must cross over the filter to 
            trigger a signal (e.g., 0.005 for 0.5%).
        
        Returns:
        --------
        str: 'BUY', 'SELL', or 'HOLD'
        """
        if self.filtered_state is None:
            raise ValueError("No analysis results. Run run_analysis() first.")
        
        # Get current values
        current_price = self.prices[-1, 0]
        filtered_price = self.filtered_state[-1, 0]
        
        # Calculate the upper and lower buffer boundaries
        upper_threshold = filtered_price * (1 + buffer_pct)
        lower_threshold = filtered_price * (1 - buffer_pct)
        
        # Calculate recent trend (last 5-10 bars) to ensure momentum
        lookback = min(10, len(self.filtered_state))
        recent_trend = self.filtered_state[-1] - self.filtered_state[-lookback]
        
        if self.mode == self.BREAKOUT_DETECTION:
            # Breakout Strategy: Require price to 'clear' the buffer zone
            if current_price > upper_threshold and recent_trend > 0:
                return 'BUY'
            elif current_price < lower_threshold and recent_trend < 0:
                return 'SELL'
            else:
                return 'HOLD'
                
        else:
            # Trend Following: Stay in trend unless price aggressively crosses back through buffer
            # This prevents exiting a good trade during a minor consolidation
            if recent_trend > 0 and current_price > lower_threshold:
                return 'BUY'
            elif recent_trend < 0 and current_price < upper_threshold:
                return 'SELL'
            else:
                return 'HOLD'


# ==================== EXAMPLE USAGE ====================

def main():
    """
    Example usage demonstrating both operational modes.
    """
    # Define ticker
    TICKER = 'LT.NS'
    
    print("\n" + "="*70)
    print(" ADAPTIVE KALMAN FILTER FOR TRADING - DEMONSTRATION")
    print("="*70 + "\n")
    
    # ========== MODE 1: BREAKOUT DETECTION ==========
    print("\n📊 MODE 1: BREAKOUT DETECTION")
    print("-" * 70)
    
    strategy_breakout = KalmanStrategy(
        ticker=TICKER,
        mode=KalmanStrategy.BREAKOUT_DETECTION,
        period='3y',
        interval='1d'
    )
    
    # Run analysis with optimization
    results_breakout = strategy_breakout.run_analysis(optimize=True)
    
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
        period='3y',
        interval='1d'
    )
    
    # Run analysis with optimization
    results_trend = strategy_trend.run_analysis(optimize=True)
    
    # Get trading signal
    signal = strategy_trend.get_trading_signal()
    print(f"\n🎯 Trading Signal: {signal}")
    
    # Plot results
    #strategy_trend.plot_results()
    
    # ========== COMPARISON ==========
    print("\n\n📈 MODE COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<30} {'Breakout Detection':<20} {'Trend Following':<20}")
    print("-" * 70)
    print(f"{'Optimal Q':<30} {results_breakout['Q']:<20.6e} {results_trend['Q']:<20.6e}")
    print(f"{'Optimal R':<30} {results_breakout['R']:<20.6e} {results_trend['R']:<20.6e}")
    print(f"{'MSE':<30} {results_breakout['mse']:<20.6f} {results_trend['mse']:<20.6f}")
    print(f"{'Filtered Change %':<30} {results_breakout['filtered_change_pct']:<20.2f} {results_trend['filtered_change_pct']:<20.2f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
