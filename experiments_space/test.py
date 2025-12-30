import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

# 1. Download Stock Data
data = yf.download("MARICO.NS", start="2020-01-01", end="2025-12-25")
prices = data['Close']

# 2. Construct the Kalman Filter
# Parameters:
# transition_matrices: How the state evolves (Price_t = Price_t-1)
# observation_matrices: How we observe the state (Measured_t = True_t)
kf = KalmanFilter(
    transition_matrices = [1],
    observation_matrices = [1],
    initial_state_mean = 0,
    initial_state_covariance = 1,
    observation_covariance = 1,     # Trust in the measurement (lower = more trust in price)
    transition_covariance = 0.0001  # Trust in the model (lower = smoother line)
)

# 3. Use EM algorithm to optimize parameters based on data
kf = kf.em(prices.values, n_iter=5)

# 4. Apply the filter to get smoothed means and covariances
state_means, state_covs = kf.filter(prices.values)
kf_line = pd.Series(state_means.flatten(), index=prices.index)

# 5. Calculate a standard 20-day SMA for comparison
sma_20 = prices.rolling(window=20).mean()
sma_50 = prices.rolling(window=50).mean()
sma_200 = prices.rolling(window=200).mean()

# 6. Visualization
plt.figure(figsize=(14, 7))
plt.plot(prices, label='Actual Price (Noisy)', alpha=0.4, color='black')
plt.plot(sma_20, label='20-Day SMA (Lagging)', color='green', linestyle='--')
plt.plot(sma_50, label='50-Day SMA (Lagging)', color='blue', linestyle='--')
plt.plot(sma_200, label='200-Day SMA (Lagging)', color='red', linestyle='--')
plt.plot(kf_line, label='Kalman Filter (Responsive)', color='yellow', linewidth=2)

plt.title('Kalman Filter vs. Simple Moving Average (SMA)')
plt.legend()
plt.show()