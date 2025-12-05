# Stock Research & Technical Analysis

A comprehensive Python-based stock technical analysis toolkit featuring multiple indicators, relative strength analysis, and an integrated web application for interactive visualization and analysis.

> 📱 **Quick Start**: Jump straight to the [Web Application](#-web-application) section to start analyzing stocks with an interactive UI!

## 📊 Overview

This project provides a complete suite of technical analysis tools organized into four main components:

1. **Lagging Indicator Analysis** - Trend-following and momentum indicators
2. **Leading Indicator Analysis** - Divergence-based predictive signals
3. **Market Analysis** - Sector analysis, batch analysis, and relative strength tools
4. **Web Interfaces** - Two Flask-based UIs (classic and modern dashboard)

## 🌐 Web Interfaces

**The easiest way to use this project is through one of the two web interfaces!**

### Option 1: Modern Dashboard UI (Recommended)

A sleek, modern interface with sidebar navigation and comprehensive market analysis.

👉 **[View detailed Website UI documentation](./website_ui/README.md)**

**Quick Start:**
```bash
cd website_ui
python app.py
# Navigate to http://127.0.0.1:5001
```

**Features:**
- **Modern dashboard layout**: Sidebar with stock watchlist
- **Market Analysis**: Sector analysis, stocks in sector, batch analysis
- **All indicators**: Leading and lagging indicators with RS analysis
- **Multiple timeframes**: 1d, 1wk, 1mo, 1h, 15m
- **Real-time charts**: Auto-generated visualizations
- **Support for global markets**: US stocks, NSE, BSE

### Option 2: Classic Web App

The original tabbed interface with all technical indicators.

👉 **[View detailed Web App documentation](./web_app/README.md)**

**Quick Start:**
```bash
cd web_app
python app.py
# Navigate to http://127.0.0.1:5000
```

**Features:**
- **Two-tier tab interface**: Organized by Lagging/Leading indicators
- **7 Technical Indicators**: All accessible from one place
- **Configurable parameters**: Customize each indicator
- **Multiple timeframes**: 1d, 1wk, 1mo, 1h, 15m
- **Real-time charts**: Auto-generated visualizations
- **Support for global markets**: US stocks, NSE, BSE

## 📈 Technical Indicators

### Lagging Indicators (Trend-Following)

Located in `lagging_indicator_analysis/`

#### 1. MACD Analysis
- **File**: [`macd_analysis.py`](./lagging_indicator_analysis/macd_analysis.py)
- **Type**: Momentum oscillator
- **Signals**: Trend, momentum, crossovers, divergences
- **Use Case**: Identify trend strength and potential reversals

#### 2. Supertrend Analysis
- **File**: [`supertrend_analysis.py`](./lagging_indicator_analysis/supertrend_analysis.py)
- **Type**: ATR-based trend follower
- **Signals**: Uptrend/Downtrend, dynamic support/resistance
- **Use Case**: Clear trend identification with stop-loss levels

#### 3. Bollinger Bands
- **File**: [`bollinger_band_analysis.py`](./lagging_indicator_analysis/bollinger_band_analysis.py)
- **Type**: Volatility bands
- **Signals**: Overbought/Oversold, squeeze, breakouts
- **Use Case**: Mean reversion and volatility expansion trades

#### 4. EMA Crossover
- **File**: [`crossover_analysis.py`](./lagging_indicator_analysis/crossover_analysis.py)
- **Type**: Multi-timeframe moving averages
- **Signals**: Golden Cross, Death Cross, trend alignment
- **Use Case**: Long-term trend confirmation

#### 5. Donchian Channels
- **File**: [`donchian_channel_analysis.py`](./lagging_indicator_analysis/donchian_channel_analysis.py)
- **Type**: Breakout indicator (Turtle Trading)
- **Signals**: Upper/Lower channel breakouts
- **Use Case**: Breakout trading and channel-based entries

### Leading Indicators (Predictive)

Located in `leading_indicator_analysis/`

#### 1. RSI Divergence
- **File**: [`rsi_divergence_analysis.py`](./leading_indicator_analysis/rsi_divergence_analysis.py)
- **Type**: RSI-based divergence detector
- **Signals**: Bullish/Bearish divergences
- **Use Case**: Early reversal detection

#### 2. RSI-Volume Divergence
- **File**: [`rsi_volume_divergence.py`](./leading_indicator_analysis/rsi_volume_divergence.py)
- **Type**: Combined RSI and volume analysis
- **Signals**: Multi-dimensional divergences, early reversals
- **Use Case**: High-confidence reversal signals with volume confirmation

#### 3. Volatility Squeeze
- **File**: [`volatility_squeeze_analysis.py`](./leading_indicator_analysis/volatility_squeeze_analysis.py)
- **Type**: Bollinger Bands + Keltner Channels
- **Signals**: Squeeze detection, breakout anticipation
- **Use Case**: Low volatility to high volatility transitions

#### 4. Relative Strength (RS) Analysis
- **File**: [`rs_analysis.py`](./leading_indicator_analysis/rs_analysis.py)
- **Type**: Multi-timeframe momentum comparison
- **Features**:
  - RS calculation across 1M, 2M, 3M, 6M, 1Y, 3Y, 5Y
  - Momentum pattern detection
  - Early turnaround signals
  - Automated ranking system
- **Use Case**: Compare stocks within sectors, identify relative winners

### Market Analysis Tools

Located in `market_analysis/`

#### 1. Sector Analysis
- **File**: [`sector_analysis.py`](./market_analysis/sector_analysis.py)
- **Type**: Sector-wide relative strength analysis
- **Features**:
  - Analyze all stocks within a sector
  - Identify consistent leaders, emerging momentum, and turnarounds
  - Technical confirmation filters (MA breakout, volume surge)
  - Composite scoring and ranking
- **Use Case**: Find the best stocks within a sector

#### 2. Stocks in Sector Analysis
- **File**: [`stock_in_sector_analysis.py`](./market_analysis/stock_in_sector_analysis.py)
- **Type**: Individual stock analysis within sector context
- **Features**:
  - Compare individual stock to sector index
  - Relative performance metrics
- **Use Case**: Evaluate stock performance vs sector benchmark

#### 3. Batch Analysis
- **File**: [`batch_analysis.py`](./market_analysis/batch_analysis.py)
- **Type**: Multi-stock analysis engine
- **Features**:
  - Analyze multiple stocks simultaneously
  - Apply technical indicators across watchlists
  - Generate comparative reports
- **Use Case**: Screen and compare multiple stocks efficiently

## 🚀 Installation

### Prerequisites
```bash
Python 3.8 or higher
```

### Install Dependencies
```bash
pip install pandas numpy yfinance matplotlib scipy flask
```

Or create a `requirements.txt`:
```txt
pandas
numpy
yfinance
matplotlib
scipy
flask
```

Then install:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Option 1: Modern Dashboard UI (Recommended)
```bash
cd website_ui
python app.py
```
Then navigate to `http://127.0.0.1:5001` in your browser.

**See [Website UI README](./website_ui/README.md) for detailed usage instructions.**

### Option 2: Classic Web App
```bash
cd web_app
python app.py
```
Then navigate to `http://127.0.0.1:5000` in your browser.

**See [Web App README](./web_app/README.md) for detailed usage instructions.**

### Option 2: Standalone Python Scripts
```bash
# MACD Analysis
python lagging_indicator_analysis/macd_analysis.py

# Supertrend Analysis  
python lagging_indicator_analysis/supertrend_analysis.py

# Bollinger Bands
python lagging_indicator_analysis/bollinger_band_analysis.py

# Crossover Analysis
python lagging_indicator_analysis/crossover_analysis.py

# Donchian Channels
python lagging_indicator_analysis/donchian_channel_analysis.py

# RSI Divergence
python leading_indicator_analysis/rsi_divergence_analysis.py

# RSI-Volume Divergence
python leading_indicator_analysis/rsi_volume_divergence.py

# Volatility Squeeze
python leading_indicator_analysis/volatility_squeeze_analysis.py

# Relative Strength Analysis
python leading_indicator_analysis/rs_analysis.py

# Sector Analysis
python market_analysis/sector_analysis.py

# Batch Analysis
python market_analysis/batch_analysis.py
```

### Option 3: Import as Module
```python
from lagging_indicator_analysis.macd_analysis import run_analysis as run_macd

# Configure and run
config = {
    'FAST': 12,
    'SLOW': 26,
    'SIGNAL': 9,
    'INTERVAL': '1d',
    'LOOKBACK_PERIODS': 730
}

results = run_macd(ticker="AAPL", config=config)
print(results)
```

## 📁 Project Structure

```
stock_research/
├── README.md                          # This file - Project overview
├── lagging_indicator_analysis/        # Trend-following indicators
│   ├── macd_analysis.py               # MACD
│   ├── supertrend_analysis.py         # Supertrend
│   ├── bollinger_band_analysis.py     # Bollinger Bands
│   ├── crossover_analysis.py          # EMA Crossover
│   └── donchian_channel_analysis.py   # Donchian Channels
├── leading_indicator_analysis/        # Predictive indicators
│   ├── rsi_divergence_analysis.py     # RSI Divergence
│   ├── rsi_volume_divergence.py       # RSI-Volume Divergence
│   ├── volatility_squeeze_analysis.py # Volatility Squeeze
│   └── rs_analysis.py                 # Relative Strength Analysis
├── market_analysis/                   # Market-wide analysis tools
│   ├── sector_analysis.py             # Sector RS analysis
│   ├── stock_in_sector_analysis.py    # Stock vs sector comparison
│   └── batch_analysis.py              # Multi-stock batch analysis
├── web_app/                           # Classic Flask web application
│   ├── README.md                      # Web app documentation
│   ├── app.py                         # Flask backend (port 5000)
│   ├── verify_api.py                  # API testing utility
│   ├── templates/
│   │   └── index.html                 # Web UI
│   └── static/
│       ├── style.css                  # Styling
│       └── script.js                  # Frontend logic
├── website_ui/                        # Modern dashboard web interface
│   ├── README.md                      # Website UI documentation
│   ├── app.py                         # Flask backend (port 5001)
│   ├── templates/
│   │   ├── base.html                  # Base layout with sidebar
│   │   ├── dashboard.html             # Main analysis dashboard
│   │   ├── market_analysis.html       # Market analysis page
│   │   ├── batch_analysis.html        # Batch analysis page
│   │   └── partials/                  # Reusable analysis panels
│   └── static/
│       ├── style.css                  # Modern styling
│       └── script.js                  # Dashboard logic
├── data/
│   ├── tickers_list.json              # Stock ticker lists
│   ├── tickers_grouped.json           # Sector-grouped tickers
│   ├── tickers_batch_analysis.json    # Batch analysis configs
│   └── watchlist.json                 # User watchlist
├── docs/
│   └── todo.txt                       # Project roadmap
└── stock_analysis.py                  # Legacy analysis script
```

## 🎯 Use Cases & Trading Strategies

### 1. Trend Following
**Indicators**: MACD, Supertrend, EMA Crossover  
**Strategy**: Enter on Golden Cross, exit on Death Cross, confirm with MACD crossover

### 2. Breakout Trading
**Indicators**: Donchian Channels, Bollinger Bands  
**Strategy**: Enter on upper channel breakout with volume confirmation

### 3. Mean Reversion
**Indicators**: Bollinger Bands, RSI  
**Strategy**: Buy at lower band with bullish RSI divergence

### 4. Multi-Indicator Confluence
**Indicators**: All indicators  
**Strategy**: Wait for 3+ indicators to align before entering

### 5. Early Reversal Detection
**Indicators**: RSI Divergence, RSI-Volume Divergence  
**Strategy**: Enter early on divergence signals before trend change

### 6. Relative Strength Stock Selection
**Tool**: RS Analysis  
**Strategy**: Select top RS stocks from sector, then apply technical analysis

## 🔧 Configuration

Each analysis module accepts a configuration dictionary. Example:

```python
# MACD Configuration
macd_config = {
    'FAST': 12,           # Fast EMA period
    'SLOW': 26,           # Slow EMA period
    'SIGNAL': 9,          # Signal line period
    'INTERVAL': '1d',     # Data interval (1d, 1wk, 1mo, 1h, 15m)
    'LOOKBACK_PERIODS': 730  # Days of historical data
}

# Supertrend Configuration
supertrend_config = {
    'PERIOD': 14,         # ATR period
    'MULTIPLIER': 3.0,    # ATR multiplier
    'INTERVAL': '1d',
    'LOOKBACK_PERIODS': 730
}

# Bollinger Bands Configuration
bollinger_config = {
    'WINDOW': 20,         # Moving average window
    'NUM_STD': 2,         # Number of standard deviations
    'INTERVAL': '1d',
    'LOOKBACK_PERIODS': 730
}
```

See [Web App README](./web_app/README.md) for complete configuration reference.

## 📊 Data Source

- **Provider**: Yahoo Finance (via `yfinance` library)
- **Coverage**: Global stocks, indices, ETFs, cryptocurrencies
- **Intervals**: 1m, 2m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo
- **Historical Data**: Up to several years depending on interval

## 🛠️ Technical Stack

- **Language**: Python 3.8+
- **Web Framework**: Flask
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib
- **Market Data**: yfinance
- **Signal Detection**: SciPy
- **Frontend**: HTML, CSS, JavaScript

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional technical indicators
- Backtesting framework
- Portfolio analysis
- Real-time alerts
- Mobile app
- Database integration

## ⚠️ Disclaimer

**This tool is for educational and research purposes only.**

- Not financial advice
- No guarantee of accuracy or profitability
- Past performance ≠ future results
- Always do your own research
- Consult a financial advisor before trading

## 📧 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

## 📝 License

For personal use only.

---

**Last Updated**: December 2024  
**Version**: 2.0

**Made with ❤️ for stock market enthusiasts and technical analysts**
