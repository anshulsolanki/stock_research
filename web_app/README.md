# Stock Technical Analysis Web Application

A comprehensive Flask-based web application for analyzing stocks using multiple technical indicators, featuring both lagging and leading indicator analysis with an intuitive tabbed interface.

## 🎯 Overview

This is the **classic tabbed interface** for the Stock Technical Analysis platform. It provides professional-grade technical analysis for stocks with real-time chart generation, configurable parameters, and an organized multi-tier tab interface that separates lagging and leading indicators.

> 💡 **Looking for a modern dashboard?** Check out the [Website UI](../website_ui/README.md) which offers a sleek sidebar-based interface with additional market analysis features (sector analysis, batch analysis). It runs on port 5001.

## ✨ Features

### Lagging Indicator Analysis
Access 5 powerful trend-following and momentum indicators:

#### 1. **MACD Analysis** (Moving Average Convergence Divergence)
- **Metrics**: MACD Line, Signal Line, Histogram
- **Signals**: Trend direction, momentum strength, crossover signals
- **Advanced**: Automatic divergence detection (bullish/bearish)
- **Configurable**: Fast (12), Slow (26), Signal (9) periods

#### 2. **Supertrend Analysis**
- **Metrics**: Supertrend value, trend status, price position
- **Signals**: Uptrend/Downtrend identification
- **Configurable**: ATR Period (14), Multiplier (3.0)

#### 3. **Bollinger Bands**
- **Metrics**: Upper/Lower/Middle bands, %B, BandWidth
- **Signals**: Overbought/Oversold, squeeze detection, breakouts
- **Advanced**: Recent signal history with buy/sell recommendations
- **Configurable**: Window (20), Standard Deviations (2)

#### 4. **EMA Crossover Analysis**
- **Metrics**: EMA 20, EMA 50, EMA 200
- **Signals**: Trend status, Golden Cross/Death Cross detection
- **Advanced**: Multi-timeframe trend confirmation
- **Configurable**: Short (20), Medium (50), Long (200) windows

#### 5. **Donchian Channels**
- **Metrics**: Upper/Lower/Middle channels, price position
- **Signals**: Breakout detection (Turtle Trading strategy)
- **Advanced**: Recent breakout history tracking
- **Configurable**: Window period (20)

### Leading Indicator Analysis
Anticipate market moves with 2 advanced divergence-based indicators:

#### 1. **RSI Divergence Analysis**
- **Metrics**: Current RSI value
- **Signals**: Bullish/Bearish divergences
- **Advanced**: Peak and trough pattern detection
- **Configurable**: RSI Period (14), Peak/Trough Order (5), Overbought (70), Oversold (30)

#### 2. **RSI-Volume Divergence**
- **Metrics**: Current RSI, Volume, Volume MA-20, Volume MA-50
- **Signals**: Bullish/Bearish RSI-Volume divergences
- **Advanced**: Early reversal signal detection combining RSI extremes with volume analysis
- **Configurable**: RSI Period (14), Volume MA Short (20), Volume MA Long (50)

### User Interface

- **Two-Tier Tab System**:
  - **Main Tabs**: Lagging vs. Leading Indicator Analysis
  - **Sub Tabs**: Individual indicators within each category
- **Configurable Parameters**: Each indicator has a collapsible configuration panel
- **Independent Analysis**: Run each indicator separately or all at once
- **Multiple Timeframes**: Support for 1d, 1wk, 1mo, 1h, 15m intervals
- **Interactive Charts**: Auto-generated matplotlib visualizations
- **Responsive Design**: Modern gradient-based UI with smooth transitions

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **Dependencies**:
  - Flask
  - matplotlib
  - pandas
  - numpy
  - yfinance
  - scipy

## 🚀 Installation

1. **Navigate to the web_app directory**:
   ```bash
   cd /Users/solankianshul/Documents/projects/stock_research/web_app
   ```

2. **Install required packages** (if not already installed):
   ```bash
   pip install flask matplotlib pandas numpy yfinance scipy
   ```

## 🎮 Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://127.0.0.1:5000
   ```

3. **Enter a stock ticker** (e.g., `AAPL`, `MARICO.NS`) and click "Analyze"

## 🌍 Ticker Format

- **US Stocks**: Use standard ticker symbols
  - Examples: `AAPL`, `MSFT`, `TSLA`, `GOOGL`
  
- **Indian Stocks (NSE)**: Add `.NS` suffix
  - Examples: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`
  
- **Indian Stocks (BSE)**: Add `.BO` suffix
  - Examples: `RELIANCE.BO`, `TATAMOTORS.BO`

## 🏗️ Architecture

### Backend (`app.py`)
- **Framework**: Flask web server
- **Port**: 5000 (localhost)
- **Endpoint**: `/analyze` (POST)
- **Functionality**: 
  - Routes requests to appropriate analysis modules
  - Converts matplotlib figures to base64-encoded images
  - Returns structured JSON responses

### Frontend

#### HTML (`templates/index.html`)
- Two-tier tabbed interface (main tabs + sub tabs)
- Configuration panels for each indicator
- Metric cards for key values
- Chart display sections
- Signal/divergence lists

#### JavaScript (`static/script.js`)
- Tab switching logic (main and sub tabs)
- Configuration panel toggles
- API request handling
- Dynamic result rendering
- Error handling and loading states

#### CSS (`static/style.css`)
- Modern gradient-based design
- Responsive layout
- Color-coded status cards (bullish/bearish)
- Smooth animations and transitions

### Analysis Modules

Located in parent directory `lagging_indicator_analysis/` and `leading_indicator_analysis/`:

- `macd_analysis.py` - MACD indicator calculations
- `supertrend_analysis.py` - Supertrend algorithm
- `bollinger_band_analysis.py` - Bollinger Bands with signals
- `crossover_analysis.py` - EMA crossover detection
- `donchian_channel_analysis.py` - Donchian breakout system
- `rsi_divergence_analysis.py` - RSI divergence detection
- `rsi_volume_divergence.py` - RSI-Volume combined analysis

Each module exports a `run_analysis(ticker, config)` function.

## 🔌 API Reference

### POST `/analyze`

**Request Body**:
```json
{
  "ticker": "AAPL",
  "analysis_type": "all",  // or "macd", "supertrend", "bollinger", "crossover", "donchian", "rsi", "rsi_volume"
  "macd_config": {
    "FAST": 12,
    "SLOW": 26,
    "SIGNAL": 9,
    "INTERVAL": "1d",
    "LOOKBACK_PERIODS": 730
  },
  "supertrend_config": {
    "PERIOD": 14,
    "MULTIPLIER": 3.0,
    "INTERVAL": "1d",
    "LOOKBACK_PERIODS": 730
  },
  "bollinger_config": {
    "WINDOW": 20,
    "NUM_STD": 2,
    "INTERVAL": "1d",
    "LOOKBACK_PERIODS": 730
  },
  "crossover_config": {
    "WINDOWS": [20, 50, 200],
    "INTERVAL": "1d",
    "LOOKBACK_PERIODS": 730
  },
  "donchian_config": {
    "WINDOW": 20,
    "INTERVAL": "1d",
    "LOOKBACK_PERIODS": 730
  },
  "rsi_config": {
    "PERIOD": 14,
    "ORDER": 5,
    "INTERVAL": "1d",
    "LOOKBACK_PERIODS": 730,
    "RSI_OVERBOUGHT": 70,
    "RSI_OVERSOLD": 30
  },
  "rsi_volume_config": {
    "RSI_PERIOD": 14,
    "ORDER": 5,
    "VOLUME_MA_SHORT": 20,
    "VOLUME_MA_LONG": 50,
    "INTERVAL": "1d",
    "LOOKBACK_PERIODS": 730,
    "RSI_OVERBOUGHT": 70,
    "RSI_OVERSOLD": 30
  }
}
```

**Response Structure**:
```json
{
  "success": true,
  "ticker": "AAPL",
  "macd": {
    "macd_line": 5.23,
    "signal_line": 4.89,
    "histogram": 0.34,
    "trend": "Bullish",
    "momentum": "Strengthening",
    "crossover_signal": "Bullish Crossover (Buy)",
    "divergences": [...],
    "chart_image": "base64_encoded_image"
  },
  "supertrend": {
    "status": "Uptrend",
    "last_price": 175.43,
    "supertrend_value": 168.25,
    "last_date": "2024-11-29",
    "last_trend": 1,
    "chart_image": "base64_encoded_image"
  },
  "bollinger": {
    "bb_upper": 180.25,
    "bb_lower": 170.15,
    "sma_20": 175.20,
    "pct_b": 0.65,
    "bandwidth": 0.058,
    "status": "Normal",
    "signals": [...],
    "chart_image": "base64_encoded_image"
  },
  "crossover": {
    "ema_20": 176.50,
    "ema_50": 174.20,
    "ema_200": 165.80,
    "trend_status": "Strong Uptrend",
    "gc_date": "2024-10-15",
    "gc_price": 168.30,
    "chart_image": "base64_encoded_image"
  },
  "donchian": {
    "dc_upper": 182.50,
    "dc_lower": 168.20,
    "dc_middle": 175.35,
    "last_price": 175.43,
    "status": "Bullish Breakout",
    "breakout_signal": "Upper Channel Breakout",
    "signals": [...],
    "chart_image": "base64_encoded_image"
  },
  "rsi": {
    "current_rsi": 62.5,
    "divergences": [...],
    "chart_image": "base64_encoded_image"
  },
  "rsi_volume": {
    "current_rsi": 62.5,
    "current_volume": 45000000,
    "volume_ma_20": 42000000,
    "volume_ma_50": 40000000,
    "bullish_divergences": [...],
    "bearish_divergences": [...],
    "early_reversals": [...],
    "chart_image": "base64_encoded_image"
  }
}
```

## ⚙️ Configuration

### Common Configuration Options

All indicators support these parameters:
- **INTERVAL**: Data granularity (`1d`, `1wk`, `1mo`, `1h`, `15m`)
- **LOOKBACK_PERIODS**: Number of days/periods to fetch (default: 730)

### Indicator-Specific Parameters

Each indicator has specific tunable parameters accessible via the configuration panel in the UI or programmatically via the API.

## 🛠️ Troubleshooting

### Port Already in Use
If port 5000 is occupied:
```python
# Modify app.py line 357
app.run(debug=True, host='127.0.0.1', port=5001)  # Change port
```

### Module Not Found Errors
Ensure parent directory is in Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:/Users/solankianshul/Documents/projects/stock_research"
```

Or add at the top of `app.py`:
```python
import sys
sys.path.insert(0, '/Users/solankianshul/Documents/projects/stock_research')
```

### Data Fetching Errors
- Verify internet connection
- Check ticker symbol validity
- Confirm ticker exists in Yahoo Finance
- For Indian stocks, ensure correct suffix (`.NS` or `.BO`)
- **Note**: 15-minute interval data is limited to 59 days of history due to Yahoo Finance restrictions

### Chart Not Displaying
- Check browser console for errors
- Verify `matplotlib.use('Agg')` is set in `app.py`
- Ensure base64 image encoding is working

## 🎯 Usage Examples

### Basic Analysis
1. Enter ticker: `AAPL`
2. Click "Analyze" button
3. View results in all tabs

### Custom MACD Analysis
1. Go to "Lagging Indicator Analysis" → "MACD Analysis"
2. Click "⚙️ MACD Configuration"
3. Adjust parameters (e.g., Fast=8, Slow=21, Signal=5)
4. Click "Run MACD Analysis"

### Compare Multiple Timeframes
1. Run analysis with `INTERVAL = 1d`
2. Note the Golden Cross date in Crossover tab
3. Re-run with `INTERVAL = 1wk` to confirm trend
4. Use `INTERVAL = 15m` for intraday entries

## 📊 Best Practices

1. **Start with Lagging Indicators**: Use MACD, Supertrend, and Crossover to identify the primary trend
2. **Confirm with Leading Indicators**: Check RSI divergences for reversal signals
3. **Use Multiple Timeframes**: Daily for swing trades, 15m/1h for day trades
4. **Check Volume**: RSI-Volume divergence can provide early warning signals
5. **Combine Signals**: Look for confluence across multiple indicators

## 🚧 Future Enhancements

- Add more leading indicators (Stochastic, Williams %R)
- Implement backtesting capabilities
- Add comparison mode for multiple tickers
- Export analysis reports as PDF
- Save/load custom configurations
- Add alerts and notifications

## 📄 License

For personal use only.

## ⚠️ Disclaimer

This tool is for educational and research purposes only. Technical analysis does not guarantee future results. Always conduct your own due diligence and consult with a financial advisor before making investment decisions.

---

**Last Updated**: December 2024  
**Version**: 2.0
