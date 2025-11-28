# Stock Research & Analysis

A comprehensive Python-based stock technical analysis toolkit with multiple lagging indicators, relative strength analysis, and an integrated web application for visualization.

## 📊 Features

### Lagging Indicators
- **MACD Analysis**: Moving Average Convergence Divergence with divergence detection
- **Supertrend Analysis**: ATR-based trend following with dynamic support/resistance
- **Bollinger Bands**: Volatility bands with RSI and candlestick pattern confirmation
- **EMA Crossover**: Multi-timeframe analysis with Golden Cross detection
- **Donchian Channels**: Breakout detection system (Turtle Trading)

### Relative Strength Analysis
- Multi-timeframe RS calculation (1M, 2M, 3M, 6M, 1Y, 3Y, 5Y)
- Momentum pattern detection (Consistent, Emerging, Slowing)
- Early turnaround signal identification
- Technical confirmation filters (MA Breakout, Volume Surge)
- Automated scoring and ranking system

### Web Application
- Flask-based web interface at `http://127.0.0.1:5000`
- Interactive analysis with configurable parameters
- Multi-tab interface for different indicators
- Real-time chart generation
- Support for multiple timeframes and intervals

## 🚀 Quick Start

### Prerequisites
```bash
python 3.8+
pip install -r requirements.txt  # (create if needed)
```

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/stock_research.git
cd stock_research
pip install pandas numpy yfinance matplotlib scipy flask
```

### Running Analysis

**Standalone Analysis:**
```bash
# MACD Analysis
python lagging_indicator_analysis/macd_analysis.py

# Supertrend Analysis
python lagging_indicator_analysis/supertrend_analysis.py

# RS Analysis
python rs_analysis.py
```

**Web Application:**
```bash
cd web_app
python app.py
# Navigate to http://127.0.0.1:5000
```

## 📁 Project Structure

```
stock_research/
├── lagging_indicator_analysis/
│   ├── macd_analysis.py           # MACD indicator
│   ├── supertrend_analysis.py     # Supertrend indicator
│   ├── bollinger_band_analysis.py # Bollinger Bands
│   ├── crossover_analysis.py      # EMA Crossover
│   └── donchian_channel_analysis.py # Donchian Channels
├── leading_indicator_analysis/     # Leading indicators (future)
├── web_app/
│   ├── app.py                     # Flask backend
│   ├── templates/
│   │   └── index.html             # Web UI
│   └── static/
│       ├── style.css
│       └── script.js
├── data/
│   ├── tickers_list.json          # Stock tickers
│   └── tickers_batch.json         # Batch configurations
├── docs/
│   └── todo.txt                   # Project roadmap
├── rs_analysis.py                 # Relative Strength analysis
└── stock_analysis.py              # Legacy analysis script
```

## 🔧 Configuration

Each analysis module has customizable parameters via config dictionaries:

```python
# Example: MACD Configuration
config = {
    'FAST': 12,
    'SLOW': 26,
    'SIGNAL': 9,
    'INTERVAL': '1d',
    'LOOKBACK_PERIODS': 730
}
results = run_analysis(ticker="AAPL", config=config)
```

## 📈 Use Cases

1. **Trend Identification**: Use MACD, Supertrend, and EMA Crossover
2. **Breakout Trading**: Donchian Channels for entry signals
3. **Mean Reversion**: Bollinger Bands in ranging markets
4. **Relative Strength**: Compare stocks within sectors (RS Analysis)
5. **Multi-Indicator Confluence**: Combine signals for high-probability setups

## 🛠️ Technical Details

- **Data Source**: Yahoo Finance (via `yfinance`)
- **Visualization**: Matplotlib
- **Web Framework**: Flask
- **Data Processing**: Pandas, NumPy
- **Signal Detection**: SciPy

## 📝 License

[Add your license here]

## 🤝 Contributing

Contributions welcome! Please read the contribution guidelines first.

## 📧 Contact

[Add your contact information]

---

**Note**: This tool is for educational and research purposes. Always do your own analysis before making investment decisions.
