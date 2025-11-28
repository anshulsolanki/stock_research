# MACD Stock Analysis Web App

A Flask-based web application for analyzing stocks using the MACD (Moving Average Convergence Divergence) technical indicator.

## Features

- **Interactive UI**: Enter any stock ticker and get instant MACD analysis
- **Visual Charts**: View MACD, Signal Line, and Histogram plots
- **Key Metrics**: See trend, momentum, and crossover signals
- **Divergence Detection**: Automatically identifies bullish and bearish divergences
- **Responsive Design**: Works on desktop and mobile devices

## Prerequisites

- Python 3.8+
- Flask
- matplotlib
- pandas
- numpy
- yfinance
- scipy

## Installation

1. Navigate to the web_app directory:
```bash
cd /Users/solankianshul/Documents/stock_research/web_app
```

2. Install Flask if not already installed:
```bash
pip install flask
```

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

3. Enter a stock ticker (e.g., `AAPL`, `MARICO.NS`) and click "Analyze"

## Ticker Format

- **US Stocks**: Use the standard ticker (e.g., `AAPL`, `MSFT`, `TSLA`)
- **Indian Stocks (NSE)**: Add `.NS` suffix (e.g., `RELIANCE.NS`, `TCS.NS`)
- **Indian Stocks (BSE)**: Add `.BO` suffix (e.g., `RELIANCE.BO`)

## How It Works

1. **Backend** (`app.py`): Flask server that calls `macd_analysis.py` to analyze stocks
2. **Frontend** (`templates/index.html`): User interface for entering tickers
3. **Styling** (`static/style.css`): Modern, gradient-based design
4. **Logic** (`static/script.js`): Handles API calls and result display

## API Endpoint

**POST** `/analyze`

Request body:
```json
{
    "ticker": "AAPL"
}
```

Response:
```json
{
    "success": true,
    "ticker": "AAPL",
    "macd_line": 5.23,
    "signal_line": 4.89,
    "histogram": 0.34,
    "trend": "Bullish",
    "momentum": "Strengthening",
    "crossover_signal": "Bullish Crossover (Buy)",
    "divergences": [...],
    "chart_image": "base64_encoded_image"
}
```

## Troubleshooting

### Port Already in Use
If port 5000 is already in use, modify `app.py`:
```python
app.run(debug=True, host='127.0.0.1', port=5001)  # Change port
```

### Module Not Found
Ensure the parent directory is accessible:
```bash
export PYTHONPATH="${PYTHONPATH}:/Users/solankianshul/Documents/stock_research"
```

### Data Fetching Errors
- Check internet connection
- Verify ticker symbol is correct
- Some tickers may not be available in Yahoo Finance

## Future Enhancements

- Add more technical indicators (RSI, Bollinger Bands, Supertrend)
- Support multiple timeframes (daily, weekly, intraday)
- Save favorite tickers
- Export analysis as PDF

## License

For personal use only.
