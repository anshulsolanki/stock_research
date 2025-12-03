# Stock Research Platform - Modern UI

This is the redesigned modern web interface for the Stock Research Platform.

## Features
- **Modern Dashboard Layout**: Clean, responsive design with a sidebar for easy navigation.
- **Stock Watchlist**: Quickly switch between stocks from the sidebar.
- **Comprehensive Analysis**:
  - **Leading Indicators**: RSI Divergence, RSI-Volume, Volatility Squeeze, RS Analysis.
  - **Lagging Indicators**: MACD, Supertrend, Bollinger Bands, Crossover, Donchian Channels.
- **Future Ready**: Placeholders for Custom Strategy and Batch Analysis.

## How to Run

1. Navigate to the project root directory.
2. Run the Flask app:
   ```bash
   python3 website_ui/app.py
   ```
3. Open your browser and go to `http://127.0.0.1:5001`.

## Structure
- `app.py`: The Flask backend (runs on port 5001).
- `templates/`: HTML templates.
  - `base.html`: Main layout with sidebar.
  - `dashboard.html`: Main analysis view.
  - `partials/`: Reusable analysis panels.
- `static/`: CSS and JavaScript files.
