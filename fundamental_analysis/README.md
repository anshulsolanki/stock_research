# Fundamental Analysis Module

A comprehensive tool for analyzing stock financial health, growth trends, and valuation metrics. This module supports both single-ticker analysis and high-performance parallel batch processing.

## 🚀 Features

### 1. Long-term Analysis (4 Years)
- **Revenue Growth**: Trend analysis & 3-year CAGR.
- **Profit Growth**: Trend analysis & 3-year CAGR.
- **ROE (Return on Equity)**: Consistency and growth.
- **EPS (Earnings Per Share)**: Growth trends.
- **Valuation**: PE Ratio comparison vs Industry averages.

### 2. Short-term Analysis (6 Quarters)
- **Quarterly Growth**: Revenue, Profit, ROE, and EPS analysis.
- **Trend Detection**: QoQ growth confirmation.

### 3. Batched & Parallel Processing
- **Multi-threading**: Process multiple tickers simultaneously (default: 3 workers) for rapid analysis.
- **Robustness**: Built-in retry logic with exponential backoff to handle API rate limits.
- **Progress Tracking**: Real-time progress bar (via `tqdm`).
- **Error Logging**: Detailed failure logs saved to `analysis_errors.json`.

### 4. Automated Reporting
- Generates a consolidated **PDF Report** containing:
  - Ranked summary of all analyzed stocks.
  - Detailed individual stock pages with visual tables.
  - Color-coded metrics (Growth/Decline).

## 💻 Usage

### Prerequisites
```bash
pip install pandas numpy yfinance matplotlib tqdm
```

### 1. Single Ticker Analysis
Run the script directly or import it in Python.

**CLI:**
```bash
# defaults to DABUR.NS if no argument provided
python fundamental_analysis.py
```

**Python:**
```python
from fundamental_analysis import run_analysis

result = run_analysis("RELIANCE.NS")
print(result)
```

### 2. Batch Analysis
Run analysis on a list of tickers from a JSON file.

**Input Format (`tickers.json`):**
```json
// Simple List
["RELIANCE.NS", "TCS.NS", "INFY.NS"]

// OR Dictionary (values are used)
{
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS"
}
```

**CLI Command:**
```bash
python fundamental_analysis.py path/to/tickers.json
```

**Output:**
- **Console**: Summary table of results.
- **PDF**: `fundamental_batch_report_YYYYMMDD_HHMMSS.pdf`
- **Errors**: `analysis_errors.json` (if any failures occur)

## ⚙️ Configuration

The script attempts to handle API rate limits automatically.
- **Concurrency**: Defaults to 3 workers to be safe.
- **Retries**: 3 retries with exponential backoff for network requests.

To modify these settings, edit `fundamental_analysis.py`:
```python
# In analyze_multiple_tickers function
with ThreadPoolExecutor(max_workers=3) as executor: # Change max_workers
```
