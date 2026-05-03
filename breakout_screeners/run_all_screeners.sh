#!/bin/bash

# -------------------------------------------------------------------------------
# Project: Stock Analysis
# Script to run all screeners sequentially
# -------------------------------------------------------------------------------

PYTHON_EXEC="/Users/solankianshul/Documents/projects/stock_research/.venv/bin/python"
BASE_DIR="/Users/solankianshul/Documents/projects/stock_research/breakout_screeners"

echo "============================================================"
echo "Starting Stock Research Screeners Batch Run"
echo "============================================================"
date

# 1. Data refresh
echo -e "\n[1/11] Running Data Refresh..."
$PYTHON_EXEC "$BASE_DIR/download_data.py"

# 2. Bear market screeners
echo -e "\n[2/11] Running Bear Market Screeners..."
$PYTHON_EXEC "$BASE_DIR/bear_market_screener/bear_market_combined_screener.py"

# 3. CANSLIM screener
echo -e "\n[3/11] Running CANSLIM Screener..."
$PYTHON_EXEC "$BASE_DIR/canslim_screener.py"

# 4. Financial wisdom screener
echo -e "\n[4/11] Running Financial Wisdom Screener..."
$PYTHON_EXEC "$BASE_DIR/fw_breakout_screener.py"

# 5. Minervini ST screener (Default)
echo -e "\n[5/11] Running Minervini ST Screener (Default)..."
$PYTHON_EXEC "$BASE_DIR/minervini_st_breakouts.py" --use-fundamentals

# 6. Minervini ST screener (With Volume Dry-Up)
echo -e "\n[6/11] Running Minervini ST Screener with Volume Dry-Up..."
$PYTHON_EXEC "$BASE_DIR/minervini_st_breakouts.py" --use-volume-dryup

# 7. Minervini LT screener
echo -e "\n[7/11] Running Minervini LT Breakout Screener..."
$PYTHON_EXEC "$BASE_DIR/minervini_lt_breakouts.py"

# 8. Qullamaggie screener
echo -e "\n[8/11] Running Qullamaggie Screener..."
$PYTHON_EXEC "$BASE_DIR/qullamaggie_screener.py"

# 9. Darvas screener
echo -e "\n[9/11] Running Darvas Screener..."
$PYTHON_EXEC "$BASE_DIR/darvas_screener.py"

# 10. Screener Results Tracker
echo -e "\n[10/11] Running Screener Results Tracker..."
$PYTHON_EXEC "$BASE_DIR/screener_results_tracker.py"

echo -e "\n============================================================"
echo "Batch Run Completed"
echo "============================================================"
date
