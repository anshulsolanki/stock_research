@echo off
REM -------------------------------------------------------------------------------
REM Project: Stock Analysis
REM Script to run all screeners sequentially on Windows
REM -------------------------------------------------------------------------------

SET PYTHON_EXEC=C:\Users\solan\projects\stock_research\venv\Scripts\python
SET BASE_DIR=C:\Users\solan\projects\stock_research\breakout_screeners

echo ============================================================
echo Starting Stock Research Screeners Batch Run
echo ============================================================
echo %DATE% %TIME%

REM 1. Data refresh
echo.
echo [1/9] Running Data Refresh...
"%PYTHON_EXEC%" "%BASE_DIR%\download_data.py"

REM 2. Bear market screeners
echo.
echo [2/9] Running Bear Market Screeners...
"%PYTHON_EXEC%" "%BASE_DIR%\bear_market_screener\bear_market_combined_screener.py"

REM 3. CANSLIM screener
echo.
echo [3/9] Running CANSLIM Screener...
"%PYTHON_EXEC%" "%BASE_DIR%\canslim_screener.py"

REM 4. Financial wisdom screener
echo.
echo [4/9] Running Financial Wisdom Screener...
"%PYTHON_EXEC%" "%BASE_DIR%\fw_breakout_screener.py"

REM 5. Minervini screener (Default)
echo.
echo [5/9] Running Minervini Screener (Default)...
"%PYTHON_EXEC%" "%BASE_DIR%\minervini_screener.py" --use-fundamentals

REM 6. Minervini screener (With Volume Dry-Up)
echo.
echo [6/9] Running Minervini Screener with Volume Dry-Up...
"%PYTHON_EXEC%" "%BASE_DIR%\minervini_screener.py" --use-volume-dryup

REM 7. Qullamaggie screener
echo.
echo [7/9] Running Qullamaggie Screener...
"%PYTHON_EXEC%" "%BASE_DIR%\qullamaggie_screener.py"

REM 8. Darvas screener
echo.
echo [8/9] Running Darvas Screener...
"%PYTHON_EXEC%" "%BASE_DIR%\darvas_screener.py"

REM 9. Screener Results Tracker
echo.
echo [9/9] Running Screener Results Tracker...
"%PYTHON_EXEC%" "%BASE_DIR%\screener_results_tracker.py"

echo.
echo ============================================================
echo Batch Run Completed
echo ============================================================
echo %DATE% %TIME%

pause