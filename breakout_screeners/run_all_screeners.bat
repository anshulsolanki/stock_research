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
echo [1/11] Running Data Refresh...
"%PYTHON_EXEC%" "%BASE_DIR%\download_data.py"

REM 2. Bear market screeners
echo.
echo [2/11] Running Bear Market Screeners...
"%PYTHON_EXEC%" "%BASE_DIR%\bear_market_screener\bear_market_combined_screener.py"

REM 3. CANSLIM screener
echo.
echo [3/11] Running CANSLIM Screener...
"%PYTHON_EXEC%" "%BASE_DIR%\canslim_screener.py"

REM 4. Financial wisdom screener
echo.
echo [4/11] Running Financial Wisdom Screener...
"%PYTHON_EXEC%" "%BASE_DIR%\fw_breakout_screener.py"

REM 5. Minervini ST screener (Default)
echo.
echo [5/11] Running Minervini ST Screener (Default)...
"%PYTHON_EXEC%" "%BASE_DIR%\minervini_st_breakouts.py" --use-fundamentals

REM 6. Minervini ST screener (With Volume Dry-Up)
echo.
echo [6/11] Running Minervini ST Screener with Volume Dry-Up...
"%PYTHON_EXEC%" "%BASE_DIR%\minervini_st_breakouts.py" --use-volume-dryup

REM 7. Minervini LT screener
echo.
echo [7/11] Running Minervini LT Breakout Screener...
"%PYTHON_EXEC%" "%BASE_DIR%\minervini_lt_breakouts.py"

REM 8. Qullamaggie screener
echo.
echo [8/11] Running Qullamaggie Screener...
"%PYTHON_EXEC%" "%BASE_DIR%\qullamaggie_screener.py"

REM 9. Darvas screener
echo.
echo [9/11] Running Darvas Screener...
"%PYTHON_EXEC%" "%BASE_DIR%\darvas_screener.py"

REM 10. Screener Results Tracker
echo.
echo [10/11] Running Screener Results Tracker...
"%PYTHON_EXEC%" "%BASE_DIR%\screener_results_tracker.py"

echo.
echo ============================================================
echo Batch Run Completed
echo ============================================================
echo %DATE% %TIME%

pause