@echo off
echo --- Starting L2 Historical Data Collector ---
echo --- This script will continuously save L2 order book data. ---
echo --- Leave this window open for at least 4-6 hours (24 hours is recommended). ---
echo --- Press CTRL+C in this window to stop the collection process. ---
echo.

REM Check if the virtual environment's Python executable exists
if exist "venv\Scripts\python.exe" (
    echo --- Using Python from virtual environment ---
    venv\Scripts\python.exe forge/data_processing/l2_collector.py
) else (
    echo --- ERROR: Virtual environment not found. ---
    echo --- Please run this script from the project root directory. ---
)

echo --- L2 Data Collection Stopped ---
pause
