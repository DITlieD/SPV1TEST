@echo off
echo --- Starting Influence Mapper GAT Model Training ---
echo --- This script will train the Graph Attention Network. ---
echo.

REM Check if the virtual environment's Python executable exists
if exist "venv\Scripts\python.exe" (
    echo --- Using Python from virtual environment ---
    venv\Scripts\python.exe train_gat_model.py
) else (
    echo --- ERROR: Virtual environment not found. ---
    echo --- Please run this script from the project root directory. ---
)

echo --- GAT Model Training Finished ---
pause
