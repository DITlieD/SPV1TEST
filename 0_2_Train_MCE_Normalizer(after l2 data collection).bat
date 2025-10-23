@echo off
echo --- Starting MCE Feature Normalizer Training ---

REM Check if the virtual environment's Python executable exists
if exist "venv\Scripts\python.exe" (
    echo --- Using Python from virtual environment ---
    venv\Scripts\python.exe train_mce_normalizer.py
) else (
    echo --- WARNING: Virtual environment not found. Using default 'python'. ---
    echo --- This may fail if dependencies are not installed globally. ---
    python train_mce_normalizer.py
)

echo --- MCE Normalizer Training Script Finished ---
pause
