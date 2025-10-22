@echo off
echo --- Starting Implicit Brain (SDE) Training ---

REM Check if the virtual environment's Python executable exists
if exist "venv\Scripts\python.exe" (
    echo --- Using Python from virtual environment ---
    venv\Scripts\python.exe train_sde_brain.py
) else (
    echo --- WARNING: Virtual environment not found. Using default 'python'. ---
    echo --- This may fail if dependencies are not installed globally. ---
    python train_sde_brain.py
)

echo --- SDE Training Script Finished ---
pause
