@echo off
ECHO #####################################################
ECHO #          SINGULARITY - IEL DEBUG SCRIPT           #
ECHO #####################################################
ECHO.

echo --- Activating Virtual Environment ---
CALL .\venv\Scripts\activate.bat
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Could not activate the virtual environment.
    pause
    EXIT /B 1
)
ECHO.

echo --- Running IEL Trainer Directly ---
python train_rl_executor.py

ECHO.
echo --- SCRIPT FINISHED ---
PAUSE
