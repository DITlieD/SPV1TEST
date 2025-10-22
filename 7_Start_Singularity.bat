@echo off
SETLOCAL
TITLE Singularity Protocol - Cortex

REM --- Configuration ---
SET VENV_PATH=venv\Scripts\activate.bat
SET PYTHON_APP=app.py
REM ---------------------

ECHO ##################################
ECHO #     The Singularity Protocol   #
ECHO #         Cortex Startup         #
ECHO ##################################
ECHO.

REM --- 1. Activate Python Virtual Environment ---
ECHO [1/2] Activating Python Environment...
IF EXIST "%VENV_PATH%" (
    CALL "%VENV_PATH%"
    ECHO [1/2] Environment activated.
) ELSE (
    ECHO [ERROR] Virtual environment not found at %VENV_PATH%. Exiting.
    PAUSE
    EXIT /B 1
)
ECHO.

REM --- 2. Launch the Python Cortex ---
ECHO [2/2] Launching Cortex (Python Core)....
ECHO --------------------------------------------------------------------
ECHO System Startup Initiated.
ECHO Monitor this window for Cortex logs (Python).
ECHO Press CTRL+C in THIS WINDOW to stop the Python application.
ECHO --------------------------------------------------------------------
ECHO.

python -X faulthandler %PYTHON_APP%

ECHO.
ECHO [System] Cortex (Python Core) has stopped.
PAUSE
ENDLOCAL
