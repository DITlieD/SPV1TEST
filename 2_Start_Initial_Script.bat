@echo off
SETLOCAL EnableDelayedExpansion
REM V3.1: Unified initial training script.

ECHO.
ECHO #####################################################
ECHO #           SINGULARITY - GENESIS PROTOCOL          #
ECHO #          INITIAL MODEL TRAINING SCRIPT            #
ECHO #####################################################
ECHO.

echo --- Activating Virtual Environment ---
CALL .\venv\Scripts\activate.bat
IF !ERRORLEVEL! NEQ 0 (
    echo ERROR: Could not activate the virtual environment.
    pause
    EXIT /B 1
)
ECHO.

echo [1/5] Preparing all historical data...
python prepare_all_data.py
IF !ERRORLEVEL! NEQ 0 (
    echo ERROR during data preparation. Halting script.
    pause
    EXIT /B !ERRORLEVEL!
)
ECHO.

echo [2/5] Training Auxiliary Models (HMM, RL Governor)...
python train_aux_models.py
IF !ERRORLEVEL! NEQ 0 (
    echo ERROR during Auxiliary Model training. Halting script.
    pause
    EXIT /B !ERRORLEVEL!
)
ECHO.

echo [3/5] Training Initial Specialist Models (Genesis)...
python chimera_trainer.py
IF !ERRORLEVEL! NEQ 0 (
    echo ERROR during Initial Specialist Model training. Halting script.
    pause
    EXIT /B !ERRORLEVEL!
)
ECHO.

echo [4/5] Training IEL Agent...
IF EXIST IEL_agent.zip (
    echo IEL agent (IEL_agent.zip^) already exists. Skipping training.
) ELSE (
    CALL Train_IEL_Agent.bat
    IF !ERRORLEVEL! NEQ 0 (
        echo ERROR during IEL Agent training step. Halting script.
        pause
        EXIT /B !ERRORLEVEL!
    ))
ECHO.

echo [5/5] Running Online Model Trainer...
python train_online_model.py
IF !ERRORLEVEL! NEQ 0 (
    echo ERROR during Online Model training. Halting script.
    pause
    EXIT /B !ERRORLEVEL!
)
ECHO.

echo --- ALL MODELS TRAINED SUCCESSFULLY ---
PAUSE
