@echo off
SETLOCAL

SET RUST_PROJECT_DIR=cerebellum_core

ECHO Compiling Cerebellum (Rust Execution Core) in Release Mode...

where cargo >nul 2>nul
IF ERRORLEVEL 1 (
    ECHO [ERROR] 'cargo' command not found. Please install the Rust toolchain.
    PAUSE
    EXIT /B 1
)

PUSHD %RUST_PROJECT_DIR%
call cargo build --release

IF ERRORLEVEL 1 (
    ECHO.
    ECHO [ERROR] FAILED to compile the Rust Cerebellum Core.
    POPD
    PAUSE
    EXIT /B 1
)
POPD
ECHO Compilation successful.

PAUSE
ENDLOCAL