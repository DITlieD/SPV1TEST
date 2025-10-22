@echo off
SETLOCAL

SET RUST_PROJECT_DIR=cerebellum_core
SET CEREBELLUM_EXE=%RUST_PROJECT_DIR%\target\release\cerebellum_core.exe

ECHO [2/4] Launching Cerebellum (Rust Core) in a separate window...

IF NOT EXIST "%CEREBELLUM_EXE%" (
    ECHO [ERROR] Cerebellum executable not found at %CEREBELLUM_EXE%.
    PAUSE
    EXIT /B 1
)

START "Cerebellum Execution Core (Rust)" /D "%RUST_PROJECT_DIR%" %CEREBELLUM_EXE%

ECHO [2/4] Cerebellum launched.

CLOSE
ENDLOCAL
