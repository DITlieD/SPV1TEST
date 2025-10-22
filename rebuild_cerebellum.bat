@echo off
echo ========================================
echo Cerebellum Rebuild Script
echo ========================================
echo.
echo Stopping any running Cerebellum process...
taskkill /F /IM cerebellum_core.exe 2>nul
timeout /t 2 /nobreak >nul
echo.
echo Rebuilding Cerebellum...
cd cerebellum_core
cargo build --release
echo.
echo ========================================
echo Cerebellum rebuild complete!
echo ========================================
pause
