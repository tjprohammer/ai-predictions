@echo off
REM Bootstrap the Ultra 80 system with 60-90 days of historical data (Updated for MLB structure)
REM Run this ONCE to initialize the system

echo 🚀 BOOTSTRAPPING ULTRA 80 SYSTEM...
echo Warming up models + scaler + conformal buffers with historical data

set START_DATE=2025-06-01
set END_DATE=2025-08-01
set RUN_MODE=TRAIN_ONLY

echo Training window: %START_DATE% to %END_DATE%
echo.

python mlb\systems\incremental_ultra_80_system.py

if %ERRORLEVEL% equ 0 (
    echo.
    echo ✅ BOOTSTRAP COMPLETE!
    echo 💾 State saved to: mlb\models\incremental_ultra80_state.joblib
    echo 📊 Backtest results in: outputs\
    echo.
    echo Next steps:
    echo   1. Run nightly_update.bat after each day's games
    echo   2. Run pregame_slate.bat to see today's predictions
) else (
    echo ❌ Bootstrap failed with error %ERRORLEVEL%
)

pause
