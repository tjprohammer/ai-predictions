@echo off
REM Nightly incremental update - train on yesterday's completed games (Updated for MLB structure)
REM Schedule this for 3AM local time after games are complete

echo 🌙 NIGHTLY ULTRA 80 UPDATE...

REM Get yesterday's date (PowerShell way for Windows)
for /f %%i in ('powershell -Command "(Get-Date).AddDays(-1).ToString('yyyy-MM-dd')"') do set START_DATE=%%i
for /f %%i in ('powershell -Command "(Get-Date).ToString('yyyy-MM-dd')"') do set END_DATE=%%i

set RUN_MODE=TRAIN_ONLY

echo Updating models with games from: %START_DATE%
echo.

python mlb\systems\incremental_ultra_80_system.py

if %ERRORLEVEL% equ 0 (
    echo.
    echo 📊 Running tracking validation...
    python mlb\tracking\validation\check_predictions_final.py
    
    echo.
    echo ✅ NIGHTLY UPDATE COMPLETE!
    echo 📈 Models updated with latest game results
    echo 💾 State saved to: mlb\models\incremental_ultra80_state.joblib
    echo 🔍 Tracking validation completed
    echo.
    echo Ready for pregame_slate.bat
) else (
    echo ❌ Nightly update failed with error %ERRORLEVEL%
)

REM Don't pause for automated runs
REM pause
