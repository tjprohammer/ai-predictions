@echo off
REM Pregame slate - fast display of today's predictions
REM Run this anytime to see current predictions without training

echo 🎯 PREGAME SLATE PREDICTIONS...

REM Get today's date
for /f %%i in ('powershell -Command "(Get-Date).ToString('yyyy-MM-dd')"') do set SLATE_DATE=%%i

set RUN_MODE=SLATE_ONLY

echo Generating predictions for: %SLATE_DATE%
echo.

python mlb-overs\pipelines\incremental_ultra_80_system.py

if %ERRORLEVEL% equ 0 (
    echo.
    echo ✅ SLATE PREDICTIONS COMPLETE!
    echo 📊 CSV saved to: outputs\slate_%SLATE_DATE%_predictions_*.csv
    echo 📄 One-pager saved to: outputs\onepager_%SLATE_DATE%.md
    echo.
    echo Check the console output above for today's recommended bets!
) else (
    echo ❌ Slate generation failed with error %ERRORLEVEL%
    echo Make sure you've run bootstrap_ultra80.bat first
)

pause
