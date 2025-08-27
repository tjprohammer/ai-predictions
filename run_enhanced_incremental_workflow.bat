@echo off
REM Enhanced Daily Workflow with Incremental Learning
REM ================================================
REM 
REM This batch script runs the daily workflow which now includes:
REM 1. Incremental learning updates from recent games (automatic)
REM 2. Predictions using the updated incremental system (primary)
REM 3. Fallback to Ultra Sharp or Enhanced Bullpen if needed
REM
REM Usage: run_enhanced_incremental_workflow.bat [YYYY-MM-DD]

echo.
echo ================================================================
echo 🚀 ENHANCED DAILY WORKFLOW with INCREMENTAL LEARNING
echo ================================================================
echo.

REM Set target date (default to today if not provided)
set TARGET_DATE=%1
if "%TARGET_DATE%"=="" (
    REM Get today's date using PowerShell
    for /f %%i in ('powershell -Command "(Get-Date).ToString('yyyy-MM-dd')"') do set TARGET_DATE=%%i
)

echo 🎯 Target Date: %TARGET_DATE%
echo.

REM Run the enhanced daily workflow (incremental learning now integrated)
echo ================================================================
echo 🧠 Running Daily Workflow with Integrated Incremental Learning
echo ================================================================
echo.

REM Check if daily workflow exists
if exist "mlb-overs\deployment\daily_api_workflow.py" (
    echo Running enhanced daily workflow for %TARGET_DATE%...
    echo The workflow will automatically:
    echo   1. Update incremental models from recent completed games
    echo   2. Generate predictions using the updated incremental system
    echo   3. Export results and update the database
    echo.
    
    cd mlb-overs\deployment
    python daily_api_workflow.py --target-date %TARGET_DATE% --stages markets,features,predict,odds,health,prob,export,audit
    cd ..\..
    
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ Error in enhanced daily workflow
    ) else (
        echo ✅ Enhanced daily workflow completed successfully
    )
) else (
    echo ❌ daily_api_workflow.py not found in mlb-overs\deployment\
    echo Please check the file path and try again.
)

echo.

REM Show results summary
echo ================================================================
echo 📋 Results Summary
echo ================================================================
echo.

echo 📁 Generated Files:
if exist "mlb-overs\deployment\exports\preds_%TARGET_DATE%.csv" (
    echo    ✅ mlb-overs\deployment\exports\preds_%TARGET_DATE%.csv (daily workflow predictions)
) else (
    echo    ❌ No daily workflow predictions file found
)

if exist "outputs\slate_%TARGET_DATE%_predictions*.csv" (
    echo    ✅ outputs\slate_%TARGET_DATE%_predictions*.csv (incremental system output)
) else (
    echo    ❌ No incremental predictions file found
)

echo.
echo 🏁 Enhanced daily workflow complete for %TARGET_DATE%!
echo.
echo 🧠 The incremental learning system is now the PRIMARY predictor in the daily workflow.
echo 📊 It automatically learns from recent completed games and generates today's predictions.
echo 🎯 Check mlb-overs\deployment\exports\ for the main prediction outputs.
echo.

pause
