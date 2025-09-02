@echo off
REM UPDATED: Optimal Incremental Learning - 60 Day Window  
REM System in development: Current performance ~50% accuracy
REM 60-day window testing configuration

set INCREMENTAL_LEARNING_DAYS=60

echo =========================================================
echo 🚀 ULTRA-60 DEVELOPMENT CONFIGURATION 
echo =========================================================
echo System currently in development:
echo - ~50%% accuracy (below target, improving)
echo - Testing various configurations
echo - 60-day learning window active
echo - Performance tracking enabled
echo =========================================================

python mlb\core\daily_api_workflow.py

echo.
echo Ultra-60 incremental learning complete. Window: %INCREMENTAL_LEARNING_DAYS% days
echo Current performance: ~50%% accuracy (developing system)
pause
