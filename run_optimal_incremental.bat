@echo off
REM Optimal Incremental Learning Workflow - 14 Day Window
REM Based on A/B testing results from 1,517 games (April-August 2025)
REM 14-day window won 3/5 performance metrics with better MAE and correlation

set INCREMENTAL_LEARNING_DAYS=14

echo Running Ultra-80 with optimal 14-day incremental learning window...
echo Based on A/B test results: 14d MAE=3.665 vs 7d MAE=3.716

python mlb\core\daily_api_workflow.py

echo.
echo Incremental learning complete. Window: %INCREMENTAL_LEARNING_DAYS% days
pause
