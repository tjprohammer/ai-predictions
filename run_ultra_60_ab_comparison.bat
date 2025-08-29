@echo off
echo.
echo =========================================================
echo 🎯 ULTRA-60 A/B COMPARISON TEST
echo =========================================================
echo Comparing 60-day optimal vs other configurations
echo This will run side-by-side analysis for validation
echo =========================================================
echo.

REM Primary: 60-day optimal configuration
set INCREMENTAL_LEARNING_DAYS=60
set ALWAYS_RUN_DUAL=true
set PUBLISH_BLEND=false
set PREDICT_ALL_TODAY=true
set ULTRA_CONFIDENCE_THRESHOLD=3.0

REM Enable detailed comparison logging
set LOG_AB_COMPARISON=true
set COMPARE_WITH_14D=true
set COMPARE_WITH_90D=true

REM Track all confidence thresholds for analysis
set CONFIDENCE_THRESHOLDS=0.5,1.0,1.5,2.0,2.5,3.0

echo Running Ultra-60 with A/B comparison analysis...
echo Primary: 60-day window (88.6%% accuracy target)
echo Comparison: 14-day (current) and 90-day (best MAE)
echo.

python mlb\core\daily_api_workflow.py

echo.
echo =========================================================
echo A/B Comparison completed!
echo Review comparison logs for performance validation
echo =========================================================
pause
