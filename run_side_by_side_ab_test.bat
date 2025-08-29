@echo off
REM Side-by-side A/B Testing: Learning vs Original Model
REM This runs both the optimized 14-day incremental learning AND the original baseline model
REM Provides complete side-by-side comparison for analysis

echo ========================================
echo    SIDE-BY-SIDE A/B TESTING MODE
echo ========================================
echo.
echo Configuration:
echo - Learning Window: 14 days (A/B tested optimal)
echo - Recency Features: 7,14,30 day windows  
echo - Original Model: Always runs for comparison
echo - Published Prediction: Learning model (with fallback)
echo.

REM Set optimal configuration based on A/B test results
set INCREMENTAL_LEARNING_DAYS=14

REM Enable recency/matchup features 
set RECENCY_WINDOWS=7,14,30

REM Predict all games today (not just upcoming)
set PREDICT_ALL_TODAY=1

REM Always run dual models for side-by-side comparison
set ALWAYS_RUN_DUAL=1

REM Publish learning model as primary (with safe fallback to original)
set PUBLISH_BLEND=0

echo Running daily workflow with side-by-side comparison...
python mlb\core\daily_api_workflow.py --stages markets,features,predict,odds,health,prob,export

echo.
echo ========================================
echo     A/B TEST ANALYSIS AVAILABLE
echo ========================================
echo.
echo Check database columns:
echo - predicted_total: Published prediction (learning model)
echo - predicted_total_learning: Incremental Ultra-80 system
echo - predicted_total_original: Original baseline model
echo.
echo For detailed analysis:
echo python mlb\analysis\ab_test_learning_windows.py --generate-report
echo.
pause
