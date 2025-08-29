@echo off
REM Blended Prediction Mode: 70% Learning + 30% Original
REM Combines the optimized incremental learning with baseline stability
REM Ideal for conservative betting approach with enhanced performance

echo ========================================
echo     BLENDED PREDICTION MODE
echo ========================================
echo.
echo Configuration:
echo - Learning Window: 14 days (A/B tested optimal)
echo - Recency Features: 7,14,30 day windows
echo - Published Prediction: 70%% learning + 30%% original
echo - Side-by-side Analysis: Available
echo.

REM Set optimal configuration
set INCREMENTAL_LEARNING_DAYS=14
set RECENCY_WINDOWS=7,14,30
set PREDICT_ALL_TODAY=1
set ALWAYS_RUN_DUAL=1
set PUBLISH_BLEND=1

echo Running daily workflow with blended predictions...
python mlb\core\daily_api_workflow.py --stages markets,features,predict,odds,health,prob,export

echo.
echo ========================================
echo    BLENDED PREDICTION COMPLETE
echo ========================================
echo.
echo Published prediction combines:
echo - 70%% Incremental Ultra-80 system (predicted_total_learning)
echo - 30%% Original baseline model (predicted_total_original)
echo.
echo This provides enhanced performance with stability guardrails.
echo.
pause
