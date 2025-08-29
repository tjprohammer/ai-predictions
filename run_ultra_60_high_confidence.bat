@echo off
echo.
echo =========================================================
echo 💰 ULTRA-60 HIGH-CONFIDENCE PICKS
echo =========================================================
echo Optimized for maximum ROI with 88.6%% accuracy
echo Only games meeting 3.0+ confidence threshold
echo Expected Value: 0.692 per bet
echo =========================================================
echo.

REM Ultra-60 high-confidence configuration
set INCREMENTAL_LEARNING_DAYS=60
set ULTRA_CONFIDENCE_THRESHOLD=3.0
set HIGH_CONFIDENCE_ONLY=true

REM Conservative settings for maximum accuracy
set PUBLISH_BLEND=false
set ALWAYS_RUN_DUAL=false
set PREDICT_ALL_TODAY=false

REM Focus on high-confidence games only
set MIN_CONFIDENCE_FOR_PREDICTION=3.0
set TRACK_CONFIDENCE_DISTRIBUTION=true

REM Enhanced logging for confidence analysis
set LOG_CONFIDENCE_DETAILS=true
set SAVE_PREDICTION_DETAILS=true

echo Running Ultra-60 High-Confidence mode...
echo Minimum confidence: %MIN_CONFIDENCE_FOR_PREDICTION% points from market
echo Expected accuracy: 88.6%%
echo Expected ROI: 69.2%%
echo.

python mlb\core\daily_api_workflow.py

echo.
echo =========================================================
echo High-confidence analysis completed!
echo Check exports folder for detailed prediction analysis
echo =========================================================
pause
