@echo off
echo.
echo =========================================================
echo 🔬 ULTRA-60 VALIDATION TEST
echo =========================================================
echo Testing the new 60-day optimal configuration
echo This will validate our A/B testing results
echo =========================================================
echo.

REM Test the 60-day configuration
set INCREMENTAL_LEARNING_DAYS=60
set ULTRA_CONFIDENCE_THRESHOLD=3.0
set VALIDATION_MODE=true
set TEST_DATE_RANGE=recent

echo Testing Ultra-60 configuration...
echo Target: 88.6%% accuracy, 69.2%% ROI
echo Confidence threshold: 3.0
echo.

echo Running validation test...
python mlb\analysis\comprehensive_ab_testing.py --test-learning-windows --start-date 2025-08-01 --end-date 2025-08-28

echo.
echo =========================================================
echo Validation test completed!
echo Compare results to expected: 88.6%% accuracy, 69.2%% ROI
echo =========================================================
pause
