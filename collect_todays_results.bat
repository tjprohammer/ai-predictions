@echo off
REM ============================================================================
REM  COLLECT TODAY'S COMPLETED GAME RESULTS
REM ============================================================================
REM  Uses existing tracking files to collect and display completed game results
REM  for both learning and ultra models, updating the UI performance tracking.
REM ============================================================================

echo.
echo ============================================================================
echo   COLLECTING TODAY'S COMPLETED GAME RESULTS
echo ============================================================================
echo   Using existing tracking system to gather completed games
echo   This will update the Performance Tracking tab in the UI
echo.

cd /d "S:\Projects\AI_Predictions"

echo [1/3] Showing today's learning model picks and results...
python mlb/tracking/results/simple_results_checker.py

echo.
echo [2/3] Collecting Ultra 80 System results separately...
python mlb/tracking/results/ultra80_results_tracker.py

echo.
echo [3/3] Running comprehensive performance analysis (both systems)...
python mlb/tracking/performance/enhanced_prediction_tracker.py

echo.
echo ============================================================================
echo   RESULTS COLLECTION COMPLETE
echo ============================================================================
echo   - Today's completed games have been processed
echo   - Performance metrics updated for both models
echo   - UI Performance Tracking tab should show latest results
echo.
echo   To view results in UI:
echo   1. Navigate to mlb-predictions-ui directory
echo   2. Run: npm start
echo   3. Click "Performance Tracking" tab
echo.

pause
