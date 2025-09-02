@echo off
REM Daily Results Collection and Tracking Update
REM Collects completed game results and updates tracking for UI display

echo 🎯 DAILY RESULTS COLLECTION AND TRACKING UPDATE
echo ================================================
echo.

REM Get today's date
for /f %%i in ('powershell -Command "(Get-Date).ToString('yyyy-MM-dd')"') do set TODAY=%%i

echo 📅 Processing results for: %TODAY%
echo.

echo 🔍 Step 1: Collecting Game Results...
echo ----------------------------------------
python mlb\tracking\results\game_result_tracker.py
if %ERRORLEVEL% neq 0 (
    echo ❌ Failed to collect game results
    goto :error
)

echo.
echo 📊 Step 2: Running Results Checker...
echo ----------------------------------------
python mlb\tracking\results\simple_results_checker.py
if %ERRORLEVEL% neq 0 (
    echo ❌ Failed to run results checker
    goto :error
)

echo.
echo 📈 Step 3: Performance Analysis...
echo ----------------------------------------
python mlb\tracking\performance\enhanced_prediction_tracker.py
if %ERRORLEVEL% neq 0 (
    echo ❌ Failed to run performance analysis
    goto :error
)

echo.
echo 🔍 Step 4: Reality Check Validation...
echo ----------------------------------------
python mlb\tracking\monitoring\todays_reality_check.py
if %ERRORLEVEL% neq 0 (
    echo ⚠️ Reality check had issues (continuing...)
)

echo.
echo 🔄 Step 5: Validating Database Updates...
echo ----------------------------------------
python mlb\tracking\validation\check_predictions_final.py
if %ERRORLEVEL% neq 0 (
    echo ❌ Failed validation check
    goto :error
)

echo.
echo 📊 Step 6: Learning Model Impact Analysis...
echo ----------------------------------------
python mlb\tracking\performance\learning_impact_tracker.py
if %ERRORLEVEL% neq 0 (
    echo ⚠️ Learning impact analysis had issues (continuing...)
)

echo.
echo 🌐 Step 7: Testing API Integration...
echo ----------------------------------------
powershell -Command "try { $result = Invoke-RestMethod 'http://localhost:8000/api/comprehensive-tracking?days=1'; Write-Host 'API Response:'; Write-Host ('Total Games: ' + $result.performance.total_games); Write-Host ('Completed Games: ' + $result.performance.completed_games); Write-Host ('Learning Predictions: ' + $result.performance.learning_predictions); Write-Host ('Ultra Predictions: ' + $result.performance.ultra_predictions); if ($result.performance.learning_mae) { Write-Host ('Learning MAE: ' + [math]::Round($result.performance.learning_mae, 2)) } if ($result.performance.ultra_mae) { Write-Host ('Ultra MAE: ' + [math]::Round($result.performance.ultra_mae, 2)) } } catch { Write-Host 'API test failed - check if server is running' }"

echo.
echo ✅ SUCCESS! Daily results collection and tracking complete!
echo.
echo 🎯 Next Steps:
echo    • Check the 📊 Performance Tracking tab in the UI
echo    • Verify completed games show results and errors
echo    • Review learning vs ultra performance comparison
echo.
echo 💡 UI Access:
echo    • Navigate to http://localhost:3000
echo    • Click "📊 Performance Tracking" tab
echo    • View updated performance metrics and completed games
echo.

goto :end

:error
echo.
echo ❌ FAILED! Daily results collection encountered errors
echo Check the logs above for details
echo.
pause
exit /b 1

:end
echo 🎉 Ready for UI review!
REM Don't pause for automated runs
REM pause
