@echo off
REM Test script for MLB tracking system organization

echo 🔍 Testing MLB Tracking System Organization
echo ============================================

echo.
echo 📂 Checking directory structure...
if exist "mlb\tracking\performance" (
    echo ✅ Performance directory exists
) else (
    echo ❌ Performance directory missing
    exit /b 1
)

if exist "mlb\tracking\results" (
    echo ✅ Results directory exists  
) else (
    echo ❌ Results directory missing
    exit /b 1
)

if exist "mlb\tracking\validation" (
    echo ✅ Validation directory exists
) else (
    echo ❌ Validation directory missing
    exit /b 1
)

if exist "mlb\tracking\monitoring" (
    echo ✅ Monitoring directory exists
) else (
    echo ❌ Monitoring directory missing
    exit /b 1
)

echo.
echo 📊 Testing validation script...
cd mlb\tracking\validation
python check_predictions_final.py
if %errorlevel% neq 0 (
    echo ❌ Validation test failed
    cd ..\..\..
    exit /b 1
)

echo.
echo ✅ SUCCESS! MLB Tracking system organized successfully!
echo.
echo 📋 Available tracking components:
echo    📊 Performance: enhanced_prediction_tracker.py, weekly_performance_tracker.py
echo    🎯 Results: game_result_tracker.py, simple_results_checker.py  
echo    🔍 Validation: check_predictions_final.py, check_residual_data.py
echo    📱 Monitoring: todays_reality_check.py, auto_prediction_tracker.py
echo.
echo 🎯 Quick usage examples:
echo    python mlb\tracking\validation\check_predictions_final.py
echo    python mlb\tracking\performance\enhanced_prediction_tracker.py
echo    python mlb\tracking\results\simple_results_checker.py
echo    python mlb\tracking\monitoring\todays_reality_check.py

cd ..\..\..
