@echo off
REM Comprehensive A/B Testing for Ultra-80 Incremental Learning System
REM This script provides easy access to various testing scenarios

echo.
echo ========================================
echo Ultra-80 Comprehensive A/B Testing Suite
echo ========================================
echo.

if "%1"=="" goto menu

if "%1"=="all" goto test_all
if "%1"=="windows" goto test_windows
if "%1"=="features" goto test_features
if "%1"=="models" goto test_models
if "%1"=="rates" goto test_rates
if "%1"=="frequencies" goto test_frequencies
if "%1"=="quick" goto test_quick
goto invalid_option

:menu
echo Available testing scenarios:
echo.
echo   1. all         - Run comprehensive testing (ALL scenarios)
echo   2. windows     - Test learning window sizes (3d, 7d, 14d, 21d, 30d)
echo   3. features    - Test feature combinations
echo   4. models      - Test model architectures (SGD, Passive-Aggressive, Online RF)
echo   5. rates       - Test learning rates
echo   6. frequencies - Test update frequencies
echo   7. quick       - Quick test (windows + features only)
echo.
echo Usage: run_comprehensive_ab_test.bat [scenario]
echo Example: run_comprehensive_ab_test.bat all
echo.
goto end

:test_all
echo Running COMPREHENSIVE A/B testing (all scenarios)...
echo This will test:
echo - Learning windows: 3d, 7d, 14d, 21d, 30d
echo - Learning rates: 0.001, 0.01, 0.1, adaptive
echo - Feature combinations: 6 different subsets
echo - Model architectures: SGD, Passive-Aggressive, Online RF
echo - Update frequencies: daily, every-3-days, weekly
echo.
echo Estimated runtime: 45-90 minutes
echo.
set /p confirm="Continue? (y/n): "
if /i "%confirm%" NEQ "y" goto end

echo.
echo Starting comprehensive testing...
python mlb\analysis\comprehensive_ab_testing.py --test-all --start-date 2025-04-01 --end-date 2025-08-27
goto end

:test_windows
echo Running LEARNING WINDOWS A/B testing...
echo Testing window sizes: 3d, 7d, 14d, 21d, 30d
echo Estimated runtime: 10-15 minutes
echo.
python mlb\analysis\comprehensive_ab_testing.py --test-learning-windows --start-date 2025-04-01 --end-date 2025-08-27
goto end

:test_features
echo Running FEATURE COMBINATIONS A/B testing...
echo Testing feature subsets:
echo - Core only (basic team stats)
echo - With pitcher features
echo - With bullpen features  
echo - With handedness splits
echo - With recency features
echo - Full baseball intelligence
echo.
echo Estimated runtime: 15-20 minutes
echo.
python mlb\analysis\comprehensive_ab_testing.py --test-feature-combinations --start-date 2025-04-01 --end-date 2025-08-27
goto end

:test_models
echo Running MODEL ARCHITECTURES A/B testing...
echo Testing models:
echo - SGD Regressor (current)
echo - Passive Aggressive Regressor
echo - Online Random Forest
echo.
echo Estimated runtime: 15-20 minutes
echo.
python mlb\analysis\comprehensive_ab_testing.py --test-model-architectures --start-date 2025-04-01 --end-date 2025-08-27
goto end

:test_rates
echo Running LEARNING RATES A/B testing...
echo Testing rates: 0.001, 0.01, 0.1, adaptive
echo Estimated runtime: 8-12 minutes
echo.
python mlb\analysis\comprehensive_ab_testing.py --test-learning-rates --start-date 2025-04-01 --end-date 2025-08-27
goto end

:test_frequencies
echo Running UPDATE FREQUENCIES A/B testing...
echo Testing frequencies: daily, every-3-days, weekly
echo Estimated runtime: 8-12 minutes
echo.
python mlb\analysis\comprehensive_ab_testing.py --test-update-frequencies --start-date 2025-04-01 --end-date 2025-08-27
goto end

:test_quick
echo Running QUICK A/B testing (windows + features)...
echo This tests the most impactful optimizations:
echo - Learning windows
echo - Feature combinations
echo.
echo Estimated runtime: 20-25 minutes
echo.
python mlb\analysis\comprehensive_ab_testing.py --test-learning-windows --start-date 2025-04-01 --end-date 2025-08-27
echo.
echo Running feature combinations test...
python mlb\analysis\comprehensive_ab_testing.py --test-feature-combinations --start-date 2025-04-01 --end-date 2025-08-27
goto end

:invalid_option
echo Invalid option: %1
echo Run without parameters to see available options
goto end

:end
echo.
echo A/B testing complete! Check the data\ab_test_results folder for detailed results.
echo.
pause
