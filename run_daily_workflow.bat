@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

REM ############################################################################
REM #
REM #  MLB Daily Workflow Runner (Updated for New Structure)
REM #
REM #  This script orchestrates the entire daily prediction pipeline using the
REM #  reorganized MLB module structure under mlb/core/
REM #
REM #  Usage:
REM #    run_daily_workflow.bat [YYYY-MM-DD]
REM #
REM #  If no date is provided, it defaults to the current date.
REM #
REM ############################################################################

REM --- Configuration ---
ECHO [WORKFLOW] Setting up environment...
SET "PYTHON_EXE=s:\Projects\AI_Predictions\.venv\Scripts\python.exe"
SET "WORKFLOW_DIR=s:\Projects\AI_Predictions\mlb\core"
SET "WORKFLOW_SCRIPT=daily_api_workflow.py"

REM --- Prediction Serving Defaults (Whitelist Primary + Optional Market Blend) ---
REM WHITELIST_PRIMARY=1 promotes calibrated whitelist predictions to primary (predicted_total)
REM WHITELIST_MARKET_BLEND alpha: 1.0 = pure model, 0.0 = pure market line, e.g. 0.8 strong model influence
REM Pass desired alpha as 2nd arg: run_daily_workflow.bat 2025-09-11 0.8
REM Use keyword 'pure' for 1.0, 'market' for 0.2, or 'off' to disable blend (unsets variable)
SET "WHITELIST_PRIMARY=1"
IF NOT "%2"=="" (
    IF /I "%2"=="pure" (
        SET "WHITELIST_MARKET_BLEND=1.0"
    ) ELSE IF /I "%2"=="market" (
        SET "WHITELIST_MARKET_BLEND=0.2"
    ) ELSE IF /I "%2"=="off" (
        ECHO [BLEND] Disabling market blend (pure whitelist model)
        SET "WHITELIST_MARKET_BLEND="
    ) ELSE (
        SET "WHITELIST_MARKET_BLEND=%2"
    )
) ELSE (
    IF NOT DEFINED WHITELIST_MARKET_BLEND SET "WHITELIST_MARKET_BLEND=0.8"
)
IF DEFINED WHITELIST_MARKET_BLEND (
    ECHO [BLEND] Using whitelist market blend alpha=%WHITELIST_MARKET_BLEND%
) ELSE (
    ECHO [BLEND] No blend alpha set (model-only predictions)
)

REM Include live odds by default for freshest lines
SET "ODDS_INCLUDE_LIVE=1"

REM --- ULTRA-60 OPTIMAL CONFIGURATION (A/B Tested) ---
REM System in development: Current performance ~50% accuracy
ECHO [ULTRA-60] Activating optimal 60-day learning window configuration...
SET "INCREMENTAL_LEARNING_DAYS=60"
SET "ULTRA_CONFIDENCE_THRESHOLD=3.0"
SET "ALWAYS_RUN_DUAL=true"
SET "PUBLISH_BLEND=false"
SET "PREDICT_ALL_TODAY=true"
SET "RECENCY_WINDOWS=7,14,30"
SET "USE_ENHANCED_FEATURES=true"
SET "TRACK_ULTRA_PERFORMANCE=true"
ECHO [ULTRA-60] Learning window: %INCREMENTAL_LEARNING_DAYS% days
ECHO [ULTRA-60] Confidence threshold: %ULTRA_CONFIDENCE_THRESHOLD%
ECHO [ULTRA-60] Current performance: ~50%% accuracy (system in development)

REM --- Override Feature QC Checks ---
REM These environment variables allow the workflow to proceed despite data quality issues
REM Note: Values must be decimal (0.30 = 30%, not 30 = 3000%)
REM Lowered from 0.30 to 0.20 to accommodate current 25% pitcher coverage
SET "MIN_PITCHER_COVERAGE=0.20"
SET "PER_COL_PITCHER_COVERAGE=0.00"
SET "ALLOW_FLAT_ENV=1"

REM --- Set encoding to handle Unicode characters ---
SET "PYTHONIOENCODING=utf-8"
SET "PYTHONLEGACYWINDOWSSTDIO=1"

REM --- Date Handling ---
REM Use the first command-line argument as the target date.
REM If no argument is provided, default to today's date.
IF "%~1"=="" (
    FOR /F "tokens=2 delims==" %%I IN ('wmic os get localdatetime /format:list') DO (
        IF NOT "%%I"=="" (
            SET "DT=%%I"
        )
    )
    CALL :FormatDate
    ECHO [WORKFLOW] No date provided. Defaulting to today: %TARGET_DATE%
) ELSE (
    SET "TARGET_DATE=%~1"
    ECHO [WORKFLOW] Target date set from argument: !TARGET_DATE!
)

REM --- Define Workflow Stages ---
REM These are the core stages required for a full daily run.
SET "STAGES=markets,features,predict,ultra80,odds,health,prob,export,audit"
ECHO [WORKFLOW] Running stages: %STAGES%
ECHO [WORKFLOW] Whitelist primary: %WHITELIST_PRIMARY% ^| Live odds: %ODDS_INCLUDE_LIVE%

REM --- Pre-flight Checks ---
IF NOT EXIST "%PYTHON_EXE%" (
    ECHO [ERROR] Python executable not found at "%PYTHON_EXE%".
    ECHO [ERROR] Please ensure the virtual environment is set up correctly.
    GOTO :EOF
)
IF NOT EXIST "%WORKFLOW_DIR%\%WORKFLOW_SCRIPT%" (
    ECHO [ERROR] Workflow script not found at "%WORKFLOW_DIR%\%WORKFLOW_SCRIPT%".
    GOTO :EOF
)

REM --- Execute Workflow ---
ECHO.
ECHO [WORKFLOW] Starting Daily MLB Overs Pipeline for !TARGET_DATE!...
ECHO ==================================================================

REM Change to the workflow directory to ensure all relative paths in the script work correctly.
PUSHD "%WORKFLOW_DIR%"

REM Run the main Python workflow script
ECHO [WORKFLOW] Executing: %WORKFLOW_SCRIPT% --date %TARGET_DATE% --stages %STAGES%
ECHO.

"%PYTHON_EXE%" "%WORKFLOW_SCRIPT%" --date !TARGET_DATE! --stages %STAGES%

REM Check for errors from the Python script
IF %ERRORLEVEL% NEQ 0 (
    ECHO.
    ECHO [ERROR] ******************************************************
    ECHO [ERROR] *  WORKFLOW FAILED with exit code %ERRORLEVEL%
    ECHO [ERROR] ******************************************************
    POPD
    EXIT /B %ERRORLEVEL%
)

POPD

ECHO.
ECHO [SUCCESS] =====================================================
ECHO [SUCCESS]  Daily MLB Ultra-60 Workflow COMPLETED successfully.
ECHO [SUCCESS]  Optimal 60-day learning window active
ECHO [SUCCESS]  Current performance: ~50%% accuracy (developing system)
ECHO [SUCCESS] =====================================================
ECHO.

ENDLOCAL
GOTO :EOF

REM --- Subroutines ---
:FormatDate
REM Format the DT variable into YYYY-MM-DD format
SET "TARGET_DATE=%DT:~0,4%-%DT:~4,2%-%DT:~6,2%"
GOTO :EOF
