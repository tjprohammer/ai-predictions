@echo off
SETLOCAL

REM ############################################################################
REM #
REM #  MLB Overs Daily Workflow Runner
REM #
REM #  This script orchestrates the entire daily prediction pipeline.
REM #  It navigates to the correct directory and executes the main workflow
REM #  script with the necessary stages.
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
SET "WORKFLOW_DIR=s:\Projects\AI_Predictions\mlb-overs\deployment"
SET "WORKFLOW_SCRIPT=daily_api_workflow.py"

REM --- Date Handling ---
REM Use the first command-line argument as the target date.
REM If no argument is provided, default to today's date.
IF "%1"=="" (
    FOR /F "tokens=2 delims==" %%I IN ('wmic os get localdatetime /format:list') DO SET "DT=%%I"
    SET "TARGET_DATE=%DT:~0,4%-%DT:~4,2%-%DT:~6,2%"
    ECHO [WORKFLOW] No date provided. Defaulting to today: %TARGET_DATE%
) ELSE (
    SET "TARGET_DATE=%1"
    ECHO [WORKFLOW] Target date set from argument: %TARGET_DATE%
)

REM --- Define Workflow Stages ---
REM These are the core stages required for a full daily run.
SET "STAGES=markets,features,predict,odds,health,prob,export,audit"
ECHO [WORKFLOW] Running stages: %STAGES%

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
ECHO [WORKFLOW] Starting Daily MLB Overs Pipeline for %TARGET_DATE%...
ECHO ==================================================================

REM Change to the workflow directory to ensure all relative paths in the script work correctly.
PUSHD "%WORKFLOW_DIR%"

REM Run the main Python workflow script
ECHO [WORKFLOW] Executing: %WORKFLOW_SCRIPT% --date %TARGET_DATE% --stages %STAGES%
ECHO.

"%PYTHON_EXE%" "%WORKFLOW_SCRIPT%" --date %TARGET_DATE% --stages %STAGES%

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
ECHO [SUCCESS]  Daily MLB Overs Workflow COMPLETED successfully.
ECHO [SUCCESS] =====================================================
ECHO.

ENDLOCAL
