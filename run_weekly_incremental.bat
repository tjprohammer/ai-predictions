@echo off
REM Weekly Incremental Learning Workflow
REM Runs the daily workflow with 7-day learning window

echo 🔄 Setting incremental learning to 7 days...
set INCREMENTAL_LEARNING_DAYS=14

echo 🚀 Running daily workflow with weekly learning...
call run_daily_workflow.bat

echo ✅ Weekly incremental workflow complete!
pause
