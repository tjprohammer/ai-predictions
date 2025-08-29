@echo off
REM 8-Day Incremental Learning Workflow
REM Runs the daily workflow with 8-day learning window

echo 🔄 Setting incremental learning to 8 days...
set INCREMENTAL_LEARNING_DAYS=8

echo 🚀 Running daily workflow with 8-day learning...
call run_daily_workflow.bat

echo ✅ 8-day incremental workflow complete!
pause
