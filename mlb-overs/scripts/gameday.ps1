# --- Pick the date ---
$DATE = '2025-08-17'
$CAL_START = (Get-Date $DATE).AddDays(-31).ToString('yyyy-MM-dd')
$CAL_END   = (Get-Date $DATE).AddDays(-1).ToString('yyyy-MM-dd')

# Console UTF-8 (optional)
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'

# 0) Seed schedule (this is what was missing)
cd S:\Projects\AI_Predictions\mlb-overs\data_collection
python working_games_ingestor.py --date $DATE --refresh

# 1) Collect inputs for the target date
python enhanced_market_collector.py   --date $DATE
python enhanced_weather_ingestor.py   --date $DATE
python season_stats_collector.py      --target-date $DATE

# 2) Rebuild predictions for the last ~30 days for isotonic calibration
cd ..\deployment
python predict_from_range.py --start $CAL_START --end $CAL_END --thr 2.0

# 3) Run the day’s workflow end-to-end
python daily_api_workflow.py --date $DATE --stages features,predict,probabilities

# 4) Quick verification
python verify_fixed_data.py
