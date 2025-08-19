# Handy Commands for MLB Prediction System

## � ENHANCED DAILY WORKFLOW - COMPLETE PIPELINE STATUS: ✅ OPERATIONAL

### **Quick Status Check**

```bash
# Complete pipeline validation
python validate_enhanced_pipeline.py

# Production status dashboard
python mlb-overs/deployment/production_status.py

# Database and API health
python -c "
import requests
response = requests.get('http://localhost:8000/comprehensive-games')
print(f'API Status: {response.status_code}')
data = response.json()
print(f'Games: {len(data.get(\"games\", []))}')
if data.get('games'):
    game = data['games'][0]
    print(f'Sample: {game[\"away_team\"]} @ {game[\"home_team\"]}')
    print(f'Prediction: {game.get(\"predicted_total\")}')
    print(f'Market: {game.get(\"market_total\")}')
    print(f'Weather: {game.get(\"temperature\")}F')
"
```

### **Full Automated Enhanced Pipeline (Recommended)**

```bash
# Complete enhanced daily workflow with all new features
cd mlb-overs/deployment; python daily_runbook.py --date 2025-08-17 --mode predictions

# Alternative: Manual step-by-step execution
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'; $env:DB_NULLPOOL='1'; python enhanced_gameday.ps1 2025-08-17
```

### Daily Pipeline (Fresh Data + Predictions + Evaluation)

```bash
# Full pipeline: markets → features → predict → evaluate
python mlb-overs/deployment/daily_api_workflow.py --stages markets,features,predict,eval

# Today's predictions only
python mlb-overs/deployment/daily_api_workflow.py --stages features,predict

# Yesterday's evaluation only
python mlb-overs/deployment/daily_api_workflow.py --stages eval

# Automated retraining (weekly)
python mlb-overs/deployment/daily_api_workflow.py --stages retrain
```

## 📚 Historical Data & Learning Loop

### Backfill Historical Data

```bash
# Backfill last month (for model training)
python mlb-overs/deployment/backfill_range.py --start 2025-07-15 --end 2025-08-14

# Backfill with predictions (for frontend history)
python mlb-overs/deployment/backfill_range.py --start 2025-07-15 --end 2025-08-14 --predict

# Skip weather if historical unavailable
python mlb-overs/deployment/backfill_range.py --start 2025-07-15 --end 2025-08-14 --no-weather
```

### Check Training Data Coverage

```bash
# Basic coverage check
python mlb-overs/deployment/coverage_check.py --start 2025-07-15 --end 2025-08-14

# Require market data for training
python mlb-overs/deployment/coverage_check.py --start 2025-07-15 --end 2025-08-14 --require-market
```

### Model Retraining

```bash
# Standard retraining (150-day window, 21-day holdout)
python mlb-overs/deployment/retrain_model.py --end 2025-08-14 --deploy --audit

# Custom window (last month training)
python mlb-overs/deployment/retrain_model.py --end 2025-08-14 --window-days 31 --holdout-days 7 --deploy --audit

# Require market data for training
python mlb-overs/deployment/retrain_model.py --end 2025-08-14 --window-days 31 --require-market --deploy
```

### Complete Backfill → Retrain Workflow

```bash
# One command: backfill + coverage check + retrain + deploy
python mlb-overs/deployment/complete_backfill_retrain.py --start 2025-07-15 --end 2025-08-14

# With predictions for frontend history
python mlb-overs/deployment/complete_backfill_retrain.py --start 2025-07-15 --end 2025-08-14 --predict

# Dry run (show commands without executing)
python mlb-overs/deployment/complete_backfill_retrain.py --start 2025-07-15 --end 2025-08-14 --dry-run
```

## 🛠️ Development & Testing

### Model Validation & Auditing

```bash
# Full audit with truth validation (30 days)
python training_bundle_audit.py --target-date 2025-08-15 --serve-days 7 --truth-days 30

# PSI monitoring (population stability)
python training_bundle_audit.py --target-date 2025-08-15 --serve-days 7 --psi-min-serving 100 --psi-min-training 1000

# Early season / low data mode
python training_bundle_audit.py --target-date 2025-08-15 --serve-days 14 --psi-min-serving 60 --psi-min-training 500
```

### Direct Prediction

```bash
# Generate predictions for specific date
python enhanced_bullpen_predictor.py --target-date 08-15-2025

# Training data snapshot
python build_training_snapshot.py --max-rows 2000
```

### Learning Loop Testing

```bash
# Test evaluation and retraining components
python test_learning_loop.py
```

## 📊 Data Pipeline Components

### Individual Ingestors (for debugging)

```bash
# ESPN odds
python mlb-overs/deployment/enhanced_market_collector.py --date 2025-08-15

# Pitcher stats
python mlb-overs/deployment/working_pitcher_ingestor.py --date 2025-08-15

# Team stats
python mlb-overs/deployment/working_team_ingestor.py --date 2025-08-15

# Weather data
python mlb-overs/deployment/working_weather_ingestor.py --date 2025-08-15

# Game results
python mlb-overs/deployment/working_games_ingestor.py --date 2025-08-15
```

## 🎯 Production Workflows

### Daily Schedule (Recommended)

```bash
# 6 AM: Fresh data collection
python mlb-overs/deployment/daily_api_workflow.py --stages markets

# 7 AM: Generate today's predictions
python mlb-overs/deployment/daily_api_workflow.py --stages features,predict

# 8 AM: Evaluate yesterday's performance
python mlb-overs/deployment/daily_api_workflow.py --stages eval --date $(date -d "yesterday" +%Y-%m-%d)

# Sunday: Weekly retraining
python mlb-overs/deployment/daily_api_workflow.py --stages retrain
```

### Seasonal Setup

```bash
# Start of season: backfill preseason + early games
python mlb-overs/deployment/complete_backfill_retrain.py --start 2025-03-01 --end 2025-04-15 --predict

# Mid-season: rolling 150-day window retraining
python mlb-overs/deployment/retrain_model.py --end $(date +%Y-%m-%d) --window-days 150 --deploy --audit

# End of season: full season training for next year baseline
python mlb-overs/deployment/retrain_model.py --end 2025-10-31 --window-days 200 --deploy --audit
```

## 🔍 Monitoring & Troubleshooting

### Check System Health

```bash
# Database coverage
python mlb-overs/deployment/coverage_check.py --start $(date -d "30 days ago" +%Y-%m-%d) --end $(date +%Y-%m-%d)

# Model performance trends
python mlb-overs/deployment/daily_api_workflow.py --stages eval --date $(date -d "7 days ago" +%Y-%m-%d)

# Bundle provenance
grep "Bundle logged" mlb-overs/deployment/daily_api_workflow.log
```

### Verify Feature Engineering Pipeline

```bash
# Test enhanced predictor (with all variance fixes)
python mlb-overs/deployment/enhanced_bullpen_predictor.py --target-date 08-16-2025

# Verify zero-variance features are fixed
python -c "
from enhanced_bullpen_predictor import EnhancedBullpenPredictor
predictor = EnhancedBullpenPredictor()
result = predictor.predict_today_games('2025-08-16')
" | findstr "ballpark.*factor\|wind_speed\|temperature\|offense_imbalance"

# Check API predictions
python -c "
import requests
response = requests.get('http://localhost:8000/api/comprehensive-games/2025-08-16')
data = response.json()
games = data.get('games', [])
print(f'API serving {len(games)} games')
for g in games[:3]:
    print(f'{g.get(\"away_team\")} @ {g.get(\"home_team\")}: {g.get(\"predicted_total\")} (market: {g.get(\"market_total\")})')
"
```

### Feature Engineering Debugging

```bash
# Compare daily workflow vs enhanced predictor predictions
python -c "
import pandas as pd
from enhanced_bullpen_predictor import EnhancedBullpenPredictor
predictor = EnhancedBullpenPredictor()
engine = predictor.get_engine()

# Get stored predictions from daily workflow
stored_query = '''SELECT away_team, home_team, predicted_total FROM enhanced_games WHERE date = '2025-08-16' ORDER BY game_id LIMIT 3'''
stored_df = pd.read_sql(stored_query, engine)
print('Daily Workflow predictions:')
for _, row in stored_df.iterrows():
    print(f'{row[\"away_team\"]} @ {row[\"home_team\"]}: {row[\"predicted_total\"]:.1f}')

# Run enhanced predictor
print('\nEnhanced Predictor predictions:')
result = predictor.predict_today_games('2025-08-16')
"

# Verify ballpark factors working
python -c "
from enhanced_bullpen_predictor import EnhancedBullpenPredictor
predictor = EnhancedBullpenPredictor()
result = predictor.predict_today_games('2025-08-16')
" | findstr "SANITY ballpark.*factor.*std.*0\.0"

# Check weather data integration
python -c "
from enhanced_bullpen_predictor import EnhancedBullpenPredictor
predictor = EnhancedBullpenPredictor()
result = predictor.predict_today_games('2025-08-16')
" | findstr "wind_speed.*[1-9]\|temperature.*[1-9]"
```

### Common Debugging

```bash
# Check API status
curl http://localhost:8000/api/comprehensive-games/$(date +%Y-%m-%d)

# Validate predictions written
python -c "from sqlalchemy import create_engine; engine = create_engine('postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'); print(pd.read_sql('SELECT COUNT(*) FROM enhanced_games WHERE predicted_total IS NOT NULL AND date = CURRENT_DATE', engine))"

# Check recent evaluation results
python -c "from sqlalchemy import create_engine; engine = create_engine('postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'); print(pd.read_sql('SELECT * FROM model_eval_daily ORDER BY eval_date DESC LIMIT 5', engine))"
```

# PowerShell

#

$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'; $env:DB_NULLPOOL='1' 
$DATE = '2025-08-17'; $CAL_START = (Get-Date $DATE).AddDays(-31).ToString('yyyy-MM-dd'); $CAL_END = (Get-Date $DATE).AddDays(-1).ToString('yyyy-MM-dd'); echo "DATE: $DATE"; echo "CAL_START: $CAL_START"; echo "CAL_END: $CAL_END"

python working_games_ingestor.py --date 2025-08-17 --refresh

## 🚀 ENHANCED DAILY WORKFLOW - COMPLETE PIPELINE

### **Full Automated Enhanced Pipeline (Recommended)**

```bash
# Complete enhanced daily workflow with all new features
python mlb-overs/deployment/daily_api_workflow.py --date 2025-08-17

# Alternative PowerShell enhanced workflow
$env:PYTHONIOENCODING='utf-8'; $env:PYTHONUTF8='1'; $env:DB_NULLPOOL='1'; python enhanced_gameday.ps1 2025-08-17
```

### **Manual Enhanced Pipeline Steps**

```bash
# STEP 1: Collect Real Market Data (Fixed market data bug)
python mlb-overs/data_collection/real_market_ingestor.py --date 2025-08-17

# STEP 2: Enhanced Weather Collection (Real-time APIs)
python mlb-overs/data_collection/weather_ingestor.py --date 2025-08-17 --force-update

# STEP 3: Enhanced ML Predictions (Pitcher fallbacks + feature fixes)
python mlb-overs/deployment/enhanced_bullpen_predictor.py --target-date 2025-08-17

# STEP 4: Advanced Betting Analysis
python mlb-overs/deployment/enhanced_analysis.py --date 2025-08-17

# Complete workflow
cd mlb-overs/deployment; python daily_runbook.py --date 2025-08-17 --mode predictions

# Individual components (with UTF-8 encoding)
$env:PYTHONIOENCODING='utf-8'; python mlb-overs/data_collection/working_games_ingestor.py --target-date 2025-08-17
$env:PYTHONIOENCODING='utf-8'; python mlb-overs/data_collection/real_market_ingestor.py --date 2025-08-17
$env:PYTHONIOENCODING='utf-8'; python mlb-overs/data_collection/weather_ingestor.py --date 2025-08-17 --force-update
python mlb-overs/deployment/enhanced_bullpen_predictor.py --target-date 2025-08-17
python mlb-overs/deployment/enhanced_analysis.py --date 2025-08-17

# Validation
python validate_enhanced_pipeline.py
```

### **Data Collection Commands**

```bash
# Real market totals from The Odds API (ENHANCED: Pregame-only by default)
python mlb-overs/data_collection/real_market_ingestor.py --date 2025-08-17

# Force include live/in-progress games (NOT recommended for model training)
python mlb-overs/data_collection/real_market_ingestor.py --date 2025-08-17 --include-live

# Real-time weather with OpenWeather API integration
python mlb-overs/data_collection/weather_ingestor.py --date 2025-08-17 --force-update

# Verify market data collection (should show pregame-only games)
python -c "from sqlalchemy import create_engine; import pandas as pd; engine = create_engine('postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'); df = pd.read_sql('SELECT COUNT(*) as market_games FROM enhanced_games WHERE date = \'2025-08-17\' AND market_total IS NOT NULL', engine); print(f'Market data: {df.iloc[0][\"market_games\"]} games')"

# Verify weather variance (should show realistic 65-95°F range)
python -c "from sqlalchemy import create_engine; import pandas as pd; engine = create_engine('postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'); df = pd.read_sql('SELECT MIN(temperature) as min_temp, MAX(temperature) as max_temp, COUNT(*) as weather_games FROM enhanced_games WHERE date = \'2025-08-17\' AND temperature IS NOT NULL', engine); print(f'Weather: {df.iloc[0][\"weather_games\"]} games, {df.iloc[0][\"min_temp\"]:.0f}°F - {df.iloc[0][\"max_temp\"]:.0f}°F')"
```

### **Enhanced Prediction Commands**

```bash
# Generate predictions with enhanced features (handles TBD pitchers)
python mlb-overs/deployment/enhanced_bullpen_predictor.py --target-date 2025-08-17

# Debug pitcher assignments (shows league average fallbacks)
python -c "from enhanced_bullpen_predictor import EnhancedBullpenPredictor; predictor = EnhancedBullpenPredictor(); result = predictor.predict_today_games('2025-08-17')"

# Verify feature engineering (check zero variance fixes)
python -c "from enhanced_bullpen_predictor import EnhancedBullpenPredictor; predictor = EnhancedBullpenPredictor(); result = predictor.predict_today_games('2025-08-17')" | findstr "SANITY\|zero variance\|ballpark.*factor"
```

### **System Validation Commands**

```bash
# Complete data pipeline verification
python -c "
from sqlalchemy import create_engine
import pandas as pd
engine = create_engine('postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
df = pd.read_sql('''
    SELECT
        COUNT(*) as total_games,
        COUNT(market_total) as market_data,
        COUNT(temperature) as weather_data,
        COUNT(predicted_total) as predictions,
        COUNT(away_pitcher) as pitcher_data
    FROM enhanced_games
    WHERE date = \'2025-08-17\'
''', engine)
print('Enhanced Pipeline Status:')
print(f'Total Games: {df.iloc[0][\"total_games\"]}')
print(f'Market Data: {df.iloc[0][\"market_data\"]} games')
print(f'Weather Data: {df.iloc[0][\"weather_data\"]} games')
print(f'Predictions: {df.iloc[0][\"predictions\"]} games')
print(f'Pitcher Data: {df.iloc[0][\"pitcher_data\"]} games')
"

# Enhanced API test
curl "http://localhost:8000/api/comprehensive-games/2025-08-17" | python -m json.tool
```

### **Troubleshooting Enhanced Pipeline**

```bash
# Test market data API connectivity
python -c "import requests, os; api_key = os.getenv('ODDS_API_KEY'); response = requests.get(f'https://api.the-odds-api.com/v4/sports/baseball_mlb/odds?apiKey={api_key}&regions=us&markets=totals'); print(f'Market API Status: {response.status_code}, Games: {len(response.json()) if response.status_code == 200 else \"Error\"}')"

# Test weather API connectivity
python -c "import requests, os; api_key = os.getenv('OPENWEATHER_API_KEY'); response = requests.get(f'http://api.openweathermap.org/data/2.5/weather?q=New York&appid={api_key}') if api_key else None; print(f'Weather API Status: {response.status_code if response else \"No API key - using fallbacks\"}')"

# Debug weather variance issues
python mlb-overs/data_collection/weather_ingestor.py --date 2025-08-17 --debug

# Check for zero variance feature warnings
python mlb-overs/deployment/enhanced_bullpen_predictor.py --target-date 2025-08-17 | findstr "zero variance\|SANITY\|ballpark"

# WORKING CLI TEST COMMANDS (Unicode-safe)
$env:PYTHONIOENCODING='utf-8'; python mlb-overs/data_collection/working_games_ingestor.py --target-date 2025-08-17
$env:PYTHONIOENCODING='utf-8'; python mlb-overs/data_collection/real_market_ingestor.py --date 2025-08-17
$env:PYTHONIOENCODING='utf-8'; python mlb-overs/data_collection/weather_ingestor.py --date 2025-08-17 --force-update
python mlb-overs/deployment/enhanced_bullpen_predictor.py --target-date 2025-08-17
python mlb-overs/deployment/enhanced_analysis.py --date 2025-08-17

# Quick pipeline validation (PowerShell-safe)
python validate_enhanced_pipeline.py
```

# Complete daily workflow - now fully operational

python daily_runbook.py --date 2025-08-17 --mode predictions

# Individual components also working

python production_status.py # System health dashboard
python enhanced_analysis.py # Detailed bet analysis
python reliability_brier.py # Calibration monitoring
