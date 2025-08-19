#!/usr/bin/env powershell
param(
    [string]$Date = (Get-Date).ToString("yyyy-MM-dd")
)

# Enhanced Gameday Pipeline - Skip problematic pitcher stats
Write-Host "ENHANCED GAMEDAY PIPELINE (QUICK TEST) - $Date" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green

$PY = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }

# Check for enhanced files
if (Test-Path "data\enhanced_historical_games_2025.parquet") {
    Write-Host "Enhanced training data found" -ForegroundColor Green
} else {
    Write-Host "Enhanced training data missing" -ForegroundColor Red
    exit 1
}

if (Test-Path "models\enhanced_mlb_predictor.joblib") {
    Write-Host "Enhanced ML model found" -ForegroundColor Green
} else {
    Write-Host "Enhanced ML model missing" -ForegroundColor Red
    exit 1
}

# 1) Basic data collection (skip pitcher stats that cause conflicts)
Write-Host "Fetching basic game data..." -ForegroundColor Cyan
& $PY -m ingestors.games --date $Date

Write-Host "Fetching probable pitchers..." -ForegroundColor Cyan
& $PY -m ingestors.probables_fill --date $Date

# Skip pitcher stats that cause conflicts
Write-Host "Skipping pitcher stats (conflict issue)..." -ForegroundColor Yellow

Write-Host "Fetching bullpen data..." -ForegroundColor Cyan
& $PY -m ingestors.bullpens_daily --date $Date

Write-Host "Fetching offense data..." -ForegroundColor Cyan
& $PY -m ingestors.teams_offense_daily --date $Date

# 2) Betting lines
Write-Host "Fetching betting lines..." -ForegroundColor Cyan
if ($env:THE_ODDS_API_KEY) {
  Write-Host "The Odds API key found - fetching real betting lines" -ForegroundColor Green
  $env:ODDS_BOOK = 'fanduel'
  & $PY -m ingestors.odds_totals --date $Date
} else {
  Write-Host "THE_ODDS_API_KEY not set - using fallback ESPN lines" -ForegroundColor Yellow
  Write-Host "To get real betting lines:" -ForegroundColor Gray
  Write-Host "1. Sign up at https://the-odds-api.com/" -ForegroundColor Gray
  Write-Host "2. Get your free API key (500 requests/month)" -ForegroundColor Gray
  Write-Host "3. Set environment variable THE_ODDS_API_KEY" -ForegroundColor Gray
}
& $PY -m ingestors.espn_totals --date $Date

# 3) Enhanced Features
Write-Host "Building enhanced features..." -ForegroundColor Cyan
if (Test-Path "build_enhanced_features.py") {
    & $PY build_enhanced_features.py
} else {
    Write-Host "Enhanced feature builder not found - using legacy" -ForegroundColor Yellow
    & $PY features\build_features.py --database-url "$env:DATABASE_URL" --out features\train.parquet
}

# 4) Enhanced ML Inference
Write-Host "Running enhanced ML predictions..." -ForegroundColor Cyan
& $PY models\enhanced_infer.py data\game_totals_today.parquet predictions_today.parquet --model-path models\enhanced_mlb_predictor.joblib

# 5) Generate outputs
Write-Host "Generating prediction outputs..." -ForegroundColor Cyan

# Simple output generation
& $PY -c @"
import pandas as pd
import json
from datetime import datetime

try:
    # Load predictions
    df = pd.read_parquet('predictions_today.parquet')
    print(f'Generated predictions for {len(df)} games')
    
    if not df.empty:
        # Create readable CSV
        df.to_csv('readable_predictions.csv', index=False)
        
        # Create UI JSON
        ui_data = {
            'generated_at': datetime.now().isoformat(),
            'model_version': 'enhanced_gameday_v2.0',
            'date': '$Date',
            'games': df.to_dict('records')[:10],  # Limit for safety
            'best_bets': []
        }
        
        with open('../mlb-predictions-ui/public/daily_recommendations.json', 'w') as f:
            json.dump(ui_data, f, indent=2)
        
        print(f'UI data saved: {len(df)} total games')
        
        # Show strong recommendations
        if 'edge' in df.columns:
            strong = df[abs(df['edge']) >= 0.5]
            if len(strong) > 0:
                print(f'Strong recommendations: {len(strong)}')
                for _, row in strong.head().iterrows():
                    rec = 'OVER' if row['edge'] > 0 else 'UNDER'
                    print(f'  {row.get("matchup", "Game")}: {rec} (edge: {row["edge"]:.2f})')
            else:
                print('No strong recommendations found')
        
except Exception as e:
    print(f'Error generating outputs: {e}')
    import traceback
    traceback.print_exc()
"@

Write-Host "Enhanced gameday pipeline complete!" -ForegroundColor Green
Write-Host "Output files:" -ForegroundColor White
Write-Host "  • predictions_today.parquet (ML predictions)" -ForegroundColor Gray
Write-Host "  • readable_predictions.csv (human-friendly format)" -ForegroundColor Gray
Write-Host "  • ../mlb-predictions-ui/public/daily_recommendations.json (UI data)" -ForegroundColor Gray
