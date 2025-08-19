param(
  [string]$Date = (Get-Date).ToString('yyyy-MM-dd')
)

$ErrorActionPreference = 'Stop'

# cd to repo root (this script is in ...\mlb-overs\scripts)
Set-Location (Join-Path $PSScriptRoot '..')

# pick venv python (fallback to PATH if not found)
$PY = Join-Path $PWD '.venv\Scripts\python.exe'
if (-not (Test-Path $PY)) { $PY = 'python' }

if (-not $env:DATABASE_URL) {
  $env:DATABASE_URL = 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'
}

Write-Host "ENHANCED GAMEDAY PIPELINE - $Date" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green

# Check if we're using enhanced features and models
if (Test-Path "features\train.parquet") {
    Write-Host "Enhanced training data found" -ForegroundColor Green
} else {
    Write-Host "Enhanced training data not found - run build_enhanced_features.py first" -ForegroundColor Yellow
}

if (Test-Path "models\enhanced_mlb_predictor.joblib") {
    Write-Host "Enhanced ML model found" -ForegroundColor Green
} else {
    Write-Host "Enhanced ML model not found - run train_enhanced_model.py first" -ForegroundColor Yellow
}

Write-Host "Checking database status..." -ForegroundColor Cyan

# 1) Games
Write-Host "Fetching games data..." -ForegroundColor Cyan
& $PY -m ingestors.games --start $Date --end $Date

# 2) Probables
Write-Host "Fetching probable pitchers..." -ForegroundColor Cyan
& $PY -m ingestors.probables_fills --date $Date

# 3) Pitchers (last 10 via Statcast)
Write-Host "Fetching pitcher stats..." -ForegroundColor Cyan
& $PY -m ingestors.pitchers_last10 --start $Date --end $Date

# 4) Bullpens (need 2-day lookback for availability)
Write-Host "Fetching bullpen data..." -ForegroundColor Cyan
$StartBP = (Get-Date $Date).AddDays(-2).ToString('yyyy-MM-dd')
& $PY -m ingestors.bullpens_daily --start $StartBP --end $Date

# 5) Offense (use small window so Statcast has data)
Write-Host "Fetching offense data..." -ForegroundColor Cyan
$StartOff = (Get-Date $Date).AddDays(-2).ToString('yyyy-MM-dd')
& $PY -m ingestors.offense_daily --start $StartOff --end $Date

# 6) Markets (FanDuel via aggregator if key, plus ESPN fallback)
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

# 7) Enhanced Features
Write-Host "Building enhanced features..." -ForegroundColor Cyan
if (Test-Path "build_enhanced_features.py") {
    & $PY build_enhanced_features.py
} else {
    & $PY features\build_features.py --database-url "$env:DATABASE_URL" --out features\train.parquet
}

# 8) Enhanced ML Inference
Write-Host "Running enhanced ML predictions..." -ForegroundColor Cyan
& $PY models\enhanced_infer.py data\game_totals_today.parquet predictions_today.parquet --model-path models\enhanced_mlb_predictor.joblib

# 9) Generate outputs
Write-Host "Generating prediction outputs..." -ForegroundColor Cyan

# Create enhanced prediction summary
& $PY -c @"
import pandas as pd
import json
import numpy as np
from datetime import datetime
from pathlib import Path

try:
    # Load predictions
    df = pd.read_parquet('predictions_today.parquet')
    print(f'Generated predictions for {len(df)} games')
    
    # Create readable CSV
    readable_cols = ['date', 'home_team', 'away_team', 'k_close', 'y_pred']
    if 'edge' in df.columns:
        readable_cols.append('edge')
        df['rec'] = df['edge'].apply(lambda e: 'Over' if e > 0 else 'Under')
        readable_cols.append('rec')
    
    available_cols = [c for c in readable_cols if c in df.columns]
    readable = df[available_cols].copy()
    readable.to_csv('readable_predictions.csv', index=False)
    
    # Create UI-compatible JSON
    ui_data = {
        'generated_at': datetime.now().isoformat(),
        'model_version': 'enhanced_gameday_v2.0',
        'date': datetime.now().strftime('%Y-%m-%d'),
        'weather_enabled': True,
        'enhanced_features': True,
        'games': [],
        'best_bets': []
    }
    
    for _, row in df.iterrows():
        edge = row.get('edge', 0.0)
        game_data = {
            'game_id': str(row.get('game_id', '')),
            'matchup': f"{row.get('away_team', 'Away')} @ {row.get('home_team', 'Home')}",
            'away_team': row.get('away_team', 'Away'),
            'home_team': row.get('home_team', 'Home'),
            'ai_prediction': float(row.get('y_pred', 8.5)),
            'market_total': float(row.get('k_close', 8.5)),
            'difference': float(edge),
            'recommendation': 'OVER' if edge > 0.5 else 'UNDER' if edge < -0.5 else 'NO_BET',
            'confidence': 'HIGH' if abs(edge) >= 1.0 else 'MEDIUM' if abs(edge) >= 0.5 else 'LOW',
            'enhanced_model': True
        }
        ui_data['games'].append(game_data)
    
    # Create best bets
    strong_bets = [g for g in ui_data['games'] if g['confidence'] in ['HIGH', 'MEDIUM']]
    strong_bets.sort(key=lambda x: abs(x['difference']), reverse=True)
    ui_data['best_bets'] = strong_bets[:5]
    
    # Save to UI directory
    ui_path = '../mlb-predictions-ui/public/daily_recommendations.json'
    try:
        with open(ui_path, 'w') as f:
            json.dump(ui_data, f, indent=2)
        print(f'UI data saved: {len(ui_data["games"])} games, {len(ui_data["best_bets"])} best bets')
    except:
        print('Could not save UI data (directory may not exist)')
    
    # Print summary
    print()
    print('ENHANCED PREDICTIONS SUMMARY')
    print('=' * 50)
    if 'edge' in df.columns:
        strong_count = len(df[df['edge'].abs() >= 1.0])
        moderate_count = len(df[(df['edge'].abs() >= 0.5) & (df['edge'].abs() < 1.0)])
        print(f'Strong Plays: {strong_count}')
        print(f'Moderate Plays: {moderate_count}')
    
    print()
    print('Top Predictions:')
    if 'edge' in df.columns:
        top_preds = df.reindex(df['edge'].abs().sort_values(ascending=False).index)
        display_cols = ['home_team', 'away_team', 'k_close', 'y_pred', 'edge', 'rec']
        available_display = [c for c in display_cols if c in top_preds.columns]
        print(top_preds[available_display].head(10).to_string(index=False))
    
except Exception as e:
    print(f'Error processing predictions: {e}')
"@

Write-Host ""
Write-Host "Enhanced gameday pipeline complete!" -ForegroundColor Green
Write-Host "Output files:" -ForegroundColor Cyan
Write-Host "  • predictions_today.parquet (ML predictions)" -ForegroundColor Gray
Write-Host "  • readable_predictions.csv (human-friendly format)" -ForegroundColor Gray
Write-Host "  • ../mlb-predictions-ui/public/daily_recommendations.json (UI data)" -ForegroundColor Gray
