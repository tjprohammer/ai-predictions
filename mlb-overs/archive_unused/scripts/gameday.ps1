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

Write-Host "🎯 ENHANCED GAMEDAY PIPELINE - $Date" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green

# Check if we're using enhanced features and models
if (Test-Path "features\train.parquet") {
    $trainData = & $PY -c "import pandas as pd; df = pd.read_parquet('features/train.parquet'); print(f'Enhanced training data: {len(df)} games')"
    Write-Host "✅ $trainData" -ForegroundColor Green
} else {
    Write-Host "⚠️  Enhanced training data not found - run build_enhanced_features.py first" -ForegroundColor Yellow
}

if (Test-Path "models\enhanced_mlb_predictor.joblib") {
    Write-Host "✅ Enhanced ML model found" -ForegroundColor Green
} else {
    Write-Host "⚠️  Enhanced ML model not found - run train_enhanced_model.py first" -ForegroundColor Yellow
}

Write-Host "`n🔍 Checking database status..." -ForegroundColor Cyan

docker exec -it mlb-postgres psql -U mlbuser -d mlb -c "SELECT game_id, home_sp_id, away_sp_id FROM games WHERE date='2025-08-11';"

docker exec -it mlb-postgres psql -U mlbuser -d mlb -c "SELECT COUNT(*) FROM pitchers_starts WHERE date < '2025-08-11';"

docker exec -it mlb-postgres psql -U mlbuser -d mlb -c "SELECT team, date, vs_rhp_xwoba, vs_lhp_xwoba FROM teams_offense_daily WHERE date<='2025-08-11' ORDER BY date DESC LIMIT 20;"

docker exec -it mlb-postgres psql -U mlbuser -d mlb -c "SELECT team, date, bp_fip, closer_back2back_flag FROM bullpens_daily WHERE date<'2025-08-11' ORDER BY date DESC LIMIT 20;"


# 1) Games
& $PY -m ingestors.games --start $Date --end $Date

#probables
python -m ingestors.probables_fills --date $Date


# 2) Pitchers (last 10 via Statcast; works with or without games in DB)
& $PY -m ingestors.pitchers_last10 --start $Date --end $Date

# 3) Bullpens (need 2-day lookback for availability)
$StartBP = (Get-Date $Date).AddDays(-2).ToString('yyyy-MM-dd')
& $PY -m ingestors.bullpens_daily --start $StartBP --end $Date

# 4) Offense (use small window so Statcast has data)
$StartOff = (Get-Date $Date).AddDays(-2).ToString('yyyy-MM-dd')
& $PY -m ingestors.offense_daily --start $StartOff --end $Date

# 5) Markets (FanDuel via aggregator if key, plus ESPN fallback)
Write-Host "`n💰 Fetching betting lines..." -ForegroundColor Cyan
if ($env:THE_ODDS_API_KEY) {
  Write-Host "✅ The Odds API key found - fetching real betting lines" -ForegroundColor Green
  $env:ODDS_BOOK = 'fanduel'
  & $PY -m ingestors.odds_totals --date $Date
} else {
  Write-Host "⚠️  THE_ODDS_API_KEY not set - using fallback ESPN lines" -ForegroundColor Yellow
  Write-Host "   To get real betting lines:" -ForegroundColor Gray
  Write-Host "   1. Sign up at https://the-odds-api.com/" -ForegroundColor Gray
  Write-Host "   2. Get your free API key (500 requests/month)" -ForegroundColor Gray
  Write-Host "   3. Set: `$env:THE_ODDS_API_KEY = 'your-key-here'" -ForegroundColor Gray
}
& $PY -m ingestors.espn_totals --date $Date

# 6) Enhanced Features + 7) Enhanced ML Inference
Write-Host "`n🎯 Building enhanced features..." -ForegroundColor Cyan
if (Test-Path "build_enhanced_features.py") {
    & $PY build_enhanced_features.py
} else {
    & $PY features\build_features.py --database-url "$env:DATABASE_URL" --out features\train.parquet
}

Write-Host "`n🤖 Running enhanced ML predictions..." -ForegroundColor Cyan
& $PY models\infer.py --database-url "$env:DATABASE_URL" --out predictions_today.parquet

# 8) Strong plays (PowerShell here-string piped to python)
@'
import pandas as pd
from models.filter import strong_plays
try:
    df = pd.read_parquet("predictions_today.parquet")
except Exception as e:
    print("could not load predictions_today.parquet:", e)
else:
    # ensure required columns exist
    if "edge" not in df.columns and set(["y_pred","k_close"]).issubset(df.columns):
        df["edge"] = df["y_pred"] - df["k_close"]
    for c in ["p_over_cal","p_under_cal"]:
        if c not in df.columns:
            df[c] = pd.NA
    # run filter with safe defaults
    try:
        sp = strong_plays(df, min_edge=0.6, min_conf=0.80)
    except Exception as e:
        print("strong_plays failed (likely missing calibrated probs). Falling back to edge-only.")
        # simple fallback: pick by absolute edge >= 0.6
        if "edge" in df.columns:
            sp = df.loc[df["edge"].abs() >= 0.6].copy()
            sp["conf"] = pd.NA
        else:
            sp = pd.DataFrame()
    sp.to_csv("strong_plays.csv", index=False)
    print("Strong plays (top 10):")
    print(sp.head(10).to_string(index=False))
'@ | & $PY -

# 9) Enhanced human-readable predictions with weather and pitcher analysis
Write-Host "`n📊 Generating enhanced predictions summary..." -ForegroundColor Cyan
@'
import pandas as pd
import numpy as np
from pathlib import Path

p = Path("predictions_today.parquet")
if not p.exists():
    print("predictions_today.parquet not found; skipping readable export")
else:
    df = pd.read_parquet(p)

    # Enhanced columns for display
    preferred = [
        "date","game_id","home_team","away_team","k_close",
        "y_pred","edge","p_over_cal","p_under_cal","conf",
        "home_pitcher_era_season","away_pitcher_era_season",
        "home_pitcher_era_l5","away_pitcher_era_l5",
        # Add weather if available
        "temp_f","wind_mph","pf_runs_3y"
    ]
    
    # Use available columns
    cols = [c for c in preferred if c in df.columns]
    if not cols:
        cols = [c for c in ["date","home_team","away_team","k_close","y_pred"] if c in df.columns]

    readable = df[cols].copy()

    # Enhanced analysis
    if "edge" in readable.columns:
        readable["rec"] = readable["edge"].apply(lambda e: "Over" if e > 0 else "Under")
        readable["strength"] = readable["edge"].apply(lambda e: 
            "STRONG" if abs(e) >= 1.0 else "MODERATE" if abs(e) >= 0.5 else "WEAK")
        readable = readable.reindex(readable["edge"].abs().sort_values(ascending=False).index)

    # Add prediction confidence analysis
    if "y_pred" in readable.columns and "k_close" in readable.columns:
        readable["pred_vs_line"] = (readable["y_pred"] - readable["k_close"]).round(2)
        readable["line_accuracy"] = readable["pred_vs_line"].apply(lambda x:
            "HIGH_CONFIDENCE" if abs(x) >= 1.5 else 
            "MEDIUM_CONFIDENCE" if abs(x) >= 0.8 else "LOW_CONFIDENCE")

    readable.to_csv("readable_predictions.csv", index=False)
    
    print("✅ ENHANCED PREDICTIONS SUMMARY")
    print("=" * 60)
    print(f"📅 Date: {readable['date'].iloc[0] if 'date' in readable.columns else 'Today'}")
    print(f"🎮 Total Games: {len(readable)}")
    
    if "strength" in readable.columns:
        strong_count = len(readable[readable["strength"] == "STRONG"])
        moderate_count = len(readable[readable["strength"] == "MODERATE"])
        print(f"🔥 Strong Plays: {strong_count}")
        print(f"📈 Moderate Plays: {moderate_count}")
    
    print(f"\n🏆 TOP PREDICTIONS:")
    display_cols = ["home_team", "away_team", "k_close", "y_pred", "edge", "rec", "strength"]
    available_display = [c for c in display_cols if c in readable.columns]
    print(readable[available_display].head(10).to_string(index=False))
    
    if "temp_f" in readable.columns:
        avg_temp = readable["temp_f"].mean()
        print(f"\n🌤️  Average Game Temperature: {avg_temp:.1f}°F")
    
    if "home_pitcher_era_season" in readable.columns:
        avg_home_era = readable["home_pitcher_era_season"].mean()
        avg_away_era = readable["away_pitcher_era_season"].mean()
        print(f"⚾ Average Pitcher ERAs: Home {avg_home_era:.2f} | Away {avg_away_era:.2f}")
'@ | & $PY -

Write-Host "`n🎯 Enhanced gameday pipeline complete!" -ForegroundColor Green
Write-Host "📁 Output files:" -ForegroundColor Cyan
Write-Host "   • predictions_today.parquet (ML predictions)" -ForegroundColor Gray
Write-Host "   • strong_plays.csv (high-confidence bets)" -ForegroundColor Gray  
Write-Host "   • readable_predictions.csv (human-friendly format)" -ForegroundColor Gray

if (Test-Path "predictions_today.parquet") {
    $predCount = & $PY -c "import pandas as pd; df = pd.read_parquet('predictions_today.parquet'); print(len(df))"
    Write-Host "✅ Generated predictions for $predCount games" -ForegroundColor Green
    
    # Generate UI-compatible output
    Write-Host "`n📱 Generating UI-compatible output..." -ForegroundColor Cyan
    $uiPath = "..\mlb-predictions-ui\public\daily_recommendations.json"
    
    @'
import pandas as pd
import json
from datetime import datetime

try:
    df = pd.read_parquet("predictions_today.parquet")
    
    ui_data = {
        "generated_at": datetime.now().isoformat(),
        "model_version": "enhanced_gameday_v2.0",
        "date": datetime.now().strftime('%Y-%m-%d'),
        "weather_enabled": True,
        "enhanced_features": True,
        "games": [],
        "best_bets": []
    }
    
    for _, row in df.iterrows():
        game_data = {
            "game_id": str(row.get("game_id", "")),
            "matchup": f"{row.get('away_team', 'Away')} @ {row.get('home_team', 'Home')}",
            "away_team": row.get("away_team", "Away"),
            "home_team": row.get("home_team", "Home"),
            "ai_prediction": float(row.get("y_pred", 8.5)),
            "market_total": float(row.get("k_close", 8.5)),
            "difference": float(row.get("edge", 0.0)),
            "recommendation": "OVER" if row.get("edge", 0) > 0.5 else "UNDER" if row.get("edge", 0) < -0.5 else "NO_BET",
            "confidence": "HIGH" if abs(row.get("edge", 0)) >= 1.0 else "MEDIUM" if abs(row.get("edge", 0)) >= 0.5 else "LOW",
            "enhanced_model": True,
            "weather_data": {
                "temperature": float(row.get("temp_f", 75)),
                "wind_speed": float(row.get("wind_mph", 5))
            }
        }
        ui_data["games"].append(game_data)
    
    # Sort by edge for best bets
    strong_bets = [g for g in ui_data["games"] if g["confidence"] in ["HIGH", "MEDIUM"]]
    strong_bets.sort(key=lambda x: abs(x["difference"]), reverse=True)
    ui_data["best_bets"] = strong_bets[:5]
    
    with open("../mlb-predictions-ui/public/daily_recommendations.json", "w") as f:
        json.dump(ui_data, f, indent=2)
    
    print(f"✅ Generated UI data: {len(ui_data['games'])} games, {len(ui_data['best_bets'])} best bets")
    
except Exception as e:
    print(f"⚠️  Could not generate UI data: {e}")
'@ | & $PY -
    
} else {
    Write-Host "❌ No predictions generated" -ForegroundColor Red
}

