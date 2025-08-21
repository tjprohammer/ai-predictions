# MLB Betting Prediction System

## 🎯 Overview

A comprehensive machine learning system for MLB over/under betting predictions with continuous learning capabilities, real-time tracking, and performance validation.

## 📁 Project Structure

```
mlb-overs/                          # Main prediction system
├── api/                            # FastAPI backend
│   ├── app.py                     # Main API with learning endpoints
│   └── endpoints/                 # API route modules
├── deployment/                     # Production workflow
│   ├── daily_api_workflow.py     # Complete daily pipeline
│   ├── DAILY_RUNBOOK.md          # Operations guide
│   └── ingestion/                 # Data collectors
├── feature_engineering/           # ML feature pipelines
├── models/                        # Trained ML models
├── data/                          # Data storage
├── prediction_tracking/           # Performance validation
│   ├── prediction_performance_tracker.py     # Main tracking system
│   └── *.py                      # Various tracking utilities
├── training_systems/             # ML training & learning
│   ├── continuous_learning_system.py # Daily model retraining
│   ├── daily_learning_pipeline.py    # Learning workflow
│   └── *.py                      # Training utilities
├── model_analysis/               # Model evaluation
├── system_validation/            # Testing & validation
├── data_analysis/                # Data exploration
└── debugging/                    # Debug utilities

mlb-predictions-ui/                 # React frontend
├── src/components/                # UI components
│   ├── ComprehensivePredictionsBoard.tsx  # Main predictions with learning integration
│   └── ModelPerformanceDashboard.tsx     # Performance analysis
└── public/                        # Static assets
```

## 🚀 Daily Workflow

### Complete Pipeline (Recommended)
```powershell
# Run full enhanced pipeline
cd mlb-overs/deployment
python daily_api_workflow.py --stages markets,features,predict,odds,health,prob,export,audit

# For specific date
python daily_api_workflow.py --date 2025-08-21 --stages markets,features,predict,odds,health,prob,export,audit
```

### Learning System Integration
```powershell
# Run continuous learning workflow
cd training_systems
python continuous_learning_system.py --date 2025-08-21

# Manual learning pipeline
python daily_learning_pipeline.py
```

### Prediction Tracking
```powershell
# Track predictions vs actual results
cd prediction_tracking
python prediction_performance_tracker.py
```

## 🎯 Key Features

### 1. **Continuous Learning System**
- Daily model retraining with recent performance data
- Learning vs current model comparison (8 vs 4 betting opportunities)
- Performance improvement tracking (3.08 MAE)

### 2. **Enhanced Predictions**
- Comprehensive game analysis with 50+ features
- Market integration with odds data
- Risk assessment and confidence scoring
- Learning model predictions integrated

### 3. **Performance Tracking**
- Real-time prediction accuracy monitoring
- Model comparison analytics
- Edge realization tracking
- Historical performance validation

### 4. **Advanced UI**
- React-based dashboard with multiple prediction views
- Learning predictions integrated into comprehensive tab
- Performance analysis with model comparisons
- Real-time data updates

## 📊 API Endpoints

### Core Predictions
- `GET /api/comprehensive-games/{date}` - Complete game analysis
- `GET /api/learning-predictions/{date}` - Learning vs current comparison
- `GET /api/latest-predictions` - Most recent predictions

### Performance Tracking  
- `GET /api/prediction-performance/summary` - Performance metrics
- `GET /api/prediction-performance/recent` - Recent predictions with results

### System Health
- `GET /api/health` - System status
- `GET /api/model-status` - Model performance metrics

## 🔧 Configuration

### Database
```sql
-- Main prediction data
enhanced_games              -- Base game data with predictions
probability_predictions     -- Enhanced probability calculations
prediction_tracking        -- Performance tracking

-- Learning system
learning_model_performance  -- Learning model metrics
model_corrections          -- Bias corrections
```

### Environment Variables
```bash
DATABASE_URL=postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb
MODEL_BUNDLE_PATH=../models/legitimate_model_latest.joblib
```

## 🎮 Usage Examples

### Start the System
```powershell
# 1. Start API server
cd mlb-overs
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000

# 2. Start React UI  
cd mlb-predictions-ui
npm start

# 3. Run daily workflow
cd mlb-overs/deployment
python daily_api_workflow.py --stages markets,features,predict,odds,health,prob,export,audit
```

### View Predictions
1. **Comprehensive View**: Learning + current predictions side-by-side
2. **Performance Analysis**: Model accuracy and learning progress  
3. **Dashboard**: Real-time betting recommendations

## 📈 Performance Monitoring

### Learning Model Results
- **8 learning opportunities** vs 4 current model bets
- **3.08 MAE** performance on recent data
- **Consensus picks**: 3 games where both models agree
- **High confidence**: 5 games with strong edges

### Prediction Tracking
```powershell
# View recent performance
python prediction_tracking/prediction_performance_tracker.py

# Example output:
# Current Model: 67.3% accuracy (37/55 calls)
# Learning Model: 74.1% accuracy (43/58 calls) 
# Learning vs Current: +6.8% accuracy improvement
```

## 🛠 Development

### Adding New Features
1. Update feature engineering in `feature_engineering/`
2. Retrain models with `training_systems/`
3. Test with `system_validation/`
4. Deploy via `daily_api_workflow.py`

### Testing
```powershell
# Run system validation
cd system_validation
python test_enhanced_api.py
python test_learning_loop.py

# Validate predictions
cd prediction_tracking  
python prediction_performance_tracker.py
```

## 📋 Workflow Stages Explained

- **🏪 markets**: Pull market data and odds
- **🔧 features**: Build ML features (50+ variables)
- **🤖 predict**: Generate base ML predictions  
- **📊 odds**: Load comprehensive odds data
- **🛡️ health**: Validate system calibration (Brier < 0.25)
- **🎯 prob**: Calculate probability predictions with Kelly sizing
- **📁 export**: Export results to files
- **🔍 audit**: Validate data quality

## 🎯 Success Metrics

- **Prediction Accuracy**: Learning model shows 6.8% improvement
- **Betting Opportunities**: 8 vs 4 (100% increase)  
- **Edge Realization**: Tracked in real-time
- **System Uptime**: 99%+ with health monitoring
- **Data Quality**: Comprehensive validation at each stage

This system provides a complete MLB betting prediction platform with continuous learning, performance tracking, and comprehensive analysis capabilities.

Modeling

Time‑series split CV

XGBoost/LGBM regressor → isotonic calibration for P(Over)

Optional residual‑to‑market model for further calibration

# run a single date

make gameday DATE=2025-08-10




-------------------------------------------------------
Database + schema

Added markets_totals table (with snapshot and close rows) and the unique arbiter for snapshots.

Confirmed indexes and FK to games.

Extended bullpens_daily table to include form & availability fields (ERA/FIP/K-BB%, HR/9, closer/setup pitches D-1, back-to-back flag).

Ingestors

Games (ingestors.games) — daily schedule with home_sp_id/away_sp_id.

Pitcher starts (ingestors.pitchers_last10) — last-10 per pitcher with CSW%, velo, xwOBA allowed, etc.

Bullpen daily (ingestors.bullpens_daily)

Scrapes MLB StatsAPI box scores.

Aggregates bullpen performance by team/day.

Tracks availability (closer/setup pitches D-1; back-to-back).

ESPN totals (ingestors.espn_totals)

Scoreboard fetch with robust team name normalization.

Inserts snapshot rows; supports close rows.

Fallback to event summary endpoint so we capture totals when the scoreboard is missing them.

Features

features/build_features.py now builds a single row per game with:

Market line: latest snapshot (else close) as k_close.

Team offense (today): xwOBA/ISO/BB%/K% for home and away, keyed via nickname normalization.

Bullpen form (today): ERA/FIP/K-BB%, HR/9.

Bullpen availability: closer/setup D-1 pitches; back-to-back flag.

Recent bullpen usage: 3-day window flag via helper (home_bp_use_3d, home_bp_fatigued, etc.).

Starter rolling form: last 3/5/10 CSW%, velo, xwOBA allowed, xSLG allowed for both starters.

Inference

models/infer.py produces a baseline prediction:

Uses k_close as anchor.

Adjusts with team offense form, starter form, bullpen FIP and availability.

Outputs y_pred, edge = y_pred - k_close, and calibrated over/under probabilities.

Fixed PowerShell merge issues; created a readable predictions table (CSV + console view).

Debugging / Dev hygiene

Resolved module path issues for ingestors in PowerShell.

Removed brittle SQL and column name assumptions.

Added sanity checks / debug prints for columns.

Confirmed we’re no longer defaulting k_close to 8.5 (that was flattening all results).

What’s next (priority-ordered)
Strengthen market data (lines)

Add one more free source (e.g., Covers, Rotowire page scrape) to cross-check totals.

Store multi-book snapshots (book column already in place).

Compute open → close deltas per game and include as features.

Pitcher vs Team module

New ingestor: roll the last 2 seasons of pitcher vs opponent from pitchers_starts.

Features: vs-opponent xwOBA allowed, K%, BB%, HR/9, and PA sample size.

Merge on home_sp_id/away_sp_id + opponent team.

Offense splits vs L/R

If not in teams_offense_daily, add an ingestor for vs RHP/LHP xwOBA/ISO.

Use probable starter hand to pick the right split for each lineup side.

Optional: incorporate projected lineup handedness mix if you scrape lineup cards later.

Park & Weather

Create/curate parks table (run factor, HR factor, altitude, roof type).

Add weather_game (temp, wind mph/dir, humidity) via a free weather page scrape by game time.

Add features: park run factor, expected temp & wind (direction-aware for stadium, if feasible).

Backfill & train a real model

Backfill a few months of games with totals + final scores.

Build a training set with all the features above and target = total_runs (or classification >/< line).

Train quick regressors (LightGBM/XGBoost/ElasticNet) and compare to the heuristic.

Add cross-validation, feature importance, and calibration.

Quality checks & reporting

Daily validation script: % of games missing totals, starter IDs, offense splits.

Simple monitoring: log prediction drift vs market; keep hit rate ledger by threshold (e.g., edges > 0.5 runs).

Export a compact “slate report” (CSV/HTML) with matchup, line, prediction, edge, confidence, key drivers.

Typical daily run order (now)
powershell
Copy
Edit
# lines + core data
python -m ingestors.games --start YYYY-MM-DD --end YYYY-MM-DD
python -m ingestors.pitchers_last10 --start YYYY-MM-DD --end YYYY-MM-DD
python -m ingestors.bullpens_daily --start YYYY-MM-DD --end YYYY-MM-DD
python -m ingestors.espn_totals --date YYYY-MM-DD

# features + predictions
python features\build_features.py --database-url $env:DATABASE_URL --out features\train.parquet
python models\infer.py --database-url $env:DATABASE_URL --out predictions_today.parquet

python predict_from_range.py --start 2025-08-15 --end 2025-08-16 --thr 1.5

$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"; python daily_api_workflow.py --date 2025-08-16 --stages features,predict,probabilities

# readable slate
@'
import os, pandas as pd
from sqlalchemy import create_engine
eng = create_engine(os.getenv("DATABASE_URL","postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb"))
pred = pd.read_parquet("predictions_today.parquet")
games = pd.read_sql("""
  SELECT game_id, date, away_team||' @ '||home_team AS matchup
  FROM games WHERE date=(SELECT MAX(date) FROM games)
""", eng)
df = pred.merge(games, on="game_id", how="left")
print(df[["game_id","date","matchup","k_close","y_pred","edge","p_over_cal","p_under_cal"]]
      .sort_values("edge", ascending=False).to_string(index=False))
df.to_csv("predictions_readable.csv", index=False)
'@ | python -
Where you’ll see immediate improvement
After the ESPN fallback and removing the 8.5 default, you should see varied k_close and non-flat predictions.

Adding starter vs team and L/R offense splits typically moves edges by ~0.2–0.5 runs in some matchups.

Park/weather can shift totals materially (Wrigley wind days, Coors, domes).

If you want, I can drop in:

the pitcher_vs_team ingestor + feature merge,

the L/R splits ingestor (if you don’t already store them),

and a tiny LightGBM trainer to replace the heuristic once we’ve backfilled a few weeks.

What do you want to tackle first: pitcher-vs-team or offense L/R splits?