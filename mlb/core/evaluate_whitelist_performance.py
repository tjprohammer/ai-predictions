#!/usr/bin/env python3
"""
Evaluate Whitelist Feature Set Performance
=========================================
Calculates basic predictive performance metrics (MAE, RMSE, bias, directional
accuracy vs market) for recent completed games using only the curated
whitelist features to simulate a lightweight baseline model.

This does NOT retrain complex pipelines; instead it fits a simple Lasso
(regression) and a Gradient Boosting model for comparison, using past N days
and evaluating on the most recent M days (temporal split).

Outputs a JSON summary and a CSV of per-game predictions under core/outputs/.
"""
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
ENGINE = create_engine(DATABASE_URL)
OUTDIR = Path(__file__).parent / 'outputs'
OUTDIR.mkdir(exist_ok=True, parents=True)

WL_PATH = Path(__file__).parent / 'learning_features_v1.json'
if not WL_PATH.exists():
    raise SystemExit("Whitelist JSON missing: learning_features_v1.json")
WL = json.loads(WL_PATH.read_text()).get('features', [])

LOOKBACK_DAYS = int(os.getenv('WL_EVAL_LOOKBACK_DAYS','45'))
TEST_DAYS = int(os.getenv('WL_EVAL_TEST_DAYS','7'))
MIN_ROWS = 150
TODAY = datetime.utcnow().date()
START_DATE = TODAY - timedelta(days=LOOKBACK_DAYS)
CUTOFF_DATE = TODAY - timedelta(days=TEST_DAYS)

print(f"Evaluating whitelist performance: train window start={START_DATE}, test cutoff={CUTOFF_DATE}, today={TODAY}")

query = text("""
        SELECT *
            FROM enhanced_games
         WHERE date >= :start
             AND total_runs IS NOT NULL
             AND total_runs BETWEEN 0 AND 30
""")

df = pd.read_sql(query, ENGINE, params={'start': START_DATE})
if 'date' not in df.columns:
    raise SystemExit("No date column returned from enhanced_games")

# Normalize date column to datetime.date
df['date'] = pd.to_datetime(df['date']).dt.date

# Apply known column aliases (mirrors daily_api_workflow mappings) BEFORE whitelist filtering
ALIAS_MAP = {
    'home_sp_season_era': 'home_sp_era',
    'away_sp_season_era': 'away_sp_era',
    'home_sp_season_k': 'home_sp_k_per_9',  # if per game counts approximated
    'away_sp_season_k': 'away_sp_k_per_9',
    'home_sp_season_bb': 'home_sp_bb_per_9',
    'away_sp_season_bb': 'away_sp_bb_per_9',
    'home_bp_season_era': 'home_bp_era',
    'away_bp_season_era': 'away_bp_era',
    'home_bp_season_fip': 'home_bp_fip',
    'away_bp_season_fip': 'away_bp_fip',
    'home_team_runs_per_game_season': 'home_team_rpg_season',
    'away_team_runs_per_game_season': 'away_team_rpg_season',
    'home_team_runs_per_game_l30': 'home_team_rpg_l30',
    'away_team_runs_per_game_l30': 'away_team_rpg_l30',
    'home_team_recent_runs': 'home_recent_runs_per_game',
    'away_team_recent_runs': 'away_recent_runs_per_game',
    'home_team_iso': 'home_team_iso_season',
    'away_team_iso': 'away_team_iso_season',
    'home_team_wrc_plus': 'home_team_wrc_plus_season',
    'away_team_wrc_plus': 'away_team_wrc_plus_season'
}

for src, dst in ALIAS_MAP.items():
    if src in df.columns and dst not in df.columns:
        df[dst] = df[src]
if df.empty or len(df) < MIN_ROWS:
    raise SystemExit("Not enough historical completed games for evaluation")

# Restrict to whitelist + target columns
available_features = [c for c in WL if c in df.columns]
missing = [c for c in WL if c not in df.columns]
if missing:
    print(f"WARNING: Missing {len(missing)} whitelist features (ignored): {missing[:10]}{'...' if len(missing)>10 else ''}")

MIN_FEATURES = int(os.getenv('WL_MIN_AVAILABLE_FEATURES','8'))
if len(available_features) < MIN_FEATURES:
    print(f"ABORT: Only {len(available_features)} whitelist features present (< {MIN_FEATURES} threshold).")
    print("Present features:", available_features)
    raise SystemExit(5)

print(f"Using {len(available_features)} whitelist features (>= {MIN_FEATURES} threshold).")

feature_df = df[available_features].copy()
feature_df = feature_df.apply(pd.to_numeric, errors='coerce')
feature_df = feature_df.fillna(feature_df.median(numeric_only=True)).fillna(0)
feature_df = feature_df.replace([np.inf,-np.inf], 0)

# Temporal split
train_mask = df['date'] < CUTOFF_DATE
train_X = feature_df[train_mask]
train_y = df.loc[train_mask, 'total_runs']

test_mask = df['date'] >= CUTOFF_DATE
test_X = feature_df[test_mask]
test_y = df.loc[test_mask, 'total_runs']

if test_X.empty or train_X.empty:
    raise SystemExit("Insufficient temporal split data")

models = {}
results_rows = []

# Lasso
try:
    lasso = LassoCV(cv=5, random_state=42, n_alphas=50, max_iter=5000)
    lasso.fit(train_X, train_y)
    lasso_pred = np.clip(lasso.predict(test_X), 3, 18)
    try:
        rmse_val = mean_squared_error(test_y, lasso_pred, squared=False)
    except TypeError:
        rmse_val = mean_squared_error(test_y, lasso_pred) ** 0.5
    models['lasso'] = {
        'MAE': float(mean_absolute_error(test_y, lasso_pred)),
        'RMSE': float(rmse_val),
        'Bias': float((lasso_pred - test_y).mean()),
    }
except Exception as e:
    print(f"Lasso failed: {e}")
    lasso_pred = None

# Gradient Boosting
try:
    gbr = GradientBoostingRegressor(random_state=42, n_estimators=300, max_depth=3, learning_rate=0.05)
    gbr.fit(train_X, train_y)
    gbr_pred = np.clip(gbr.predict(test_X), 3, 18)
    try:
        rmse_val = mean_squared_error(test_y, gbr_pred, squared=False)
    except TypeError:
        rmse_val = mean_squared_error(test_y, gbr_pred) ** 0.5
    models['gbr'] = {
        'MAE': float(mean_absolute_error(test_y, gbr_pred)),
        'RMSE': float(rmse_val),
        'Bias': float((gbr_pred - test_y).mean()),
    }
except Exception as e:
    print(f"GBR failed: {e}")
    gbr_pred = None

# Market directional accuracy & simple anchor baseline
market = df.loc[test_mask, 'market_total']
market_mae = float(mean_absolute_error(test_y, market)) if market.notna().any() else None
if market.notna().any():
    try:
        market_rmse = float(mean_squared_error(test_y, market, squared=False))
    except TypeError:
        market_rmse = float(mean_squared_error(test_y, market) ** 0.5)
else:
    market_rmse = None

# Directional accuracy (over/under actual vs. prediction vs market) using mid 0.15 tolerance
if market.notna().any():
    def dir_acc(pred):
        side_pred = np.sign(pred - market)
        side_actual = np.sign(test_y - market)
        # Treat very small edges as hold (0)
        side_pred[np.abs(pred - market) < 0.15] = 0
        side_actual[np.abs(test_y - market) < 0.15] = 0
        return float((side_pred == side_actual).mean())
    if lasso_pred is not None:
        models['lasso']['DirectionalAcc'] = dir_acc(lasso_pred)
    if gbr_pred is not None:
        models['gbr']['DirectionalAcc'] = dir_acc(gbr_pred)

summary = {
    'window': {
        'train_start': START_DATE.isoformat(),
        'cutoff': CUTOFF_DATE.isoformat(),
        'today': TODAY.isoformat()
    },
    'n_train': int(train_X.shape[0]),
    'n_test': int(test_X.shape[0]),
    'features_used': available_features,
    'missing_features': missing,
    'models': models,
    'market_baseline': {'MAE': market_mae, 'RMSE': market_rmse},
}

(out_json := OUTDIR / 'whitelist_eval_summary.json').write_text(json.dumps(summary, indent=2))
print(f"Wrote summary → {out_json}")

# Per-game output for inspection
if lasso_pred is not None or gbr_pred is not None:
    out_games = df.loc[test_mask, ['date','game_id','home_team','away_team','total_runs','market_total']].copy()
    if lasso_pred is not None:
        out_games['lasso_pred'] = lasso_pred
    if gbr_pred is not None:
        out_games['gbr_pred'] = gbr_pred
    (out_csv := OUTDIR / 'whitelist_eval_games.csv').write_text(out_games.to_csv(index=False))
    print(f"Wrote per-game predictions → {out_csv}")

print(json.dumps(summary, indent=2))
