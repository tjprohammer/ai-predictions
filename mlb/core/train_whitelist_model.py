#!/usr/bin/env python3
"""
Train Whitelist Model
=====================
Trains simple baseline models (LassoCV and GradientBoostingRegressor) using only
whitelist features from enhanced_games with a temporal split. Persists the best
model, feature list, and training metadata for serving.

Usage:
  python train_whitelist_model.py --lookback-days 120 --test-days 14 --outdir models/whitelist
"""
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import hashlib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# NOTE: Enhanced to allow dynamic whitelist sources via --whitelist-json / --whitelist-csv
# If neither provided, falls back to legacy learning_features_v1.json (if present)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
ENGINE = create_engine(DATABASE_URL)
BASE_DIR = Path(__file__).parent
WL_PATH = BASE_DIR / 'learning_features_v1.json'


def feature_hash(features: list) -> str:
    s = "|".join(sorted(features))
    return hashlib.sha1(s.encode('utf-8')).hexdigest()[:12]


def load_data(lookback_days: int):
    today = datetime.utcnow().date()
    start = today - timedelta(days=lookback_days)
    q = text("""
        SELECT * FROM enhanced_games
        WHERE date >= :start AND total_runs IS NOT NULL AND total_runs BETWEEN 0 AND 30
    """)
    df = pd.read_sql(q, ENGINE, params={"start": start})
    if 'date' not in df.columns:
        raise SystemExit("enhanced_games missing 'date'")
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df, today


def prepare_features(df: pd.DataFrame, whitelist: list):
    available = [c for c in whitelist if c in df.columns]
    X = df[available].copy()
    X = X.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    y = df['total_runs']
    return X, y, available


def temporal_split(df, X, y, test_days: int):
    cutoff = df['date'].max() - timedelta(days=test_days)
    train_mask = df['date'] < cutoff
    test_mask = df['date'] >= cutoff
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask], cutoff


def train_and_select(train_X, train_y, test_X, test_y):
    results = {}
    artifacts = {}

    # Lasso
    try:
        lasso = LassoCV(cv=5, random_state=42, n_alphas=60, max_iter=6000)
        lasso.fit(train_X, train_y)
        l_pred = np.clip(lasso.predict(test_X), 3, 18)
        try:
            l_rmse = mean_squared_error(test_y, l_pred, squared=False)
        except TypeError:
            l_rmse = mean_squared_error(test_y, l_pred) ** 0.5
        results['lasso'] = {
            'MAE': float(mean_absolute_error(test_y, l_pred)),
            'RMSE': float(l_rmse),
            'Bias': float((l_pred - test_y).mean())
        }
        artifacts['lasso'] = {'model': lasso, 'pred': l_pred}
    except Exception as e:
        print(f"Lasso failed: {e}")

    # Gradient Boosting
    try:
        gbr = GradientBoostingRegressor(random_state=42, n_estimators=350, max_depth=3, learning_rate=0.05)
        gbr.fit(train_X, train_y)
        g_pred = np.clip(gbr.predict(test_X), 3, 18)
        try:
            g_rmse = mean_squared_error(test_y, g_pred, squared=False)
        except TypeError:
            g_rmse = mean_squared_error(test_y, g_pred) ** 0.5
        results['gbr'] = {
            'MAE': float(mean_absolute_error(test_y, g_pred)),
            'RMSE': float(g_rmse),
            'Bias': float((g_pred - test_y).mean())
        }
        artifacts['gbr'] = {'model': gbr, 'pred': g_pred}
    except Exception as e:
        print(f"GBR failed: {e}")

    if not results:
        raise SystemExit("No models trained")

    # Select by MAE (then RMSE)
    best_name = sorted(results.items(), key=lambda kv: (kv[1]['MAE'], kv[1]['RMSE']))[0][0]
    return best_name, artifacts[best_name]['model'], results


def load_whitelist(args) -> tuple[list,str]:
    """Load whitelist features from provided CLI arguments.

    Precedence:
      1. --whitelist-json (expects {'features': [...]})
      2. --whitelist-csv (expects a column named 'feature' or single-column list)
      3. legacy learning_features_v1.json if present
    Returns (features, source_label)
    """
    # JSON path
    if args.whitelist_json:
        p = Path(args.whitelist_json)
        if not p.exists():
            raise SystemExit(f"Whitelist JSON not found: {p}")
        try:
            data = json.loads(p.read_text())
            feats = data.get('features') or data.get('whitelist') or []
        except Exception as e:
            raise SystemExit(f"Failed parsing whitelist json {p}: {e}")
        if not feats:
            raise SystemExit(f"No 'features' array in {p}")
        return [f for f in feats if isinstance(f,str)], f"json:{p.name}"
    # CSV path
    if args.whitelist_csv:
        p = Path(args.whitelist_csv)
        if not p.exists():
            raise SystemExit(f"Whitelist CSV not found: {p}")
        try:
            df = pd.read_csv(p)
        except Exception as e:
            raise SystemExit(f"Failed reading whitelist csv {p}: {e}")
        col = None
        for candidate in ['feature','features','name','col','column']:
            if candidate in df.columns:
                col = candidate; break
        if col is None:
            # Single column unnamed? then first column
            if len(df.columns) == 1:
                col = df.columns[0]
            else:
                raise SystemExit(f"Could not infer feature column in {p} (looked for 'feature')")
        feats = [f for f in df[col].tolist() if isinstance(f,str)]
        return feats, f"csv:{p.name}"
    # Legacy fallback
    if WL_PATH.exists():
        try:
            data = json.loads(WL_PATH.read_text())
            feats = data.get('features', [])
            if feats:
                return [f for f in feats if isinstance(f,str)], f"json:{WL_PATH.name}"
        except Exception as e:
            print(f"Failed loading legacy whitelist: {e}")
    raise SystemExit("No whitelist source provided (use --whitelist-json or --whitelist-csv)")


def main():
    parser = argparse.ArgumentParser(description='Train whitelist baseline model (dynamic feature list)')
    parser.add_argument('--lookback-days', type=int, default=int(os.getenv('WL_TRAIN_LOOKBACK_DAYS','120')))
    parser.add_argument('--test-days', type=int, default=int(os.getenv('WL_TRAIN_TEST_DAYS','14')))
    parser.add_argument('--outdir', default=str(BASE_DIR.parent / 'models'))
    parser.add_argument('--whitelist-json', help='Path to whitelist JSON containing {"features": [...]}')
    parser.add_argument('--whitelist-csv', help='Path to whitelist CSV with a column named feature')
    parser.add_argument('--tag', help='Optional tag to include in output directory name')
    parser.add_argument('--allow-suspect', action='store_true', help='Allow saving model even if leakage heuristic flags it')
    args = parser.parse_args()

    wl, wl_source = load_whitelist(args)
    if not wl:
        raise SystemExit('Empty whitelist after loading')
    print(f"Loaded whitelist: {len(wl)} features from {wl_source}")

    df, today = load_data(args.lookback_days)
    if len(df) < 200:
        raise SystemExit('Insufficient rows for training')

    X, y, used = prepare_features(df, wl)
    train_X, test_X, train_y, test_y, cutoff = temporal_split(df, X, y, args.test_days)

    best_name, model, results = train_and_select(train_X, train_y, test_X, test_y)

    # Simple leakage heuristic: extremely low MAE plus presence of suspicious columns
    suspicious_tokens = ['score', 'current_', 'live_', 'inning']
    if results[best_name]['MAE'] < 1.2:
        suspect_cols = [c for c in used if any(tok in c.lower() for tok in suspicious_tokens)]
        if suspect_cols and not args.allow_suspect:
            print("WARNING: Suspect low MAE with potential leakage columns detected:")
            print(f"  MAE={results[best_name]['MAE']:.3f} Features={len(used)} SuspiciousCols={suspect_cols[:8]}")
            print("Abort save (use --allow-suspect to override). Remove leakage columns and retry.")
            return

    # Bundle metadata
    bundle = {
        'trained_at': datetime.utcnow().isoformat(),
        'lookback_days': args.lookback_days,
        'test_days': args.test_days,
        'train_rows': int(train_X.shape[0]),
        'test_rows': int(test_X.shape[0]),
        'features': used,
        'feature_sha': feature_hash(used),
        'model_type': best_name,
        'metrics': results[best_name],
        'cutoff_date': cutoff.isoformat(),
        'today': today.isoformat(),
        'whitelist_source': wl_source,
        'whitelist_size': len(used)
    }

    outdir = Path(args.outdir)
    stamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    parts = ["whitelist", stamp, best_name, f"f{len(used)}", bundle['feature_sha']]
    if args.tag:
        parts.insert(1, args.tag)
    model_dir = outdir / ("_".join(parts))
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump({'model': model, 'features': used, 'bundle': bundle}, model_dir / 'model.joblib')
    (model_dir / 'bundle.json').write_text(json.dumps(bundle, indent=2))
    print(f"Saved model → {model_dir}")

    # Write quick evaluation CSV
    eval_df = pd.DataFrame({'date': df.loc[df['date'] >= cutoff, 'date'], 'actual': test_y})
    # recompute predictions with best model
    if best_name == 'lasso':
        pred = np.clip(model.predict(test_X), 3, 18)
    else:
        pred = np.clip(model.predict(test_X), 3, 18)
    eval_df['pred'] = pred
    eval_df.to_csv(model_dir / 'eval_recent.csv', index=False)
    print(json.dumps(bundle, indent=2))

if __name__ == '__main__':
    main()
