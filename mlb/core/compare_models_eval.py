#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# Reuse the contract logic from the workflow
from mlb.core.daily_api_workflow import enforce_feature_contract


def find_latest_whitelist_model_dir() -> Path | None:
    override = os.getenv('WHITELIST_MODEL_DIR')
    if override:
        p = Path(override)
        if p.is_file() and p.name.endswith('model.joblib'):
            p = p.parent
        if p.exists() and (p / 'model.joblib').exists():
            return p
    # Search common roots
    here = Path(__file__).parent
    repo = here.parent.parent
    candidates = []
    for base in (repo / 'models', here.parent / 'models', Path.cwd() / 'models'):
        if base.exists():
            for d in base.glob('whitelist_*'):
                if d.is_dir() and (d / 'model.joblib').exists():
                    candidates.append(d)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"n": 0}
    y = y_true[mask]
    p = y_pred[mask]
    mae = float(np.mean(np.abs(p - y)))
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    bias = float(np.mean(p - y))
    return {"n": int(mask.sum()), "MAE": mae, "RMSE": rmse, "Bias": bias}


def diracc_vs_market(y_true: np.ndarray, y_pred: np.ndarray, market: np.ndarray) -> float | None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    market = np.asarray(market, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(market)
    if mask.sum() == 0:
        return None
    y = y_true[mask]
    p = y_pred[mask]
    m = market[mask]
    # Correct if prediction is on the same side of market as actual
    s = np.sign(y - m) == np.sign(p - m)
    return float(np.mean(s))


def main():
    ap = argparse.ArgumentParser(description="Compare whitelist model vs prior models and market over a window")
    ap.add_argument('--start-date', help='YYYY-MM-DD', default=(datetime.utcnow() - timedelta(days=21)).strftime('%Y-%m-%d'))
    ap.add_argument('--end-date', help='YYYY-MM-DD', default=datetime.utcnow().strftime('%Y-%m-%d'))
    ap.add_argument('--outdir', default='outputs', help='Directory to write outputs')
    ap.add_argument('--clip-low', type=float, default=3.0)
    ap.add_argument('--clip-high', type=float, default=18.0)
    args = ap.parse_args()

    engine = create_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'))

    # Load whitelist bundle
    model_dir = find_latest_whitelist_model_dir()
    if not model_dir:
        print('ERROR: No whitelist model directory found')
        raise SystemExit(2)
    import joblib
    bundle = joblib.load(model_dir / 'model.joblib')
    model = bundle.get('model')
    features = bundle.get('features') or []
    if not model or not features:
        print('ERROR: Whitelist bundle missing model or features')
        raise SystemExit(3)

    q = text('''
        SELECT *
          FROM enhanced_games
         WHERE "date" BETWEEN :s AND :e
           AND total_runs IS NOT NULL
         ORDER BY "date", game_id
    ''')
    df = pd.read_sql(q, engine, params={'s': args.start_date, 'e': args.end_date})
    if df.empty:
        print('No completed games found for window')
        return

    # Build X for whitelist model
    X = enforce_feature_contract(df.copy(), features)
    yhat_wl = np.clip(model.predict(X), args.clip_low, args.clip_high)

    # Baselines if present
    y_true = pd.to_numeric(df['total_runs'], errors='coerce').to_numpy()
    market = pd.to_numeric(df.get('market_total'), errors='coerce').to_numpy()

    # Previous learning model columns (naming from workflow)
    prev = pd.to_numeric(df.get('predicted_total'), errors='coerce').to_numpy() if 'predicted_total' in df.columns else None
    ultra = pd.to_numeric(df.get('predicted_total_learning'), errors='coerce').to_numpy() if 'predicted_total_learning' in df.columns else None

    # Assemble per-row output
    out = df[['game_id', 'date']].copy()
    out['y_true'] = y_true
    out['market_total'] = market
    out['yhat_whitelist'] = yhat_wl
    if prev is not None:
        out['yhat_prev'] = prev
    if ultra is not None:
        out['yhat_ultra80'] = ultra

    # Compute metrics
    results = {
        'window': {'start': args.start_date, 'end': args.end_date},
        'counts': {'rows': int(len(out))}
    }
    results['whitelist'] = metrics(y_true, yhat_wl)
    results['whitelist']['DirAcc_vs_market'] = diracc_vs_market(y_true, yhat_wl, market)

    results['market'] = metrics(y_true, market)

    if prev is not None:
        results['previous_model'] = metrics(y_true, prev)
        results['previous_model']['DirAcc_vs_market'] = diracc_vs_market(y_true, prev, market)
    if ultra is not None:
        results['ultra80'] = metrics(y_true, ultra)
        results['ultra80']['DirAcc_vs_market'] = diracc_vs_market(y_true, ultra, market)

    # Write outputs
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    csv_path = outdir / f'comparison_per_game_{ts}.csv'
    json_path = outdir / f'comparison_metrics_{ts}.json'
    out.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(results, indent=2))

    # Print concise summary
    def fmt(d):
        if not d or d.get('n', 0) == 0:
            return 'n=0'
        return f"n={d['n']} MAE={d['MAE']:.3f} RMSE={d['RMSE']:.3f} Bias={d['Bias']:.3f}"

    print('Window:', args.start_date, '→', args.end_date)
    print('Whitelist   :', fmt(results.get('whitelist')), 'DirAcc=', results['whitelist'].get('DirAcc_vs_market'))
    if 'previous_model' in results:
        print('Previous    :', fmt(results.get('previous_model')), 'DirAcc=', results['previous_model'].get('DirAcc_vs_market'))
    if 'ultra80' in results:
        print('Ultra80     :', fmt(results.get('ultra80')), 'DirAcc=', results['ultra80'].get('DirAcc_vs_market'))
    print('Market      :', fmt(results.get('market')))
    print('Wrote:', csv_path)
    print('Metrics:', json_path)


if __name__ == '__main__':
    main()
