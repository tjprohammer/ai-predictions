#!/usr/bin/env python3
"""
Build & store a real training_feature_snapshot in the model bundle.
"""

import os
import json
import joblib
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import create_engine, text
import argparse
import shutil
import tempfile

# --- logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("build_training_snapshot")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb"
)

def safe_align(df, feature_columns, fill_values=None):
    X = df.copy()
    for c in feature_columns:
        if c not in X.columns:
            X[c] = np.nan
    X = X[feature_columns]
    X = X.apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    if isinstance(fill_values, dict) and fill_values:
        for c, v in fill_values.items():
            if c in X.columns:
                X[c] = X[c].fillna(v)
    med = X.median(numeric_only=True)
    if isinstance(med, pd.Series):
        X = X.fillna(med)
    X = X.fillna(0)
    return X

def infer_period(b, default_days=120):
    # Prefer explicit training_period in bundle
    tp = b.get("training_period")
    if isinstance(tp, dict) and "start" in tp and "end" in tp:
        return tp["start"], tp["end"]
    # Else back off from training_date ~N days
    td = b.get("training_date")
    try:
        td = datetime.fromisoformat(str(td).replace("Z", "+00:00"))
    except Exception:
        td = datetime.utcnow()
    start = (td - timedelta(days=default_days)).strftime("%Y-%m-%d")
    end   = td.strftime("%Y-%m-%d")
    return start, end

def main():
    ap = argparse.ArgumentParser(description="Create a real training_feature_snapshot in the bundle.")
    ap.add_argument("--model-path", default="../models/legitimate_model_latest.joblib")
    ap.add_argument("--start-date", help="YYYY-MM-DD; default uses bundle.training_period or training_date-120d")
    ap.add_argument("--end-date", help="YYYY-MM-DD; default uses bundle.training_period or training_date")
    ap.add_argument("--max-rows", type=int, default=5000, help="Rows to store in snapshot (post-sample)")
    ap.add_argument("--fetch-multiplier", type=float, default=3.0, help="Overfetch factor before sampling")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    bundle_path = Path(args.model_path)
    if not bundle_path.exists():
        log.error("Bundle not found: %s", bundle_path)
        raise SystemExit(1)

    log.info("📦 Loading bundle: %s", bundle_path)
    b = joblib.load(bundle_path)

    # Resolve dates
    start_date, end_date = args.start_date, args.end_date
    if not (start_date and end_date):
        start_date, end_date = infer_period(b)
    log.info("Using training window: %s → %s", start_date, end_date)

    # Predictor / pipeline
    try:
        from enhanced_bullpen_predictor import EnhancedBullpenPredictor
    except ImportError as e:
        log.error("Cannot import EnhancedBullpenPredictor: %s", e)
        raise SystemExit(2)
    predictor = EnhancedBullpenPredictor()

    # Fetch historical games with truth (to mimic training rows)
    engine = create_engine(DATABASE_URL)
    target_fetch = int(args.max_rows * args.fetch_multiplier)

    q = text("""
        SELECT
          lgf.*,
          COALESCE(NULLIF(lgf.market_total, 0), eg.market_total) AS market_total_final,
          COALESCE(NULLIF(eg.away_sp_season_era, 0),
                   NULLIF(lgf.away_sp_season_era, 0),
                   NULLIF(lgf.away_sp_season_era, 4.5)) AS away_sp_season_era_final,
          COALESCE(NULLIF(eg.home_sp_season_era, 0),
                   NULLIF(lgf.home_sp_season_era, 0),
                   NULLIF(lgf.home_sp_season_era, 4.5)) AS home_sp_season_era_final
        FROM legitimate_game_features lgf
        JOIN enhanced_games eg
          ON eg.game_id = lgf.game_id AND eg."date" = lgf."date"
        WHERE lgf.total_runs IS NOT NULL
          AND lgf."date" BETWEEN :start_date AND :end_date
        ORDER BY lgf."date" DESC
        LIMIT :limit
    """)

    log.info("Querying historical rows (limit=%d)...", target_fetch)
    raw = pd.read_sql(q, engine, params={"start_date": start_date, "end_date": end_date, "limit": target_fetch})
    engine.dispose()

    if raw.empty:
        log.error("No historical rows returned in the given window.")
        raise SystemExit(3)

    # Coalesce like serving
    raw["market_total"] = raw.pop("market_total_final")
    raw["away_sp_season_era"] = raw.pop("away_sp_season_era_final")
    if "home_sp_season_era_final" in raw.columns:
        raw["home_sp_season_era"] = raw.pop("home_sp_season_era_final")
    raw = raw.loc[:, ~raw.columns.duplicated(keep="last")]

    # Engineer + align to bundle schema
    feats = predictor.engineer_features(raw)
    log.info("Engineered features: %s", feats.shape)
    try:
        X = predictor.align_serving_features(feats, strict=False)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=b["feature_columns"])
    except Exception:
        log.exception("align_serving_features failed; using safe_align fallback")
        X = safe_align(feats, b["feature_columns"], getattr(predictor, "fill_values", {}) or b.get("feature_fill_values", {}))

    # Keep only bundle columns and sample
    X = X[b["feature_columns"]]
    n_before = len(X)
    if n_before > args.max_rows:
        X = X.sample(n=args.max_rows, random_state=args.seed)
    X = X.reset_index(drop=True)

    # Downcast for bundle size and fill NaNs
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    # Fill NaNs before saving to avoid bias in variance/PSI calculations
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    
    float_cols = X.select_dtypes(include=["float64"]).columns
    int_cols   = X.select_dtypes(include=["int64"]).columns
    X[float_cols] = X[float_cols].astype("float32")
    X[int_cols]   = X[int_cols].astype("int32")

    log.info("Snapshot rows: %d (fetched %d)", len(X), n_before)
    
    # Generate robust reference stats for anomaly detection
    ref_stats = {}
    for c in b['feature_columns']:
        if c in X.columns:
            s = pd.to_numeric(X[c], errors='coerce').dropna()
            if len(s) >= 100:
                med = float(s.median())
                mad = float(1.4826*np.median(np.abs(s - med)))  # Robust scale estimator
                p01 = float(s.quantile(0.01))
                p99 = float(s.quantile(0.99))
                mean_val = float(s.mean())
                std_val = float(s.std())
                ref_stats[c] = {
                    "mean": mean_val,
                    "std": std_val,
                    "median": med,
                    "mad": mad,
                    "p01": p01,
                    "p99": p99,
                }
            else:
                # Insufficient data
                ref_stats[c] = {
                    "mean": 0.0, "std": 1.0, "median": 0.0, 
                    "mad": 1.0, "p01": -3.0, "p99": 3.0
                }
        else:
            # Placeholder for missing features
            ref_stats[c] = {
                "mean": 0.0, "std": 1.0, "median": 0.0, 
                "mad": 1.0, "p01": -3.0, "p99": 3.0
            }

    # Write back into bundle (atomic with backup)
    tmpdir = Path(tempfile.mkdtemp())
    tmp_path = tmpdir / (bundle_path.name + ".tmp")
    bak_path = bundle_path.with_suffix(".bak." + datetime.utcnow().strftime("%Y%m%d-%H%M%S") + ".joblib")

    b["training_feature_snapshot"] = X
    b["reference_stats"] = ref_stats
    b.setdefault("trainer_version", b.get("trainer_version") or "v2.2")
    # Optional: stash snapshot_rows into training_period for transparency
    tp = b.get("training_period") or {}
    tp["snapshot_rows"] = int(len(X))
    b["training_period"] = tp

    log.info("Writing backup → %s", bak_path.name)
    shutil.copy2(bundle_path, bak_path)

    log.info("Saving updated bundle atomically…")
    joblib.dump(b, tmp_path, compress=3)
    shutil.move(str(tmp_path), str(bundle_path))
    shutil.rmtree(tmpdir, ignore_errors=True)
    log.info("✅ training_feature_snapshot stored in bundle. Done.")

if __name__ == "__main__":
    main()
