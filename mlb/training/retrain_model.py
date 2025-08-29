#!/usr/bin/env python3
"""
retrain_model.py
================
Build a fresh model from finished games in a rolling window, evaluate, audit, and (optionally) deploy.

Examples:
  python retrain_model.py --end 2025-08-15 --window-days 150 --deploy
"""
import os, sys, json, logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import joblib

# ⬇️ add these imports for serving alignment
from enhanced_bullpen_predictor import EnhancedBullpenPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("retrain")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
MODEL_DIR = Path("../models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LATEST_PATH = MODEL_DIR / "legitimate_model_latest.joblib"

def _fetch_training_frame(engine, start_date, end_date, require_market=False):
    """
    Pull finished games and serving inputs. Make sure 'total_runs' and 'market_total'
    are unique Series (no duplicate column names).
    """
    market_filter = "AND eg.market_total IS NOT NULL" if require_market else ""

    q = text(f"""
        SELECT
          lgf.*,
          eg.market_total AS market_total_final,   -- avoid name clash
          COALESCE(lgf.total_runs,
                   CASE WHEN eg.home_score IS NOT NULL AND eg.away_score IS NOT NULL
                        THEN eg.home_score + eg.away_score END) AS total_runs_label,
          COALESCE(NULLIF(eg.home_sp_season_era, 0), lgf.home_sp_season_era, 4.5) AS home_sp_season_era_final,
          COALESCE(NULLIF(eg.away_sp_season_era, 0), lgf.away_sp_season_era, 4.5) AS away_sp_season_era_final
        FROM legitimate_game_features lgf
        JOIN enhanced_games eg
          ON eg.game_id = lgf.game_id AND eg."date" = lgf."date"
        WHERE lgf."date" BETWEEN :s AND :e
          AND (lgf.total_runs IS NOT NULL OR (eg.home_score IS NOT NULL AND eg.away_score IS NOT NULL))
          {market_filter}
    """)
    df = pd.read_sql(q, engine, params={"s": start_date, "e": end_date})
    if df.empty:
        return df

    df = df.copy()

    # Unify label
    if "total_runs_label" in df.columns:
        if "total_runs" in df.columns:
            df.drop(columns=["total_runs"], inplace=True)
        df.rename(columns={"total_runs_label": "total_runs"}, inplace=True)

    # Unify market_total (prefer the EG value we just aliased)
    if "market_total_final" in df.columns:
        if "market_total" in df.columns:
            df.drop(columns=["market_total"], inplace=True)
        df.rename(columns={"market_total_final": "market_total"}, inplace=True)

    # Normalize SP ERA column names used by feature builder
    if "home_sp_season_era_final" in df.columns:
        df["home_sp_season_era"] = df.pop("home_sp_season_era_final")
    if "away_sp_season_era_final" in df.columns:
        df["away_sp_season_era"] = df.pop("away_sp_season_era_final")

    return df

def _engineer(df):
    from enhanced_bullpen_predictor import EnhancedBullpenPredictor
    P = EnhancedBullpenPredictor()
    base = df.copy()
    # ensure market_total column name matches serving
    if "market_total_final" in base.columns:
        base["market_total"] = base.pop("market_total_final")
    # ✅ ensure serving parity - create expected_total just like in daily pipeline
    if "market_total" in base.columns:
        base["expected_total"] = pd.to_numeric(base["market_total"], errors="coerce")
    feats = P.engineer_features(base)
    return feats

def _pick_feature_columns(feats: pd.DataFrame):
    # avoid leakage & identifiers; keep numeric columns only
    drop = {
        "total_runs", "home_score", "away_score",
        "game_id", "date", "game_time_utc", "created_at",
        "recommendation", "edge", "confidence"
    }
    num = feats.select_dtypes(include=[np.number]).copy()
    cols = [c for c in num.columns if c not in drop and not c.endswith('_id')]
    return cols

def _make_fill_values(X: pd.DataFrame):
    med = X.median(numeric_only=True).to_dict()
    # simple clip to avoid NaNs
    return {k: (0.0 if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)) for k, v in med.items()}

def _time_decay_weights(dates: pd.Series, half_life_days=60.0):
    # newer games get higher weight; w = 0.5 ** (age / half_life)
    maxd = pd.to_datetime(dates).max()
    age = (maxd - pd.to_datetime(dates)).dt.days.astype(float)
    hl = max(1e-6, float(half_life_days))
    w = np.power(0.5, age / hl).astype(float)
    return np.clip(w, 0.05, 1.0)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"),
                    help="End date (inclusive) for training window")
    ap.add_argument("--window-days", type=int, default=150)
    ap.add_argument("--holdout-days", type=int, default=21)
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--min-samples-leaf", type=int, default=2)
    ap.add_argument("--deploy", action="store_true", help="If set, swap in latest on success")
    ap.add_argument("--audit", action="store_true", help="Run training_bundle_audit after saving")
    ap.add_argument("--require-market", action="store_true", help="Only train on games with market_total data")
    args = ap.parse_args()

    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    start = end - timedelta(days=args.window_days - 1)
    holdout_start = end - timedelta(days=args.holdout_days - 1)

    log.info("Training window: %s → %s (holdout starts: %s)", start, end, holdout_start)

    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    try:
        raw = _fetch_training_frame(engine, start, end, require_market=args.require_market)
    finally:
        engine.dispose()

    if raw.empty:
        msg = "No finished games found in window"
        if args.require_market:
            msg += " (with market data)"
        log.error(msg + ". Aborting.")
        sys.exit(2)

    # --- build features & align to serving schema (61 cols) ---
    feats = _engineer(raw)
    P = EnhancedBullpenPredictor()

    try:
        X = P.align_serving_features(feats, strict=False)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=P.feature_columns)
    except Exception as e:
        log.error("align_serving_features failed: %s", e)
        # fallback: reindex to serving columns
        X = feats.reindex(columns=P.feature_columns)

    # Ensure only numeric features for model training
    X = X.select_dtypes(include=[np.number])
    cols = list(X.columns)  # serving feature order
    # fill values (medians; fail-safe to 0.0)
    fill_values = _make_fill_values(X)
    X = X.fillna(pd.Series(fill_values)).fillna(0.0)

    # X / y split (unchanged y)
    y = pd.to_numeric(raw["total_runs"], errors="coerce")
    good = y.notna()
    X, y, raw = X.loc[good], y.loc[good], raw.loc[good]

    # time-based split
    dates = pd.to_datetime(raw["date"])
    train_mask = dates < pd.to_datetime(holdout_start)
    test_mask  = ~train_mask

    if train_mask.sum() < 200 or test_mask.sum() < 20:
        log.warning("Small split (train=%d, test=%d). Consider longer window.", train_mask.sum(), test_mask.sum())

    Xtr, ytr = X.loc[train_mask], y.loc[train_mask]
    Xte, yte = X.loc[test_mask],  y.loc[test_mask]

    # sample weights (time decay)
    w = _time_decay_weights(raw.loc[train_mask, "date"], half_life_days=60)

    # model
    rf = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=-1,
        random_state=42,
        oob_score=False
    )
    # Simple preprocessing: scale numeric features (forest not sensitive, but we will reuse same preproc at serve)
    preproc = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True))
    ])
    Xtr_proc = preproc.fit_transform(Xtr)
    Xte_proc = preproc.transform(Xte)
    rf.fit(Xtr_proc, ytr, sample_weight=w)

    # eval
    yhat_tr = rf.predict(Xtr_proc)
    yhat_te = rf.predict(Xte_proc)

    mae_tr = mean_absolute_error(ytr, yhat_tr) if len(ytr) else np.nan
    mae_te = mean_absolute_error(yte, yhat_te) if len(yte) else np.nan

    # bias correction: mean(actual - pred) on holdout (safer than train)
    bias_correction = float((yte - yhat_te).mean()) if len(yte) else 0.0
    log.info("Holdout bias (actual - pred) = %+0.3f runs", bias_correction)

    log.info("Train MAE = %.3f | Holdout MAE = %.3f (n_te=%d)", mae_tr, mae_te, len(yte))

    # market baseline on holdout (if available)
    if "market_total" in raw.columns:
        mkt_te_series = raw.loc[test_mask, "market_total"]
        # Handle potential duplicate columns by taking the first column
        if isinstance(mkt_te_series, pd.DataFrame):
            mkt_te_series = mkt_te_series.iloc[:, 0]
        mkt_te = pd.to_numeric(mkt_te_series, errors="coerce")
        if mkt_te.notna().sum() > 10:
            mae_market = mean_absolute_error(yte, mkt_te)
            log.info("Holdout MAE — model: %.3f  vs market: %.3f (Δ=%.3f)",
                     mae_te, mae_market, mae_te - mae_market)
        else:
            mae_market = None
    else:
        mae_market = None

    # bundle metadata
    training_feature_snapshot = X.sample(min(len(X), 1000), random_state=42).copy()
    ref_stats = {}
    for c in X.columns:
        s = pd.to_numeric(X[c], errors="coerce")
        med = float(s.median()) if s.notna().sum() else 0.0
        mad = float(1.4826 * np.median(np.abs(s - med))) if s.notna().sum() else 1.0
        ref_stats[c] = {"median": med, "mad": max(mad, 1e-3)}

    bundle = {
        "model": rf,
        "model_type": "legitimate_random_forest",
        "feature_columns": cols,
        "feature_fill_values": fill_values,
    "preproc": preproc,
    "scaler": None,
        "label_definition": "total_runs",
        "training_period": {"start": str(start), "end": str(end), "holdout_start": str(holdout_start)},
        "training_feature_snapshot": training_feature_snapshot,
        "evaluation_metrics": {
            "mae_train": float(mae_tr) if np.isfinite(mae_tr) else None,
            "mae_holdout": float(mae_te) if np.isfinite(mae_te) else None,
            "mae_market_holdout": float(mae_market) if (mae_market is not None) else None,
            "n_train": int(len(Xtr)),
            "n_holdout": int(len(Xte)),
        },
        "reference_stats": ref_stats,
        "training_date": datetime.utcnow().isoformat(),
        "schema_version": "1",
        "trainer_version": "v2.2_retrain_loop",
        "feature_sha": str(hash(str(sorted(cols)))),
        "bias_correction": bias_correction,
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        "version": f"rf_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
    }

    out_path = MODEL_DIR / f"legitimate_model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.joblib"
    joblib.dump(bundle, out_path)
    log.info("Wrote new bundle: %s", out_path)

    # optional: audit
    if args.audit:
        try:
            import training_bundle_audit as tba
            tba.audit_training_bundle(
                target_date=args.end,
                apply_bias_correction=False,
                model_path=str(out_path),
                enforce_metadata=True,
                serve_days=7,
                psi_min_serving=100,
                psi_min_training=1000,
                dry_run=True
            )
            log.info("Audit completed.")
        except Exception as e:
            log.error("Audit failed: %s", e)
            sys.exit(3)

    # Create adaptive learning model format for learning_model_predictor.py
    adaptive_learning_bundle = {
        'model': rf,  # Same trained model
        'feature_columns': cols,  # Same feature columns
        'learning_config': {
            'model_type': 'RandomForestRegressor',
            'retrained_date': datetime.utcnow().isoformat(),
            'feature_count': len(cols),
            'test_mae': float(mae_te) if np.isfinite(mae_te) else None
        }
    }
    
    adaptive_path = MODEL_DIR / "adaptive_learning_model.joblib"
    joblib.dump(adaptive_learning_bundle, adaptive_path)
    log.info("Created adaptive learning model: %s", adaptive_path)

    # optional: deploy (atomic swap)
    if args.deploy:
        try:
            # copy (symlinks can be finicky on Windows)
            joblib.dump(bundle, LATEST_PATH)
            log.info("Deployed new bundle to: %s", LATEST_PATH)
        except Exception as e:
            log.error("Failed to deploy latest: %s", e)
            sys.exit(4)

if __name__ == "__main__":
    main()
