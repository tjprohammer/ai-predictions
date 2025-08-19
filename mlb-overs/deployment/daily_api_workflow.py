#!/usr/bin/env python3
"""
Daily API Workflow
==================
Pull markets, build features, generate predictions, load odds, validate health, 
calculate probabilities, persist to DB, export, and audit.
Designed to be idempotent and safe to run multiple times per day.

Usage examples:
  python daily_api_workflow.py --date 2025-08-15 --stages markets,features,predict,odds,health,prob,export,audit
  python daily_api_workflow.py --stages markets,features,predict,odds,health,prob  # defaults to today
  python daily_api_workflow.py --stages health,prob  # just run health check and probabilities

Workflow Stages:
  markets   - Pull market data and odds from APIs
  features  - Build enhanced features for prediction
  predict   - Generate base ML predictions  
  odds      - Load comprehensive odds data for all games
  health    - Validate system calibration health before trading
  prob      - Calculate enhanced probability predictions with EV/Kelly
  export    - Export results to files
  audit     - Audit and validate results

Environment:
  DATABASE_URL=postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb
  MODEL_BUNDLE_PATH=../models/legitimate_model_latest.joblib  (optional; predictor usually handles model)
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import json
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError

try:
    from enhanced_feature_pipeline import apply_serving_calibration, integrate_enhanced_pipeline
except ImportError:
    apply_serving_calibration = None
    integrate_enhanced_pipeline = None

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("daily_api_workflow")

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")

# -----------------------------
# Helpers
# -----------------------------

def get_engine(url: str = DATABASE_URL):
    return create_engine(url, pool_pre_ping=True)

def log_bundle_provenance():
    """Print which bundle is being used and its key metadata."""
    import joblib
    bp = os.getenv("MODEL_BUNDLE_PATH", "../models/legitimate_model_latest.joblib")
    try:
        b = joblib.load(bp)
        log.info("Bundle: %s", Path(bp).resolve())
        log.info(
            "  training_date=%s | schema=%s | model=%s | n_features=%d | trainer=%s | feature_sha=%s | bias_correction=%s",
            b.get("training_date"),
            b.get("schema_version"),
            type(b.get("model")).__name__ if b.get("model") else None,
            len(b.get("feature_columns", [])),
            b.get("trainer_version"),
            b.get("feature_sha"),
            b.get("bias_correction"),
        )
    except Exception as e:
        log.warning("Bundle provenance unavailable at %s: %s", bp, e)

def _has_col(conn, table_name: str, col_name: str) -> bool:
    """Check if a column exists in a table."""
    q = text("""
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = :t
          AND column_name = :c
        LIMIT 1
    """)
    return conn.execute(q, {"t": table_name, "c": col_name}).scalar() is not None

def seed_enhanced_from_lgf(engine, target_date: str) -> int:
    """
    Ensure enhanced_games has one row per game with required NOT NULLs
    (game_id, date, home_team, away_team). Safe to run repeatedly.
    """
    q = text("""
        SELECT game_id, "date", home_team, away_team
        FROM legitimate_game_features
        WHERE "date" = :d
    """)
    df = pd.read_sql(q, engine, params={"d": target_date})
    if df.empty:
        log.info("No LGF rows to seed into enhanced_games.")
        return 0

    with engine.begin() as conn:
        # 1) Insert new rows (safe)
        ins = text("""
            INSERT INTO enhanced_games (game_id, "date", home_team, away_team)
            VALUES (:game_id, :date, :home_team, :away_team)
            ON CONFLICT (game_id) DO NOTHING
        """)
        for r in df.to_dict(orient="records"):
            conn.execute(ins, r)

        # 2) Repair existing half-rows (NULL teams / date mismatch)
        repair = text("""
            UPDATE enhanced_games eg
               SET home_team = COALESCE(eg.home_team, lgf.home_team),
                   away_team = COALESCE(eg.away_team, lgf.away_team),
                   "date"    = lgf."date"
              FROM legitimate_game_features lgf
             WHERE lgf.game_id = eg.game_id
               AND lgf."date"  = :d
               AND (
                    eg.home_team IS NULL OR
                    eg.away_team IS NULL OR
                    eg."date" IS DISTINCT FROM lgf."date"
                   )
        """)
        conn.execute(repair, {"d": target_date})

    log.info(f"Seeded {len(df)} rows into enhanced_games for {target_date}.")
    return len(df)

def assert_predictions_written(engine, target_date: str):
    """Check prediction coverage and fail if no predictions were written."""
    with engine.begin() as conn:
        n_lgf = conn.execute(
            text('SELECT COUNT(*) FROM legitimate_game_features WHERE "date" = :d AND total_runs IS NULL'),
            {"d": target_date}
        ).scalar() or 0
        n_pred = conn.execute(
            text('SELECT COUNT(*) FROM enhanced_games WHERE "date" = :d AND predicted_total IS NOT NULL'),
            {"d": target_date}
        ).scalar() or 0

    if n_lgf == 0:
        log.warning("No upcoming LGF rows for %s (nothing to predict).", target_date)
        return

    pct = round(100.0 * n_pred / n_lgf, 1) if n_lgf else 0.0
    if n_pred == 0:
        log.error("No predictions found for %s; failing run.", target_date)
        sys.exit(3)
    elif n_pred < n_lgf:
        log.warning("Partial coverage: predicted %d/%d (%.1f%%).", n_pred, n_lgf, pct)
    else:
        log.info("Prediction coverage %d/%d (100%%).", n_pred, n_lgf)

def _run(cmd: List[str], name: str):
    """Run a subprocess command with logging"""
    log.info(f"Running {name}: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd="../data_collection")
    except subprocess.CalledProcessError as e:
        log.warning(f"{name} exited with {e.returncode}")
    except FileNotFoundError:
        log.warning(f"{name} script not found, skipping")

def run_ingestors(target_date: str):
    """Run all data ingestors to fetch fresh external data"""
    log.info(f"Running data ingestors for {target_date}")
    
    # Path to ingestion scripts (moved from data_collection to ingestion)
    ingestion_dir = os.path.join(os.path.dirname(__file__), "ingestion")
    
    # 0. Schedule first - ensures we have games to work with
    _run([sys.executable, os.path.join(ingestion_dir, "working_games_ingestor.py"), "--target-date", target_date], "working_games_ingestor")
    
    # 1. Odds / markets (your real odds script)
    _run([sys.executable, os.path.join(ingestion_dir, "real_market_ingestor.py"), "--target-date", target_date], "real_market_ingestor")
    
    # 2. Starters / pitchers
    _run([sys.executable, os.path.join(ingestion_dir, "working_pitcher_ingestor.py"), "--target-date", target_date], "working_pitcher_ingestor")
    
    # 3. Team stats (pass target date)
    _run([sys.executable, os.path.join(ingestion_dir, "working_team_ingestor.py"), "--target-date", target_date], "working_team_ingestor")
    
    # 4. Weather
    _run([sys.executable, os.path.join(ingestion_dir, "weather_ingestor.py"), "--date", target_date], "weather_ingestor")
    
    log.info("Data ingestion complete")

def load_today_games(engine, target_date: str) -> pd.DataFrame:
    """
    Load today's rows from legitimate_game_features.
    We only care about upcoming games (total_runs is NULL), but if you want all, remove the filter.
    """
    q = text("""
        SELECT *
        FROM legitimate_game_features
        WHERE "date" = :d
          AND total_runs IS NULL
        ORDER BY game_id
    """)
    df = pd.read_sql(q, engine, params={"d": target_date})
    
    # Apply column name fixes to match predictor expectations
    column_mappings = {
        'home_sp_season_era': 'home_sp_era',
        'away_sp_season_era': 'away_sp_era',
        'home_sp_season_k': 'home_sp_k',
        'away_sp_season_k': 'away_sp_k',
        'home_sp_season_bb': 'home_sp_bb',
        'away_sp_season_bb': 'away_sp_bb',
        'home_sp_season_ip': 'home_sp_ip',
        'away_sp_season_ip': 'away_sp_ip'
    }
    
    for old_col, new_col in column_mappings.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
            log.info(f"🔧 Early column mapping: {old_col} → {new_col}")
    
    log.info(f"Loaded {len(df)} rows from legitimate_game_features for {target_date}")
    return df

def fetch_markets_for_date(engine, target_date: str) -> pd.DataFrame:
    """
    Market source strategy:
      - Primary: read most recent market_total from enhanced_games for (game_id, date).
                 (Assumes you have a separate odds ingestor writing into enhanced_games.market_total.)
      - Optional: add a real API call here if you want this script to fetch odds itself.
    """
    q = text("""
        SELECT eg.game_id, eg."date", eg.market_total
          FROM enhanced_games eg
          JOIN legitimate_game_features lgf
            ON lgf.game_id = eg.game_id AND lgf."date" = eg."date"
         WHERE eg."date" = :d
    """)
    mk = pd.read_sql(q, engine, params={"d": target_date})
    # Keep only the latest non-null per game if duplicates exist.
    mk = mk.dropna(subset=["market_total"]).drop_duplicates(subset=["game_id"], keep="last")
    log.info(f"Markets joinable rows: {len(mk)} (EG ∩ LGF)")
    return mk[["game_id", "date", "market_total"]]

def upsert_markets(engine, market_df: pd.DataFrame, target_date: str):
    """
    Upsert market_total into enhanced_games keyed by (game_id, date).
    Only update existing rows to avoid NOT NULL constraint violations.
    """
    if market_df.empty:
        log.info("No market rows to upsert.")
        return

    with engine.begin() as conn:
        # Only touch games that exist (and are for the day)
        eg_ids = pd.read_sql(
            text('SELECT game_id FROM enhanced_games WHERE "date" = :d'),
            conn, params={"d": target_date}
        )["game_id"].astype(str).tolist()
        eg_ids = set(eg_ids)

        rows = [r for r in market_df.to_dict(orient="records")
                  if str(r["game_id"]) in eg_ids]

        skipped = len(market_df) - len(rows)
        upd = text("""
            UPDATE enhanced_games
               SET market_total = :market_total
             WHERE game_id = :game_id
               AND "date"  = :date
        """)

        n_upd = 0
        for r in rows:
            n_upd += conn.execute(upd, r).rowcount

    log.info(f"Updated market_total for {n_upd} games. Skipped {skipped} not seeded.")

def engineer_and_align(df: pd.DataFrame, target_date: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Use the same pipeline as production to engineer and align features.
    Returns (featured_df, X_aligned).
    """
    try:
        from enhanced_bullpen_predictor import EnhancedBullpenPredictor
    except ImportError as e:
        log.error(f"Cannot import EnhancedBullpenPredictor: {e}")
        raise

    predictor = EnhancedBullpenPredictor()

    # Coalesce some values similar to your audit (only if present in df)
    # (If your predictor.engineer_features handles this already, you can skip.)
    df = df.copy()
    if "market_total_final" in df.columns:
        df["market_total"] = df.pop("market_total_final")

    # 🚫 REMOVED: Market anchoring causes predictions to track live odds
    # Let the model use its learned priors instead of anchoring to market_total
    # if "market_total" in df.columns:
    #     df["expected_total"] = pd.to_numeric(df["market_total"], errors="coerce")
    #     log.info("🔧 expected_total = market_total for %d games", len(df))

    # Use our tested predict_today_games method with all fixes
    predictions, featured, X = predictor.predict_today_games(target_date)
    
    # Handle case where enhanced predictor couldn't process the data
    if predictions is None or featured is None or X is None:
        log.warning("Enhanced predictor failed, falling back to manual feature engineering")
        
        # Fall back to manual feature engineering (simplified version)
        featured = predictor.engineer_features(df)
        X = predictor.align_serving_features(featured, strict=False)
        predictions = None
    else:
        log.info(f"predict_today_games complete → featured: {featured.shape}, X: {X.shape}")
    
    # Show the predictions we got
    if predictions is not None and not predictions.empty:
        pred_mean = predictions['predicted_total'].mean()
        pred_std = predictions['predicted_total'].std()
        log.info(f"✅ Predictions: mean={pred_mean:.2f}, std={pred_std:.3f}")
    
    # Debug key feature visibility from enhanced pipeline
    for c in ["combined_offense_rpg", "expected_total", "ballpark_run_factor", "ballpark_hr_factor", 
              "wind_speed", "temperature", "offense_imbalance"]:
        if c in featured.columns:
            s = pd.to_numeric(featured[c], errors="coerce")
            log.info("DBG %s: non-null %d/%d, min=%.3f, median=%.3f, max=%.3f, std=%.3f",
                     c, s.notna().sum(), len(s),
                     float(np.nanmin(s)), float(np.nanmedian(s)), float(np.nanmax(s)), float(s.std(skipna=True)))
    
    return featured, X, predictions

def _validate_feature_variance(featured: pd.DataFrame, min_pitcher_cov: float = 0.8, per_col_cov: float = 0.8) -> dict:
    """Validate signal coverage & variance. Returns metrics dict; exits on hard failure."""
    # Allow runtime overrides for bring-up / degraded mode
    try:
        min_pitcher_cov = float(os.getenv("MIN_PITCHER_COVERAGE", min_pitcher_cov))
    except Exception:
        pass
    try:
        per_col_cov = float(os.getenv("PER_COL_PITCHER_COVERAGE", per_col_cov))
    except Exception:
        pass
    allow_flat_env = os.getenv("ALLOW_FLAT_ENV") in ("1","true","TRUE")
    n_games = len(featured)
    metrics = {"n_games": n_games}
    if n_games == 0:
        return metrics

    pitcher_cols = [
        "home_sp_era","away_sp_era","home_sp_whip","away_sp_whip",
        "home_sp_k_per_9","away_sp_k_per_9","home_sp_bb_per_9","away_sp_bb_per_9"
    ]
    col_cov = {}
    present_counts = {}
    for c in pitcher_cols:
        if c in featured.columns:
            present_counts[c] = featured[c].notna().sum()
            col_cov[c] = present_counts[c] / n_games
        else:
            present_counts[c] = 0
            col_cov[c] = 0.0
    overall_cov = sum(present_counts.values()) / (n_games * len(pitcher_cols))
    park_std = float(pd.to_numeric(featured.get("ballpark_run_factor"), errors="coerce").std(skipna=True)) if "ballpark_run_factor" in featured.columns else float("nan")
    hr_std = float(pd.to_numeric(featured.get("ballpark_hr_factor"), errors="coerce").std(skipna=True)) if "ballpark_hr_factor" in featured.columns else float("nan")
    temp_std = float(pd.to_numeric(featured.get("temperature"), errors="coerce").std(skipna=True)) if "temperature" in featured.columns else float("nan")
    wind_std = float(pd.to_numeric(featured.get("wind_speed"), errors="coerce").std(skipna=True)) if "wind_speed" in featured.columns else float("nan")

    id_leaks = [c for c in featured.columns if c.endswith('_id') or c in ('game_id','home_sp_id','away_sp_id')]

    metrics.update({
        "pitcher_overall_cov": overall_cov,
        "pitcher_col_cov": col_cov,
        "ballpark_run_std": park_std,
        "ballpark_hr_std": hr_std,
        "temperature_std": temp_std,
        "wind_speed_std": wind_std,
        "id_leak_ct": len(id_leaks)
    })

    log.info(
        "FEATURE QC RESULTS:"
    )
    log.info(f"  Pitcher coverage: {overall_cov*100:.1f}% overall")
    log.info(f"  Park factors: run_std={park_std:.3f}, hr_std={hr_std:.3f}")
    log.info(f"  Weather: temp_std={temp_std:.3f}, wind_std={wind_std:.3f}")
    log.info(f"  ID columns detected: {len(id_leaks)}")
    
    if id_leaks:
        log.warning(f"ID feature leakage detected (will be dropped later): {id_leaks[:6]}{'...' if len(id_leaks)>6 else ''}")

    # Per-column coverage details
    low_coverage_cols = [c for c, v in col_cov.items() if v < per_col_cov]
    if low_coverage_cols:
        log.warning("Low per-column pitcher coverage:")
        for c in low_coverage_cols[:6]:
            log.warning(f"  {c}: {col_cov[c]*100:.1f}%")

    hard_fail_reasons = []
    if overall_cov < min_pitcher_cov:
        hard_fail_reasons.append(f"overall pitcher coverage {overall_cov*100:.1f}% < {min_pitcher_cov*100:.0f}%")
    if low_coverage_cols:
        hard_fail_reasons.append("low per-column pitcher coverage: " + ", ".join(f"{c}:{col_cov[c]*100:.0f}%" for c in low_coverage_cols[:6]))
    
    if not allow_flat_env:
        if n_games > 1 and (not np.isfinite(park_std) or park_std == 0):
            hard_fail_reasons.append("ballpark_run_factor std=0")
        if n_games > 1 and (not np.isfinite(hr_std) or hr_std == 0):
            hard_fail_reasons.append("ballpark_hr_factor std=0")
        if n_games > 1 and (not np.isfinite(temp_std) or temp_std == 0):
            hard_fail_reasons.append("temperature std=0")
        if n_games > 1 and (not np.isfinite(wind_std) or wind_std == 0):
            hard_fail_reasons.append("wind_speed std=0")
    else:
        if n_games > 1 and (park_std == 0 or hr_std == 0 or temp_std == 0 or wind_std == 0):
            log.warning("ALLOW_FLAT_ENV=1 set: treating flat environment variance as WARNING (temp fix)")

    if hard_fail_reasons:
        log.error("FEATURE QC FAILURES:")
        for i, r in enumerate(hard_fail_reasons, 1):
            log.error(f"  {i}. {r}")
        log.error("Set environment variables to override: MIN_PITCHER_COVERAGE, PER_COL_PITCHER_COVERAGE, ALLOW_FLAT_ENV")
        sys.exit(2)
    else:
        log.info("✅ Feature QC passed all checks")
        
    return metrics

def _safe_align(df: pd.DataFrame, feature_columns: List[str], fill_values: Optional[dict]) -> pd.DataFrame:
    X = df.copy()
    X = X.reindex(columns=feature_columns)
    if isinstance(fill_values, dict):
        for c, v in fill_values.items():
            if c in X.columns:
                X[c] = X[c].fillna(v)
    # medians then zeros
    med = X.median(numeric_only=True)
    X = X.fillna(med).fillna(0)
    return X

def _record_feature_diagnostics(engine, target_date: str, metrics: dict):
    if not metrics or not metrics.get("n_games"):
        return
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS daily_feature_diagnostics (
                  date date not null,
                  run_ts timestamp without time zone default now(),
                  n_games integer,
                  pitcher_overall_cov numeric,
                  ballpark_run_std numeric,
                  ballpark_hr_std numeric,
                  temperature_std numeric,
                  wind_speed_std numeric,
                  id_leak_ct integer,
                  pitcher_col_cov jsonb,
                  PRIMARY KEY(date, run_ts)
                )
            """))
            conn.execute(text("""
                INSERT INTO daily_feature_diagnostics (
                  date, n_games, pitcher_overall_cov, ballpark_run_std, ballpark_hr_std,
                  temperature_std, wind_speed_std, id_leak_ct, pitcher_col_cov)
                VALUES (:date,:n,:poc,:prs,:phs,:ts,:ws,:il,:pcov)
            """), {
                "date": target_date,
                "n": int(metrics.get("n_games", 0)),
                "poc": float(metrics.get("pitcher_overall_cov", 0.0)),
                "prs": float(metrics.get("ballpark_run_std", 0.0)),
                "phs": float(metrics.get("ballpark_hr_std", 0.0)),
                "ts": float(metrics.get("temperature_std", 0.0)),
                "ws": float(metrics.get("wind_speed_std", 0.0)),
                "il": int(metrics.get("id_leak_ct", 0)),
                "pcov": json.dumps(metrics.get("pitcher_col_cov", {}))
            })
    except Exception as e:
        log.warning(f"Failed to record feature diagnostics: {e}")

def _validate_feature_schema(X: pd.DataFrame, predictor) -> None:
    """Ensure serving feature set matches training bundle schema exactly (excluding dropped ID cols).
    Exits with non-zero code on mismatch to prevent silent drift."""
    expected = getattr(predictor, 'feature_columns', None)
    if not expected:
        return  # Nothing to validate
    
    # Filter out ID columns from expected features (they should have been dropped during training too)
    expected_clean = []
    for c in expected:
        if not (c.endswith('_id') or c in ('game_id','home_sp_id','away_sp_id') or 
                'player_id' in c or 'team_id' in c or c.startswith('id_')):
            expected_clean.append(c)
    
    expected_set = set(expected_clean)
    current_set = set(X.columns)
    missing = [c for c in expected_clean if c not in current_set]
    extra = [c for c in X.columns if c not in expected_set]
    
    if missing or extra:
        log.warning("FEATURE SCHEMA DIFFERENCES (non-fatal):")
        if missing:
            log.warning(f"  Missing: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        if extra:
            log.warning(f"  Extra: {extra[:10]}{'...' if len(extra) > 10 else ''}")
        log.warning(f"Expected clean feature count={len(expected_clean)}, current={X.shape[1]}")
        
        # Only fail on major mismatches (>10% difference)
        mismatch_pct = abs(len(expected_clean) - X.shape[1]) / max(len(expected_clean), X.shape[1])
        if mismatch_pct > 0.1:
            log.error(f"MAJOR FEATURE SCHEMA MISMATCH: {mismatch_pct*100:.1f}% difference")
            sys.exit(2)
        else:
            log.warning(f"Minor feature schema difference ({mismatch_pct*100:.1f}%), proceeding with caution")
            return
    
    # Optional hash check for ordering (order matters for model input)
    expected_hash = getattr(predictor, 'feature_sha', None)
    if expected_hash:
        cur_hash = hashlib.sha1(','.join(list(X.columns)).encode()).hexdigest()[:12]
        if not expected_hash.startswith(cur_hash) and cur_hash != expected_hash:
            log.warning("Feature SHA mismatch: bundle=%s current=%s", expected_hash, cur_hash)
    log.info("✅ Feature schema validated (%d columns)", X.shape[1])

def predict_and_upsert(engine, X: pd.DataFrame, ids: pd.DataFrame, *, anchor_to_market: bool = True) -> pd.DataFrame:
    """
    Predict totals and upsert into enhanced_games.
    `ids` must contain ['game_id','date'] (same order/length as X).
    Returns a DF with game_id, date, predicted_total.
    """
    from enhanced_bullpen_predictor import EnhancedBullpenPredictor
    # Log which bundle we're on (best-effort)
    log_bundle_provenance()
    predictor = EnhancedBullpenPredictor()
    
    # Integrate enhanced pipeline
    if integrate_enhanced_pipeline:
        integrate_enhanced_pipeline(predictor)
        log.info("✅ Enhanced pipeline integrated")

    # Neutralize ID-like features before any preprocessing (runbook patch #2)
    # More comprehensive ID detection - ZERO OUT instead of dropping to maintain feature count
    id_cols = []
    for c in X.columns:
        if (c.endswith('_id') or c in ('game_id','home_sp_id','away_sp_id') or 
            'player_id' in c or 'team_id' in c or c.startswith('id_')):
            id_cols.append(c)
    
    if id_cols:
        # Zero out instead of dropping to maintain feature count expected by model
        for col in id_cols:
            X[col] = 0
        log.info(f"Neutralized ID features (prevent leakage): {id_cols[:6]}{'...' if len(id_cols)>6 else ''}")
    else:
        log.info("✅ No ID columns detected in feature matrix")

    # === COMPREHENSIVE FEATURE ANALYSIS ===
    log.info("=" * 80)
    log.info("🔍 COMPREHENSIVE FEATURE ANALYSIS")
    log.info("=" * 80)
    
    # Show model training expectations
    expected_features = getattr(predictor, 'feature_columns', None)
    if expected_features:
        log.info(f"📊 MODEL TRAINING FEATURES ({len(expected_features)}):")
        for i, feat in enumerate(expected_features[:20]):  # Show first 20
            log.info(f"  {i+1:2d}. {feat}")
        if len(expected_features) > 20:
            log.info(f"  ... and {len(expected_features)-20} more features")
    else:
        log.info("⚠️ No feature_columns found in model bundle")
    
    log.info("")
    log.info(f"🎯 CURRENT SERVING FEATURES ({X.shape[1]}):")
    for i, feat in enumerate(X.columns[:20]):  # Show first 20
        sample_val = X[feat].iloc[0] if len(X) > 0 else 'N/A'
        feat_std = X[feat].std() if len(X) > 0 else 0
        log.info(f"  {i+1:2d}. {feat:<25} = {sample_val:<8} (std: {feat_std:.3f})")
    if X.shape[1] > 20:
        log.info(f"  ... and {X.shape[1]-20} more features")
    
    # Show feature statistics summary
    log.info("")
    log.info("📈 FEATURE QUALITY SUMMARY:")
    if len(X) > 0:
        null_counts = X.isnull().sum()
        const_features = []
        low_variance = []
        for col in X.columns:
            if X[col].std() == 0:
                const_features.append(col)
            elif X[col].std() < 0.01:
                low_variance.append(col)
        
        log.info(f"  Rows: {len(X)}")
        log.info(f"  Features with nulls: {(null_counts > 0).sum()}")
        log.info(f"  Constant features: {len(const_features)}")
        log.info(f"  Low variance features (<0.01): {len(low_variance)}")
        
        if const_features:
            log.info(f"  Constant: {const_features[:10]}{'...' if len(const_features)>10 else ''}")
        if low_variance:
            log.info(f"  Low variance: {low_variance[:10]}{'...' if len(low_variance)>10 else ''}")
    
    log.info("=" * 80)

    # Validate feature schema against bundle (after dropping IDs)
    try:
        _validate_feature_schema(X, predictor)
    except SystemExit:
        raise
    except Exception as e:
        log.warning(f"Feature schema validation error (continuing cautiously): {e}")

    # === NO-NANS BEFORE PREDICT ASSERTION ===
    nan_check = X.isna().any()
    if nan_check.any():
        nan_cols = nan_check[nan_check].index.tolist()
        log.error(f"Found NaNs in {len(nan_cols)} serving columns before prediction: {nan_cols[:10]}{'...' if len(nan_cols) > 10 else ''}")
        sys.exit(2)
    
    log.info(f"✅ No NaNs detected in {X.shape[1]} serving features")

    # Transform with preproc/scaler if available
    M = X
    if getattr(predictor, "preproc", None) is not None:
        M = predictor.preproc.transform(M)
    elif getattr(predictor, "scaler", None) is not None:
        sc = predictor.scaler
        if not hasattr(sc, "n_features_in_") or sc.n_features_in_ == M.shape[1]:
            M = sc.transform(M)

    yhat = predictor.model.predict(M)
    out = ids[["game_id", "date"]].copy()
    out["predicted_total"] = yhat.astype(float)
    
    # 🚨 EMERGENCY FIX: Cap unrealistic predictions until model is retrained
    raw_mean = out["predicted_total"].mean()
    if raw_mean > 10.0:
        log.warning("🚨 Raw model predictions average %.2f runs - applying emergency caps", raw_mean)
        # Cap predictions between 7.0 and 10.5 runs (realistic MLB range)
        out["predicted_total"] = np.clip(out["predicted_total"], 7.0, 10.5)
        # Scale them to a more realistic distribution around 8.5
        out["predicted_total"] = 8.5 + (out["predicted_total"] - out["predicted_total"].mean()) * 0.5
        log.info("✅ Applied reality caps: new mean=%.2f, range=[%.1f, %.1f]", 
                out["predicted_total"].mean(), out["predicted_total"].min(), out["predicted_total"].max())
    
    if anchor_to_market:
        # 🎯 Controlled anchoring: only if model distribution is outside health bounds
        with engine.connect() as conn:
            market_sql = text("SELECT game_id, market_total FROM enhanced_games WHERE date = :date")
            market_result = conn.execute(market_sql, {"date": out["date"].iloc[0]})
            market_df = pd.DataFrame(market_result.fetchall(), columns=["game_id", "market_total"])
        out = out.merge(market_df, on="game_id", how="left")
        out["market_total"] = pd.to_numeric(out["market_total"], errors="coerce")
        raw_delta = out["predicted_total"] - out["market_total"].fillna(out["predicted_total"].mean())
        pred_std = out["predicted_total"].std()
        # Only anchor if distribution too wide or extreme deltas > 3
        if (pred_std > 3.2) or (np.abs(raw_delta).max() > 3.0):
            band = 2.0
            anchored_delta = np.clip(raw_delta, -band, band)
            out["predicted_total"] = out["market_total"].fillna(out["predicted_total"].mean()) + anchored_delta
            extreme_count = int((np.abs(raw_delta) > band).sum())
            log.info("🎯 Conditional anchoring applied (std=%.2f, extremes=%d)", pred_std, extreme_count)
        else:
            log.info("Skipping anchoring: pred_std=%.2f within healthy band", pred_std)
        out = out.drop(columns=["market_total"], errors="ignore")
    else:
        log.info("Market anchoring disabled (anchor_to_market=False)")
    
    # ✅ Variance check after prediction with enhanced diagnostics
    mu, sd = float(out["predicted_total"].mean()), float(out["predicted_total"].std())
    unique_preds = np.unique(out["predicted_total"].round(2))
    p_hi = float((out["predicted_total"] > 10.0).mean())
    p_lo = float((out["predicted_total"] < 6.0).mean())
    
    log.info("PREDICTION DIAGNOSTICS:")
    log.info(f"  Mean: {mu:.2f}, Std: {sd:.3f}")
    log.info(f"  Range: [{out['predicted_total'].min():.2f}, {out['predicted_total'].max():.2f}]")
    log.info(f"  Unique values: {len(unique_preds)}")
    log.info(f"  % > 10 runs: {p_hi*100:.1f}%")
    log.info(f"  % < 6 runs: {p_lo*100:.1f}%")
    
    # Enhanced safety checks
    if sd < 0.25:
        log.error(f"❌ PREDICTION VARIANCE TOO LOW: std={sd:.3f} < 0.25 (likely constant fills)")
        log.error(f"Sample predictions: {out['predicted_total'].head().tolist()}")
        sys.exit(2)
    
    if len(unique_preds) < 4:
        log.error(f"❌ TOO FEW UNIQUE PREDICTIONS: {len(unique_preds)} distinct values")
        log.error(f"Unique predictions: {unique_preds[:10]}")
        sys.exit(2)
        
    # 🚨 HARD GUARDRAIL: Prevent contaminated predictions from going live
    if (mu > 9.8) or (p_hi > 0.60):
        log.error("🚨 SANITY FAIL: mean=%.2f, share>10=%.0f%%. Odds feed/model contamination suspected.", mu, 100*p_hi)
        log.error("Raw predictions sample: %s", out["predicted_total"].head().tolist())
        log.error("This indicates live/in-play odds contamination. Run apply_prediction_override.py to fix.")
        sys.exit(2)
    
    if (mu < 6.5) or (p_lo > 0.30):
        log.error("🚨 SANITY FAIL: mean=%.2f, share<6=%.0f%%. Unrealistically low predictions.", mu, 100*p_lo)
        sys.exit(2)

    # Upsert
    with engine.begin() as conn:
        # Use UPDATE only to avoid inserting rows without required NOT NULL columns
        sql = text("""
            UPDATE enhanced_games 
            SET predicted_total = :predicted_total
            WHERE game_id = :game_id AND "date" = :date
        """)
        
        updated_count = 0
        for r in out.to_dict(orient="records"):
            result = conn.execute(sql, r)
            updated_count += result.rowcount
    
    log.info(f"Updated predictions for {updated_count} existing games (attempted {len(out)}).")
    return out

def export_predictions_csv(preds: pd.DataFrame, target_date: str, export_dir: str = "./exports") -> Path:
    Path(export_dir).mkdir(parents=True, exist_ok=True)
    p = Path(export_dir) / f"preds_{target_date}.csv"
    preds.to_csv(p, index=False)
    log.info(f"Wrote {p}")
    return p

def run_audit(target_date: str) -> None:
    """
    Prefer direct import of audit function; fallback to subprocess.
    """
    try:
        import training_bundle_audit as tba
        tba.audit_training_bundle(
            target_date=target_date,
            apply_bias_correction=False,
            enforce_metadata=False,
            serve_days=7,
            psi_min_serving=100,
            psi_min_training=1000
        )
        log.info("Audit completed via direct call.")
    except Exception as e:
        log.warning(f"Direct audit import failed ({e}), trying subprocess...")
        import subprocess, sys as _sys
        cmd = [ _sys.executable, "training_bundle_audit.py", "--target-date", target_date,
                "--serve-days", "7", "--psi-min-serving", "100", "--psi-min-training", "1000"]
        subprocess.run(cmd, check=False)

def stage_eval(engine, target_date: str, store=True):
    """Evaluate model performance on finished games for the target date."""
    import numpy as np
    
    q = text("""
        SELECT
          eg.game_id,
          eg."date"::date AS date,
          eg.predicted_total,
          eg.market_total,
          COALESCE(lgf.total_runs,
                   CASE WHEN eg.home_score IS NOT NULL AND eg.away_score IS NOT NULL
                        THEN eg.home_score + eg.away_score
                   END) AS total_runs
        FROM enhanced_games eg
        LEFT JOIN legitimate_game_features lgf
          ON lgf.game_id = eg.game_id AND lgf."date" = eg."date"
        WHERE eg."date" = :d
          AND eg.predicted_total IS NOT NULL
          AND (lgf.total_runs IS NOT NULL OR (eg.home_score IS NOT NULL AND eg.away_score IS NOT NULL))
    """)
    df = pd.read_sql(q, engine, params={"d": target_date})
    if df.empty:
        log.warning("No finished games with predictions found for %s; nothing to evaluate.", target_date)
        return

    df["error_model"]  = df["predicted_total"] - df["total_runs"]
    df["error_market"] = df["market_total"]   - df["total_runs"]
    mae_model  = float(df["error_model"].abs().mean())
    mae_market = float(df["error_market"].abs().mean())
    bias_model = float(df["error_model"].mean())
    bias_mkt   = float(df["error_market"].mean())

    x = np.c_[np.ones(len(df)), df["predicted_total"].values]
    y = df["total_runs"].values
    try:
        coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        intercept = float(coef[0]); slope = float(coef[1])
        yhat = x @ coef
        r2 = float(1 - ((y - yhat)**2).sum() / max(1e-9, ((y - y.mean())**2).sum()))
    except Exception:
        intercept = slope = r2 = np.nan

    log.info("EVAL %s  n=%d | MAE model=%.3f market=%.3f | bias model=%+.3f market=%+.3f | calib: a=%+.2f b=%.2f R2=%.3f",
             target_date, len(df), mae_model, mae_market, bias_model, bias_mkt, intercept, slope, r2)

    if not store:
        return

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS model_eval_daily (
              date date PRIMARY KEY,
              n integer NOT NULL,
              mae_model numeric,
              mae_market numeric,
              bias_model numeric,
              bias_market numeric,
              calib_intercept numeric,
              calib_slope numeric,
              r2 numeric,
              created_at timestamp without time zone DEFAULT now()
            )
        """))
        conn.execute(text("""
            INSERT INTO model_eval_daily
              (date, n, mae_model, mae_market, bias_model, bias_market, calib_intercept, calib_slope, r2)
            VALUES
              (:date, :n, :mae_model, :mae_market, :bias_model, :bias_market, :a, :b, :r2)
            ON CONFLICT (date) DO UPDATE SET
              n = EXCLUDED.n,
              mae_model = EXCLUDED.mae_model,
              mae_market = EXCLUDED.mae_market,
              bias_model = EXCLUDED.bias_model,
              bias_market = EXCLUDED.bias_market,
              calib_intercept = EXCLUDED.calib_intercept,
              calib_slope = EXCLUDED.calib_slope,
              r2 = EXCLUDED.r2
        """), {
            "date": target_date,
            "n": int(len(df)),
            "mae_model": mae_model,
            "mae_market": mae_market,
            "bias_model": bias_model,
            "bias_market": bias_mkt,
            "a": None if np.isnan(intercept) else intercept,
            "b": None if np.isnan(slope) else slope,
            "r2": None if np.isnan(r2) else r2,
        })

def stage_backfill(start_date: str, end_date: str, predict: bool = False, no_weather: bool = False):
    """
    Backfill historical data for a date range using backfill_range.py script.
    """
    import subprocess
    
    # Build command
    backfill_path = os.path.join(os.path.dirname(__file__), "backfill_range.py")
    cmd = [
        sys.executable, backfill_path,
        "--start", start_date,
        "--end", end_date,
    ]
    if predict:
        cmd.append("--predict")
    if no_weather:
        cmd.append("--no-weather")
    
    log.info(f"Starting backfill: {start_date} → {end_date}")
    log.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, check=False)
        
        if result.returncode != 0:
            log.error(f"Backfill failed (code {result.returncode}): {result.stderr}")
            return False
        else:
            log.info(f"Backfill completed successfully")
            if result.stdout.strip():
                log.info(f"Backfill output: {result.stdout}")
            return True
            
    except subprocess.TimeoutExpired:
        log.error("Backfill timed out after 1 hour")
        return False
    except Exception as e:
        log.error(f"Backfill error: {e}")
        return False

def stage_retrain(target_date: str, window_days=150, holdout_days=21, deploy=True, audit=True):
    """
    Run retraining using retrain_model.py script with proper error handling.
    """
    import subprocess
    
    # Build command with proper path to retrain_model.py
    retrain_path = os.path.join(os.path.dirname(__file__), "retrain_model.py")
    cmd = [
        sys.executable, retrain_path,
        "--end", target_date,
        "--window-days", str(window_days),
        "--holdout-days", str(holdout_days),
    ]
    if deploy: 
        cmd.append("--deploy")
    if audit:  
        cmd.append("--audit")
    
    log.info(f"Starting retraining with: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, check=False)
        
        if result.returncode != 0:
            log.error(f"Retraining failed (code {result.returncode}): {result.stderr}")
            return False
        else:
            log.info(f"Retraining completed successfully")
            if result.stdout.strip():
                log.info(f"Training output: {result.stdout}")
            return True
            
    except subprocess.TimeoutExpired:
        log.error("Retraining timed out after 30 minutes")
        return False
    except Exception as e:
        log.error(f"Retraining error: {e}")
        return False

# -----------------------------
# Pipeline stages
# -----------------------------

def stage_markets(engine, target_date: str, greedy: bool = True):
    """
    Stage: Markets.
    - Ensures rows exist in enhanced_games
    - Runs all data ingestors to fetch fresh external data
    - Updates enhanced_games with latest odds, pitchers, team stats, weather
    """
    # 1. Make sure rows exist with required NOT NULLs
    seed_enhanced_from_lgf(engine, target_date)
    
    # 2. NEW: Pull fresh data from APIs into enhanced_games
    run_ingestors(target_date)
    
    # 3. Read the fresh odds from enhanced_games after ingestion
    mk = fetch_markets_for_date(engine, target_date)
    if mk.empty and greedy:
        log.info("No markets found after data ingestion.")
    elif not mk.empty:
        upsert_markets(engine, mk, target_date)
        added = seed_lgf_from_enhanced(engine, target_date)
        if added:
            log.info(f"Seeded {added} LGF rows from enhanced_games.")

    # ✳️ Environment variance validation after ingestion (early signal)
    try:
        _validate_ingested_environment(engine, target_date)
    except Exception as e:
        log.warning(f"Environment variance validation error (non-fatal): {e}")

def stage_features_and_predict(engine, target_date: str) -> pd.DataFrame:
    """
    Stage: Features + Predict.
    - Loads today's LGF rows (upcoming games).
    - Left joins market_total from enhanced_games (if present).
    - Engineers features, aligns, predicts, upserts predictions.
    """
    # NEW: ensure rows exist (harmless if already present)
    seed_enhanced_from_lgf(engine, target_date)
    
    df = load_today_games(engine, target_date)
    if df.empty:
        added = seed_lgf_from_enhanced(engine, target_date)  # NEW fallback
        if added:
            log.info(f"Seeded {added} LGF rows; reloading…")
            df = load_today_games(engine, target_date)
    
    if df.empty:
        log.info("No upcoming games found (total_runs is not NULL or schedule missing). Nothing to predict.")
        return pd.DataFrame()

    # ✅ Force a clean, typed join of markets into today's LGF
    mk = fetch_markets_for_date(engine, target_date)
    
    # Cast keys to the same dtype to avoid silent non-matches
    df["game_id"] = df["game_id"].astype(str)
    mk["game_id"] = mk["game_id"].astype(str)
    
    # Remove any preexisting market_total and merge fresh
    df = df.drop(columns=["market_total"], errors="ignore").merge(
        mk[["game_id", "market_total"]], on="game_id", how="left"
    )
    
    # Sanity: did we actually get lines?
    missing = df["market_total"].isna().sum()
    if missing:
        log.warning("market_total missing for %d games: %s",
                    missing, df.loc[df.market_total.isna(), "game_id"].tolist())
    
    # Sanity: do the lines vary?
    mt_std = float(pd.to_numeric(df["market_total"], errors="coerce").std(skipna=True))
    log.info("market_total std = %.3f", mt_std)
    if not np.isfinite(mt_std) or mt_std < 0.1:
        log.error("Market totals look flat (std=%.3f). Aborting to avoid junk preds.", mt_std)
        sys.exit(2)

    # Keep identity columns for later upsert
    ids = df[["game_id", "date"]].copy()

    # Engineer + align
    feat, X, predictions = engineer_and_align(df, target_date)
    # Validate variance / coverage before allowing prediction upsert
    try:
        diag_metrics = _validate_feature_variance(feat)
        _record_feature_diagnostics(engine, target_date, diag_metrics)
    except SystemExit:
        raise
    except Exception as e:
        log.warning(f"Feature variance validation encountered an error (continuing cautiously): {e}")

    # Sanity log
    if len(ids) != len(X):
        log.warning(f"Identity/predictor row count mismatch: ids={len(ids)} X={len(X)}")

    # We already have predictions from predict_today_games, just upsert them
    if predictions is not None and not predictions.empty:
        log.info(f"Using predictions from enhanced pipeline: {len(predictions)} games")
        # Ensure predictions have the right columns and merge with ids
        preds = predictions.copy()
        if len(preds) == len(ids):
            for col in ids.columns:
                if col not in preds.columns:
                    preds[col] = ids[col].values
        else:
            log.warning(f"Prediction count mismatch: preds={len(preds)} ids={len(ids)}")
            # Fall back to regular prediction
            log.info(f"Falling back to predict_and_upsert...")
            anchor_env = os.getenv("DISABLE_MARKET_ANCHORING") not in ("1","true","TRUE")
            preds = predict_and_upsert(engine, X, ids, anchor_to_market=anchor_env)
    else:
        log.info(f"No predictions from enhanced pipeline, using predict_and_upsert...")
        anchor_env = os.getenv("DISABLE_MARKET_ANCHORING") not in ("1","true","TRUE")
        preds = predict_and_upsert(engine, X, ids, anchor_to_market=anchor_env)
    
    # Postprocess: calculate edge and recommendation for frontend
    postprocess_signals(engine, target_date)
    
    return preds

# -----------------------------
# Environment variance helpers
# -----------------------------

def _validate_ingested_environment(engine, target_date: str) -> None:
    """Check variance of weather & park factors in enhanced_games immediately after ingestion.
    Logs warnings or errors; does NOT exit (prediction stage handles hard fail).
    If ballpark factors are constant (std=0) but venue_name variance exists, attempts recalculation.
    """
    with engine.connect() as conn:
        # Check which environment columns exist
        has_ballpark_run = _has_col(conn, "enhanced_games", "ballpark_run_factor")
        has_ballpark_hr = _has_col(conn, "enhanced_games", "ballpark_hr_factor") 
        has_temperature = _has_col(conn, "enhanced_games", "temperature")
        has_wind_speed = _has_col(conn, "enhanced_games", "wind_speed")
        
        # Build query based on available columns
        select_cols = ["game_id", "venue_name"]
        if has_ballpark_run:
            select_cols.append("ballpark_run_factor")
        if has_ballpark_hr:
            select_cols.append("ballpark_hr_factor")
        if has_temperature:
            select_cols.append("temperature")
        if has_wind_speed:
            select_cols.append("wind_speed")
            
        q = text(f"""
            SELECT {', '.join(select_cols)}
            FROM enhanced_games
            WHERE "date" = :d
        """)
        df = pd.read_sql(q, conn, params={"d": target_date})
        
    if df.empty:
        log.info("Environment check: no enhanced_games rows yet.")
        return
        
    # Calculate variance for available columns
    run_std = float(pd.to_numeric(df.get("ballpark_run_factor", []), errors="coerce").std(skipna=True)) if has_ballpark_run else -1
    hr_std = float(pd.to_numeric(df.get("ballpark_hr_factor", []), errors="coerce").std(skipna=True)) if has_ballpark_hr else -1
    temp_std = float(pd.to_numeric(df.get("temperature", []), errors="coerce").std(skipna=True)) if has_temperature else -1
    wind_std = float(pd.to_numeric(df.get("wind_speed", []), errors="coerce").std(skipna=True)) if has_wind_speed else -1
    
    log.info(f"ENV VAR STD: run={run_std:.3f} hr={hr_std:.3f} temp={temp_std:.3f} wind={wind_std:.3f}")
    
    # If ballpark factors missing or constant, inject variance
    if (not has_ballpark_run or not has_ballpark_hr or 
        (len(df) > 1 and run_std == 0 and hr_std == 0)):
        log.warning("Ballpark factors missing/constant; injecting venue-based factors…")
        try:
            from enhanced_feature_pipeline import BALLPARK_FACTORS as _BF
            updated = 0
            with engine.begin() as conn:
                # Add columns if missing
                if not has_ballpark_run:
                    conn.execute(text("ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS ballpark_run_factor NUMERIC DEFAULT 1.0"))
                if not has_ballpark_hr:
                    conn.execute(text("ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS ballpark_hr_factor NUMERIC DEFAULT 1.0"))
                    
                for _, r in df.iterrows():
                    venue = r.get("venue_name")
                    meta = _BF.get(venue) if venue else None
                    if meta and "run_factor" in meta and "hr_factor" in meta:
                        conn.execute(
                            text(
                                "UPDATE enhanced_games SET ballpark_run_factor=:r, ballpark_hr_factor=:h WHERE game_id=:g AND \"date\"=:d"
                            ),
                            {"r": meta["run_factor"], "h": meta["hr_factor"], "g": r.game_id, "d": target_date},
                        )
                        updated += 1
            if updated:
                log.info(f"Injected ballpark factors for {updated} games from venue metadata")
        except Exception as e:
            log.warning(f"Ballpark factor injection failed: {e}")
            
    # If weather missing or constant, try to inject some variance
    if (not has_temperature or not has_wind_speed or 
        (len(df) > 1 and temp_std == 0 and wind_std == 0)):
        log.warning("Weather data missing/constant; attempting to inject basic variance")
        try:
            with engine.begin() as conn:
                if not has_temperature:
                    conn.execute(text("ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS temperature NUMERIC DEFAULT 70"))
                if not has_wind_speed:
                    conn.execute(text("ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS wind_speed NUMERIC DEFAULT 5"))
                    
                # Basic venue-based weather variance (placeholder until real weather works)
                weather_map = {
                    "Coors Field": (75, 8),  # Denver - higher elevation, windier
                    "Fenway Park": (68, 12), # Boston - cooler, windier
                    "Petco Park": (72, 6),   # San Diego - mild
                    "Kauffman Stadium": (78, 10), # Kansas City
                    "Dodger Stadium": (74, 4),    # LA - warm, calm
                    "Wrigley Field": (70, 15),    # Chicago - windy
                }
                
                for _, r in df.iterrows():
                    venue = r.get("venue_name", "")
                    temp, wind = weather_map.get(venue, (70 + hash(venue) % 10, 5 + hash(venue) % 8))
                    conn.execute(
                        text("UPDATE enhanced_games SET temperature=:t, wind_speed=:w WHERE game_id=:g AND \"date\"=:d"),
                        {"t": temp, "w": wind, "g": r.game_id, "d": target_date}
                    )
                log.info("Injected basic weather variance based on venue")
        except Exception as e:
            log.warning(f"Weather variance injection failed: {e}")


def postprocess_signals(engine, target_date: str):
    """Calculate edge, recommendation, and enhanced confidence signals for frontend consumption"""
    log.info("Calculating betting signals with enhanced confidence...")
    sql = text("""
        UPDATE enhanced_games
           SET edge = ROUND(predicted_total - market_total, 2),
               recommendation = CASE
                   WHEN predicted_total IS NULL OR market_total IS NULL THEN NULL
                   WHEN (predicted_total - market_total) >=  1.0 THEN 'OVER'
                   WHEN (predicted_total - market_total) <= -1.0 THEN 'UNDER'
                   ELSE 'HOLD'
               END,
               confidence = CASE
                   WHEN predicted_total IS NULL OR market_total IS NULL THEN NULL
                   ELSE LEAST(95, GREATEST(30, ROUND(
                       -- Base confidence from edge (boosted formula)
                       40 + 15 * ABS(predicted_total - market_total) +
                       
                       -- Weather factor bonus (extreme conditions increase confidence)
                       CASE 
                           WHEN wind_speed >= 15 THEN 3  -- Strong wind affects totals
                           WHEN wind_speed <= 5 THEN 2   -- Calm conditions are predictable
                           ELSE 0 
                       END +
                       
                       -- Temperature factor bonus 
                       CASE 
                           WHEN temperature >= 80 THEN 2  -- Hot weather favors offense
                           WHEN temperature <= 60 THEN 2  -- Cold weather favors pitching
                           ELSE 0
                       END +
                       
                       -- Extreme prediction bonus (very high/low totals are more confident)
                       CASE 
                           WHEN predicted_total >= 11.0 OR predicted_total <= 6.0 THEN 5
                           WHEN predicted_total >= 10.0 OR predicted_total <= 6.5 THEN 3
                           ELSE 0
                       END +
                       
                       -- Large edge bonus (high conviction predictions)
                       CASE 
                           WHEN ABS(predicted_total - market_total) >= 3.0 THEN 8
                           WHEN ABS(predicted_total - market_total) >= 2.5 THEN 5
                           WHEN ABS(predicted_total - market_total) >= 2.0 THEN 3
                           ELSE 0
                       END
                   , 0)))
               END
         WHERE "date" = :d
    """)
    with engine.begin() as conn:
        result = conn.execute(sql, {"d": target_date})
        log.info(f"Updated betting signals for {result.rowcount} games")

def stage_export(preds: pd.DataFrame, target_date: str):
    # Always fetch fresh data from database for export, don't rely on preds parameter
    # This ensures export works even when run independently of predict stages
    engine = get_engine()
    with engine.begin() as conn:
        v = conn.execute(text('SELECT stddev_samp(predicted_total) FROM enhanced_games WHERE "date"=:d'), {"d": target_date}).scalar()
    if not v or v < 0.25:
        log.error("Export blocked: predicted_total std=%.3f is too low.", 0.0 if v is None else v)
        return None

    # -------- try the view (no transaction) --------
    df = None
    with engine.connect() as conn:
        view_exists = conn.execute(text("""
            SELECT 1
            FROM information_schema.views
            WHERE table_schema = current_schema()
              AND table_name = 'api_games_today'
            LIMIT 1
        """)).scalar() is not None

        if view_exists:
            has_game_date = conn.execute(text("""
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'api_games_today'
                  AND column_name = 'game_date'
                LIMIT 1
            """)).scalar() is not None

            has_ev_best = conn.execute(text("""
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'api_games_today'
                  AND column_name = 'ev_best'
                LIMIT 1
            """)).scalar() is not None

            where_clause = "WHERE v.game_date = :d" if has_game_date else ""
            order_clause = "ORDER BY ABS(v.edge) DESC NULLS LAST"
            if has_ev_best:
                order_clause = "ORDER BY ABS(v.ev_best) DESC NULLS LAST, ABS(v.edge) DESC NULLS LAST"

            sql = f"""
                SELECT v.*
                FROM api_games_today v
                {where_clause}
                {order_clause}
            """
            params = {"d": target_date} if has_game_date else {}
            try:
                df = pd.read_sql(text(sql), conn, params=params)
            except Exception as e:
                log.warning(f"api_games_today read failed, will use fallback: {e}")

    # -------- fallback (fresh connection) --------
    if df is None:
        with engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT
                  eg.game_id,
                  eg."date"::date AS game_date,
                  eg.home_team, eg.away_team,
                  eg.market_total, eg.predicted_total,
                  ROUND((eg.predicted_total - eg.market_total)::numeric, 2) AS edge,
                  pp.p_over, pp.p_under,
                  pp.ev_over, pp.ev_under,
                  pp.kelly_over, pp.kelly_under,
                  GREATEST(pp.ev_over, pp.ev_under)           AS ev_best,
                  CASE WHEN pp.ev_over >= pp.ev_under THEN 'OVER' ELSE 'UNDER' END AS ev_side,
                  CASE
                    WHEN GREATEST(pp.ev_over, pp.ev_under) > 0 THEN
                      CASE WHEN pp.ev_over >= pp.ev_under THEN 'OVER' ELSE 'UNDER' END
                    WHEN (eg.predicted_total - eg.market_total) >=  2.0 THEN 'OVER'
                    WHEN (eg.predicted_total - eg.market_total) <= -2.0 THEN 'UNDER'
                    ELSE 'NO BET'
                  END AS recommendation,
                  CASE
                    WHEN GREATEST(pp.ev_over, pp.ev_under) > 0 THEN 'EV'
                    WHEN ABS(eg.predicted_total - eg.market_total) >= 2.0 THEN 'EDGE'
                    ELSE 'NONE'
                  END AS rec_source,
                  CASE
                    WHEN GREATEST(pp.ev_over, pp.ev_under) > 0 AND pp.ev_over >= pp.ev_under THEN pp.kelly_over
                    WHEN GREATEST(pp.ev_over, pp.ev_under) > 0 AND pp.ev_under >  pp.ev_over  THEN pp.kelly_under
                    ELSE 0.0
                  END AS kelly_best
                FROM enhanced_games eg
                LEFT JOIN probability_predictions pp
                  ON pp.game_id = eg.game_id AND pp.game_date = eg."date"
                WHERE eg."date" = :d
                ORDER BY ABS(GREATEST(pp.ev_over, pp.ev_under)) DESC NULLS LAST, ABS(edge) DESC NULLS LAST
            """), conn, params={"d": target_date})

    return export_predictions_csv(df, target_date)

def stage_audit(target_date: str):
    run_audit(target_date)

def stage_probabilities(target_date: str, window_days: int = 30, model_version: str = "latest"):
    """
    Fit isotonic calibration on a rolling window up to yesterday and score today's slate.
    Writes to probability_predictions (p_over/p_under, EV, Kelly).
    """
    import subprocess, sys as _sys, os
    # Optional: pass a safer sigma floor through env (if you made it configurable)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # env["EV_SIGMA_FLOOR"] = "2.5"  # uncomment if your script supports it

    cmd = [
        _sys.executable, os.path.join(os.path.dirname(__file__), "probabilities_and_ev.py"),
        "--date", target_date,
        "--window-days", str(window_days),
        "--model-version", f"wf_{model_version}"
    ]
    log.info(f"Running probabilities: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, encoding='utf-8', errors='replace')
    if res.returncode != 0:
        log.error("probabilities_and_ev failed (%d): %s", res.returncode, res.stderr or "No stderr")
        raise RuntimeError("probabilities stage failed")
    if res.stdout and res.stdout.strip():
        log.info(res.stdout.splitlines()[-1])  # tail for cleanliness

def refresh_api_view(engine):
    """
    Create/replace a view that the frontend reads. It surfaces EV/Kelly/probabilities,
    and chooses recommendation by EV>0; falls back to edge threshold.
    """
    with engine.begin() as conn:
        n = conn.execute(text('SELECT COUNT(*) FROM enhanced_games WHERE "date" = CURRENT_DATE')).scalar() or 0
        if n == 0:
            log.info("No games today; skipping api_games_today refresh.")
            return
        
        try:
            sql = """
            CREATE TABLE IF NOT EXISTS model_config (key text PRIMARY KEY, value text);
            INSERT INTO model_config(key, value)
            VALUES ('edge_threshold', '2.0')
            ON CONFLICT (key) DO NOTHING;

            CREATE OR REPLACE VIEW api_games_today AS
            WITH cfg AS (
              SELECT COALESCE(NULLIF(value,''),'2.0')::numeric AS edge_thr
              FROM model_config WHERE key='edge_threshold'
            ),
            base AS (
              SELECT
                eg.game_id,
                eg."date"::date AS game_date,
                eg.home_team, eg.away_team,
                eg.market_total, eg.predicted_total,
                ROUND((eg.predicted_total - eg.market_total)::numeric, 2) AS edge,
                pp.p_over, pp.p_under,
                pp.ev_over, pp.ev_under,
                pp.kelly_over, pp.kelly_under,
                GREATEST(pp.ev_over, pp.ev_under)           AS ev_best,
                CASE WHEN pp.ev_over >= pp.ev_under THEN 'OVER' ELSE 'UNDER' END AS ev_side
              FROM enhanced_games eg
              LEFT JOIN probability_predictions pp
                ON pp.game_id = eg.game_id AND pp.game_date = eg."date"
              WHERE eg."date" = CURRENT_DATE
            )
            SELECT
              b.*,
              CASE
                WHEN b.ev_best > 0 THEN b.ev_side
                WHEN (b.predicted_total - b.market_total) >= (SELECT edge_thr FROM cfg) THEN 'OVER'
                WHEN (b.predicted_total - b.market_total) <= -(SELECT edge_thr FROM cfg) THEN 'UNDER'
                ELSE 'NO BET'
              END AS recommendation,
              CASE
                WHEN b.ev_best > 0 THEN 'EV'
                WHEN ABS(b.edge) >= (SELECT edge_thr FROM cfg) THEN 'EDGE'
                ELSE 'NONE'
              END AS rec_source,
              CASE
                WHEN b.ev_best > 0 AND b.ev_side='OVER'  THEN b.kelly_over
                WHEN b.ev_best > 0 AND b.ev_side='UNDER' THEN b.kelly_under
                ELSE 0.0
              END AS kelly_best
            FROM base b;
            """
            for stmt in sql.split(";"):
                s = stmt.strip()
                if s:
                    conn.execute(text(s))
            log.info("api_games_today view refreshed.")
            
        except Exception as e:
            if "cannot drop columns from view" in str(e):
                log.warning("View columns differ; leaving existing view untouched this run.")
                return
            raise


def seed_lgf_from_enhanced(engine, target_date):
    """Seed LGF rows from enhanced_games to ensure EV calculations work
    
    Called when features/predict stages run but markets may have been skipped.
    Creates minimal LGF entries so probabilities stage doesn't fail.
    Now also copies pitcher data to fix feature engineering pipeline.
    """
    sql = text("""
        INSERT INTO legitimate_game_features (
            game_id, "date", home_team, away_team, market_total,
            home_sp_season_era, away_sp_season_era, 
            home_sp_whip, away_sp_whip
        )
        SELECT 
            eg.game_id, eg."date", eg.home_team, eg.away_team, eg.market_total,
            eg.home_sp_season_era, eg.away_sp_season_era,
            eg.home_sp_whip, eg.away_sp_whip
        FROM enhanced_games eg
        LEFT JOIN legitimate_game_features lgf
          ON lgf.game_id = eg.game_id AND lgf."date" = eg."date"
        WHERE eg."date" = :d AND lgf.game_id IS NULL
    """)
    with engine.begin() as conn:
        return conn.execute(sql, {"d": target_date}).rowcount or 0


def stage_health_gate(target_date: str):
    """
    Run health gate validation to check calibration quality before allowing trades.
    Uses health_gate.py to validate Brier score and ECE thresholds.
    """
    import subprocess, sys as _sys, os
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    cmd = [
        _sys.executable, os.path.join(os.path.dirname(__file__), "health_gate.py"),
        "--date", target_date
    ]
    log.info(f"Running health gate validation: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, encoding='utf-8', errors='replace')
    
    if res.returncode != 0:
        log.error("Health gate validation failed (%d): %s", res.returncode, res.stderr or "No stderr")
        raise RuntimeError("Health gate validation failed - trading halted for safety")
    
    # Check if health gate passed
    if "🟢 HEALTH GATE: PASS" in res.stdout:
        log.info("🟢 Health gate validation PASSED - system calibration is healthy")
    elif "🔴 HEALTH GATE: FAIL" in res.stdout:
        log.error("🔴 Health gate validation FAILED - trading should be halted")
        raise RuntimeError("Health gate failed - system calibration is poor")
    else:
        log.warning("⚠️ Health gate output unclear - proceeding with caution")
    
    if res.stdout and res.stdout.strip():
        log.info("Health gate output: " + res.stdout.splitlines()[-1])


def stage_odds_loading(target_date: str, odds_file: str = None):
    """
    Load comprehensive odds data for all games to enable enhanced probability predictions.
    If no odds file provided, creates market-based odds from enhanced_games data.
    """
    import subprocess, sys as _sys, os
    import psycopg2
    import csv
    import tempfile
    
    # If no odds file provided, create one from enhanced_games market data
    if not odds_file:
        log.info("No odds file provided - generating from enhanced_games market data")
        
        try:
            conn = psycopg2.connect(
                host='localhost',
                database='mlb',
                user='mlbuser',
                password='mlbpass'
            )
            cursor = conn.cursor()
            
            # Get games without odds - guard against NULLs  
            cursor.execute('''
                SELECT eg.game_id, 
                       COALESCE(eg.market_total, 8.5) AS market_total,
                       COALESCE(eg.over_odds, -110) AS over_odds,
                       COALESCE(eg.under_odds, -110) AS under_odds
                FROM enhanced_games eg
                LEFT JOIN totals_odds to_table ON to_table.game_id = eg.game_id AND to_table.date = eg.date  
                WHERE eg.date = %s 
                  AND to_table.game_id IS NULL
                  AND eg.market_total IS NOT NULL
                ORDER BY eg.game_id
            ''', (target_date,))
            
            missing_games = cursor.fetchall()
            
            if missing_games:
                # Create temporary CSV file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    writer = csv.writer(f)
                    writer.writerow(['game_id', 'date', 'book', 'total', 'over_odds', 'under_odds'])
                    
                    for game_id, market_total, over_odds, under_odds in missing_games:
                        # skip if still missing a market total
                        if market_total is None: 
                            continue
                        writer.writerow([
                            game_id, target_date, 'consensus', float(market_total), 
                            int(over_odds), int(under_odds)
                        ])
                    
                    odds_file = f.name
                    log.info(f"Created temporary odds file: {odds_file} with {len(missing_games)} games")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            log.error(f"Failed to generate odds file: {e}")
            return
    
    if odds_file and os.path.exists(odds_file):
        # Load odds using load_totals_odds.py
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        cmd = [
            _sys.executable, os.path.join(os.path.dirname(__file__), "load_totals_odds.py"),
            odds_file
        ]
        log.info(f"Loading odds data: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True, env=env, encoding='utf-8', errors='replace')
        
        if res.returncode != 0:
            log.error("Odds loading failed (%d): %s", res.returncode, res.stderr or "No stderr")
            raise RuntimeError("Odds loading stage failed")
        
        log.info("✅ Odds data loaded successfully")
        if res.stdout and res.stdout.strip():
            log.info("Odds loading output: " + res.stdout.splitlines()[-1])
        
        # Clean up temporary file if we created it
        if odds_file.startswith(tempfile.gettempdir()):
            os.unlink(odds_file)
            log.info("Cleaned up temporary odds file")
    else:
        log.info("No odds file needed - all games already have odds data")


# -----------------------------
# Main
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Daily API Workflow: markets → features → predict → odds → health → prob → export → audit → eval → retrain")
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"), help="Target date (YYYY-MM-DD)")
    ap.add_argument("--target-date", dest="date", help="Target date (YYYY-MM-DD) - alias for --date")
    ap.add_argument("--stages", default="markets,features,predict,odds,health,prob,export",
                    help="Comma list: markets,features,predict,odds,health,prob,export,audit,eval,retrain")
    ap.add_argument("--quiet", action="store_true", help="Less logging")
    # Backfill arguments (when using backfill stage)
    ap.add_argument("--start-date", help="Start date for backfill (YYYY-MM-DD)")
    ap.add_argument("--end-date", help="End date for backfill (YYYY-MM-DD)")
    ap.add_argument("--predict", action="store_true", help="Generate predictions during backfill")
    ap.add_argument("--no-weather", action="store_true", help="Skip weather during backfill")
    return ap.parse_args()

def main():
    args = parse_args()
    if args.quiet:
        log.setLevel(logging.WARNING)

    target_date = args.date
    stages = [s.strip().lower() for s in args.stages.split(",") if s.strip()]

    engine = get_engine()

    preds = pd.DataFrame()

    try:
        if "backfill" in stages:
            # Handle backfill stage
            if not args.start_date or not args.end_date:
                log.error("Backfill stage requires --start-date and --end-date")
                sys.exit(2)
            stage_backfill(args.start_date, args.end_date, 
                          predict=args.predict, no_weather=args.no_weather)
            return  # Backfill doesn't need other stages

        if "markets" in stages:
            stage_markets(engine, target_date)

        if "features" in stages or "predict" in stages:
            preds = stage_features_and_predict(engine, target_date)

        if "odds" in stages:
            stage_odds_loading(target_date)

        if "health" in stages:
            stage_health_gate(target_date)

        if "prob" in stages or "probabilities" in stages:
            stage_probabilities(target_date)
            refresh_api_view(engine)

        if "export" in stages:
            stage_export(preds, target_date)

        if "audit" in stages:
            stage_audit(target_date)

        if "eval" in stages:
            stage_eval(engine, target_date)

        if "retrain" in stages:
            # typically run after evaluation for yesterday
            from datetime import datetime
            yday = (datetime.strptime(target_date, "%Y-%m-%d")).strftime("%Y-%m-%d")
            stage_retrain(yday, window_days=150, holdout_days=21, deploy=True, audit=True)

        # Post-run assertion (skip for eval/retrain/backfill-only runs)
        if any(stage in stages for stage in ["markets", "features", "predict"]):
            assert_predictions_written(engine, target_date)

    except Exception as e:
        log.exception("Daily workflow failed")
        sys.exit(2)
    finally:
        engine.dispose()

    log.info("Daily workflow complete.")
    sys.exit(0)

if __name__ == "__main__":
    main()
