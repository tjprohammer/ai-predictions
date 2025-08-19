#!/usr/bin/env python3
"""
Training Bundle Audit
====================
Comprehensive audit of the model bundle to identify systematic bias and feature drift
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
from sqlalchemy import create_engine, text
import argparse
from datetime import datetime, timedelta
import json
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
EPS = 1e-5  # Tolerance for "constant" features
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb"
)

def safe_align(df, feature_columns, fill_values=None):
    """Return a model-ready DataFrame, even if upstream alignment fails."""
    X = df.copy()

    # ensure all training columns exist
    for c in feature_columns:
        if c not in X.columns:
            X[c] = np.nan

    # order columns exactly as training
    X = X[feature_columns]

    # numeric coerce + inf -> NaN
    X = X.apply(pd.to_numeric, errors="coerce")
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # training-time fill values (bundle > predictor)
    if isinstance(fill_values, dict) and fill_values:
        for c, v in fill_values.items():
            if c in X.columns:
                X[c] = X[c].fillna(v)

    # per-column medians (Series) then final zero
    med = X.median(numeric_only=True)
    if isinstance(med, pd.Series):
        X = X.fillna(med)
    X = X.fillna(0)

    return X

def _fetch_truth_df(engine, start_date):
    """Robust truth data fetching with multiple fallback strategies"""
    attempts = []

    # 1) Strict join but with date cast (fixes timestamp vs date)
    attempts.append((
        "join on (game_id, date::date)",
        text("""
        SELECT eg."date"::date AS date, eg.game_id, eg.predicted_total, eg.market_total, lgf.total_runs
        FROM enhanced_games eg
        JOIN legitimate_game_features lgf
          ON lgf.game_id = eg.game_id
         AND lgf."date"   = eg."date"::date
        WHERE eg.predicted_total IS NOT NULL
          AND lgf.total_runs IS NOT NULL
          AND eg."date"::date >= :start_date
          AND eg.market_total BETWEEN 5 AND 15
        ORDER BY eg."date" DESC
        LIMIT 1000
        """)
    ))

    # 2) Join on game_id only (handles rare date drift)
    attempts.append((
        "join on game_id only",
        text("""
        SELECT eg."date"::date AS date, eg.game_id, eg.predicted_total, eg.market_total, lgf.total_runs
        FROM enhanced_games eg
        JOIN legitimate_game_features lgf
          ON lgf.game_id = eg.game_id
        WHERE eg.predicted_total IS NOT NULL
          AND lgf.total_runs IS NOT NULL
          AND eg."date"::date >= :start_date
          AND eg.market_total BETWEEN 5 AND 15
        ORDER BY eg."date" DESC
        LIMIT 1000
        """)
    ))

    # 3) Relaxed date constraints (game_id + close dates)
    attempts.append((
        "relaxed date matching",
        text("""
        SELECT eg."date"::date AS date, eg.game_id, eg.predicted_total, eg.market_total, lgf.total_runs
        FROM enhanced_games eg
        JOIN legitimate_game_features lgf
          ON lgf.game_id = eg.game_id
         AND ABS(EXTRACT(EPOCH FROM (lgf."date" - eg."date"::date))/86400) <= 1
        WHERE eg.predicted_total IS NOT NULL
          AND lgf.total_runs IS NOT NULL
          AND eg."date"::date >= :start_date
          AND eg.market_total BETWEEN 5 AND 15
        ORDER BY eg."date" DESC
        LIMIT 1000
        """)
    ))

    for label, q in attempts:
        try:
            df = pd.read_sql(q, engine, params={"start_date": start_date})
            logger.info(f"Truth bias attempt [{label}]: {len(df)} rows")
            if len(df) >= 30:  # enough to be meaningful
                return df
        except Exception as e:
            logger.warning(f"Truth bias attempt [{label}] failed: {e}")

    # Diagnostics if all attempts are tiny
    try:
        diag = pd.read_sql(text("""
            SELECT eg.game_id, eg."date" AS eg_date, eg.predicted_total,
                   lgf."date" AS lgf_date, lgf.total_runs
            FROM enhanced_games eg
            LEFT JOIN legitimate_game_features lgf
                   ON lgf.game_id = eg.game_id AND lgf."date" = eg."date"::date
            WHERE eg."date"::date >= :start_date
            ORDER BY eg."date" DESC
            LIMIT 50
        """), engine, params={"start_date": start_date})
        if not diag.empty:
            logger.info("Sample unmatched rows (predictions vs LGF):")
            logger.info("EG rows with predicted_total: %d", diag[diag['predicted_total'].notna()].shape[0])
            logger.info("LGF rows with total_runs: %d", diag[diag['total_runs'].notna()].shape[0])
            logger.info("Matched rows: %d", diag[(diag['predicted_total'].notna()) & (diag['total_runs'].notna())].shape[0])
            logger.info("Sample data:\n%s", diag.head(10).to_string(index=False))
    except Exception as e:
        logger.warning(f"Diagnostic query failed: {e}")
    
    return pd.DataFrame()


def audit_training_bundle(target_date=None, apply_bias_correction=False, db_url=None, model_path=None, enforce_metadata=False, market_days=5, truth_days=30, dry_run=False, serve_days=7, psi_bins=10, psi_min_serving=100, psi_min_training=1000):
    """Comprehensive audit of the training bundle"""
    
    logger.info("Audit script v2.2 - Production-ready edition")
    
    # Use provided db_url or default
    if db_url is None:
        db_url = DATABASE_URL
    
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info("=" * 80)
    logger.info("🔍 TRAINING BUNDLE AUDIT v2.2")
    logger.info("=" * 80)
    
    # 1) Inspect the model bundle
    logger.info("\n1️⃣ MODEL BUNDLE INSPECTION")
    logger.info("-" * 40)
    
    bundle_path = Path(model_path) if model_path else Path("../models/legitimate_model_latest.joblib")
    if not bundle_path.exists():
        logger.error(f"❌ Bundle not found: {bundle_path}")
        return
    
    try:
        b = joblib.load(bundle_path)
        logger.info(f"✅ Bundle loaded: {bundle_path}")
        logger.info(f"   Keys: {list(b.keys())}")
        logger.info(f"   Model: {type(b['model']).__name__}")
        logger.info(f"   N_features: {len(b['feature_columns'])}")
        logger.info(f"   Has_preproc: {'preproc' in b and b['preproc'] is not None}")
        logger.info(f"   Has_scaler: {'scaler' in b and b['scaler'] is not None}")
        logger.info(f"   Fill_values available: {len(b.get('feature_fill_values', {}))}")
        logger.info(f"   Aliases: {b.get('feature_aliases', 'None')}")
        
        if hasattr(b['model'], 'n_estimators'):
            logger.info(f"   Random Forest trees: {b['model'].n_estimators}")
        
        # Early scaler sanity check
        scaler = b.get('scaler')
        if scaler is not None and hasattr(scaler, 'n_features_in_'):
            if scaler.n_features_in_ != len(b['feature_columns']):
                logger.warning(f"⚠️  Scaler expects {scaler.n_features_in_} features but bundle has {len(b['feature_columns'])}")
        
        # Get snapshot early for reference stats
        snap = b.get("training_feature_snapshot")
            
    except Exception as e:
        logger.error(f"❌ Error loading bundle: {e}")
        return
    
    # Enhanced metadata check with empty DataFrame detection
    required_meta = ["label_definition", "evaluation_metrics", "training_period", "training_feature_snapshot"]
    
    missing_meta = []
    for k in required_meta:
        v = b.get(k)
        if v is None:
            missing_meta.append(k)
        elif isinstance(v, dict) and v == {}:
            missing_meta.append(k)
        elif k == "training_feature_snapshot" and isinstance(v, pd.DataFrame) and v.empty:
            missing_meta.append(k)
    
    if missing_meta:
        logger.warning(f"⚠️  Missing required training metadata: {missing_meta}")
        if enforce_metadata:
            logger.error("❌ METADATA ENFORCEMENT FAILED - Required metadata missing")
            raise SystemExit(3)
    else:
        logger.info("✅ All required training metadata present")
    
    # 2) Feature importance analysis with robust fallback 
    logger.info("\n2️⃣ FEATURE IMPORTANCE ANALYSIS")
    logger.info("-" * 40)
    
    # Define importances robustly (guard against length mismatch)
    fi_aligned = False  # Track if feature importance names align with data
    if hasattr(b['model'], "feature_importances_"):
        fi = b['model'].feature_importances_
        if len(fi) == len(b['feature_columns']):
            importances = pd.Series(fi, index=b['feature_columns'])
            fi_aligned = True  # Successful alignment
        else:
            logger.warning(f"Feature importances length mismatch: model={len(fi)}, columns={len(b['feature_columns'])}. Using anonymous names.")
            importances = pd.Series(fi, index=[f"feat_{i}" for i in range(len(fi))])
            fi_aligned = False  # Length mismatch = no alignment
        
        top_20 = importances.sort_values(ascending=False).head(20)
        logger.info("Top 20 feature importances:")
        for i, (feat, imp) in enumerate(top_20.items(), 1):
            logger.info(f"   {i:2d}. {feat:<25} {imp:.6f}")
            
        # Check for suspicious patterns (only if fi_aligned)
        if fi_aligned:
            suspicious = []
            whitelist = {"pitching_vs_offense"}  # Deterministically computed features
            for feat in top_20.index:
                low = feat.lower()
                if any(w in low for w in ["vs", "versus", "interaction"]) and feat not in whitelist:
                    suspicious.append(feat)
            
            if suspicious:
                logger.warning(f"⚠️  Suspicious features (may be hard to replicate at serving):")
                for feat in suspicious:
                    logger.warning(f"   - {feat} (importance: {importances[feat]:.6f})")
        else:
            logger.warning("⚠️  Feature importance names don't align - skipping name-based diagnostics")
    else:
        # Fallback: zeros so later .get() calls won't crash
        importances = pd.Series(0.0, index=b['feature_columns'])
        logger.warning("❌ Model has no feature_importances_ attribute")
        fi_aligned = False
    
    # 3) Schema drift analysis with production-like serving alignment
    logger.info("\n3️⃣ SCHEMA DRIFT ANALYSIS")
    logger.info("-" * 40)
    
    engine = None
    try:
        # Import with fallback
        try:
            from enhanced_bullpen_predictor import EnhancedBullpenPredictor
        except ImportError as e:
            logger.error(f"❌ Cannot import EnhancedBullpenPredictor: {e}")
            logger.error("   Ensure this script is run from the deployment directory")
            return
            
        predictor = EnhancedBullpenPredictor()
        engine = create_engine(db_url)
        
        # Get today's data to check what we actually produce
        query = text("""
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
            LEFT JOIN enhanced_games eg
              ON eg.game_id = lgf.game_id AND eg."date" = lgf."date"
            WHERE lgf."date" = :target_date AND lgf.total_runs IS NULL
        """)
        games_df = pd.read_sql(query, engine, params={'target_date': target_date})
        
        logger.info(f"📈 Loaded {len(games_df)} games for analysis from {target_date}")
        
        if len(games_df) > 0:
            # Use the coalesced values like in production
            games_df['market_total'] = games_df.pop('market_total_final')
            games_df['away_sp_season_era'] = games_df.pop('away_sp_season_era_final') 
            if 'home_sp_season_era_final' in games_df.columns:
                games_df['home_sp_season_era'] = games_df.pop('home_sp_season_era_final')
            games_df = games_df.loc[:, ~games_df.columns.duplicated(keep='last')]
            
            # Engineer features like in production serving - use predict_today_games to get real pitcher stats
            try:
                # Store target date in predictor for use by enhanced pipeline
                predictor._current_target_date = target_date
                
                # Use predict_today_games to get the same real pitcher data hydration as production
                predictions, featured_df, X = predictor.predict_today_games(target_date)
                logger.info(f"✅ Feature engineering completed: {featured_df.shape}")
                
                # If predict_today_games failed, fall back to manual engineering
                if featured_df is None:
                    featured_df = predictor.engineer_features(games_df)
                    logger.info(f"✅ Fallback feature engineering completed: {featured_df.shape}")
                    
            except Exception:
                logger.exception("predict_today_games failed; falling back to engineer_features")
                try:
                    featured_df = predictor.engineer_features(games_df)
                    logger.info(f"✅ Fallback feature engineering completed: {featured_df.shape}")
                except Exception:
                    logger.exception("engineer_features also failed; using raw columns")
                    featured_df = games_df.copy()   # <- keep going with raw features
            
            # BEFORE alignment: what's truly missing from serving?
            train_feats = set(b['feature_columns'])
            serving_raw = set(featured_df.columns)
            missing_pre_align = [f for f in b['feature_columns'] if f not in serving_raw]
            
            # Use the same serving alignment as production, but with a safe fallback
            try:
                X = predictor.align_serving_features(featured_df, strict=False)
                # guard: some implementations return arrays
                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X, columns=b['feature_columns'])
                logger.info("✅ align_serving_features completed successfully")
            except Exception as e:
                logger.exception("align_serving_features failed; using safe_align fallback")
                X = safe_align(
                    featured_df,
                    b['feature_columns'],
                    b.get('feature_fill_values') or getattr(predictor, 'fill_values', {}) or {}
                )
            
            # Debug logging for troubleshooting
            logger.info(f"type(X)={type(X)}, shape={getattr(X, 'shape', None)}")
            logger.info(f"X columns (first 10): {list(X.columns)[:10] if hasattr(X, 'columns') else 'n/a'}")
            
            # Shape validation after alignment
            if X.shape[1] != len(b['feature_columns']):
                logger.error(f"❌ Shape mismatch after alignment: X has {X.shape[1]} columns, expected {len(b['feature_columns'])}")
                logger.error(f"   This could cause model errors. Check feature alignment logic.")
            else:
                logger.info(f"✅ Shape validation passed: {X.shape[1]} columns match expected count")

            # Check for all-NaN features (sanity warning)
            all_nan_features = [c for c in X.columns if X[c].isna().all()]
            if all_nan_features:
                logger.warning(f"⚠️  All-NaN features detected: {len(all_nan_features)} features")
                logger.warning(f"   Sample all-NaN features: {all_nan_features[:5]}")
                
                # Check for critical team aggregates that should be computed
                critical_team_features = {'home_team_xwoba', 'away_team_xwoba', 'home_team_iso', 'away_team_iso',
                                        'home_team_avg', 'away_team_avg', 'combined_woba', 'combined_wrcplus'}
                missing_critical = [c for c in critical_team_features if c in all_nan_features]
                if missing_critical:
                    logger.warning(f"⚠️  Critical team aggregates missing: {missing_critical}")
                    logger.warning("   Consider computing proxies or adding team stats pipeline stage")
            
            # Define placeholder_only for compatibility
            placeholder_only = [c for c in b['feature_columns'] if c in X.columns and X[c].isna().all()]
            logger.info(
                "Placeholder-only (all-NaN before fill) features: %d; sample: %s",
                len(placeholder_only), placeholder_only[:5]
            )
            
            # Add range checks to catch corrupt slates fast
            def _range_check(df, col, lo, hi):
                if col in df.columns:
                    s = pd.to_numeric(df[col], errors="coerce")
                    bad = df[(s < lo) | (s > hi)]
                    if len(bad):
                        median_val = s.median()
                        logger.warning(
                            f"Out-of-range values in {col}: {len(bad)} rows "
                            f"(min={s.min():.3f}, max={s.max():.3f}, median={median_val:.3f})"
                        )
                        logger.warning(f"   Sample out-of-range rows: {bad.index[:3].tolist()}")
            
            for c, lo, hi in [
                ("home_sp_era", 0.5, 10.0),
                ("away_sp_era", 0.5, 10.0), 
                ("home_sp_whip", 0.6, 2.5),
                ("away_sp_whip", 0.6, 2.5),
                ("home_sp_k_per_9", 3.0, 15.0),
                ("away_sp_k_per_9", 3.0, 15.0),
                ("market_total", 4.0, 16.0),
            ]:
                _range_check(X, c, lo, hi)
            
            serving_feats = set(X.columns)
            
            logger.info(f"Training expects: {len(train_feats)} features")
            logger.info(f"Serving produces: {len(serving_feats)} features") 
            logger.info(f"Missing pre-align: {len(missing_pre_align)}")
            if missing_pre_align:
                logger.info(f"Missing (first 20): {missing_pre_align[:20]}")
            if placeholder_only:
                logger.warning(f"Placeholder-only features (first 20): {placeholder_only[:20]}")
            
            missing_at_serving = [f for f in train_feats if f not in serving_feats]
            extra_at_serving = [f for f in serving_feats if f not in train_feats]
            logger.info(f"Missing at serving (post-align): {len(missing_at_serving)}")
            logger.info(f"Extra at serving: {len(extra_at_serving)}")
            
            # Robust constant check helper
            def _const_cols(df, cols):
                out = []
                for f in cols:
                    if f in df.columns:
                        s = float(df[f].std(skipna=True))
                        if not np.isfinite(s) or s < EPS:
                            out.append(f)
                return out
            
            # Check top features that are missing or constant at serving (robust)
            # Only if feature importance names align with data
            if fi_aligned:
                top_20 = importances.sort_values(ascending=False).head(20)
                top_20_names = top_20.index.tolist()
                missing_top = [f for f in top_20_names if f not in X.columns]
                const_top_prefill = _const_cols(featured_df, top_20_names)
                const_top_final = _const_cols(X, top_20_names)
                
                if missing_top:
                    logger.error("❌ Top features missing at serving:")
                    for feat in missing_top:
                        importance = importances.get(feat, 0.0)
                        logger.error(f"   - {feat:<30} (importance: {importance:.6f})")
                
                if const_top_prefill:
                    logger.warning("⚠️  Top features constant BEFORE fill:")
                    for feat in const_top_prefill:
                        importance = importances.get(feat, 0.0)
                        logger.warning(f"   - {feat:<30} (importance: {importance:.6f})")
                
                if const_top_final and not const_top_prefill:
                    logger.warning("⚠️  Top features became constant AFTER alignment/fill:")
                    for feat in const_top_final:
                        importance = importances.get(feat, 0.0)
                        unique_vals = X[feat].nunique(dropna=True) if feat in X.columns else 0
                        sample_val = X[feat].iloc[0] if feat in X.columns and len(X) > 0 else 'N/A'
                        logger.warning(f"   - {feat:<30} (importance: {importance:.6f})")
                        logger.warning(f"     → Unique values: {unique_vals}, sample: {sample_val}")
                
                if not missing_top and not const_top_prefill and not const_top_final:
                    logger.info("✅ No top-importance features are missing or constant at serving")
                    
                # Surface top-feature variance for this slate (handy sanity check)
                if len(X) > 0:
                    top = importances.sort_values(ascending=False).head(10).index.tolist()
                    present = [c for c in top if c in X.columns]
                    if present:
                        top_stds = X[present].std(numeric_only=True).sort_values()
                        logger.info("Top feature STDs this slate:\n%s", top_stds)
                        
                # Robust anomaly check using reference stats or snapshot
                ref = b.get("reference_stats")
                ref_stats = {}

                # Use only entries that already have 'median' and 'mad'
                if isinstance(ref, dict):
                    for c, d in ref.items():
                        if isinstance(d, dict) and ("median" in d and "mad" in d):
                            try:
                                ref_stats[c] = {"median": float(d["median"]), "mad": float(d["mad"])}
                            except (TypeError, ValueError):
                                pass

                # If still empty or missing keys for some columns, compute from training snapshot
                if (not ref_stats) and isinstance(snap, pd.DataFrame) and not snap.empty:
                    ref_df = snap.select_dtypes(include=[np.number])
                    for c in ref_df.columns:
                        s = pd.to_numeric(ref_df[c], errors="coerce").dropna()
                        if len(s) >= 100:
                            med = float(s.median())
                            mad = float(1.4826 * np.median(np.abs(s - med)))
                            ref_stats[c] = {"median": med, "mad": mad}

                def anomalyrate(col, serve, stats, fallback_snap=None, z_floor=0.5, z=4.0):
                    d = stats.get(col, {})
                    med = d.get("median")
                    mad = d.get("mad")

                    # Try to compute from snapshot if missing/incomplete
                    if (med is None or mad is None) and isinstance(fallback_snap, pd.DataFrame) and col in fallback_snap.columns:
                        s = pd.to_numeric(fallback_snap[col], errors="coerce").dropna()
                        if len(s) >= 100:
                            med = float(s.median())
                            mad = float(1.4826 * np.median(np.abs(s - med)))

                    if med is None or mad is None:
                        return None  # no reference available → skip gracefully

                    sigma = max(mad, z_floor)
                    zscores = (serve[col].astype(float) - med) / sigma
                    return float((zscores.abs() > z).mean())

                key_cols = ["expected_total","home_sp_era","away_sp_era"]
                rates = {}
                
                if not ref_stats:
                    logger.info("ℹ️ Skipping anomaly-rate check (no reference stats available)")
                else:
                    for c in key_cols:
                        if c in X.columns:
                            r = anomalyrate(c, X, ref_stats, fallback_snap=snap)
                            if r is not None:
                                rates[c] = r

                # Only warn if >30% beyond 4 robust σ AND training MAD isn't degenerate
                anomalies = []
                for c, r in rates.items():
                    if ref_stats.get(c, {}).get("mad", 0) >= 0.05 and r > 0.3:
                        anomalies.append((c, r))
                
                if anomalies:
                    logger.warning("⚠️  Robust anomaly rates (>4σ):")
                    for feat, rate in anomalies:
                        logger.warning(f"   - {feat}: {rate:.1%} outliers")
                else:
                    logger.info("✅ No significant anomaly rates detected")
            else:
                logger.warning("⚠️  Skipping top feature analysis - feature importance names don't align")
                
        else:
            logger.warning(f"❌ No games found for {target_date} to check serving schema")
            
    except Exception as e:
        logger.exception("❌ Error in schema drift analysis")
    finally:
        if engine:
            engine.dispose()
    
    # 4) Training metadata check
    logger.info("\n4️⃣ TRAINING METADATA")
    logger.info("-" * 40)
    
    metadata_keys = ['training_date', 'label_definition', 'evaluation_metrics', 'bias_correction', 
                    'training_period', 'model_type', 'training_feature_snapshot']
    
    for key in metadata_keys:
        if key in b:
            logger.info(f"✅ {key}: {b[key]}")
        else:
            logger.info(f"❌ Missing: {key}")
    
    # 5) Feature variance in bundle (if training snapshot exists)
    logger.info("\n5️⃣ TRAINING FEATURE VARIANCE")
    logger.info("-" * 40)
    
    snap = b.get("training_feature_snapshot", None)
    if isinstance(snap, pd.DataFrame) and len(snap):
        cols = ["home_sp_k_per_9","away_sp_k_per_9","home_sp_whip","away_sp_whip",
                "home_sp_bb_per_9","away_sp_bb_per_9","home_sp_starts","away_sp_starts",
                "expected_total", "pitching_vs_offense", "combined_bb_rate"]
        present = [c for c in cols if c in snap.columns]
        
        logger.info(f"Training snapshot: {len(snap)} rows, {len(snap.columns)} features")
        
        if present:
            variances = snap[present].std(numeric_only=True).sort_values()
            logger.info("Training feature standard deviations:")
            for feat, std_val in variances.items():
                status = "❌ FLAT" if std_val < 0.01 else "✅ OK" if std_val > 0.1 else "⚠️  LOW"
                logger.info(f"   {feat:<25} {std_val:8.6f} {status}")
        else:
            logger.info("❌ No key features found in training snapshot")
            
        # Enhanced drift analysis with small-sample robustness
        if 'X' in locals() and len(X):
            # 5a) Build a pooled serving window when needed (last N days)
            serve_start = (datetime.strptime(target_date, '%Y-%m-%d') - timedelta(days=serve_days-1)).strftime('%Y-%m-%d')
            try:
                pooled_q = text("""
                    SELECT
                      lgf.*,
                      COALESCE(NULLIF(lgf.market_total, 0), eg.market_total) AS market_total_final
                    FROM legitimate_game_features lgf
                    LEFT JOIN enhanced_games eg
                      ON eg.game_id = lgf.game_id AND eg."date" = lgf."date"
                    WHERE lgf."date" BETWEEN :start AND :end
                """)
                with create_engine(db_url).connect() as conn:
                    pooled_raw = pd.read_sql(pooled_q, conn, params={"start": serve_start, "end": target_date})
                if not pooled_raw.empty:
                    pooled_raw['market_total'] = pooled_raw.pop('market_total_final')
                    pooled_raw = pooled_raw.loc[:, ~pooled_raw.columns.duplicated(keep='last')]
                    pooled_df = predictor.engineer_features(pooled_raw)
                    try:
                        pooledX = predictor.align_serving_features(pooled_df, strict=False)
                        if not isinstance(pooledX, pd.DataFrame):
                            pooledX = pd.DataFrame(pooledX, columns=b['feature_columns'])
                    except Exception:
                        pooledX = safe_align(pooled_df, b['feature_columns'], 
                                           getattr(predictor, 'fill_values', {}) or b.get('feature_fill_values', {}))
                    pooledX = pooledX[b['feature_columns']].apply(pd.to_numeric, errors='coerce')
                else:
                    pooledX = X.copy()
            except Exception:
                logger.warning("Failed to build pooled serving window, using today only")
                pooledX = X.copy()  # fall back to today only

            # 5b) Choose drift mode
            ref = snap  # training snapshot
            serve_for_drift = pooledX.dropna(axis=1, how='all')
            enough_for_psi = (len(serve_for_drift) >= psi_min_serving) and (len(ref) >= psi_min_training)

            def _non_placeholder(cols_df):
                # keep columns with ≥80% non-NaN coverage
                return [c for c in cols_df.columns if cols_df[c].notna().mean() >= 0.8]

            drift_cols = [c for c in ['expected_total','combined_bb_rate','home_sp_whip','away_sp_whip','home_sp_k_per_9','away_sp_k_per_9']
                          if c in ref.columns and c in serve_for_drift.columns]
            drift_cols = [c for c in drift_cols if c in _non_placeholder(serve_for_drift)]

            logger.info(f"Drift analysis: {len(serve_for_drift)} serving rows ({serve_days}d pool), enough_for_psi={enough_for_psi}")

            if enough_for_psi:
                # PSI with quantile bins from training
                def _psi(train, serve, bins=psi_bins):
                    qt = np.quantile(train, np.linspace(0, 1, bins+1))
                    qt[0], qt[-1] = -np.inf, np.inf
                    pt = np.histogram(train, bins=qt)[0] / max(1, len(train))
                    ps = np.histogram(serve, bins=qt)[0] / max(1, len(serve))
                    pt = np.clip(pt, 1e-6, None); ps = np.clip(ps, 1e-6, None)
                    return float(np.sum((pt-ps)*np.log(pt/ps)))
                    
                rows = []
                for c in drift_cols:
                    tr = pd.to_numeric(ref[c], errors='coerce').dropna().values
                    sv = pd.to_numeric(serve_for_drift[c], errors='coerce').dropna().values
                    if len(tr)>10 and len(sv)>10:
                        rows.append((c, _psi(tr, sv)))
                        
                if rows:
                    drift_df = pd.DataFrame(rows, columns=['feature','psi']).sort_values('psi', ascending=False)
                    logger.info("PSI (train→serve pooled):\n%s", drift_df.to_string(index=False))
                    high = drift_df[drift_df.psi > 0.25]
                    if len(high):
                        logger.warning("⚠️  High drift in %d features (PSI>0.25)", len(high))
                        for _, row in high.head(3).iterrows():
                            logger.warning(f"   - {row['feature']}: PSI = {row['psi']:.3f}")
                else:
                    logger.info("ℹ️ Insufficient feature overlap for PSI analysis")
                    drift_df = pd.DataFrame(columns=['feature','psi'])
            else:
                # Small-sample robust checks (today or small pool)
                logger.info("ℹ️ Using small-sample drift checks (serve=%d, train=%d)", len(serve_for_drift), len(ref))
                rows = []
                for c in drift_cols:
                    tr = pd.to_numeric(ref[c], errors='coerce')
                    sv = pd.to_numeric(serve_for_drift[c], errors='coerce')
                    if tr.notna().sum()>20 and sv.notna().sum()>5:
                        mu, sd = tr.mean(), tr.std(ddof=1)
                        sd = max(sd, 1e-6)
                        # standardized mean shift
                        d_mean = float((sv.mean() - mu)/sd)
                        # robust median shift (MAD)
                        mad = 1.4826*np.median(np.abs(tr - tr.median()))
                        mad = max(mad, 1e-6)
                        d_med = float((sv.median() - tr.median())/mad)
                        # tail exceedance using training 1–99th pct
                        p01, p99 = tr.quantile(0.01), tr.quantile(0.99)
                        tail_rate = float(((sv < p01) | (sv > p99)).mean())
                        rows.append((c, d_mean, d_med, tail_rate))
                        
                if rows:
                    smalldrift = pd.DataFrame(rows, columns=['feature','z_mean','z_median','tail_1p_99p']).sort_values('tail_1p_99p', ascending=False)
                    logger.info("Small-sample drift summary:\n%s", smalldrift.to_string(index=False))
                    # flag if any two metrics breach mild thresholds
                    flags = smalldrift[(smalldrift.tail_1p_99p>0.25) | (smalldrift.z_mean.abs()>1.0) | (smalldrift.z_median.abs()>1.5)]
                    if len(flags):
                        logger.warning("⚠️  Possible drift (small-sample) in %d features", len(flags))
                        for _, row in flags.head(3).iterrows():
                            logger.warning(f"   - {row['feature']}: tail={row['tail_1p_99p']:.2f}, z_mean={row['z_mean']:+.2f}")
                    drift_df = smalldrift  # so it lands in the JSON report as `psi`
                else:
                    logger.info("ℹ️ Insufficient data for small-sample drift analysis")
                    drift_df = pd.DataFrame(columns=['feature','z_mean','z_median','tail_1p_99p'])
        else:
            logger.info("ℹ️ No serving data available for drift analysis")
            drift_df = pd.DataFrame(columns=['feature'])
                
    elif isinstance(snap, pd.DataFrame):
        logger.info("⚠️  Training snapshot is empty DataFrame")
    else:
        logger.info("❌ No training feature snapshot saved in bundle")
    
    # 6) Expected total dependency check
    logger.info("\n6️⃣ EXPECTED_TOTAL DEPENDENCY")
    logger.info("-" * 40)
    
    if hasattr(b['model'], "feature_importances_") and fi_aligned:
        exp_total_imp = importances.get('expected_total', 0.0)
        market_total_imp = importances.get('market_total', 0.0)
        
        logger.info(f"expected_total importance: {exp_total_imp:.6f}")
        logger.info(f"market_total importance: {market_total_imp:.6f}")
        
        total_dependency = exp_total_imp + market_total_imp
        if total_dependency > 0.1:
            logger.info(f"⚠️  High market dependency: {total_dependency:.3f} (>10%)")
            logger.info("   Consider blending approach or retraining with reduced market reliance")
        else:
            logger.info(f"✅ Reasonable market dependency: {total_dependency:.3f}")
    else:
        logger.info("ℹ️ Skipping dependency check (feature names not aligned with importances)")
    
    # 7) Systematic bias analysis vs actual results AND market
    logger.info("\n7️⃣ SYSTEMATIC BIAS ANALYSIS")  
    logger.info("-" * 40)
    
    engine = None
    try:
        engine = create_engine(db_url)
        
        # Bias vs market (quick recent check)
        bias_query = text("""
            SELECT predicted_total, market_total
            FROM enhanced_games
            WHERE predicted_total IS NOT NULL 
              AND market_total IS NOT NULL
              AND market_total BETWEEN 6 AND 12
              AND "date" >= :start_date
            ORDER BY "date" DESC
            LIMIT 100
        """)
        start_date = (datetime.strptime(target_date, '%Y-%m-%d') - timedelta(days=market_days)).strftime('%Y-%m-%d')
        bias_df = pd.read_sql(bias_query, engine, params={'start_date': start_date})
        
        if len(bias_df) > 10:
            deltas = bias_df['predicted_total'] - bias_df['market_total']
            mean_bias_market = deltas.mean()
            std_bias = deltas.std()
            
            logger.info(f"Recent predictions vs market (n={len(bias_df)}):")
            logger.info(f"   Mean bias (pred - market): {mean_bias_market:+.3f}")
            logger.info(f"   Std deviation: {std_bias:.3f}")
            logger.info(f"   Range: {deltas.min():+.2f} to {deltas.max():+.2f}")
            
            if abs(mean_bias_market) > 0.5:
                logger.warning(f"⚠️  Significant systematic bias vs market: {mean_bias_market:+.3f}")
        else:
            logger.warning("❌ Insufficient recent prediction data for market bias analysis")
            mean_bias_market = None
        
        # Bias vs actual results (more important!)
        truth_start = (datetime.strptime(target_date, '%Y-%m-%d') - timedelta(days=truth_days)).strftime('%Y-%m-%d')
        hist = _fetch_truth_df(engine, truth_start)
        
        bias_truth = None
        if not hist.empty:
            y  = hist['total_runs'].astype(float).values
            yp = hist['predicted_total'].astype(float).values
            mm = hist['market_total'].astype(float).values

            bias_truth = float((yp - y).mean())
            mae_model  = float(np.abs(yp - y).mean())
            mae_market = float(np.abs(mm - y).mean())
            market_bias = float((mm - y).mean())

            # Add calibration analysis (slope/R²)
            x1 = np.c_[np.ones_like(yp), yp]
            try:
                coef, _, _, _ = np.linalg.lstsq(x1, y, rcond=None)
                yhat = x1 @ coef
                ss_res = float(np.sum((y - yhat)**2))
                ss_tot = float(np.sum((y - y.mean())**2))
                r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
                calibration_intercept = float(coef[0])
                calibration_slope     = float(coef[1])
            except np.linalg.LinAlgError:
                calibration_intercept = np.nan
                calibration_slope = np.nan
                r2 = np.nan

            logger.info(f"\nBias vs actual results (n={len(hist)}):")
            logger.info(f"   Model bias (pred - actual): {bias_truth:+.3f}")
            logger.info(f"   Market bias (market - actual): {market_bias:+.3f}")
            logger.info(f"   MAE — model: {mae_model:.3f} | market: {mae_market:.3f} | delta: {mae_model - mae_market:+.3f}")
            if not np.isnan(calibration_slope):
                logger.info(f"   Calibration vs actuals: intercept={calibration_intercept:+.2f}, slope={calibration_slope:.2f}, R²={r2:.3f}")
                if calibration_slope < 0.7 or calibration_slope > 1.3:
                    logger.warning(f"⚠️  Poor calibration slope: {calibration_slope:.2f} (should be ~1.0)")
            
            if abs(bias_truth) > 0.5:
                logger.warning(f"⚠️  Significant bias vs truth; consider bundle bias_correction = {-bias_truth:.3f}")
                
                # Offer to save bias correction (respecting dry-run flag)
                if apply_bias_correction and not dry_run:
                    b['bias_correction'] = float(-bias_truth)
                    joblib.dump(b, bundle_path)
                    logger.info("✅ Saved bias_correction into bundle")
                elif dry_run:
                    logger.info("   DRY RUN: Would save bias_correction = {:.3f}".format(-bias_truth))
                else:
                    logger.info("   Use --apply-bias-correction to save bias correction to bundle")
            else:
                logger.info(f"✅ Bias vs truth within acceptable range: {bias_truth:+.3f}")
        else:
            logger.warning("❌ No historical data with actual results for truth bias analysis (after fallbacks)")
            
    except Exception as e:
        logger.error(f"❌ Error in bias analysis: {e}")
    finally:
        if engine:
            engine.dispose()
    
    # 8) Expected_total dependency ablation test
    logger.info("\n8️⃣ EXPECTED_TOTAL DEPENDENCY ABLATION")
    logger.info("-" * 40)
    
    # Guard: ensure predictor exists for ablation
    if 'predictor' not in locals():
        logger.warning("❌ Predictor not available for ablation test (import failed)")
        return
    
    if 'X' in locals() and len(X) > 0:
        def _transform(M, preproc, scaler):
            # Robust transform helper (handles scaler mismatch like production)
            if preproc is not None:
                return preproc.transform(M)
            if scaler is not None and (not hasattr(scaler, "n_features_in_") or scaler.n_features_in_ == M.shape[1]):
                return scaler.transform(M)
            return M.values
        
        def _ensure_df(M, cols):
            """Ensure M is a DataFrame for easy column assignment"""
            return M if isinstance(M, pd.DataFrame) else pd.DataFrame(M, columns=cols)
        
        try:
            # Normal predictions - fill NaNs first since we removed numeric coercion from predictor
            X_pred = safe_align(
                X, 
                b['feature_columns'], 
                b.get('feature_fill_values') or getattr(predictor, 'fill_values', {}) or {}
            )
            Xt = _transform(X_pred, predictor.preproc, predictor.scaler)
            pred = predictor.model.predict(Xt)
            normal_spread = float(np.ptp(pred))
            
            # Guard ablation division by zero
            if normal_spread <= 1e-9:
                logger.warning("Normal spread ~0; ablation ratios not informative")
                normal_spread = 1e-9  # prevent divide-by-zero
            
            logger.info(f"Prediction spread analysis:")
            logger.info(f"   Normal spread: {normal_spread:.2f}")
            
            # Ablation: clamp expected_total to median
            if 'expected_total' in X_pred.columns:
                X_e = X_pred.copy()
                X_e['expected_total'] = X_pred['expected_total'].median()
                pred_e = predictor.model.predict(_transform(X_e, predictor.preproc, predictor.scaler))
                spread_e = float(np.ptp(pred_e))
                logger.info(f"   With expected_total fixed: {spread_e:.2f}")
                logger.info(f"   Spread retention: {spread_e/normal_spread*100:.1f}%")
                
                if spread_e < normal_spread * 0.3:
                    logger.warning("⚠️  Model is over-anchored to expected_total (market prior)")
                    logger.warning("   Consider blending approach or retraining with reduced market reliance")
                else:
                    logger.info("✅ Reasonable expected_total dependency")
            else:
                logger.warning("❌ No expected_total column found for ablation test")
            
            # Ablation: clamp market_total if present
            if 'market_total' in X_pred.columns:
                X_m = X_pred.copy()
                X_m['market_total'] = X_pred['market_total'].median()
                pred_m = predictor.model.predict(_transform(X_m, predictor.preproc, predictor.scaler))
                spread_m = float(np.ptp(pred_m))
                logger.info(f"   With market_total fixed:   {spread_m:.2f}")
                
        except Exception as e:
            logger.error(f"❌ Error in ablation test: {e}")
    else:
        logger.warning("❌ No serving data available for ablation test")
    
    # 9) Recommendations
    logger.info("\n9️⃣ RECOMMENDATIONS")
    logger.info("-" * 40)
    
    recommendations = []
    
    # Check for missing important features (only if fi_aligned)
    if fi_aligned and 'missing_top' in locals() and missing_top:
        recommendations.append(f"🔧 Engineer missing top features: {missing_top[:5]}")
    
    # Check for constant top features (only if fi_aligned)
    if fi_aligned and 'const_top_prefill' in locals() and const_top_prefill:
        recommendations.append(f"🔧 Fix constant top features: {const_top_prefill}")
    if fi_aligned and 'const_top_final' in locals() and const_top_final:
        recommendations.append(f"🔧 Fix features constant after alignment: {const_top_final}")
    
    # Check preprocessing
    if not (b.get('preproc') or b.get('scaler')):
        recommendations.append("🔧 Add preprocessing pipeline to bundle")
    
    # Check bias vs truth
    if 'bias_truth' in locals() and bias_truth is not None and abs(bias_truth) > 0.5:
        recommendations.append(f"🔧 Add bias correction: {-bias_truth:.3f}")
    
    # Check market dependency
    if 'total_dependency' in locals() and total_dependency > 0.15:
        recommendations.append("🔧 Consider reducing market total dependency")
    
    # Check expected_total over-reliance
    if 'spread_e' in locals() and 'normal_spread' in locals():
        if spread_e < normal_spread * 0.3:
            recommendations.append("🔧 Model over-anchored to expected_total - consider blending")
    
    # Check placeholder-only feature count
    if 'placeholder_only' in locals() and len(placeholder_only) >= 0.25 * len(b['feature_columns']):
        recommendations.append(f"🔧 Reduce placeholder-only features by adding proxies/joins for "
                               f"{placeholder_only[:6]}…")
    
    if recommendations:
        logger.info("Priority fixes:")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"   {i}. {rec}")
    else:
        logger.info("✅ No critical issues found!")
    
    # 10) Machine-readable audit report
    logger.info("\n🔟 GENERATING AUDIT REPORT")
    logger.info("-" * 40)
    
    report = {
        "target_date": target_date,
        "missing_pre_align": missing_pre_align[:50] if 'missing_pre_align' in locals() else [],
        "placeholder_only": placeholder_only[:50] if 'placeholder_only' in locals() else [],
        "const_top_prefill": const_top_prefill if 'const_top_prefill' in locals() else [],
        "const_top_final": const_top_final if 'const_top_final' in locals() else [],
        "missing_at_serving": missing_at_serving[:50] if 'missing_at_serving' in locals() else [],
        "extra_at_serving": extra_at_serving[:50] if 'extra_at_serving' in locals() else [],
        "serving_sample": X.head(3).to_dict(orient="records") if 'X' in locals() and isinstance(X, pd.DataFrame) and len(X) else [],
        "mean_bias_market": float(mean_bias_market) if 'mean_bias_market' in locals() and mean_bias_market is not None else None,
        "bias_truth": float(bias_truth) if 'bias_truth' in locals() and bias_truth is not None else None,
        "calibration": {
            "slope": float(calibration_slope) if 'calibration_slope' in locals() and not np.isnan(calibration_slope) else None,
            "intercept": float(calibration_intercept) if 'calibration_intercept' in locals() and not np.isnan(calibration_intercept) else None,
            "r2": float(r2) if 'r2' in locals() and not np.isnan(r2) else None,
        },
        "spread_analysis": {
            "normal_spread": float(normal_spread) if 'normal_spread' in locals() else None,
            "expected_total_fixed_spread": float(spread_e) if 'spread_e' in locals() else None,
            "market_total_fixed_spread": float(spread_m) if 'spread_m' in locals() else None,
        },
        "recommendations": recommendations,
        "bundle_metadata": {
            "has_preproc": b.get('preproc') is not None,
            "has_scaler": b.get('scaler') is not None,
            "has_bias_correction": 'bias_correction' in b,
            "bias_correction_value": float(b.get('bias_correction', 0.0)),
        },
        "fi_aligned": fi_aligned,
        "psi": drift_df.to_dict(orient="records") if 'drift_df' in locals() else [],
        "counts": {
            "n_games": int(len(games_df)) if 'games_df' in locals() else 0,
            "n_missing_pre_align": int(len(missing_pre_align)) if 'missing_pre_align' in locals() else 0,
            "n_placeholder_only": int(len(placeholder_only)) if 'placeholder_only' in locals() else 0,
        }
    }
    
    # Add severity assessment for CI/monitoring (metadata enforcement trumps other issues)
    severity = "pass"
    
    # Check for drift flags
    high_drift = False
    if 'drift_df' in locals() and not drift_df.empty:
        if 'psi' in drift_df.columns:
            high_drift = bool((drift_df['psi'] > 0.25).any())
        elif {'tail_1p_99p','z_mean','z_median'}.issubset(drift_df.columns):
            high_drift = bool(
                (drift_df['tail_1p_99p'] > 0.25).any() |
                (drift_df['z_mean'].abs() > 2.0).any() |
                (drift_df['z_median'].abs() > 3.0).any()
            )
    
    if enforce_metadata and missing_meta:
        severity = "fail"  # Metadata enforcement takes priority
    elif 'placeholder_only' in locals() and len(placeholder_only) > 0:
        severity = "warn"
    elif fi_aligned and (('missing_top' in locals() and missing_top) or
                       ('const_top_final' in locals() and const_top_final)):
        severity = "fail"
    elif high_drift:
        severity = "warn"  # Add drift to warnings
        
    report["status"] = severity
    
    Path("training_bundle_audit_report.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    logger.info("📝 Wrote training_bundle_audit_report.json")
    
    # Optional: non-zero exit for CI on fail
    if severity == "fail":
        logger.error("❌ AUDIT FAILED - Critical issues detected")
        raise SystemExit(2)
    elif severity == "warn":
        logger.warning("⚠️  AUDIT WARNING - Issues detected but not critical")
    else:
        logger.info("✅ AUDIT PASSED - No issues detected")
    
    logger.info("\n" + "=" * 80)
    logger.info("🏁 AUDIT COMPLETE")
    logger.info("=" * 80)

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Audit training bundle for systematic issues')
    parser.add_argument('--target-date', type=str, help='Target date (YYYY-MM-DD)', 
                       default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--apply-bias-correction', action='store_true',
                       help='Write bias_correction to the bundle if significant bias is detected')
    parser.add_argument('--model-path', type=str, help='Path to model bundle')
    parser.add_argument('--enforce-metadata', action='store_true',
                       help='Exit non-zero if required training metadata is missing')
    parser.add_argument('--market-days', type=int, default=5,
                       help='Days to look back for market bias analysis (default: 5)')
    parser.add_argument('--truth-days', type=int, default=30,
                       help='Days to look back for truth bias analysis (default: 30)')
    parser.add_argument('--serve-days', type=int, default=7,
                       help='Days to pool for drift analysis (default: 7)')
    parser.add_argument('--psi-bins', type=int, default=10,
                       help='Number of bins for PSI calculation (default: 10)')
    parser.add_argument('--psi-min-serving', type=int, default=100,
                       help='Minimum serving rows required for PSI (default: 100)')
    parser.add_argument('--psi-min-training', type=int, default=1000,
                       help='Minimum training rows required for PSI (default: 1000)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Never write bias corrections even if --apply-bias-correction is passed')
    args = parser.parse_args()
    
    audit_training_bundle(
        args.target_date, 
        args.apply_bias_correction, 
        model_path=args.model_path, 
        enforce_metadata=args.enforce_metadata,
        market_days=args.market_days,
        truth_days=args.truth_days,
        dry_run=args.dry_run,
        serve_days=args.serve_days,
        psi_bins=args.psi_bins,
        psi_min_serving=args.psi_min_serving,
        psi_min_training=args.psi_min_training
    )

if __name__ == "__main__":
    main()
