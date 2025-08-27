#!/usr/bin/env python3
"""
Ultra Sharp Pipeline - Residual vs Market Learning

SURGICAL IMPROVEMENTS for 30%+ ROI:
- Predict residual vs market, not raw runs
- Time-ordered CV + OOF calibration
- Roof bucketing (closed vs open environments)
- Enhanced feature imputation
- Clipped residual targets with sigma estimation

Usage:
  python ultra_sharp_pipeline.py train --db "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb" --start 2024-04-01 --end 2025-08-25 --model_dir models_ultra
  python ultra_sharp_pipeline.py price --db "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb" --date 2025-08-26 --model_dir models_ultra --out exports/ultra_preds_2025-08-26.csv
"""

import os
import argparse
import logging
import joblib
import numpy as np
import hashlib
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy import create_engine, text

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# ---------- Serve-time column harmonizer ----------
def _harmonize_serve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map EG aliases into the names our prep expects; derive roof_open if missing."""
    d = df.copy()
    mapping = {
        "total_runs": "eg_total_runs",
        "market_total": "eg_market_total",
        "date": "eg_date",
        "home_sp_rest_days": "eg_home_sp_rest_days",
        "away_sp_rest_days": "eg_away_sp_rest_days",
        "roof_state": "eg_roof_state",
    }
    for to, frm in mapping.items():
        if to not in d.columns and frm in d.columns:
            d[to] = d[frm]

    # roof_open (int 0/1) if missing
    if "roof_open" not in d.columns:
        rs = d.get("roof_state", pd.Series([""]*len(d))).astype(str).str.lower()
        rt = d.get("roof_type", pd.Series([""]*len(d))).astype(str).str.lower()
        d["roof_open"] = (~rs.str.contains("closed") & ~rt.str.contains("dome")).astype(int)
    return d


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _hash_mod(s: str, mod: int = 256) -> int:
    """Deterministic stable hash to [0, mod)"""
    return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16) % mod


def _is_closed_row(row):
    """Treat domes as closed; retractable unknowns → open bucket unless explicitly closed"""
    roof_open = row.get("roof_open")
    roof_state = str(row.get("roof_state") or "").lower()
    roof_type = str(row.get("roof_type") or "").lower()
    return (roof_open is False) or ("closed" in roof_state) or ("dome" in roof_type)


def _prep_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leak-proof feature prep with expanded pregame whitelist."""
    
    # EXPANDED SAFE WHITELIST: only pregame features, no post-game aggregates
    SAFE_WHITELIST = [
        # weather / park (all pregame forecast/static)
        "temperature","humidity","wind_speed","wind_direction_deg","dew_point",
        "cloud_cover","uv_index","precip_prob","pressure","air_pressure",
        "ballpark_run_factor","ballpark_hr_factor","park_cf_bearing_deg","altitude_ft",
        # roof / environment
        "roof_type","roof_status","roof_open",
        # context/schedule
        "day_night","is_weekend","doubleheader","getaway_day","day_after_night","month",
        # starters (pregame identity + rest/hand only)
        "home_sp_id","away_sp_id","home_sp_days_rest","away_sp_days_rest",
        "home_sp_hand","away_sp_hand","probable_pitchers_confirmed",
        # umpire assignment (static rates, not market-based)
        "plate_umpire","plate_umpire_bb_pct","plate_umpire_strike_zone_consistency","plate_umpire_rpg",
        # teams (IDs only for encoding; NO season totals or L7/L14 aggregates)
        "home_team","away_team",
        # NEW: pregame as-of features (computed with strict temporal integrity)
        "home_sp_era_l3_asof","away_sp_era_l3_asof",
        "home_sp_whip_l3_asof","away_sp_whip_l3_asof", 
        "home_bp_ip_3d_asof","away_bp_ip_3d_asof",
        "home_bp_era_30d_asof","away_bp_era_30d_asof",
        "home_runs_pg_14_asof","away_runs_pg_14_asof",
    ]
    
    X = df[[c for c in SAFE_WHITELIST if c in df.columns]].copy()
    missing_features = [c for c in SAFE_WHITELIST if c not in df.columns]
    log.info(f"Whitelisted features: {len(X.columns)} of {len(SAFE_WHITELIST)} available")
    if missing_features:
        log.info(f"Missing features: {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}")

    # Derive roof_state from roof_status when needed
    if "roof_status" in X.columns and "roof_state" not in X.columns:
        X["roof_state"] = X["roof_status"]

    # --- Helper function for safe numeric conversion ---
    def _num(s): 
        return pd.to_numeric(s, errors="coerce")

    # --- Convert booleans to ints ---
    bool_cols = ["roof_open","doubleheader","getaway_day","day_after_night","probable_pitchers_confirmed","is_weekend"]
    for col in bool_cols:
        if col in X.columns:
            X[col] = X[col].astype(bool).astype(int)

    # --- Category encodings (drop raw IDs later) ---
    cat_cols = ["home_team","away_team","home_sp_id","away_sp_id","day_night","roof_type","roof_status","plate_umpire"]
    for col in cat_cols:
        if col in X.columns:
            X[f"{col}_enc"] = X[col].astype("category").cat.codes

    # --- Weather→Park Physics Engineering ---
    # Wind toward center field (physics-based)
    if {"wind_speed","wind_direction_deg","park_cf_bearing_deg"} <= set(X.columns):
        theta = np.deg2rad(_num(X["wind_direction_deg"]) - _num(X["park_cf_bearing_deg"]))
        X["wind_out_cf"] = _num(X["wind_speed"]) * np.cos(theta)
    else:
        X["wind_out_cf"] = 0.0

    # --- Pregame As-Of Feature Engineering & Fallbacks ---
    # SP ERA L3 with fallbacks
    if "home_sp_era_l3_asof" in X.columns:
        X["home_sp_era_l3_asof"] = _num(X["home_sp_era_l3_asof"]).fillna(4.20)  # league median
    else:
        X["home_sp_era_l3_asof"] = 4.20
    
    if "away_sp_era_l3_asof" in X.columns:
        X["away_sp_era_l3_asof"] = _num(X["away_sp_era_l3_asof"]).fillna(4.20)
    else:
        X["away_sp_era_l3_asof"] = 4.20
    
    # SP WHIP L3 with fallbacks  
    if "home_sp_whip_l3_asof" in X.columns:
        X["home_sp_whip_l3_asof"] = _num(X["home_sp_whip_l3_asof"]).fillna(1.30)  # league median
    else:
        X["home_sp_whip_l3_asof"] = 1.30
        
    if "away_sp_whip_l3_asof" in X.columns:
        X["away_sp_whip_l3_asof"] = _num(X["away_sp_whip_l3_asof"]).fillna(1.30)
    else:
        X["away_sp_whip_l3_asof"] = 1.30
    
    # Bullpen ERA 30d with fallbacks
    if "home_bp_era_30d_asof" in X.columns:
        X["home_bp_era_30d_asof"] = _num(X["home_bp_era_30d_asof"]).fillna(4.10)  # league median
    else:
        X["home_bp_era_30d_asof"] = 4.10
        
    if "away_bp_era_30d_asof" in X.columns:
        X["away_bp_era_30d_asof"] = _num(X["away_bp_era_30d_asof"]).fillna(4.10)
    else:
        X["away_bp_era_30d_asof"] = 4.10
    
    # Bullpen IP 3d with fallbacks (0 = well-rested)
    if "home_bp_ip_3d_asof" in X.columns:
        X["home_bp_ip_3d_asof"] = _num(X["home_bp_ip_3d_asof"]).fillna(0.0)
    else:
        X["home_bp_ip_3d_asof"] = 0.0
        
    if "away_bp_ip_3d_asof" in X.columns:
        X["away_bp_ip_3d_asof"] = _num(X["away_bp_ip_3d_asof"]).fillna(0.0)
    else:
        X["away_bp_ip_3d_asof"] = 0.0
    
    # Team runs per game L14 with fallbacks
    if "home_runs_pg_14_asof" in X.columns:
        X["home_runs_pg_14_asof"] = _num(X["home_runs_pg_14_asof"]).fillna(4.5)  # league median
    else:
        X["home_runs_pg_14_asof"] = 4.5
        
    if "away_runs_pg_14_asof" in X.columns:
        X["away_runs_pg_14_asof"] = _num(X["away_runs_pg_14_asof"]).fillna(4.5)
    else:
        X["away_runs_pg_14_asof"] = 4.5
    
    # --- Derived matchup features (key signal!) ---
    X["sp_era_diff_asof"] = X["home_sp_era_l3_asof"] - X["away_sp_era_l3_asof"]
    X["sp_whip_diff_asof"] = X["home_sp_whip_l3_asof"] - X["away_sp_whip_l3_asof"]
    X["bp_era_diff_30d_asof"] = X["home_bp_era_30d_asof"] - X["away_bp_era_30d_asof"] 
    X["team_runs_pg14_diff_asof"] = X["home_runs_pg_14_asof"] - X["away_runs_pg_14_asof"]

    # Air density proxy (forecast temp + pressure → ball flight)
    if "temperature" in X.columns:
        temp_K = (_num(X["temperature"]) - 32.0) * 5.0/9.0 + 273.15
        pressure = _num(X.get("pressure", X.get("air_pressure", 1013.25)))
        X["air_density_proxy"] = (pressure.fillna(1013.25) / temp_K.replace(0, np.nan)).fillna(1013.25/295.15)
    else:
        X["air_density_proxy"] = 1013.25/295.15

    # Altitude effects (if available)
    if "altitude_ft" in X.columns:
        X["altitude_kft"] = _num(X["altitude_ft"]).fillna(0.0) / 1000.0
    else:
        X["altitude_kft"] = 0.0

    # --- Weather×Park Interactions (key for run environment) ---
    if {"temperature","ballpark_run_factor"} <= set(X.columns):
        X["temp_park_interaction"] = _num(X["temperature"]) * _num(X["ballpark_run_factor"])
    else:
        X["temp_park_interaction"] = 0.0
        
    if {"ballpark_hr_factor"} <= set(X.columns):
        X["wind_park_interaction"] = X["wind_out_cf"] * _num(X["ballpark_hr_factor"])
    else:
        X["wind_park_interaction"] = 0.0

    # --- Seasonality (month sin/cos) ---
    if "month" in X.columns:
        m = _num(X["month"]).clip(1,12)
        X["month_sin"] = np.sin(2*np.pi*(m-1)/12)
        X["month_cos"] = np.cos(2*np.pi*(m-1)/12)
    else:
        X["month_sin"] = 0.0
        X["month_cos"] = 0.0

    # --- Roof environment flags (for bucketing logic) ---
    X["is_dome_or_closed"] = X.apply(_is_closed_row, axis=1).astype(int)
    if "roof_state" in X.columns:
        X["is_retractable"] = X["roof_state"].astype(str).str.contains("unknown|retract", case=False, regex=True).astype(int)
    else:
        X["is_retractable"] = 0

    # --- Safe numeric conversions for remaining features ---
    numeric_cols = ['home_sp_days_rest', 'away_sp_days_rest', 'plate_umpire_bb_pct', 
                   'plate_umpire_strike_zone_consistency', 'plate_umpire_rpg']
    for col in numeric_cols:
        if col in X.columns:
            X[col] = _num(X[col]).fillna(0)

    # --- Clean up: drop raw IDs and string helpers (keep encoded versions) ---
    drop_cols = [c for c in X.columns if c.endswith("_id")] + \
                ["home_team", "away_team", "plate_umpire", "roof_state", "roof_status", "roof_type"]
    X.drop(columns=drop_cols, errors="ignore", inplace=True)

    # --- Final cleanup: convert any remaining object columns ---
    for col in list(X.columns):
        if X[col].dtype == 'object':
            try:
                X[col] = _num(X[col]).fillna(0)
            except:
                log.warning(f"Dropping non-numeric column: {col}")
                X = X.drop(columns=[col])

    log.info(f"Final feature matrix: {X.shape} (expanded pregame whitelist)")
    return X



def _train_residual_model(df: pd.DataFrame, model_dir: Path, label_clip=4.0, bucket_name="all"):
    """Train residual model with proper time-ordered CV and leak-proof features"""
    need = {"total_runs","market_total","date"}
    if not need.issubset(df.columns): 
        raise RuntimeError(f"Missing cols: {need - set(df.columns)}")

    df = df.sort_values("date").reset_index(drop=True).copy()
    
    # Closing residual target (actual - closing)
    y_raw = (pd.to_numeric(df["total_runs"], errors="coerce") -
             pd.to_numeric(df["market_total"], errors="coerce"))
    
    # robust, data-driven clip (3 * MAD) bounded to [2, 5]
    mad = np.median(np.abs(y_raw - np.median(y_raw))) if len(y_raw) else 0.0
    robust_sigma = 1.4826 * mad if mad > 0 else 3.0
    label_clip = float(np.clip(3.0 * robust_sigma, 2.0, 5.0))
    df["resid"] = y_raw.clip(-label_clip, label_clip)
    
    df = df.dropna(subset=["resid"])

    y = df["resid"].astype(float).values
    X = _prep_features(df)
    
    # Enhanced feature diagnostics and sanity checks
    if len(X):
        nunique = X.nunique().sum()
        log.info(f"Feature matrix: shape={X.shape}, total_unique_values={int(nunique)}")
        
        # Check null rates for key as-of features
        asof_features = [c for c in X.columns if "_asof" in c]
        if asof_features:
            log.info(f"As-of features available: {len(asof_features)}")
            for feat in asof_features[:8]:  # first 8 to avoid spam
                null_rate = X[feat].isna().mean()
                if null_rate > 0:
                    log.info(f"  {feat}: {100*null_rate:.1f}% null")
        
        # Simple correlation check with target (just for key features)
        key_features = ["sp_era_diff_asof", "sp_whip_diff_asof", "bp_era_diff_30d_asof", 
                       "team_runs_pg14_diff_asof", "wind_out_cf", "temp_park_interaction"]
        valid_key_features = [f for f in key_features if f in X.columns]
        if valid_key_features and len(y) > 50:
            log.info("Key feature correlations with residual:")
            for feat in valid_key_features:
                if X[feat].std() > 1e-6:  # has variance
                    corr = np.corrcoef(X[feat].fillna(0), y)[0,1]
                    if np.isfinite(corr):
                        log.info(f"  {feat}: {corr:.3f}")
                    else:
                        log.info(f"  {feat}: flat/invalid")

    # sanity: target has spread
    y_std = float(np.std(y))
    log.info(f"Residual y: mean={np.mean(y):.3f}, std={y_std:.3f}, min={np.min(y):.2f}, max={np.max(y):.2f}")
    log.info(f"Using robust label_clip={label_clip:.3f} (3*MAD bounded [2,5])")
    if y_std < 1e-3:
        raise RuntimeError("Residual variance ~0. Check market_total/total_runs sourcing.")

    n = len(df)
    tscv = TimeSeriesSplit(n_splits=5)
    oof = np.full(n, np.nan)
    models = []
    fold_id = 0
    for tr_pos, va_pos in tscv.split(np.arange(n)):
        fold_id += 1
        # Assert no overlap
        if set(tr_pos).intersection(set(va_pos)):
            raise RuntimeError("Train/valid overlap in TimeSeriesSplit!")

        log.info(f"Fold {fold_id}: train={len(tr_pos)}, valid={len(va_pos)} | "
                 f"train_dates=[{df['date'].iloc[tr_pos[0]]}..{df['date'].iloc[tr_pos[-1]]}] "
                 f"→ valid_starts={df['date'].iloc[va_pos[0]]}")

        m = HistGradientBoostingRegressor(
            max_depth=5, max_iter=500, learning_rate=0.06,
            l2_regularization=0.0, min_samples_leaf=20, random_state=42
        )
        # optional sample weighting (env ULTRA_SAMPLE_WEIGHTS=true)
        sw = None
        if os.getenv("ULTRA_SAMPLE_WEIGHTS", "false").lower() in ("1","true","yes"):
            # emphasize in-range, information-rich residuals
            sw = 0.5 + 0.5 * (np.abs(y[tr_pos]) / label_clip)
        else:
            # default: emphasize larger residuals
            sw = 0.5 + 0.5*np.clip(np.abs(y[tr_pos])/label_clip, 0, 1.5)  # 0.5..1.25
        m.fit(X.iloc[tr_pos], y[tr_pos], sample_weight=sw)
        oof[va_pos] = m.predict(X.iloc[va_pos])
        models.append(m)

    valid_mask = ~np.isnan(oof)
    if valid_mask.sum() < n * 0.6:
        log.warning(f"OOF coverage low: {valid_mask.sum()}/{n}")

    # Isotonic on OOF only
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof[valid_mask], y[valid_mask])

    # metrics
    oof_cal = iso.predict(oof[valid_mask])
    mae = float(mean_absolute_error(y[valid_mask], oof_cal))
    sigma = float(np.std(y[valid_mask] - oof_cal))
    
    # Additional diagnostics
    oof_corr = float(np.corrcoef(oof[valid_mask], y[valid_mask])[0,1]) if valid_mask.sum() > 10 else 0.0
    r2 = max(0.0, oof_corr**2)
    resid_std = float(np.std(oof_cal))
    
    log.info(f"OOF metrics: MAE={mae:.6f}, σ={sigma:.6f}, corr={oof_corr:.3f}, R²={r2:.3f}, σ(pred)={resid_std:.3f}")

    if mae < 1e-3 and sigma < 1e-2:
        # extra debugging: show any residual leak columns still around
        suspicious = [c for c in X.columns if any(k in c.lower() for k in ("total","run","score","resid"))]
        log.error(f"Leakage detected; suspicious feature names: {suspicious[:12]}")
        raise RuntimeError("Calibrator leakage detected: OOF MAE/σ ~ 0.")

    bundle = {
        "bucket": bucket_name,
        "feature_cols": list(X.columns),
        "models": models,
        "calibrator": iso,
        "target": "residual",
        "label_clip": label_clip,
        "train_mae": float(mae),
        "train_sigma": float(sigma),
        "train_corr": float(oof_corr),
        "train_r2": float(r2),
        "pred_std": float(resid_std),
        "trained_at": datetime.utcnow().isoformat(timespec="seconds")
    }
    outp = model_dir / f"ultra_bundle_{bucket_name}.joblib"
    joblib.dump(bundle, outp)
    return outp, mae, sigma


def _predict_with_bundle(bundle_path: Path, df_today: pd.DataFrame) -> np.ndarray:
    """Predict residuals with trained bundle using final calibrator (with collapse guard)"""
    b = joblib.load(bundle_path)
    X = (_prep_features(df_today)
         .reindex(columns=b["feature_cols"])
         .fillna(0))
    
    # Log missing features
    missing = [c for c in b["feature_cols"] if c not in X.columns]
    if missing:
        frac = len(missing)/max(1,len(b["feature_cols"]))
        log.warning("Serving missing %d/%d features (%.1f%%). Example: %s",
                    len(missing), len(b["feature_cols"]), 100*frac, missing[:8])
        if frac > 0.3:
            log.warning("Too many missing features → predictions likely flat.")
    
    raw = np.mean([m.predict(X) for m in b["models"]], axis=0)
    
    # Check raw variance
    if np.ptp(raw) < 1e-6:
        log.warning("Raw model predictions are constant; check feature coverage/alignment.")
    
    # Guard against calibrator collapse
    cal = b["calibrator"]
    x_min, x_max = cal.X_thresholds_[0], cal.X_thresholds_[-1]
    frac_oob = np.mean((raw < x_min) | (raw > x_max))
    
    if (np.ptp(raw) < 0.15) or (frac_oob > 0.25):
        # too flat or mostly out-of-domain → skip calibration
        log.warning("Bypassing collapsed calibrator: raw_range=%.3f, oob_frac=%.3f", np.ptp(raw), frac_oob)
        preds = raw
    else:
        preds = cal.predict(raw)
        
    return np.clip(preds, -b["label_clip"], b["label_clip"])


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def cmd_train(args):
    """Train residual models with roof bucketing"""
    log.info("🚀 Starting ultra sharp training (residual vs market)")
    engine = create_engine(args.db)
    
    # Use game_conditions + enhanced_games + pregame_features for comprehensive data
    sql = """
      SELECT
        gc.*,
        eg.total_runs,
        eg.market_total,
        eg.date,
        -- tiny pregame whitelist from EG:
        eg.plate_umpire,
        eg.plate_umpire_bb_pct,
        eg.plate_umpire_strike_zone_consistency,
        eg.plate_umpire_rpg,
        eg.roof_type,
        eg.roof_status,
        -- pregame as-of features:
        pf.home_sp_era_l3_asof,
        pf.away_sp_era_l3_asof,
        pf.home_sp_whip_l3_asof,
        pf.away_sp_whip_l3_asof,
        pf.home_bp_ip_3d_asof,
        pf.away_bp_ip_3d_asof,
        pf.home_bp_era_30d_asof,
        pf.away_bp_era_30d_asof,
        pf.home_runs_pg_14_asof,
        pf.away_runs_pg_14_asof
      FROM game_conditions gc
      JOIN enhanced_games eg ON eg.game_id::text = gc.game_id::text
      LEFT JOIN pregame_features_v1 pf ON pf.game_id = eg.game_id
      WHERE eg.date BETWEEN :s AND :e
        AND eg.total_runs IS NOT NULL
        AND eg.market_total IS NOT NULL
    """
    
    df = pd.read_sql(text(sql), engine, params={"s": args.start, "e": args.end})
    df = _harmonize_serve_columns(df)
    
    # Fix roof bucketing alias
    if "roof_status" in df.columns and "roof_state" not in df.columns:
        df["roof_state"] = df["roof_status"]
    
    log.info("Loaded %d games from %s to %s", len(df), args.start, args.end)
    
    if df.empty:
        log.error("No training data found!")
        raise SystemExit(1)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Buckets: closed vs open environments
    df_closed = df[df.apply(_is_closed_row, axis=1)]
    df_open   = df[~df.apply(_is_closed_row, axis=1)]

    log.info("Data split: %d total | %d closed | %d open", len(df), len(df_closed), len(df_open))

    paths = []
    mae_all = []
    sigma_all = []

    # Train ALL bucket (fallback)
    p, mae, sigma = _train_residual_model(df, model_dir, bucket_name="all")
    log.info("ALL bucket: %d games, MAE=%.3f, σ=%.3f → %s", len(df), mae, sigma, p)
    paths.append(p)
    mae_all.append(mae)
    sigma_all.append(sigma)

    # Train OPEN bucket if enough data
    if len(df_open) > 300:
        p, mae, sigma = _train_residual_model(df_open, model_dir, bucket_name="open")
        log.info("OPEN bucket: %d games, MAE=%.3f, σ=%.3f → %s", len(df_open), mae, sigma, p)
        paths.append(p)
        mae_all.append(mae)
        sigma_all.append(sigma)
    else:
        log.warning("Only %d open games, skipping bucket", len(df_open))

    # Train CLOSED bucket if enough data  
    if len(df_closed) > 150:
        p, mae, sigma = _train_residual_model(df_closed, model_dir, bucket_name="closed")
        log.info("CLOSED bucket: %d games, MAE=%.3f, σ=%.3f → %s", len(df_closed), mae, sigma, p)
        paths.append(p)
        mae_all.append(mae)
        sigma_all.append(sigma)
    else:
        log.warning("Only %d closed games, skipping bucket", len(df_closed))

    # Save registry
    registry = {
        "bundles": [str(p) for p in paths], 
        "avg_mae": float(np.mean(mae_all)),
        "avg_sigma": float(np.mean(sigma_all)),
        "train_window": {"start": str(args.start), "end": str(args.end)},
        "trained_at": datetime.utcnow().isoformat()
    }
    joblib.dump(registry, model_dir / "ultra_registry.joblib")
    log.info("✅ Models saved to %s | avg OOF-MAE=%.3f, avg σ=%.3f", model_dir, registry["avg_mae"], registry["avg_sigma"])


def cmd_price(args):
    """Price with ultra residual models"""
    log.info("🧮 Pricing with ultra residual models")
    engine = create_engine(args.db)
    reg = joblib.load(Path(args.model_dir) / "ultra_registry.joblib")

    # Today's games from game_conditions + enhanced_games + pregame_features
    q = """
      SELECT
        gc.*,
        eg.market_total,
        eg.date,
        -- tiny pregame whitelist from EG:
        eg.plate_umpire,
        eg.plate_umpire_bb_pct,
        eg.plate_umpire_strike_zone_consistency,
        eg.plate_umpire_rpg,
        eg.roof_type,
        eg.roof_status,
        -- pregame as-of features:
        pf.home_sp_era_l3_asof,
        pf.away_sp_era_l3_asof,
        pf.home_sp_whip_l3_asof,
        pf.away_sp_whip_l3_asof,
        pf.home_bp_ip_3d_asof,
        pf.away_bp_ip_3d_asof,
        pf.home_bp_era_30d_asof,
        pf.away_bp_era_30d_asof,
        pf.home_runs_pg_14_asof,
        pf.away_runs_pg_14_asof
      FROM game_conditions gc
      JOIN enhanced_games eg ON eg.game_id::text = gc.game_id::text
      LEFT JOIN pregame_features_v1 pf ON pf.game_id = eg.game_id
      WHERE eg.date = :d
        AND eg.market_total IS NOT NULL
        AND eg.total_runs IS NULL
    """
    today = pd.read_sql(text(q), engine, params={"d": args.date})
    today = _harmonize_serve_columns(today)
    
    # Fix roof bucketing alias
    if "roof_status" in today.columns and "roof_state" not in today.columns:
        today["roof_state"] = today["roof_status"]
    
    # ---- Deduplicate: keep latest snapshot per game_id if duplicates exist ----
    if "game_id" in today.columns:
        dupes = today["game_id"].duplicated(keep=False).sum()
        if dupes:
            log.warning(f"WARNING: {dupes} duplicated rows across game_id — keeping latest per game.")
            sort_keys = [c for c in ["game_time_utc","created_at","date"] if c in today.columns]
            if sort_keys:
                today = today.sort_values(["game_id"] + sort_keys).groupby("game_id", as_index=False).tail(1)
            else:
                today = today.drop_duplicates(subset=["game_id"], keep="last")
        
    if today.empty:
        log.warning("Nothing to price for %s", args.date)
        return

    def pick_bundle(row):
        # Prefer bucketed model; fallback to ALL
        if _is_closed_row(row):
            p = Path(args.model_dir) / "ultra_bundle_closed.joblib"
            if p.exists(): return p
        else:
            p = Path(args.model_dir) / "ultra_bundle_open.joblib"
            if p.exists(): return p
        return Path(args.model_dir) / "ultra_bundle_all.joblib"

    preds = []
    for bucket, dfb in today.groupby(today.apply(lambda r: "closed" if _is_closed_row(r) else "open", axis=1)):
        bundle_path = pick_bundle(dfb.iloc[0])
        rhat = _predict_with_bundle(bundle_path, dfb)
        preds.append(pd.DataFrame({
            "game_id": dfb["game_id"].astype(str),
            "date": dfb["date"].astype(str),
            "market_total": pd.to_numeric(dfb["market_total"], errors="coerce"),
            "resid_hat": rhat
        }))
    out = pd.concat(preds, ignore_index=True)
    out["predicted_total"] = (out["market_total"] + out["resid_hat"]).clip(5.0, 13.0)

    # Stability gate with enhanced diagnostics
    resid_std = float(out["resid_hat"].std())
    resid_range = float(out["resid_hat"].max() - out["resid_hat"].min())
    
    if resid_std < 0.25:
        log.error("Residual std %.3f too low → HOLD day (feature misalignment?).", resid_std)
    
    log.info(f"Prediction diagnostics: σ(resid)={resid_std:.3f}, range={resid_range:.3f}, n_games={len(out)}")

    # Basic sanity on residuals & totals
    pt = out["market_total"] + out["resid_hat"]
    mu, sd = float(pt.mean()), float(pt.std())
    
    if not np.isfinite(resid_std) or resid_std < 0.20:
        log.warning("Residual std looks too low (%.3f) — check inputs/calibration", resid_std)
    if sd < 0.25 or mu < 6.0 or mu > 11.5:
        log.warning("Pred total sanity: mean=%.2f std=%.3f", mu, sd)

    # Include sigma in CSV for downstream EV
    try:
        # Pick the bucket used for the first row to read sigma (small approximation)
        first_bundle = pick_bundle(today.iloc[0])
        sigma = joblib.load(first_bundle).get("sigma", 0.90)
        out["sigma"] = sigma
    except Exception:
        out["sigma"] = 0.90

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out[["game_id","date","predicted_total","sigma"]].to_csv(args.out, index=False)
    log.info("Wrote %s (%d games, σ=%.3f)", args.out, len(out), out["sigma"].iloc[0])

    # Optional: upsert into DB
    if args.upsert:
        with engine.begin() as conn:
            conn.execute(text("""
                ALTER TABLE enhanced_games
                ADD COLUMN IF NOT EXISTS predicted_total_ultra NUMERIC;
            """))
            sql = text("""
                UPDATE enhanced_games eg
                   SET predicted_total_ultra = :pt
                 WHERE eg.game_id::text = :gid
                   AND eg.date = :d
            """)
            n = 0
            for r in out.to_dict(orient="records"):
                n += conn.execute(sql, {"pt": float(r["predicted_total"]), "gid": str(r["game_id"]), "d": r["date"]}).rowcount
        log.info("Upserted predicted_total_ultra for %d games", n)


def build_parser():
    """Build command line argument parser"""
    p = argparse.ArgumentParser(description='Ultra Sharp Pipeline - Residual vs Market Learning')
    
    sub = p.add_subparsers(dest='cmd', required=True)
    
    # Training command
    pt = sub.add_parser('train', help='Train residual models with roof bucketing')
    pt.add_argument('--db', required=True, help='Database URL')
    pt.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    pt.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    pt.add_argument('--model_dir', required=True, help='Model directory')
    
    # Pricing command
    pp = sub.add_parser('price', help='Price slate with residual models')
    pp.add_argument('--db', required=True, help='Database URL')
    pp.add_argument('--date', required=True, help='Date to price (YYYY-MM-DD)')
    pp.add_argument('--model_dir', required=True, help='Model directory')
    pp.add_argument('--out', required=True, help='Output predictions CSV file')
    pp.add_argument('--upsert', action='store_true', help='Write predictions back into enhanced_games.predicted_total_ultra')
    
    return p


if __name__ == '__main__':
    args = build_parser().parse_args()
    
    if args.cmd == 'train':
        cmd_train(args)
    elif args.cmd == 'price':
        cmd_price(args)
