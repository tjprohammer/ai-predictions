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
  python ultra_sharp_pipeline.py train --db "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/ml        m = HistGradientBoostingRegressor(
            loss="absolute_error",      # <--- was default squared
            max_depth=4,                # <--- reduced from 5 for stability
            max_iter=800,
            learning_rate=0.05,
            l2_regularization=0.01,     # <--- added regularization
            min_samples_leaf=25,        # <--- increased from 20
            random_state=42
        )rt 2024-04-01 --end 2025-08-25 --model_dir models_ultra
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

# --- Stable categorical encoders (persisted in bundle) ---
_CAT_COLS = ["home_team", "away_team", "home_sp_id", "away_sp_id"]

def _fit_cat_encoders(df: pd.DataFrame) -> dict:
    """Fit stable categorical encoders that persist across train/serve."""
    enc = {}
    for c in _CAT_COLS:
        if c in df.columns:
            vals = df[c].astype(str).fillna("NA").unique().tolist()
            vals.sort()
            enc[c] = {v: i for i, v in enumerate(vals)}
    return enc

def _apply_cat_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Apply categorical encoders with consistent mapping."""
    Z = df.copy()
    for c, mp in encoders.items():
        if c in Z.columns:
            Z[f"{c}_encoded"] = Z[c].astype(str).map(mp).fillna(-1).astype(int)
    return Z
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy import create_engine, text

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Stable categorical encoders (persisted in bundle) ---
_CAT_COLS = ["home_team", "away_team", "home_sp_id", "away_sp_id", "plate_umpire"]

def _fit_cat_encoders(df: pd.DataFrame) -> dict:
    """Fit stable categorical encoders on training data."""
    enc = {}
    for c in _CAT_COLS:
        if c in df.columns:
            vals = df[c].astype(str).fillna("NA").unique().tolist()
            vals.sort()
            enc[c] = {v: i for i, v in enumerate(vals)}
    return enc

def _apply_cat_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Apply categorical encoders consistently at train and serve time."""
    Z = df.copy()
    for c, mp in encoders.items():
        if c in Z.columns:
            Z[f"{c}_encoded"] = Z[c].astype(str).map(mp).fillna(-1).astype(int)
    return Z

# ---------- Serve-time column harmonizer ----------
def _harmonize_serve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map EG aliases into the names our prep expects; derive roof_open if missing."""
    d = df.copy()
    mapping = {
        "total_runs": "eg_total_runs",
        "market_total": "eg_market_total",
        "opening_total": "eg_opening_total",
        "opening_over_odds": "eg_opening_over_odds",
        "opening_under_odds": "eg_opening_under_odds",
        "opening_captured_at": "eg_opening_captured_at",
        "opening_is_proxy": "eg_opening_is_proxy",
        "date": "eg_date",
        "home_sp_rest_days": "eg_home_sp_rest_days",
        "away_sp_rest_days": "eg_away_sp_rest_days",
        "roof_state": "eg_roof_state",
    }
    for to, frm in mapping.items():
        if to not in d.columns and frm in d.columns:
            d[to] = d[frm]

    # Derive month and is_weekend from date if missing
    if "date" in d.columns and "month" not in d.columns:
        dt = pd.to_datetime(d["date"], errors="coerce")
        d["month"] = dt.dt.month
        d["is_weekend"] = dt.dt.weekday.isin([5,6]).astype(int)

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
        # encoded variants (these will exist after _apply_cat_encoders)
        "home_team_encoded","away_team_encoded",
        "home_sp_id_encoded","away_sp_id_encoded",
        "plate_umpire_encoded",
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

    # --- Encode day/night binary ---
    if "day_night" in X.columns:
        dn = X["day_night"].astype(str).str.upper()
        X["is_day_game"] = (dn.str.startswith("D")).astype(int)
    else:
        X["is_day_game"] = 0

    # --- Encode pitcher handedness ---
    for c in ("home_sp_hand","away_sp_hand"):
        if c in X.columns:
            s = X[c].astype(str).str.upper().str[0]
            X[c + "_is_L"] = (s == "L").astype(int)
            X[c + "_is_R"] = (s == "R").astype(int)

    # --- Category encodings (handled by training-time encoders; see _fit/_apply_cat_encoders) ---
    # (categoricals now encoded consistently at train/serve time via stable mappings)

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

    # --- Enhanced Static Park Features (no weather needed) ---
    # Ballpark run factor interactions with team strength
    if {"ballpark_run_factor", "home_runs_pg_14_asof", "away_runs_pg_14_asof"} <= set(X.columns):
        team_run_avg = (X["home_runs_pg_14_asof"] + X["away_runs_pg_14_asof"]) / 2.0
        X["park_team_run_interaction"] = _num(X["ballpark_run_factor"]) * team_run_avg
    else:
        X["park_team_run_interaction"] = 0.0
    
    # HR factor with power pitching interaction
    if {"ballpark_hr_factor", "sp_era_diff_asof"} <= set(X.columns):
        # Lower ERA diff = better pitching = less HR vulnerability
        X["park_hr_pitching_interaction"] = _num(X["ballpark_hr_factor"]) * (-X["sp_era_diff_asof"])
    else:
        X["park_hr_pitching_interaction"] = 0.0
    
    # Altitude effects on run scoring (physics-based)
    if "altitude_kft" in X.columns:
        X["altitude_run_boost"] = X["altitude_kft"] * 0.1  # ~10% per 1000ft rule of thumb
    else:
        X["altitude_run_boost"] = 0.0

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



def _train_residual_model(df: pd.DataFrame, model_dir: Path, target_mode: str, label_clip=4.0, bucket_name="all"):
    """Train residual model with proper time-ordered CV and leak-proof features"""
    need = {"total_runs","market_total","date"}
    if not need.issubset(df.columns): 
        raise RuntimeError(f"Missing cols: {need - set(df.columns)}")

    df = df.sort_values("date").reset_index(drop=True).copy()
    
    # Normalize target mode consistently
    tm = (target_mode or "closing").lower()
    
    # Build raw target y BEFORE feature prep (so we can safely drop target-like cols later)
    if tm == "steam":
        y_raw = (pd.to_numeric(df.get("market_total"), errors="coerce") -
                 pd.to_numeric(df.get("opening_total"), errors="coerce"))
        ref_name = "closing_minus_opening"
    elif tm in ("open", "opening"):
        y_raw = (pd.to_numeric(df.get("total_runs"), errors="coerce") -
                 pd.to_numeric(df.get("opening_total"), errors="coerce"))
        ref_name = "opening_total"
    elif tm in ("close", "closing"):
        y_raw = (pd.to_numeric(df.get("total_runs"), errors="coerce") -
                 pd.to_numeric(df.get("market_total"), errors="coerce"))
        ref_name = "market_total"
    else:
        raise ValueError(f"Unknown target_mode={target_mode}")
    
    # robust, data-driven clip (3 * MAD) bounded to [2, 5]
    mad = np.median(np.abs(y_raw - np.median(y_raw))) if len(y_raw) else 0.0
    robust_sigma = 1.4826 * mad if mad > 0 else 3.0
    label_clip = float(np.clip(3.0 * robust_sigma, 2.0, 5.0))
    df["resid"] = y_raw.clip(-label_clip, label_clip)
    
    df = df.dropna(subset=["resid"])

    # Stable encoders learned on TRAIN space only
    encoders = _fit_cat_encoders(df)
    df_enc = _apply_cat_encoders(df, encoders)

    y = df_enc["resid"].astype(float).values
    X = _prep_features(df_enc)
    
    # Enhanced feature diagnostics and sanity checks
    if len(X):
        nunique = X.nunique().sum()
        log.info(f"Feature matrix: shape={X.shape}, total_unique_values={int(nunique)}")
        
        # Check for encoded categorical features
        encoded_features = [c for c in X.columns if c.endswith("_encoded")]
        if encoded_features:
            log.info(f"Encoded categorical features found: {encoded_features}")
            for feat in encoded_features:
                n_cats = X[feat].nunique()
                log.info(f"  {feat}: {n_cats} unique values")
        else:
            log.warning("No encoded categorical features found! Team/SP identity may be missing.")
        
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
            loss="absolute_error",      # <--- was default squared
            max_depth=4, max_iter=800,
            learning_rate=0.05,
            l2_regularization=0.0,
            min_samples_leaf=25,
            random_state=42
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

    # Smart calibration: only use isotonic if it actually helps
    from sklearn.metrics import mean_absolute_error
    mae_raw = float(mean_absolute_error(y[valid_mask], oof[valid_mask]))
    oof_cal = iso.predict(oof[valid_mask])
    mae_iso = float(mean_absolute_error(y[valid_mask], oof_cal))
    
    # Decision: use isotonic only if it improves MAE by meaningful margin
    if mae_iso + 0.02 >= mae_raw:
        calib_mode = "raw"         # skip iso - raw is good enough
        calib_alpha = 0.0
        log.info(f"Calibration: using RAW (mae_raw={mae_raw:.4f} vs mae_iso={mae_iso:.4f})")
        final_oof = oof[valid_mask]
    else:
        # Gentle blend instead of full isotonic
        calib_mode = "blend"
        calib_alpha = 0.5          # 50/50 blend
        log.info(f"Calibration: using BLEND (mae_raw={mae_raw:.4f} vs mae_iso={mae_iso:.4f})")
        final_oof = 0.5 * oof_cal + 0.5 * oof[valid_mask]

    # metrics on final calibrated predictions
    mae = float(mean_absolute_error(y[valid_mask], final_oof))
    sigma = float(np.std(y[valid_mask] - final_oof))
    
    # Additional diagnostics
    oof_corr = float(np.corrcoef(oof[valid_mask], y[valid_mask])[0,1]) if valid_mask.sum() > 10 else 0.0
    r2 = max(0.0, oof_corr**2)
    resid_std = float(np.std(final_oof))
    
    log.info(f"OOF metrics: MAE={mae:.6f}, σ={sigma:.6f}, corr={oof_corr:.3f}, R²={r2:.3f}, σ(pred)={resid_std:.3f}")

    # Log key feature statistics for debugging
    encoded_features = [c for c in X.columns if '_encoded' in c]
    if encoded_features:
        log.info(f"Categorical features in final model: {len(encoded_features)}")
        for feat in encoded_features:
            n_unique = X[feat].nunique()
            feat_std = float(X[feat].std()) if n_unique > 1 else 0.0
            log.info(f"  {feat}: {n_unique} unique, std={feat_std:.3f}")

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
        "calibration_mode": calib_mode,      # <--- NEW
        "calibration_alpha": calib_alpha,    # <--- NEW
        "encoders": encoders,        # <--- NEW
        "target": "residual",
        "target_mode": tm,          # <--- Use normalized mode
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
    """Predict residuals with trained bundle using smart calibration and per-game uncertainty"""
    b = joblib.load(bundle_path)

    # Apply the SAME encoders learned at train time
    df_today = _apply_cat_encoders(df_today, b.get("encoders", {}))

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
    
    # Get predictions from all models for uncertainty estimation
    per_model = np.vstack([m.predict(X) for m in b["models"]])  # [n_models, n_rows]
    raw = np.mean(per_model, axis=0)
    
    # Per-game uncertainty from model ensemble dispersion
    sigma_raw = np.std(per_model, axis=0) + 1e-6  # avoid zeros
    
    # Diagnostics
    raw_sd = float(np.std(raw))
    raw_rng = float(np.ptp(raw))
    log.info("Serve raw preds: range=%.3f, sd=%.3f (before calibrator)", raw_rng, raw_sd)
    
    # Smart calibration based on training decision
    mode = b.get("calibration_mode", "blend")
    alpha = b.get("calibration_alpha", 0.5)
    cal = b["calibrator"]
    
    if mode == "raw":
        preds = raw
        sigma_hat = sigma_raw  # no calibration adjustment
        log.info("Using RAW predictions (no calibration)")
    elif mode == "blend":
        iso_preds = cal.predict(raw)
        preds = alpha * iso_preds + (1-alpha) * raw
        # Adjust uncertainty for partial calibration
        sigma_hat = sigma_raw * (1.0 - 0.5*alpha)  # partial shrinkage
        log.info(f"Using BLEND predictions (α={alpha:.2f})")
    else:  # "iso"
        preds = cal.predict(raw)
        # Local isotonic slope for uncertainty scaling
        sigma_hat = _adjust_sigma_for_isotonic(sigma_raw, cal, raw)
        log.info("Using ISOTONIC predictions")
    
    # Cap excessive shrinkage as final guard
    if np.std(preds) < 0.4 * np.std(raw):
        log.warning("Excessive shrinkage detected, blending with raw")
        preds = 0.5 * preds + 0.5 * raw
        
    return np.clip(preds, -b["label_clip"], b["label_clip"]), sigma_hat


def _adjust_sigma_for_isotonic(sigma_raw: np.ndarray, calibrator, raw_preds: np.ndarray) -> np.ndarray:
    """Adjust uncertainty estimates for isotonic regression slope"""
    try:
        xt, yt = calibrator.X_thresholds_, calibrator.y_thresholds_
        # Compute slopes between knots
        slopes = np.diff(yt) / np.clip(np.diff(xt), 1e-6, None)
        # Map each prediction to its segment slope
        idx = np.searchsorted(xt[1:-1], raw_preds, side='right')
        local_slopes = slopes[np.clip(idx, 0, len(slopes)-1)]
        # Bound slopes to reasonable range
        local_slopes = np.clip(local_slopes, 0.2, 2.0)
        return sigma_raw * local_slopes
    except:
        # Fallback if isotonic introspection fails
        return sigma_raw


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def cmd_train(args):
    """Train residual models with roof bucketing"""
    log.info("🚀 Starting ultra sharp training (residual vs market)")
    
    # Determine target mode from CLI or environment
    target_mode = (args.target or os.getenv("ULTRA_TARGET", "closing")).lower()
    log.info(f"Training target: {target_mode}")
    
    engine = create_engine(args.db)
    
    # Use game_conditions + enhanced_games + pregame_features for comprehensive data
    sql = """
      SELECT
        gc.*,
        eg.total_runs,
        eg.market_total,
        eg.opening_total,
        eg.opening_is_proxy,
        eg.date,
        -- team and SP identity columns:
        eg.home_team,
        eg.away_team,
        eg.home_sp_id,
        eg.away_sp_id,
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
    
    # Filter proxy openings for opening/steam targets
    if target_mode in ("open","opening","steam"):
        if "opening_is_proxy" in df.columns:
            n0 = len(df)
            real_openings = df[(df["opening_is_proxy"].astype(str).str.lower() != "true") & (df["opening_total"].notna())]
            proxy_pct = 100 * (1 - len(real_openings) / max(1, n0))
            
            if proxy_pct > 50:
                log.warning(f"Opening target selected but {proxy_pct:.1f}% are proxies. Consider using closing target until more real opening lines available.")
                
            df = real_openings
            log.info("Filtered proxy openings: %d → %d rows (%.1f%% real openings)", n0, len(df), 100-proxy_pct)
        else:
            log.warning("opening_is_proxy not present; training on opening/steam may be contaminated.")
    
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
    p, mae, sigma = _train_residual_model(df, model_dir, target_mode=target_mode, bucket_name="all")
    log.info("ALL bucket: %d games, MAE=%.3f, σ=%.3f → %s", len(df), mae, sigma, p)
    paths.append(p)
    mae_all.append(mae)
    sigma_all.append(sigma)

    # Train OPEN bucket if enough data
    if len(df_open) > 300:
        p, mae, sigma = _train_residual_model(df_open, model_dir, target_mode=target_mode, bucket_name="open")
        log.info("OPEN bucket: %d games, MAE=%.3f, σ=%.3f → %s", len(df_open), mae, sigma, p)
        paths.append(p)
        mae_all.append(mae)
        sigma_all.append(sigma)
    else:
        log.warning("Only %d open games, skipping bucket", len(df_open))

    # Train CLOSED bucket if enough data  
    if len(df_closed) > 150:
        p, mae, sigma = _train_residual_model(df_closed, model_dir, target_mode=target_mode, bucket_name="closed")
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
        "target_mode": target_mode,  # <--- Use actual normalized mode
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
        eg.opening_total,
        eg.opening_is_proxy,
        eg.date,
        -- team and SP identity columns:
        eg.home_team,
        eg.away_team,
        eg.home_sp_id,
        eg.away_sp_id,
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
        rhat, sigma_hat = _predict_with_bundle(bundle_path, dfb)
        preds.append(pd.DataFrame({
            "game_id": dfb["game_id"].astype(str),
            "date": dfb["date"].astype(str),
            "market_total": pd.to_numeric(dfb["market_total"], errors="coerce"),
            "resid_hat": rhat,
            "sigma_hat": sigma_hat,
            "bucket": bucket
        }))
    out = pd.concat(preds, ignore_index=True)
    out["predicted_total"] = (out["market_total"] + out["resid_hat"]).clip(5.0, 13.0)

    # ROI-first high-conviction filtering
    from scipy.stats import norm
    
    # Z-edge: standardized prediction strength
    z_edge = np.abs(out["resid_hat"]) / np.maximum(1e-3, out["sigma_hat"])
    out["z_edge"] = z_edge
    
    # Convert to directional probability (normal assumption)
    p_dir = norm.cdf(z_edge)
    out["prob_dir"] = p_dir
    
    # Expected value at -110 odds
    win_ret = 100.0/110.0  # 0.909
    ev = p_dir * win_ret - (1 - p_dir) * 1.0
    out["ev_110"] = ev
    
    # Enhanced high-conviction filters with stability and hygiene checks
    
    # Add plate umpire hygiene filter
    has_umpire = today.get("plate_umpire", pd.Series([None]*len(today))).notna()
    
    # Add roof status hygiene for weather-sensitive predictions
    closed_roof = today.get("roof_state", today.get("roof_status", "unknown")).isin(["closed", "dome"])
    
    # Add SP coverage filter (require as-of ERA for both sides)
    sp_coverage = (
        today.get("home_sp_era_l3_asof", pd.Series([None]*len(today))).notna() &
        today.get("away_sp_era_l3_asof", pd.Series([None]*len(today))).notna()
    )
    
    # Calculate prediction stability (sign agreement across models) 
    # Use model dispersion as proxy for stability
    stability = 1.0 - np.minimum(1.0, out["sigma_hat"] / np.maximum(0.5, np.abs(out["resid_hat"])))
    out["stability"] = stability
    
    # Conservative high-conviction filters
    high_conv_mask = (
        (z_edge >= 1.2) &                           # ~60%+ directional confidence  
        (np.abs(out["resid_hat"]) >= 0.5) &         # magnitude floor
        (out["sigma_hat"] <= 2.0) &                 # reasonable uncertainty
        (stability >= 0.4) &                        # relaxed stability threshold
        # has_umpire &                             # skip for now - data missing
        sp_coverage                                 # SP ERA coverage for both teams
    )
    
    # Optional: closed domes only (for early production safety)
    if os.getenv("ULTRA_CLOSED_ONLY", "false").lower() in ("1", "true", "yes"):
        high_conv_mask = high_conv_mask & closed_roof
        log.info("ULTRA_CLOSED_ONLY=true: filtering to closed/dome games only")
    
    # Optional: require plate umpire data if available
    if os.getenv("ULTRA_REQUIRE_UMPIRE", "false").lower() in ("1", "true", "yes"):
        high_conv_mask = high_conv_mask & has_umpire
        log.info("ULTRA_REQUIRE_UMPIRE=true: requiring plate umpire data")
    
    out["high_confidence"] = high_conv_mask.astype(int)
    n_high_conf = high_conv_mask.sum()
    
    # Enhanced diagnostics
    resid_std = float(out["resid_hat"].std())
    resid_range = float(out["resid_hat"].max() - out["resid_hat"].min())
    
    if n_high_conf > 0:
        avg_z = float(out.loc[high_conv_mask, "z_edge"].mean())
        avg_prob = float(out.loc[high_conv_mask, "prob_dir"].mean())
        avg_ev = float(out.loc[high_conv_mask, "ev_110"].mean())
        log.info(f"High-confidence predictions: {n_high_conf}/{len(out)} games ({100*n_high_conf/len(out):.1f}%)")
        log.info(f"High-conf metrics: avg_z={avg_z:.2f}, avg_prob={avg_prob:.3f}, avg_ev={avg_ev:.3f}")
    else:
        log.warning("No high-confidence predictions found (z_edge >= 1.2, |resid| >= 0.5)")
    
    if resid_std < 0.15:  # Lowered threshold for more realistic expectations
        log.error("Residual std %.3f too low → HOLD day (feature misalignment?).", resid_std)
    
    log.info(f"Prediction diagnostics: σ(resid)={resid_std:.3f}, range={resid_range:.3f}, n_games={len(out)}")
    # Will be determined after loading bundle metadata below

    # Basic sanity on residuals & totals
    pt = out["market_total"] + out["resid_hat"]
    mu, sd = float(pt.mean()), float(pt.std())
    
    if not np.isfinite(resid_std) or resid_std < 0.20:
        log.warning("Residual std looks too low (%.3f) — check inputs/calibration", resid_std)
    if sd < 0.25 or mu < 6.0 or mu > 11.5:
        log.warning("Pred total sanity: mean=%.2f std=%.3f", mu, sd)

    # Include sigma and calibration mode in output
    try:
        # Pick the bucket used for the first row to read metadata
        first_bundle = pick_bundle(today.iloc[0])
        bundle_data = joblib.load(first_bundle)
        sigma = bundle_data.get("sigma", 0.90)
        calibration_mode = bundle_data.get("calibration_mode", "UNKNOWN")
        out["sigma"] = sigma
        log.info(f"Smart calibration: {calibration_mode} mode, ROI filters applied")
    except Exception:
        out["sigma"] = 0.90
        log.info("Smart calibration: UNKNOWN mode (bundle read failed), ROI filters applied")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    
    # Enhanced output with stability and hygiene flags
    output_cols = [
        "game_id", "date", "predicted_total", "resid_hat", "sigma_hat", 
        "z_edge", "prob_dir", "ev_110", "stability", "high_confidence", "sigma"
    ]
    
    # Add hygiene flag columns for transparency
    out["has_umpire"] = has_umpire.astype(int)
    out["sp_coverage"] = sp_coverage.astype(int) 
    out["closed_roof"] = closed_roof.astype(int)
    
    # Include hygiene flags in output for analysis
    output_cols.extend(["has_umpire", "sp_coverage", "closed_roof"])
    
    out[output_cols].to_csv(args.out, index=False)
    log.info("Wrote %s (%d games, σ=%.3f, %d high-confidence)", args.out, len(out), out["sigma"].iloc[0], out["high_confidence"].sum())

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
    pt.add_argument('--target', choices=['opening', 'closing', 'steam'], default=None, 
                    help='Target type (overrides ULTRA_TARGET env var)')
    
    # Pricing command
    pp = sub.add_parser('price', help='Price slate with residual models')
    pp.add_argument('--db', required=True, help='Database URL')
    pp.add_argument('--date', required=True, help='Date to price (YYYY-MM-DD)')
    pp.add_argument('--model_dir', required=True, help='Model directory')
    pp.add_argument('--out', required=True, help='Output predictions CSV file')
    pp.add_argument('--upsert', action='store_true', help='Write predictions back into enhanced_games.predicted_total_ultra')
    pp.add_argument('--target', choices=['opening', 'closing', 'steam'], default=None,
                    help='Target type for serving (should match training)')
    
    return p


if __name__ == '__main__':
    args = build_parser().parse_args()
    
    if args.cmd == 'train':
        cmd_train(args)
    elif args.cmd == 'price':
        cmd_price(args)
