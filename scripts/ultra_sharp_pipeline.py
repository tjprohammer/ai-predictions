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
    return (roof_open is False) or ("closed" in roof_state)


def _prep_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leak-proof feature prep with robust drops + safe imputes."""
    X = df.copy()

    # --- drop target & outcome columns (including suffixed/mangled variants) ---
    leak_keys = ("resid", "total_runs", "market_total", "home_score", "away_score")
    to_drop = [c for c in X.columns
               if any(k in c.lower().replace(".", "_") for k in leak_keys)]
    if to_drop:
        log.warning(f"Dropping leak-like columns: {sorted(to_drop)[:8]}{'...' if len(to_drop)>8 else ''}")
    X.drop(columns=to_drop, errors="ignore", inplace=True)

    # drop any datetime-like cols
    for c in list(X.columns):
        if np.issubdtype(X[c].dtype, np.datetime64):
            log.warning(f"Dropping datetime column: {c}")
            X.drop(columns=[c], inplace=True, errors="ignore")

    # --- env defaults ---
    if "wind_out_cf" not in X.columns: X["wind_out_cf"] = np.nan
    if "air_density_proxy" not in X.columns: X["air_density_proxy"] = np.nan
    if "altitude_ft" not in X.columns: X["altitude_ft"] = 0.0

    # roof flags
    X["is_dome_or_closed"] = X.apply(_is_closed_row, axis=1).astype(int)
    X["is_retractable"] = X.get("roof_state", pd.Series([""]*len(X))).astype(str)\
                           .str.contains("unknown|retract", case=False, regex=True).astype(int)

    # imputes
    X["wind_out_cf"]      = pd.to_numeric(X["wind_out_cf"], errors="coerce").fillna(0.0)
    X["air_density_proxy"]= pd.to_numeric(X["air_density_proxy"], errors="coerce").fillna(1013.25/295.15)
    X["altitude_ft"]      = pd.to_numeric(X["altitude_ft"], errors="coerce").fillna(0.0)
    X["altitude_kft"]     = X["altitude_ft"] / 1000.0

    # seasonality
    if "month" in X.columns:
        X["month_sin"] = np.sin(2*np.pi*(X["month"].astype(float)-1)/12)
        X["month_cos"] = np.cos(2*np.pi*(X["month"].astype(float)-1)/12)
    else:
        X["month_sin"] = 0.0; X["month_cos"] = 0.0

    # Safe numeric conversions for all potential features
    numeric_cols = ['is_weekend', 'is_playoff', 'home_bp_fatigue', 'away_bp_fatigue',
                   'home_runs_l7', 'away_runs_l7', 'home_ra_l7', 'away_ra_l7',
                   'home_ops_l7', 'away_ops_l7', 'home_era_l7', 'away_era_l7',
                   'park_factor_runs', 'park_factor_hr',
                   'home_sp_era_l3', 'away_sp_era_l3', 'home_sp_whip_l3', 'away_sp_whip_l3']
    
    for col in numeric_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    # team/SP encodings (drop raw IDs later)
    for col in ["home_team", "away_team", "home_sp_id", "away_sp_id"]:
        if col in X.columns:
            X[f"{col}_encoded"] = X[col].astype('category').cat.codes

    # Convert boolean columns
    bool_cols = ['roof_open']
    for col in bool_cols:
        if col in X.columns:
            X[col] = X[col].astype(bool).astype(int)

    # drop raw IDs & helpers
    X.drop(columns=[c for c in X.columns if c.endswith("_id")] + 
                    ["game_id","date","created_at","roof_state"], errors="ignore", inplace=True)

    # final sanity
    for k in leak_keys:
        assert not any(k in c.lower().replace(".", "_") for c in X.columns), f"{k} still present"

    # Convert any remaining string columns to numeric or drop
    for col in list(X.columns):
        if X[col].dtype == 'object':
            try:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
            except:
                log.warning(f"Dropping non-numeric column: {col}")
                X = X.drop(columns=[col])

    return X



def _train_residual_model(df: pd.DataFrame, model_dir: Path, label_clip=4.0, bucket_name="all"):
    """Train residual model with proper time-ordered CV and leak-proof features"""
    need = {"total_runs","market_total","date"}
    if not need.issubset(df.columns): 
        raise RuntimeError(f"Missing cols: {need - set(df.columns)}")

    df = df.sort_values("date").reset_index(drop=True).copy()
    df["resid"] = (pd.to_numeric(df["total_runs"], errors="coerce") -
                   pd.to_numeric(df["market_total"], errors="coerce")).clip(-label_clip, label_clip)
    df = df.dropna(subset=["resid"])

    y = df["resid"].astype(float).values
    X = _prep_features(df)

    # sanity: target has spread
    y_std = float(np.std(y))
    log.info(f"Residual y: mean={np.mean(y):.3f}, std={y_std:.3f}, min={np.min(y):.2f}, max={np.max(y):.2f}")
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
        m.fit(X.iloc[tr_pos], y[tr_pos])
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
    log.info(f"OOF metrics: MAE={mae:.6f}, σ={sigma:.6f}")

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
        "trained_at": datetime.utcnow().isoformat(timespec="seconds")
    }
    outp = model_dir / f"ultra_bundle_{bucket_name}.joblib"
    joblib.dump(bundle, outp)
    return outp, mae, sigma


def _predict_with_bundle(bundle_path: Path, df_today: pd.DataFrame) -> np.ndarray:
    """Predict residuals with trained bundle using final calibrator"""
    b = joblib.load(bundle_path)
    X = (_prep_features(df_today)
         .reindex(columns=b["feature_cols"])
         .fillna(0))
    raw = np.mean([m.predict(X) for m in b["models"]], axis=0)
    preds = b["calibrator"].predict(raw)               # calibrated residuals
    return np.clip(preds, -b["label_clip"], b["label_clip"])


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def cmd_train(args):
    """Train residual models with roof bucketing"""
    log.info("🚀 Starting ultra sharp training (residual vs market)")
    engine = create_engine(args.db)
    
    # Enhanced query with aliased columns to avoid duplicates
    sql = """
      SELECT
        gc.*,
        eg.total_runs AS eg_total_runs,
        eg.market_total AS eg_market_total,
        eg.date AS eg_date
      FROM game_conditions gc
      JOIN enhanced_games eg ON gc.game_id::text = eg.game_id::text
      WHERE eg.date BETWEEN :s AND :e
        AND eg.total_runs IS NOT NULL
        AND eg.market_total IS NOT NULL
    """
    
    df = pd.read_sql(text(sql), engine, params={"s": args.start, "e": args.end})
    
    # Use aliased columns and rename for consistency
    if "eg_total_runs" in df.columns:
        df["total_runs"] = df["eg_total_runs"]
    if "eg_market_total" in df.columns:
        df["market_total"] = df["eg_market_total"]
    if "eg_date" in df.columns:
        df["date"] = df["eg_date"]
    
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
        "trained_at": datetime.utcnow().isoformat()
    }
    joblib.dump(registry, model_dir / "ultra_registry.joblib")
    log.info("✅ Models saved to %s | avg OOF-MAE=%.3f, avg σ=%.3f", model_dir, registry["avg_mae"], registry["avg_sigma"])


def cmd_price(args):
    """Price with ultra residual models"""
    log.info("🧮 Pricing with ultra residual models")
    engine = create_engine(args.db)
    reg = joblib.load(Path(args.model_dir) / "ultra_registry.joblib")

    # Today's slate with aliased columns to avoid duplicates
    q = """
      SELECT
        gc.*,
        eg.market_total AS eg_market_total,
        eg.date AS eg_date
      FROM game_conditions gc
      JOIN enhanced_games eg ON gc.game_id::text = eg.game_id::text
      WHERE eg.date = :d
        AND eg.market_total IS NOT NULL
        AND eg.total_runs IS NULL
    """
    today = pd.read_sql(text(q), engine, params={"d": args.date})
    
    # Use aliased columns and rename for consistency
    if "eg_market_total" in today.columns:
        today["market_total"] = today["eg_market_total"]
    if "eg_date" in today.columns:
        today["date"] = today["eg_date"]
        
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

    # Basic sanity on residuals & totals
    resid_std = float(out["resid_hat"].std())
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
