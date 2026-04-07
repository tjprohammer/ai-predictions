"""Strikeout bias analysis — diagnose the over-prediction pattern."""
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.features.contracts import (
    FIELD_ROLE_CORE_PREDICTOR,
    STRIKEOUTS_TARGET_COLUMN,
    feature_columns_for_roles,
)
from src.models.common import (
    chronological_split,
    compute_sample_weights,
    encode_frame,
    load_feature_snapshots,
    load_latest_artifact,
)
from src.utils.logging import get_logger

log = get_logger(__name__)


def main():
    # Load features & artifact
    frame = load_feature_snapshots("strikeouts")
    trainable = frame[frame[STRIKEOUTS_TARGET_COLUMN].notna()].copy()
    train_frame, val_frame = chronological_split(trainable)

    artifact = load_latest_artifact("strikeouts")
    feature_columns = artifact["feature_columns"]
    category_columns = artifact["category_columns"]

    X_val = encode_frame(
        val_frame[feature_columns],
        category_columns,
        artifact["training_columns"],
    )
    y_val = val_frame[STRIKEOUTS_TARGET_COLUMN].astype(float).values
    preds = artifact["model"].predict(X_val)
    residuals = preds - y_val  # positive = over-predicted

    print(f"=== STRIKEOUT BIAS ANALYSIS (val set: {len(y_val)} rows) ===")
    print(f"  Mean residual (bias): {residuals.mean():+.3f}")
    print(f"  Median residual:      {np.median(residuals):+.3f}")
    print(f"  Std residual:         {residuals.std():.3f}")
    print(f"  MAE:                  {mean_absolute_error(y_val, preds):.3f}")
    print(f"  Avg prediction:       {preds.mean():.2f}")
    print(f"  Avg actual:           {y_val.mean():.2f}")
    print()

    # --- Bucketed analysis by actual strikeout range ---
    print("=== BY ACTUAL K BUCKET ===")
    for lo, hi in [(0, 2), (3, 4), (5, 6), (7, 8), (9, 99)]:
        mask = (y_val >= lo) & (y_val <= hi)
        if mask.sum() == 0:
            continue
        b_res = residuals[mask]
        label = f"{lo}-{hi}" if hi < 99 else f"{lo}+"
        print(f"  K={label:5s} n={mask.sum():4d}  bias={b_res.mean():+.2f}  mae={abs(b_res).mean():.2f}  pred_avg={preds[mask].mean():.2f}  actual_avg={y_val[mask].mean():.2f}")
    print()

    # --- Bucketed by predicted K range ---
    print("=== BY PREDICTED K BUCKET ===")
    for lo, hi in [(0, 3), (3, 4.5), (4.5, 6), (6, 7.5), (7.5, 99)]:
        mask = (preds >= lo) & (preds < hi)
        if mask.sum() == 0:
            continue
        b_res = residuals[mask]
        label = f"{lo:.1f}-{hi:.1f}" if hi < 99 else f"{lo:.1f}+"
        print(f"  pred={label:10s} n={mask.sum():4d}  bias={b_res.mean():+.2f}  mae={abs(b_res).mean():.2f}  pred_avg={preds[mask].mean():.2f}  actual_avg={y_val[mask].mean():.2f}")
    print()

    # --- Bucketed by baseline_strikeouts (starter tier proxy) ---
    if "baseline_strikeouts" in val_frame.columns:
        baseline = val_frame["baseline_strikeouts"].values
        print("=== BY BASELINE STRIKEOUTS (starter tier) ===")
        for lo, hi in [(0, 3), (3, 5), (5, 7), (7, 99)]:
            mask = (baseline >= lo) & (baseline < hi)
            if mask.sum() == 0:
                continue
            b_res = residuals[mask]
            label = f"{lo}-{hi}" if hi < 99 else f"{lo}+"
            print(f"  baseline={label:6s} n={mask.sum():4d}  bias={b_res.mean():+.2f}  mae={abs(b_res).mean():.2f}  pred_avg={preds[mask].mean():.2f}  actual_avg={y_val[mask].mean():.2f}")
        print()

    # --- Bucketed by projected_innings ---
    if "projected_innings" in val_frame.columns:
        proj = val_frame["projected_innings"].values
        print("=== BY PROJECTED INNINGS (leash proxy) ===")
        for lo, hi in [(0, 4.5), (4.5, 5.5), (5.5, 6.5), (6.5, 99)]:
            mask = (proj >= lo) & (proj < hi)
            if mask.sum() == 0:
                continue
            b_res = residuals[mask]
            label = f"{lo:.1f}-{hi:.1f}" if hi < 99 else f"{lo:.1f}+"
            print(f"  IP={label:10s} n={mask.sum():4d}  bias={b_res.mean():+.2f}  mae={abs(b_res).mean():.2f}")
        print()

    # --- Simple bias correction ---
    bias = residuals.mean()
    corrected = preds - bias
    mae_raw = mean_absolute_error(y_val, preds)
    mae_corrected = mean_absolute_error(y_val, corrected)
    print("=== SIMPLE BIAS CORRECTION ===")
    print(f"  Global bias: {bias:+.3f}")
    print(f"  MAE (raw):       {mae_raw:.3f}")
    print(f"  MAE (corrected): {mae_corrected:.3f}")
    print(f"  Improvement:     {mae_raw - mae_corrected:+.4f}")
    print()

    # --- Bucketed bias correction by baseline K tier ---
    if "baseline_strikeouts" in val_frame.columns:
        print("=== BUCKETED BIAS CORRECTION (by baseline K tier) ===")
        tiers = [(0, 3, "low"), (3, 5, "mid"), (5, 7, "high"), (7, 99, "ace")]
        corrected_bucketed = preds.copy()
        for lo, hi, name in tiers:
            mask = (baseline >= lo) & (baseline < hi)
            if mask.sum() < 5:
                continue
            tier_bias = residuals[mask].mean()
            corrected_bucketed[mask] -= tier_bias
            print(f"  {name:5s} (baseline {lo}-{hi}): bias={tier_bias:+.3f}, n={mask.sum()}")
        mae_bucketed = mean_absolute_error(y_val, corrected_bucketed)
        print(f"  MAE (bucketed correction): {mae_bucketed:.3f}")
        print(f"  Improvement over raw:      {mae_raw - mae_bucketed:+.4f}")
        print()

    # --- Market calibration (model+market blend) ---
    # Check if market lines exist in val features
    if "market_line" in val_frame.columns:
        mkt = pd.to_numeric(val_frame["market_line"], errors="coerce")
    else:
        # Try to get from the feature snapshots directly
        mkt = None
        for col in val_frame.columns:
            if "market" in col.lower() and "line" in col.lower():
                mkt = pd.to_numeric(val_frame[col], errors="coerce")
                print(f"  Found market column: {col}")
                break
    
    if mkt is not None and mkt.notna().sum() > 0:
        has_mkt = mkt.notna()
        print(f"=== MARKET CALIBRATION ({has_mkt.sum()}/{len(mkt)} with market) ===")
        from sklearn.linear_model import Ridge
        X_cal = np.column_stack([preds[has_mkt.values], mkt[has_mkt].values])
        y_cal = y_val[has_mkt.values]
        cal = Ridge(alpha=1.0)
        cal.fit(X_cal, y_cal)
        cal_preds = np.full_like(preds, np.nan)
        cal_preds[has_mkt.values] = cal.predict(X_cal)
        cal_preds[~has_mkt.values] = preds[~has_mkt.values] - bias  # fallback simple correction for non-market
        mae_cal = mean_absolute_error(y_val[has_mkt.values], cal_preds[has_mkt.values])
        mae_raw_mkt = mean_absolute_error(y_val[has_mkt.values], preds[has_mkt.values])
        print(f"  Model weight:  {cal.coef_[0]:.3f}")
        print(f"  Market weight: {cal.coef_[1]:.3f}")
        print(f"  Intercept:     {cal.intercept_:.3f}")
        print(f"  MAE (raw, mkt subset):  {mae_raw_mkt:.3f}")
        print(f"  MAE (calibrated):       {mae_cal:.3f}")
        print(f"  Improvement:            {mae_raw_mkt - mae_cal:+.4f}")
    else:
        print("=== MARKET CALIBRATION: no market lines in val features ===")
    
    # --- Residual correction via isotonic regression ---
    try:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip")
        # Sort by prediction for isotonic fitting
        sort_idx = np.argsort(preds)
        iso.fit(preds[sort_idx], y_val[sort_idx])
        iso_preds = iso.predict(preds)
        mae_iso = mean_absolute_error(y_val, iso_preds)
        print(f"\n=== ISOTONIC RESIDUAL CORRECTION ===")
        print(f"  MAE (isotonic): {mae_iso:.3f}")
        print(f"  Improvement:    {mae_raw - mae_iso:+.4f}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
