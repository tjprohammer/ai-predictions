"""Hit calibration analysis — diagnose calibration issues and test corrections."""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss, mean_absolute_error

from src.features.contracts import (
    FIELD_ROLE_CORE_PREDICTOR,
    HITS_TARGET_COLUMN,
    feature_columns_for_roles,
)
from src.models.common import (
    chronological_split,
    encode_frame,
    load_feature_snapshots,
    load_latest_artifact,
)
from src.utils.logging import get_logger

log = get_logger(__name__)


def main():
    frame = load_feature_snapshots("hits")
    trainable = frame[frame[HITS_TARGET_COLUMN].notna()].copy()
    train_frame, val_frame = chronological_split(trainable)

    artifact = load_latest_artifact("hits")
    feature_columns = artifact["feature_columns"]
    category_columns = artifact.get("category_columns", [])

    X_val = encode_frame(
        val_frame[feature_columns],
        category_columns,
        artifact.get("training_columns", list(
            encode_frame(val_frame[feature_columns], category_columns).columns
        )),
    )
    y_val = val_frame[HITS_TARGET_COLUMN].astype(float).values  # binary: got_hit
    
    # Get raw probabilities
    model = artifact["model"]
    if hasattr(model, "predict_proba"):
        raw_probs = model.predict_proba(X_val)[:, 1]
    else:
        raw_probs = model.predict(X_val)
    
    print(f"=== HIT CALIBRATION ANALYSIS (val set: {len(y_val)} rows) ===")
    print(f"  Hit rate (actual):      {y_val.mean():.4f}")
    print(f"  Avg predicted prob:     {raw_probs.mean():.4f}")
    print(f"  Brier score:            {brier_score_loss(y_val, raw_probs):.4f}")
    try:
        print(f"  Log-loss:               {log_loss(y_val, raw_probs):.4f}")
    except Exception:
        pass
    print()

    # --- Calibration by probability bucket ---
    print("=== CALIBRATION BY PROBABILITY BUCKET ===")
    for lo, hi in [(0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]:
        mask = (raw_probs >= lo) & (raw_probs < hi)
        if mask.sum() == 0:
            continue
        actual_rate = y_val[mask].mean()
        pred_avg = raw_probs[mask].mean()
        label = f"{lo:.1f}-{hi:.1f}"
        cal_error = pred_avg - actual_rate
        print(f"  pred={label:8s} n={mask.sum():5d}  pred_avg={pred_avg:.3f}  actual_rate={actual_rate:.3f}  cal_error={cal_error:+.3f}")
    print()

    # --- Calibration by lineup confirmation status ---
    lineup_cols = [c for c in val_frame.columns if "confirmed" in c.lower() or "lineup" in c.lower()]
    print(f"  Lineup-related columns: {lineup_cols}")
    if "lineup_confirmed" in val_frame.columns:
        for status in [True, False, 1, 0]:
            mask = val_frame["lineup_confirmed"].values == status
            if mask.sum() == 0:
                continue
            actual_rate = y_val[mask].mean()
            pred_avg = raw_probs[mask].mean()
            print(f"  confirmed={status}: n={mask.sum()}, pred={pred_avg:.3f}, actual={actual_rate:.3f}, gap={pred_avg - actual_rate:+.3f}")
    elif "confirmed_hitters" in val_frame.columns:
        conf = val_frame["confirmed_hitters"].values
        for lo, hi, label in [(0, 1, "none"), (1, 5, "partial"), (5, 999, "full")]:
            mask = (conf >= lo) & (conf < hi)
            if mask.sum() == 0:
                continue
            actual_rate = y_val[mask].mean()
            pred_avg = raw_probs[mask].mean()
            print(f"  confirmed={label}: n={mask.sum()}, pred={pred_avg:.3f}, actual={actual_rate:.3f}, gap={pred_avg - actual_rate:+.3f}")
    print()

    # --- Test: Platt (sigmoid) calibration ---
    from sklearn.linear_model import LogisticRegression
    try:
        platt = LogisticRegression(max_iter=1000, solver="lbfgs")
        platt.fit(raw_probs.reshape(-1, 1), y_val)
        platt_probs = platt.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
        print("=== PLATT (SIGMOID) CALIBRATION ===")
        print(f"  Brier (raw):    {brier_score_loss(y_val, raw_probs):.4f}")
        print(f"  Brier (Platt):  {brier_score_loss(y_val, platt_probs):.4f}")
        print(f"  Log-loss (raw):   {log_loss(y_val, raw_probs):.4f}")
        print(f"  Log-loss (Platt): {log_loss(y_val, platt_probs):.4f}")
        print(f"  Avg prob (Platt): {platt_probs.mean():.4f}")
        print(f"  Platt coef={platt.coef_[0][0]:.3f}, intercept={platt.intercept_[0]:.3f}")
    except Exception as e:
        print(f"  Platt failed: {e}")
    print()

    # --- Test: Isotonic calibration ---
    from sklearn.isotonic import IsotonicRegression
    try:
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
        sort_idx = np.argsort(raw_probs)
        iso.fit(raw_probs[sort_idx], y_val[sort_idx])
        iso_probs = iso.predict(raw_probs)
        print("=== ISOTONIC CALIBRATION ===")
        print(f"  Brier (raw):      {brier_score_loss(y_val, raw_probs):.4f}")
        print(f"  Brier (isotonic): {brier_score_loss(y_val, iso_probs):.4f}")
        print(f"  Log-loss (raw):     {log_loss(y_val, raw_probs):.4f}")
        print(f"  Log-loss (isotonic):{log_loss(y_val, iso_probs):.4f}")
        print(f"  Avg prob (isotonic):{iso_probs.mean():.4f}")
    except Exception as e:
        print(f"  Isotonic failed: {e}")
    print()

    # --- Test: Simple shift (move mean prediction to match actual rate) ---
    shift = y_val.mean() - raw_probs.mean()
    shifted = np.clip(raw_probs + shift, 0.01, 0.99)
    print("=== SIMPLE SHIFT CALIBRATION ===")
    print(f"  Shift amount:     {shift:+.4f}")
    print(f"  Brier (raw):      {brier_score_loss(y_val, raw_probs):.4f}")
    print(f"  Brier (shifted):  {brier_score_loss(y_val, shifted):.4f}")
    print(f"  Log-loss (raw):     {log_loss(y_val, raw_probs):.4f}")
    print(f"  Log-loss (shifted): {log_loss(y_val, shifted):.4f}")
    print()

    # --- Check if existing artifact has calibration ---
    if "calibrator" in artifact:
        print(f"  Existing calibrator in artifact: {type(artifact['calibrator'])}")
    elif "calibration_method" in artifact:
        print(f"  Existing calibration method: {artifact['calibration_method']}")
    else:
        print("  No calibration in current artifact.")
    
    # --- Edge analysis: how does calibration affect betting decisions ---
    # Simulate market prices as fair odds
    if "market_price" in val_frame.columns:
        mkt_price = pd.to_numeric(val_frame["market_price"], errors="coerce")
        has_mkt = mkt_price.notna()
        if has_mkt.sum() > 0:
            # market_price is American odds, implied prob
            print(f"\n=== EDGE QUALITY WITH CALIBRATION ({has_mkt.sum()} w/ market) ===")
            # For hits, fair_price is model's implied line; market_price is the sportsbook line
            # A "play" is when model thinks player is more likely to get a hit than market implies


if __name__ == "__main__":
    main()
