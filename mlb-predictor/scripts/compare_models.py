"""One-off old vs new model comparison on the same feature snapshots."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error

from src.features.contracts import HITS_TARGET_COLUMN, TOTALS_TARGET_COLUMN
from src.models.common import encode_frame, load_feature_snapshots

MODEL_DIR = Path("data/models")


def _load(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def _compare_totals() -> None:
    print("=" * 60)
    print("TOTALS LANE COMPARISON")
    print("=" * 60)

    old = _load(MODEL_DIR / "totals" / "totals_v1_20260330_190448.pkl")
    new = _load(MODEL_DIR / "totals" / "totals_v1_20260404_172142.pkl")

    frame = load_feature_snapshots("totals")
    frame = frame[frame[TOTALS_TARGET_COLUMN].notna()].copy()
    print(f"Feature snapshots with actuals: {len(frame)} rows")
    print(f"Date range: {frame['game_date'].min()} to {frame['game_date'].max()}")
    print()

    X_old = encode_frame(frame[old["feature_columns"]], old["category_columns"], old["training_columns"])
    pred_old = old["model"].predict(X_old)

    X_new = encode_frame(frame[new["feature_columns"]], new["category_columns"], new["training_columns"])
    pred_new = new["model"].predict(X_new)

    actual = frame[TOTALS_TARGET_COLUMN].values.astype(float)

    print(f"Old model: {old['model_version']}")
    print(f"  Features: {len(old['feature_columns'])} columns")
    print(f"  MAE: {mean_absolute_error(actual, pred_old):.4f}")
    print(f"  Mean pred: {np.mean(pred_old):.3f}, Mean actual: {np.mean(actual):.3f}")
    print(f"  Bias (pred-actual): {np.mean(pred_old - actual):.4f}")

    print(f"New model: {new['model_version']}")
    print(f"  Features: {len(new['feature_columns'])} columns")
    print(f"  MAE: {mean_absolute_error(actual, pred_new):.4f}")
    print(f"  Mean pred: {np.mean(pred_new):.3f}, Mean actual: {np.mean(actual):.3f}")
    print(f"  Bias (pred-actual): {np.mean(pred_new - actual):.4f}")

    if "market_total" in frame.columns:
        market = frame["market_total"].values.astype(float)
        mask = ~np.isnan(market)
        if mask.sum() > 0:
            mkt_mae = mean_absolute_error(actual[mask], market[mask])
            print(f"Market line MAE: {mkt_mae:.4f} (n={mask.sum()})")
            old_beats = int((np.abs(pred_old[mask] - actual[mask]) < np.abs(market[mask] - actual[mask])).sum())
            new_beats = int((np.abs(pred_new[mask] - actual[mask]) < np.abs(market[mask] - actual[mask])).sum())
            print(f"Old beats market: {old_beats}/{mask.sum()} ({100 * old_beats / mask.sum():.1f}%)")
            print(f"New beats market: {new_beats}/{mask.sum()} ({100 * new_beats / mask.sum():.1f}%)")

    better = int((np.abs(pred_new - actual) < np.abs(pred_old - actual)).sum())
    worse = int((np.abs(pred_new - actual) > np.abs(pred_old - actual)).sum())
    tied = len(actual) - better - worse
    print(f"\nHead-to-head: New better on {better}, Old better on {worse}, Tied {tied} ({len(actual)} total)")


def _compare_hits() -> None:
    print()
    print("=" * 60)
    print("HITS LANE COMPARISON")
    print("=" * 60)

    old = _load(MODEL_DIR / "hits" / "hits_v1_20260330_190453.pkl")
    new = _load(MODEL_DIR / "hits" / "hits_v1_20260404_172558.pkl")

    frame = load_feature_snapshots("hits")
    frame = frame[frame[HITS_TARGET_COLUMN].notna()].copy()
    print(f"Feature snapshots with actuals: {len(frame)} rows")
    print(f"Date range: {frame['game_date'].min()} to {frame['game_date'].max()}")
    print()

    X_old = encode_frame(frame[old["feature_columns"]], old["category_columns"], old["training_columns"])
    prob_old = old["model"].predict_proba(X_old)[:, 1]

    X_new = encode_frame(frame[new["feature_columns"]], new["category_columns"], new["training_columns"])
    prob_new = new["model"].predict_proba(X_new)[:, 1]

    actual = frame[HITS_TARGET_COLUMN].values.astype(int)

    print(f"Old model: {old['model_version']}")
    print(f"  Features: {len(old['feature_columns'])} columns")
    print(f"  Brier score: {brier_score_loss(actual, prob_old):.6f}")
    print(f"  Log loss: {log_loss(actual, np.clip(prob_old, 1e-6, 1 - 1e-6)):.6f}")
    print(f"  Mean pred prob: {np.mean(prob_old):.4f}, Base rate: {np.mean(actual):.4f}")

    print(f"New model: {new['model_version']}")
    print(f"  Features: {len(new['feature_columns'])} columns")
    print(f"  Brier score: {brier_score_loss(actual, prob_new):.6f}")
    print(f"  Log loss: {log_loss(actual, np.clip(prob_new, 1e-6, 1 - 1e-6)):.6f}")
    print(f"  Mean pred prob: {np.mean(prob_new):.4f}, Base rate: {np.mean(actual):.4f}")

    # Calibration check at different thresholds
    print("\nCalibration (predicted bucket -> actual hit rate):")
    for lo, hi in [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]:
        mask_old = (prob_old >= lo) & (prob_old < hi)
        mask_new = (prob_new >= lo) & (prob_new < hi)
        actual_old = np.mean(actual[mask_old]) if mask_old.sum() > 0 else float("nan")
        actual_new = np.mean(actual[mask_new]) if mask_new.sum() > 0 else float("nan")
        print(f"  [{lo:.1f}, {hi:.1f}): Old {actual_old:.3f} (n={mask_old.sum()})  New {actual_new:.3f} (n={mask_new.sum()})")


def main() -> None:
    _compare_totals()
    _compare_hits()


if __name__ == "__main__":
    main()
