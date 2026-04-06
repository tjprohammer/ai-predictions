"""Reproducible baseline benchmark for totals and first-5 totals lanes.

Phase 1 of the Totals Improvement Plan: freeze honest baselines so every
future experiment can be measured against the same yardstick.

Usage:
    # Freeze current baselines
    python -m src.backtest.benchmark_totals --freeze

    # Compare latest artifact against frozen baselines
    python -m src.backtest.benchmark_totals --compare
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features.contracts import (
    FIRST5_TOTALS_TARGET_COLUMN,
    TOTALS_TARGET_COLUMN,
)
from src.models.common import (
    chronological_split,
    load_feature_snapshots,
    load_latest_artifact,
)
from src.models.train_totals import compute_baseline
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)

BENCHMARK_FILENAME = "frozen_baselines.json"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _benchmark_one_lane(
    lane: str,
    target_col: str,
) -> dict:
    """Compute all baseline metrics for a single lane."""
    frame = load_feature_snapshots(lane)
    if frame.empty:
        log.info("No feature snapshots for %s", lane)
        return {}

    trainable = frame[frame[target_col].notna()].copy()
    if trainable.empty:
        log.info("No labeled rows for %s", lane)
        return {}

    train_frame, val_frame = chronological_split(trainable)
    if val_frame.empty:
        log.info("Not enough rows for validation split in %s", lane)
        return {}

    y_train = train_frame[target_col].astype(float)
    y_val = val_frame[target_col].astype(float)
    train_mean = float(y_train.mean())
    train_median = float(y_train.median())

    # Split metadata for reproducibility
    split_info = {
        "total_rows": len(trainable),
        "train_rows": len(train_frame),
        "val_rows": len(val_frame),
        "train_date_min": str(train_frame["game_date"].min()),
        "train_date_max": str(train_frame["game_date"].max()),
        "val_date_min": str(val_frame["game_date"].min()),
        "val_date_max": str(val_frame["game_date"].max()),
    }

    # --- Baselines ---
    baselines = {}

    # 1. Train mean
    baselines["train_mean"] = _mae_rmse(
        y_val.values, np.full(len(y_val), train_mean),
    )
    baselines["train_mean"]["value"] = train_mean

    # 2. Train median
    baselines["train_median"] = _mae_rmse(
        y_val.values, np.full(len(y_val), train_median),
    )
    baselines["train_median"]["value"] = train_median

    # 3. Team average (per-team lookup from training set)
    home_avgs = train_frame.groupby("home_team")[target_col].mean()
    away_avgs = train_frame.groupby("away_team")[target_col].mean()
    team_avg_preds = compute_baseline(val_frame, home_avgs, away_avgs, train_mean)
    baselines["team_average"] = _mae_rmse(y_val.values, team_avg_preds)

    # 4. Market total (where available)
    if "market_total" in val_frame.columns:
        market_mask = val_frame["market_total"].notna()
        if market_mask.sum() > 0:
            market_vals = val_frame.loc[
                market_mask.index[market_mask], "market_total"
            ].astype(float)
            market_metrics = _mae_rmse(
                y_val[market_mask.values].values, market_vals.values,
            )
            market_metrics["rows"] = int(market_mask.sum())
            baselines["market_total"] = market_metrics

    # --- Current model (if artifact exists) ---
    model_metrics = None
    try:
        artifact = load_latest_artifact(lane)
        from src.models.common import encode_frame

        feature_columns = artifact["feature_columns"]
        missing = [c for c in feature_columns if c not in val_frame.columns]
        if missing:
            log.warning("Artifact features missing from val_frame: %s", missing)
        else:
            X_val = encode_frame(
                val_frame[feature_columns],
                artifact["category_columns"],
                artifact["training_columns"],
            )
            raw_model_output = artifact["model"].predict(X_val)

            if artifact.get("architecture") == "baseline_plus_residual":
                bl_home = pd.Series(artifact["baseline_home_avgs"])
                bl_away = pd.Series(artifact["baseline_away_avgs"])
                bl_fallback = artifact["baseline_fallback"]
                bl = compute_baseline(val_frame, bl_home, bl_away, bl_fallback)
                predictions = bl + raw_model_output
            else:
                predictions = raw_model_output

            model_metrics = _mae_rmse(y_val.values, predictions)
            model_metrics["model_name"] = artifact.get("model_name", "unknown")
            model_metrics["model_version"] = artifact.get("model_version", "unknown")
            model_metrics["architecture"] = artifact.get("architecture", "direct")
    except FileNotFoundError:
        log.info("No artifact for %s — baselines only", lane)

    # --- Best baseline ---
    best_baseline_name = min(
        baselines, key=lambda k: baselines[k].get("mae", float("inf")),
    )
    best_baseline_mae = baselines[best_baseline_name]["mae"]

    beats_best_baseline = False
    if model_metrics and model_metrics["mae"] < best_baseline_mae:
        beats_best_baseline = True

    return {
        "lane": lane,
        "target_column": target_col,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "split": split_info,
        "baselines": baselines,
        "best_baseline": {
            "name": best_baseline_name,
            "mae": best_baseline_mae,
        },
        "current_model": model_metrics,
        "beats_best_baseline": beats_best_baseline,
    }


# ------------------------------------------------------------------
# Freeze
# ------------------------------------------------------------------

def freeze_baselines() -> Path:
    """Compute and persist frozen baseline benchmarks for totals lanes."""
    settings = get_settings()

    totals = _benchmark_one_lane("totals", TOTALS_TARGET_COLUMN)
    first5 = _benchmark_one_lane("first5_totals", FIRST5_TOTALS_TARGET_COLUMN)

    payload = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "totals": totals,
        "first5_totals": first5,
    }

    out_dir = settings.report_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / BENCHMARK_FILENAME
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # Print summary table
    for section_name, section in [("totals", totals), ("first5_totals", first5)]:
        if not section:
            continue
        log.info("=" * 60)
        log.info("FROZEN BASELINE — %s", section_name)
        log.info(
            "  Split: %s train, %s val  (%s → %s | %s → %s)",
            section["split"]["train_rows"],
            section["split"]["val_rows"],
            section["split"]["train_date_min"],
            section["split"]["train_date_max"],
            section["split"]["val_date_min"],
            section["split"]["val_date_max"],
        )
        log.info("  Baselines:")
        for name, vals in section["baselines"].items():
            extra = f"  (rows={vals['rows']})" if "rows" in vals else ""
            log.info("    %-15s MAE=%.3f  RMSE=%.3f%s", name, vals["mae"], vals["rmse"], extra)
        log.info("  Best baseline: %s (MAE=%.3f)", section["best_baseline"]["name"], section["best_baseline"]["mae"])
        if section["current_model"]:
            m = section["current_model"]
            status = "BEATS" if section["beats_best_baseline"] else "LOSES TO"
            log.info(
                "  Current model: %s (%s) MAE=%.3f — %s best baseline",
                m["model_name"], m.get("architecture", "direct"), m["mae"], status,
            )
        log.info("=" * 60)

    log.info("Frozen baselines saved to %s", out_path)
    return out_path


# ------------------------------------------------------------------
# Compare
# ------------------------------------------------------------------

def compare_against_frozen() -> None:
    """Load frozen baselines and compare the latest artifact against them."""
    settings = get_settings()
    frozen_path = settings.report_dir / BENCHMARK_FILENAME
    if not frozen_path.exists():
        log.error("No frozen baselines found at %s — run --freeze first", frozen_path)
        return

    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))

    for lane_key, target_col in [
        ("totals", TOTALS_TARGET_COLUMN),
        ("first5_totals", FIRST5_TOTALS_TARGET_COLUMN),
    ]:
        frozen_lane = frozen.get(lane_key, {})
        if not frozen_lane:
            continue

        current = _benchmark_one_lane(lane_key, target_col)
        if not current:
            continue

        frozen_best = frozen_lane["best_baseline"]["mae"]
        frozen_best_name = frozen_lane["best_baseline"]["name"]

        log.info("=" * 60)
        log.info("COMPARISON — %s", lane_key)
        log.info(
            "  Frozen best baseline: %s MAE=%.3f (from %s)",
            frozen_best_name, frozen_best, frozen_lane.get("frozen_at", "?"),
        )

        if current["current_model"]:
            m = current["current_model"]
            delta = m["mae"] - frozen_best
            direction = "WORSE" if delta > 0 else "BETTER"
            log.info(
                "  Current model: %s MAE=%.3f  (%s by %.3f vs frozen best baseline)",
                m["model_name"], m["mae"], direction, abs(delta),
            )

        # Compare current baselines for data drift detection
        for bname in ["train_mean", "train_median", "team_average"]:
            frozen_val = frozen_lane.get("baselines", {}).get(bname, {}).get("mae")
            current_val = current.get("baselines", {}).get(bname, {}).get("mae")
            if frozen_val is not None and current_val is not None:
                drift = current_val - frozen_val
                if abs(drift) > 0.05:
                    log.warning(
                        "  DATA DRIFT: %s MAE changed %.3f → %.3f (Δ=%.3f)",
                        bname, frozen_val, current_val, drift,
                    )
        log.info("=" * 60)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Totals baseline benchmark")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true", help="Freeze current baselines")
    group.add_argument("--compare", action="store_true", help="Compare latest artifact against frozen baselines")
    args = parser.parse_args()

    if args.freeze:
        freeze_baselines()
    else:
        compare_against_frozen()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
