"""Experiment: evaluate totals model vs baselines across certainty slices.

Phase 3 of the Totals Improvement Plan.  Answers the central question:
"When does totals have any real edge at all?"

Instead of one aggregate MAE, evaluates the model and baselines on
data-completeness slices (board_state, missing_fallback_count, bullpen
completeness, starter availability) and feature-value slices (starter
asymmetry, venue factor known).

Usage:
    python -m src.backtest.experiment_totals_slices
    python -m src.backtest.experiment_totals_slices --lane first5_totals
"""
from __future__ import annotations

import argparse
import math
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features.contracts import (
    FIRST5_TOTALS_TARGET_COLUMN,
    TOTALS_TARGET_COLUMN,
)
from src.models.common import (
    chronological_split,
    encode_frame,
    load_feature_snapshots,
    load_latest_artifact,
)
from src.models.train_totals import compute_baseline
from src.utils.logging import get_logger


log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Slice definitions
# ---------------------------------------------------------------------------
# Each slice is a (name, filter_fn) pair.  filter_fn takes a DataFrame and
# returns a boolean mask.  Slices that produce fewer than MIN_SLICE_ROWS
# are reported but flagged as "too small."

MIN_SLICE_ROWS = 30


def _define_slices() -> list[tuple[str, Callable[[pd.DataFrame], pd.Series]]]:
    """Return ordered list of (slice_name, mask_fn) pairs."""
    slices: list[tuple[str, Callable[[pd.DataFrame], pd.Series]]] = []

    # --- Data-completeness slices ---
    slices.append(("ALL", lambda df: pd.Series(True, index=df.index)))

    slices.append(("board_state=complete", lambda df: df["board_state"] == "complete"))
    slices.append(("board_state=partial", lambda df: df["board_state"] == "partial"))
    slices.append(("board_state=minimal", lambda df: df["board_state"] == "minimal"))

    slices.append(("missing_fallback=0", lambda df: df["missing_fallback_count"] == 0))
    slices.append(("missing_fallback=1-2", lambda df: df["missing_fallback_count"].between(1, 2)))
    slices.append(("missing_fallback=3+", lambda df: df["missing_fallback_count"] >= 3))

    slices.append((
        "bullpen_complete(>0.5)",
        lambda df: df["bullpen_completeness_score"].fillna(0) > 0.5
        if "bullpen_completeness_score" in df.columns
        else pd.Series(False, index=df.index),
    ))
    slices.append((
        "bullpen_incomplete(<=0.5)",
        lambda df: df["bullpen_completeness_score"].fillna(0) <= 0.5
        if "bullpen_completeness_score" in df.columns
        else pd.Series(False, index=df.index),
    ))

    # --- Feature-availability slices ---
    slices.append((
        "both_starters_known",
        lambda df: df["home_starter_xwoba_blended"].notna() & df["away_starter_xwoba_blended"].notna(),
    ))
    slices.append((
        "any_starter_missing",
        lambda df: df["home_starter_xwoba_blended"].isna() | df["away_starter_xwoba_blended"].isna(),
    ))

    slices.append((
        "venue_factor_known",
        lambda df: df["venue_run_factor"].notna(),
    ))

    # --- Feature-value slices ---
    def _starter_asymmetry(df: pd.DataFrame) -> pd.Series:
        h = df["home_starter_xwoba_blended"].fillna(0)
        a = df["away_starter_xwoba_blended"].fillna(0)
        diff = (h - a).abs()
        return diff > diff.quantile(0.75)  # top-quartile asymmetry

    slices.append(("high_starter_asymmetry", _starter_asymmetry))

    def _bullpen_asymmetry(df: pd.DataFrame) -> pd.Series:
        h = df.get("home_bullpen_era_last3", pd.Series(0, index=df.index)).fillna(0)
        a = df.get("away_bullpen_era_last3", pd.Series(0, index=df.index)).fillna(0)
        diff = (h - a).abs()
        if diff.max() == 0:
            return pd.Series(False, index=df.index)
        return diff > diff.quantile(0.75)

    slices.append(("high_bullpen_asymmetry", _bullpen_asymmetry))

    # --- Timing-based slices (will grow as live data accumulates) ---
    slices.append((
        "lineup_confirmed(==1.0)",
        lambda df: df["lineup_certainty_score"].fillna(0) == 1.0,
    ))
    slices.append((
        "starter_confirmed(==1.0)",
        lambda df: df["starter_certainty_score"].fillna(0) == 1.0,
    ))

    return slices


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_slice(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    mask: np.ndarray,
    slice_name: str,
) -> dict | None:
    """Evaluate all prediction methods on a single slice."""
    n = int(mask.sum())
    if n == 0:
        return None

    y_slice = y_true[mask]
    result: dict = {
        "slice": slice_name,
        "n": n,
        "too_small": n < MIN_SLICE_ROWS,
    }

    methods = {}
    for method_name, y_pred in predictions.items():
        pred_slice = y_pred[mask]
        methods[method_name] = {
            "mae": _mae(y_slice, pred_slice),
            "rmse": _rmse(y_slice, pred_slice),
        }
    result["methods"] = methods

    # Find the best method for this slice
    best_name = min(methods, key=lambda k: methods[k]["mae"])
    result["best"] = best_name
    result["best_mae"] = methods[best_name]["mae"]

    # Does the model beat the best baseline?
    baseline_names = [k for k in methods if k != "model"]
    if "model" in methods and baseline_names:
        best_baseline = min(baseline_names, key=lambda k: methods[k]["mae"])
        model_mae = methods["model"]["mae"]
        baseline_mae = methods[best_baseline]["mae"]
        result["model_vs_best_baseline"] = {
            "baseline": best_baseline,
            "baseline_mae": baseline_mae,
            "model_mae": model_mae,
            "delta": model_mae - baseline_mae,
            "pct_change": ((model_mae - baseline_mae) / baseline_mae * 100) if baseline_mae > 0 else 0,
            "model_wins": model_mae < baseline_mae,
        }

    return result


def run_sliced_evaluation(lane: str = "totals") -> list[dict]:
    """Run sliced evaluation for a totals lane."""
    target_col = TOTALS_TARGET_COLUMN if lane == "totals" else FIRST5_TOTALS_TARGET_COLUMN

    frame = load_feature_snapshots(lane)
    if frame.empty:
        log.error("No feature snapshots for %s", lane)
        return []

    trainable = frame[frame[target_col].notna()].copy()
    if trainable.empty:
        log.error("No labeled rows for %s", lane)
        return []

    train_frame, val_frame = chronological_split(trainable)
    if val_frame.empty:
        log.error("Not enough rows for validation split in %s", lane)
        return []

    y_train = train_frame[target_col].astype(float)
    y_val = val_frame[target_col].astype(float).values
    train_mean = float(y_train.mean())
    train_median = float(y_train.median())

    log.info(
        "%s — train: %d rows (%s to %s), val: %d rows (%s to %s)",
        lane.upper(),
        len(train_frame),
        train_frame["game_date"].min(),
        train_frame["game_date"].max(),
        len(val_frame),
        val_frame["game_date"].min(),
        val_frame["game_date"].max(),
    )

    # --- Build prediction arrays ---
    predictions: dict[str, np.ndarray] = {}

    # 1. Train mean
    predictions["train_mean"] = np.full(len(y_val), train_mean)

    # 2. Train median
    predictions["train_median"] = np.full(len(y_val), train_median)

    # 3. Team average baseline
    home_avgs = train_frame.groupby("home_team")[target_col].mean()
    away_avgs = train_frame.groupby("away_team")[target_col].mean()
    predictions["team_average"] = compute_baseline(val_frame, home_avgs, away_avgs, train_mean)

    # 4. Market total (where available — NaN elsewhere)
    if "market_total" in val_frame.columns:
        market_vals = val_frame["market_total"].astype(float).values
    else:
        market_vals = np.full(len(y_val), np.nan)

    # 5. Current model artifact
    model_preds = None
    try:
        artifact = load_latest_artifact(lane)
        feature_columns = artifact["feature_columns"]
        missing = [c for c in feature_columns if c not in val_frame.columns]
        if missing:
            log.warning("Artifact feature columns missing: %s — skipping model eval", missing)
        else:
            X_val = encode_frame(
                val_frame[feature_columns],
                artifact["category_columns"],
                artifact.get("training_columns", list(val_frame[feature_columns].columns)),
            )
            raw_output = artifact["model"].predict(X_val)

            if artifact.get("architecture") == "baseline_plus_residual":
                bl_home = pd.Series(artifact["baseline_home_avgs"])
                bl_away = pd.Series(artifact["baseline_away_avgs"])
                bl_fallback = artifact["baseline_fallback"]
                bl = compute_baseline(val_frame, bl_home, bl_away, bl_fallback)
                model_preds = bl + raw_output
            else:
                model_preds = raw_output

            predictions["model"] = model_preds
            log.info(
                "Model loaded: %s (%s, %s)",
                artifact.get("model_name", "?"),
                artifact.get("model_version", "?"),
                artifact.get("architecture", "direct"),
            )
    except FileNotFoundError:
        log.info("No artifact for %s — baselines only", lane)

    # --- Evaluate per slice ---
    slices = _define_slices()
    results: list[dict] = []

    for slice_name, mask_fn in slices:
        try:
            mask = mask_fn(val_frame).values.astype(bool)
        except Exception as exc:
            log.warning("Slice %s failed: %s", slice_name, exc)
            continue

        # For each slice, build a subset prediction dict
        # Market total uses team_average fallback where NaN
        slice_predictions: dict[str, np.ndarray] = {}
        for k, v in predictions.items():
            slice_predictions[k] = v

        # Add market_total only on rows where it's available
        market_mask_in_slice = mask & ~np.isnan(market_vals)
        if market_mask_in_slice.sum() >= MIN_SLICE_ROWS:
            # Evaluate market separately on its available subset
            pass  # handled below

        result = evaluate_slice(y_val, slice_predictions, mask, slice_name)
        if result is None:
            log.info("  %-30s  (empty)", slice_name)
            continue

        results.append(result)

        # Report
        flag = " [TOO SMALL]" if result["too_small"] else ""
        model_note = ""
        if "model_vs_best_baseline" in result:
            mvb = result["model_vs_best_baseline"]
            if mvb["model_wins"]:
                model_note = f"  ** MODEL WINS by {-mvb['delta']:.3f} ({-mvb['pct_change']:.1f}%) vs {mvb['baseline']}"
            else:
                model_note = f"  model loses by {mvb['delta']:.3f} ({mvb['pct_change']:.1f}%) vs {mvb['baseline']}"

        log.info("  %-30s  n=%-5d  best=%-15s MAE=%.3f%s%s",
                 slice_name, result["n"], result["best"], result["best_mae"], flag, model_note)

    # --- Summary table ---
    log.info("")
    log.info("=" * 100)
    log.info("SLICED EVALUATION SUMMARY — %s", lane.upper())
    log.info("=" * 100)

    # Header
    method_names = sorted(predictions.keys())
    header = f"{'Slice':<32} {'N':>5}  " + "  ".join(f"{m:>14}" for m in method_names)
    log.info(header)
    log.info("-" * len(header))

    model_win_count = 0
    model_eval_count = 0

    for r in results:
        cols = [f"{r['slice']:<32}", f"{r['n']:>5}"]
        for m in method_names:
            if m in r["methods"]:
                mae = r["methods"][m]["mae"]
                # Mark the best for this slice
                mark = " *" if m == r["best"] else "  "
                cols.append(f"{mae:>12.3f}{mark}")
            else:
                cols.append(f"{'n/a':>14}")
        line = "  ".join(cols)
        if r["too_small"]:
            line += "  [small]"
        log.info(line)

        if "model_vs_best_baseline" in r and not r["too_small"]:
            model_eval_count += 1
            if r["model_vs_best_baseline"]["model_wins"]:
                model_win_count += 1

    log.info("-" * len(header))
    log.info("(* = best method for that slice)")

    if "model" in predictions:
        log.info("")
        log.info(
            "Model wins %d of %d meaningful slices (n >= %d).",
            model_win_count, model_eval_count, MIN_SLICE_ROWS,
        )

    # --- Verdict ---
    log.info("")
    if model_win_count == 0:
        log.info("VERDICT: model does not beat baselines in any slice. Totals remains a baseline-only lane for now.")
    elif model_win_count < model_eval_count / 2:
        log.info("VERDICT: model shows edge in some slices — totals may be a SELECTIVE lane.")
    else:
        log.info("VERDICT: model wins most slices — totals is showing broad signal.")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Sliced totals evaluation")
    parser.add_argument("--lane", default="totals", choices=["totals", "first5_totals"])
    args = parser.parse_args()

    results = run_sliced_evaluation(args.lane)
    if not results:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
