from __future__ import annotations

import argparse
import math
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from src.features.contracts import TOTALS_META_COLUMNS, TOTALS_TARGET_COLUMN
from src.models.common import calibrate_with_market, encode_frame, load_feature_snapshots, load_latest_artifact
from src.models.train_totals import compute_baseline
from src.utils.db import run_sql, upsert_rows
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Full-game totals lane status
# ---------------------------------------------------------------------------
# This lane is research-only: predictions are stored for auditing inputs and
# diagnosing the feature pipeline, but directional O/U picks should not be
# surfaced publicly until the lane proves signal.
LANE_STATUS = "research_only"

# ---------------------------------------------------------------------------
# Slate collapse detector
# ---------------------------------------------------------------------------
# When the model's predictions for an entire slate lack differentiation,
# every prediction on that slate is suppressed.  Thresholds are calibrated
# to the structural failure observed on 2025-04-06 (all ~5.9 flat).

_COLLAPSE_STD_THRESHOLD = 0.40        # std dev of predictions too low
_COLLAPSE_ONE_SIDE_THRESHOLD = 0.75   # >75 % of picks on one side
_COLLAPSE_RANGE_THRESHOLD = 1.5       # prediction max-min too narrow
_COLLAPSE_AVG_GAP_THRESHOLD = 1.5     # |avg(pred) - avg(market)| too large


def _detect_slate_collapse(
    predicted_totals: list[float],
    market_totals: list[float | None],
) -> str | None:
    """Return a collapse reason string if the slate looks degenerate, else None."""
    preds = np.array(predicted_totals, dtype=float)
    if len(preds) < 3:
        return None  # too few games to judge

    pred_std = float(np.std(preds))
    pred_range = float(np.ptp(preds))

    if pred_std < _COLLAPSE_STD_THRESHOLD:
        return f"slate_std_too_low({pred_std:.2f})"
    if pred_range < _COLLAPSE_RANGE_THRESHOLD:
        return f"slate_range_too_narrow({pred_range:.2f})"

    valid_markets = [m for m in market_totals if m is not None and not (isinstance(m, float) and math.isnan(m))]
    if valid_markets:
        sides = ["over" if p >= m else "under" for p, m in zip(preds, valid_markets) if m is not None]
        if sides:
            dominant = max(sides.count("over"), sides.count("under"))
            fraction = dominant / len(sides)
            if fraction > _COLLAPSE_ONE_SIDE_THRESHOLD:
                return f"slate_one_sided({fraction:.0%})"
            avg_gap = float(np.mean(preds[: len(valid_markets)]) - np.mean(valid_markets))
            if abs(avg_gap) > _COLLAPSE_AVG_GAP_THRESHOLD:
                return f"slate_avg_gap_extreme({avg_gap:+.2f})"

    return None


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _load_or_train_artifact() -> dict | None:
    try:
        return load_latest_artifact("totals")
    except FileNotFoundError:
        log.info("No totals artifact found; attempting a training pass")
        from src.models.train_totals import main as train_totals_main

        train_totals_main()
        try:
            return load_latest_artifact("totals")
        except FileNotFoundError:
            log.info("No totals artifact available after training attempt")
            return None


def _reload_artifact_after_failure(exc: Exception) -> dict | None:
    log.warning("Latest totals artifact failed to score with the current runtime; retraining and retrying once: %s", exc)
    from src.models.train_totals import main as train_totals_main

    train_totals_main()
    try:
        return load_latest_artifact("totals")
    except FileNotFoundError:
        log.info("No totals artifact available after retry training")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Score totals predictions from latest artifact")
    parser.add_argument("--target-date", help="Target date in YYYY-MM-DD format")
    args = parser.parse_args()
    target_date = date.fromisoformat(args.target_date) if args.target_date else date.today()

    settings = get_settings()
    artifact = _load_or_train_artifact()
    if artifact is None:
        return 0
    frame = load_feature_snapshots("totals")
    if frame.empty:
        log.info("No totals feature snapshots found")
        return 0

    scoring = frame[pd.to_datetime(frame["game_date"]).dt.date == target_date].copy()
    if scoring.empty:
        log.info("No totals features found for %s", target_date)
        return 0

    try:
        feature_columns = artifact["feature_columns"]
        X = encode_frame(scoring[feature_columns], artifact["category_columns"], artifact["training_columns"])
        raw_model_output = artifact["model"].predict(X)
        if artifact.get("architecture") == "baseline_plus_residual":
            baseline = compute_baseline(
                scoring,
                pd.Series(artifact["baseline_home_avgs"]),
                pd.Series(artifact["baseline_away_avgs"]),
                artifact["baseline_fallback"],
            )
            predictions = baseline + raw_model_output
        else:
            predictions = raw_model_output
    except Exception as exc:
        artifact = _reload_artifact_after_failure(exc)
        if artifact is None:
            return 0
        feature_columns = artifact["feature_columns"]
        X = encode_frame(scoring[feature_columns], artifact["category_columns"], artifact["training_columns"])
        raw_model_output = artifact["model"].predict(X)
        if artifact.get("architecture") == "baseline_plus_residual":
            baseline = compute_baseline(
                scoring,
                pd.Series(artifact["baseline_home_avgs"]),
                pd.Series(artifact["baseline_away_avgs"]),
                artifact["baseline_fallback"],
            )
            predictions = baseline + raw_model_output
        else:
            predictions = raw_model_output
    residual_std = max(float(artifact.get("residual_std", 1.0)), 1.0)
    market_calibrator = artifact.get("market_calibrator")
    calibrated_predictions, calibration_mask = calibrate_with_market(
        predictions, scoring["market_total"], market_calibrator,
    )
    calibration_residual_std = (
        max(float(market_calibrator["calibration_residual_std"]), 1.0)
        if market_calibrator is not None
        else residual_std
    )

    # --- Slate collapse detection ---
    market_list = [
        float(v) if v is not None and not pd.isna(v) else None
        for v in scoring["market_total"]
    ]
    collapse_reason = _detect_slate_collapse(
        [float(p) for p in calibrated_predictions],
        market_list,
    )
    if collapse_reason:
        log.warning("Slate collapse detected for %s: %s", target_date, collapse_reason)

    prediction_ts = datetime.now(timezone.utc)
    rows = []
    for idx, (row, raw_pred, cal_pred, was_calibrated) in enumerate(
        zip(
            scoring.itertuples(index=False),
            predictions,
            calibrated_predictions,
            calibration_mask,
        )
    ):
        predicted_total = float(cal_pred)
        effective_std = calibration_residual_std if was_calibrated else residual_std
        over_probability = None
        under_probability = None
        edge = None
        if row.market_total is not None and not pd.isna(row.market_total):
            over_probability = _sigmoid((predicted_total - float(row.market_total)) / effective_std)
            under_probability = 1.0 - over_probability
            edge = abs(over_probability - 0.5)

        # Confidence / suppression: full-game totals is research-only.
        # Within that, individual games may have additional suppress reasons.
        suppress_reason = collapse_reason or "lane_research_only"
        confidence_level = "suppress"

        rows.append(
            {
                "game_id": int(row.game_id),
                "game_date": row.game_date,
                "prediction_ts": prediction_ts,
                "model_name": artifact["model_name"],
                "model_version": artifact["model_version"],
                "predicted_total_runs": predicted_total,
                "over_probability": over_probability,
                "under_probability": under_probability,
                "market_total": row.market_total,
                "market_sportsbook": getattr(row, "market_sportsbook", None),
                "market_snapshot_ts": getattr(row, "line_snapshot_ts", None),
                "edge": edge,
                "confidence_level": confidence_level,
                "suppress_reason": suppress_reason,
                "lane_status": LANE_STATUS,
            }
        )
    log.info(
        "Full-game totals: %d predictions, lane_status=%s%s",
        len(rows),
        LANE_STATUS,
        f", slate_collapse={collapse_reason}" if collapse_reason else "",
    )

    run_sql(
        """
        DELETE FROM predictions_totals
        WHERE game_date = :target_date
        """,
        {
            "target_date": target_date,
        },
    )
    upsert_rows(
        "predictions_totals",
        rows,
        ["game_id", "prediction_ts", "model_name", "model_version"],
    )
    output_dir = settings.report_dir / "totals"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"predictions_{target_date.isoformat()}.parquet"
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    log.info("Scored %s totals predictions -> %s", len(rows), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())