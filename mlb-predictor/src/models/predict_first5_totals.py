from __future__ import annotations

import argparse
import math
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from src.features.contracts import FIRST5_TOTALS_META_COLUMNS, FIRST5_TOTALS_TARGET_COLUMN
from src.models.common import calibrate_with_market, encode_frame, load_feature_snapshots, load_latest_artifact
from src.utils.db import run_sql, upsert_rows
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


# ---------------------------------------------------------------------------
# Enhanced calibration helpers
# ---------------------------------------------------------------------------

_BOARD_STATE_MAP = {"complete": 2, "partial": 1, "minimal": 0}


def _apply_enhanced_calibrator(
    raw_predictions: np.ndarray,
    scoring_frame: pd.DataFrame,
    enhanced_calibrator: dict | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply enhanced calibrator (model + market + certainty signals).

    Returns ``(calibrated_predictions, used_mask)``.
    """
    calibrated = raw_predictions.copy()
    mask = np.zeros(len(raw_predictions), dtype=bool)

    if enhanced_calibrator is None:
        return calibrated, mask

    has_market = scoring_frame["market_total"].notna().values
    has_gap = (
        scoring_frame["starter_quality_gap"].notna().values
        if "starter_quality_gap" in scoring_frame.columns
        else has_market
    )
    usable = has_market & has_gap
    if not usable.any():
        return calibrated, mask

    board_numeric = (
        scoring_frame["board_state"]
        .map(_BOARD_STATE_MAP)
        .fillna(0)
        .astype(float)
        .values
    )

    X_cal = np.column_stack([
        raw_predictions[usable],
        scoring_frame.loc[scoring_frame.index[usable], "market_total"].astype(float).values,
        (
            scoring_frame.loc[scoring_frame.index[usable], "starter_quality_gap"].fillna(0).astype(float).values
            if "starter_quality_gap" in scoring_frame.columns
            else np.zeros(int(usable.sum()))
        ),
        scoring_frame.loc[scoring_frame.index[usable], "starter_certainty_score"].fillna(0).astype(float).values,
        board_numeric[usable],
    ])
    calibrated[usable] = enhanced_calibrator["calibrator"].predict(X_cal)
    mask[usable] = True
    return calibrated, mask


# ---------------------------------------------------------------------------
# Publish / suppress logic
# ---------------------------------------------------------------------------

def _compute_confidence_level(row) -> tuple[str, str | None]:
    """Determine confidence_level and suppress_reason for a prediction row.

    Returns (confidence_level, suppress_reason).
    confidence_level is one of: "high", "medium", "low", "suppress".
    suppress_reason is None when not suppressed.
    """
    starter_cert = getattr(row, "starter_certainty_score", None)
    starter_cert = float(starter_cert) if starter_cert is not None and not pd.isna(starter_cert) else 0.0
    lineup_cert = getattr(row, "lineup_certainty_score", None)
    if lineup_cert is not None and not pd.isna(lineup_cert):
        lineup_cert = float(lineup_cert)
    else:
        lineup_cert = None
    quality_gap = getattr(row, "starter_quality_gap", None)
    quality_gap = float(quality_gap) if quality_gap is not None and not pd.isna(quality_gap) else 0.0
    asymmetry = getattr(row, "starter_asymmetry_score", None)
    asymmetry = float(asymmetry) if asymmetry is not None and not pd.isna(asymmetry) else None
    board_state = getattr(row, "board_state", "minimal")
    if pd.isna(board_state):
        board_state = "minimal"
    market_total = getattr(row, "market_total", None)
    has_market = market_total is not None and not pd.isna(market_total)

    # Suppress: no starters known at all
    if starter_cert == 0.0:
        return "suppress", "no_starter_data"

    # Suppress: no market line available
    if not has_market:
        return "suppress", "no_market_line"

    # Use asymmetry score when available; fall back to quality_gap thresholds
    effective_asymmetry = asymmetry if asymmetry is not None else (quality_gap / 0.10)

    # High: actual/box-confirmed starters (high starter_cert), lineup not mushy, strong signal
    lineup_ok_for_high = lineup_cert is None or lineup_cert >= 0.4
    if (
        starter_cert >= 0.75
        and effective_asymmetry >= 0.15
        and board_state != "minimal"
        and lineup_ok_for_high
    ):
        return "high", None

    # Medium: some starter info + some mismatch signal
    if starter_cert >= 0.5 and effective_asymmetry >= 0.08:
        return "medium", None

    # Low: everything else that isn't suppressed
    return "low", None


def _asymmetry_bucket(row) -> str:
    """Classify a game into low / medium / high starter asymmetry."""
    asymmetry = getattr(row, "starter_asymmetry_score", None)
    if asymmetry is None or (isinstance(asymmetry, float) and pd.isna(asymmetry)):
        quality_gap = getattr(row, "starter_quality_gap", None)
        if quality_gap is None or (isinstance(quality_gap, float) and pd.isna(quality_gap)):
            return "unknown"
        asymmetry = float(quality_gap) / 0.10
    else:
        asymmetry = float(asymmetry)
    if asymmetry >= 0.30:
        return "high"
    if asymmetry >= 0.12:
        return "medium"
    return "low"


def _load_or_train_artifact() -> dict | None:
    try:
        return load_latest_artifact("first5_totals")
    except FileNotFoundError:
        log.info("No first-five totals artifact found; attempting a training pass")
        from src.models.train_first5_totals import main as train_first5_totals_main

        train_first5_totals_main()
        try:
            return load_latest_artifact("first5_totals")
        except FileNotFoundError:
            log.info("No first-five totals artifact available after training attempt")
            return None


def _reload_artifact_after_failure(exc: Exception) -> dict | None:
    log.warning("Latest first-five totals artifact failed to score with the current runtime; retraining and retrying once: %s", exc)
    from src.models.train_first5_totals import main as train_first5_totals_main

    train_first5_totals_main()
    try:
        return load_latest_artifact("first5_totals")
    except FileNotFoundError:
        log.info("No first-five totals artifact available after retry training")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Score first-five totals predictions from latest artifact")
    parser.add_argument("--target-date", help="Target date in YYYY-MM-DD format")
    args = parser.parse_args()
    target_date = date.fromisoformat(args.target_date) if args.target_date else date.today()

    settings = get_settings()
    artifact = _load_or_train_artifact()
    if artifact is None:
        return 0
    frame = load_feature_snapshots("first5_totals")
    if frame.empty:
        log.info("No first-five totals feature snapshots found")
        return 0

    scoring = frame[pd.to_datetime(frame["game_date"]).dt.date == target_date].copy()
    if scoring.empty:
        log.info("No first-five totals features found for %s", target_date)
        return 0

    try:
        feature_columns = artifact["feature_columns"]
        X = encode_frame(scoring[feature_columns], artifact["category_columns"], artifact["training_columns"])
        predictions = artifact["model"].predict(X)
    except Exception as exc:
        artifact = _reload_artifact_after_failure(exc)
        if artifact is None:
            return 0
        feature_columns = artifact["feature_columns"]
        X = encode_frame(scoring[feature_columns], artifact["category_columns"], artifact["training_columns"])
        predictions = artifact["model"].predict(X)
    residual_std = max(float(artifact.get("residual_std", 1.0)), 1.0)

    # Prefer enhanced calibrator; fall back to basic market calibrator
    enhanced_calibrator = artifact.get("enhanced_calibrator")
    market_calibrator = artifact.get("market_calibrator")

    if enhanced_calibrator is not None:
        calibrated_predictions, calibration_mask = _apply_enhanced_calibrator(
            predictions, scoring, enhanced_calibrator,
        )
        calibration_residual_std = max(float(enhanced_calibrator["calibration_residual_std"]), 1.0)
    elif market_calibrator is not None:
        calibrated_predictions, calibration_mask = calibrate_with_market(
            predictions, scoring["market_total"], market_calibrator,
        )
        calibration_residual_std = max(float(market_calibrator["calibration_residual_std"]), 1.0)
    else:
        # No calibrator — anchor to the market line so raw model bias
        # does not push every game to one side.
        _FALLBACK_RESIDUAL_SCALE = 0.50
        market_values = scoring["market_total"].values.astype(float)
        calibrated_predictions = np.where(
            np.isfinite(market_values),
            market_values + _FALLBACK_RESIDUAL_SCALE * (predictions - market_values),
            predictions,
        )
        calibration_mask = np.zeros(len(predictions), dtype=bool)
        calibration_residual_std = residual_std
    prediction_ts = datetime.now(timezone.utc)
    rows = []
    suppressed_count = 0
    for idx, (row, raw_pred, cal_pred, was_calibrated) in enumerate(
        zip(
            scoring.itertuples(index=False),
            predictions,
            calibrated_predictions,
            calibration_mask,
        )
    ):
        predicted_total = float(cal_pred)
        predicted_total_fundamentals = float(raw_pred)
        effective_std = calibration_residual_std if was_calibrated else residual_std
        over_probability = None
        under_probability = None
        edge = None
        if row.market_total is not None and not pd.isna(row.market_total):
            over_probability = _sigmoid((predicted_total - float(row.market_total)) / effective_std)
            under_probability = 1.0 - over_probability
            edge = over_probability - 0.5

        confidence_level, suppress_reason = _compute_confidence_level(row)
        if confidence_level == "suppress":
            suppressed_count += 1

        bucket = _asymmetry_bucket(row)

        rows.append(
            {
                "game_id": int(row.game_id),
                "game_date": row.game_date,
                "prediction_ts": prediction_ts,
                "model_name": artifact["model_name"],
                "model_version": artifact["model_version"],
                "predicted_total_runs": predicted_total,
                "predicted_total_fundamentals": predicted_total_fundamentals,
                "over_probability": over_probability,
                "under_probability": under_probability,
                "market_total": row.market_total,
                "market_sportsbook": getattr(row, "market_sportsbook", None),
                "market_snapshot_ts": getattr(row, "line_snapshot_ts", None),
                "edge": edge,
                "confidence_level": confidence_level,
                "suppress_reason": suppress_reason,
                "asymmetry_bucket": bucket,
            }
        )
    if suppressed_count:
        log.info("Suppressed %d of %d first5 predictions (insufficient certainty)", suppressed_count, len(rows))
    bucket_counts = {}
    for r in rows:
        b = r.get("asymmetry_bucket", "unknown")
        bucket_counts[b] = bucket_counts.get(b, 0) + 1
    log.info("First5 asymmetry buckets: %s", bucket_counts)

    # Strip non-DB columns before upsert; keep them for parquet
    db_rows = [{k: v for k, v in r.items() if k != "asymmetry_bucket"} for r in rows]
    run_sql(
        """
        DELETE FROM predictions_first5_totals
        WHERE game_date = :target_date
        """,
        {
            "target_date": target_date,
        },
    )
    upsert_rows(
        "predictions_first5_totals",
        db_rows,
        ["game_id", "prediction_ts", "model_name", "model_version"],
    )
    output_dir = settings.report_dir / "first5_totals"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"predictions_{target_date.isoformat()}.parquet"
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    log.info("Scored %s first-five totals predictions -> %s", len(rows), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())