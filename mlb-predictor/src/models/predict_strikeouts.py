from __future__ import annotations

import argparse
import math
from datetime import date, datetime, timezone

import pandas as pd

from src.models.common import calibrate_with_market, encode_frame, load_feature_snapshots, load_latest_artifact
from src.utils.db import query_df, run_sql, upsert_rows
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _coerce_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _expects_market_line(row: object) -> bool:
    for field_name in ("projected_innings", "recent_avg_ip_3", "recent_avg_ip_5"):
        value = _coerce_float(getattr(row, field_name, None))
        if value is not None and value >= 2.0:
            return True
    baseline = _coerce_float(getattr(row, "baseline_strikeouts", None))
    return baseline is not None and baseline >= 2.0


def _load_or_train_artifact() -> dict | None:
    try:
        return load_latest_artifact("strikeouts")
    except FileNotFoundError:
        log.info("No strikeout artifact found; attempting a training pass")
        from src.models.train_strikeouts import main as train_strikeouts_main

        train_strikeouts_main()
        try:
            return load_latest_artifact("strikeouts")
        except FileNotFoundError:
            log.info("No strikeout artifact available after training attempt")
            return None


def _reload_artifact_after_failure(exc: Exception) -> dict | None:
    log.warning("Latest strikeout artifact failed to score with the current runtime; retraining and retrying once: %s", exc)
    from src.models.train_strikeouts import main as train_strikeouts_main

    train_strikeouts_main()
    try:
        return load_latest_artifact("strikeouts")
    except FileNotFoundError:
        log.info("No strikeout artifact available after retry training")
        return None


def _fetch_market_map(target_date: date) -> dict[tuple[int, int], float]:
    frame = query_df(
        """
        WITH ranked AS (
            SELECT
                ppm.game_id,
                ppm.player_id,
                ppm.sportsbook,
                ppm.line_value,
                ppm.snapshot_ts,
                ROW_NUMBER() OVER (
                    PARTITION BY ppm.game_id, ppm.player_id, ppm.sportsbook, ppm.market_type
                    ORDER BY ppm.snapshot_ts DESC
                ) AS row_rank
            FROM player_prop_markets ppm
            WHERE ppm.game_date = :target_date
              AND ppm.market_type = 'pitcher_strikeouts'
        )
        SELECT game_id, player_id, line_value
        FROM ranked
        WHERE row_rank = 1
        """,
        {"target_date": target_date},
    )
    if frame.empty:
        return {}
    market_map = {}
    for (game_id, player_id), rows in frame.groupby(["game_id", "player_id"]):
        line_values = pd.to_numeric(rows["line_value"], errors="coerce").dropna()
        market_map[(int(game_id), int(player_id))] = round(float(line_values.median()), 2) if not line_values.empty else None
    return market_map


def main() -> int:
    parser = argparse.ArgumentParser(description="Score pitcher strikeout predictions from latest artifact")
    parser.add_argument("--target-date", help="Target date in YYYY-MM-DD format")
    args = parser.parse_args()
    target_date = date.fromisoformat(args.target_date) if args.target_date else date.today()

    settings = get_settings()
    artifact = _load_or_train_artifact()
    if artifact is None:
        return 0

    frame = load_feature_snapshots("strikeouts")
    if frame.empty:
        log.info("No strikeout feature snapshots found")
        return 0

    scoring = frame[pd.to_datetime(frame["game_date"]).dt.date == target_date].copy()
    if scoring.empty:
        log.info("No strikeout features found for %s", target_date)
        return 0

    try:
        feature_columns = artifact["feature_columns"]
        X = encode_frame(scoring[feature_columns], artifact["category_columns"], artifact["training_columns"])
        raw_predictions = artifact["model"].predict(X)
    except Exception as exc:
        artifact = _reload_artifact_after_failure(exc)
        if artifact is None:
            return 0
        feature_columns = artifact["feature_columns"]
        X = encode_frame(scoring[feature_columns], artifact["category_columns"], artifact["training_columns"])
        raw_predictions = artifact["model"].predict(X)

    # --- Apply isotonic correction (monotonic decompression) ---
    predictions = raw_predictions.copy()
    isotonic_calibrator = artifact.get("isotonic_calibrator")
    if isotonic_calibrator is not None:
        predictions = isotonic_calibrator.predict(predictions)
        log.info("Applied isotonic correction to %d predictions", len(predictions))

    # --- Apply market calibration where market lines are available ---
    market_calibrator = artifact.get("market_calibrator")
    residual_std = max(float(artifact.get("residual_std", 1.0)), 1.0)
    residual_std_calibrated = max(float(artifact.get("residual_std_calibrated", residual_std)), 1.0)
    prediction_ts = datetime.now(timezone.utc)
    market_map = _fetch_market_map(target_date)
    used_market_calibration = False

    # Build a Series of market lines aligned with scoring rows for batch calibration
    market_lines_series = pd.Series(
        [market_map.get((int(r.game_id), int(r.pitcher_id))) for r in scoring.itertuples(index=False)],
        dtype=float,
    )
    if market_calibrator is not None and market_lines_series.notna().any():
        calibrated, cal_mask = calibrate_with_market(
            predictions, market_lines_series, market_calibrator,
        )
        n_calibrated = int(cal_mask.sum())
        if n_calibrated > 0:
            predictions = calibrated
            used_market_calibration = True
            log.info("Applied market calibration to %d/%d predictions", n_calibrated, len(predictions))

    rows = []
    missing_market_lines = 0
    for row, predicted_strikeouts, market_line_val in zip(
        scoring.itertuples(index=False), predictions, market_lines_series
    ):
        market_line = None if pd.isna(market_line_val) else float(market_line_val)
        over_probability = None
        under_probability = None
        edge = None
        if market_line is not None:
            # Use the calibrated residual_std when market calibration was applied
            effective_std = residual_std_calibrated if used_market_calibration else residual_std
            over_probability = _sigmoid((float(predicted_strikeouts) - float(market_line)) / effective_std)
            under_probability = 1.0 - over_probability
            edge = abs(over_probability - 0.5)
        elif _expects_market_line(row):
            missing_market_lines += 1
        rows.append(
            {
                "game_id": int(row.game_id),
                "game_date": row.game_date,
                "pitcher_id": int(row.pitcher_id),
                "team": row.team,
                "prediction_ts": prediction_ts,
                "model_name": artifact["model_name"],
                "model_version": artifact["model_version"],
                "predicted_strikeouts": float(predicted_strikeouts),
                "over_probability": over_probability,
                "under_probability": under_probability,
                "market_line": market_line,
                "edge": edge,
            }
        )

    if missing_market_lines:
        log.warning(
            "Strikeout market coverage for %s is incomplete: %s of %s pitchers are missing market lines; probabilities and edge were left null",
            target_date,
            missing_market_lines,
            len(rows),
        )

    run_sql(
        """
        DELETE FROM predictions_pitcher_strikeouts
        WHERE game_date = :target_date
        """,
        {
            "target_date": target_date,
        },
    )
    upsert_rows(
        "predictions_pitcher_strikeouts",
        rows,
        ["game_id", "pitcher_id", "prediction_ts", "model_name", "model_version"],
    )
    output_dir = settings.report_dir / "strikeouts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"predictions_{target_date.isoformat()}.parquet"
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    log.info("Scored %s strikeout predictions -> %s", len(rows), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())