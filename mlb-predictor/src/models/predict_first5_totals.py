from __future__ import annotations

import argparse
import math
from datetime import date, datetime, timezone

import pandas as pd

from src.features.contracts import FIRST5_TOTALS_META_COLUMNS, FIRST5_TOTALS_TARGET_COLUMN
from src.models.common import encode_frame, load_feature_snapshots, load_latest_artifact
from src.utils.db import run_sql, upsert_rows
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def main() -> int:
    parser = argparse.ArgumentParser(description="Score first-five totals predictions from latest artifact")
    parser.add_argument("--target-date", help="Target date in YYYY-MM-DD format")
    args = parser.parse_args()
    target_date = date.fromisoformat(args.target_date) if args.target_date else date.today()

    settings = get_settings()
    try:
        artifact = load_latest_artifact("first5_totals")
    except FileNotFoundError:
        log.info("No first-five totals artifact found; attempting a training pass")
        from src.models.train_first5_totals import main as train_first5_totals_main

        train_first5_totals_main()
        try:
            artifact = load_latest_artifact("first5_totals")
        except FileNotFoundError:
            log.info("No first-five totals artifact available after training attempt")
            return 0
    frame = load_feature_snapshots("first5_totals")
    if frame.empty:
        log.info("No first-five totals feature snapshots found")
        return 0

    scoring = frame[pd.to_datetime(frame["game_date"]).dt.date == target_date].copy()
    if scoring.empty:
        log.info("No first-five totals features found for %s", target_date)
        return 0

    feature_columns = artifact["feature_columns"]
    X = encode_frame(scoring[feature_columns], artifact["category_columns"], artifact["training_columns"])
    predictions = artifact["model"].predict(X)
    residual_std = max(float(artifact.get("residual_std", 1.0)), 1.0)
    prediction_ts = datetime.now(timezone.utc)
    rows = []
    for row, predicted_total in zip(scoring.itertuples(index=False), predictions):
        over_probability = None
        under_probability = None
        edge = None
        if row.market_total is not None and not pd.isna(row.market_total):
            over_probability = _sigmoid((float(predicted_total) - float(row.market_total)) / residual_std)
            under_probability = 1.0 - over_probability
            edge = abs(over_probability - 0.5)
        rows.append(
            {
                "game_id": int(row.game_id),
                "game_date": row.game_date,
                "prediction_ts": prediction_ts,
                "model_name": artifact["model_name"],
                "model_version": artifact["model_version"],
                "predicted_total_runs": float(predicted_total),
                "over_probability": over_probability,
                "under_probability": under_probability,
                "market_total": row.market_total,
                "edge": edge,
            }
        )

    run_sql(
        """
        DELETE FROM predictions_first5_totals
        WHERE game_date = :target_date
          AND model_name = :model_name
          AND model_version = :model_version
        """,
        {
            "target_date": target_date,
            "model_name": artifact["model_name"],
            "model_version": artifact["model_version"],
        },
    )
    upsert_rows(
        "predictions_first5_totals",
        rows,
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