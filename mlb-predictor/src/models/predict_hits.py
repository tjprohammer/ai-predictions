from __future__ import annotations

import argparse
from datetime import date, datetime, timezone

import pandas as pd

from src.models.common import encode_frame, load_feature_snapshots, load_latest_artifact
from src.utils.db import run_sql, upsert_rows
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)


def _fair_american(probability: float | None) -> int | None:
    if probability is None or probability <= 0 or probability >= 1:
        return None
    if probability >= 0.5:
        return int(round(-100 * probability / (1 - probability)))
    return int(round(100 * (1 - probability) / probability))


def main() -> int:
    parser = argparse.ArgumentParser(description="Score 1+ hit predictions from latest artifact")
    parser.add_argument("--target-date", help="Target date in YYYY-MM-DD format")
    args = parser.parse_args()
    target_date = date.fromisoformat(args.target_date) if args.target_date else date.today()

    settings = get_settings()
    try:
        artifact = load_latest_artifact("hits")
    except FileNotFoundError:
        log.info("No hits artifact found; attempting a training pass")
        from src.models.train_hits import main as train_hits_main

        train_hits_main()
        try:
            artifact = load_latest_artifact("hits")
        except FileNotFoundError:
            log.info("No hits artifact available after training attempt")
            return 0
    frame = load_feature_snapshots("hits")
    if frame.empty:
        log.info("No hits feature snapshots found")
        return 0

    scoring = frame[pd.to_datetime(frame["game_date"]).dt.date == target_date].copy()
    if scoring.empty:
        log.info("No hit features found for %s", target_date)
        return 0

    feature_columns = artifact["feature_columns"]
    X = encode_frame(scoring[feature_columns], artifact["category_columns"], artifact["training_columns"])
    probabilities = artifact["model"].predict_proba(X)[:, 1]
    prediction_ts = datetime.now(timezone.utc)
    rows = []
    for row, probability in zip(scoring.itertuples(index=False), probabilities):
        rows.append(
            {
                "game_id": int(row.game_id),
                "game_date": row.game_date,
                "player_id": int(row.player_id),
                "team": row.team,
                "prediction_ts": prediction_ts,
                "model_name": artifact["model_name"],
                "model_version": artifact["model_version"],
                "predicted_hit_probability": float(probability),
                "fair_price": _fair_american(float(probability)),
                "market_price": None,
                "edge": None,
            }
        )

    run_sql(
        """
        DELETE FROM predictions_player_hits
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
        "predictions_player_hits",
        rows,
        ["game_id", "player_id", "prediction_ts", "model_name", "model_version"],
    )
    output_dir = settings.report_dir / "hits"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"predictions_{target_date.isoformat()}.parquet"
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    log.info("Scored %s hit predictions -> %s", len(rows), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())