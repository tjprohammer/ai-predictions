from __future__ import annotations

import math

import pandas as pd
from sklearn.metrics import log_loss

from src.features.contracts import HITS_META_COLUMNS, HITS_TARGET_COLUMN, TOTALS_META_COLUMNS, TOTALS_TARGET_COLUMN
from src.models.common import encode_frame, load_feature_snapshots, load_latest_artifact
from src.utils.db import upsert_rows
from src.utils.logging import get_logger


log = get_logger(__name__)


def _run_totals_backtest() -> int:
    artifact = load_latest_artifact("totals")
    frame = load_feature_snapshots("totals")
    frame = frame[frame[TOTALS_TARGET_COLUMN].notna()].copy()
    if frame.empty:
        return 0
    X = encode_frame(frame[artifact["feature_columns"]], artifact["category_columns"], artifact["training_columns"])
    predictions = artifact["model"].predict(X)
    rows = []
    for row, prediction in zip(frame.itertuples(index=False), predictions):
        actual = float(row.actual_total_runs)
        error = abs(float(prediction) - actual)
        rows.append(
            {
                "game_id": int(row.game_id),
                "game_date": row.game_date,
                "feature_cutoff_ts": row.feature_cutoff_ts,
                "model_name": artifact["model_name"],
                "model_version": artifact["model_version"],
                "predicted_total_runs": float(prediction),
                "actual_total_runs": int(actual),
                "market_total": row.market_total,
                "absolute_error": error,
                "squared_error": error * error,
            }
        )
    upsert_rows("backtest_totals", rows, ["game_id", "feature_cutoff_ts", "model_name", "model_version"])
    return len(rows)


def _run_hits_backtest() -> int:
    artifact = load_latest_artifact("hits")
    frame = load_feature_snapshots("hits")
    frame = frame[frame[HITS_TARGET_COLUMN].notna()].copy()
    if frame.empty:
        return 0
    X = encode_frame(frame[artifact["feature_columns"]], artifact["category_columns"], artifact["training_columns"])
    probabilities = artifact["model"].predict_proba(X)[:, 1]
    rows = []
    for row, probability in zip(frame.itertuples(index=False), probabilities):
        actual = bool(row.got_hit)
        clipped = min(max(float(probability), 1e-6), 1 - 1e-6)
        rows.append(
            {
                "game_id": int(row.game_id),
                "game_date": row.game_date,
                "player_id": int(row.player_id),
                "feature_cutoff_ts": row.feature_cutoff_ts,
                "model_name": artifact["model_name"],
                "model_version": artifact["model_version"],
                "predicted_hit_probability": float(probability),
                "actual_hit": actual,
                "brier_score": (float(probability) - float(actual)) ** 2,
                "log_loss": log_loss([int(actual)], [clipped], labels=[0, 1]),
            }
        )
    upsert_rows("backtest_player_hits", rows, ["game_id", "player_id", "feature_cutoff_ts", "model_name", "model_version"])
    return len(rows)


def main() -> int:
    totals_rows = _run_totals_backtest()
    hits_rows = _run_hits_backtest()
    log.info("Backtest stored %s totals rows and %s hits rows", totals_rows, hits_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())