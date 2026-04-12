"""Score today's batters for total bases using the trained regression model.

Architecture mirrors predict_strikeouts.py — loads the latest artifact, scores
feature snapshots, and upserts results into predictions_total_bases.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.models.common import encode_frame, load_feature_snapshots, load_latest_artifact
from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.db import table_exists, upsert_rows
from src.utils.logging import get_logger


log = get_logger(__name__)


def _load_or_train_artifact() -> dict | None:
    try:
        return load_latest_artifact("total_bases")
    except FileNotFoundError:
        log.info("No total_bases artifact found; attempting training pass")
        from src.models.train_total_bases import main as train_main
        train_main()
        try:
            return load_latest_artifact("total_bases")
        except FileNotFoundError:
            log.info("No total_bases artifact available after training attempt")
            return None


def _over_probability_from_line(predicted: float, line: float, residual_std: float) -> float:
    """Approximate P(actual TB >= line + 0.5) assuming Gaussian residuals."""
    from scipy.stats import norm  # lazy import
    return float(norm.sf(line, loc=predicted, scale=max(residual_std, 0.3)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict batter total bases")
    add_date_range_args(parser)
    args = parser.parse_args()
    start_date, end_date = resolve_date_range(args)

    if not table_exists("predictions_total_bases"):
        log.warning("predictions_total_bases table not found — run DB migrations first")
        return 0

    artifact = _load_or_train_artifact()
    if artifact is None:
        log.info("No total_bases artifact available — skipping prediction")
        return 0

    frame = load_feature_snapshots("total_bases")
    if frame.empty:
        log.info("No total_bases feature rows for %s – %s", start_date, end_date)
        return 0
    if "game_date" not in frame.columns:
        log.warning("total_bases feature snapshots are missing game_date — skipping prediction")
        return 0

    game_dates = pd.to_datetime(frame["game_date"], errors="coerce").dt.date
    frame = frame[(game_dates >= start_date) & (game_dates <= end_date)].copy()
    if frame.empty:
        log.info("No total_bases feature rows for %s – %s", start_date, end_date)
        return 0

    feature_columns = artifact["feature_columns"]
    category_columns = artifact.get("category_columns", [])
    training_columns = artifact.get("training_columns")
    model = artifact["model"]
    residual_std = artifact.get("residual_std", 1.0)

    missing = [c for c in feature_columns if c not in frame.columns]
    if missing:
        log.warning("Missing %d feature column(s) — filling with NaN: %s", len(missing), missing[:5])
        for col in missing:
            frame[col] = np.nan

    X = encode_frame(frame[feature_columns], category_columns, training_columns=training_columns)
    predicted_tb = model.predict(X)

    prediction_ts = datetime.now(timezone.utc)
    rows = []
    for idx, record in enumerate(frame.to_dict(orient="records")):
        tb_pred = float(predicted_tb[idx])
        market_line = record.get("market_tb_line")
        edge = None
        over_prob = None
        under_prob = None
        if market_line is not None:
            try:
                over_prob = _over_probability_from_line(tb_pred, float(market_line), residual_std)
                under_prob = 1.0 - over_prob
                edge = round(tb_pred - float(market_line), 4)
            except Exception:
                pass
        rows.append({
            "game_id": record.get("game_id"),
            "game_date": record.get("game_date"),
            "player_id": record.get("player_id"),
            "player_name": record.get("player_name"),
            "team": record.get("team"),
            "prediction_ts": prediction_ts,
            "model_name": artifact.get("model_name", "unknown"),
            "model_version": artifact.get("model_version", "unknown"),
            "predicted_tb": round(tb_pred, 3),
            "over_probability": round(over_prob, 4) if over_prob is not None else None,
            "under_probability": round(under_prob, 4) if under_prob is not None else None,
            "market_line": market_line,
            "market_over_price": record.get("market_tb_over_price"),
            "market_under_price": record.get("market_tb_under_price"),
            "edge": edge,
        })

    if not rows:
        log.info("No predictions generated")
        return 0

    inserted = upsert_rows(
        "predictions_total_bases",
        rows,
        conflict_columns=["game_id", "player_id", "prediction_ts", "model_name", "model_version"],
    )
    log.info("Upserted %d total-bases predictions for %s – %s", inserted, start_date, end_date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
