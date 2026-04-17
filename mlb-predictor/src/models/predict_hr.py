from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from src.models.common import encode_frame, load_feature_snapshots, load_latest_artifact
from src.models.hr_reasoning import build_hr_reasoning_lines
from src.utils.db import query_df, run_sql, upsert_rows
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)

# Books label these differently; we take any priced YES on a HR prop.
HR_MARKET_TYPES = (
    "player_home_run",
    "to_hit_a_home_run",
    "home_run",
    "home_runs",
    "hr",
    "batter_home_runs",
)


def _load_or_train_artifact() -> dict | None:
    try:
        return load_latest_artifact("hr")
    except FileNotFoundError:
        log.info("No HR artifact found; attempting a training pass")
        from src.models.train_hr import main as train_hr_main

        # Do not use parent's sys.argv (--target-date from predict_hr breaks train_hr's argparse).
        train_hr_main([])
        try:
            return load_latest_artifact("hr")
        except FileNotFoundError:
            log.info("No HR artifact available after training attempt")
            return None


def _bind_list(prefix: str, values: list[object], params: dict[str, object]) -> str:
    placeholders: list[str] = []
    for index, value in enumerate(values):
        key = f"{prefix}_{index}"
        params[key] = value
        placeholders.append(f":{key}")
    return ", ".join(placeholders) or "NULL"


def _clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), 1e-5, 1.0 - 1e-5)


def _apply_probability_calibration(probabilities: np.ndarray, artifact: dict) -> np.ndarray:
    calibration_method = artifact.get("calibration_method") or "identity"
    calibrator = artifact.get("calibrator")
    if calibrator is None or calibration_method == "identity":
        return _clip_probabilities(probabilities)
    reshaped = np.asarray(probabilities, dtype=float).reshape(-1, 1)
    if calibration_method == "sigmoid":
        return _clip_probabilities(calibrator.predict_proba(reshaped)[:, 1])
    if calibration_method == "isotonic":
        return _clip_probabilities(calibrator.predict(np.asarray(probabilities, dtype=float)))
    return _clip_probabilities(probabilities)


def _implied_probability(american_price: float | None) -> float | None:
    if american_price is None:
        return None
    if american_price > 0:
        return 100.0 / (american_price + 100.0)
    if american_price < 0:
        absolute = abs(american_price)
        return absolute / (absolute + 100.0)
    return None


def _fair_american(probability: float | None) -> int | None:
    if probability is None or probability <= 0 or probability >= 1:
        return None
    # Below ~0.02% HR, American odds explode to useless placeholders (e.g. +99999900).
    if probability < 2e-4:
        return None
    if probability >= 0.5:
        return int(round(-100 * probability / (1 - probability)))
    raw = int(round(100 * (1 - probability) / probability))
    # Cap long-shot fair lines for storage / UI (still rare-event friendly).
    return raw if abs(raw) <= 500_000 else None


def _fetch_hr_market_map(target_date: date) -> dict[tuple[int, int], int | None]:
    params: dict[str, object] = {"target_date": target_date}
    market_type_placeholders = _bind_list("market_type", list(HR_MARKET_TYPES), params)
    frame = query_df(
        f"""
        WITH ranked AS (
            SELECT
                ppm.game_id,
                ppm.player_id,
                ppm.sportsbook,
                ppm.market_type,
                ppm.line_value,
                ppm.over_price,
                ppm.snapshot_ts,
                ROW_NUMBER() OVER (
                    PARTITION BY ppm.game_id, ppm.player_id, ppm.sportsbook, ppm.market_type
                    ORDER BY ppm.snapshot_ts DESC
                ) AS row_rank
            FROM player_prop_markets ppm
            WHERE ppm.game_date = :target_date
              AND ppm.market_type IN ({market_type_placeholders})
        )
        SELECT game_id, player_id, line_value, over_price
        FROM ranked
        WHERE row_rank = 1
          AND (line_value IS NULL OR line_value <= 0.5)
        """,
        params,
    )
    if frame.empty:
        return {}
    market_map: dict[tuple[int, int], int | None] = {}
    for (game_id, player_id), rows in frame.groupby(["game_id", "player_id"]):
        over_prices = pd.to_numeric(rows["over_price"], errors="coerce").dropna()
        market_map[(int(game_id), int(player_id))] = int(over_prices.max()) if not over_prices.empty else None
    return market_map


def main() -> int:
    parser = argparse.ArgumentParser(description="Score P(HR) from latest HR artifact")
    parser.add_argument("--target-date", help="Target date in YYYY-MM-DD format")
    args = parser.parse_args()
    target_date = date.fromisoformat(args.target_date) if args.target_date else date.today()

    settings = get_settings()
    artifact = _load_or_train_artifact()
    if artifact is None:
        return 0
    frame = load_feature_snapshots("hr")
    if frame.empty:
        log.info("No HR feature snapshots found")
        return 0

    scoring = frame[pd.to_datetime(frame["game_date"]).dt.date == target_date].copy()
    if scoring.empty:
        log.info("No HR features found for %s", target_date)
        return 0

    feature_columns = artifact["feature_columns"]
    X = encode_frame(scoring[feature_columns], artifact["category_columns"], artifact["training_columns"])
    raw_probabilities = artifact["model"].predict_proba(X)[:, 1]
    probabilities = _apply_probability_calibration(raw_probabilities, artifact)
    prediction_ts = datetime.now(timezone.utc)
    market_map = _fetch_hr_market_map(target_date)
    rows = []
    for idx, probability in enumerate(probabilities):
        row = scoring.iloc[idx]
        reasoning = build_hr_reasoning_lines(row.to_dict())
        market_price = market_map.get((int(row["game_id"]), int(row["player_id"])))
        implied_probability = _implied_probability(market_price)
        rows.append(
            {
                "game_id": int(row["game_id"]),
                "game_date": row["game_date"],
                "player_id": int(row["player_id"]),
                "team": row["team"],
                "prediction_ts": prediction_ts,
                "model_name": artifact["model_name"],
                "model_version": artifact["model_version"],
                "predicted_hr_probability": float(probability),
                "fair_price": _fair_american(float(probability)),
                "market_price": market_price,
                "edge": None if implied_probability is None else float(probability) - implied_probability,
                "reasoning_json": json.dumps(reasoning),
            }
        )

    run_sql(
        """
        DELETE FROM predictions_player_hr
        WHERE game_date = :target_date
        """,
        {"target_date": target_date},
    )
    upsert_rows(
        "predictions_player_hr",
        rows,
        ["game_id", "player_id", "prediction_ts", "model_name", "model_version"],
    )
    output_dir = settings.report_dir / "hr"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"predictions_{target_date.isoformat()}.parquet"
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    log.info("Scored %s HR predictions -> %s", len(rows), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
