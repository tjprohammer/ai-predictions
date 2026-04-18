"""Train a binary classifier for P(bet wins) on team best-bet markets from ``prediction_outcomes_daily``.

Run after you have graded rows (e.g. post-``product_surfaces``). Produces
``{MODEL_DIR}/board_action_score/action_classifier.joblib`` consumed by ``board_action_score``.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.board_action_score import (
    artifact_paths,
    clear_loaded_model_cache,
    feature_vector_from_outcome_row,
    training_feature_names,
)
from src.utils.best_bets import BEST_BET_MARKET_KEYS
from src.utils.db import query_df
from src.utils.logging import get_logger
from src.utils.settings import get_settings

log = get_logger(__name__)

_DEFAULT_MIN_ROWS = 80


def _outcomes_frame(start: date, end: date):
    placeholders = ", ".join(f":m{i}" for i in range(len(BEST_BET_MARKET_KEYS)))
    params = {f"m{i}": mk for i, mk in enumerate(BEST_BET_MARKET_KEYS)}
    params["start"] = start
    params["end"] = end
    q = f"""
        SELECT game_date, market, probability, opposite_probability, market_line,
               success, meta_payload
        FROM prediction_outcomes_daily
        WHERE game_date BETWEEN :start AND :end
          AND market IN ({placeholders})
          AND success IS NOT NULL
    """
    return query_df(q, params)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train board action score from prediction_outcomes_daily")
    parser.add_argument("--min-rows", type=int, default=_DEFAULT_MIN_ROWS, help="Minimum graded rows to train")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=540,
        help="Train on outcomes from [today-lookback, yesterday]",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=max(30, args.lookback_days))

    try:
        frame = _outcomes_frame(start, end)
    except Exception as exc:
        log.warning("train_board_action_score: query failed (missing table?): %s", exc)
        return 0

    if frame is None or frame.empty or len(frame) < args.min_rows:
        log.warning(
            "train_board_action_score: insufficient rows (%s; need %s). Skipping.",
            0 if frame is None else len(frame),
            args.min_rows,
        )
        return 0

    X_list = []
    y_list = []
    for _, row in frame.iterrows():
        rd = row.to_dict()
        try:
            xv = feature_vector_from_outcome_row(rd)
            X_list.append(xv[0])
            y_list.append(1 if bool(rd.get("success")) else 0)
        except Exception:
            continue

    if len(y_list) < args.min_rows:
        log.warning("train_board_action_score: too few rows after feature build (%s)", len(y_list))
        return 0

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.int32)
    pos_rate = float(y.mean())
    log.info(
        "train_board_action_score: samples=%s positive_rate=%.3f date_range=%s..%s",
        len(y),
        pos_rate,
        start,
        end,
    )

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    try:
        proba = pipe.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba)
        log.info("train_board_action_score: train-set ROC-AUC=%.4f", auc)
    except Exception as exc:
        log.warning("train_board_action_score: metric failed: %s", exc)

    out_dir = settings.model_dir / "board_action_score"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact, meta_path = artifact_paths()

    import joblib

    joblib.dump(pipe, artifact)
    meta = {
        "feature_names": list(training_feature_names()),
        "train_start": start.isoformat(),
        "train_end": end.isoformat(),
        "n_samples": int(len(y)),
        "positive_rate": pos_rate,
        "markets": list(BEST_BET_MARKET_KEYS),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("train_board_action_score: wrote %s", artifact)
    clear_loaded_model_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
