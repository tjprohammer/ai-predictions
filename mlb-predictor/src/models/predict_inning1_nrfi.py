"""Score inning-1 NRFI probability for a slate (uses latest trained artifact)."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from src.models.common import encode_frame, load_feature_snapshots, load_latest_artifact
from src.utils.db import run_sql, upsert_rows
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)

_CLIP_EPS = 1e-5


def _clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), _CLIP_EPS, 1.0 - _CLIP_EPS)


def _apply_nrfi_calibration(probabilities: np.ndarray, artifact: dict) -> np.ndarray:
    calibration_method = artifact.get("calibration_method") or "identity"
    calibrator = artifact.get("calibrator")
    if calibrator is None or calibration_method == "identity":
        return _clip_probabilities(probabilities)
    raw = np.asarray(probabilities, dtype=float)
    if calibration_method == "sigmoid":
        return _clip_probabilities(calibrator.predict_proba(raw.reshape(-1, 1))[:, 1])
    if calibration_method == "isotonic":
        return _clip_probabilities(calibrator.predict(raw))
    return _clip_probabilities(probabilities)


def _confidence_level(row: object) -> tuple[str, str | None]:
    bs = getattr(row, "board_state", None)
    if str(bs or "").lower() == "minimal":
        return "suppress", "minimal board inputs"
    sc = getattr(row, "starter_certainty_score", None)
    sc = float(sc) if sc is not None and not pd.isna(sc) else 0.0
    if sc < 0.28:
        return "low", "low starter certainty"
    if sc < 0.45:
        return "medium", None
    return "high", None


def main() -> int:
    parser = argparse.ArgumentParser(description="Score inning-1 NRFI probabilities")
    parser.add_argument("--target-date", help="Target date in YYYY-MM-DD format")
    args = parser.parse_args()
    target_date = date.fromisoformat(args.target_date) if args.target_date else date.today()

    try:
        artifact = load_latest_artifact("inning1_nrfi")
    except FileNotFoundError:
        log.info("No inning1_nrfi model artifact — train with src.models.train_inning1_nrfi")
        return 0
    except Exception as exc:
        log.warning("Could not load inning1_nrfi artifact: %s", exc)
        return 0

    frame = load_feature_snapshots("inning1_nrfi")
    if frame.empty:
        log.info("No inning1_nrfi feature snapshots — run inning1_nrfi_builder after first5_totals")
        return 0

    scoring = frame[pd.to_datetime(frame["game_date"]).dt.date == target_date].copy()
    if scoring.empty:
        log.info("No inning1_nrfi features for %s", target_date)
        return 0

    feature_columns = artifact["feature_columns"]
    X = encode_frame(scoring[feature_columns], artifact["category_columns"], artifact["training_columns"])
    model = artifact["model"]
    raw_proba = model.predict_proba(X)[:, 1]
    proba = _apply_nrfi_calibration(raw_proba, artifact)

    prediction_ts = datetime.now(timezone.utc)
    rows = []
    for idx, row in enumerate(scoring.itertuples(index=False)):
        p_nrfi = float(proba[idx])
        p_yrfi = float(1.0 - p_nrfi)
        side = "nrfi" if p_nrfi >= 0.5 else "yrfi"
        conf, reason = _confidence_level(row)
        rows.append(
            {
                "game_id": int(row.game_id),
                "game_date": row.game_date,
                "prediction_ts": prediction_ts,
                "model_name": str(artifact["model_name"]),
                "model_version": str(artifact["model_version"]),
                "predicted_nrfi_probability": round(p_nrfi, 5),
                "predicted_yrfi_probability": round(p_yrfi, 5),
                "recommended_side": side,
                "confidence_level": conf,
                "suppress_reason": reason,
            }
        )

    run_sql(
        "DELETE FROM predictions_inning1_nrfi WHERE game_date = :target_date",
        {"target_date": target_date},
    )
    upsert_rows(
        "predictions_inning1_nrfi",
        rows,
        ["game_id", "prediction_ts", "model_name", "model_version"],
    )
    settings = get_settings()
    out_dir = settings.report_dir / "inning1_nrfi"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"predictions_{target_date.isoformat()}.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    log.info("Scored %s inning1_nrfi predictions -> %s", len(rows), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
