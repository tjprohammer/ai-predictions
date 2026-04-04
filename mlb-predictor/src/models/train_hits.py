from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features.contracts import HITS_META_COLUMNS, HITS_TARGET_COLUMN
from src.models.common import chronological_split, encode_frame, load_feature_snapshots, save_artifact, save_report
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)

LOGISTIC_MAX_ITER = 4000


def _clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)


def _score_probabilities(y_true, probabilities: np.ndarray) -> dict[str, float]:
    clipped = _clip_probabilities(probabilities)
    return {
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, clipped)),
    }


def _build_logistic_classifier():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=LOGISTIC_MAX_ITER, solver="lbfgs"),
    )


def _fit_sigmoid_calibrator(probabilities: np.ndarray, y_true) -> LogisticRegression | None:
    unique_targets = set(int(value) for value in y_true)
    if len(unique_targets) < 2:
        return None
    calibrator = LogisticRegression(max_iter=LOGISTIC_MAX_ITER, solver="lbfgs")
    calibrator.fit(probabilities.reshape(-1, 1), y_true)
    return calibrator


def _fit_best_calibrator(calibration_probabilities: np.ndarray, y_calibration, evaluation_probabilities: np.ndarray, y_evaluation):
    calibration_metrics = {
        "identity": _score_probabilities(y_evaluation, evaluation_probabilities),
    }
    best_method = "identity"
    best_calibrator = None
    best_log_loss = calibration_metrics["identity"]["log_loss"]

    sigmoid_calibrator = _fit_sigmoid_calibrator(calibration_probabilities, y_calibration)
    if sigmoid_calibrator is not None:
        sigmoid_probabilities = sigmoid_calibrator.predict_proba(evaluation_probabilities.reshape(-1, 1))[:, 1]
        calibration_metrics["sigmoid"] = _score_probabilities(y_evaluation, sigmoid_probabilities)
        if calibration_metrics["sigmoid"]["log_loss"] < best_log_loss:
            best_method = "sigmoid"
            best_calibrator = sigmoid_calibrator
            best_log_loss = calibration_metrics["sigmoid"]["log_loss"]

    isotonic_calibrator = IsotonicRegression(out_of_bounds="clip")
    isotonic_calibrator.fit(calibration_probabilities, y_calibration)
    isotonic_probabilities = isotonic_calibrator.predict(evaluation_probabilities)
    calibration_metrics["isotonic"] = _score_probabilities(y_evaluation, isotonic_probabilities)
    if calibration_metrics["isotonic"]["log_loss"] < best_log_loss:
        best_method = "isotonic"
        best_calibrator = isotonic_calibrator

    return best_method, best_calibrator, calibration_metrics


def main() -> int:
    settings = get_settings()
    frame = load_feature_snapshots("hits")
    if frame.empty:
        log.info("No hits feature snapshots found")
        return 0

    trainable = frame[frame[HITS_TARGET_COLUMN].notna()].copy()
    if trainable.empty:
        log.info("No labeled hit rows found")
        return 0

    train_frame, val_frame = chronological_split(trainable)
    if val_frame.empty:
        log.info("Not enough hits rows for validation split")
        return 0

    calibration_frame, eval_frame = chronological_split(val_frame, validation_fraction=0.5)
    if eval_frame.empty:
        eval_frame = calibration_frame.copy()

    excluded_columns = set(HITS_META_COLUMNS + [HITS_TARGET_COLUMN, "player_name"])
    feature_columns = [column for column in trainable.columns if column not in excluded_columns]
    category_columns = [column for column in ["team", "opponent", "home_away"] if column in feature_columns]
    X_train = encode_frame(train_frame[feature_columns], category_columns)
    training_columns = list(X_train.columns)
    X_calibration = encode_frame(calibration_frame[feature_columns], category_columns, training_columns=training_columns)
    X_eval = encode_frame(eval_frame[feature_columns], category_columns, training_columns=training_columns)
    y_train = train_frame[HITS_TARGET_COLUMN].astype(int)
    y_calibration = calibration_frame[HITS_TARGET_COLUMN].astype(int)
    y_eval = eval_frame[HITS_TARGET_COLUMN].astype(int)

    candidates = {
        "logistic": _build_logistic_classifier(),
        "hgb": HistGradientBoostingClassifier(random_state=42, max_depth=4),
    }
    metrics = {}
    best_name = None
    best_model = None
    best_calibration_method = "identity"
    best_calibrator = None
    best_log_loss = float("inf")
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        calibration_probabilities = model.predict_proba(X_calibration)[:, 1]
        evaluation_probabilities = model.predict_proba(X_eval)[:, 1]
        calibration_method, calibrator, calibration_metrics = _fit_best_calibrator(
            calibration_probabilities,
            y_calibration,
            evaluation_probabilities,
            y_eval,
        )
        current_log_loss = calibration_metrics[calibration_method]["log_loss"]
        metrics[name] = {
            "calibration_method": calibration_method,
            "raw": _score_probabilities(y_eval, evaluation_probabilities),
            "calibrated": calibration_metrics[calibration_method],
            "calibration_candidates": calibration_metrics,
            "calibration_rows": int(len(calibration_frame)),
            "evaluation_rows": int(len(eval_frame)),
        }
        if current_log_loss < best_log_loss:
            best_log_loss = current_log_loss
            best_name = name
            best_model = model
            best_calibration_method = calibration_method
            best_calibrator = calibrator

    artifact_name = f"hits_{settings.model_version_prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    artifact = {
        "lane": "hits",
        "model_name": best_name,
        "model_version": artifact_name,
        "trained_at": datetime.now(timezone.utc),
        "feature_columns": feature_columns,
        "training_columns": training_columns,
        "category_columns": category_columns,
        "calibration_method": best_calibration_method,
        "metrics": metrics,
        "calibrator": best_calibrator,
        "model": best_model,
    }
    artifact_path = save_artifact("hits", artifact_name, artifact)
    report_path = save_report("hits", artifact_name, artifact | {"model": None, "calibrator": None})
    log.info("Saved hits artifact %s and report %s", artifact_path, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())