from __future__ import annotations

import argparse
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features.contracts import (
    FIELD_ROLE_CORE_PREDICTOR,
    FIELD_ROLE_ENVIRONMENT_CONTEXT,
    HR_TARGET_COLUMN,
    feature_columns_for_roles,
    feature_field_roles,
)
from src.models.common import chronological_split, compute_sample_weights, encode_frame, load_feature_snapshots, save_artifact, save_report
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)

LOGISTIC_MAX_ITER = 4000


def _clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    # Floor above 1e-6 so UI / storage does not cluster at ~0.0001% from numerical clipping alone.
    return np.clip(np.asarray(probabilities, dtype=float), 1e-5, 1.0 - 1e-5)


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


def _isotonic_eligible(
    isotonic_probabilities: np.ndarray,
    raw_evaluation_probabilities: np.ndarray,
) -> bool:
    """Isotonic can collapse to all zeros on rare-event HR data and still win log_loss."""
    iso = np.asarray(isotonic_probabilities, dtype=float)
    raw = np.asarray(raw_evaluation_probabilities, dtype=float)
    if iso.size == 0:
        return False
    if float(np.ptp(iso)) < 1e-12:
        return False
    if float(np.max(iso)) <= 1e-8 and float(np.max(raw)) > 1e-4:
        return False
    return True


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
    if (
        _isotonic_eligible(isotonic_probabilities, evaluation_probabilities)
        and calibration_metrics["isotonic"]["log_loss"] < best_log_loss
    ):
        best_method = "isotonic"
        best_calibrator = isotonic_calibrator

    return best_method, best_calibrator, calibration_metrics


def main(argv: list[str] | None = None) -> int:
    """Train HR classifier from feature snapshots.

    When called from ``predict_hr`` (no artifact yet), pass ``main([])`` so argparse
    does not inherit ``--target-date`` from the parent's ``sys.argv`` (that caused
    exit code 2 and failed Update Lineups & Markets at the predict_hr step).
    """
    parser = argparse.ArgumentParser(description="Train HR classifier from feature snapshots")
    parser.add_argument(
        "--train-end-date",
        metavar="YYYY-MM-DD",
        default=None,
        help="Use only rows with game_date on or before this date (excludes later games from training)",
    )
    parser.add_argument(
        "--train-start-date",
        metavar="YYYY-MM-DD",
        default=None,
        help="Use only rows with game_date on or after this date (e.g. season-opening window)",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    frame = load_feature_snapshots("hr")
    if frame.empty:
        log.info("No HR feature snapshots found — run src.features.hr_builder for a date range with games")
        return 0

    trainable = frame[frame[HR_TARGET_COLUMN].notna()].copy()
    if args.train_start_date:
        cutoff_start = date.fromisoformat(args.train_start_date)
        gd = pd.to_datetime(trainable["game_date"]).dt.date
        trainable = trainable[gd >= cutoff_start].copy()
        log.info("Filtered to game_date >= %s → %s rows", cutoff_start, len(trainable))
    if args.train_end_date:
        cutoff = date.fromisoformat(args.train_end_date)
        gd = pd.to_datetime(trainable["game_date"]).dt.date
        trainable = trainable[gd <= cutoff].copy()
        log.info("Filtered to game_date <= %s → %s rows", cutoff, len(trainable))
    if trainable.empty:
        log.info("No labeled HR rows (need completed games in player_game_batting, or relax --train-end-date)")
        return 0

    train_frame, val_frame = chronological_split(trainable)
    if val_frame.empty:
        log.info("Not enough HR rows for validation split")
        return 0

    calibration_frame, eval_frame = chronological_split(val_frame, validation_fraction=0.3)
    if eval_frame.empty:
        eval_frame = calibration_frame.copy()

    feature_columns = feature_columns_for_roles(
        "hr",
        [FIELD_ROLE_CORE_PREDICTOR, FIELD_ROLE_ENVIRONMENT_CONTEXT],
        available_columns=list(trainable.columns),
    )
    category_columns = [column for column in ["team", "opponent", "home_away"] if column in feature_columns]
    X_train = encode_frame(train_frame[feature_columns], category_columns)
    training_columns = list(X_train.columns)
    X_calibration = encode_frame(calibration_frame[feature_columns], category_columns, training_columns=training_columns)
    X_eval = encode_frame(eval_frame[feature_columns], category_columns, training_columns=training_columns)
    y_train = train_frame[HR_TARGET_COLUMN].astype(int)
    y_calibration = calibration_frame[HR_TARGET_COLUMN].astype(int)
    y_eval = eval_frame[HR_TARGET_COLUMN].astype(int)

    candidates = {
        "logistic": _build_logistic_classifier(),
        "hgb": HistGradientBoostingClassifier(random_state=42, max_depth=4),
        "hgb_shallow": HistGradientBoostingClassifier(random_state=42, max_depth=2, min_samples_leaf=80),
        "hgb_balanced": HistGradientBoostingClassifier(
            random_state=42,
            max_depth=3,
            min_samples_leaf=40,
            class_weight="balanced",
        ),
    }
    metrics = {}
    best_name = None
    best_model = None
    best_calibration_method = "identity"
    best_calibrator = None
    best_log_loss = float("inf")
    train_weights = compute_sample_weights(train_frame["game_date"])

    for name, model in candidates.items():
        if hasattr(model, "steps"):
            last_step_name = model.steps[-1][0]
            model.fit(X_train, y_train, **{f"{last_step_name}__sample_weight": train_weights})
        else:
            model.fit(X_train, y_train, sample_weight=train_weights)
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

    base_rate = float(y_train.mean())
    baselines = {
        "base_rate": {
            "brier": float(brier_score_loss(y_eval, [base_rate] * len(y_eval))),
            "log_loss": float(log_loss(y_eval, np.clip([base_rate] * len(y_eval), 1e-5, 1 - 1e-5), labels=[0, 1])),
        },
    }
    log.info("Baselines: %s", baselines)
    log.info(
        "Best HR model '%s' — raw: brier=%.4f log_loss=%.4f | calibrated (%s): brier=%.4f log_loss=%.4f",
        best_name,
        metrics[best_name]["raw"]["brier"],
        metrics[best_name]["raw"]["log_loss"],
        metrics[best_name]["calibration_method"],
        metrics[best_name]["calibrated"]["brier"],
        metrics[best_name]["calibrated"]["log_loss"],
    )

    model_brier = metrics[best_name]["calibrated"]["brier"]
    base_rate_brier = baselines["base_rate"]["brier"]
    lane_status = "below_baseline" if model_brier >= base_rate_brier else "above_baseline"
    if model_brier >= base_rate_brier:
        log.warning(
            "LANE STATUS: HR model (Brier=%.4f) does NOT beat base_rate (Brier=%.4f). Rare-event lane — check features/labels.",
            model_brier,
            base_rate_brier,
        )

    artifact_name = f"hr_{settings.model_version_prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    artifact = {
        "lane": "hr",
        "lane_status": lane_status,
        "model_name": best_name,
        "model_version": artifact_name,
        "trained_at": datetime.now(timezone.utc),
        "field_roles": feature_field_roles("hr"),
        "selected_feature_roles": [
            FIELD_ROLE_CORE_PREDICTOR,
            FIELD_ROLE_ENVIRONMENT_CONTEXT,
        ],
        "feature_columns": feature_columns,
        "training_columns": training_columns,
        "category_columns": category_columns,
        "calibration_method": best_calibration_method,
        "metrics": metrics,
        "baselines": baselines,
        "calibrator": best_calibrator,
        "model": best_model,
    }
    artifact_path = save_artifact("hr", artifact_name, artifact)
    report_path = save_report("hr", artifact_name, artifact | {"model": None, "calibrator": None})
    log.info("Saved HR artifact %s and report %s", artifact_path, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
