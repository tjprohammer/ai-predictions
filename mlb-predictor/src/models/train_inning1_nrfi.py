"""Train a classifier for P(no run in the 1st inning) — NRFI vs YRFI."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features.contracts import (
    FIELD_ROLE_CORE_PREDICTOR,
    INNING1_NRFI_TARGET_COLUMN,
    feature_columns_for_roles,
    feature_field_roles,
)
from src.models.common import chronological_split, compute_sample_weights, encode_frame, load_feature_snapshots, save_artifact, save_report
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)

_DEFAULT_MIN_LABELED_ROWS = 120
_LOGIT_MAX_ITER = 3000
_CLIP_EPS = 1e-5


def _clip_probabilities(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), _CLIP_EPS, 1.0 - _CLIP_EPS)


def _score_probabilities(y_true, probabilities: np.ndarray) -> dict[str, float]:
    clipped = _clip_probabilities(probabilities)
    return {
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, clipped)),
    }


def _fit_sigmoid_calibrator(probabilities: np.ndarray, y_true) -> LogisticRegression | None:
    unique_targets = set(int(value) for value in y_true)
    if len(unique_targets) < 2:
        return None
    calibrator = LogisticRegression(max_iter=_LOGIT_MAX_ITER, solver="lbfgs")
    calibrator.fit(probabilities.reshape(-1, 1), y_true)
    return calibrator


def _isotonic_eligible(
    isotonic_probabilities: np.ndarray,
    raw_evaluation_probabilities: np.ndarray,
) -> bool:
    iso = np.asarray(isotonic_probabilities, dtype=float)
    raw = np.asarray(raw_evaluation_probabilities, dtype=float)
    if iso.size == 0:
        return False
    if float(np.ptp(iso)) < 1e-12:
        return False
    if float(np.max(iso)) <= 1e-8 and float(np.max(raw)) > 1e-4:
        return False
    return True


def _fit_best_calibrator(
    calibration_probabilities: np.ndarray,
    y_calibration,
    evaluation_probabilities: np.ndarray,
    y_evaluation,
):
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


def _better_candidate(
    ll: float,
    br: float,
    auc: float,
    best_ll: float,
    best_brier: float,
    best_auc: float,
) -> bool:
    if ll < best_ll - 1e-12:
        return True
    if abs(ll - best_ll) > 1e-12:
        return False
    if br < best_brier - 1e-12:
        return True
    if abs(br - best_brier) > 1e-12:
        return False
    return auc > best_auc


def main() -> int:
    parser = argparse.ArgumentParser(description="Train inning-1 NRFI classifier from feature snapshots")
    parser.add_argument(
        "--train-end-date",
        metavar="YYYY-MM-DD",
        default=None,
        help="Use only rows with game_date on or before this date",
    )
    parser.add_argument(
        "--train-start-date",
        metavar="YYYY-MM-DD",
        default=None,
        help="Use only rows with game_date on or after this date",
    )
    parser.add_argument(
        "--min-labeled-rows",
        type=int,
        default=_DEFAULT_MIN_LABELED_ROWS,
        help="Minimum labeled rows to train (default: %s)" % _DEFAULT_MIN_LABELED_ROWS,
    )
    args = parser.parse_args()
    min_labeled = max(1, int(args.min_labeled_rows))

    settings = get_settings()
    frame = load_feature_snapshots("inning1_nrfi")
    if frame.empty:
        log.info("No inning1_nrfi feature snapshots found — run src.features.inning1_nrfi_builder")
        return 0

    trainable = frame[frame[INNING1_NRFI_TARGET_COLUMN].notna()].copy()
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
    if len(trainable) < min_labeled:
        log.info(
            "Not enough labeled inning-1 rows (%s); need at least %s (boxscore ingest populates total_runs_inning1).",
            len(trainable),
            min_labeled,
        )
        return 0

    train_frame, val_frame = chronological_split(trainable)
    if val_frame.empty or len(val_frame) < 20:
        log.info("Validation split too small for inning1_nrfi")
        return 0

    val_cal_frame, val_eval_frame = chronological_split(val_frame, validation_fraction=0.5)
    use_calibrator = len(val_eval_frame) >= 15 and len(val_cal_frame) >= 15

    feature_columns = feature_columns_for_roles(
        "inning1_nrfi",
        [FIELD_ROLE_CORE_PREDICTOR],
        available_columns=list(trainable.columns),
    )
    category_columns: list[str] = []
    X_train = encode_frame(train_frame[feature_columns], category_columns)
    training_columns = list(X_train.columns)
    X_val = encode_frame(val_frame[feature_columns], category_columns, training_columns=training_columns)
    if use_calibrator:
        X_val_cal = encode_frame(val_cal_frame[feature_columns], category_columns, training_columns=training_columns)
        X_val_eval = encode_frame(val_eval_frame[feature_columns], category_columns, training_columns=training_columns)
    y_train = train_frame[INNING1_NRFI_TARGET_COLUMN].astype(int)
    y_val = val_frame[INNING1_NRFI_TARGET_COLUMN].astype(int)
    y_val_cal = val_cal_frame[INNING1_NRFI_TARGET_COLUMN].astype(int)
    y_val_eval = val_eval_frame[INNING1_NRFI_TARGET_COLUMN].astype(int)

    train_weights = compute_sample_weights(train_frame["game_date"])

    candidates: dict[str, object] = {
        "logit": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=_LOGIT_MAX_ITER, class_weight="balanced", random_state=42, solver="lbfgs"),
        ),
        "gbc": GradientBoostingClassifier(
            n_estimators=240,
            max_depth=3,
            learning_rate=0.06,
            subsample=0.85,
            random_state=42,
        ),
        "hgb": HistGradientBoostingClassifier(
            max_depth=4,
            max_iter=220,
            learning_rate=0.06,
            random_state=42,
        ),
    }

    best_name = None
    best_model = None
    best_ll = float("inf")
    best_brier = float("inf")
    best_auc = -1.0
    metrics: dict[str, dict[str, float]] = {}

    for name, model in candidates.items():
        if hasattr(model, "steps"):
            last_step_name = model.steps[-1][0]
            model.fit(X_train, y_train, **{f"{last_step_name}__sample_weight": train_weights})
        else:
            model.fit(X_train, y_train, sample_weight=train_weights)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_val)[:, 1]
        else:
            proba = model.decision_function(X_val)
            proba = 1.0 / (1.0 + np.exp(-proba))
        try:
            auc = float(roc_auc_score(y_val, proba))
        except ValueError:
            auc = 0.5
        ll = float(log_loss(y_val, _clip_probabilities(proba)))
        br = float(brier_score_loss(y_val, proba))
        metrics[name] = {"roc_auc": auc, "log_loss": ll, "brier": br}
        if best_model is None or _better_candidate(ll, br, auc, best_ll, best_brier, best_auc):
            best_ll = ll
            best_brier = br
            best_auc = auc
            best_name = name
            best_model = model

    assert best_model is not None and best_name is not None

    calibration_method = "identity"
    calibrator = None
    calibration_split_metrics: dict[str, object] = {}

    if use_calibrator:
        cal_raw = best_model.predict_proba(X_val_cal)[:, 1]
        eval_raw = best_model.predict_proba(X_val_eval)[:, 1]
        calibration_method, calibrator, calibration_split_metrics = _fit_best_calibrator(
            cal_raw,
            y_val_cal,
            eval_raw,
            y_val_eval,
        )

    def _apply_cal_raw(raw: np.ndarray) -> np.ndarray:
        if calibrator is None or calibration_method == "identity":
            return raw
        raw = np.asarray(raw, dtype=float)
        if calibration_method == "sigmoid":
            return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
        if calibration_method == "isotonic":
            return calibrator.predict(raw)
        return raw

    val_proba_raw = best_model.predict_proba(X_val)[:, 1]
    val_proba = _clip_probabilities(_apply_cal_raw(val_proba_raw))
    base_rate = float(y_train.mean())
    baseline_ll = float(log_loss(y_val, np.full(len(y_val), base_rate)))
    model_ll = float(log_loss(y_val, val_proba))
    try:
        val_auc = float(roc_auc_score(y_val, val_proba_raw))
    except ValueError:
        val_auc = 0.5
    val_ll_raw = float(log_loss(y_val, _clip_probabilities(val_proba_raw)))

    artifact_name = f"inning1_nrfi_{settings.model_version_prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    artifact = {
        "lane": "inning1_nrfi",
        "model_name": best_name,
        "model_version": artifact_name,
        "trained_at": datetime.now(timezone.utc),
        "field_roles": feature_field_roles("inning1_nrfi"),
        "selected_feature_roles": [FIELD_ROLE_CORE_PREDICTOR],
        "feature_columns": feature_columns,
        "training_columns": training_columns,
        "category_columns": category_columns,
        "metrics": metrics,
        "selection_criterion": "val_log_loss_tiebreak_brier_auc",
        "validation_roc_auc": val_auc,
        "validation_log_loss_raw": val_ll_raw,
        "validation_log_loss": model_ll,
        "calibration_method": calibration_method,
        "calibrator": calibrator,
        "calibration_split_metrics": calibration_split_metrics,
        "baseline_log_loss": baseline_ll,
        "model_log_loss": model_ll,
        "labeled_train_rows": len(train_frame),
        "labeled_val_rows": len(val_frame),
        "train_nrfi_rate": base_rate,
        "model": best_model,
    }
    artifact_path = save_artifact("inning1_nrfi", artifact_name, artifact)
    report_payload = {k: v for k, v in artifact.items() if k not in ("model", "calibrator")}
    report_path = save_report("inning1_nrfi", artifact_name, report_payload)
    log.info(
        "Saved inning1_nrfi artifact %s (best=%s val_log_loss=%.4f raw=%.4f auc=%.3f cal=%s vs baseline %.4f) report %s",
        artifact_path,
        best_name,
        model_ll,
        val_ll_raw,
        val_auc,
        calibration_method,
        baseline_ll,
        report_path,
    )
    if model_ll >= baseline_ll:
        log.warning(
            "Inning-1 NRFI model log_loss (%.4f) does not beat constant base-rate (%.4f); treat as experimental.",
            model_ll,
            baseline_ll,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
