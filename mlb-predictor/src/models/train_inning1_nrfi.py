"""Train a classifier for P(no run in the 1st inning) — NRFI vs YRFI."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
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

_MIN_LABELED_ROWS = 120


def main() -> int:
    settings = get_settings()
    frame = load_feature_snapshots("inning1_nrfi")
    if frame.empty:
        log.info("No inning1_nrfi feature snapshots found — run src.features.inning1_nrfi_builder")
        return 0

    trainable = frame[frame[INNING1_NRFI_TARGET_COLUMN].notna()].copy()
    if len(trainable) < _MIN_LABELED_ROWS:
        log.info(
            "Not enough labeled inning-1 rows (%s); need at least %s (boxscore ingest populates total_runs_inning1).",
            len(trainable),
            _MIN_LABELED_ROWS,
        )
        return 0

    train_frame, val_frame = chronological_split(trainable)
    if val_frame.empty or len(val_frame) < 20:
        log.info("Validation split too small for inning1_nrfi")
        return 0

    feature_columns = feature_columns_for_roles(
        "inning1_nrfi",
        [FIELD_ROLE_CORE_PREDICTOR],
        available_columns=list(trainable.columns),
    )
    category_columns: list[str] = []
    X_train = encode_frame(train_frame[feature_columns], category_columns)
    X_val = encode_frame(val_frame[feature_columns], category_columns, training_columns=list(X_train.columns))
    y_train = train_frame[INNING1_NRFI_TARGET_COLUMN].astype(int)
    y_val = val_frame[INNING1_NRFI_TARGET_COLUMN].astype(int)

    train_weights = compute_sample_weights(train_frame["game_date"])

    candidates: dict[str, object] = {
        "logit": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42, solver="lbfgs"),
        ),
        "gbc": GradientBoostingClassifier(
            n_estimators=240,
            max_depth=3,
            learning_rate=0.06,
            subsample=0.85,
            random_state=42,
        ),
    }

    best_name = None
    best_model = None
    best_auc = -1.0
    metrics: dict[str, dict[str, float]] = {}

    for name, model in candidates.items():
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
        ll = float(log_loss(y_val, np.clip(proba, 1e-6, 1 - 1e-6)))
        br = float(brier_score_loss(y_val, proba))
        metrics[name] = {"roc_auc": auc, "log_loss": ll, "brier": br}
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_model = model

    assert best_model is not None and best_name is not None

    val_proba = best_model.predict_proba(X_val)[:, 1]
    base_rate = float(y_train.mean())
    baseline_ll = float(log_loss(y_val, np.full(len(y_val), base_rate)))
    model_ll = float(log_loss(y_val, np.clip(val_proba, 1e-6, 1 - 1e-6)))

    artifact_name = f"inning1_nrfi_{settings.model_version_prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    artifact = {
        "lane": "inning1_nrfi",
        "model_name": best_name,
        "model_version": artifact_name,
        "trained_at": datetime.now(timezone.utc),
        "field_roles": feature_field_roles("inning1_nrfi"),
        "selected_feature_roles": [FIELD_ROLE_CORE_PREDICTOR],
        "feature_columns": feature_columns,
        "training_columns": list(X_train.columns),
        "category_columns": category_columns,
        "metrics": metrics,
        "validation_roc_auc": best_auc,
        "baseline_log_loss": baseline_ll,
        "model_log_loss": model_ll,
        "labeled_train_rows": len(train_frame),
        "labeled_val_rows": len(val_frame),
        "train_nrfi_rate": base_rate,
        "model": best_model,
    }
    artifact_path = save_artifact("inning1_nrfi", artifact_name, artifact)
    report_payload = {k: v for k, v in artifact.items() if k != "model"}
    report_path = save_report("inning1_nrfi", artifact_name, report_payload)
    log.info(
        "Saved inning1_nrfi artifact %s (best=%s val_auc=%.3f log_loss=%.3f vs baseline %.3f) report %s",
        artifact_path,
        best_name,
        best_auc,
        model_ll,
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
