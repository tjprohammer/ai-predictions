from __future__ import annotations

from datetime import datetime, timezone

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from src.features.contracts import HITS_META_COLUMNS, HITS_TARGET_COLUMN
from src.models.common import chronological_split, encode_frame, load_feature_snapshots, save_artifact, save_report
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)


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

    feature_columns = [column for column in trainable.columns if column not in set(HITS_META_COLUMNS + [HITS_TARGET_COLUMN])]
    category_columns = [column for column in ["team", "opponent", "home_away", "player_name"] if column in feature_columns]
    X_train = encode_frame(train_frame[feature_columns], category_columns)
    X_val = encode_frame(val_frame[feature_columns], category_columns, training_columns=list(X_train.columns))
    y_train = train_frame[HITS_TARGET_COLUMN].astype(int)
    y_val = val_frame[HITS_TARGET_COLUMN].astype(int)

    candidates = {
        "logistic": LogisticRegression(max_iter=1000),
        "hgb": HistGradientBoostingClassifier(random_state=42, max_depth=4),
    }
    metrics = {}
    best_name = None
    best_model = None
    best_log_loss = float("inf")
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_val)[:, 1]
        current_log_loss = log_loss(y_val, probabilities)
        metrics[name] = {
            "log_loss": current_log_loss,
            "brier": brier_score_loss(y_val, probabilities),
        }
        if current_log_loss < best_log_loss:
            best_log_loss = current_log_loss
            best_name = name
            best_model = model

    artifact_name = f"hits_{settings.model_version_prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    artifact = {
        "lane": "hits",
        "model_name": best_name,
        "model_version": artifact_name,
        "trained_at": datetime.now(timezone.utc),
        "feature_columns": feature_columns,
        "training_columns": list(X_train.columns),
        "category_columns": category_columns,
        "metrics": metrics,
        "model": best_model,
    }
    artifact_path = save_artifact("hits", artifact_name, artifact)
    report_path = save_report("hits", artifact_name, artifact | {"model": None})
    log.info("Saved hits artifact %s and report %s", artifact_path, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())