from __future__ import annotations

import math
from datetime import datetime, timezone

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features.contracts import (
    FIELD_ROLE_CORE_PREDICTOR,
    TOTALS_META_COLUMNS,
    TOTALS_TARGET_COLUMN,
    feature_columns_for_roles,
    feature_field_roles,
)
from src.models.common import chronological_split, encode_frame, load_feature_snapshots, save_artifact, save_report
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)


def main() -> int:
    settings = get_settings()
    frame = load_feature_snapshots("totals")
    if frame.empty:
        log.info("No totals feature snapshots found")
        return 0

    trainable = frame[frame[TOTALS_TARGET_COLUMN].notna()].copy()
    if trainable.empty:
        log.info("No labeled totals feature rows found")
        return 0

    train_frame, val_frame = chronological_split(trainable)
    if val_frame.empty:
        log.info("Not enough totals rows for validation split")
        return 0

    feature_columns = feature_columns_for_roles(
        "totals",
        [FIELD_ROLE_CORE_PREDICTOR],
        available_columns=list(trainable.columns),
    )
    category_columns = []
    X_train = encode_frame(train_frame[feature_columns], category_columns)
    X_val = encode_frame(val_frame[feature_columns], category_columns, training_columns=list(X_train.columns))
    y_train = train_frame[TOTALS_TARGET_COLUMN].astype(float)
    y_val = val_frame[TOTALS_TARGET_COLUMN].astype(float)

    candidates = {
        "ridge": Ridge(alpha=1.0),
        "gbr": GradientBoostingRegressor(random_state=42, learning_rate=0.05, n_estimators=300, max_depth=3),
    }
    best_name = None
    best_model = None
    best_rmse = math.inf
    best_predictions = None
    metrics = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        mae = mean_absolute_error(y_val, predictions)
        rmse = math.sqrt(mean_squared_error(y_val, predictions))
        metrics[name] = {"mae": mae, "rmse": rmse}
        if rmse < best_rmse:
            best_name = name
            best_model = model
            best_rmse = rmse
            best_predictions = predictions

    artifact_name = f"totals_{settings.model_version_prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    residual_std = float((y_val - best_predictions).std()) if len(y_val) else 1.0
    artifact = {
        "lane": "totals",
        "model_name": best_name,
        "model_version": artifact_name,
        "trained_at": datetime.now(timezone.utc),
        "field_roles": feature_field_roles("totals"),
        "selected_feature_roles": [FIELD_ROLE_CORE_PREDICTOR],
        "feature_columns": feature_columns,
        "training_columns": list(X_train.columns),
        "category_columns": category_columns,
        "metrics": metrics,
        "residual_std": residual_std if residual_std > 0 else 1.0,
        "model": best_model,
    }
    artifact_path = save_artifact("totals", artifact_name, artifact)
    report_path = save_report("totals", artifact_name, artifact | {"model": None})
    log.info("Saved totals artifact %s and report %s", artifact_path, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())