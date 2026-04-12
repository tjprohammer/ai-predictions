"""Train a regression model to predict batter total bases per game.

Architecture mirrors train_strikeouts.py — GradientBoosting regressor with
time-decay sample weights and an optional market-line calibration layer.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features.contracts import (
    FIELD_ROLE_CORE_PREDICTOR,
    TOTAL_BASES_TARGET_COLUMN,
    feature_columns_for_roles,
    feature_field_roles,
)
from src.models.common import (
    chronological_split,
    compute_sample_weights,
    encode_frame,
    fit_market_calibrator,
    load_feature_snapshots,
    save_artifact,
    save_report,
)
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)


def main() -> int:
    settings = get_settings()
    frame = load_feature_snapshots("total_bases")
    if frame.empty:
        log.info("No total-bases feature snapshots found")
        return 0

    trainable = frame[frame[TOTAL_BASES_TARGET_COLUMN].notna()].copy()
    if len(trainable) < 100:
        log.info("Not enough labeled total-bases rows (%d) — skipping training", len(trainable))
        return 0

    train_frame, val_frame = chronological_split(trainable)
    if val_frame.empty:
        log.info("Not enough rows for a validation split")
        return 0

    feature_columns = feature_columns_for_roles(
        "total_bases",
        [FIELD_ROLE_CORE_PREDICTOR],
        available_columns=list(trainable.columns),
    )
    category_columns = [c for c in ["home_away"] if c in feature_columns]
    X_train = encode_frame(train_frame[feature_columns], category_columns)
    X_val = encode_frame(val_frame[feature_columns], category_columns, training_columns=list(X_train.columns))
    y_train = train_frame[TOTAL_BASES_TARGET_COLUMN].astype(float)
    y_val = val_frame[TOTAL_BASES_TARGET_COLUMN].astype(float)

    candidates = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "gbr": GradientBoostingRegressor(random_state=42, learning_rate=0.05, n_estimators=300, max_depth=3),
        "hgb": HistGradientBoostingRegressor(random_state=42, max_depth=4, learning_rate=0.05, max_iter=500),
    }
    train_weights = compute_sample_weights(train_frame["game_date"])
    best_name, best_model, best_rmse, best_predictions = None, None, math.inf, None
    metrics = {}

    for name, model in candidates.items():
        if hasattr(model, "steps"):
            last_step = model.steps[-1][0]
            model.fit(X_train, y_train, **{f"{last_step}__sample_weight": train_weights})
        else:
            model.fit(X_train, y_train, sample_weight=train_weights)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        rmse = math.sqrt(mean_squared_error(y_val, preds))
        metrics[name] = {"mae": float(mae), "rmse": float(rmse)}
        if rmse < best_rmse:
            best_name, best_model, best_rmse, best_predictions = name, model, rmse, preds

    # Baselines
    train_mean = float(y_train.mean())
    baselines = {
        "train_mean": {"mae": float(mean_absolute_error(y_val, [train_mean] * len(y_val))),
                       "rmse": float(math.sqrt(mean_squared_error(y_val, [train_mean] * len(y_val))))},
    }

    # Optional market calibration (requires market_tb_line in features)
    market_calibrator: dict = {}
    calibration_metrics: dict = {}
    market_col = "market_tb_line"
    if market_col in val_frame.columns:
        try:
            market_values = pd.to_numeric(val_frame[market_col], errors="coerce")
            market_calibrator = fit_market_calibrator(
                best_predictions,
                market_values,
                y_val,
            )
            if market_calibrator:
                cal_preds = market_calibrator["calibrator"].predict(
                    np.column_stack([best_predictions, market_values.fillna(train_mean).values])
                    if market_calibrator.get("calibration_rows", 0) > 0
                    else best_predictions.reshape(-1, 1)
                )
                calibration_metrics = {
                    "calibrated_mae": float(mean_absolute_error(y_val, cal_preds)),
                    "calibrated_rmse": float(math.sqrt(mean_squared_error(y_val, cal_preds))),
                }
        except Exception as exc:  # noqa: BLE001
            log.warning("Market calibration failed for total_bases: %s", exc)

    artifact_name = f"total_bases_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    artifact = {
        "lane": "total_bases",
        "model_name": best_name,
        "model_version": artifact_name,
        "trained_at": datetime.now(timezone.utc),
        "field_roles": feature_field_roles("total_bases"),
        "selected_feature_roles": [FIELD_ROLE_CORE_PREDICTOR],
        "feature_columns": feature_columns,
        "training_columns": list(X_train.columns),
        "category_columns": category_columns,
        "metrics": metrics,
        "baselines": baselines,
        "calibration_metrics": calibration_metrics,
        "market_calibrator": market_calibrator,
        "model": best_model,
        "residual_std": float(np.std(y_val - best_predictions)) if best_predictions is not None else 1.0,
    }
    save_artifact("total_bases", artifact_name, artifact)
    save_report("total_bases", artifact_name, {
        "lane": "total_bases", "best_model": best_name,
        "best_mae": metrics.get(best_name, {}).get("mae"),
        "best_rmse": best_rmse, "baselines": baselines,
        "calibration_metrics": calibration_metrics,
        "train_rows": len(train_frame), "val_rows": len(val_frame),
    })
    log.info("total_bases training complete — best=%s rmse=%.4f mae=%.4f", best_name, best_rmse,
             metrics.get(best_name, {}).get("mae", float("nan")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
