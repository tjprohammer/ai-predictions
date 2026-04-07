from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features.contracts import (
    FIELD_ROLE_CORE_PREDICTOR,
    STRIKEOUTS_TARGET_COLUMN,
    feature_columns_for_roles,
    feature_field_roles,
)
from src.models.common import chronological_split, compute_sample_weights, encode_frame, fit_market_calibrator, load_feature_snapshots, save_artifact, save_report
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)


def main() -> int:
    settings = get_settings()
    frame = load_feature_snapshots("strikeouts")
    if frame.empty:
        log.info("No strikeout feature snapshots found")
        return 0

    trainable = frame[frame[STRIKEOUTS_TARGET_COLUMN].notna()].copy()
    if trainable.empty:
        log.info("No labeled strikeout rows found")
        return 0

    train_frame, val_frame = chronological_split(trainable)
    if val_frame.empty:
        log.info("Not enough strikeout rows for validation split")
        return 0

    feature_columns = feature_columns_for_roles(
        "strikeouts",
        [FIELD_ROLE_CORE_PREDICTOR],
        available_columns=list(trainable.columns),
    )
    category_columns = [column for column in ["throws"] if column in feature_columns]
    X_train = encode_frame(train_frame[feature_columns], category_columns)
    X_val = encode_frame(val_frame[feature_columns], category_columns, training_columns=list(X_train.columns))
    y_train = train_frame[STRIKEOUTS_TARGET_COLUMN].astype(float)
    y_val = val_frame[STRIKEOUTS_TARGET_COLUMN].astype(float)

    candidates = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "gbr": GradientBoostingRegressor(random_state=42, learning_rate=0.05, n_estimators=300, max_depth=3),
        "gbr_deep": GradientBoostingRegressor(random_state=42, learning_rate=0.03, n_estimators=500, max_depth=4),
        "hgb": HistGradientBoostingRegressor(random_state=42, max_depth=4, learning_rate=0.05, max_iter=500),
    }
    best_name = None
    best_model = None
    best_rmse = math.inf
    best_predictions = None
    metrics = {}

    # Time-decay sample weights: recent games count more
    train_weights = compute_sample_weights(train_frame["game_date"])

    for name, model in candidates.items():
        # Pipeline models need the weight prefixed with the step name
        if hasattr(model, "steps"):
            last_step_name = model.steps[-1][0]
            model.fit(X_train, y_train, **{f"{last_step_name}__sample_weight": train_weights})
        else:
            model.fit(X_train, y_train, sample_weight=train_weights)
        predictions = model.predict(X_val)
        mae = mean_absolute_error(y_val, predictions)
        rmse = math.sqrt(mean_squared_error(y_val, predictions))
        metrics[name] = {"mae": mae, "rmse": rmse}
        if rmse < best_rmse:
            best_name = name
            best_model = model
            best_rmse = rmse
            best_predictions = predictions

    # --- Baseline benchmarks ---
    baselines = {}
    train_mean = float(y_train.mean())
    baselines["train_mean"] = {
        "mae": float(mean_absolute_error(y_val, [train_mean] * len(y_val))),
        "rmse": float(math.sqrt(mean_squared_error(y_val, [train_mean] * len(y_val)))),
    }
    train_median = float(y_train.median())
    baselines["train_median"] = {
        "mae": float(mean_absolute_error(y_val, [train_median] * len(y_val))),
        "rmse": float(math.sqrt(mean_squared_error(y_val, [train_median] * len(y_val)))),
    }
    log.info("Baselines: %s", baselines)

    # --- Feature importance (GBR / HistGBR expose feature_importances_) ---
    feature_importance = {}
    importance_model = best_model
    if hasattr(importance_model, "steps"):
        importance_model = importance_model[-1]
    if hasattr(importance_model, "feature_importances_"):
        _fi = importance_model.feature_importances_
        _cols = list(X_train.columns)
        _ranked = sorted(zip(_cols, _fi), key=lambda t: t[1], reverse=True)
        for col, imp in _ranked:
            feature_importance[col] = round(float(imp), 4)
        log.info("Feature importance (top 10): %s", dict(_ranked[:10]))

    # --- Market calibration (blend model prediction with market line) ---
    market_calibrator = None
    if "market_line" in val_frame.columns:
        market_lines = pd.to_numeric(val_frame["market_line"], errors="coerce")
        market_calibrator = fit_market_calibrator(
            best_predictions, market_lines, y_val, min_rows=15,
        )
        if market_calibrator is not None:
            log.info(
                "Market calibrator: model_weight=%.3f market_weight=%.3f intercept=%.3f (fitted on %d rows)",
                market_calibrator["model_weight"],
                market_calibrator["market_weight"],
                market_calibrator["intercept"],
                market_calibrator["calibration_rows"],
            )
        else:
            log.info("Market calibrator: not enough market lines to fit")

    # --- Isotonic residual correction (monotonic mapping: raw_pred → actual) ---
    isotonic_calibrator = None
    if len(y_val) >= 30:
        sort_idx = np.argsort(best_predictions)
        isotonic_calibrator = IsotonicRegression(out_of_bounds="clip")
        isotonic_calibrator.fit(best_predictions[sort_idx], y_val.values[sort_idx])
        iso_preds = isotonic_calibrator.predict(best_predictions)
        iso_mae = mean_absolute_error(y_val, iso_preds)
        raw_mae = mean_absolute_error(y_val, best_predictions)
        if iso_mae >= raw_mae:
            log.info("Isotonic correction does not improve MAE (%.3f >= %.3f); discarding", iso_mae, raw_mae)
            isotonic_calibrator = None
        else:
            log.info("Isotonic correction: MAE %.3f -> %.3f (improvement: +%.4f)", raw_mae, iso_mae, raw_mae - iso_mae)

    # --- Compute residual_std from best available calibration ---
    calibrated_preds = best_predictions.copy()
    if isotonic_calibrator is not None:
        calibrated_preds = isotonic_calibrator.predict(calibrated_preds)
    residual_std_raw = float((y_val - best_predictions).std()) if len(y_val) else 1.0
    residual_std_calibrated = float((y_val - calibrated_preds).std()) if len(y_val) else 1.0

    artifact_name = f"strikeouts_{settings.model_version_prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    artifact = {
        "lane": "strikeouts",
        "model_name": best_name,
        "model_version": artifact_name,
        "trained_at": datetime.now(timezone.utc),
        "field_roles": feature_field_roles("strikeouts"),
        "selected_feature_roles": [FIELD_ROLE_CORE_PREDICTOR],
        "feature_columns": feature_columns,
        "training_columns": list(X_train.columns),
        "category_columns": category_columns,
        "metrics": metrics,
        "baselines": baselines,
        "feature_importance": feature_importance,
        "residual_std": residual_std_raw if residual_std_raw > 0 else 1.0,
        "residual_std_calibrated": residual_std_calibrated if residual_std_calibrated > 0 else 1.0,
        "market_calibrator": market_calibrator,
        "isotonic_calibrator": isotonic_calibrator,
        "model": best_model,
    }
    artifact_path = save_artifact("strikeouts", artifact_name, artifact)
    report_path = save_report("strikeouts", artifact_name, artifact | {"model": None})
    log.info("Saved strikeout artifact %s and report %s", artifact_path, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())