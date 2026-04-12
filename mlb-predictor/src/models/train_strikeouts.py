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
from src.models.common import add_strikeout_derived_features, calibrate_with_market, chronological_split, compute_sample_weights, encode_frame, fit_market_calibrator, load_feature_snapshots, save_artifact, save_report
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)

STRIKEOUT_MARKET_CALIBRATION_MIN_ROWS = 40
STRIKEOUT_ISOTONIC_BLEND_WEIGHTS = (1.0, 0.8, 0.65, 0.5, 0.35, 0.2)


def _dominant_market_side_fraction(predictions: np.ndarray, market_lines: pd.Series) -> float:
    paired_mask = market_lines.notna().values
    if not paired_mask.any():
        return 0.0
    line_values = market_lines[paired_mask].astype(float).to_numpy()
    sides = np.where(predictions[paired_mask] >= line_values, "over", "under")
    side_counts = pd.Series(sides).value_counts()
    if side_counts.empty:
        return 0.0
    return float(side_counts.iloc[0] / len(sides))


def _market_calibration_rejection_reason(
    base_predictions: np.ndarray,
    calibrated_predictions: np.ndarray,
    market_lines: pd.Series,
    actuals: pd.Series,
) -> str | None:
    paired_mask = market_lines.notna().values
    if not paired_mask.any():
        return "no_market_rows"

    base_mae = float(
        mean_absolute_error(actuals[paired_mask], base_predictions[paired_mask])
    )
    calibrated_mae = float(
        mean_absolute_error(actuals[paired_mask], calibrated_predictions[paired_mask])
    )
    if calibrated_mae >= base_mae:
        return f"non_improving_mae_{calibrated_mae:.3f}_vs_{base_mae:.3f}"

    base_values = base_predictions[paired_mask]
    calibrated_values = calibrated_predictions[paired_mask]
    base_std = float(np.std(base_values)) if len(base_values) else 0.0
    calibrated_std = float(np.std(calibrated_values)) if len(calibrated_values) else 0.0
    base_dominant_fraction = _dominant_market_side_fraction(base_predictions, market_lines)
    calibrated_dominant_fraction = _dominant_market_side_fraction(
        calibrated_predictions,
        market_lines,
    )
    rounded_counts = pd.Series(np.round(calibrated_values, 2)).value_counts()
    largest_bucket = int(rounded_counts.iloc[0]) if not rounded_counts.empty else 0
    unique_bucket_count = int(len(rounded_counts))

    if (
        calibrated_dominant_fraction >= 0.9
        and calibrated_dominant_fraction > base_dominant_fraction
        and largest_bucket >= max(3, len(calibrated_values) // 5)
        and unique_bucket_count <= max(6, len(calibrated_values) // 2)
    ):
        return (
            "one_sided_distribution_"
            f"{calibrated_dominant_fraction:.0%}_bucket_{largest_bucket}"
        )

    if (
        calibrated_dominant_fraction >= 0.95
        and calibrated_std < max(0.55, base_std * 0.55)
    ):
        return (
            "collapsed_variance_"
            f"{calibrated_std:.2f}_vs_{base_std:.2f}"
        )

    return None


def _prediction_bucket_stats(predictions: np.ndarray) -> tuple[float, int, int]:
    prediction_std = float(np.std(predictions)) if len(predictions) else 0.0
    rounded_counts = pd.Series(np.round(predictions, 2)).value_counts()
    largest_bucket = int(rounded_counts.iloc[0]) if not rounded_counts.empty else 0
    unique_bucket_count = int(len(rounded_counts))
    return prediction_std, largest_bucket, unique_bucket_count


def _isotonic_rejection_reason(
    base_predictions: np.ndarray,
    adjusted_predictions: np.ndarray,
    market_lines: pd.Series | None,
) -> str | None:
    if len(adjusted_predictions) < 12:
        return None

    base_std, _, _ = _prediction_bucket_stats(base_predictions)
    adjusted_std, largest_bucket, unique_bucket_count = _prediction_bucket_stats(
        adjusted_predictions
    )
    if adjusted_std < max(0.80, base_std * 0.65):
        return f"collapsed_variance_{adjusted_std:.2f}_vs_{base_std:.2f}"

    if (
        largest_bucket >= max(8, len(adjusted_predictions) // 10)
        and unique_bucket_count <= max(10, len(adjusted_predictions) // 5)
    ):
        return f"bucketed_distribution_{largest_bucket}_unique_{unique_bucket_count}"

    if market_lines is not None and market_lines.notna().any():
        base_dominant_fraction = _dominant_market_side_fraction(base_predictions, market_lines)
        adjusted_dominant_fraction = _dominant_market_side_fraction(
            adjusted_predictions,
            market_lines,
        )
        if (
            adjusted_dominant_fraction >= 0.82
            and adjusted_dominant_fraction > base_dominant_fraction + 0.08
        ):
            return (
                "one_sided_market_split_"
                f"{adjusted_dominant_fraction:.0%}_vs_{base_dominant_fraction:.0%}"
            )

    return None


def _select_isotonic_blend_weight(
    base_predictions: np.ndarray,
    isotonic_predictions: np.ndarray,
    actuals: pd.Series,
    market_lines: pd.Series | None,
) -> tuple[float, np.ndarray, dict[str, object]]:
    raw_mae = float(mean_absolute_error(actuals, base_predictions))
    diagnostics: list[dict[str, object]] = []
    best_choice: tuple[float, np.ndarray, float] | None = None

    for weight in STRIKEOUT_ISOTONIC_BLEND_WEIGHTS:
        candidate_predictions = base_predictions + weight * (
            isotonic_predictions - base_predictions
        )
        candidate_mae = float(mean_absolute_error(actuals, candidate_predictions))
        rejection_reason = _isotonic_rejection_reason(
            base_predictions,
            candidate_predictions,
            market_lines,
        )
        diagnostics.append(
            {
                "weight": weight,
                "mae": candidate_mae,
                "rejection_reason": rejection_reason,
            }
        )
        if rejection_reason is not None or candidate_mae >= raw_mae:
            continue
        if best_choice is None or candidate_mae < best_choice[2]:
            best_choice = (weight, candidate_predictions, candidate_mae)

    if best_choice is None:
        return 0.0, base_predictions.copy(), {"raw_mae": raw_mae, "candidates": diagnostics}

    return best_choice[0], best_choice[1], {
        "raw_mae": raw_mae,
        "selected_mae": best_choice[2],
        "candidates": diagnostics,
    }


def main() -> int:
    settings = get_settings()
    frame = add_strikeout_derived_features(load_feature_snapshots("strikeouts"))
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

    market_lines = None
    if "market_line" in val_frame.columns:
        market_lines = pd.to_numeric(val_frame["market_line"], errors="coerce")

    # --- Isotonic residual correction (monotonic mapping: raw_pred → actual) ---
    isotonic_calibrator = None
    isotonic_blend_weight = 0.0
    isotonic_metrics: dict[str, object] = {}
    if len(y_val) >= 30:
        sort_idx = np.argsort(best_predictions)
        isotonic_calibrator = IsotonicRegression(out_of_bounds="clip")
        isotonic_calibrator.fit(best_predictions[sort_idx], y_val.values[sort_idx])
        iso_preds = isotonic_calibrator.predict(best_predictions)
        isotonic_blend_weight, selected_isotonic_predictions, isotonic_metrics = _select_isotonic_blend_weight(
            best_predictions,
            iso_preds,
            y_val,
            market_lines,
        )
        if isotonic_blend_weight <= 0:
            raw_mae = float(isotonic_metrics.get("raw_mae", mean_absolute_error(y_val, best_predictions)))
            iso_mae = float(mean_absolute_error(y_val, iso_preds))
            log.info(
                "Isotonic correction discarded after guardrails (raw MAE %.3f, full isotonic MAE %.3f)",
                raw_mae,
                iso_mae,
            )
            isotonic_calibrator = None
        else:
            raw_mae = float(isotonic_metrics.get("raw_mae", mean_absolute_error(y_val, best_predictions)))
            selected_mae = float(isotonic_metrics.get("selected_mae", mean_absolute_error(y_val, selected_isotonic_predictions)))
            log.info(
                "Isotonic correction accepted with blend %.2f: MAE %.3f -> %.3f (improvement: +%.4f)",
                isotonic_blend_weight,
                raw_mae,
                selected_mae,
                raw_mae - selected_mae,
            )

    fundamentals_predictions = best_predictions.copy()
    if isotonic_calibrator is not None:
        full_isotonic_predictions = isotonic_calibrator.predict(fundamentals_predictions)
        fundamentals_predictions = best_predictions + isotonic_blend_weight * (
            full_isotonic_predictions - best_predictions
        )

    # --- Market calibration (blend model prediction with market line) ---
    market_calibrator = None
    market_calibration_metrics = {}
    calibrated_predictions = fundamentals_predictions.copy()
    if market_lines is not None:
        candidate_calibrator = fit_market_calibrator(
            fundamentals_predictions,
            market_lines,
            y_val,
            min_rows=STRIKEOUT_MARKET_CALIBRATION_MIN_ROWS,
        )
        if candidate_calibrator is not None:
            candidate_predictions, calibrated_mask = calibrate_with_market(
                fundamentals_predictions,
                market_lines,
                candidate_calibrator,
            )
            rejection_reason = _market_calibration_rejection_reason(
                fundamentals_predictions,
                candidate_predictions,
                market_lines,
                y_val,
            )
            if rejection_reason is not None:
                log.warning(
                    "Market calibrator rejected: %s",
                    rejection_reason,
                )
            elif calibrated_mask.any():
                market_calibrator = candidate_calibrator
                calibrated_predictions = candidate_predictions
                base_market_mae = float(
                    mean_absolute_error(
                        y_val[calibrated_mask],
                        fundamentals_predictions[calibrated_mask],
                    )
                )
                calibrated_market_mae = float(
                    mean_absolute_error(
                        y_val[calibrated_mask],
                        candidate_predictions[calibrated_mask],
                    )
                )
                market_calibration_metrics = {
                    "market_subset_mae": base_market_mae,
                    "market_subset_mae_calibrated": calibrated_market_mae,
                }
                log.info(
                    "Market calibrator accepted: model_weight=%.3f market_weight=%.3f intercept=%.3f (fitted on %d rows, MAE %.3f -> %.3f)",
                    market_calibrator["model_weight"],
                    market_calibrator["market_weight"],
                    market_calibrator["intercept"],
                    market_calibrator["calibration_rows"],
                    base_market_mae,
                    calibrated_market_mae,
                )
        else:
            log.info(
                "Market calibrator: not enough market lines to fit (need %d)",
                STRIKEOUT_MARKET_CALIBRATION_MIN_ROWS,
            )

    # --- Compute residual_std from best available calibration ---
    residual_std_raw = float((y_val - best_predictions).std()) if len(y_val) else 1.0
    residual_std_calibrated = float((y_val - calibrated_predictions).std()) if len(y_val) else 1.0

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
        "market_calibration_metrics": market_calibration_metrics,
        "isotonic_calibrator": isotonic_calibrator,
        "isotonic_blend_weight": isotonic_blend_weight,
        "isotonic_metrics": isotonic_metrics,
        "model": best_model,
    }
    artifact_path = save_artifact("strikeouts", artifact_name, artifact)
    report_path = save_report("strikeouts", artifact_name, artifact | {"model": None})
    log.info("Saved strikeout artifact %s and report %s", artifact_path, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())