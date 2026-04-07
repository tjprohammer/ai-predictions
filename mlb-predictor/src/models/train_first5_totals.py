from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features.contracts import (
    FIELD_ROLE_CORE_PREDICTOR,
    FIRST5_TOTALS_META_COLUMNS,
    FIRST5_TOTALS_TARGET_COLUMN,
    feature_columns_for_roles,
    feature_field_roles,
)
from src.models.common import chronological_split, compute_sample_weights, encode_frame, fit_market_calibrator, load_feature_snapshots, save_artifact, save_report
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enhanced calibration: model + market + certainty signals
# ---------------------------------------------------------------------------

_CALIBRATION_EXTRA_COLUMNS = [
    "starter_quality_gap",
    "starter_certainty_score",
    "board_state_numeric",
]


def _board_state_numeric(frame):
    """Encode board_state as numeric: complete=2, partial=1, minimal=0."""
    mapping = {"complete": 2, "partial": 1, "minimal": 0}
    return frame["board_state"].map(mapping).fillna(0).astype(float)


def _fit_enhanced_calibrator(
    raw_predictions: np.ndarray,
    val_frame,
    y_val,
    *,
    min_rows: int = 20,
):
    """Fit a Ridge calibrator using model prediction + market + certainty signals."""
    market_ok = val_frame["market_total"].notna()
    gap_ok = val_frame["starter_quality_gap"].notna() if "starter_quality_gap" in val_frame.columns else market_ok
    mask = market_ok & gap_ok

    n_usable = int(mask.sum())
    if n_usable < min_rows:
        return None

    board_numeric = _board_state_numeric(val_frame)
    X_cal = np.column_stack([
        raw_predictions[mask.values],
        val_frame.loc[mask.index[mask], "market_total"].astype(float).values,
        val_frame.loc[mask.index[mask], "starter_quality_gap"].fillna(0).astype(float).values if "starter_quality_gap" in val_frame.columns else np.zeros(n_usable),
        val_frame.loc[mask.index[mask], "starter_certainty_score"].fillna(0).astype(float).values,
        board_numeric[mask].values,
    ])
    y_cal = y_val[mask].astype(float).values

    calibrator = Ridge(alpha=1.0)
    calibrator.fit(X_cal, y_cal)

    calibrated = calibrator.predict(X_cal)
    residual_std = float(np.std(y_cal - calibrated))

    col_names = ["raw_prediction", "market_total"] + _CALIBRATION_EXTRA_COLUMNS
    return {
        "calibrator": calibrator,
        "calibration_columns": col_names,
        "calibration_rows": n_usable,
        "calibration_residual_std": residual_std if residual_std > 0 else 1.0,
        "model_weight": float(calibrator.coef_[0]),
        "market_weight": float(calibrator.coef_[1]),
        "extra_weights": {name: float(w) for name, w in zip(col_names[2:], calibrator.coef_[2:])},
        "intercept": float(calibrator.intercept_),
    }


def main() -> int:
    settings = get_settings()
    frame = load_feature_snapshots("first5_totals")
    if frame.empty:
        log.info("No first-five totals feature snapshots found")
        return 0

    trainable = frame[frame[FIRST5_TOTALS_TARGET_COLUMN].notna()].copy()
    if trainable.empty:
        log.info("No labeled first-five totals rows found")
        return 0

    train_frame, val_frame = chronological_split(trainable)
    if val_frame.empty:
        log.info("Not enough first-five totals rows for validation split")
        return 0

    feature_columns = feature_columns_for_roles(
        "first5_totals",
        [FIELD_ROLE_CORE_PREDICTOR],
        available_columns=list(trainable.columns),
    )
    category_columns = []
    X_train = encode_frame(train_frame[feature_columns], category_columns)
    X_val = encode_frame(val_frame[feature_columns], category_columns, training_columns=list(X_train.columns))
    y_train = train_frame[FIRST5_TOTALS_TARGET_COLUMN].astype(float)
    y_val = val_frame[FIRST5_TOTALS_TARGET_COLUMN].astype(float)

    candidates = {
        "ridge": Ridge(alpha=1.0),
        "lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.1, max_iter=5000)),
        "elasticnet": make_pipeline(StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
        "gbr": GradientBoostingRegressor(random_state=42, learning_rate=0.05, n_estimators=300, max_depth=3),
    }
    best_name = None
    best_model = None
    best_rmse = math.inf
    best_predictions = None
    metrics = {}

    # Time-decay sample weights: recent games count more
    train_weights = compute_sample_weights(train_frame["game_date"])

    for name, model in candidates.items():
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
    if "home_team" in train_frame.columns and "away_team" in train_frame.columns:
        _home_avgs = train_frame.groupby("home_team")[FIRST5_TOTALS_TARGET_COLUMN].mean()
        _away_avgs = train_frame.groupby("away_team")[FIRST5_TOTALS_TARGET_COLUMN].mean()
        _matchup_pred = (
            val_frame["home_team"].map(_home_avgs).fillna(train_mean)
            + val_frame["away_team"].map(_away_avgs).fillna(train_mean)
        ) / 2
        baselines["team_average"] = {
            "mae": float(mean_absolute_error(y_val, _matchup_pred)),
            "rmse": float(math.sqrt(mean_squared_error(y_val, _matchup_pred))),
        }
    if "market_total" in val_frame.columns:
        market_mask = val_frame["market_total"].notna()
        if market_mask.sum() > 0:
            market_vals = val_frame.loc[market_mask.index[market_mask], "market_total"].astype(float)
            baselines["market_total"] = {
                "mae": float(mean_absolute_error(y_val[market_mask.values], market_vals)),
                "rmse": float(math.sqrt(mean_squared_error(y_val[market_mask.values], market_vals))),
                "rows": int(market_mask.sum()),
            }
    log.info("Baselines: %s", baselines)

    # --- Honest lane status ---
    best_baseline_mae = min(b["mae"] for b in baselines.values() if "mae" in b)
    best_baseline_name = min(baselines, key=lambda k: baselines[k].get("mae", float("inf")))
    model_mae = metrics[best_name]["mae"]
    if model_mae > best_baseline_mae:
        log.warning(
            "LANE STATUS: first5_totals model (MAE=%.3f) does NOT beat %s baseline (MAE=%.3f). "
            "This lane is not extracting signal above simple baselines. "
            "Treat as derivative/experimental until totals redesign is validated.",
            model_mae, best_baseline_name, best_baseline_mae,
        )
    lane_status = "below_baseline" if model_mae > best_baseline_mae else "above_baseline"

    artifact_name = f"first5_totals_{settings.model_version_prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    residual_std = float((y_val - best_predictions).std()) if len(y_val) else 1.0

    # --- Market calibration layer (basic) ---
    market_calibrator = fit_market_calibrator(
        best_predictions,
        val_frame["market_total"],
        y_val,
    )

    # --- Enhanced calibration layer (model + market + certainty) ---
    enhanced_calibrator = _fit_enhanced_calibrator(
        best_predictions, val_frame, y_val,
    )

    calibration_metrics = None
    if enhanced_calibrator is not None:
        from sklearn.metrics import mean_absolute_error as _mae, mean_squared_error as _mse

        cal_mask = val_frame["market_total"].notna()
        gap_ok = val_frame["starter_quality_gap"].notna() if "starter_quality_gap" in val_frame.columns else cal_mask
        cal_mask = cal_mask & gap_ok
        board_numeric = _board_state_numeric(val_frame)
        X_cal = np.column_stack([
            best_predictions[cal_mask.values],
            val_frame.loc[cal_mask.index[cal_mask], "market_total"].astype(float).values,
            val_frame.loc[cal_mask.index[cal_mask], "starter_quality_gap"].fillna(0).astype(float).values if "starter_quality_gap" in val_frame.columns else np.zeros(int(cal_mask.sum())),
            val_frame.loc[cal_mask.index[cal_mask], "starter_certainty_score"].fillna(0).astype(float).values,
            board_numeric[cal_mask].values,
        ])
        cal_preds = enhanced_calibrator["calibrator"].predict(X_cal)
        calibration_metrics = {
            "calibrated_mae": float(_mae(y_val[cal_mask.values], cal_preds)),
            "calibrated_rmse": float(_mse(y_val[cal_mask.values], cal_preds) ** 0.5),
            "raw_mae_same_rows": float(_mae(y_val[cal_mask.values], best_predictions[cal_mask.values])),
            "raw_rmse_same_rows": float(_mse(y_val[cal_mask.values], best_predictions[cal_mask.values]) ** 0.5),
            "calibration_rows": enhanced_calibrator["calibration_rows"],
            "model_weight": enhanced_calibrator["model_weight"],
            "market_weight": enhanced_calibrator["market_weight"],
            "extra_weights": enhanced_calibrator["extra_weights"],
            "intercept": enhanced_calibrator["intercept"],
        }
    elif market_calibrator is not None:
        from sklearn.metrics import mean_absolute_error as _mae, mean_squared_error as _mse

        cal_mask = val_frame["market_total"].notna()
        cal_preds = market_calibrator["calibrator"].predict(
            np.column_stack(
                [best_predictions[cal_mask.values], val_frame.loc[cal_mask.index[cal_mask], "market_total"].astype(float).values]
            )
        )
        calibration_metrics = {
            "calibrated_mae": float(_mae(y_val[cal_mask.values], cal_preds)),
            "calibrated_rmse": float(_mse(y_val[cal_mask.values], cal_preds) ** 0.5),
            "raw_mae_same_rows": float(_mae(y_val[cal_mask.values], best_predictions[cal_mask.values])),
            "raw_rmse_same_rows": float(_mse(y_val[cal_mask.values], best_predictions[cal_mask.values]) ** 0.5),
            "calibration_rows": market_calibrator["calibration_rows"],
            "model_weight": market_calibrator["model_weight"],
            "market_weight": market_calibrator["market_weight"],
            "intercept": market_calibrator["intercept"],
        }

    artifact = {
        "lane": "first5_totals",
        "lane_status": lane_status,
        "model_name": best_name,
        "model_version": artifact_name,
        "trained_at": datetime.now(timezone.utc),
        "field_roles": feature_field_roles("first5_totals"),
        "selected_feature_roles": [FIELD_ROLE_CORE_PREDICTOR],
        "feature_columns": feature_columns,
        "training_columns": list(X_train.columns),
        "category_columns": category_columns,
        "metrics": metrics,
        "baselines": baselines,
        "calibration_metrics": calibration_metrics,
        "residual_std": residual_std if residual_std > 0 else 1.0,
        "market_calibrator": market_calibrator,
        "enhanced_calibrator": enhanced_calibrator,
        "model": best_model,
    }
    artifact_path = save_artifact("first5_totals", artifact_name, artifact)
    report_payload = {k: v for k, v in artifact.items() if k not in ("model", "market_calibrator", "enhanced_calibrator")}
    if market_calibrator is not None:
        report_payload["market_calibrator"] = {k: v for k, v in market_calibrator.items() if k != "calibrator"}
    if enhanced_calibrator is not None:
        report_payload["enhanced_calibrator"] = {k: v for k, v in enhanced_calibrator.items() if k != "calibrator"}
    report_path = save_report("first5_totals", artifact_name, report_payload)
    log.info("Saved first-five totals artifact %s and report %s", artifact_path, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())