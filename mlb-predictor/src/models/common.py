from __future__ import annotations

import json
import pickle
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from src.utils.settings import get_settings


class ArtifactRuntimeMismatchError(FileNotFoundError):
    pass


def _artifact_metadata_path(artifact_path: Path) -> Path:
    return artifact_path.with_suffix(".meta.json")


def _current_sklearn_version() -> str | None:
    try:
        import sklearn
    except ImportError:
        return None
    return str(sklearn.__version__)


def _build_artifact_metadata(lane: str, artifact_name: str, artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "lane": lane,
        "artifact_name": artifact_name,
        "model_name": artifact.get("model_name"),
        "model_version": artifact.get("model_version") or artifact_name,
        "trained_at": artifact.get("trained_at"),
        "python_version": sys.version.split()[0],
        "pandas_version": pd.__version__,
        "sklearn_version": _current_sklearn_version(),
    }


def _read_artifact_metadata(artifact_path: Path) -> dict[str, Any] | None:
    metadata_path = _artifact_metadata_path(artifact_path)
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_feature_snapshots(lane: str) -> pd.DataFrame:
    settings = get_settings()
    lane_dir = settings.feature_dir / lane
    files = sorted(lane_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    column_order: list[str] = []
    frames = []
    for file_path in files:
        frame = pd.read_parquet(file_path)
        if frame.empty:
            continue
        for column in frame.columns:
            if column not in column_order:
                column_order.append(column)
        frames.append(frame.dropna(axis=1, how="all"))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.reindex(columns=column_order)

    sort_columns = [column for column in ["game_date", "feature_cutoff_ts", "prediction_ts"] if column in combined.columns]
    if sort_columns:
        combined = combined.sort_values(sort_columns).reset_index(drop=True)

    dedupe_columns: list[str] = []
    if "game_id" in combined.columns:
        dedupe_columns.append("game_id")
    if "player_id" in combined.columns:
        dedupe_columns.append("player_id")
    elif "pitcher_id" in combined.columns:
        dedupe_columns.append("pitcher_id")
    if dedupe_columns:
        combined = combined.drop_duplicates(subset=dedupe_columns, keep="last").reset_index(drop=True)

    return combined


def encode_frame(frame: pd.DataFrame, category_columns: list[str], training_columns: list[str] | None = None) -> pd.DataFrame:
    encoded = pd.get_dummies(frame, columns=category_columns, dummy_na=False)
    for column in encoded.columns:
        series = encoded[column]
        if pd.api.types.is_bool_dtype(series):
            encoded[column] = series.astype("int8")
        elif not pd.api.types.is_numeric_dtype(series):
            encoded[column] = pd.to_numeric(series, errors="coerce")
    encoded = encoded.fillna(0)
    if training_columns is not None:
        encoded = encoded.reindex(columns=training_columns, fill_value=0)
    return encoded


def save_artifact(lane: str, artifact_name: str, artifact: dict[str, Any]) -> Path:
    settings = get_settings()
    output_dir = settings.model_dir / lane
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{artifact_name}.pkl"
    with output_path.open("wb") as handle:
        pickle.dump(artifact, handle)
    metadata = _build_artifact_metadata(lane, artifact_name, artifact)
    _artifact_metadata_path(output_path).write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return output_path


def load_latest_artifact(lane: str) -> dict[str, Any]:
    settings = get_settings()
    files = sorted((settings.model_dir / lane).glob("*.pkl"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No model artifacts found for {lane}")
    runtime_sklearn_version = _current_sklearn_version()
    mismatch_reasons: list[str] = []
    for artifact_path in files:
        metadata = _read_artifact_metadata(artifact_path)
        if runtime_sklearn_version is not None:
            if metadata is None:
                mismatch_reasons.append(f"{artifact_path.name} has no artifact metadata")
                continue
            artifact_sklearn_version = metadata.get("sklearn_version")
            if artifact_sklearn_version != runtime_sklearn_version:
                mismatch_reasons.append(
                    f"{artifact_path.name} was trained with sklearn {artifact_sklearn_version or 'unknown'}"
                )
                continue
        with artifact_path.open("rb") as handle:
            return pickle.load(handle)
    if mismatch_reasons:
        raise ArtifactRuntimeMismatchError(
            f"No compatible {lane} model artifact is available for sklearn {runtime_sklearn_version}: "
            + "; ".join(mismatch_reasons)
        )
    raise FileNotFoundError(f"No model artifacts found for {lane}")


def save_report(lane: str, report_name: str, payload: dict[str, Any]) -> Path:
    settings = get_settings()
    output_dir = settings.report_dir / lane
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{report_name}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return output_path


def chronological_split(frame: pd.DataFrame, validation_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame, frame
    sorted_frame = frame.sort_values([column for column in ["game_date", "feature_cutoff_ts", "prediction_ts"] if column in frame.columns]).reset_index(drop=True)
    split_index = max(1, int(len(sorted_frame) * (1 - validation_fraction)))
    return sorted_frame.iloc[:split_index].copy(), sorted_frame.iloc[split_index:].copy()


# ---------------------------------------------------------------------------
# Market calibration layer
# ---------------------------------------------------------------------------

import math

import numpy as np
from sklearn.linear_model import Ridge as _CalibrationRidge


def fit_market_calibrator(
    raw_predictions: np.ndarray,
    market_values: pd.Series,
    actuals: pd.Series,
    *,
    min_rows: int = 20,
) -> dict[str, Any] | None:
    """Fit a Ridge blend of raw model prediction and market_total → actual.

    Returns a dict suitable for storing in the model artifact under key
    ``"market_calibrator"``, or ``None`` if there are too few rows with
    market data to fit reliably.
    """
    mask = market_values.notna()
    n_usable = int(mask.sum())
    if n_usable < min_rows:
        return None

    X_cal = np.column_stack(
        [raw_predictions[mask.values], market_values[mask].astype(float).values]
    )
    y_cal = actuals[mask].astype(float).values

    calibrator = _CalibrationRidge(alpha=1.0)
    calibrator.fit(X_cal, y_cal)

    calibrated = calibrator.predict(X_cal)
    residual_std = float(np.std(y_cal - calibrated))

    return {
        "calibrator": calibrator,
        "calibration_columns": ["raw_prediction", "market_total"],
        "calibration_rows": n_usable,
        "calibration_residual_std": residual_std if residual_std > 0 else 1.0,
        "model_weight": float(calibrator.coef_[0]),
        "market_weight": float(calibrator.coef_[1]),
        "intercept": float(calibrator.intercept_),
    }


def calibrate_with_market(
    raw_predictions: np.ndarray,
    market_values: pd.Series,
    calibrator_info: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply trained market calibrator to raw predictions.

    Returns ``(calibrated_predictions, used_calibration_mask)``.
    Rows without ``market_total`` get the raw prediction unchanged.
    If *calibrator_info* is ``None`` all rows are returned raw.
    """
    calibrated = raw_predictions.copy()
    mask = np.zeros(len(raw_predictions), dtype=bool)

    if calibrator_info is None:
        return calibrated, mask

    has_market = market_values.notna().values
    if not has_market.any():
        return calibrated, mask

    X_cal = np.column_stack(
        [raw_predictions[has_market], market_values[has_market].astype(float).values]
    )
    calibrated[has_market] = calibrator_info["calibrator"].predict(X_cal)
    mask[has_market] = True

    return calibrated, mask