from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.settings import get_settings


def load_feature_snapshots(lane: str) -> pd.DataFrame:
    settings = get_settings()
    lane_dir = settings.feature_dir / lane
    files = sorted(lane_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(file_path) for file_path in files]
    return pd.concat(frames, ignore_index=True)


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
    return output_path


def load_latest_artifact(lane: str) -> dict[str, Any]:
    settings = get_settings()
    files = sorted((settings.model_dir / lane).glob("*.pkl"), key=lambda path: path.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No model artifacts found for {lane}")
    with files[-1].open("rb") as handle:
        return pickle.load(handle)


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