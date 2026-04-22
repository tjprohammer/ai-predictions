from __future__ import annotations

import json
import pickle
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge as _CalibrationRidge
from sklearn.metrics import brier_score_loss, log_loss as sk_log_loss

from src.utils.settings import get_settings


class ArtifactRuntimeMismatchError(FileNotFoundError):
    pass


STRIKEOUT_DERIVED_FEATURE_COLUMNS = [
    "matchup_k_factor",
    "matchup_baseline_strikeouts",
    "matchup_recent_strikeouts_3",
    "matchup_recent_strikeouts_5",
    "matchup_season_k_per_start",
]


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


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(index=frame.index, dtype=float)


def add_strikeout_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        enriched = frame.copy()
        for column in STRIKEOUT_DERIVED_FEATURE_COLUMNS:
            if column not in enriched.columns:
                enriched[column] = pd.Series(dtype=float)
        return enriched

    enriched = frame.copy()
    opponent_lineup_k = _numeric_series(enriched, "opponent_lineup_k_pct_recent")
    opponent_team_k = _numeric_series(enriched, "opponent_k_pct_blended")
    opponent_k_rate = opponent_lineup_k.fillna(opponent_team_k).fillna(0.225)
    venue_k_factor = _numeric_series(enriched, "venue_k_factor").fillna(1.0).clip(0.85, 1.15)
    umpire_k_factor = _numeric_series(enriched, "ump_k_rate_adj").fillna(1.0).clip(0.90, 1.10)
    matchup_k_factor = (opponent_k_rate / 0.225).clip(0.78, 1.32) * venue_k_factor * umpire_k_factor

    baseline_strikeouts = _numeric_series(enriched, "baseline_strikeouts")
    recent_avg_strikeouts_3 = _numeric_series(enriched, "recent_avg_strikeouts_3")
    recent_avg_strikeouts_5 = _numeric_series(enriched, "recent_avg_strikeouts_5")
    season_k_per_start = _numeric_series(enriched, "season_k_per_start")

    enriched["matchup_k_factor"] = matchup_k_factor
    enriched["matchup_baseline_strikeouts"] = baseline_strikeouts * matchup_k_factor
    enriched["matchup_recent_strikeouts_3"] = recent_avg_strikeouts_3 * matchup_k_factor
    enriched["matchup_recent_strikeouts_5"] = recent_avg_strikeouts_5 * matchup_k_factor
    enriched["matchup_season_k_per_start"] = season_k_per_start * matchup_k_factor
    return enriched


_LEAGUE_AVERAGE_DEFAULTS: dict[str, float] = {
    # Team offense (2025 MLB league averages)
    "home_xwoba_blended": 0.310, "away_xwoba_blended": 0.310,
    "home_iso_blended": 0.145, "away_iso_blended": 0.145,
    "home_bb_pct_blended": 0.083, "away_bb_pct_blended": 0.083,
    "home_k_pct_blended": 0.225, "away_k_pct_blended": 0.225,
    "home_runs_rate_blended": 4.5, "away_runs_rate_blended": 4.5,
    "home_hits_rate_blended": 8.3, "away_hits_rate_blended": 8.3,
    # Starters
    "home_starter_xwoba_blended": 0.310, "away_starter_xwoba_blended": 0.310,
    "home_starter_csw_blended": 0.295, "away_starter_csw_blended": 0.295,
    # Lineups
    "home_lineup_top5_xwoba": 0.320, "away_lineup_top5_xwoba": 0.320,
    "home_lineup_k_pct": 0.225, "away_lineup_k_pct": 0.225,
    # Venue
    "venue_run_factor": 1.0, "venue_hr_factor": 1.0,
    # Weather
    "temperature_f": 72.0, "humidity_pct": 55.0,
    "wind_speed_mph": 8.0, "wind_direction_deg": 180.0,
    # Hits lane – player Statcast
    "xba_14": 0.245, "xwoba_14": 0.310, "hard_hit_pct_14": 0.380,
    "season_prior_xba": 0.245, "season_prior_xwoba": 0.310,
    "opposing_starter_xwoba": 0.310, "opposing_starter_csw": 0.295,
    # Strikeouts lane – pitcher Statcast
    "recent_whiff_pct_5": 0.245, "recent_csw_pct_5": 0.295,
    "recent_xwoba_5": 0.310,
}


def encode_frame(frame: pd.DataFrame, category_columns: list[str], training_columns: list[str] | None = None) -> pd.DataFrame:
    encoded = pd.get_dummies(frame, columns=category_columns, dummy_na=False)
    for column in encoded.columns:
        series = encoded[column]
        if pd.api.types.is_bool_dtype(series):
            encoded[column] = series.astype("int8")
        elif not pd.api.types.is_numeric_dtype(series):
            encoded[column] = pd.to_numeric(series, errors="coerce")
    encoded = encoded.fillna(
        {col: val for col, val in _LEAGUE_AVERAGE_DEFAULTS.items() if col in encoded.columns}
    ).fillna(0)
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


def mean_pinball_loss(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series, *, tau: float = 0.5) -> float:
    """Pinball loss for quantile regression; ``tau=0.5`` equals ``0.5 * MAE`` for symmetric errors."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    e = y - p
    t = float(tau)
    return float(np.mean(np.maximum(t * e, (t - 1.0) * e)))


def regression_metrics_by_month(
    game_dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    min_rows: int = 8,
) -> dict[str, dict[str, float]]:
    """Validation MAE/RMSE by calendar month (YYYY-MM)."""
    if game_dates is None or len(game_dates) != len(y_true):
        return {}
    months = pd.to_datetime(game_dates, errors="coerce").dt.strftime("%Y-%m")
    out: dict[str, dict[str, float]] = {}
    for m in sorted(months.dropna().unique()):
        mask = (months == m).values
        if int(mask.sum()) < min_rows:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        out[str(m)] = {
            "n": float(int(mask.sum())),
            "mae": float(np.mean(np.abs(yt - yp))),
            "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
        }
    return out


def regression_val_temporal_halves_mae(y_true: np.ndarray, y_pred: np.ndarray, *, min_total: int = 40) -> dict[str, float] | None:
    """Split validation (already time-ordered) into first/second half MAE — quick drift check."""
    n = len(y_true)
    if n < min_total:
        return None
    h = n // 2
    return {
        "first_half_mae": float(np.mean(np.abs(y_true[:h] - y_pred[:h]))),
        "second_half_mae": float(np.mean(np.abs(y_true[h:] - y_pred[h:]))),
        "n_first": float(h),
        "n_second": float(n - h),
    }


def reliability_bins_binary(
    y_true: np.ndarray | pd.Series,
    probabilities: np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
    min_bin_n: int = 5,
) -> list[dict[str, float]]:
    """Histogram-style reliability: mean predicted prob vs empirical positive rate per bin."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) != len(p) or len(y) < n_bins * min_bin_n:
        return []
    edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] = float(np.min(p)) - 1e-9
    edges[-1] = float(np.max(p)) + 1e-9
    rows: list[dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if int(mask.sum()) < min_bin_n:
            continue
        rows.append(
            {
                "bin": float(i),
                "n": float(int(mask.sum())),
                "mean_p": float(np.mean(p[mask])),
                "positive_rate": float(np.mean(y[mask])),
            }
        )
    return rows


def classification_metrics_by_month(
    game_dates: pd.Series,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    min_rows: int = 20,
) -> dict[str, dict[str, float]]:
    months = pd.to_datetime(game_dates, errors="coerce").dt.strftime("%Y-%m")
    out: dict[str, dict[str, float]] = {}
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    for m in sorted(months.dropna().unique()):
        mask = (months == m).values
        if int(mask.sum()) < min_rows:
            continue
        try:
            out[str(m)] = {
                "n": float(int(mask.sum())),
                "brier": float(brier_score_loss(y[mask], p[mask])),
                "log_loss": float(sk_log_loss(y[mask], p[mask], labels=[0, 1])),
            }
        except Exception:
            continue
    return out


def log_loss_ou_vs_market_line(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    market_total: pd.Series,
    *,
    std: float,
    min_rows: int = 30,
) -> dict[str, float] | None:
    """Binary log loss for P(over) vs posted total, using a logistic on (pred − line) / std."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mkt = pd.to_numeric(market_total, errors="coerce")
    mask = mkt.notna().values
    if int(mask.sum()) < min_rows:
        return None
    m = mkt[mask].astype(float).values
    yt = y[mask]
    yp = p[mask]
    y_over = (yt > m).astype(int)
    scale = max(float(std), 0.5)
    logits = (yp - m) / scale
    prob = np.clip(1.0 / (1.0 + np.exp(-logits)), 1e-6, 1.0 - 1e-6)
    return {
        "n": float(int(mask.sum())),
        "log_loss": float(sk_log_loss(y_over, prob, labels=[0, 1])),
    }


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
# Time-decay sample weights
# ---------------------------------------------------------------------------

def compute_sample_weights(
    game_dates: pd.Series,
    *,
    half_life_days: int = 180,
    min_weight: float = 0.25,
) -> np.ndarray:
    """Compute exponential time-decay weights so recent games count more.

    Parameters
    ----------
    game_dates : Series of date-like values
    half_life_days : days until a sample's weight drops to 50% of the most
        recent sample. Default 180 ≈ one full season of decay.
    min_weight : floor so old samples are never fully discarded.

    Returns
    -------
    1-D numpy array of weights, same length as *game_dates*.
    """
    import numpy as np

    dates = pd.to_datetime(game_dates)
    most_recent = dates.max()
    days_ago = (most_recent - dates).dt.total_seconds() / 86_400
    decay = np.exp(-np.log(2) * days_ago / half_life_days)
    return np.clip(decay, min_weight, 1.0)


# ---------------------------------------------------------------------------
# Market calibration layer
# ---------------------------------------------------------------------------


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

    model_weight = float(calibrator.coef_[0])
    market_weight = float(calibrator.coef_[1])
    if not np.isfinite(model_weight) or not np.isfinite(market_weight):
        return None
    if model_weight <= 0 or market_weight < 0:
        return None

    calibrated = calibrator.predict(X_cal)
    residual_std = float(np.std(y_cal - calibrated))

    return {
        "calibrator": calibrator,
        "calibration_columns": ["raw_prediction", "market_total"],
        "calibration_rows": n_usable,
        "calibration_residual_std": residual_std if residual_std > 0 else 1.0,
        "model_weight": model_weight,
        "market_weight": market_weight,
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
    if float(calibrator_info.get("model_weight", 1.0)) <= 0:
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