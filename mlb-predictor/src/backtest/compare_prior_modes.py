from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.common import encode_frame


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = PROJECT_DIR / ".venv" / "Scripts" / "python.exe"
EXPERIMENT_ROOT = PROJECT_DIR / "data" / "experiments" / "prior_modes"


PRESETS: dict[str, dict[str, str]] = {
    "standard_tuned": {
        "PRIOR_BLEND_MODE": "standard",
        "PRIOR_WEIGHT_MULTIPLIER": "0.8",
        "TEAM_FULL_WEIGHT_GAMES": "20",
        "PITCHER_FULL_WEIGHT_STARTS": "6",
        "MIN_PA_FULL_WEIGHT": "80",
    },
    "reduced": {
        "PRIOR_BLEND_MODE": "reduced",
        "PRIOR_WEIGHT_MULTIPLIER": "1.0",
        "TEAM_FULL_WEIGHT_GAMES": "30",
        "PITCHER_FULL_WEIGHT_STARTS": "10",
        "MIN_PA_FULL_WEIGHT": "120",
    },
    "current_only": {
        "PRIOR_BLEND_MODE": "current_only",
        "PRIOR_WEIGHT_MULTIPLIER": "1.0",
        "TEAM_FULL_WEIGHT_GAMES": "30",
        "PITCHER_FULL_WEIGHT_STARTS": "10",
        "MIN_PA_FULL_WEIGHT": "120",
    },
}


def _python_executable() -> str:
    return str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable))


def _latest_file(directory: Path, suffix: str) -> Path | None:
    files = sorted(directory.glob(f"*{suffix}"), key=lambda path: path.stat().st_mtime)
    return files[-1] if files else None


def _load_report(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_features(feature_root: Path, lane: str, target_date: date) -> pd.DataFrame:
    lane_dir = feature_root / lane
    files = sorted(lane_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(file_path) for file_path in files]
    frame = pd.concat(frames, ignore_index=True)
    if "game_date" not in frame.columns:
        return pd.DataFrame()
    return frame[pd.to_datetime(frame["game_date"]).dt.date == target_date].copy()


def _score_totals(mode_dir: Path, target_date: date) -> pd.DataFrame:
    artifact_path = _latest_file(mode_dir / "models" / "totals", ".pkl")
    if artifact_path is None:
        return pd.DataFrame()
    artifact = _load_pickle(artifact_path)
    frame = _load_features(mode_dir / "features", "totals", target_date)
    if frame.empty:
        return pd.DataFrame()
    X = encode_frame(frame[artifact["feature_columns"]], artifact["category_columns"], artifact["training_columns"])
    predictions = artifact["model"].predict(X)
    residual_std = max(float(artifact.get("residual_std", 1.0)), 1.0)
    rows = []
    for row, prediction in zip(frame.itertuples(index=False), predictions):
        market_total = getattr(row, "market_total", None)
        over_probability = None
        if market_total is not None and not pd.isna(market_total):
            over_probability = 1.0 / (1.0 + math.exp(-(float(prediction) - float(market_total)) / residual_std))
        rows.append(
            {
                "game_id": int(row.game_id),
                "away_team": row.away_team,
                "home_team": row.home_team,
                "market_total": market_total,
                "predicted_total_runs": float(prediction),
                "over_probability": over_probability,
            }
        )
    return pd.DataFrame(rows)


def _score_hits(mode_dir: Path, target_date: date) -> pd.DataFrame:
    artifact_path = _latest_file(mode_dir / "models" / "hits", ".pkl")
    if artifact_path is None:
        return pd.DataFrame()
    artifact = _load_pickle(artifact_path)
    frame = _load_features(mode_dir / "features", "hits", target_date)
    if frame.empty:
        return pd.DataFrame()
    X = encode_frame(frame[artifact["feature_columns"]], artifact["category_columns"], artifact["training_columns"])
    probabilities = artifact["model"].predict_proba(X)[:, 1]
    rows = []
    for row, probability in zip(frame.itertuples(index=False), probabilities):
        rows.append(
            {
                "game_id": int(row.game_id),
                "player_id": int(row.player_id),
                "player_name": row.player_name,
                "team": row.team,
                "opponent": row.opponent,
                "lineup_slot": row.lineup_slot,
                "is_confirmed_lineup": row.is_confirmed_lineup,
                "projected_plate_appearances": row.projected_plate_appearances,
                "streak_len_capped": row.streak_len_capped,
                "predicted_hit_probability": float(probability),
            }
        )
    return pd.DataFrame(rows)


def _merge_totals(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for label, frame in results.items():
        if frame.empty:
            continue
        renamed = frame.rename(
            columns={
                "predicted_total_runs": f"predicted_total_runs_{label}",
                "over_probability": f"over_probability_{label}",
            }
        )
        keep_columns = [
            "game_id",
            "away_team",
            "home_team",
            "market_total",
            f"predicted_total_runs_{label}",
            f"over_probability_{label}",
        ]
        merged = renamed[keep_columns] if merged is None else merged.merge(renamed[keep_columns], on=["game_id", "away_team", "home_team", "market_total"], how="outer")
    return merged if merged is not None else pd.DataFrame()


def _merge_hits(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for label, frame in results.items():
        if frame.empty:
            continue
        renamed = frame.rename(columns={"predicted_hit_probability": f"predicted_hit_probability_{label}"})
        keep_columns = [
            "game_id",
            "player_id",
            "player_name",
            "team",
            "opponent",
            "lineup_slot",
            "is_confirmed_lineup",
            "projected_plate_appearances",
            "streak_len_capped",
            f"predicted_hit_probability_{label}",
        ]
        merged = renamed[keep_columns] if merged is None else merged.merge(
            renamed[keep_columns],
            on=[
                "game_id",
                "player_id",
                "player_name",
                "team",
                "opponent",
                "lineup_slot",
                "is_confirmed_lineup",
                "projected_plate_appearances",
                "streak_len_capped",
            ],
            how="outer",
        )
    if merged is None:
        return pd.DataFrame()
    sort_column = next((column for column in merged.columns if column.startswith("predicted_hit_probability_")), None)
    if sort_column:
        merged = merged.sort_values(sort_column, ascending=False)
    return merged


def _mode_env(label: str, overrides: dict[str, str]) -> tuple[dict[str, str], Path]:
    mode_dir = EXPERIMENT_ROOT / label
    env = os.environ.copy()
    env.update(overrides)
    env["FEATURE_DIR"] = str(mode_dir / "features")
    env["MODEL_DIR"] = str(mode_dir / "models")
    env["REPORT_DIR"] = str(mode_dir / "reports")
    env["MODEL_VERSION_PREFIX"] = label
    return env, mode_dir


def _run_module(module_name: str, env: dict[str, str], *args: str) -> None:
    command = [_python_executable(), "-m", module_name, *args]
    completed = subprocess.run(command, cwd=PROJECT_DIR, env=env, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )


def _train_and_score_mode(label: str, overrides: dict[str, str], train_start: date, train_end: date, target_date: date) -> dict[str, Any]:
    env, mode_dir = _mode_env(label, overrides)
    for lane in ("features", "models", "reports"):
        (mode_dir / lane).mkdir(parents=True, exist_ok=True)

    _run_module("src.features.totals_builder", env, "--start-date", train_start.isoformat(), "--end-date", train_end.isoformat())
    _run_module("src.features.hits_builder", env, "--start-date", train_start.isoformat(), "--end-date", train_end.isoformat())
    _run_module("src.features.totals_builder", env, "--target-date", target_date.isoformat())
    _run_module("src.features.hits_builder", env, "--target-date", target_date.isoformat())
    _run_module("src.models.train_totals", env)
    _run_module("src.models.train_hits", env)

    totals_report = _load_report(_latest_file(mode_dir / "reports" / "totals", ".json"))
    hits_report = _load_report(_latest_file(mode_dir / "reports" / "hits", ".json"))
    totals_scores = _score_totals(mode_dir, target_date)
    hits_scores = _score_hits(mode_dir, target_date)
    return {
        "label": label,
        "overrides": overrides,
        "mode_dir": mode_dir,
        "totals_report": totals_report,
        "hits_report": hits_report,
        "totals_scores": totals_scores,
        "hits_scores": hits_scores,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run standard tuned, reduced, and current-only prior comparisons")
    parser.add_argument("--train-start-date", required=True, help="Historical training start date YYYY-MM-DD")
    parser.add_argument("--train-end-date", required=True, help="Historical training end date YYYY-MM-DD")
    parser.add_argument("--target-date", required=True, help="Target scoring date YYYY-MM-DD")
    args = parser.parse_args()

    train_start = date.fromisoformat(args.train_start_date)
    train_end = date.fromisoformat(args.train_end_date)
    target_date = date.fromisoformat(args.target_date)

    output_root = EXPERIMENT_ROOT / "comparison_outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    results = []
    for label, overrides in PRESETS.items():
        results.append(_train_and_score_mode(label, overrides, train_start, train_end, target_date))

    summary_rows = []
    totals_frames: dict[str, pd.DataFrame] = {}
    hits_frames: dict[str, pd.DataFrame] = {}
    for result in results:
        totals_report = result["totals_report"]
        hits_report = result["hits_report"]
        totals_model_name = totals_report.get("model_name")
        hits_model_name = hits_report.get("model_name")
        totals_metric = (totals_report.get("metrics") or {}).get(totals_model_name or "", {})
        hits_metric = (hits_report.get("metrics") or {}).get(hits_model_name or "", {})
        summary_rows.append(
            {
                "label": result["label"],
                **result["overrides"],
                "totals_model_name": totals_model_name,
                "totals_mae": totals_metric.get("mae"),
                "totals_rmse": totals_metric.get("rmse"),
                "hits_model_name": hits_model_name,
                "hits_log_loss": hits_metric.get("log_loss"),
                "hits_brier": hits_metric.get("brier"),
                "totals_scored_rows": int(len(result["totals_scores"])),
                "hits_scored_rows": int(len(result["hits_scores"])),
            }
        )
        totals_frames[result["label"]] = result["totals_scores"]
        hits_frames[result["label"]] = result["hits_scores"]

    summary = pd.DataFrame(summary_rows)
    totals_compare = _merge_totals(totals_frames)
    hits_compare = _merge_hits(hits_frames)

    summary_path = output_root / f"prior_mode_summary_{target_date.isoformat()}.csv"
    totals_path = output_root / f"prior_mode_totals_compare_{target_date.isoformat()}.csv"
    hits_path = output_root / f"prior_mode_hits_compare_{target_date.isoformat()}.csv"
    summary.to_csv(summary_path, index=False)
    totals_compare.to_csv(totals_path, index=False)
    hits_compare.to_csv(hits_path, index=False)

    manifest = {
        "train_start_date": train_start.isoformat(),
        "train_end_date": train_end.isoformat(),
        "target_date": target_date.isoformat(),
        "summary_csv": str(summary_path),
        "totals_compare_csv": str(totals_path),
        "hits_compare_csv": str(hits_path),
        "modes": [
            {
                "label": result["label"],
                "mode_dir": str(result["mode_dir"]),
                "overrides": result["overrides"],
            }
            for result in results
        ],
    }
    manifest_path = output_root / f"prior_mode_manifest_{target_date.isoformat()}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"\nWrote summary: {summary_path}")
    print(f"Wrote totals comparison: {totals_path}")
    print(f"Wrote hits comparison: {hits_path}")
    print(f"Wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())