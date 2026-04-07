from __future__ import annotations

import contextlib
import importlib
import io
import json
import subprocess
import sys
import threading
import traceback
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from src.features.first5_totals_builder import main as build_first5_totals_features_main
from src.features.hits_builder import main as build_hits_features_main
from src.features.strikeouts_builder import main as build_strikeouts_features_main
from src.features.totals_builder import main as build_totals_features_main
from src.ingestors.boxscores import main as ingest_boxscores_main
from src.ingestors.games import main as ingest_games_main
from src.ingestors.lineups import main as import_lineups_main
from src.ingestors.market_totals import main as import_market_totals_main
from src.ingestors.player_batting import main as ingest_player_batting_main
from src.ingestors.prepare_slate_inputs import main as prepare_slate_inputs_main
from src.ingestors.starters import main as ingest_starters_main
from src.models.predict_first5_totals import main as predict_first5_totals_main
from src.models.predict_hits import main as predict_hits_main
from src.models.predict_strikeouts import main as predict_strikeouts_main
from src.models.predict_totals import main as predict_totals_main
from src.transforms.bullpens_daily import main as refresh_bullpens_daily_main
from src.transforms.freeze_markets import main as freeze_markets_main
from src.transforms.offense_daily import main as refresh_offense_daily_main
from src.transforms.product_surfaces import main as refresh_product_surfaces_main
from src.utils.db import get_dialect_name, query_df, table_exists, upsert_rows
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)
settings = get_settings()
DB_DIALECT = get_dialect_name()
STATIC_DIR = Path(__file__).with_name("static")
INDEX_FILE = STATIC_DIR / "index.html"
HOT_HITTERS_FILE = STATIC_DIR / "hot-hitters.html"
PITCHERS_FILE = STATIC_DIR / "pitchers.html"
RESULTS_FILE = STATIC_DIR / "results.html"
TOTALS_FILE = STATIC_DIR / "totals.html"
GAME_FILE = STATIC_DIR / "game.html"
DOCTOR_FILE = STATIC_DIR / "doctor.html"
FAVICON_FILE = STATIC_DIR / "favicon.svg"
UPDATE_MODULE_MAINS = {
    "src.ingestors.games": ingest_games_main,
    "src.ingestors.starters": ingest_starters_main,
    "src.ingestors.prepare_slate_inputs": prepare_slate_inputs_main,
    "src.ingestors.lineups": import_lineups_main,
    "src.ingestors.market_totals": import_market_totals_main,
    "src.ingestors.boxscores": ingest_boxscores_main,
    "src.ingestors.player_batting": ingest_player_batting_main,
    "src.transforms.offense_daily": refresh_offense_daily_main,
    "src.transforms.bullpens_daily": refresh_bullpens_daily_main,
    "src.transforms.freeze_markets": freeze_markets_main,
    "src.transforms.product_surfaces": refresh_product_surfaces_main,
    "src.features.totals_builder": build_totals_features_main,
    "src.features.first5_totals_builder": build_first5_totals_features_main,
    "src.features.hits_builder": build_hits_features_main,
    "src.features.strikeouts_builder": build_strikeouts_features_main,
    "src.models.predict_totals": predict_totals_main,
    "src.models.predict_first5_totals": predict_first5_totals_main,
    "src.models.predict_hits": predict_hits_main,
    "src.models.predict_strikeouts": predict_strikeouts_main,
}

app = FastAPI(title="MLB Predictor", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


HTML_SHELL_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


def _html_file_response(path: Path) -> FileResponse:
    return FileResponse(path, headers=HTML_SHELL_HEADERS)


class PipelineRunRequest(BaseModel):
    target_date: date = Field(default_factory=date.today)
    refresh_aggregates: bool = True
    rebuild_features: bool = True


UpdateAction = Literal[
    "refresh_everything",
    "prepare_slate",
    "import_manual_inputs",
    "refresh_results",
    "rebuild_predictions",
    "grade_predictions",
]


class UpdateJobRunRequest(BaseModel):
    action: UpdateAction
    target_date: date = Field(default_factory=date.today)


UpdateJobStatus = Literal["queued", "running", "succeeded", "failed"]


UPDATE_JOB_LOCK = threading.Lock()
UPDATE_JOBS: dict[str, dict[str, Any]] = {}
UPDATE_JOB_HISTORY_LIMIT = 12
UPDATE_JOB_STORE_PATH = settings.report_dir / "update_jobs" / "history.json"


def _persist_pipeline_run(job: dict[str, Any]) -> None:
    """Upsert the pipeline_runs row for this job."""
    try:
        if not _table_exists("pipeline_runs"):
            return
        upsert_rows(
            "pipeline_runs",
            [
                {
                    "job_id": job["job_id"],
                    "action": job["action"],
                    "label": job["label"],
                    "target_date": str(job["target_date"]),
                    "status": job["status"],
                    "total_steps": job["total_steps"],
                    "completed_steps": job.get("completed_steps", 0),
                    "error": job.get("error"),
                    "started_at": job.get("started_at"),
                    "finished_at": job.get("finished_at"),
                }
            ],
            ["job_id"],
        )
    except Exception as exc:
        log.warning("Failed to persist pipeline_run for %s: %s", job["job_id"], exc)


def _persist_pipeline_step(job_id: str, step_index: int, step: dict[str, Any]) -> None:
    """Upsert the pipeline_run_steps row for this step."""
    try:
        if not _table_exists("pipeline_run_steps"):
            return
        upsert_rows(
            "pipeline_run_steps",
            [
                {
                    "job_id": job_id,
                    "step_index": step_index,
                    "module_name": step.get("module", ""),
                    "returncode": step.get("returncode"),
                    "stdout": (step.get("stdout") or "")[:4000],
                    "stderr": (step.get("stderr") or "")[:4000],
                    "finished_at": _utc_now_iso(),
                }
            ],
            ["job_id", "step_index"],
        )
    except Exception as exc:
        log.warning("Failed to persist pipeline_run_step for %s step %d: %s", job_id, step_index, exc)
DESKTOP_HISTORY_REQUIREMENTS: tuple[tuple[str, str, str], ...] = (
    ("team_offense_daily", "game_date", "team offense history"),
    ("bullpens_daily", "game_date", "bullpen history"),
    ("player_game_batting", "game_date", "hitter game logs"),
    ("player_trend_daily", "game_date", "hot hitter trend history"),
    ("pitcher_starts", "game_date", "pitcher start history"),
)


def _safe_frame(query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    try:
        return query_df(query, params)
    except Exception as exc:
        log.warning("Query failed: %s", exc)
        return pd.DataFrame()


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    cleaned = frame.copy().astype(object)
    cleaned = cleaned.where(pd.notnull(cleaned), None)
    return cleaned.to_dict(orient="records")


def _table_exists(table_name: str) -> bool:
    try:
        return table_exists(table_name)
    except Exception as exc:
        log.warning("Table lookup failed for %s: %s", table_name, exc)
        return False


def _sql_json_text(column: str, key: str, dialect: str | None = None) -> str:
    active_dialect = (dialect or DB_DIALECT).lower()
    if active_dialect == "sqlite":
        return f"json_extract({column}, '$.{key}')"
    return f"{column} ->> '{key}'"


def _sql_real(expression: str, dialect: str | None = None) -> str:
    active_dialect = (dialect or DB_DIALECT).lower()
    target_type = "REAL" if active_dialect == "sqlite" else "DOUBLE PRECISION"
    return f"CAST({expression} AS {target_type})"


def _sql_integer(expression: str, dialect: str | None = None) -> str:
    active_dialect = (dialect or DB_DIALECT).lower()
    target_type = "INTEGER" if active_dialect == "sqlite" else "SMALLINT"
    return f"CAST({expression} AS {target_type})"


def _sql_boolean(expression: str, dialect: str | None = None) -> str:
    active_dialect = (dialect or DB_DIALECT).lower()
    if active_dialect != "sqlite":
        return f"CAST({expression} AS BOOLEAN)"
    normalized = f"LOWER(TRIM(COALESCE({expression}, '')))"
    return (
        "CASE "
        f"WHEN {normalized} IN ('true', '1', 't', 'yes', 'y') THEN 1 "
        f"WHEN {normalized} IN ('false', '0', 'f', 'no', 'n') THEN 0 "
        "ELSE NULL END"
    )


def _sql_ratio(numerator_expression: str, denominator_expression: str) -> str:
    return (
        f"CASE WHEN SUM({denominator_expression}) = 0 THEN NULL "
        f"ELSE (1.0 * SUM({numerator_expression}) / SUM({denominator_expression})) END"
    )


def _sql_year(expression: str, dialect: str | None = None) -> str:
    active_dialect = (dialect or DB_DIALECT).lower()
    if active_dialect == "sqlite":
        return f"CAST(strftime('%Y', {expression}) AS INTEGER)"
    return f"EXTRACT(YEAR FROM {expression})"


def _sql_year_param(param_name: str, dialect: str | None = None) -> str:
    active_dialect = (dialect or DB_DIALECT).lower()
    if active_dialect == "sqlite":
        return _sql_year(f":{param_name}", active_dialect)
    return _sql_year(f"CAST(:{param_name} AS DATE)", active_dialect)


def _sql_order_nulls_last(expression: str, direction: str = "ASC") -> str:
    normalized_direction = direction.upper()
    return f"CASE WHEN {expression} IS NULL THEN 1 ELSE 0 END, {expression} {normalized_direction}"


def _sql_bind_list(prefix: str, values: list[Any], params: dict[str, Any]) -> str:
    placeholders: list[str] = []
    for index, value in enumerate(values):
        key = f"{prefix}_{index}"
        params[key] = value
        placeholders.append(f":{key}")
    if not placeholders:
        return "NULL"
    return ", ".join(placeholders)


def _artifact_ready(lane: str) -> bool:
    return any((settings.model_dir / lane).glob("*.pkl"))


def _desktop_history_row_counts(target_date: date) -> dict[str, int]:
    row_counts: dict[str, int] = {}
    if DB_DIALECT != "sqlite":
        return row_counts

    for table_name, date_column, _label in DESKTOP_HISTORY_REQUIREMENTS:
        if not _table_exists(table_name):
            row_counts[table_name] = 0
            continue
        frame = _safe_frame(
            f"SELECT COUNT(*) AS row_count FROM {table_name} WHERE {date_column} < :target_date",
            {"target_date": target_date},
        )
        row_counts[table_name] = int(frame.iloc[0]["row_count"]) if not frame.empty else 0
    return row_counts


def _desktop_rebuild_blocker(target_date: date) -> dict[str, Any] | None:
    if DB_DIALECT != "sqlite":
        return None

    row_counts = _desktop_history_row_counts(target_date)
    missing = [
        {"table": table_name, "label": label, "row_count": row_counts.get(table_name, 0)}
        for table_name, _date_column, label in DESKTOP_HISTORY_REQUIREMENTS
        if row_counts.get(table_name, 0) <= 0
    ]
    if not missing:
        return None

    missing_labels = ", ".join(item["label"] for item in missing)
    missing_tables = ", ".join(item["table"] for item in missing)
    return {
        "code": "desktop_history_missing",
        "message": (
            "Desktop historical data is incomplete. Rebuild predictions is blocked because the bundled "
            f"SQLite seed has no prior rows in {missing_labels} ({missing_tables}). "
            "Rebuild the desktop bundle with a populated SQLite seed exported from Postgres before running predictions."
        ),
        "missing": missing,
        "row_counts": row_counts,
    }


def _action_blocker(action: UpdateAction, target_date: date) -> dict[str, Any] | None:
    if action not in {"refresh_everything", "rebuild_predictions"}:
        return None
    return _desktop_rebuild_blocker(target_date)


def _pipeline_blocker(target_date: date) -> dict[str, Any] | None:
    return _desktop_rebuild_blocker(target_date)


def _blocked_update_payload(action: UpdateAction, target_date: date, blocker: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": False,
        "action": action,
        "label": _update_job_label(action),
        "target_date": target_date.isoformat(),
        "steps": [],
        "status": _fetch_status(target_date),
        "message": blocker["message"],
        "blocker": blocker,
    }


def _blocked_pipeline_payload(target_date: date, blocker: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": False,
        "target_date": target_date.isoformat(),
        "steps": [],
        "status": _fetch_status(target_date),
        "message": blocker["message"],
        "blocker": blocker,
    }


def _fetch_lineup_snapshot_keys(target_date: date) -> set[tuple[int, int]]:
    if not _table_exists("lineups"):
        return set()
    frame = _safe_frame(
        """
        SELECT DISTINCT game_id, player_id
        FROM lineups
        WHERE game_date = :target_date
          AND game_id IS NOT NULL
          AND player_id IS NOT NULL
        """,
        {"target_date": target_date},
    )
    if frame.empty:
        return set()
    return {
        (int(row["game_id"]), int(row["player_id"]))
        for row in _frame_records(frame)
        if row.get("game_id") is not None and row.get("player_id") is not None
    }


def _annotate_lineup_confidence(
    records: list[dict[str, Any]],
    lineup_snapshot_keys: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    for record in records:
        game_id = record.get("game_id")
        player_id = record.get("player_id")
        snapshot_key = None
        if game_id is not None and player_id is not None:
            snapshot_key = (int(game_id), int(player_id))
        has_lineup_snapshot = snapshot_key in lineup_snapshot_keys if snapshot_key is not None else False
        is_confirmed_lineup = bool(record.get("is_confirmed_lineup")) if record.get("is_confirmed_lineup") is not None else False
        is_inferred_lineup = not has_lineup_snapshot and not is_confirmed_lineup
        record["has_lineup_snapshot"] = has_lineup_snapshot
        record["is_inferred_lineup"] = is_inferred_lineup
        record["lineup_source"] = "confirmed" if is_confirmed_lineup else ("snapshot" if has_lineup_snapshot else "inferred")
    return records


def _format_lineup_source_name(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return "Snapshot"
    if normalized == "projected_template":
        return "Projected template"
    if normalized == "manual_template":
        return "Manual template"
    return normalized.replace("_", " ").replace("-", " ").title()


def _summarize_team_lineup(records: list[dict[str, Any]]) -> dict[str, Any]:
    confirmed_records = [record for record in records if record.get("is_confirmed_lineup")]
    snapshot_records = [record for record in records if record.get("has_lineup_snapshot")]

    if confirmed_records:
        displayed_records = confirmed_records
        scope = "confirmed"
    elif snapshot_records:
        displayed_records = snapshot_records
        scope = "snapshot"
    elif records:
        displayed_records = list(records)
        scope = "projected"
    else:
        displayed_records = []
        scope = "empty"

    displayed_source_names = {
        str(record.get("lineup_source_name") or "").strip().lower()
        for record in displayed_records
        if str(record.get("lineup_source_name") or "").strip()
    }
    if scope == "confirmed":
        source_summary = "Confirmed lineup"
    elif "projected_template" in displayed_source_names:
        source_summary = "Projected template lineup"
    elif "manual_template" in displayed_source_names:
        source_summary = "Manual template lineup"
    elif scope == "snapshot" and len(displayed_source_names) == 1:
        source_summary = f"{_format_lineup_source_name(next(iter(displayed_source_names)))} lineup snapshot"
    elif scope == "snapshot":
        source_summary = "Latest lineup snapshot"
    elif scope == "projected":
        source_summary = "Projected lineup"
    else:
        source_summary = "Lineup pending"

    return {
        "lineup": displayed_records,
        "lineup_scope": scope,
        "lineup_source_summary": source_summary,
        "lineup_counts": {
            "total_rows": len(records),
            "displayed_rows": len(displayed_records),
            "confirmed_rows": len(confirmed_records),
            "snapshot_rows": len(snapshot_records),
            "inferred_rows": sum(1 for record in records if record.get("is_inferred_lineup")),
        },
    }


def _is_final_game_status(status: Any) -> bool:
    normalized = str(status or "").strip().lower()
    if not normalized:
        return False
    final_markers = ("final", "completed", "game over", "closed")
    return any(marker in normalized for marker in final_markers)


def _build_hit_actual_meta(actual_hits: Any, is_final_game: bool) -> dict[str, str]:
    if actual_hits is not None:
        return {"actual_status": "graded", "actual_status_label": ""}
    if is_final_game:
        return {"actual_status": "dnp", "actual_status_label": "No box score row"}
    return {"actual_status": "pending", "actual_status_label": "Outcome pending"}


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _format_rate(value: float | None, digits: int = 0) -> str:
    if value is None:
        return "pending"
    return f"{value * 100:.{digits}f}%"


def _format_metric(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "pending"
    return f"{value:.{digits}f}"


def _hitter_heat_score(player: dict[str, Any]) -> float:
    return (
        (_to_float(player.get("hit_rate_7")) or 0.0) * 1.2
        + (((_to_float(player.get("hit_rate_7")) or 0.0) - ((_to_float(player.get("hit_rate_30")) or 0.0))) * 1.5)
        + ((_to_float(player.get("xwoba_14")) or 0.0) * 1.1)
        + ((_to_float(player.get("hard_hit_pct_14")) or 0.0) * 0.5)
        + ((int(player.get("streak_len") or player.get("streak_len_capped") or 0)) * 0.04)
    )


def _classify_hitter_form(player: dict[str, Any]) -> dict[str, Any]:
    hit_rate_7 = _to_float(player.get("hit_rate_7"))
    hit_rate_30 = _to_float(player.get("hit_rate_30"))
    xwoba_14 = _to_float(player.get("xwoba_14"))
    hard_hit_pct_14 = _to_float(player.get("hard_hit_pct_14"))
    batting_avg_last7 = _to_float(player.get("batting_avg_last7"))
    hit_games_last7 = int(player.get("hit_games_last7") or 0)
    games_last7 = int(player.get("games_last7") or 0)
    streak = int(player.get("streak_len") or player.get("streak_len_capped") or 0)
    hit_delta = None if hit_rate_7 is None or hit_rate_30 is None else hit_rate_7 - hit_rate_30

    evidence = [
        f"7G {_format_rate(hit_rate_7)} vs 30G {_format_rate(hit_rate_30)}"
        + (f" ({hit_delta * 100:+.0f} pts)" if hit_delta is not None else ""),
        f"xwOBA14 {_format_metric(xwoba_14)}",
        f"HH14 {_format_rate(hard_hit_pct_14)}",
        f"Streak {streak}",
    ]

    hot_reasons: list[str] = []
    warm_reasons: list[str] = []
    cold_reasons: list[str] = []
    if hit_delta is not None and hit_delta >= 0.12:
        hot_reasons.append(f"7G hit rate {_format_rate(hit_rate_7)} is {hit_delta * 100:+.0f} points above the 30G baseline")
    if xwoba_14 is not None and xwoba_14 >= 0.38:
        hot_reasons.append(f"xwOBA over the last 14 games is {_format_metric(xwoba_14)}")
    if streak >= 4:
        hot_reasons.append(f"Riding a {streak}-game hit streak")
    if hard_hit_pct_14 is not None and hard_hit_pct_14 >= 0.45:
        hot_reasons.append(f"Hard-hit rate over the last 14 games is {_format_rate(hard_hit_pct_14)}")

    if streak >= 2:
        warm_reasons.append(f"On a {streak}-game hit streak")
    if hit_rate_7 is not None and hit_rate_7 >= 0.65:
        warm_reasons.append(f"7G hit rate is {_format_rate(hit_rate_7)}")
    if games_last7 >= 5 and hit_games_last7 >= 4:
        warm_reasons.append(f"Has hits in {hit_games_last7} of the last {games_last7} games")
    if batting_avg_last7 is not None and batting_avg_last7 >= 0.320:
        warm_reasons.append(f"Batting {_format_metric(batting_avg_last7)} over the last 7 games")
    if xwoba_14 is not None and xwoba_14 >= 0.345:
        warm_reasons.append(f"xwOBA over the last 14 games is {_format_metric(xwoba_14)}")
    if hard_hit_pct_14 is not None and hard_hit_pct_14 >= 0.40:
        warm_reasons.append(f"Hard-hit rate over the last 14 games is {_format_rate(hard_hit_pct_14)}")

    if hit_delta is not None and hit_delta <= -0.12:
        cold_reasons.append(f"7G hit rate {_format_rate(hit_rate_7)} is {hit_delta * 100:+.0f} points below the 30G baseline")
    if xwoba_14 is not None and xwoba_14 <= 0.285:
        cold_reasons.append(f"xwOBA over the last 14 games is only {_format_metric(xwoba_14)}")

    label = "Steady"
    tone = ""
    reasons = evidence[:1]
    if hot_reasons:
        label = "Hot"
        tone = "good"
        reasons = hot_reasons
    elif warm_reasons:
        label = "Streaking" if streak >= 2 else "Hitting well"
        tone = "good"
        reasons = warm_reasons
    elif cold_reasons:
        label = "Cold"
        tone = "warn"
        reasons = cold_reasons

    summary = reasons[0] if reasons else evidence[0]
    detail = " · ".join((reasons[1:] if len(reasons) > 1 else evidence[1:]))
    return {
        "label": label,
        "tone": tone,
        "summary": summary,
        "detail": detail,
        "reasons": reasons,
        "heat_score": round(_hitter_heat_score(player), 4),
        "evidence": evidence,
    }


def _normalize_bat_side(value: Any) -> str | None:
    normalized = str(value or "").strip().upper()
    if normalized in {"R", "L", "S"}:
        return normalized
    return None


def _summarize_lineup_handedness(players: list[dict[str, Any]], confirmed_key: str) -> dict[str, Any]:
    counts = {"R": 0, "L": 0, "S": 0}
    confirmed_hitters = 0
    total_hitters = 0
    for player in players:
        bat_side = _normalize_bat_side(player.get("bats"))
        if bat_side:
            counts[bat_side] += 1
        if player.get(confirmed_key):
            confirmed_hitters += 1
        total_hitters += 1

    known_hitters = counts["R"] + counts["L"] + counts["S"]
    return {
        "counts": counts,
        "known_hitters": known_hitters,
        "confirmed_hitters": confirmed_hitters,
        "total_hitters": total_hitters,
    }


def _fetch_lineup_handedness_by_game(target_date: date) -> dict[int, dict[str, dict[str, Any]]]:
    if not _table_exists("lineups"):
        return {}

    lineup_slot_order = _sql_order_nulls_last("lineup_slot")
    frame = _safe_frame(
        f"""
        WITH ranked_lineups AS (
            SELECT
                l.game_id,
                l.team,
                l.player_id,
                l.is_confirmed,
                l.lineup_slot,
                dp.bats,
                DENSE_RANK() OVER (
                    PARTITION BY l.game_id, l.team
                    ORDER BY l.snapshot_ts DESC
                ) AS snapshot_rank
            FROM lineups l
            LEFT JOIN dim_players dp ON dp.player_id = l.player_id
            WHERE l.game_date = :target_date
        )
        SELECT
            game_id,
            team,
            player_id,
            is_confirmed,
            lineup_slot,
            bats
        FROM ranked_lineups
        WHERE snapshot_rank = 1
        ORDER BY game_id, team, {lineup_slot_order}, player_id
        """,
        {"target_date": target_date},
    )
    if frame.empty:
        return {}

    context_by_game: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in _frame_records(frame):
        context_by_game[int(row["game_id"])][str(row["team"])].append(row)

    return {
        game_id: {
            team: _summarize_lineup_handedness(players, confirmed_key="is_confirmed")
            for team, players in teams.items()
        }
        for game_id, teams in context_by_game.items()
    }


def _fetch_hitter_pitch_hand_splits(
    target_date: date,
    player_ids: list[int] | set[int],
) -> dict[int, dict[str, dict[str, Any]]]:
    if not player_ids or not _table_exists("player_game_batting") or not _table_exists("pitcher_starts"):
        return {}

    params: dict[str, Any] = {"target_date": target_date}
    player_id_placeholders = _sql_bind_list("player_id", sorted(int(player_id) for player_id in player_ids), params)
    split_batting_avg = _sql_ratio("b.hits", "b.at_bats")
    frame = _safe_frame(
        f"""
        WITH ranked_starters AS (
            SELECT
                ps.game_id,
                ps.team,
                dp.throws,
                ROW_NUMBER() OVER (
                    PARTITION BY ps.game_id, ps.team
                    ORDER BY
                        CASE
                            WHEN ps.ip IS NOT NULL
                              OR ps.pitch_count IS NOT NULL
                              OR ps.strikeouts IS NOT NULL THEN 0
                            ELSE 1
                        END,
                        CASE WHEN COALESCE(ps.is_probable, FALSE) THEN 1 ELSE 0 END,
                        ps.pitcher_id
                ) AS row_rank
            FROM pitcher_starts ps
            LEFT JOIN dim_players dp ON dp.player_id = ps.pitcher_id
            WHERE ps.game_date < :target_date
        )
        SELECT
            b.player_id,
            rs.throws AS pitcher_throws,
            COUNT(*) AS split_games,
            SUM(b.hits) AS split_hits,
            SUM(b.at_bats) AS split_at_bats,
            {split_batting_avg} AS split_batting_avg,
            AVG(b.xwoba) AS split_xwoba,
            AVG(b.hard_hit_pct) AS split_hard_hit_pct
        FROM player_game_batting b
        INNER JOIN ranked_starters rs
            ON rs.game_id = b.game_id
           AND rs.team = b.opponent
           AND rs.row_rank = 1
        WHERE b.game_date < :target_date
                    AND b.player_id IN ({player_id_placeholders})
          AND rs.throws IN ('R', 'L')
        GROUP BY b.player_id, rs.throws
        """,
                params,
    )
    if frame.empty:
        return {}

    split_map: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in _frame_records(frame):
        player_id = int(row["player_id"])
        throw_hand = str(row["pitcher_throws"])
        split_map[player_id][throw_hand] = {
            "vs_pitcher_hand_games": row["split_games"],
            "vs_pitcher_hand_hits": row["split_hits"],
            "vs_pitcher_hand_at_bats": row["split_at_bats"],
            "vs_pitcher_hand_batting_avg": row["split_batting_avg"],
            "vs_pitcher_hand_xwoba": row["split_xwoba"],
            "vs_pitcher_hand_hard_hit_pct": row["split_hard_hit_pct"],
        }
    return split_map


def _fetch_recent_hit_history_map(
    target_date: date,
    player_ids: list[int] | set[int],
    limit: int = 10,
) -> dict[int, list[dict[str, Any]]]:
    if not player_ids or not _table_exists("player_game_batting"):
        return {}

    params: dict[str, Any] = {
        "target_date": target_date,
        "history_limit": int(limit),
    }
    player_id_placeholders = _sql_bind_list(
        "history_player_id",
        sorted(int(player_id) for player_id in player_ids),
        params,
    )
    frame = _safe_frame(
        f"""
        WITH recent AS (
            SELECT
                b.player_id,
                b.game_date,
                b.game_id,
                b.opponent,
                b.hits,
                b.at_bats,
                b.home_runs,
                (
                    COALESCE(b.singles, 0)
                    + 2 * COALESCE(b.doubles, 0)
                    + 3 * COALESCE(b.triples, 0)
                    + 4 * COALESCE(b.home_runs, 0)
                ) AS total_bases,
                ROW_NUMBER() OVER (
                    PARTITION BY b.player_id
                    ORDER BY b.game_date DESC, b.game_id DESC
                ) AS row_rank
            FROM player_game_batting b
            WHERE b.game_date < :target_date
              AND b.player_id IN ({player_id_placeholders})
        )
        SELECT
            player_id,
            game_date,
            game_id,
            opponent,
            hits,
            at_bats,
            home_runs,
            total_bases
        FROM recent
        WHERE row_rank <= :history_limit
        ORDER BY player_id, game_date DESC, game_id DESC
        """,
        params,
    )
    if frame.empty:
        return {}

    history_map: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in _frame_records(frame):
        player_id = int(row["player_id"])
        hits = int(row.get("hits") or 0)
        at_bats = int(row.get("at_bats") or 0)
        history_map[player_id].append(
            {
                "game_date": row.get("game_date"),
                "game_id": row.get("game_id"),
                "opponent": row.get("opponent"),
                "hits": hits,
                "at_bats": at_bats,
                "home_runs": int(row.get("home_runs") or 0),
                "total_bases": int(row.get("total_bases") or 0),
                "had_hit": hits > 0,
            }
        )
    return dict(history_map)


def _attach_hitter_matchup_context(
    player: dict[str, Any],
    opposing_pitcher_throws: Any,
    split_map: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    throw_hand = str(opposing_pitcher_throws or "").strip().upper()[:1]
    if throw_hand not in {"R", "L"}:
        player["opposing_pitcher_throws"] = None
        return player

    player["opposing_pitcher_throws"] = throw_hand
    split = split_map.get(int(player["player_id"]), {}).get(throw_hand, {})
    player.update(
        {
            "vs_pitcher_hand_games": split.get("vs_pitcher_hand_games"),
            "vs_pitcher_hand_hits": split.get("vs_pitcher_hand_hits"),
            "vs_pitcher_hand_at_bats": split.get("vs_pitcher_hand_at_bats"),
            "vs_pitcher_hand_batting_avg": split.get("vs_pitcher_hand_batting_avg"),
            "vs_pitcher_hand_xwoba": split.get("vs_pitcher_hand_xwoba"),
            "vs_pitcher_hand_hard_hit_pct": split.get("vs_pitcher_hand_hard_hit_pct"),
        }
    )
    return player


def _fetch_pitcher_strikeout_market_map(
    target_date: date,
    game_id: int | None = None,
) -> dict[tuple[int, int], dict[str, Any]]:
    if not _table_exists("player_prop_markets"):
        return {}

    filters = ["ppm.game_date = :target_date", "ppm.market_type = 'pitcher_strikeouts'"]
    params: dict[str, Any] = {"target_date": target_date}
    if game_id is not None:
        filters.append("ppm.game_id = :game_id")
        params["game_id"] = game_id

    frame = _safe_frame(
        f"""
        WITH ranked AS (
            SELECT
                ppm.*,
                ROW_NUMBER() OVER (
                    PARTITION BY ppm.game_id, ppm.player_id, ppm.sportsbook, ppm.market_type
                    ORDER BY ppm.snapshot_ts DESC
                ) AS row_rank
            FROM player_prop_markets ppm
            WHERE {' AND '.join(filters)}
        )
        SELECT
            game_id,
            player_id,
            player_name,
            team,
            sportsbook,
            market_type,
            line_value,
            over_price,
            under_price,
            snapshot_ts,
            source_name
        FROM ranked
        WHERE row_rank = 1
        ORDER BY game_id, player_id, sportsbook
        """,
        params,
    )
    if frame.empty:
        return {}

    records = _frame_records(frame)
    by_pitcher: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_pitcher[(int(record["game_id"]), int(record["player_id"]))].append(record)

    market_map: dict[tuple[int, int], dict[str, Any]] = {}
    for key, rows in by_pitcher.items():
        line_values = [float(row["line_value"]) for row in rows if row.get("line_value") is not None]
        over_prices = [int(row["over_price"]) for row in rows if row.get("over_price") is not None]
        under_prices = [int(row["under_price"]) for row in rows if row.get("under_price") is not None]
        market_map[key] = {
            "market_type": "pitcher_strikeouts",
            "player_name": next((row.get("player_name") for row in rows if row.get("player_name")), None),
            "team": next((row.get("team") for row in rows if row.get("team")), None),
            "consensus_line": round(float(pd.Series(line_values).median()), 2) if line_values else None,
            "line_min": min(line_values) if line_values else None,
            "line_max": max(line_values) if line_values else None,
            "best_over_price": max(over_prices) if over_prices else None,
            "best_under_price": max(under_prices) if under_prices else None,
            "sportsbook_count": len(rows),
            "sportsbooks": [str(row.get("sportsbook")) for row in rows if row.get("sportsbook")],
            "source_names": sorted({str(row.get("source_name")) for row in rows if row.get("source_name")}),
            "latest_snapshot_ts": max((row.get("snapshot_ts") for row in rows if row.get("snapshot_ts") is not None), default=None),
        }
    return market_map


def _merge_strikeout_market_context(
    projection: dict[str, Any] | None,
    market_context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if projection is None:
        return None
    merged = dict(projection)
    current_market = dict(merged.get("market") or {})
    market_line = _to_float((market_context or {}).get("consensus_line"))
    if market_line is None:
        market_line = _to_float(current_market.get("consensus_line"))
    projected = _to_float(merged.get("projected_strikeouts"))
    merged["market"] = {
        "consensus_line": market_line,
        "line_min": None if market_context is None else market_context.get("line_min"),
        "line_max": None if market_context is None else market_context.get("line_max"),
        "best_over_price": None if market_context is None else market_context.get("best_over_price"),
        "best_under_price": None if market_context is None else market_context.get("best_under_price"),
        "sportsbook_count": None if market_context is None else market_context.get("sportsbook_count"),
        "sportsbooks": [] if market_context is None else market_context.get("sportsbooks") or [],
        "source_names": [] if market_context is None else market_context.get("source_names") or [],
        "latest_snapshot_ts": None if market_context is None else market_context.get("latest_snapshot_ts"),
        "projection_delta": None if projected is None or market_line is None else round(projected - market_line, 2),
    }
    return merged


def _fetch_pitcher_strikeout_prediction_map(
    target_date: date,
    game_id: int | None = None,
) -> dict[tuple[int, int], dict[str, Any]]:
    if not _table_exists("predictions_pitcher_strikeouts"):
        return {}

    prediction_filters = ["p.game_date = :target_date"]
    feature_filters = ["f.game_date = :target_date"]
    params: dict[str, Any] = {"target_date": target_date}
    if game_id is not None:
        prediction_filters.append("p.game_id = :game_id")
        feature_filters.append("f.game_id = :game_id")
        params["game_id"] = game_id

    feature_join = ""
    feature_select = ""
    if _table_exists("game_features_pitcher_strikeouts"):
        feature_join = f"""
        , ranked_features AS (
            SELECT
                f.*,
                ROW_NUMBER() OVER (
                    PARTITION BY f.game_id, f.pitcher_id
                    ORDER BY f.prediction_ts DESC
                ) AS row_rank
            FROM game_features_pitcher_strikeouts f
            WHERE {' AND '.join(feature_filters)}
        )
        """
        feature_select = """
            CAST(NULLIF(f.feature_payload ->> 'baseline_strikeouts', '') AS DOUBLE PRECISION) AS baseline_strikeouts,
            CAST(NULLIF(f.feature_payload ->> 'opponent_lineup_k_pct', '') AS DOUBLE PRECISION) AS opponent_lineup_k_pct,
            CAST(NULLIF(f.feature_payload ->> 'opponent_k_pct_blended', '') AS DOUBLE PRECISION) AS opponent_k_pct_blended,
            CAST(NULLIF(f.feature_payload ->> 'same_hand_share', '') AS DOUBLE PRECISION) AS same_hand_share,
            CAST(NULLIF(f.feature_payload ->> 'opposite_hand_share', '') AS DOUBLE PRECISION) AS opposite_hand_share,
            CAST(NULLIF(f.feature_payload ->> 'switch_share', '') AS DOUBLE PRECISION) AS switch_share,
            CAST(NULLIF(f.feature_payload ->> 'lineup_right_count', '') AS INTEGER) AS lineup_right_count,
            CAST(NULLIF(f.feature_payload ->> 'lineup_left_count', '') AS INTEGER) AS lineup_left_count,
            CAST(NULLIF(f.feature_payload ->> 'lineup_switch_count', '') AS INTEGER) AS lineup_switch_count,
            CAST(NULLIF(f.feature_payload ->> 'known_hitters', '') AS INTEGER) AS known_hitters,
            CAST(NULLIF(f.feature_payload ->> 'confirmed_hitters', '') AS INTEGER) AS confirmed_hitters,
            CAST(NULLIF(f.feature_payload ->> 'total_hitters', '') AS INTEGER) AS total_hitters,
            CAST(NULLIF(f.feature_payload ->> 'handedness_adjustment_applied', '') AS BOOLEAN) AS handedness_adjustment_applied,
            CAST(NULLIF(f.feature_payload ->> 'handedness_data_missing', '') AS BOOLEAN) AS handedness_data_missing,
            CAST(NULLIF(f.feature_payload ->> 'recent_avg_strikeouts_3', '') AS DOUBLE PRECISION) AS recent_avg_strikeouts_3,
            CAST(NULLIF(f.feature_payload ->> 'recent_avg_strikeouts_5', '') AS DOUBLE PRECISION) AS recent_avg_strikeouts_5,
        """

    frame = _safe_frame(
        f"""
        WITH ranked_predictions AS (
            SELECT
                p.*,
                ROW_NUMBER() OVER (
                    PARTITION BY p.game_id, p.pitcher_id
                    ORDER BY p.prediction_ts DESC
                ) AS row_rank
            FROM predictions_pitcher_strikeouts p
            WHERE {' AND '.join(prediction_filters)}
        )
        {feature_join}
        SELECT
            p.game_id,
            p.pitcher_id,
            p.prediction_ts,
            p.model_name,
            p.model_version,
            p.predicted_strikeouts,
            p.over_probability,
            p.under_probability,
            p.market_line,
            {feature_select if feature_select else ''}
            dp.full_name AS pitcher_name
        FROM ranked_predictions p
        LEFT JOIN dim_players dp ON dp.player_id = p.pitcher_id
        {'LEFT JOIN ranked_features f ON f.game_id = p.game_id AND f.pitcher_id = p.pitcher_id AND f.row_rank = 1' if feature_select else ''}
        WHERE p.row_rank = 1
        """,
        params,
    )
    if frame.empty:
        return {}

    prediction_map: dict[tuple[int, int], dict[str, Any]] = {}
    for row in _frame_records(frame):
        known_hitters = int(row.get("known_hitters") or 0)
        prediction_map[(int(row["game_id"]), int(row["pitcher_id"]))] = {
            "source": "model",
            "pitcher_name": row.get("pitcher_name"),
            "prediction_ts": row.get("prediction_ts"),
            "model_name": row.get("model_name"),
            "model_version": row.get("model_version"),
            "projected_strikeouts": row.get("predicted_strikeouts"),
            "baseline_strikeouts": row.get("baseline_strikeouts"),
            "opponent_lineup_k_pct": row.get("opponent_lineup_k_pct"),
            "opponent_k_pct_blended": row.get("opponent_k_pct_blended"),
            "opponent_k_pct_used": row.get("opponent_lineup_k_pct") if row.get("opponent_lineup_k_pct") is not None else row.get("opponent_k_pct_blended"),
            "sample_starts": 5 if row.get("recent_avg_strikeouts_5") is not None else (3 if row.get("recent_avg_strikeouts_3") is not None else None),
            "handedness_adjustment_applied": bool(row.get("handedness_adjustment_applied")) if row.get("handedness_adjustment_applied") is not None else False,
            "handedness_data_missing": bool(row.get("handedness_data_missing")) if row.get("handedness_data_missing") is not None else known_hitters == 0,
            "lineup_handedness": {
                "counts": {
                    "R": int(row.get("lineup_right_count") or 0),
                    "L": int(row.get("lineup_left_count") or 0),
                    "S": int(row.get("lineup_switch_count") or 0),
                },
                "known_hitters": known_hitters,
                "confirmed_hitters": row.get("confirmed_hitters"),
                "total_hitters": row.get("total_hitters"),
                "same_hand_share": row.get("same_hand_share"),
                "opposite_hand_share": row.get("opposite_hand_share"),
                "switch_share": row.get("switch_share"),
            },
            "market": {
                "consensus_line": row.get("market_line"),
            },
            "over_probability": row.get("over_probability"),
            "under_probability": row.get("under_probability"),
        }
    return prediction_map


def _estimate_starter_strikeout_projection(
    starter: dict[str, Any] | None,
    recent_form: dict[str, Any] | None,
    opponent_lineup_k_pct: Any,
    opponent_k_pct_blended: Any,
    opponent_lineup_handedness: dict[str, Any] | None = None,
    market_context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if starter is None:
        return None

    recent = recent_form or {}
    baseline = _to_float(recent.get("avg_strikeouts")) or _to_float(starter.get("strikeouts"))
    opponent_lineup = _to_float(opponent_lineup_k_pct)
    opponent_blended = _to_float(opponent_k_pct_blended)
    opponent_used = opponent_lineup if opponent_lineup is not None else opponent_blended
    avg_ip = _to_float(recent.get("avg_ip")) or _to_float(starter.get("ip"))
    whiff_pct = _to_float(recent.get("whiff_pct")) or _to_float(starter.get("whiff_pct"))
    csw_pct = _to_float(recent.get("csw_pct")) or _to_float(starter.get("csw_pct"))
    throw_hand = str(starter.get("throws") or "").strip().upper()[:1]
    handedness = opponent_lineup_handedness or {}
    counts = handedness.get("counts") or {}
    known_hitters = int(handedness.get("known_hitters") or 0)

    same_hand_share = None
    opposite_hand_share = None
    switch_share = None
    if throw_hand in {"R", "L"} and known_hitters > 0:
        same_count = counts.get(throw_hand, 0)
        opposite_count = counts.get("L" if throw_hand == "R" else "R", 0)
        switch_count = counts.get("S", 0)
        same_hand_share = same_count / known_hitters
        opposite_hand_share = opposite_count / known_hitters
        switch_share = switch_count / known_hitters

    market_line = _to_float((market_context or {}).get("consensus_line"))

    projected = None
    if baseline is not None:
        projection = baseline
        if opponent_used is not None:
            projection *= _clamp(1 + ((opponent_used - 0.22) / 0.22) * 0.70, 0.82, 1.22)
        if avg_ip is not None:
            projection *= _clamp(avg_ip / 5.4, 0.85, 1.18)
        if whiff_pct is not None:
            projection *= _clamp(1 + ((whiff_pct - 0.27) / 0.27) * 0.35, 0.90, 1.12)
        elif csw_pct is not None:
            projection *= _clamp(1 + ((csw_pct - 0.29) / 0.29) * 0.25, 0.92, 1.10)
        if same_hand_share is not None and opposite_hand_share is not None:
            projection *= _clamp(1 + ((same_hand_share - opposite_hand_share) * 0.18) - ((switch_share or 0.0) * 0.05), 0.93, 1.08)
        projected = round(projection, 1)

    market_delta = None if projected is None or market_line is None else round(projected - market_line, 2)
    handedness_data_missing = known_hitters == 0
    handedness_adjustment_applied = same_hand_share is not None and opposite_hand_share is not None

    return {
        "projected_strikeouts": projected,
        "baseline_strikeouts": baseline,
        "opponent_lineup_k_pct": opponent_lineup,
        "opponent_k_pct_blended": opponent_blended,
        "opponent_k_pct_used": opponent_used,
        "sample_starts": recent.get("sample_starts"),
        "handedness_adjustment_applied": handedness_adjustment_applied,
        "handedness_data_missing": handedness_data_missing,
        "lineup_handedness": {
            "counts": {
                "R": counts.get("R", 0),
                "L": counts.get("L", 0),
                "S": counts.get("S", 0),
            },
            "known_hitters": known_hitters,
            "confirmed_hitters": handedness.get("confirmed_hitters"),
            "total_hitters": handedness.get("total_hitters"),
            "same_hand_share": same_hand_share,
            "opposite_hand_share": opposite_hand_share,
            "switch_share": switch_share,
        },
        "market": {
            "consensus_line": market_line,
            "line_min": None if market_context is None else market_context.get("line_min"),
            "line_max": None if market_context is None else market_context.get("line_max"),
            "best_over_price": None if market_context is None else market_context.get("best_over_price"),
            "best_under_price": None if market_context is None else market_context.get("best_under_price"),
            "sportsbook_count": None if market_context is None else market_context.get("sportsbook_count"),
            "sportsbooks": [] if market_context is None else market_context.get("sportsbooks") or [],
            "source_names": [] if market_context is None else market_context.get("source_names") or [],
            "latest_snapshot_ts": None if market_context is None else market_context.get("latest_snapshot_ts"),
            "projection_delta": market_delta,
        },
    }


def _fetch_status(target_date: date) -> dict[str, Any]:
    db_connected = not _safe_frame("SELECT 1 AS ok").empty
    totals_count = 0
    hits_count = 0
    strikeouts_count = 0
    rebuild_blocker = _desktop_rebuild_blocker(target_date)
    if _table_exists("predictions_totals"):
        totals = _safe_frame(
            "SELECT COUNT(*) AS row_count FROM predictions_totals WHERE game_date = :target_date",
            {"target_date": target_date},
        )
        if not totals.empty:
            totals_count = int(totals.iloc[0]["row_count"])
    if _table_exists("predictions_player_hits"):
        hits = _safe_frame(
            "SELECT COUNT(*) AS row_count FROM predictions_player_hits WHERE game_date = :target_date",
            {"target_date": target_date},
        )
        if not hits.empty:
            hits_count = int(hits.iloc[0]["row_count"])
    if _table_exists("predictions_pitcher_strikeouts"):
        strikeouts = _safe_frame(
            "SELECT COUNT(*) AS row_count FROM predictions_pitcher_strikeouts WHERE game_date = :target_date",
            {"target_date": target_date},
        )
        if not strikeouts.empty:
            strikeouts_count = int(strikeouts.iloc[0]["row_count"])
    return {
        "target_date": target_date,
        "db_connected": db_connected,
        "totals_artifact_ready": _artifact_ready("totals"),
        "hits_artifact_ready": _artifact_ready("hits"),
        "strikeouts_artifact_ready": _artifact_ready("strikeouts"),
        "totals_predictions": totals_count,
        "hits_predictions": hits_count,
        "strikeouts_predictions": strikeouts_count,
        "rebuild_blocker": rebuild_blocker,
        "tables": {
            "games": _table_exists("games"),
            "game_features_totals": _table_exists("game_features_totals"),
            "player_features_hits": _table_exists("player_features_hits"),
            "game_features_pitcher_strikeouts": _table_exists("game_features_pitcher_strikeouts"),
            "predictions_totals": _table_exists("predictions_totals"),
            "predictions_player_hits": _table_exists("predictions_player_hits"),
            "predictions_pitcher_strikeouts": _table_exists("predictions_pitcher_strikeouts"),
            "player_trend_daily": _table_exists("player_trend_daily"),
            "pitcher_trend_daily": _table_exists("pitcher_trend_daily"),
            "model_scorecards_daily": _table_exists("model_scorecards_daily"),
        },
    }


def _redact_database_url(database_url: str) -> str:
    if not database_url:
        return ""
    if database_url.startswith("sqlite"):
        return database_url
    if "://" not in database_url or "@" not in database_url:
        return database_url
    scheme, remainder = database_url.split("://", 1)
    credentials, suffix = remainder.split("@", 1)
    if not credentials:
        return database_url
    if ":" in credentials:
        username, _password = credentials.split(":", 1)
        credentials = f"{username}:***"
    else:
        credentials = "***"
    return f"{scheme}://{credentials}@{suffix}"


def _fetch_pipeline_runs(limit: int = 20) -> list[dict[str, Any]]:
    if not _table_exists("pipeline_runs"):
        return []
    runs_frame = _safe_frame(
        """
        SELECT job_id, action, label, target_date, status,
               total_steps, completed_steps, error,
               created_at, started_at, finished_at
        FROM pipeline_runs
        ORDER BY created_at DESC
        LIMIT :lim
        """,
        {"lim": max(1, min(limit, 100))},
    )
    return _frame_records(runs_frame)


def _fetch_game_readiness_payload(target_date: date | None = None) -> dict[str, Any]:
    td = target_date or date.today()
    if not _table_exists("game_readiness"):
        return {"games": [], "summary": {"green": 0, "yellow": 0, "red": 0, "total": 0}}
    frame = _safe_frame(
        """
        SELECT game_id, game_date, away_team, home_team,
               has_away_starter, has_home_starter, has_market, has_venue,
               has_away_lineup, has_home_lineup, has_weather,
               checks_passed, checks_total, warnings, badge
        FROM game_readiness
        WHERE game_date = :target_date
        ORDER BY game_id
        """,
        {"target_date": str(td)},
    )
    records = _frame_records(frame)
    summary = {
        "green": sum(1 for r in records if r.get("badge") == "green"),
        "yellow": sum(1 for r in records if r.get("badge") == "yellow"),
        "red": sum(1 for r in records if r.get("badge") == "red"),
        "total": len(records),
    }
    return {"games": records, "summary": summary}


def _fetch_source_health_payload(hours: int = 24) -> dict[str, Any]:
    capped_hours = max(1, min(hours, 168))
    if not _table_exists("source_health"):
        return {
            "sources": [],
            "hours": capped_hours,
            "summary": {"total_sources": 0, "healthy_sources": 0, "sources_with_failures": 0},
        }
    cutoff = datetime.now(timezone.utc) - timedelta(hours=capped_hours)
    frame = _safe_frame(
        """
        SELECT source_name, checked_at, is_available, response_time_ms, error_message
        FROM source_health
        ORDER BY checked_at DESC
        """
    )
    grouped: dict[str, dict[str, Any]] = {}
    for row in _frame_records(frame):
        raw_ts = str(row.get("checked_at") or "").replace("Z", "+00:00")
        try:
            checked_at = datetime.fromisoformat(raw_ts)
        except ValueError:
            continue
        if checked_at.tzinfo is None:
            checked_at = checked_at.replace(tzinfo=timezone.utc)
        if checked_at < cutoff:
            continue
        source_name = str(row.get("source_name") or "unknown")
        entry = grouped.setdefault(
            source_name,
            {
                "source_name": source_name,
                "latest_checked_at": checked_at.isoformat(),
                "success_count": 0,
                "failure_count": 0,
                "avg_response_time_ms": None,
                "latest_error": None,
                "_response_times": [],
            },
        )
        if row.get("is_available"):
            entry["success_count"] += 1
        else:
            entry["failure_count"] += 1
            if not entry["latest_error"] and row.get("error_message"):
                entry["latest_error"] = row.get("error_message")
        if row.get("response_time_ms") is not None:
            entry["_response_times"].append(float(row["response_time_ms"]))
    sources = []
    for entry in grouped.values():
        times = entry.pop("_response_times")
        entry["avg_response_time_ms"] = round(sum(times) / len(times), 1) if times else None
        sources.append(entry)
    sources.sort(key=lambda item: item["source_name"])
    summary = {
        "total_sources": len(sources),
        "healthy_sources": sum(1 for item in sources if int(item.get("failure_count") or 0) == 0),
        "sources_with_failures": sum(1 for item in sources if int(item.get("failure_count") or 0) > 0),
    }
    return {"sources": sources, "hours": capped_hours, "summary": summary}


def _doctor_check(name: str, label: str, ok: bool, severity: str, detail: str) -> dict[str, Any]:
    return {
        "name": name,
        "label": label,
        "ok": ok,
        "severity": severity,
        "detail": detail,
    }


def _doctor_payload(
    target_date: date,
    source_health_hours: int = 24,
    pipeline_limit: int = 5,
    update_history_limit: int = 5,
) -> dict[str, Any]:
    status = _fetch_status(target_date)
    readiness = _fetch_game_readiness_payload(target_date)
    source_health = _fetch_source_health_payload(source_health_hours)
    pipeline_runs = _fetch_pipeline_runs(pipeline_limit)
    active_job = _active_update_job_payload()
    recent_jobs = _update_job_history_payload()[: max(1, min(update_history_limit, 20))]

    artifacts_ready = all(
        bool(status.get(key))
        for key in ("totals_artifact_ready", "hits_artifact_ready", "strikeouts_artifact_ready")
    )
    prediction_rows = int(status.get("totals_predictions") or 0) + int(status.get("hits_predictions") or 0) + int(status.get("strikeouts_predictions") or 0)
    readiness_summary = readiness.get("summary", {})
    readiness_total = int(readiness_summary.get("total") or 0)
    readiness_red = int(readiness_summary.get("red") or 0)
    source_summary = source_health.get("summary", {})
    source_failures = int(source_summary.get("sources_with_failures") or 0)
    blocker = status.get("rebuild_blocker")

    checks = [
        _doctor_check(
            "database_connection",
            "Database connection",
            bool(status.get("db_connected")),
            "critical",
            "Local runtime database is reachable." if status.get("db_connected") else "The current runtime database is not reachable.",
        ),
        _doctor_check(
            "model_artifacts",
            "Model artifacts ready",
            artifacts_ready,
            "warning",
            "Totals, hits, and strikeout artifacts are available." if artifacts_ready else "One or more model artifacts are missing from the local runtime.",
        ),
        _doctor_check(
            "prediction_outputs",
            "Prediction outputs present",
            prediction_rows > 0,
            "warning",
            f"Found {prediction_rows} prediction rows for {target_date.isoformat()}." if prediction_rows > 0 else f"No prediction rows are present for {target_date.isoformat()}.",
        ),
        _doctor_check(
            "desktop_history_seed",
            "Desktop history seed",
            blocker is None,
            "critical",
            "Historical SQLite seed looks complete for rebuild flows." if blocker is None else str(blocker.get("message") or "Desktop history is incomplete."),
        ),
        _doctor_check(
            "game_readiness",
            "Game readiness",
            readiness_total == 0 or readiness_red == 0,
            "warning",
            "No readiness rows are available yet." if readiness_total == 0 else f"Readiness rows: green={readiness_summary.get('green', 0)} yellow={readiness_summary.get('yellow', 0)} red={readiness_red}.",
        ),
        _doctor_check(
            "source_health",
            "Source health",
            source_failures == 0,
            "warning",
            "No recent source failures recorded." if source_failures == 0 else f"Recent source failures recorded for {source_failures} source(s).",
        ),
    ]
    critical_failures = sum(1 for item in checks if not item["ok"] and item["severity"] == "critical")
    warning_failures = sum(1 for item in checks if not item["ok"] and item["severity"] != "critical")
    overall_status = "error" if critical_failures else ("warn" if warning_failures else "ok")

    return {
        "target_date": target_date.isoformat(),
        "overall": {
            "status": overall_status,
            "critical_failures": critical_failures,
            "warning_failures": warning_failures,
            "checks_total": len(checks),
        },
        "runtime": {
            "db_dialect": DB_DIALECT,
            "database_url": _redact_database_url(settings.database_url),
            "data_dir": str(settings.data_dir),
            "model_dir": str(settings.model_dir),
            "report_dir": str(settings.report_dir),
            "feature_dir": str(settings.feature_dir),
        },
        "checks": checks,
        "status": status,
        "game_readiness": readiness,
        "source_health": source_health,
        "pipeline_runs": {"runs": pipeline_runs},
        "update_jobs": {"active_job": active_job, "recent_jobs": recent_jobs},
    }


def _fetch_player_trend(player_id: int, target_date: date, limit: int = 10) -> list[dict[str, Any]]:
    if not _table_exists("player_trend_daily"):
        return []
    frame = _safe_frame(
        """
        SELECT *
        FROM player_trend_daily
        WHERE player_id = :player_id
          AND game_date <= :target_date
        ORDER BY game_date DESC
        LIMIT :limit
        """,
        {"player_id": player_id, "target_date": target_date, "limit": limit},
    )
    return _frame_records(frame)


def _fetch_pitcher_trend(pitcher_id: int, target_date: date, limit: int = 10) -> list[dict[str, Any]]:
    if not _table_exists("pitcher_trend_daily"):
        return []
    frame = _safe_frame(
        """
        SELECT *
        FROM pitcher_trend_daily
        WHERE pitcher_id = :pitcher_id
          AND game_date <= :target_date
        ORDER BY game_date DESC
        LIMIT :limit
        """,
        {"pitcher_id": pitcher_id, "target_date": target_date, "limit": limit},
    )
    return _frame_records(frame)


def _fetch_pitcher_recent_starts(pitcher_id: int | None, target_date: date, limit: int = 5) -> list[dict[str, Any]]:
    if pitcher_id is None or not _table_exists("pitcher_starts"):
        return []
    history_frame = _safe_frame(
        """
        SELECT
            ps.game_date,
            ps.game_id,
            ps.team,
            ps.ip,
            ps.strikeouts,
            ps.pitch_count,
            CASE
                WHEN g.home_team = ps.team THEN g.away_team
                WHEN g.away_team = ps.team THEN g.home_team
                ELSE NULL
            END AS opponent
        FROM pitcher_starts ps
        LEFT JOIN games g
          ON g.game_id = ps.game_id
         AND g.game_date = ps.game_date
        WHERE ps.pitcher_id = :pitcher_id
          AND ps.game_date < :target_date
        ORDER BY ps.game_date DESC, ps.game_id DESC
        LIMIT :history_limit
        """,
        {"pitcher_id": pitcher_id, "target_date": target_date, "history_limit": limit},
    )
    return [
        {
            "game_date": row.get("game_date"),
            "game_id": row.get("game_id"),
            "team": row.get("team"),
            "opponent": row.get("opponent"),
            "ip": _coerce_float(row.get("ip")),
            "strikeouts": int(row.get("strikeouts") or 0),
            "pitch_count": int(row.get("pitch_count") or 0)
            if row.get("pitch_count") is not None
            else None,
        }
        for row in _frame_records(history_frame)
    ]


def _fetch_model_scorecards(target_date: date, window_days: int = 14) -> dict[str, Any]:
    if not _table_exists("model_scorecards_daily"):
        return {"latest_score_date": None, "window_days": window_days, "rows": []}
    start_date = target_date - pd.Timedelta(days=max(window_days - 1, 0))
    frame = _safe_frame(
        """
        SELECT *
        FROM model_scorecards_daily
        WHERE score_date BETWEEN :start_date AND :target_date
        ORDER BY score_date DESC, market, model_name, model_version
        """,
        {"start_date": start_date, "target_date": target_date},
    )
    if frame.empty:
        return {"latest_score_date": None, "window_days": window_days, "rows": []}

    frame["score_date"] = pd.to_datetime(frame["score_date"])
    latest_rows = frame.sort_values(["market", "score_date", "model_name", "model_version"], ascending=[True, False, True, True]).groupby("market", as_index=False).head(1)
    rows: list[dict[str, Any]] = []
    for record in _frame_records(latest_rows):
        trailing = frame[
            (frame["market"] == record["market"])
            & (frame["model_name"] == record["model_name"])
            & (frame["model_version"] == record["model_version"])
        ].copy()
        graded_weight = pd.to_numeric(trailing["graded_count"], errors="coerce").fillna(0)
        prediction_weight = pd.to_numeric(trailing["predictions_count"], errors="coerce").fillna(0)
        clv_weight = pd.to_numeric(
            trailing["clv_count"] if "clv_count" in trailing.columns else pd.Series([0] * len(trailing)),
            errors="coerce",
        ).fillna(0)

        def _weighted_average(column: str, weights: pd.Series) -> float | None:
            values = pd.to_numeric(trailing[column], errors="coerce")
            valid = values.notna() & (weights > 0)
            if not valid.any():
                return None
            return float((values[valid] * weights[valid]).sum() / weights[valid].sum())

        rows.append(
            {
                **record,
                "latest_score_date": record.get("score_date"),
                "trailing_days": int(trailing["score_date"].nunique()),
                "trailing_predictions": int(prediction_weight.sum()),
                "trailing_graded": int(graded_weight.sum()),
                "trailing_clv_count": int(clv_weight.sum()),
                "trailing_success_rate": _weighted_average("success_rate", graded_weight),
                "trailing_avg_absolute_error": _weighted_average("avg_absolute_error", graded_weight),
                "trailing_brier_score": _weighted_average("brier_score", graded_weight),
                "trailing_beat_market_rate": _weighted_average("beat_market_rate", graded_weight),
                "trailing_avg_clv_side_value": _weighted_average("avg_clv_side_value", clv_weight),
                "trailing_positive_clv_rate": _weighted_average("positive_clv_rate", clv_weight),
            }
        )
    latest_score_date = max(record["latest_score_date"] for record in rows if record.get("latest_score_date")) if rows else None
    return {"latest_score_date": latest_score_date, "window_days": window_days, "rows": rows}


def _fetch_totals_predictions(target_date: date) -> list[dict[str, Any]]:
    if not _table_exists("predictions_totals"):
        return []
    game_start_order = _sql_order_nulls_last("g.game_start_ts")
    frame = _safe_frame(
        f"""
        WITH ranked AS (
            SELECT
                p.*,
                ROW_NUMBER() OVER (PARTITION BY p.game_id ORDER BY p.prediction_ts DESC) AS row_rank
            FROM predictions_totals p
            WHERE p.game_date = :target_date
        )
        SELECT
            r.game_id,
            r.game_date,
            COALESCE(g.away_team, 'TBD') AS away_team,
            COALESCE(g.home_team, 'TBD') AS home_team,
            g.game_start_ts,
            r.model_name,
            r.model_version,
            r.prediction_ts,
            r.predicted_total_runs,
            r.market_total,
            r.over_probability,
            r.under_probability,
            r.edge,
            r.confidence_level,
            r.suppress_reason,
            r.lane_status
        FROM ranked r
        LEFT JOIN games g ON g.game_id = r.game_id
        WHERE r.row_rank = 1
        ORDER BY {game_start_order}, away_team, home_team
        """,
        {"target_date": target_date},
    )
    return _frame_records(frame)


def _fetch_hit_predictions(
    target_date: date,
    limit: int,
    min_probability: float,
    confirmed_only: bool,
    include_inferred: bool,
) -> list[dict[str, Any]]:
    if not _table_exists("predictions_player_hits"):
        return []

    player_name_expr = _sql_json_text("f.feature_payload", "player_name")
    lineup_slot_expr = _sql_integer(f"NULLIF({_sql_json_text('f.feature_payload', 'lineup_slot')}, '')")
    confirmed_lineup_expr = _sql_boolean(f"NULLIF({_sql_json_text('f.feature_payload', 'is_confirmed_lineup')}, '')")
    projected_pa_expr = _sql_real(f"NULLIF({_sql_json_text('f.feature_payload', 'projected_plate_appearances')}, '')")
    streak_len_expr = _sql_integer(f"NULLIF({_sql_json_text('f.feature_payload', 'streak_len_capped')}, '')")
    streak_len_full_expr = _sql_integer(f"NULLIF({_sql_json_text('f.feature_payload', 'streak_len')}, '')")
    confirmed_order = _sql_order_nulls_last(confirmed_lineup_expr, "DESC")
    projected_pa_order = _sql_order_nulls_last(projected_pa_expr, "DESC")
    streak_order = _sql_order_nulls_last(streak_len_expr, "DESC")
    frame = _safe_frame(
        f"""
        WITH ranked_predictions AS (
            SELECT
                p.*,
                ROW_NUMBER() OVER (
                    PARTITION BY p.game_id, p.player_id
                    ORDER BY p.prediction_ts DESC
                ) AS row_rank
            FROM predictions_player_hits p
            WHERE p.game_date = :target_date
        ),
        ranked_features AS (
            SELECT
                f.*,
                ROW_NUMBER() OVER (
                    PARTITION BY f.game_id, f.player_id
                    ORDER BY f.prediction_ts DESC
                ) AS row_rank
            FROM player_features_hits f
            WHERE f.game_date = :target_date
        )
        SELECT
            p.game_id,
            p.game_date,
            p.player_id,
            COALESCE({player_name_expr}, dp.full_name, CAST(p.player_id AS TEXT)) AS player_name,
            COALESCE(f.team, p.team) AS team,
            COALESCE(
                f.opponent,
                CASE
                    WHEN g.home_team = COALESCE(f.team, p.team) THEN g.away_team
                    WHEN g.away_team = COALESCE(f.team, p.team) THEN g.home_team
                    ELSE NULL
                END,
                'TBD'
            ) AS opponent,
            {lineup_slot_expr} AS lineup_slot,
            {confirmed_lineup_expr} AS is_confirmed_lineup,
            {projected_pa_expr} AS projected_plate_appearances,
            {streak_len_expr} AS streak_len_capped,
            {streak_len_full_expr} AS streak_len,
            p.prediction_ts,
            p.predicted_hit_probability,
            p.fair_price,
            p.market_price,
            p.edge
        FROM ranked_predictions p
        LEFT JOIN ranked_features f
            ON f.game_id = p.game_id
           AND f.player_id = p.player_id
           AND f.row_rank = 1
        LEFT JOIN games g ON g.game_id = p.game_id
        LEFT JOIN dim_players dp ON dp.player_id = p.player_id
        WHERE p.row_rank = 1
          AND p.predicted_hit_probability >= :min_probability
                ORDER BY p.predicted_hit_probability DESC,
                                 {confirmed_order},
                                 {projected_pa_order},
                                 {streak_order},
                                 player_name
        LIMIT :limit
        """,
        {
            "target_date": target_date,
            "limit": limit,
            "min_probability": min_probability,
        },
    )
    records = _frame_records(frame)
    records = _annotate_lineup_confidence(records, _fetch_lineup_snapshot_keys(target_date))
    if confirmed_only:
        records = [record for record in records if record.get("is_confirmed_lineup")]
    if not include_inferred:
        records = [record for record in records if not record.get("is_inferred_lineup")]
    return records


def _fetch_game_board(
    target_date: date,
    hit_limit_per_team: int,
    min_probability: float,
    confirmed_only: bool,
    include_inferred: bool,
) -> list[dict[str, Any]]:
    if not _table_exists("games"):
        return []

    game_start_order = _sql_order_nulls_last("g.game_start_ts")
    recent_batting_avg_expr = _sql_ratio("recent.hits", "recent.at_bats")
    season_batting_avg_expr = _sql_ratio("b.hits", "b.at_bats")
    game_year_expr = _sql_year("b.game_date")
    target_year_expr = _sql_year_param("target_date")
    hit_player_name_expr = _sql_json_text("f.feature_payload", "player_name")
    hit_lineup_slot_expr = _sql_integer(f"NULLIF({_sql_json_text('f.feature_payload', 'lineup_slot')}, '')")
    hit_confirmed_lineup_expr = _sql_boolean(f"NULLIF({_sql_json_text('f.feature_payload', 'is_confirmed_lineup')}, '')")
    hit_projected_pa_expr = _sql_real(f"NULLIF({_sql_json_text('f.feature_payload', 'projected_plate_appearances')}, '')")
    hit_streak_len_expr = _sql_integer(f"NULLIF({_sql_json_text('f.feature_payload', 'streak_len_capped')}, '')")
    hit_streak_len_full_expr = _sql_integer(f"NULLIF({_sql_json_text('f.feature_payload', 'streak_len')}, '')")
    hit_rate_blended_expr = _sql_real(_sql_json_text("f.feature_payload", "hit_rate_blended"))
    xwoba_14_expr = _sql_real(_sql_json_text("f.feature_payload", "xwoba_14"))
    opp_starter_xwoba_expr = _sql_real(_sql_json_text("f.feature_payload", "opposing_starter_xwoba"))
    opp_starter_csw_expr = _sql_real(_sql_json_text("f.feature_payload", "opposing_starter_csw"))
    team_run_environment_expr = _sql_real(_sql_json_text("f.feature_payload", "team_run_environment"))
    park_hr_factor_expr = _sql_real(_sql_json_text("f.feature_payload", "park_hr_factor"))
    projected_pa_order = _sql_order_nulls_last("j.projected_plate_appearances", "DESC")
    lineup_slot_order = _sql_order_nulls_last("j.lineup_slot")
    final_lineup_slot_order = _sql_order_nulls_last("lineup_slot")

    games_frame = _safe_frame(
        f"""
        WITH ranked_predictions AS (
            SELECT
                p.*,
                ROW_NUMBER() OVER (PARTITION BY p.game_id ORDER BY p.prediction_ts DESC) AS row_rank
            FROM predictions_totals p
            WHERE p.game_date = :target_date
        ),
        ranked_features AS (
            SELECT
                f.*,
                ROW_NUMBER() OVER (PARTITION BY f.game_id ORDER BY f.prediction_ts DESC) AS row_rank
            FROM game_features_totals f
            WHERE f.game_date = :target_date
        )
        SELECT
            g.game_id,
            g.game_date,
            g.status,
            g.away_team,
            g.home_team,
            g.game_start_ts,
            g.away_runs,
            g.home_runs,
            g.total_runs,
            g.home_win,
            COALESCE(v.venue_name, g.venue_name) AS venue_name,
            v.city AS venue_city,
            v.state AS venue_state,
            v.roof_type,
            p.model_name,
            p.model_version,
            p.prediction_ts,
            p.predicted_total_runs,
            p.market_total,
            p.over_probability,
            p.under_probability,
            p.edge,
            p.confidence_level,
            p.suppress_reason,
            p.lane_status,
            CAST(f.feature_payload ->> 'away_runs_rate_blended' AS DOUBLE PRECISION) AS away_expected_runs,
            CAST(f.feature_payload ->> 'home_runs_rate_blended' AS DOUBLE PRECISION) AS home_expected_runs,
            CAST(f.feature_payload ->> 'away_bullpen_pitches_last3' AS DOUBLE PRECISION) AS away_bullpen_pitches_last3,
            CAST(f.feature_payload ->> 'home_bullpen_pitches_last3' AS DOUBLE PRECISION) AS home_bullpen_pitches_last3,
            CAST(f.feature_payload ->> 'away_bullpen_innings_last3' AS DOUBLE PRECISION) AS away_bullpen_innings_last3,
            CAST(f.feature_payload ->> 'home_bullpen_innings_last3' AS DOUBLE PRECISION) AS home_bullpen_innings_last3,
            CAST(f.feature_payload ->> 'away_bullpen_b2b' AS DOUBLE PRECISION) AS away_bullpen_b2b,
            CAST(f.feature_payload ->> 'home_bullpen_b2b' AS DOUBLE PRECISION) AS home_bullpen_b2b,
            CAST(f.feature_payload ->> 'away_bullpen_runs_allowed_last3' AS DOUBLE PRECISION) AS away_bullpen_runs_allowed_last3,
            CAST(f.feature_payload ->> 'home_bullpen_runs_allowed_last3' AS DOUBLE PRECISION) AS home_bullpen_runs_allowed_last3,
            CAST(f.feature_payload ->> 'away_bullpen_earned_runs_last3' AS DOUBLE PRECISION) AS away_bullpen_earned_runs_last3,
            CAST(f.feature_payload ->> 'home_bullpen_earned_runs_last3' AS DOUBLE PRECISION) AS home_bullpen_earned_runs_last3,
            CAST(f.feature_payload ->> 'away_bullpen_hits_allowed_last3' AS DOUBLE PRECISION) AS away_bullpen_hits_allowed_last3,
            CAST(f.feature_payload ->> 'home_bullpen_hits_allowed_last3' AS DOUBLE PRECISION) AS home_bullpen_hits_allowed_last3,
            CAST(f.feature_payload ->> 'away_bullpen_era_last3' AS DOUBLE PRECISION) AS away_bullpen_era_last3,
            CAST(f.feature_payload ->> 'home_bullpen_era_last3' AS DOUBLE PRECISION) AS home_bullpen_era_last3,
            CAST(f.feature_payload ->> 'away_lineup_top5_xwoba' AS DOUBLE PRECISION) AS away_lineup_top5_xwoba,
            CAST(f.feature_payload ->> 'home_lineup_top5_xwoba' AS DOUBLE PRECISION) AS home_lineup_top5_xwoba,
            CAST(f.feature_payload ->> 'away_lineup_k_pct' AS DOUBLE PRECISION) AS away_lineup_k_pct,
            CAST(f.feature_payload ->> 'home_lineup_k_pct' AS DOUBLE PRECISION) AS home_lineup_k_pct,
            CAST(f.feature_payload ->> 'away_k_pct_blended' AS DOUBLE PRECISION) AS away_k_pct_blended,
            CAST(f.feature_payload ->> 'home_k_pct_blended' AS DOUBLE PRECISION) AS home_k_pct_blended,
            CAST(f.feature_payload ->> 'venue_run_factor' AS DOUBLE PRECISION) AS venue_run_factor,
            CAST(f.feature_payload ->> 'venue_hr_factor' AS DOUBLE PRECISION) AS venue_hr_factor,
            CAST(f.feature_payload ->> 'temperature_f' AS DOUBLE PRECISION) AS temperature_f,
            CAST(f.feature_payload ->> 'wind_speed_mph' AS DOUBLE PRECISION) AS wind_speed_mph,
            CAST(f.feature_payload ->> 'wind_direction_deg' AS DOUBLE PRECISION) AS wind_direction_deg,
            CAST(f.feature_payload ->> 'humidity_pct' AS DOUBLE PRECISION) AS humidity_pct,
            CAST(f.feature_payload ->> 'line_movement' AS DOUBLE PRECISION) AS line_movement,
            CAST(f.feature_payload ->> 'starter_certainty_score' AS DOUBLE PRECISION) AS starter_certainty_score,
            CAST(f.feature_payload ->> 'lineup_certainty_score' AS DOUBLE PRECISION) AS lineup_certainty_score,
            CAST(f.feature_payload ->> 'weather_freshness_score' AS DOUBLE PRECISION) AS weather_freshness_score,
            CAST(f.feature_payload ->> 'market_freshness_score' AS DOUBLE PRECISION) AS market_freshness_score,
            CAST(f.feature_payload ->> 'bullpen_completeness_score' AS DOUBLE PRECISION) AS bullpen_completeness_score,
            CAST(f.feature_payload ->> 'missing_fallback_count' AS INTEGER) AS missing_fallback_count,
            f.feature_payload ->> 'board_state' AS board_state
        FROM games g
        LEFT JOIN dim_venues v ON v.venue_id = g.venue_id
        LEFT JOIN ranked_predictions p ON p.game_id = g.game_id AND p.row_rank = 1
        LEFT JOIN ranked_features f ON f.game_id = g.game_id AND f.row_rank = 1
        WHERE g.game_date = :target_date
        ORDER BY {game_start_order}, g.away_team, g.home_team
        """,
        {"target_date": target_date},
    )
    game_records = _frame_records(games_frame)
    if not game_records:
        return []

    starters_frame = _safe_frame(
        """
        WITH ranked_starters AS (
            SELECT
                s.game_id,
                s.team,
                s.pitcher_id,
                s.is_probable,
                s.days_rest,
                s.xwoba_against,
                s.csw_pct,
                s.avg_fb_velo,
                s.whiff_pct,
                dp.full_name AS pitcher_name,
                dp.throws,
                ROW_NUMBER() OVER (
                    PARTITION BY s.game_id, s.team
                    ORDER BY COALESCE(s.is_probable, FALSE) DESC, s.pitcher_id
                ) AS row_rank
            FROM pitcher_starts s
            LEFT JOIN dim_players dp ON dp.player_id = s.pitcher_id
            WHERE s.game_date = :target_date
        )
        SELECT
            game_id,
            team,
            pitcher_id,
            COALESCE(pitcher_name, CAST(pitcher_id AS TEXT)) AS pitcher_name,
            throws,
            is_probable,
            days_rest,
            xwoba_against,
            csw_pct,
            avg_fb_velo,
            whiff_pct
        FROM ranked_starters
        WHERE row_rank = 1
        """,
        {"target_date": target_date},
    )
    starter_records = _frame_records(starters_frame)

    hit_frame = _safe_frame(
        f"""
        WITH ranked_predictions AS (
            SELECT
                p.*,
                ROW_NUMBER() OVER (
                    PARTITION BY p.game_id, p.player_id
                    ORDER BY p.prediction_ts DESC
                ) AS row_rank
            FROM predictions_player_hits p
            WHERE p.game_date = :target_date
        ),
        ranked_features AS (
            SELECT
                f.*,
                ROW_NUMBER() OVER (
                    PARTITION BY f.game_id, f.player_id
                    ORDER BY f.prediction_ts DESC
                ) AS row_rank
            FROM player_features_hits f
            WHERE f.game_date = :target_date
        ),
        selected_players AS (
            SELECT DISTINCT p.player_id
            FROM ranked_predictions p
            WHERE p.row_rank = 1
              AND p.predicted_hit_probability >= :min_probability
        ),
        recent_batting AS (
            SELECT
                recent.player_id,
                COUNT(*) AS games_last7,
                SUM(recent.hits) AS hits_last7,
                SUM(recent.at_bats) AS at_bats_last7,
                {recent_batting_avg_expr} AS batting_avg_last7
            FROM (
                SELECT
                    b.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY b.player_id
                        ORDER BY b.game_date DESC, b.game_id DESC
                    ) AS row_rank
                FROM player_game_batting b
                INNER JOIN selected_players sp ON sp.player_id = b.player_id
                WHERE b.game_date < :target_date
            ) recent
            WHERE recent.row_rank <= 7
            GROUP BY recent.player_id
        ),
        season_batting AS (
            SELECT
                b.player_id,
                COUNT(*) AS games_season,
                SUM(b.hits) AS season_hits,
                SUM(b.at_bats) AS season_at_bats,
                                {season_batting_avg_expr} AS batting_avg_season
            FROM player_game_batting b
            INNER JOIN selected_players sp ON sp.player_id = b.player_id
            WHERE b.game_date < :target_date
                            AND {game_year_expr} = {target_year_expr}
            GROUP BY b.player_id
        ),
        joined AS (
            SELECT
                p.game_id,
                p.game_date,
                p.player_id,
                                COALESCE({hit_player_name_expr}, dp.full_name, CAST(p.player_id AS TEXT)) AS player_name,
                dp.bats,
                dp.position,
                COALESCE(f.team, p.team) AS team,
                COALESCE(
                    f.opponent,
                    CASE
                        WHEN g.home_team = COALESCE(f.team, p.team) THEN g.away_team
                        WHEN g.away_team = COALESCE(f.team, p.team) THEN g.home_team
                        ELSE NULL
                    END,
                    'TBD'
                ) AS opponent,
                {hit_lineup_slot_expr} AS lineup_slot,
                {hit_confirmed_lineup_expr} AS is_confirmed_lineup,
                {hit_projected_pa_expr} AS projected_plate_appearances,
                {hit_streak_len_expr} AS streak_len_capped,
                {hit_streak_len_full_expr} AS streak_len,
                p.predicted_hit_probability,
                p.fair_price,
                p.market_price,
                p.edge,
                {hit_rate_blended_expr} AS hit_rate_blended,
                {xwoba_14_expr} AS xwoba_14,
                {opp_starter_xwoba_expr} AS opposing_starter_xwoba,
                {opp_starter_csw_expr} AS opposing_starter_csw,
                {team_run_environment_expr} AS team_run_environment,
                {park_hr_factor_expr} AS park_hr_factor,
                recent_batting.games_last7,
                recent_batting.hits_last7,
                recent_batting.at_bats_last7,
                recent_batting.batting_avg_last7,
                season_batting.games_season,
                season_batting.season_hits,
                season_batting.season_at_bats,
                season_batting.batting_avg_season,
                actual.plate_appearances AS actual_plate_appearances,
                actual.at_bats AS actual_at_bats,
                actual.hits AS actual_hits,
                actual.runs AS actual_runs,
                actual.rbi AS actual_rbi,
                actual.walks AS actual_walks,
                actual.home_runs AS actual_home_runs,
                actual.stolen_bases AS actual_stolen_bases,
                (
                    COALESCE(actual.singles, 0)
                    + 2 * COALESCE(actual.doubles, 0)
                    + 3 * COALESCE(actual.triples, 0)
                    + 4 * COALESCE(actual.home_runs, 0)
                ) AS actual_total_bases
            FROM ranked_predictions p
            LEFT JOIN ranked_features f
                ON f.game_id = p.game_id
               AND f.player_id = p.player_id
               AND f.row_rank = 1
            LEFT JOIN games g ON g.game_id = p.game_id
            LEFT JOIN dim_players dp ON dp.player_id = p.player_id
            LEFT JOIN recent_batting ON recent_batting.player_id = p.player_id
            LEFT JOIN season_batting ON season_batting.player_id = p.player_id
            LEFT JOIN player_game_batting actual
                ON actual.game_id = p.game_id
               AND actual.player_id = p.player_id
            WHERE p.row_rank = 1
              AND p.predicted_hit_probability >= :min_probability
        ),
        limited AS (
            SELECT
                j.*,
                ROW_NUMBER() OVER (
                    PARTITION BY j.game_id, j.team
                    ORDER BY j.predicted_hit_probability DESC,
                             COALESCE(j.is_confirmed_lineup, FALSE) DESC,
                             {projected_pa_order},
                             {lineup_slot_order},
                             j.player_name
                ) AS team_rank
            FROM joined j
        )
        SELECT
            game_id,
            game_date,
            player_id,
            player_name,
            bats,
            position,
            team,
            opponent,
            lineup_slot,
            is_confirmed_lineup,
            projected_plate_appearances,
            streak_len_capped,
            streak_len,
            predicted_hit_probability,
            fair_price,
            market_price,
            edge,
            hit_rate_blended,
            xwoba_14,
            opposing_starter_xwoba,
            opposing_starter_csw,
            team_run_environment,
            park_hr_factor,
            games_last7,
            hits_last7,
            at_bats_last7,
            batting_avg_last7,
            games_season,
            season_hits,
            season_at_bats,
            batting_avg_season,
            actual_plate_appearances,
            actual_at_bats,
            actual_hits,
            actual_runs,
            actual_rbi,
            actual_walks,
            actual_home_runs,
            actual_stolen_bases,
            actual_total_bases
        FROM limited
        WHERE team_rank <= :hit_limit_per_team
        ORDER BY game_id, team, predicted_hit_probability DESC, {final_lineup_slot_order}, player_name
        """,
        {
            "target_date": target_date,
            "min_probability": min_probability,
            "hit_limit_per_team": hit_limit_per_team,
        },
    )
    if confirmed_only and not hit_frame.empty and "is_confirmed_lineup" in hit_frame.columns:
        hit_frame = hit_frame[hit_frame["is_confirmed_lineup"] == True].copy()
    hit_records = _frame_records(hit_frame)
    hit_records = _annotate_lineup_confidence(hit_records, _fetch_lineup_snapshot_keys(target_date))
    if not include_inferred:
        hit_records = [record for record in hit_records if not record.get("is_inferred_lineup")]
    hit_split_map = _fetch_hitter_pitch_hand_splits(
        target_date,
        [int(hit["player_id"]) for hit in hit_records if hit.get("player_id") is not None],
    )
    lineup_handedness_by_game = _fetch_lineup_handedness_by_game(target_date)
    pitcher_k_market_map = _fetch_pitcher_strikeout_market_map(target_date)
    pitcher_k_prediction_map = _fetch_pitcher_strikeout_prediction_map(target_date)
    first5_totals_map = _fetch_first5_totals_map(target_date)

    games_by_id: dict[int, dict[str, Any]] = {}
    for record in game_records:
        game_id = int(record["game_id"])
        is_final = _is_final_game_status(record.get("status"))
        games_by_id[game_id] = {
            "game_id": game_id,
            "game_date": record["game_date"],
            "status": record["status"],
            "away_team": record["away_team"],
            "home_team": record["home_team"],
            "game_start_ts": record["game_start_ts"],
            "actual_result": {
                "away_runs": record["away_runs"] if is_final else None,
                "home_runs": record["home_runs"] if is_final else None,
                "total_runs": record["total_runs"] if is_final else None,
                "home_win": record["home_win"] if is_final else None,
                "is_final": is_final,
            },
            "venue": {
                "name": record["venue_name"],
                "city": record["venue_city"],
                "state": record["venue_state"],
                "roof_type": record["roof_type"],
            },
            "weather": {
                "temperature_f": record["temperature_f"],
                "wind_speed_mph": record["wind_speed_mph"],
                "wind_direction_deg": record["wind_direction_deg"],
                "humidity_pct": record["humidity_pct"],
            },
            "totals": {
                "model_name": record["model_name"],
                "model_version": record["model_version"],
                "prediction_ts": record["prediction_ts"],
                "predicted_total_runs": record["predicted_total_runs"],
                "market_total": record["market_total"],
                "over_probability": record["over_probability"],
                "under_probability": record["under_probability"],
                "edge": record["edge"],
                "confidence_level": record.get("confidence_level"),
                "suppress_reason": record.get("suppress_reason"),
                "lane_status": record.get("lane_status", "research_only"),
                "away_expected_runs": record["away_expected_runs"],
                "home_expected_runs": record["home_expected_runs"],
                "away_bullpen_pitches_last3": record["away_bullpen_pitches_last3"],
                "home_bullpen_pitches_last3": record["home_bullpen_pitches_last3"],
                "away_bullpen_innings_last3": record["away_bullpen_innings_last3"],
                "home_bullpen_innings_last3": record["home_bullpen_innings_last3"],
                "away_bullpen_b2b": record["away_bullpen_b2b"],
                "home_bullpen_b2b": record["home_bullpen_b2b"],
                "away_bullpen_runs_allowed_last3": record["away_bullpen_runs_allowed_last3"],
                "home_bullpen_runs_allowed_last3": record["home_bullpen_runs_allowed_last3"],
                "away_bullpen_earned_runs_last3": record["away_bullpen_earned_runs_last3"],
                "home_bullpen_earned_runs_last3": record["home_bullpen_earned_runs_last3"],
                "away_bullpen_hits_allowed_last3": record["away_bullpen_hits_allowed_last3"],
                "home_bullpen_hits_allowed_last3": record["home_bullpen_hits_allowed_last3"],
                "away_bullpen_era_last3": record["away_bullpen_era_last3"],
                "home_bullpen_era_last3": record["home_bullpen_era_last3"],
                "away_lineup_top5_xwoba": record["away_lineup_top5_xwoba"],
                "home_lineup_top5_xwoba": record["home_lineup_top5_xwoba"],
                "away_lineup_k_pct": record["away_lineup_k_pct"],
                "home_lineup_k_pct": record["home_lineup_k_pct"],
                "away_k_pct_blended": record["away_k_pct_blended"],
                "home_k_pct_blended": record["home_k_pct_blended"],
                "venue_run_factor": record["venue_run_factor"],
                "venue_hr_factor": record["venue_hr_factor"],
                "line_movement": record["line_movement"],
            },
            "certainty": {
                "starter_certainty": record["starter_certainty_score"],
                "lineup_certainty": record["lineup_certainty_score"],
                "weather_freshness": record["weather_freshness_score"],
                "market_freshness": record["market_freshness_score"],
                "bullpen_completeness": record["bullpen_completeness_score"],
                "missing_fallback_count": record["missing_fallback_count"],
                "board_state": record["board_state"],
            },
            "starters": {
                "away": None,
                "home": None,
            },
            "lineup_handedness": lineup_handedness_by_game.get(game_id, {}),
            "hit_targets": {
                record["away_team"]: [],
                record["home_team"]: [],
            },
            "first5_totals": first5_totals_map.get(game_id) or {"supported": False},
        }

    for starter in starter_records:
        game = games_by_id.get(int(starter["game_id"]))
        if not game:
            continue
        side = None
        if starter["team"] == game["away_team"]:
            side = "away"
        elif starter["team"] == game["home_team"]:
            side = "home"
        if side is None:
            continue
        game["starters"][side] = {
            "team": starter["team"],
            "pitcher_id": starter["pitcher_id"],
            "pitcher_name": starter["pitcher_name"],
            "throws": starter["throws"],
            "is_probable": starter["is_probable"],
            "days_rest": starter["days_rest"],
            "xwoba_against": starter["xwoba_against"],
            "csw_pct": starter["csw_pct"],
            "avg_fb_velo": starter["avg_fb_velo"],
            "whiff_pct": starter["whiff_pct"],
            "recent_form": _fetch_starter_recent_form(starter["pitcher_id"], target_date),
        }

    for game in games_by_id.values():
        away_recent = (game["starters"]["away"] or {}).get("recent_form")
        home_recent = (game["starters"]["home"] or {}).get("recent_form")
        if game["starters"]["away"]:
            away_key = (game["game_id"], int(game["starters"]["away"]["pitcher_id"])) if game["starters"]["away"].get("pitcher_id") is not None else None
            away_market = pitcher_k_market_map.get(
                away_key
            ) if away_key is not None else None
            modeled_projection = pitcher_k_prediction_map.get(away_key) if away_key is not None else None
            game["starters"]["away"]["k_projection"] = _merge_strikeout_market_context(
                modeled_projection,
                away_market,
            ) or _estimate_starter_strikeout_projection(
                game["starters"]["away"],
                away_recent,
                game["totals"].get("home_lineup_k_pct"),
                game["totals"].get("home_k_pct_blended"),
                game.get("lineup_handedness", {}).get(game["home_team"]),
                away_market,
            )
        if game["starters"]["home"]:
            home_key = (game["game_id"], int(game["starters"]["home"]["pitcher_id"])) if game["starters"]["home"].get("pitcher_id") is not None else None
            home_market = pitcher_k_market_map.get(
                home_key
            ) if home_key is not None else None
            modeled_projection = pitcher_k_prediction_map.get(home_key) if home_key is not None else None
            game["starters"]["home"]["k_projection"] = _merge_strikeout_market_context(
                modeled_projection,
                home_market,
            ) or _estimate_starter_strikeout_projection(
                game["starters"]["home"],
                home_recent,
                game["totals"].get("away_lineup_k_pct"),
                game["totals"].get("away_k_pct_blended"),
                game.get("lineup_handedness", {}).get(game["away_team"]),
                home_market,
            )

    for hit in hit_records:
        game = games_by_id.get(int(hit["game_id"]))
        if not game:
            continue
        team = hit["team"]
        if team not in game["hit_targets"]:
            game["hit_targets"][team] = []
        opposing_starter = game["starters"]["home"] if team == game["away_team"] else game["starters"]["away"]
        actual_meta = _build_hit_actual_meta(hit["actual_hits"], bool(game["actual_result"]["is_final"]))
        game["hit_targets"][team].append(
            _attach_hitter_matchup_context(
                {
                "player_id": hit["player_id"],
                "player_name": hit["player_name"],
                "bats": hit["bats"],
                "position": hit["position"],
                "team": team,
                "opponent": hit["opponent"],
                "lineup_slot": hit["lineup_slot"],
                "is_confirmed_lineup": hit["is_confirmed_lineup"],
                "projected_plate_appearances": hit["projected_plate_appearances"],
                "streak_len_capped": hit["streak_len_capped"],
                "streak_len": hit.get("streak_len") or hit["streak_len_capped"],
                "predicted_hit_probability": hit["predicted_hit_probability"],
                "fair_price": hit["fair_price"],
                "market_price": hit["market_price"],
                "edge": hit["edge"],
                "hit_rate_blended": hit["hit_rate_blended"],
                "xwoba_14": hit["xwoba_14"],
                "opposing_starter_xwoba": hit["opposing_starter_xwoba"],
                "opposing_starter_csw": hit["opposing_starter_csw"],
                "team_run_environment": hit["team_run_environment"],
                "park_hr_factor": hit["park_hr_factor"],
                "games_last7": hit["games_last7"],
                "hits_last7": hit["hits_last7"],
                "at_bats_last7": hit["at_bats_last7"],
                "batting_avg_last7": hit["batting_avg_last7"],
                "games_season": hit["games_season"],
                "season_hits": hit["season_hits"],
                "season_at_bats": hit["season_at_bats"],
                "batting_avg_season": hit["batting_avg_season"],
                "actual_plate_appearances": hit["actual_plate_appearances"],
                "actual_at_bats": hit["actual_at_bats"],
                "actual_hits": hit["actual_hits"],
                "actual_runs": hit["actual_runs"],
                "actual_rbi": hit["actual_rbi"],
                "actual_walks": hit["actual_walks"],
                "actual_home_runs": hit["actual_home_runs"],
                "actual_stolen_bases": hit["actual_stolen_bases"],
                "actual_total_bases": hit["actual_total_bases"],
                "opposing_pitcher_name": opposing_starter["pitcher_name"] if opposing_starter else None,
                "opposing_pitcher_throws": opposing_starter["throws"] if opposing_starter else None,
                **actual_meta,
                },
                opposing_starter["throws"] if opposing_starter else None,
                hit_split_map,
            )
        )

    # --- Freshness / confidence badges ---
    ingest_freshness = _fetch_ingest_freshness(target_date)
    market_freezes = _fetch_market_freeze_map(target_date)
    readiness_map = _fetch_game_readiness_map(target_date)
    for game in games_by_id.values():
        total_freeze = market_freezes.get((int(game["game_id"]), "total"), {})
        game["totals"]["market_locked"] = bool(total_freeze)
        game["totals"]["locked_sportsbook"] = total_freeze.get("frozen_sportsbook")
        game["totals"]["locked_snapshot_ts"] = total_freeze.get("frozen_snapshot_ts")
        game["totals"]["locked_line_value"] = total_freeze.get("frozen_line_value")
        game["data_quality"] = _compute_data_quality_badge(game, ingest_freshness, readiness_map)

    return [games_by_id[game_id] for game_id in games_by_id]


def _fetch_ingest_freshness(target_date: date) -> dict[str, str | None]:
    """Return the most recent ingested_at per source_name for target_date."""
    if not _table_exists("raw_ingest_events"):
        return {}
    frame = _safe_frame(
        """
        SELECT source_name, MAX(ingested_at) AS latest_ingested_at
        FROM raw_ingest_events
        WHERE target_date = :target_date
        GROUP BY source_name
        """,
        {"target_date": target_date.isoformat()},
    )
    if frame.empty:
        return {}
    return {
        str(row["source_name"]): str(row["latest_ingested_at"])
        for row in _frame_records(frame)
    }


def _fetch_market_freeze_map(target_date: date) -> dict[tuple[int, str], dict[str, Any]]:
    """Return {(game_id, market_type): freeze_row} for the target date."""
    if not _table_exists("market_selection_freezes"):
        return {}
    frame = _safe_frame(
        """
        SELECT msf.game_id, msf.market_type, msf.frozen_sportsbook,
               msf.frozen_line_value, msf.frozen_snapshot_ts, msf.reason
        FROM market_selection_freezes msf
        JOIN games g ON g.game_id = msf.game_id
        WHERE g.game_date = :target_date
        """,
        {"target_date": target_date},
    )
    if frame.empty:
        return {}
    return {
        (int(row["game_id"]), str(row["market_type"])): row
        for row in _frame_records(frame)
    }


def _fetch_game_readiness_map(target_date: date) -> dict[int, dict[str, Any]]:
    """Return {game_id: readiness_row} from the validator's game_readiness table."""
    if not _table_exists("game_readiness"):
        return {}
    target_date_str = str(target_date)
    frame = _safe_frame(
        "SELECT * FROM game_readiness WHERE game_date = :target_date",
        {"target_date": target_date_str},
    )
    if frame.empty:
        return {}
    return {int(row["game_id"]): row for row in _frame_records(frame)}


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _totals_lean_direction(totals: dict[str, Any]) -> str | None:
    over_probability = _coerce_float(totals.get("over_probability"))
    under_probability = _coerce_float(totals.get("under_probability"))
    if over_probability is None or under_probability is None:
        return None
    return "over" if over_probability >= under_probability else "under"


def _build_totals_rationale(detail: dict[str, Any]) -> dict[str, Any]:
    totals = detail.get("totals") or {}
    weather = detail.get("weather") or {}
    starters = detail.get("starters") or {}
    direction = _totals_lean_direction(totals)
    confidence = max(
        value
        for value in [
            _coerce_float(totals.get("over_probability")),
            _coerce_float(totals.get("under_probability")),
        ]
        if value is not None
    ) if any(
        value is not None
        for value in [
            _coerce_float(totals.get("over_probability")),
            _coerce_float(totals.get("under_probability")),
        ]
    ) else None
    predicted_total = _coerce_float(totals.get("predicted_total_runs"))
    market_total = _coerce_float(totals.get("market_total"))
    edge_delta = None if predicted_total is None or market_total is None else predicted_total - market_total
    away_expected = _coerce_float(totals.get("away_expected_runs"))
    home_expected = _coerce_float(totals.get("home_expected_runs"))
    venue_run_factor = _coerce_float(totals.get("venue_run_factor"))
    temperature_f = _coerce_float(weather.get("temperature_f"))
    line_movement = _coerce_float(totals.get("line_movement"))
    away_top5 = _coerce_float(totals.get("away_lineup_top5_xwoba"))
    home_top5 = _coerce_float(totals.get("home_lineup_top5_xwoba"))
    away_lineup_k = _coerce_float(totals.get("away_lineup_k_pct"))
    home_lineup_k = _coerce_float(totals.get("home_lineup_k_pct"))
    bullpen_burden = sum(
        value or 0.0
        for value in [
            _coerce_float(totals.get("away_bullpen_pitches_last3")),
            _coerce_float(totals.get("home_bullpen_pitches_last3")),
        ]
    )
    b2b_bullpens = sum(
        int(value or 0)
        for value in [
            _coerce_float(totals.get("away_bullpen_b2b")),
            _coerce_float(totals.get("home_bullpen_b2b")),
        ]
    )

    signals: list[str] = []
    risks: list[str] = []

    if edge_delta is not None:
        signals.append(f"Model total sits {abs(edge_delta):.1f} runs {direction or 'away from'} the market.")
    if direction == "over":
        if venue_run_factor is not None and venue_run_factor >= 1.03:
            signals.append(f"Park factor {venue_run_factor:.2f} points to a friendlier run environment.")
        if temperature_f is not None and temperature_f >= 78:
            signals.append(f"Warm {temperature_f:.0f}° weather boosts carry and run scoring.")
        if max(value or 0.0 for value in [away_top5, home_top5]) >= 0.34:
            signals.append("At least one top-of-order group brings premium xwOBA form.")
        if bullpen_burden >= 165 or b2b_bullpens >= 1:
            signals.append("Recent bullpen workload suggests thinner late-inning coverage.")
        if line_movement is not None and line_movement > 0.15:
            signals.append("The market has already drifted upward toward the over.")
    elif direction == "under":
        if venue_run_factor is not None and venue_run_factor <= 0.97:
            signals.append(f"Park factor {venue_run_factor:.2f} suppresses baseline scoring.")
        if temperature_f is not None and temperature_f <= 60:
            signals.append(f"Cooler {temperature_f:.0f}° weather leans run-suppressive.")
        if max(value or 0.0 for value in [away_lineup_k, home_lineup_k]) >= 0.24:
            signals.append("Strikeout-heavy lineup shape trims ball-in-play volume.")
        if max(value or 0.0 for value in [away_top5, home_top5]) <= 0.315:
            signals.append("Neither projected top-of-order group rates as especially dangerous.")
        if line_movement is not None and line_movement < -0.15:
            signals.append("The market has already drifted downward toward the under.")

    if not totals.get("market_locked"):
        risks.append("pregame line is not frozen yet")
    if edge_delta is not None and abs(edge_delta) < 0.35:
        risks.append("model edge over the market is thin")
    if temperature_f is None:
        risks.append("weather snapshot missing")
    if not starters.get("away") or not starters.get("home"):
        risks.append("starter mapping incomplete")
    if line_movement is not None and direction == "over" and line_movement < -0.2:
        risks.append("market moved against the over lean")
    if line_movement is not None and direction == "under" and line_movement > 0.2:
        risks.append("market moved against the under lean")

    headline = signals[0] if signals else (
        f"Core projection leans {direction}." if direction else "No strong totals rationale is available yet."
    )
    return {
        "direction": direction,
        "confidence": confidence,
        "headline": headline,
        "signals": signals[:4],
        "risk_flags": risks[:3],
    }


def _fetch_totals_outcome_review(game_id: int, target_date: date) -> dict[str, Any] | None:
    if not _table_exists("prediction_outcomes_daily"):
        return None
    frame = _safe_frame(
        """
        SELECT game_id, recommended_side, actual_side, graded, success, beat_market,
               probability, predicted_value, market_line, actual_value, absolute_error,
               forecast_temperature_f, observed_temperature_f, weather_delta_temperature_f,
               entry_market_sportsbook, closing_market_sportsbook, closing_market_same_sportsbook,
               closing_market_line, clv_line_delta, clv_side_value, beat_closing_line
        FROM prediction_outcomes_daily
        WHERE game_date = :target_date
          AND game_id = :game_id
          AND market = 'totals'
        ORDER BY prediction_ts DESC
        LIMIT 1
        """,
        {"target_date": str(target_date), "game_id": game_id},
    )
    records = _frame_records(frame)
    return records[0] if records else None


def _build_totals_review_block(detail: dict[str, Any], outcome: dict[str, Any] | None) -> dict[str, Any]:
    rationale = _build_totals_rationale(detail)
    actual = detail.get("actual_result") or {}
    totals = detail.get("totals") or {}
    market_total = _coerce_float(totals.get("market_total"))
    actual_total = _coerce_float(actual.get("total_runs"))
    grading = {
        "graded": False,
        "result": "pending" if not actual.get("is_final") else "missing",
        "summary": "No graded totals result is available yet.",
        "recommended_side": None,
        "actual_side": None,
        "model_error": None,
        "market_error": None,
        "beat_market": None,
        "clv_side_value": None,
        "beat_closing_line": None,
        "entry_market_sportsbook": None,
        "closing_market_sportsbook": None,
        "closing_market_same_sportsbook": None,
        "weather_shift": None,
    }
    if not outcome:
        return {"rationale": rationale, "grading": grading}

    recommended_side = outcome.get("recommended_side")
    actual_side = outcome.get("actual_side")
    result = _graded_pick_result(recommended_side, actual_side, bool(actual.get("is_final")))
    model_error = _coerce_float(outcome.get("absolute_error"))
    actual_total = actual_total if actual_total is not None else _coerce_float(outcome.get("actual_value"))
    market_error = None if actual_total is None or market_total is None else abs(actual_total - market_total)
    weather_delta = _coerce_float(outcome.get("weather_delta_temperature_f"))
    clv_side_value = _coerce_float(outcome.get("clv_side_value"))
    weather_shift = None
    if weather_delta is not None:
        weather_shift = f"Observed weather landed {weather_delta:+.1f}° versus forecast."
    total_label = "?" if actual_total is None else f"{actual_total:.0f}"
    market_label = "?" if market_total is None else f"{market_total:.1f}"
    if result == "won":
        summary = f"{str(recommended_side).title()} won against a final total of {total_label}."
    elif result == "lost":
        summary = f"{str(recommended_side).title()} lost; the game finished {str(actual_side).title()} at {total_label}."
    elif result == "push":
        summary = f"The market pushed at {market_label}; no decision on the totals pick."
    elif result == "pending":
        summary = "The game is not final yet, so the totals pick is still pending."
    else:
        summary = "No graded totals result is available yet."

    grading.update(
        {
            "graded": bool(outcome.get("graded")),
            "result": result,
            "summary": summary,
            "recommended_side": recommended_side,
            "actual_side": actual_side,
            "model_error": model_error,
            "market_error": market_error,
            "beat_market": outcome.get("beat_market"),
            "clv_side_value": clv_side_value,
            "beat_closing_line": outcome.get("beat_closing_line"),
            "entry_market_sportsbook": outcome.get("entry_market_sportsbook"),
            "closing_market_sportsbook": outcome.get("closing_market_sportsbook"),
            "closing_market_same_sportsbook": outcome.get("closing_market_same_sportsbook"),
            "weather_shift": weather_shift,
        }
    )
    return {"rationale": rationale, "grading": grading}


def _review_meta_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        with contextlib.suppress(json.JSONDecodeError):
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
    return {}


def _fetch_top_review_payload(target_date: date, market: str = "totals", limit: int = 6) -> dict[str, Any]:
    if not _table_exists("prediction_outcomes_daily"):
        return {"target_date": str(target_date), "market": market, "summary": {}, "misses": [], "best_calls": []}
    frame = _safe_frame(
        """
        SELECT game_date, game_id, market, recommended_side, actual_side, graded, success,
               beat_market, probability, predicted_value, market_line, actual_value,
               absolute_error, weather_delta_temperature_f, entry_market_sportsbook,
               closing_market_sportsbook, closing_market_same_sportsbook, closing_market_line,
               clv_line_delta, clv_side_value, beat_closing_line, meta_payload
        FROM prediction_outcomes_daily
        WHERE game_date = :target_date
          AND market = :market
          AND entity_type = 'game'
          AND graded = TRUE
        ORDER BY game_id
        """,
        {"target_date": str(target_date), "market": market},
    )
    rows: list[dict[str, Any]] = []
    for record in _frame_records(frame):
        meta = _review_meta_payload(record.get("meta_payload"))
        predicted_value = _coerce_float(record.get("predicted_value"))
        market_line = _coerce_float(record.get("market_line"))
        actual_value = _coerce_float(record.get("actual_value"))
        miss_severity = None if predicted_value is None or actual_value is None else abs(predicted_value - actual_value)
        edge_gap = None if predicted_value is None or market_line is None else abs(predicted_value - market_line)
        rows.append(
            {
                "game_id": int(record["game_id"]),
                "game_date": record.get("game_date"),
                "away_team": meta.get("away_team"),
                "home_team": meta.get("home_team"),
                "recommended_side": record.get("recommended_side"),
                "actual_side": record.get("actual_side"),
                "result": _graded_pick_result(record.get("recommended_side"), record.get("actual_side"), True),
                "success": record.get("success"),
                "beat_market": record.get("beat_market"),
                "probability": _coerce_float(record.get("probability")),
                "predicted_value": predicted_value,
                "market_line": market_line,
                "actual_value": actual_value,
                "absolute_error": _coerce_float(record.get("absolute_error")),
                "weather_delta_temperature_f": _coerce_float(record.get("weather_delta_temperature_f")),
                "entry_market_sportsbook": record.get("entry_market_sportsbook"),
                "closing_market_sportsbook": record.get("closing_market_sportsbook"),
                "closing_market_same_sportsbook": record.get("closing_market_same_sportsbook"),
                "closing_market_line": _coerce_float(record.get("closing_market_line")),
                "clv_line_delta": _coerce_float(record.get("clv_line_delta")),
                "clv_side_value": _coerce_float(record.get("clv_side_value")),
                "beat_closing_line": record.get("beat_closing_line"),
                "miss_severity": miss_severity,
                "edge_gap": edge_gap,
            }
        )

    misses = sorted(
        [row for row in rows if row.get("result") == "lost"],
        key=lambda row: (-(row.get("miss_severity") or 0.0), -(row.get("edge_gap") or 0.0)),
    )[:limit]
    best_calls = sorted(
        [row for row in rows if row.get("result") == "won"],
        key=lambda row: (
            0 if row.get("beat_market") else 1,
            -(row.get("edge_gap") or 0.0),
            row.get("miss_severity") or 999.0,
        ),
    )[:limit]
    summary = {
        "graded_games": len(rows),
        "wins": sum(1 for row in rows if row.get("result") == "won"),
        "losses": sum(1 for row in rows if row.get("result") == "lost"),
        "pushes": sum(1 for row in rows if row.get("result") == "push"),
    }
    return {
        "target_date": str(target_date),
        "market": market,
        "summary": summary,
        "misses": misses,
        "best_calls": best_calls,
    }


def _fetch_clv_review_payload(
    start_date: date,
    end_date: date,
    *,
    market: str = "totals",
    limit: int = 8,
) -> dict[str, Any]:
    if not _table_exists("prediction_outcomes_daily"):
        return {"start_date": str(start_date), "end_date": str(end_date), "market": market, "summary": {}, "best_clv": [], "worst_clv": []}
    frame = _safe_frame(
        """
        SELECT game_date, game_id, recommended_side, actual_side, success, beat_market,
               market_line, entry_market_sportsbook, closing_market_sportsbook,
               closing_market_same_sportsbook, closing_market_line, clv_line_delta, clv_side_value,
               beat_closing_line, meta_payload
        FROM prediction_outcomes_daily
        WHERE game_date BETWEEN :start_date AND :end_date
          AND market = :market
          AND entity_type = 'game'
          AND recommended_side IS NOT NULL
          AND clv_side_value IS NOT NULL
        ORDER BY game_date DESC, game_id
        """,
        {"start_date": str(start_date), "end_date": str(end_date), "market": market},
    )
    rows: list[dict[str, Any]] = []
    for record in _frame_records(frame):
        meta = _review_meta_payload(record.get("meta_payload"))
        rows.append(
            {
                "game_id": int(record["game_id"]),
                "game_date": record.get("game_date"),
                "away_team": meta.get("away_team"),
                "home_team": meta.get("home_team"),
                "recommended_side": record.get("recommended_side"),
                "actual_side": record.get("actual_side"),
                "success": record.get("success"),
                "beat_market": record.get("beat_market"),
                "market_line": _coerce_float(record.get("market_line")),
                "entry_market_sportsbook": record.get("entry_market_sportsbook"),
                "closing_market_sportsbook": record.get("closing_market_sportsbook"),
                "closing_market_same_sportsbook": record.get("closing_market_same_sportsbook"),
                "closing_market_line": _coerce_float(record.get("closing_market_line")),
                "clv_line_delta": _coerce_float(record.get("clv_line_delta")),
                "clv_side_value": _coerce_float(record.get("clv_side_value")),
                "beat_closing_line": record.get("beat_closing_line"),
            }
        )
    best_clv = sorted(rows, key=lambda row: (-(row.get("clv_side_value") or 0.0), row.get("game_date") or ""))[:limit]
    worst_clv = sorted(rows, key=lambda row: ((row.get("clv_side_value") or 0.0), row.get("game_date") or ""))[:limit]
    clv_values = [row["clv_side_value"] for row in rows if row.get("clv_side_value") is not None]
    summary = {
        "rows": len(rows),
        "avg_clv_side_value": None if not clv_values else float(sum(clv_values) / len(clv_values)),
        "positive_clv_rate": None if not rows else float(sum(1 for row in rows if row.get("beat_closing_line")) / len(rows)),
    }
    return {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "market": market,
        "summary": summary,
        "best_clv": best_clv,
        "worst_clv": worst_clv,
    }


def _compute_data_quality_badge(
    game: dict[str, Any],
    ingest_freshness: dict[str, str | None],
    readiness_map: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute a confidence badge and freshness timestamps for a game card."""
    readiness_map = readiness_map or {}
    game_id = game.get("game_id")
    readiness = readiness_map.get(game_id) if game_id else None

    # Use validator results if available, else fall back to board-level checks
    if readiness:
        badge = readiness.get("badge", "yellow")
        warnings_str = readiness.get("warnings") or ""
        has_away_starter = bool(readiness.get("has_away_starter"))
        has_home_starter = bool(readiness.get("has_home_starter"))
        has_market = bool(readiness.get("has_market"))
        has_weather = bool(readiness.get("has_weather"))
        has_confirmed_lineup = bool(readiness.get("has_away_lineup")) and bool(readiness.get("has_home_lineup"))
        checks_passed = readiness.get("checks_passed", 0)
        checks_total = readiness.get("checks_total", 0)
        badge_reason = warnings_str.replace(";", "; ") if warnings_str else "all checks passed"
        badge_label = {"green": "Ready", "yellow": "Partial data", "red": "Missing data"}.get(badge, "Unknown")
    else:
        has_away_starter = game["starters"].get("away") is not None
        has_home_starter = game["starters"].get("home") is not None
        has_market = game["totals"].get("market_total") is not None
        has_weather = game["weather"].get("temperature_f") is not None

        has_confirmed_lineup = False
        for team_hits in game.get("hit_targets", {}).values():
            for hitter in team_hits:
                if hitter.get("is_confirmed_lineup"):
                    has_confirmed_lineup = True

        if not has_away_starter or not has_home_starter or not has_market:
            badge = "red"
            badge_label = "Missing data"
            reasons = []
            if not has_away_starter:
                reasons.append("away starter missing")
            if not has_home_starter:
                reasons.append("home starter missing")
            if not has_market:
                reasons.append("no market line")
            badge_reason = "; ".join(reasons)
        elif not has_confirmed_lineup or not has_weather:
            badge = "yellow"
            reasons = []
            if not has_confirmed_lineup:
                reasons.append("projected lineups")
            if not has_weather:
                reasons.append("no weather data")
            badge_label = "Partial data"
            badge_reason = "; ".join(reasons)
        else:
            badge = "green"
            badge_label = "Ready"
            badge_reason = "confirmed lineups, fresh market, starters mapped"
        checks_passed = None
        checks_total = None

    has_prediction = game["totals"].get("predicted_total_runs") is not None

    # --- Certainty composite score (0-100) from feature builder signals ---
    cert = game.get("certainty") or {}
    cert_components = [
        cert.get("starter_certainty"),
        cert.get("lineup_certainty"),
        cert.get("weather_freshness"),
        cert.get("market_freshness"),
        cert.get("bullpen_completeness"),
    ]
    present = [v for v in cert_components if v is not None]
    certainty_pct = round(100 * sum(present) / len(present)) if present else None

    return {
        "badge": badge,
        "badge_label": badge_label,
        "badge_reason": badge_reason,
        "has_away_starter": has_away_starter,
        "has_home_starter": has_home_starter,
        "has_market": has_market,
        "has_prediction": has_prediction,
        "has_weather": has_weather,
        "has_confirmed_lineup": has_confirmed_lineup,
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "certainty_pct": certainty_pct,
        "board_state": cert.get("board_state"),
        "freshness": {
            "schedule": ingest_freshness.get("mlb_statsapi"),
            "weather": ingest_freshness.get("open_meteo"),
            "market": ingest_freshness.get("market_totals"),
            "lineup": ingest_freshness.get("lineup_csv"),
        },
    }


def _fetch_full_hit_review(target_date: date) -> dict[str, Any]:
    default = {
        "total_targets": 0,
        "confirmed_targets": 0,
        "market_backed_targets": 0,
        "graded_targets": 0,
        "landed_targets": 0,
        "missed_targets": 0,
        "pending_targets": 0,
        "no_boxscore_targets": 0,
        "landed_rate": None,
    }
    if not _table_exists("predictions_player_hits"):
        return default

    features_join = ""
    if _table_exists("player_features_hits"):
        features_join = """
        LEFT JOIN ranked_features f
            ON f.game_id = p.game_id
           AND f.player_id = p.player_id
           AND f.row_rank = 1
        """

    frame = _safe_frame(
        f"""
        WITH ranked_predictions AS (
            SELECT
                p.game_id,
                p.player_id,
                p.market_price,
                ROW_NUMBER() OVER (
                    PARTITION BY p.game_id, p.player_id
                    ORDER BY p.prediction_ts DESC
                ) AS row_rank
            FROM predictions_player_hits p
            WHERE p.game_date = :target_date
        )
        {',' if features_join else ''}
        {"""
        ranked_features AS (
            SELECT
                f.game_id,
                f.player_id,
                CAST(NULLIF(f.feature_payload ->> 'is_confirmed_lineup', '') AS BOOLEAN) AS is_confirmed_lineup,
                ROW_NUMBER() OVER (
                    PARTITION BY f.game_id, f.player_id
                    ORDER BY f.prediction_ts DESC
                ) AS row_rank
            FROM player_features_hits f
            WHERE f.game_date = :target_date
        )
        """ if features_join else ''}
        SELECT
            COUNT(*) AS total_targets,
            SUM(CASE WHEN COALESCE(f.is_confirmed_lineup, FALSE) THEN 1 ELSE 0 END) AS confirmed_targets,
            SUM(CASE WHEN p.market_price IS NOT NULL THEN 1 ELSE 0 END) AS market_backed_targets,
            SUM(CASE WHEN actual.hits IS NOT NULL THEN 1 ELSE 0 END) AS graded_targets,
            SUM(CASE WHEN actual.hits > 0 THEN 1 ELSE 0 END) AS landed_targets,
            SUM(CASE WHEN actual.hits = 0 THEN 1 ELSE 0 END) AS missed_targets,
            SUM(
                CASE
                    WHEN actual.hits IS NULL
                     AND NOT (
                        LOWER(COALESCE(g.status, '')) LIKE '%final%'
                        OR LOWER(COALESCE(g.status, '')) LIKE '%completed%'
                        OR LOWER(COALESCE(g.status, '')) LIKE '%game over%'
                        OR LOWER(COALESCE(g.status, '')) LIKE '%closed%'
                     ) THEN 1
                    ELSE 0
                END
            ) AS pending_targets,
            SUM(
                CASE
                    WHEN actual.hits IS NULL
                     AND (
                        LOWER(COALESCE(g.status, '')) LIKE '%final%'
                        OR LOWER(COALESCE(g.status, '')) LIKE '%completed%'
                        OR LOWER(COALESCE(g.status, '')) LIKE '%game over%'
                        OR LOWER(COALESCE(g.status, '')) LIKE '%closed%'
                     ) THEN 1
                    ELSE 0
                END
            ) AS no_boxscore_targets
        FROM ranked_predictions p
        {features_join}
        INNER JOIN games g
            ON g.game_id = p.game_id
           AND g.game_date = :target_date
        LEFT JOIN player_game_batting actual
            ON actual.game_id = p.game_id
           AND actual.player_id = p.player_id
        WHERE p.row_rank = 1
        """,
        {"target_date": target_date},
    )
    if frame.empty:
        return default

    record = _frame_records(frame)[0]
    graded_targets = int(record.get("graded_targets") or 0)
    landed_targets = int(record.get("landed_targets") or 0)
    return {
        **default,
        **record,
        "landed_rate": round(landed_targets / graded_targets, 4) if graded_targets else None,
    }


def _summarize_board_rows(games: list[dict[str, Any]], target_date: date) -> dict[str, Any]:
    totals_review = {
        "graded_games": 0,
        "wins": 0,
        "losses": 0,
        "pushes": 0,
        "win_rate": None,
        "avg_model_error": None,
        "avg_market_error": None,
        "avg_model_bias": None,
        "model_beats_market_games": 0,
        "model_beats_market_rate": None,
    }
    hit_review = {
        "displayed_targets": 0,
        "confirmed_targets": 0,
        "graded_targets": 0,
        "landed_targets": 0,
        "missed_targets": 0,
        "pending_targets": 0,
        "no_boxscore_targets": 0,
        "landed_rate": None,
    }
    summary = {
        "total_games": len(games),
        "market_games": 0,
        "final_games": 0,
        "pending_games": 0,
        "totals_review": totals_review,
        "hit_review": hit_review,
        "full_hit_review": _fetch_full_hit_review(target_date),
    }

    model_errors: list[float] = []
    market_errors: list[float] = []
    model_biases: list[float] = []
    comparable_games = 0

    for game in games:
        totals = game.get("totals") or {}
        actual = game.get("actual_result") or {}
        predicted_total = _to_float(totals.get("predicted_total_runs"))
        market_total = _to_float(totals.get("market_total"))
        actual_total = _to_float(actual.get("total_runs"))

        if market_total is not None:
            summary["market_games"] += 1

        if actual.get("is_final"):
            summary["final_games"] += 1
        else:
            summary["pending_games"] += 1

        if predicted_total is not None and actual_total is not None:
            model_errors.append(abs(predicted_total - actual_total))
            model_biases.append(predicted_total - actual_total)

        if market_total is not None and actual_total is not None:
            market_errors.append(abs(market_total - actual_total))

        if predicted_total is not None and market_total is not None and actual_total is not None:
            totals_review["graded_games"] += 1
            if actual_total > market_total:
                actual_side = "over"
            elif actual_total < market_total:
                actual_side = "under"
            else:
                actual_side = "push"

            if actual_side == "push":
                totals_review["pushes"] += 1
            else:
                lean_side = "over" if predicted_total >= market_total else "under"
                if lean_side == actual_side:
                    totals_review["wins"] += 1
                else:
                    totals_review["losses"] += 1

            comparable_games += 1
            if abs(predicted_total - actual_total) < abs(market_total - actual_total):
                totals_review["model_beats_market_games"] += 1

        for players in (game.get("hit_targets") or {}).values():
            for player in players:
                hit_review["displayed_targets"] += 1
                if player.get("is_confirmed_lineup"):
                    hit_review["confirmed_targets"] += 1

                actual_status = str(player.get("actual_status") or "pending")
                actual_hits = _to_float(player.get("actual_hits"))
                if actual_status == "pending":
                    hit_review["pending_targets"] += 1
                    continue
                if actual_status == "dnp":
                    hit_review["no_boxscore_targets"] += 1
                    continue

                hit_review["graded_targets"] += 1
                if actual_hits > 0:
                    hit_review["landed_targets"] += 1
                else:
                    hit_review["missed_targets"] += 1

    decisions = totals_review["wins"] + totals_review["losses"]
    if decisions:
        totals_review["win_rate"] = round(totals_review["wins"] / decisions, 4)
    if model_errors:
        totals_review["avg_model_error"] = round(sum(model_errors) / len(model_errors), 3)
    if market_errors:
        totals_review["avg_market_error"] = round(sum(market_errors) / len(market_errors), 3)
    if model_biases:
        totals_review["avg_model_bias"] = round(sum(model_biases) / len(model_biases), 3)
    if comparable_games:
        totals_review["model_beats_market_rate"] = round(
            totals_review["model_beats_market_games"] / comparable_games,
            4,
        )
    if hit_review["graded_targets"]:
        hit_review["landed_rate"] = round(
            hit_review["landed_targets"] / hit_review["graded_targets"],
            4,
        )

    return summary


def _fetch_team_recent_offense(team: str, target_date: date) -> dict[str, Any]:
    default = {
        "sample_games": 0,
        "runs_per_game": None,
        "hits_per_game": None,
        "xwoba": None,
        "iso": None,
        "bb_pct": None,
        "k_pct": None,
        "hard_hit_pct": None,
        "last_game_date": None,
    }
    if not _table_exists("team_offense_daily"):
        return default
    frame = _safe_frame(
        """
        WITH recent AS (
            SELECT *
            FROM team_offense_daily
            WHERE team = :team
              AND game_date < :target_date
            ORDER BY game_date DESC
            LIMIT 7
        )
        SELECT
            COUNT(*) AS sample_games,
            AVG(runs) AS runs_per_game,
            AVG(hits) AS hits_per_game,
            AVG(xwoba) AS xwoba,
            AVG(iso) AS iso,
            AVG(bb_pct) AS bb_pct,
            AVG(k_pct) AS k_pct,
            AVG(hard_hit_pct) AS hard_hit_pct,
            MAX(game_date) AS last_game_date
        FROM recent
        """,
        {"team": team, "target_date": target_date},
    )
    if frame.empty:
        return default
    record = _frame_records(frame)[0]
    return {**default, **record}


def _fetch_team_last_result(team: str, target_date: date) -> dict[str, Any] | None:
    if not _table_exists("games"):
        return None
    game_start_desc_order = _sql_order_nulls_last("game_start_ts", "DESC")
    frame = _safe_frame(
        f"""
        SELECT
            game_id,
            game_date,
            game_start_ts,
            status,
            away_team,
            home_team,
            away_runs,
            home_runs,
            total_runs,
            CASE
                WHEN home_team = :team THEN home_runs
                ELSE away_runs
            END AS team_runs,
            CASE
                WHEN home_team = :team THEN away_runs
                ELSE home_runs
            END AS opponent_runs,
            CASE
                WHEN home_team = :team THEN away_team
                ELSE home_team
            END AS opponent,
            CASE
                WHEN (home_team = :team AND COALESCE(home_runs, -1) > COALESCE(away_runs, -1))
                  OR (away_team = :team AND COALESCE(away_runs, -1) > COALESCE(home_runs, -1)) THEN 'W'
                WHEN home_runs IS NOT NULL AND away_runs IS NOT NULL THEN 'L'
                ELSE NULL
            END AS result
        FROM games
        WHERE game_date < :target_date
          AND (home_team = :team OR away_team = :team)
                    AND total_runs IS NOT NULL
                    AND (
                            home_win IS NOT NULL
                            OR LOWER(COALESCE(status, '')) LIKE '%final%'
                            OR LOWER(COALESCE(status, '')) LIKE '%completed%'
                            OR LOWER(COALESCE(status, '')) LIKE '%game over%'
                            OR LOWER(COALESCE(status, '')) LIKE '%closed%'
                    )
                ORDER BY game_date DESC, {game_start_desc_order}
        LIMIT 1
        """,
        {"team": team, "target_date": target_date},
    )
    if frame.empty:
        return None
    return _frame_records(frame)[0]


def _fetch_team_recent_totals_history(team: str, target_date: date, limit: int = 5) -> list[dict[str, Any]]:
    if not _table_exists("games"):
        return []

    market_join = ""
    market_select = "CAST(NULL AS DOUBLE PRECISION) AS market_total"
    if _table_exists("game_markets"):
        market_join = """
        LEFT JOIN (
            WITH ranked_market_books AS (
                SELECT
                    gm.game_id,
                    gm.line_value,
                    ROW_NUMBER() OVER (
                        PARTITION BY gm.game_id, gm.sportsbook, gm.market_type
                        ORDER BY gm.snapshot_ts DESC
                    ) AS sportsbook_rank
                FROM game_markets gm
                WHERE gm.market_type = 'total'
            )
            SELECT
                game_id,
                AVG(line_value) AS market_total
            FROM ranked_market_books
            WHERE sportsbook_rank = 1
              AND line_value IS NOT NULL
            GROUP BY game_id
        ) market ON market.game_id = recent.game_id
        """
        market_select = "market.market_total"

    game_start_desc_order = _sql_order_nulls_last("g.game_start_ts", "DESC")
    recent_game_start_desc_order = _sql_order_nulls_last("recent.game_start_ts", "DESC")
    frame = _safe_frame(
        f"""
        WITH recent AS (
            SELECT
                g.game_id,
                g.game_date,
                g.game_start_ts,
                g.status,
                g.away_team,
                g.home_team,
                g.away_runs,
                g.home_runs,
                g.total_runs,
                CASE
                    WHEN g.home_team = :team THEN g.home_runs
                    ELSE g.away_runs
                END AS team_runs,
                CASE
                    WHEN g.home_team = :team THEN g.away_runs
                    ELSE g.home_runs
                END AS opponent_runs,
                CASE
                    WHEN g.home_team = :team THEN g.away_team
                    ELSE g.home_team
                END AS opponent,
                CASE
                    WHEN g.home_team = :team THEN 'home'
                    ELSE 'away'
                END AS venue_side
            FROM games g
            WHERE g.game_date < :target_date
              AND (g.home_team = :team OR g.away_team = :team)
              AND g.total_runs IS NOT NULL
              AND (
                  LOWER(COALESCE(g.status, '')) LIKE '%final%'
                    OR LOWER(COALESCE(g.status, '')) LIKE '%completed%'
                    OR LOWER(COALESCE(g.status, '')) LIKE '%game over%'
                    OR LOWER(COALESCE(g.status, '')) LIKE '%closed%'
              )
                        ORDER BY g.game_date DESC, {game_start_desc_order}
            LIMIT :limit
        )
        SELECT
            recent.game_id,
            recent.game_date,
            recent.game_start_ts,
            recent.status,
            recent.away_team,
            recent.home_team,
            recent.team_runs,
            recent.opponent_runs,
            recent.total_runs,
            recent.opponent,
            recent.venue_side,
            {market_select}
        FROM recent
        {market_join}
        ORDER BY recent.game_date DESC, {recent_game_start_desc_order}
        """,
        {"team": team, "target_date": target_date, "limit": limit},
    )
    rows = _frame_records(frame)
    for row in rows:
        actual_side = _actual_side(row.get("total_runs"), row.get("market_total"))
        row["actual_side"] = actual_side
        row["market_backed"] = _to_float(row.get("market_total")) is not None
        total_runs = _to_float(row.get("total_runs"))
        market_total = _to_float(row.get("market_total"))
        row["delta_vs_market"] = None if total_runs is None or market_total is None else round(total_runs - market_total, 2)
    return rows


def _fetch_first5_totals_map(target_date: date) -> dict[int, dict[str, Any]]:
    prediction_created_order = _sql_order_nulls_last("p.created_at", "DESC")
    feature_cutoff_order = _sql_order_nulls_last("f.feature_cutoff_ts", "DESC")
    market_line_order = _sql_order_nulls_last("gm.line_value", "DESC")
    frame = _safe_frame(
        f"""
        WITH ranked_predictions AS (
            SELECT
                p.*,
                ROW_NUMBER() OVER (
                    PARTITION BY p.game_id
                    ORDER BY p.prediction_ts DESC, {prediction_created_order}
                ) AS row_rank
            FROM predictions_first5_totals p
            WHERE p.game_date = :target_date
        ),
        ranked_features AS (
            SELECT
                f.*,
                ROW_NUMBER() OVER (
                    PARTITION BY f.game_id
                    ORDER BY f.prediction_ts DESC, {feature_cutoff_order}
                ) AS row_rank
            FROM game_features_first5_totals f
            WHERE f.game_date = :target_date
        ),
        ranked_markets AS (
            SELECT
                gm.game_id,
                gm.line_value,
                gm.snapshot_ts,
                ROW_NUMBER() OVER (
                    PARTITION BY gm.game_id, COALESCE(gm.sportsbook, ''), gm.market_type
                    ORDER BY gm.snapshot_ts DESC, {market_line_order}
                ) AS sportsbook_rank
            FROM game_markets gm
            WHERE gm.game_date = :target_date
              AND gm.market_type = 'first_five_total'
        ),
        market AS (
            SELECT
                game_id,
                AVG(line_value) AS market_total
            FROM ranked_markets
            WHERE sportsbook_rank = 1
              AND line_value IS NOT NULL
            GROUP BY game_id
        )
        SELECT
            g.game_id,
            g.away_runs_first5,
            g.home_runs_first5,
            g.total_runs_first5,
            p.model_name,
            p.model_version,
            p.prediction_ts,
            p.predicted_total_runs,
            COALESCE(p.market_total, market.market_total) AS market_total,
            p.over_probability,
            p.under_probability,
            p.edge,
            CAST(f.feature_payload ->> 'away_runs_rate_blended' AS DOUBLE PRECISION) AS away_expected_runs,
            CAST(f.feature_payload ->> 'home_runs_rate_blended' AS DOUBLE PRECISION) AS home_expected_runs
        FROM games g
        LEFT JOIN ranked_predictions p ON p.game_id = g.game_id AND p.row_rank = 1
        LEFT JOIN ranked_features f ON f.game_id = g.game_id AND f.row_rank = 1
        LEFT JOIN market ON market.game_id = g.game_id
        WHERE g.game_date = :target_date
        """,
        {"target_date": target_date},
    )
    payload_by_game: dict[int, dict[str, Any]] = {}
    for row in _frame_records(frame):
        game_id = row.get("game_id")
        if game_id is None:
            continue
        predicted_total = _to_float(row.get("predicted_total_runs"))
        market_total = _to_float(row.get("market_total"))
        actual_total = _to_float(row.get("total_runs_first5"))
        supported = any(
            value is not None
            for value in (
                predicted_total,
                market_total,
                actual_total,
                _to_float(row.get("away_expected_runs")),
                _to_float(row.get("home_expected_runs")),
            )
        )
        recommended_side = _recommended_side(predicted_total, market_total)
        actual_side = _actual_side(actual_total, market_total) if actual_total is not None else None
        payload_by_game[int(game_id)] = {
            "supported": supported,
            "model_name": row.get("model_name"),
            "model_version": row.get("model_version"),
            "prediction_ts": row.get("prediction_ts"),
            "predicted_total_runs": row.get("predicted_total_runs"),
            "market_total": row.get("market_total"),
            "over_probability": row.get("over_probability"),
            "under_probability": row.get("under_probability"),
            "edge": row.get("edge"),
            "market_backed": market_total is not None,
            "away_expected_runs": row.get("away_expected_runs"),
            "home_expected_runs": row.get("home_expected_runs"),
            "away_runs": row.get("away_runs_first5"),
            "home_runs": row.get("home_runs_first5"),
            "actual_total_runs": row.get("total_runs_first5"),
            "recommended_side": recommended_side,
            "actual_side": actual_side,
            "result": _graded_pick_result(recommended_side, actual_side, actual_total is not None),
            "delta_vs_market": None if predicted_total is None or market_total is None else round(predicted_total - market_total, 2),
        }
    return payload_by_game


def _fetch_totals_board(target_date: date) -> dict[str, Any]:
    board_rows = _fetch_game_board(
        target_date,
        hit_limit_per_team=1,
        min_probability=1.0,
        confirmed_only=False,
        include_inferred=True,
    )
    first5_totals_map = _fetch_first5_totals_map(target_date)
    rows: list[dict[str, Any]] = []
    first5_supported_games = 0
    for game in board_rows:
        totals = dict(game.get("totals") or {})
        market_total = _to_float(totals.get("market_total"))
        predicted_total = _to_float(totals.get("predicted_total_runs"))
        actual = dict(game.get("actual_result") or {})
        actual_total = _to_float(actual.get("total_runs"))
        actual_side = _actual_side(actual_total, market_total) if actual.get("is_final") else None

        # Full-game totals lane is research-only: suppress directional calls.
        lane_status = totals.get("lane_status") or "research_only"
        if lane_status == "research_only":
            recommended_side = None
        else:
            recommended_side = _recommended_side(predicted_total, market_total)
        game_id = int(game.get("game_id") or 0)
        first5_totals = dict(first5_totals_map.get(game_id) or {"supported": False})
        if first5_totals.get("supported"):
            first5_supported_games += 1
        totals.update(
            {
                "recommended_side": recommended_side,
                "actual_side": actual_side,
                "result": _graded_pick_result(recommended_side, actual_side, bool(actual.get("is_final"))),
                "delta_vs_market": None if predicted_total is None or market_total is None else round(predicted_total - market_total, 2),
                "lane_status": lane_status,
            }
        )
        rows.append(
            {
                "game_id": game.get("game_id"),
                "game_date": game.get("game_date"),
                "status": game.get("status"),
                "game_start_ts": game.get("game_start_ts"),
                "away_team": game.get("away_team"),
                "home_team": game.get("home_team"),
                "venue": game.get("venue") or {},
                "weather": game.get("weather") or {},
                "totals": totals,
                "first5_totals": first5_totals,
                "starters": game.get("starters") or {},
                "actual_result": actual,
                "recent_offense": {
                    "away": _fetch_team_recent_offense(str(game.get("away_team") or ""), target_date),
                    "home": _fetch_team_recent_offense(str(game.get("home_team") or ""), target_date),
                },
                "recent_totals": {
                    "away": _fetch_team_recent_totals_history(str(game.get("away_team") or ""), target_date),
                    "home": _fetch_team_recent_totals_history(str(game.get("home_team") or ""), target_date),
                },
            }
        )

    summary = _summarize_board_rows(board_rows, target_date)
    summary["first5_supported_games"] = first5_supported_games
    return {
        "summary": summary,
        "first5_supported": first5_supported_games > 0,
        "games": rows,
    }


def _fetch_starter_recent_form(pitcher_id: int | None, target_date: date) -> dict[str, Any]:
    default = {
        "sample_starts": 0,
        "avg_ip": None,
        "avg_strikeouts": None,
        "avg_walks": None,
        "avg_pitch_count": None,
        "xwoba_against": None,
        "csw_pct": None,
        "whiff_pct": None,
        "avg_fb_velo": None,
        "last_start_date": None,
        "recent_starts": [],
    }
    if pitcher_id is None or not _table_exists("pitcher_starts"):
        return default
    frame = _safe_frame(
        """
        WITH recent AS (
            SELECT *
            FROM pitcher_starts
            WHERE pitcher_id = :pitcher_id
              AND game_date < :target_date
            ORDER BY game_date DESC
            LIMIT 5
        )
        SELECT
            COUNT(*) AS sample_starts,
            AVG(ip) AS avg_ip,
            AVG(strikeouts) AS avg_strikeouts,
            AVG(walks) AS avg_walks,
            AVG(pitch_count) AS avg_pitch_count,
            AVG(xwoba_against) AS xwoba_against,
            AVG(csw_pct) AS csw_pct,
            AVG(whiff_pct) AS whiff_pct,
            AVG(avg_fb_velo) AS avg_fb_velo,
            MAX(game_date) AS last_start_date
        FROM recent
        """,
        {"pitcher_id": pitcher_id, "target_date": target_date},
    )
    if frame.empty:
        return default
    record = _frame_records(frame)[0]
    recent_starts = _fetch_pitcher_recent_starts(pitcher_id, target_date, limit=5)
    return {**default, **record, "recent_starts": recent_starts}


def _fetch_game_detail(game_id: int, target_date: date, include_inferred: bool = False) -> dict[str, Any] | None:
    if not _table_exists("games"):
        return None

    game_frame = _safe_frame(
        """
        WITH ranked_predictions AS (
            SELECT
                p.*, 
                ROW_NUMBER() OVER (PARTITION BY p.game_id ORDER BY p.prediction_ts DESC) AS row_rank
            FROM predictions_totals p
            WHERE p.game_date = :target_date
              AND p.game_id = :game_id
        ),
        ranked_features AS (
            SELECT
                f.*,
                ROW_NUMBER() OVER (PARTITION BY f.game_id ORDER BY f.prediction_ts DESC) AS row_rank
            FROM game_features_totals f
            WHERE f.game_date = :target_date
              AND f.game_id = :game_id
        )
        SELECT
            g.game_id,
            g.game_date,
            g.status,
            g.away_team,
            g.home_team,
            g.game_start_ts,
            g.away_runs,
            g.home_runs,
            g.total_runs,
            g.home_win,
            COALESCE(v.venue_name, g.venue_name) AS venue_name,
            v.city AS venue_city,
            v.state AS venue_state,
            v.roof_type,
            p.model_name,
            p.model_version,
            p.prediction_ts,
            p.predicted_total_runs,
            p.market_total,
            p.over_probability,
            p.under_probability,
            p.edge,
            p.confidence_level,
            p.suppress_reason,
            p.lane_status,
            CAST(f.feature_payload ->> 'away_runs_rate_blended' AS DOUBLE PRECISION) AS away_expected_runs,
            CAST(f.feature_payload ->> 'home_runs_rate_blended' AS DOUBLE PRECISION) AS home_expected_runs,
            CAST(f.feature_payload ->> 'away_bullpen_pitches_last3' AS DOUBLE PRECISION) AS away_bullpen_pitches_last3,
            CAST(f.feature_payload ->> 'home_bullpen_pitches_last3' AS DOUBLE PRECISION) AS home_bullpen_pitches_last3,
            CAST(f.feature_payload ->> 'away_bullpen_innings_last3' AS DOUBLE PRECISION) AS away_bullpen_innings_last3,
            CAST(f.feature_payload ->> 'home_bullpen_innings_last3' AS DOUBLE PRECISION) AS home_bullpen_innings_last3,
            CAST(f.feature_payload ->> 'away_bullpen_b2b' AS DOUBLE PRECISION) AS away_bullpen_b2b,
            CAST(f.feature_payload ->> 'home_bullpen_b2b' AS DOUBLE PRECISION) AS home_bullpen_b2b,
            CAST(f.feature_payload ->> 'away_bullpen_runs_allowed_last3' AS DOUBLE PRECISION) AS away_bullpen_runs_allowed_last3,
            CAST(f.feature_payload ->> 'home_bullpen_runs_allowed_last3' AS DOUBLE PRECISION) AS home_bullpen_runs_allowed_last3,
            CAST(f.feature_payload ->> 'away_bullpen_earned_runs_last3' AS DOUBLE PRECISION) AS away_bullpen_earned_runs_last3,
            CAST(f.feature_payload ->> 'home_bullpen_earned_runs_last3' AS DOUBLE PRECISION) AS home_bullpen_earned_runs_last3,
            CAST(f.feature_payload ->> 'away_bullpen_hits_allowed_last3' AS DOUBLE PRECISION) AS away_bullpen_hits_allowed_last3,
            CAST(f.feature_payload ->> 'home_bullpen_hits_allowed_last3' AS DOUBLE PRECISION) AS home_bullpen_hits_allowed_last3,
            CAST(f.feature_payload ->> 'away_bullpen_era_last3' AS DOUBLE PRECISION) AS away_bullpen_era_last3,
            CAST(f.feature_payload ->> 'home_bullpen_era_last3' AS DOUBLE PRECISION) AS home_bullpen_era_last3,
            CAST(f.feature_payload ->> 'away_lineup_top5_xwoba' AS DOUBLE PRECISION) AS away_lineup_top5_xwoba,
            CAST(f.feature_payload ->> 'home_lineup_top5_xwoba' AS DOUBLE PRECISION) AS home_lineup_top5_xwoba,
            CAST(f.feature_payload ->> 'away_lineup_k_pct' AS DOUBLE PRECISION) AS away_lineup_k_pct,
            CAST(f.feature_payload ->> 'home_lineup_k_pct' AS DOUBLE PRECISION) AS home_lineup_k_pct,
            CAST(f.feature_payload ->> 'away_k_pct_blended' AS DOUBLE PRECISION) AS away_k_pct_blended,
            CAST(f.feature_payload ->> 'home_k_pct_blended' AS DOUBLE PRECISION) AS home_k_pct_blended,
            CAST(f.feature_payload ->> 'venue_run_factor' AS DOUBLE PRECISION) AS venue_run_factor,
            CAST(f.feature_payload ->> 'venue_hr_factor' AS DOUBLE PRECISION) AS venue_hr_factor,
            CAST(f.feature_payload ->> 'temperature_f' AS DOUBLE PRECISION) AS temperature_f,
            CAST(f.feature_payload ->> 'wind_speed_mph' AS DOUBLE PRECISION) AS wind_speed_mph,
            CAST(f.feature_payload ->> 'wind_direction_deg' AS DOUBLE PRECISION) AS wind_direction_deg,
            CAST(f.feature_payload ->> 'humidity_pct' AS DOUBLE PRECISION) AS humidity_pct,
            CAST(f.feature_payload ->> 'line_movement' AS DOUBLE PRECISION) AS line_movement,
            CAST(f.feature_payload ->> 'starter_certainty_score' AS DOUBLE PRECISION) AS starter_certainty_score,
            CAST(f.feature_payload ->> 'lineup_certainty_score' AS DOUBLE PRECISION) AS lineup_certainty_score,
            CAST(f.feature_payload ->> 'weather_freshness_score' AS DOUBLE PRECISION) AS weather_freshness_score,
            CAST(f.feature_payload ->> 'market_freshness_score' AS DOUBLE PRECISION) AS market_freshness_score,
            CAST(f.feature_payload ->> 'bullpen_completeness_score' AS DOUBLE PRECISION) AS bullpen_completeness_score,
            CAST(f.feature_payload ->> 'missing_fallback_count' AS INTEGER) AS missing_fallback_count,
            f.feature_payload ->> 'board_state' AS board_state
        FROM games g
        LEFT JOIN dim_venues v ON v.venue_id = g.venue_id
        LEFT JOIN ranked_predictions p ON p.game_id = g.game_id AND p.row_rank = 1
        LEFT JOIN ranked_features f ON f.game_id = g.game_id AND f.row_rank = 1
        WHERE g.game_id = :game_id
          AND g.game_date = :target_date
        LIMIT 1
        """,
        {"game_id": game_id, "target_date": target_date},
    )
    if game_frame.empty:
        return None
    game = _frame_records(game_frame)[0]
    is_final = _is_final_game_status(game.get("status"))
    pitcher_k_market_map = _fetch_pitcher_strikeout_market_map(target_date, game_id=int(game["game_id"]))
    pitcher_k_prediction_map = _fetch_pitcher_strikeout_prediction_map(target_date, game_id=int(game["game_id"]))

    starters_frame = _safe_frame(
        """
        WITH ranked_starters AS (
            SELECT
                s.game_id,
                s.team,
                s.pitcher_id,
                s.is_probable,
                s.days_rest,
                s.ip,
                s.strikeouts,
                s.walks,
                s.pitch_count,
                s.xwoba_against,
                s.csw_pct,
                s.avg_fb_velo,
                s.whiff_pct,
                dp.full_name AS pitcher_name,
                dp.throws,
                ROW_NUMBER() OVER (
                    PARTITION BY s.game_id, s.team
                    ORDER BY COALESCE(s.is_probable, FALSE) DESC, s.pitcher_id
                ) AS row_rank
            FROM pitcher_starts s
            LEFT JOIN dim_players dp ON dp.player_id = s.pitcher_id
            WHERE s.game_id = :game_id
              AND s.game_date = :target_date
        )
        SELECT
            game_id,
            team,
            pitcher_id,
            COALESCE(pitcher_name, CAST(pitcher_id AS TEXT)) AS pitcher_name,
            throws,
            is_probable,
            days_rest,
            ip,
            strikeouts,
            walks,
            pitch_count,
            xwoba_against,
            csw_pct,
            avg_fb_velo,
            whiff_pct
        FROM ranked_starters
        WHERE row_rank = 1
        """,
        {"game_id": game_id, "target_date": target_date},
    )
    starter_records = _frame_records(starters_frame)

    lineup_records: list[dict[str, Any]] = []
    if _table_exists("player_features_hits"):
        recent_batting_avg_expr = _sql_ratio("hits", "at_bats")
        season_batting_avg_expr = _sql_ratio("b.hits", "b.at_bats")
        game_year_expr = _sql_year("b.game_date")
        target_year_expr = _sql_year_param("target_date")
        player_name_expr = _sql_json_text("f.feature_payload", "player_name")
        lineup_slot_expr = _sql_integer(f"NULLIF({_sql_json_text('f.feature_payload', 'lineup_slot')}, '')")
        confirmed_lineup_expr = _sql_boolean(f"NULLIF({_sql_json_text('f.feature_payload', 'is_confirmed_lineup')}, '')")
        projected_pa_expr = _sql_real(f"NULLIF({_sql_json_text('f.feature_payload', 'projected_plate_appearances')}, '')")
        streak_len_expr = _sql_integer(f"NULLIF({_sql_json_text('f.feature_payload', 'streak_len_capped')}, '')")
        streak_len_full_expr = _sql_integer(f"NULLIF({_sql_json_text('f.feature_payload', 'streak_len')}, '')")
        hit_rate_7_expr = _sql_real(f"NULLIF({_sql_json_text('f.feature_payload', 'hit_rate_7')}, '')")
        hit_rate_14_expr = _sql_real(f"NULLIF({_sql_json_text('f.feature_payload', 'hit_rate_14')}, '')")
        hit_rate_30_expr = _sql_real(f"NULLIF({_sql_json_text('f.feature_payload', 'hit_rate_30')}, '')")
        hit_rate_blended_expr = _sql_real(f"NULLIF({_sql_json_text('f.feature_payload', 'hit_rate_blended')}, '')")
        xba_14_expr = _sql_real(f"NULLIF({_sql_json_text('f.feature_payload', 'xba_14')}, '')")
        xwoba_14_expr = _sql_real(f"NULLIF({_sql_json_text('f.feature_payload', 'xwoba_14')}, '')")
        hard_hit_pct_14_expr = _sql_real(f"NULLIF({_sql_json_text('f.feature_payload', 'hard_hit_pct_14')}, '')")
        k_pct_14_expr = _sql_real(f"NULLIF({_sql_json_text('f.feature_payload', 'k_pct_14')}, '')")
        lineup_slot_order = _sql_order_nulls_last(lineup_slot_expr)
        lineup_frame = _safe_frame(
            f"""
            WITH ranked_predictions AS (
                SELECT
                    p.*, 
                    ROW_NUMBER() OVER (
                        PARTITION BY p.game_id, p.player_id
                        ORDER BY p.prediction_ts DESC
                    ) AS row_rank
                FROM predictions_player_hits p
                WHERE p.game_id = :game_id
                  AND p.game_date = :target_date
            ),
            ranked_features AS (
                SELECT
                    f.*, 
                    ROW_NUMBER() OVER (
                        PARTITION BY f.game_id, f.player_id
                        ORDER BY f.prediction_ts DESC
                    ) AS row_rank
                FROM player_features_hits f
                WHERE f.game_id = :game_id
                  AND f.game_date = :target_date
            ),
            ranked_lineups AS (
                SELECT
                    l.game_id,
                    l.player_id,
                    l.player_name,
                    l.team,
                    l.lineup_slot,
                    l.field_position,
                    l.is_confirmed,
                    l.source_name,
                    l.source_url,
                    l.snapshot_ts,
                    ROW_NUMBER() OVER (
                        PARTITION BY l.game_id, l.player_id
                        ORDER BY l.snapshot_ts DESC, l.lineup_slot ASC, l.player_id
                    ) AS row_rank
                FROM lineups l
                WHERE l.game_id = :game_id
                  AND l.game_date = :target_date
            ),
            selected_players AS (
                SELECT DISTINCT game_id, player_id
                FROM ranked_lineups
                WHERE row_rank = 1
                UNION
                SELECT DISTINCT game_id, player_id
                FROM ranked_features
                WHERE row_rank = 1
            ),
            recent_batting AS (
                SELECT
                    player_id,
                    COUNT(*) AS games_last7,
                    SUM(CASE WHEN hits > 0 THEN 1 ELSE 0 END) AS hit_games_last7,
                    SUM(hits) AS hits_last7,
                    SUM(at_bats) AS at_bats_last7,
                    SUM(plate_appearances) AS plate_appearances_last7,
                    {recent_batting_avg_expr} AS batting_avg_last7,
                    AVG(xwoba) AS xwoba_last7,
                    AVG(hard_hit_pct) AS hard_hit_pct_last7
                FROM (
                    SELECT
                        b.*, 
                        ROW_NUMBER() OVER (
                            PARTITION BY b.player_id
                            ORDER BY b.game_date DESC, b.game_id DESC
                        ) AS row_rank
                    FROM player_game_batting b
                    INNER JOIN selected_players sp ON sp.player_id = b.player_id
                    WHERE b.game_date < :target_date
                ) recent
                WHERE row_rank <= 7
                GROUP BY player_id
            ),
            season_batting AS (
                SELECT
                    b.player_id,
                    COUNT(*) AS games_season,
                    SUM(b.hits) AS season_hits,
                    SUM(b.at_bats) AS season_at_bats,
                                        {season_batting_avg_expr} AS batting_avg_season
                FROM player_game_batting b
                INNER JOIN selected_players sp ON sp.player_id = b.player_id
                WHERE b.game_date < :target_date
                                    AND {game_year_expr} = {target_year_expr}
                GROUP BY b.player_id
            )
            SELECT
                sp.game_id,
                sp.player_id,
                COALESCE(rl.player_name, {player_name_expr}, dp.full_name, CAST(sp.player_id AS TEXT)) AS player_name,
                COALESCE(rl.team, f.team, CASE WHEN g.home_team = dp.team_abbr THEN g.home_team ELSE g.away_team END) AS team,
                COALESCE(
                    f.opponent,
                    CASE
                        WHEN g.home_team = COALESCE(rl.team, f.team, dp.team_abbr) THEN g.away_team
                        WHEN g.away_team = COALESCE(rl.team, f.team, dp.team_abbr) THEN g.home_team
                        ELSE NULL
                    END,
                    'TBD'
                ) AS opponent,
                COALESCE(rl.lineup_slot, {lineup_slot_expr}) AS lineup_slot,
                COALESCE(rl.is_confirmed, {confirmed_lineup_expr}) AS is_confirmed_lineup,
                rl.source_name AS lineup_source_name,
                rl.source_url AS lineup_source_url,
                rl.snapshot_ts AS lineup_snapshot_ts,
                {projected_pa_expr} AS projected_plate_appearances,
                {streak_len_expr} AS streak_len_capped,
                {streak_len_full_expr} AS streak_len,
                {hit_rate_7_expr} AS hit_rate_7,
                {hit_rate_14_expr} AS hit_rate_14,
                {hit_rate_30_expr} AS hit_rate_30,
                {hit_rate_blended_expr} AS hit_rate_blended,
                {xba_14_expr} AS xba_14,
                {xwoba_14_expr} AS xwoba_14,
                {hard_hit_pct_14_expr} AS hard_hit_pct_14,
                {k_pct_14_expr} AS k_pct_14,
                p.predicted_hit_probability,
                p.fair_price,
                p.market_price,
                p.edge,
                dp.bats,
                COALESCE(rl.field_position, dp.position) AS position,
                season_batting.games_season,
                season_batting.season_hits,
                season_batting.season_at_bats,
                season_batting.batting_avg_season,
                actual.hits AS actual_hits,
                actual.plate_appearances AS actual_plate_appearances,
                actual.at_bats AS actual_at_bats,
                actual.runs AS actual_runs,
                actual.rbi AS actual_rbi,
                actual.walks AS actual_walks,
                actual.home_runs AS actual_home_runs,
                actual.stolen_bases AS actual_stolen_bases,
                (
                    COALESCE(actual.singles, 0)
                    + 2 * COALESCE(actual.doubles, 0)
                    + 3 * COALESCE(actual.triples, 0)
                    + 4 * COALESCE(actual.home_runs, 0)
                ) AS actual_total_bases,
                recent_batting.games_last7,
                recent_batting.hit_games_last7,
                recent_batting.hits_last7,
                recent_batting.at_bats_last7,
                recent_batting.plate_appearances_last7,
                recent_batting.batting_avg_last7,
                recent_batting.xwoba_last7,
                recent_batting.hard_hit_pct_last7
            FROM selected_players sp
            LEFT JOIN ranked_features f
                ON f.game_id = sp.game_id
               AND f.player_id = sp.player_id
               AND f.row_rank = 1
            LEFT JOIN ranked_predictions p
                ON p.game_id = sp.game_id
               AND p.player_id = sp.player_id
               AND p.row_rank = 1
            LEFT JOIN ranked_lineups rl
                ON rl.game_id = sp.game_id
               AND rl.player_id = sp.player_id
               AND rl.row_rank = 1
            LEFT JOIN dim_players dp ON dp.player_id = sp.player_id
            LEFT JOIN games g ON g.game_id = sp.game_id
            LEFT JOIN season_batting ON season_batting.player_id = sp.player_id
            LEFT JOIN player_game_batting actual
                ON actual.game_id = sp.game_id
               AND actual.player_id = sp.player_id
            LEFT JOIN recent_batting ON recent_batting.player_id = sp.player_id
            ORDER BY
                CASE
                    WHEN COALESCE(rl.team, f.team, dp.team_abbr) = g.away_team THEN 0
                    WHEN COALESCE(rl.team, f.team, dp.team_abbr) = g.home_team THEN 1
                    ELSE 2
                END,
                {lineup_slot_order},
                player_name
            """,
            {"game_id": game_id, "target_date": target_date},
        )
        lineup_records = _frame_records(lineup_frame)
        lineup_records = _annotate_lineup_confidence(lineup_records, _fetch_lineup_snapshot_keys(target_date))
        if not include_inferred:
            lineup_records = [record for record in lineup_records if not record.get("is_inferred_lineup")]
        hit_history_map = _fetch_recent_hit_history_map(
            target_date,
            [int(player["player_id"]) for player in lineup_records if player.get("player_id") is not None],
            limit=15,
        )
        lineup_split_map = _fetch_hitter_pitch_hand_splits(
            target_date,
            [int(player["player_id"]) for player in lineup_records if player.get("player_id") is not None],
        )
    else:
        hit_history_map = {}
        lineup_split_map = {}

    total_freeze = _fetch_market_freeze_map(target_date).get((int(game_id), "total"), {})

    detail = {
        "game_id": int(game["game_id"]),
        "game_date": game["game_date"],
        "status": game["status"],
        "away_team": game["away_team"],
        "home_team": game["home_team"],
        "game_start_ts": game["game_start_ts"],
        "venue": {
            "name": game["venue_name"],
            "city": game["venue_city"],
            "state": game["venue_state"],
            "roof_type": game["roof_type"],
        },
        "weather": {
            "temperature_f": game["temperature_f"],
            "wind_speed_mph": game["wind_speed_mph"],
            "wind_direction_deg": game["wind_direction_deg"],
            "humidity_pct": game["humidity_pct"],
        },
        "actual_result": {
            "away_runs": game["away_runs"] if is_final else None,
            "home_runs": game["home_runs"] if is_final else None,
            "total_runs": game["total_runs"] if is_final else None,
            "home_win": game["home_win"] if is_final else None,
            "is_final": is_final,
        },
        "totals": {
            "model_name": game["model_name"],
            "model_version": game["model_version"],
            "prediction_ts": game["prediction_ts"],
            "predicted_total_runs": game["predicted_total_runs"],
            "market_total": game["market_total"],
            "market_locked": bool(total_freeze),
            "locked_sportsbook": total_freeze.get("frozen_sportsbook"),
            "locked_snapshot_ts": total_freeze.get("frozen_snapshot_ts"),
            "locked_line_value": total_freeze.get("frozen_line_value"),
            "over_probability": game["over_probability"],
            "under_probability": game["under_probability"],
            "edge": game["edge"],
            "away_expected_runs": game["away_expected_runs"],
            "home_expected_runs": game["home_expected_runs"],
            "away_bullpen_pitches_last3": game["away_bullpen_pitches_last3"],
            "home_bullpen_pitches_last3": game["home_bullpen_pitches_last3"],
            "away_bullpen_innings_last3": game["away_bullpen_innings_last3"],
            "home_bullpen_innings_last3": game["home_bullpen_innings_last3"],
            "away_bullpen_b2b": game["away_bullpen_b2b"],
            "home_bullpen_b2b": game["home_bullpen_b2b"],
            "away_bullpen_runs_allowed_last3": game["away_bullpen_runs_allowed_last3"],
            "home_bullpen_runs_allowed_last3": game["home_bullpen_runs_allowed_last3"],
            "away_bullpen_earned_runs_last3": game["away_bullpen_earned_runs_last3"],
            "home_bullpen_earned_runs_last3": game["home_bullpen_earned_runs_last3"],
            "away_bullpen_hits_allowed_last3": game["away_bullpen_hits_allowed_last3"],
            "home_bullpen_hits_allowed_last3": game["home_bullpen_hits_allowed_last3"],
            "away_bullpen_era_last3": game["away_bullpen_era_last3"],
            "home_bullpen_era_last3": game["home_bullpen_era_last3"],
            "away_lineup_top5_xwoba": game["away_lineup_top5_xwoba"],
            "home_lineup_top5_xwoba": game["home_lineup_top5_xwoba"],
            "away_lineup_k_pct": game["away_lineup_k_pct"],
            "home_lineup_k_pct": game["home_lineup_k_pct"],
            "away_k_pct_blended": game["away_k_pct_blended"],
            "home_k_pct_blended": game["home_k_pct_blended"],
            "venue_run_factor": game["venue_run_factor"],
            "venue_hr_factor": game["venue_hr_factor"],
            "line_movement": game["line_movement"],
        },
        "certainty": {
            "starter_certainty": game["starter_certainty_score"],
            "lineup_certainty": game["lineup_certainty_score"],
            "weather_freshness": game["weather_freshness_score"],
            "market_freshness": game["market_freshness_score"],
            "bullpen_completeness": game["bullpen_completeness_score"],
            "missing_fallback_count": game["missing_fallback_count"],
            "board_state": game["board_state"],
        },
        "starters": {"away": None, "home": None},
        "teams": {
            "away": {
                "team": game["away_team"],
                "recent_offense": _fetch_team_recent_offense(game["away_team"], target_date),
                "last_result": _fetch_team_last_result(game["away_team"], target_date),
                "lineup": [],
            },
            "home": {
                "team": game["home_team"],
                "recent_offense": _fetch_team_recent_offense(game["home_team"], target_date),
                "last_result": _fetch_team_last_result(game["home_team"], target_date),
                "lineup": [],
            },
        },
    }

    for starter in starter_records:
        side = None
        if starter["team"] == detail["away_team"]:
            side = "away"
        elif starter["team"] == detail["home_team"]:
            side = "home"
        if side is None:
            continue
        detail["starters"][side] = {
            "team": starter["team"],
            "pitcher_id": starter["pitcher_id"],
            "pitcher_name": starter["pitcher_name"],
            "throws": starter["throws"],
            "is_probable": starter["is_probable"],
            "days_rest": starter["days_rest"],
            "ip": starter["ip"],
            "strikeouts": starter["strikeouts"],
            "walks": starter["walks"],
            "pitch_count": starter["pitch_count"],
            "xwoba_against": starter["xwoba_against"],
            "csw_pct": starter["csw_pct"],
            "avg_fb_velo": starter["avg_fb_velo"],
            "whiff_pct": starter["whiff_pct"],
            "recent_form": _fetch_starter_recent_form(starter["pitcher_id"], target_date),
        }

    for player in lineup_records:
        player.update(_build_hit_actual_meta(player.get("actual_hits"), is_final))
        player_id = player.get("player_id")
        player["recent_hit_history"] = [] if player_id is None else hit_history_map.get(int(player_id), [])
        side = "away" if player["team"] == detail["away_team"] else "home"
        opposing_starter = detail["starters"]["home"] if side == "away" else detail["starters"]["away"]
        _attach_hitter_matchup_context(
            player,
            opposing_starter["throws"] if opposing_starter else None,
            lineup_split_map,
        )
        detail["teams"][side]["lineup"].append(player)

    for side in ("away", "home"):
        team_summary = _summarize_team_lineup(detail["teams"][side]["lineup"])
        detail["teams"][side]["lineup"] = team_summary["lineup"]
        detail["teams"][side]["lineup_scope"] = team_summary["lineup_scope"]
        detail["teams"][side]["lineup_source_summary"] = team_summary["lineup_source_summary"]
        detail["teams"][side]["lineup_counts"] = team_summary["lineup_counts"]
        team_lineup = detail["teams"][side]["lineup"]
        detail["teams"][side]["confirmed_lineup"] = team_summary["lineup_scope"] == "confirmed"
        detail["teams"][side]["lineup_handedness"] = _summarize_lineup_handedness(
            team_lineup,
            confirmed_key="is_confirmed_lineup",
        )

    detail["lineup_handedness"] = {
        detail["away_team"]: detail["teams"]["away"]["lineup_handedness"],
        detail["home_team"]: detail["teams"]["home"]["lineup_handedness"],
    }

    detail["review"] = _build_totals_review_block(
        detail,
        _fetch_totals_outcome_review(int(game_id), target_date),
    )

    if detail["starters"]["away"]:
        away_key = (detail["game_id"], int(detail["starters"]["away"]["pitcher_id"])) if detail["starters"]["away"].get("pitcher_id") is not None else None
        away_market = pitcher_k_market_map.get(away_key) if away_key is not None else None
        modeled_projection = pitcher_k_prediction_map.get(away_key) if away_key is not None else None
        detail["starters"]["away"]["k_projection"] = _merge_strikeout_market_context(
            modeled_projection,
            away_market,
        ) or _estimate_starter_strikeout_projection(
            detail["starters"]["away"],
            detail["starters"]["away"].get("recent_form"),
            detail["totals"].get("home_lineup_k_pct"),
            detail["totals"].get("home_k_pct_blended"),
            detail["teams"]["home"].get("lineup_handedness"),
            away_market,
        )
    if detail["starters"]["home"]:
        home_key = (detail["game_id"], int(detail["starters"]["home"]["pitcher_id"])) if detail["starters"]["home"].get("pitcher_id") is not None else None
        home_market = pitcher_k_market_map.get(home_key) if home_key is not None else None
        modeled_projection = pitcher_k_prediction_map.get(home_key) if home_key is not None else None
        detail["starters"]["home"]["k_projection"] = _merge_strikeout_market_context(
            modeled_projection,
            home_market,
        ) or _estimate_starter_strikeout_projection(
            detail["starters"]["home"],
            detail["starters"]["home"].get("recent_form"),
            detail["totals"].get("away_lineup_k_pct"),
            detail["totals"].get("away_k_pct_blended"),
            detail["teams"]["away"].get("lineup_handedness"),
            home_market,
        )

    return detail


def _fetch_hot_hitters(
    target_date: date,
    min_probability: float,
    confirmed_only: bool,
    limit: int,
    include_inferred: bool,
) -> dict[str, Any]:
    def _empty_payload(
        *,
        suppressed_inferred_count: int = 0,
        inferred_count: int = 0,
        available_count: int = 0,
        available_confirmed_count: int = 0,
        available_market_count: int = 0,
        confirmed_fallback_active: bool = False,
    ) -> dict[str, Any]:
        return {
            "rows": [],
            "summary": {
                "count": 0,
                "total_hot_count": 0,
                "confirmed_count": 0,
                "inferred_count": inferred_count,
                "suppressed_inferred_count": suppressed_inferred_count,
                "available_count": available_count,
                "available_confirmed_count": available_confirmed_count,
                "available_market_count": available_market_count,
                "confirmed_fallback_active": confirmed_fallback_active,
                "games_count": 0,
                "average_hit_probability": None,
                "latest_prediction_ts": None,
            },
        }

    empty = _empty_payload()
    if not _table_exists("player_features_hits") or not _table_exists("games"):
        return empty

    recent_batting_avg_expr = _sql_ratio("hits", "at_bats")
    season_batting_avg_expr = _sql_ratio("b.hits", "b.at_bats")
    game_year_expr = _sql_year("b.game_date")
    target_year_expr = _sql_year_param("target_date")
    player_name_expr = _sql_json_text("f.feature_payload", "player_name")
    lineup_slot_expr = _sql_integer(f"NULLIF({_sql_json_text('f.feature_payload', 'lineup_slot')}, '')")
    confirmed_lineup_expr = _sql_boolean(f"NULLIF({_sql_json_text('f.feature_payload', 'is_confirmed_lineup')}, '')")
    projected_pa_expr = _sql_real(f"NULLIF({_sql_json_text('f.feature_payload', 'projected_plate_appearances')}, '')")
    streak_len_expr = _sql_integer(f"NULLIF({_sql_json_text('f.feature_payload', 'streak_len_capped')}, '')")
    streak_len_full_expr = _sql_integer(f"NULLIF({_sql_json_text('f.feature_payload', 'streak_len')}, '')")
    hit_rate_7_expr = _sql_real(f"NULLIF({_sql_json_text('f.feature_payload', 'hit_rate_7')}, '')")
    hit_rate_14_expr = _sql_real(f"NULLIF({_sql_json_text('f.feature_payload', 'hit_rate_14')}, '')")
    hit_rate_30_expr = _sql_real(f"NULLIF({_sql_json_text('f.feature_payload', 'hit_rate_30')}, '')")
    hit_rate_blended_expr = _sql_real(f"NULLIF({_sql_json_text('f.feature_payload', 'hit_rate_blended')}, '')")
    xba_14_expr = _sql_real(f"NULLIF({_sql_json_text('f.feature_payload', 'xba_14')}, '')")
    xwoba_14_expr = _sql_real(f"NULLIF({_sql_json_text('f.feature_payload', 'xwoba_14')}, '')")
    hard_hit_pct_14_expr = _sql_real(f"NULLIF({_sql_json_text('f.feature_payload', 'hard_hit_pct_14')}, '')")
    k_pct_14_expr = _sql_real(f"NULLIF({_sql_json_text('f.feature_payload', 'k_pct_14')}, '')")
    game_start_order = _sql_order_nulls_last("g.game_start_ts")
    lineup_slot_order = _sql_order_nulls_last("lineup_slot")

    frame = _safe_frame(
        f"""
        WITH ranked_features AS (
            SELECT
                f.*,
                ROW_NUMBER() OVER (
                    PARTITION BY f.game_id, f.player_id
                    ORDER BY f.prediction_ts DESC
                ) AS row_rank
            FROM player_features_hits f
            WHERE f.game_date = :target_date
        ),
        ranked_predictions AS (
            SELECT
                p.*,
                ROW_NUMBER() OVER (
                    PARTITION BY p.game_id, p.player_id
                    ORDER BY p.prediction_ts DESC
                ) AS row_rank
            FROM predictions_player_hits p
            WHERE p.game_date = :target_date
        ),
        ranked_starters AS (
            SELECT
                s.game_id,
                s.team,
                s.pitcher_id,
                dp.full_name AS pitcher_name,
                dp.throws,
                ROW_NUMBER() OVER (
                    PARTITION BY s.game_id, s.team
                    ORDER BY COALESCE(s.is_probable, FALSE) DESC, s.pitcher_id
                ) AS row_rank
            FROM pitcher_starts s
            LEFT JOIN dim_players dp ON dp.player_id = s.pitcher_id
            WHERE s.game_date = :target_date
        ),
        selected_players AS (
            SELECT DISTINCT player_id
            FROM ranked_features
            WHERE row_rank = 1
        ),
        recent_batting AS (
            SELECT
                player_id,
                COUNT(*) AS games_last7,
                SUM(CASE WHEN hits > 0 THEN 1 ELSE 0 END) AS hit_games_last7,
                SUM(hits) AS hits_last7,
                SUM(at_bats) AS at_bats_last7,
                SUM(plate_appearances) AS plate_appearances_last7,
                {recent_batting_avg_expr} AS batting_avg_last7,
                AVG(xwoba) AS xwoba_last7,
                AVG(hard_hit_pct) AS hard_hit_pct_last7
            FROM (
                SELECT
                    b.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY b.player_id
                        ORDER BY b.game_date DESC, b.game_id DESC
                    ) AS row_rank
                FROM player_game_batting b
                INNER JOIN selected_players sp ON sp.player_id = b.player_id
                WHERE b.game_date < :target_date
            ) recent
            WHERE row_rank <= 7
            GROUP BY player_id
        ),
        season_batting AS (
            SELECT
                b.player_id,
                COUNT(*) AS games_season,
                SUM(b.hits) AS season_hits,
                SUM(b.at_bats) AS season_at_bats,
                                {season_batting_avg_expr} AS batting_avg_season
            FROM player_game_batting b
            INNER JOIN selected_players sp ON sp.player_id = b.player_id
            WHERE b.game_date < :target_date
                            AND {game_year_expr} = {target_year_expr}
            GROUP BY b.player_id
        )
        SELECT
            f.game_id,
            g.game_date,
            g.game_start_ts,
            g.status AS game_status,
            g.away_team,
            g.home_team,
            f.prediction_ts,
            f.player_id,
            COALESCE({player_name_expr}, dp.full_name, CAST(f.player_id AS TEXT)) AS player_name,
            COALESCE(f.team, CASE WHEN g.home_team = dp.team_abbr THEN g.home_team ELSE g.away_team END) AS team,
            COALESCE(
                f.opponent,
                CASE
                    WHEN g.home_team = COALESCE(f.team, dp.team_abbr) THEN g.away_team
                    WHEN g.away_team = COALESCE(f.team, dp.team_abbr) THEN g.home_team
                    ELSE NULL
                END,
                'TBD'
            ) AS opponent,
                {lineup_slot_expr} AS lineup_slot,
                {confirmed_lineup_expr} AS is_confirmed_lineup,
                {projected_pa_expr} AS projected_plate_appearances,
                {streak_len_expr} AS streak_len_capped,
                {streak_len_full_expr} AS streak_len,
                {hit_rate_7_expr} AS hit_rate_7,
                {hit_rate_14_expr} AS hit_rate_14,
                {hit_rate_30_expr} AS hit_rate_30,
                {hit_rate_blended_expr} AS hit_rate_blended,
                {xba_14_expr} AS xba_14,
                {xwoba_14_expr} AS xwoba_14,
                {hard_hit_pct_14_expr} AS hard_hit_pct_14,
                {k_pct_14_expr} AS k_pct_14,
            p.predicted_hit_probability,
            p.fair_price,
            p.market_price,
            p.edge,
            dp.bats,
            dp.position,
            season_batting.games_season,
            season_batting.season_hits,
            season_batting.season_at_bats,
            season_batting.batting_avg_season,
            actual.hits AS actual_hits,
            actual.plate_appearances AS actual_plate_appearances,
            actual.at_bats AS actual_at_bats,
            actual.runs AS actual_runs,
            actual.rbi AS actual_rbi,
            actual.walks AS actual_walks,
            actual.home_runs AS actual_home_runs,
            actual.stolen_bases AS actual_stolen_bases,
            (
                COALESCE(actual.singles, 0)
                + 2 * COALESCE(actual.doubles, 0)
                + 3 * COALESCE(actual.triples, 0)
                + 4 * COALESCE(actual.home_runs, 0)
            ) AS actual_total_bases,
            recent_batting.games_last7,
            recent_batting.hit_games_last7,
            recent_batting.hits_last7,
            recent_batting.at_bats_last7,
            recent_batting.plate_appearances_last7,
            recent_batting.batting_avg_last7,
            recent_batting.xwoba_last7,
            recent_batting.hard_hit_pct_last7,
            rs.pitcher_id AS opposing_pitcher_id,
            COALESCE(rs.pitcher_name, CAST(rs.pitcher_id AS TEXT)) AS opposing_pitcher_name,
            rs.throws AS opposing_pitcher_throws
        FROM ranked_features f
        LEFT JOIN ranked_predictions p
            ON p.game_id = f.game_id
           AND p.player_id = f.player_id
           AND p.row_rank = 1
        LEFT JOIN dim_players dp ON dp.player_id = f.player_id
        LEFT JOIN games g ON g.game_id = f.game_id AND g.game_date = :target_date
        LEFT JOIN season_batting ON season_batting.player_id = f.player_id
        LEFT JOIN recent_batting ON recent_batting.player_id = f.player_id
        LEFT JOIN player_game_batting actual
            ON actual.game_id = f.game_id
           AND actual.player_id = f.player_id
        LEFT JOIN ranked_starters rs
            ON rs.game_id = f.game_id
           AND rs.team = COALESCE(
                f.opponent,
                CASE
                    WHEN g.home_team = COALESCE(f.team, dp.team_abbr) THEN g.away_team
                    WHEN g.away_team = COALESCE(f.team, dp.team_abbr) THEN g.home_team
                    ELSE NULL
                END
           )
           AND rs.row_rank = 1
        WHERE f.row_rank = 1
            ORDER BY {game_start_order}, team, {lineup_slot_order}, player_name
        """,
        {"target_date": target_date},
    )
    if frame.empty:
        return empty

    records = _frame_records(frame)
    records = _annotate_lineup_confidence(records, _fetch_lineup_snapshot_keys(target_date))
    available_count = len(records)
    available_confirmed_count = sum(1 for record in records if record.get("is_confirmed_lineup"))
    available_market_count = sum(1 for record in records if _to_float(record.get("market_price")) is not None)
    confirmed_fallback_active = False
    if confirmed_only:
        confirmed_records = [record for record in records if record.get("is_confirmed_lineup")]
        if confirmed_records:
            records = confirmed_records
        else:
            confirmed_fallback_active = available_count > 0
    suppressed_inferred_count = 0
    if not include_inferred:
        suppressed_inferred_count = sum(1 for record in records if record.get("is_inferred_lineup"))
        records = [record for record in records if not record.get("is_inferred_lineup")]
    if not records:
        return _empty_payload(
            suppressed_inferred_count=suppressed_inferred_count,
            available_count=available_count,
            available_confirmed_count=available_confirmed_count,
            available_market_count=available_market_count,
            confirmed_fallback_active=confirmed_fallback_active,
        )

    hit_split_map = _fetch_hitter_pitch_hand_splits(
        target_date,
        [int(record["player_id"]) for record in records if record.get("player_id") is not None],
    )
    hit_history_map = _fetch_recent_hit_history_map(
        target_date,
        [int(record["player_id"]) for record in records if record.get("player_id") is not None],
        limit=10,
    )

    all_hot_rows: list[dict[str, Any]] = []
    for record in records:
        form = _classify_hitter_form(record)
        if form["label"] not in {"Hot", "Streaking", "Hitting well"}:
            continue
        player_id = int(record["player_id"]) if record.get("player_id") is not None else None
        actual_meta = _build_hit_actual_meta(record.get("actual_hits"), _is_final_game_status(record.get("game_status")))
        enriched = _attach_hitter_matchup_context(
            {
                **record,
                **actual_meta,
                "form": form,
                "recent_hit_history": [] if player_id is None else hit_history_map.get(player_id, []),
            },
            record.get("opposing_pitcher_throws"),
            hit_split_map,
        )
        all_hot_rows.append(enriched)

    if min_probability > 0:
        hot_rows = [
            row
            for row in all_hot_rows
            if (_to_float(row.get("predicted_hit_probability")) or 0.0) >= min_probability
        ]
    else:
        hot_rows = list(all_hot_rows)

    hot_rows.sort(
        key=lambda row: (
            -float((row.get("form") or {}).get("heat_score") or 0.0),
            -float(_to_float(row.get("predicted_hit_probability")) or 0.0),
            -float(_to_float(row.get("edge")) or 0.0),
            int(row.get("lineup_slot") or 99),
            str(row.get("player_name") or ""),
        )
    )
    hot_rows = hot_rows[:limit]

    probabilities = [_to_float(row.get("predicted_hit_probability")) for row in hot_rows]
    valid_probabilities = [value for value in probabilities if value is not None]
    return {
        "rows": hot_rows,
        "summary": {
            "count": len(hot_rows),
            "total_hot_count": len(all_hot_rows),
            "confirmed_count": sum(1 for row in hot_rows if row.get("is_confirmed_lineup")),
            "inferred_count": sum(1 for row in hot_rows if row.get("is_inferred_lineup")),
            "suppressed_inferred_count": suppressed_inferred_count,
            "available_count": available_count,
            "available_confirmed_count": available_confirmed_count,
            "available_market_count": available_market_count,
            "confirmed_fallback_active": confirmed_fallback_active,
            "games_count": len({int(row["game_id"]) for row in hot_rows if row.get("game_id") is not None}),
            "average_hit_probability": round(sum(valid_probabilities) / len(valid_probabilities), 4) if valid_probabilities else None,
            "latest_prediction_ts": max((row.get("prediction_ts") for row in hot_rows if row.get("prediction_ts") is not None), default=None),
        },
    }


def _recommended_side(predicted_value: Any, market_line: Any) -> str | None:
    predicted = _to_float(predicted_value)
    line = _to_float(market_line)
    if predicted is None or line is None:
        return None
    return "over" if predicted >= line else "under"


def _actual_side(actual_value: Any, market_line: Any) -> str | None:
    actual = _to_float(actual_value)
    line = _to_float(market_line)
    if actual is None or line is None:
        return None
    if actual > line:
        return "over"
    if actual < line:
        return "under"
    return "push"


def _graded_pick_result(recommended_side: str | None, actual_side: str | None, is_final: bool) -> str:
    if recommended_side is None:
        return "no_line"
    if actual_side is None:
        return "pending" if not is_final else "missing"
    if actual_side == "push":
        return "push"
    return "won" if recommended_side == actual_side else "lost"


def _summarize_category(rows: list[dict[str, Any]], *, result_key: str = "result") -> dict[str, Any]:
    summary = {
        "total": len(rows),
        "graded": 0,
        "pending": 0,
        "won": 0,
        "lost": 0,
        "push": 0,
        "hit": 0,
        "no_hit": 0,
        "market_backed": 0,
        "no_line": 0,
        "missing": 0,
    }
    for row in rows:
        result = str(row.get(result_key) or "pending")
        if row.get("market_backed"):
            summary["market_backed"] += 1
        if result in {"won", "lost", "push", "hit", "no_hit"}:
            summary["graded"] += 1
        if result in summary:
            summary[result] += 1
        else:
            summary["pending"] += 1
    return summary


def _fetch_daily_results(target_date: date, hit_min_probability: float = 0.5) -> dict[str, Any]:
    totals_rows: list[dict[str, Any]] = []
    hitter_rows: list[dict[str, Any]] = []
    strikeout_rows: list[dict[str, Any]] = []

    if _table_exists("predictions_totals") and _table_exists("games"):
        totals_frame = _safe_frame(
            """
            WITH ranked_predictions AS (
                SELECT
                    p.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY p.game_id
                        ORDER BY p.prediction_ts DESC
                    ) AS row_rank
                FROM predictions_totals p
                WHERE p.game_date = :target_date
            )
            SELECT
                p.game_id,
                p.game_date,
                g.game_start_ts,
                g.status,
                g.away_team,
                g.home_team,
                p.predicted_total_runs,
                p.market_total,
                p.over_probability,
                p.under_probability,
                p.edge,
                g.total_runs AS actual_total_runs
            FROM ranked_predictions p
            INNER JOIN games g
                ON g.game_id = p.game_id
               AND g.game_date = p.game_date
            WHERE p.row_rank = 1
            ORDER BY g.game_start_ts, p.game_id
            """,
            {"target_date": target_date},
        )
        for row in _frame_records(totals_frame):
            is_final = _is_final_game_status(row.get("status"))
            recommended_side = None  # Full-game totals is research_only — suppress directional picks
            actual_side = _actual_side(row.get("actual_total_runs"), row.get("market_total")) if is_final else None
            totals_rows.append(
                {
                    "game_id": row.get("game_id"),
                    "game_date": row.get("game_date"),
                    "game_start_ts": row.get("game_start_ts"),
                    "status": row.get("status"),
                    "is_final": is_final,
                    "matchup": f"{row.get('away_team')} at {row.get('home_team')}",
                    "away_team": row.get("away_team"),
                    "home_team": row.get("home_team"),
                    "predicted_total_runs": row.get("predicted_total_runs"),
                    "market_total": row.get("market_total"),
                    "over_probability": row.get("over_probability"),
                    "under_probability": row.get("under_probability"),
                    "edge": row.get("edge"),
                    "actual_total_runs": row.get("actual_total_runs"),
                    "recommended_side": recommended_side,
                    "actual_side": actual_side,
                    "result": _graded_pick_result(recommended_side, actual_side, is_final),
                    "market_backed": row.get("market_total") is not None,
                }
            )

    if _table_exists("predictions_player_hits") and _table_exists("player_features_hits") and _table_exists("games"):
        lineup_slot_order = _sql_order_nulls_last("lineup_slot")
        hitter_frame = _safe_frame(
            f"""
            WITH ranked_predictions AS (
                SELECT
                    p.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY p.game_id, p.player_id
                        ORDER BY p.prediction_ts DESC
                    ) AS row_rank
                FROM predictions_player_hits p
                WHERE p.game_date = :target_date
            ),
            ranked_features AS (
                SELECT
                    f.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY f.game_id, f.player_id
                        ORDER BY f.prediction_ts DESC
                    ) AS row_rank
                FROM player_features_hits f
                WHERE f.game_date = :target_date
            )
            SELECT
                p.game_id,
                p.game_date,
                g.game_start_ts,
                g.status AS game_status,
                p.player_id,
                COALESCE(f.feature_payload ->> 'player_name', dp.full_name, CAST(p.player_id AS TEXT)) AS player_name,
                COALESCE(f.team, p.team) AS team,
                COALESCE(
                    f.opponent,
                    CASE
                        WHEN g.home_team = COALESCE(f.team, p.team) THEN g.away_team
                        WHEN g.away_team = COALESCE(f.team, p.team) THEN g.home_team
                        ELSE NULL
                    END,
                    'TBD'
                ) AS opponent,
                CAST(NULLIF(f.feature_payload ->> 'lineup_slot', '') AS SMALLINT) AS lineup_slot,
                p.predicted_hit_probability,
                p.fair_price,
                p.market_price,
                p.edge,
                actual.hits AS actual_hits,
                actual.at_bats AS actual_at_bats,
                actual.plate_appearances AS actual_plate_appearances
            FROM ranked_predictions p
            INNER JOIN games g
                ON g.game_id = p.game_id
               AND g.game_date = p.game_date
            LEFT JOIN ranked_features f
                ON f.game_id = p.game_id
               AND f.player_id = p.player_id
               AND f.row_rank = 1
            LEFT JOIN dim_players dp ON dp.player_id = p.player_id
            LEFT JOIN player_game_batting actual
                ON actual.game_id = p.game_id
               AND actual.player_id = p.player_id
            WHERE p.row_rank = 1
              AND p.predicted_hit_probability >= :hit_min_probability
                        ORDER BY g.game_start_ts, p.predicted_hit_probability DESC, {lineup_slot_order}, player_name
            """,
            {"target_date": target_date, "hit_min_probability": hit_min_probability},
        )
        for row in _frame_records(hitter_frame):
            is_final = _is_final_game_status(row.get("game_status"))
            actual_hits = _to_float(row.get("actual_hits"))
            actual_meta = _build_hit_actual_meta(row.get("actual_hits"), is_final)
            hitter_rows.append(
                {
                    "game_id": row.get("game_id"),
                    "game_date": row.get("game_date"),
                    "game_start_ts": row.get("game_start_ts"),
                    "game_status": row.get("game_status"),
                    "is_final": is_final,
                    "player_id": row.get("player_id"),
                    "player_name": row.get("player_name"),
                    "team": row.get("team"),
                    "opponent": row.get("opponent"),
                    "lineup_slot": row.get("lineup_slot"),
                    "predicted_hit_probability": row.get("predicted_hit_probability"),
                    "fair_price": row.get("fair_price"),
                    "market_price": row.get("market_price"),
                    "edge": row.get("edge"),
                    "actual_hits": row.get("actual_hits"),
                    "actual_at_bats": row.get("actual_at_bats"),
                    "actual_plate_appearances": row.get("actual_plate_appearances"),
                    "result": "hit" if actual_hits and actual_hits > 0 else ("no_hit" if actual_hits == 0 else ("missing" if is_final else "pending")),
                    "market_backed": row.get("market_price") is not None,
                    "actual_status": actual_meta.get("actual_status"),
                    "actual_status_label": actual_meta.get("actual_status_label"),
                }
            )

    if _table_exists("predictions_pitcher_strikeouts") and _table_exists("games"):
        strikeout_frame = _safe_frame(
            """
            WITH ranked_predictions AS (
                SELECT
                    p.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY p.game_id, p.pitcher_id
                        ORDER BY p.prediction_ts DESC
                    ) AS row_rank
                FROM predictions_pitcher_strikeouts p
                WHERE p.game_date = :target_date
            )
            SELECT
                p.game_id,
                p.game_date,
                g.game_start_ts,
                g.status AS game_status,
                p.pitcher_id,
                COALESCE(dp.full_name, CAST(p.pitcher_id AS TEXT)) AS pitcher_name,
                p.team,
                CASE
                    WHEN p.team = g.home_team THEN g.away_team
                    WHEN p.team = g.away_team THEN g.home_team
                    ELSE NULL
                END AS opponent,
                p.predicted_strikeouts,
                p.market_line,
                p.over_probability,
                p.under_probability,
                p.edge,
                ps.strikeouts AS actual_strikeouts
            FROM ranked_predictions p
            INNER JOIN games g
                ON g.game_id = p.game_id
               AND g.game_date = p.game_date
            LEFT JOIN dim_players dp ON dp.player_id = p.pitcher_id
            LEFT JOIN pitcher_starts ps
                ON ps.game_id = p.game_id
               AND ps.pitcher_id = p.pitcher_id
               AND ps.game_date = p.game_date
            WHERE p.row_rank = 1
            ORDER BY g.game_start_ts, CASE WHEN p.market_line IS NULL THEN 1 ELSE 0 END, p.team, pitcher_name
            """,
            {"target_date": target_date},
        )
        for row in _frame_records(strikeout_frame):
            is_final = _is_final_game_status(row.get("game_status"))
            recommended_side = _recommended_side(row.get("predicted_strikeouts"), row.get("market_line"))
            actual_side = _actual_side(row.get("actual_strikeouts"), row.get("market_line")) if is_final else None
            strikeout_rows.append(
                {
                    "game_id": row.get("game_id"),
                    "game_date": row.get("game_date"),
                    "game_start_ts": row.get("game_start_ts"),
                    "game_status": row.get("game_status"),
                    "is_final": is_final,
                    "pitcher_id": row.get("pitcher_id"),
                    "pitcher_name": row.get("pitcher_name"),
                    "team": row.get("team"),
                    "opponent": row.get("opponent"),
                    "predicted_strikeouts": row.get("predicted_strikeouts"),
                    "market_line": row.get("market_line"),
                    "over_probability": row.get("over_probability"),
                    "under_probability": row.get("under_probability"),
                    "edge": row.get("edge"),
                    "actual_strikeouts": row.get("actual_strikeouts"),
                    "recommended_side": recommended_side,
                    "actual_side": actual_side,
                    "result": _graded_pick_result(recommended_side, actual_side, is_final),
                    "market_backed": row.get("market_line") is not None,
                }
            )

    final_games = sum(1 for row in totals_rows if row.get("is_final"))
    total_games = len(totals_rows)
    return {
        "summary": {
            "final_games": final_games,
            "pending_games": max(total_games - final_games, 0),
            "total_games": total_games,
            "totals": _summarize_category(totals_rows),
            "hitters": _summarize_category(hitter_rows),
            "strikeouts": _summarize_category(strikeout_rows),
        },
        "totals": totals_rows,
        "hitters": hitter_rows,
        "strikeouts": strikeout_rows,
    }


def _run_module(module_name: str, *args: str) -> dict[str, Any]:
    if getattr(sys, "frozen", False):
        return _run_module_in_process(module_name, *args)

    command = [sys.executable, "-m", module_name, *args]
    completed = subprocess.run(
        command,
        cwd=settings.base_dir,
        capture_output=True,
        text=True,
    )
    return {
        "module": module_name,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _resolve_module_main(module_name: str):
    module_main = UPDATE_MODULE_MAINS.get(module_name)
    if module_main is not None:
        return module_main

    module = importlib.import_module(module_name)
    module_main = getattr(module, "main", None)
    if not callable(module_main):
        raise AttributeError(f"{module_name} does not define a callable main()")
    return module_main


def _run_module_in_process(module_name: str, *args: str) -> dict[str, Any]:
    command = [sys.executable, "-m", module_name, *args]
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    original_argv = list(sys.argv)

    try:
        module_main = _resolve_module_main(module_name)

        sys.argv = [module_name, *args]
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            try:
                result = module_main()
                returncode = 0 if result is None else int(result)
            except SystemExit as exc:
                if isinstance(exc.code, int):
                    returncode = exc.code
                else:
                    returncode = 0 if exc.code in {None, ""} else 1
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=stderr_buffer)
        returncode = 1
    finally:
        sys.argv = original_argv

    return {
        "module": module_name,
        "command": command,
        "returncode": returncode,
        "stdout": stdout_buffer.getvalue().strip(),
        "stderr": stderr_buffer.getvalue().strip(),
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _public_update_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "action": job["action"],
        "label": job["label"],
        "target_date": job["target_date"],
        "status": job["status"],
        "created_at": job["created_at"],
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "current_step": job.get("current_step"),
        "completed_steps": job["completed_steps"],
        "total_steps": job["total_steps"],
        "steps": [dict(step) for step in job["steps"]],
        "error": job.get("error"),
        "status_snapshot": job.get("status_snapshot"),
    }


def _trim_finished_jobs_locked() -> None:
    finished_job_ids = [
        existing_job_id
        for existing_job_id, existing_job in UPDATE_JOBS.items()
        if existing_job["status"] in {"succeeded", "failed"}
    ]
    if len(finished_job_ids) <= UPDATE_JOB_HISTORY_LIMIT:
        return
    finished_job_ids.sort(key=lambda job_id: UPDATE_JOBS[job_id]["created_at"])
    for job_id in finished_job_ids[: len(finished_job_ids) - UPDATE_JOB_HISTORY_LIMIT]:
        UPDATE_JOBS.pop(job_id, None)


def _persist_update_jobs() -> None:
    with UPDATE_JOB_LOCK:
        payload = [
            _public_update_job(job)
            for job in sorted(
                UPDATE_JOBS.values(),
                key=lambda item: item["created_at"],
                reverse=True,
            )
        ]
    UPDATE_JOB_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = UPDATE_JOB_STORE_PATH.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temp_path.replace(UPDATE_JOB_STORE_PATH)


def _hydrate_persisted_job(payload: dict[str, Any]) -> dict[str, Any] | None:
    job_id = str(payload.get("job_id") or "").strip()
    action = payload.get("action")
    target_date = str(payload.get("target_date") or "").strip()
    created_at = str(payload.get("created_at") or "").strip()
    if not job_id or action not in {"refresh_everything", "prepare_slate", "import_manual_inputs", "refresh_results", "rebuild_predictions", "grade_predictions"}:
        return None
    if not target_date or not created_at:
        return None
    steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
    status = str(payload.get("status") or "failed").strip().lower()
    error = payload.get("error")
    if status in {"queued", "running"}:
        status = "failed"
        error = error or "Application restarted before the update job finished."
    job = {
        "job_id": job_id,
        "action": action,
        "label": _update_job_label(action),
        "target_date": target_date,
        "sequence": [],
        "status": status,
        "created_at": created_at,
        "started_at": payload.get("started_at"),
        "finished_at": payload.get("finished_at") or _utc_now_iso(),
        "current_step": None,
        "completed_steps": int(payload.get("completed_steps") or len(steps)),
        "total_steps": int(payload.get("total_steps") or len(steps)),
        "steps": steps,
        "error": error,
        "status_snapshot": payload.get("status_snapshot"),
    }
    return job


def _load_persisted_update_jobs() -> None:
    if not UPDATE_JOB_STORE_PATH.exists():
        return
    try:
        payload = json.loads(UPDATE_JOB_STORE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Failed to load persisted update jobs: %s", exc)
        return
    if not isinstance(payload, list):
        return
    loaded_jobs = [job for item in payload if isinstance(item, dict) for job in [_hydrate_persisted_job(item)] if job is not None]
    with UPDATE_JOB_LOCK:
        UPDATE_JOBS.clear()
        for job in loaded_jobs:
            UPDATE_JOBS[job["job_id"]] = job
        _trim_finished_jobs_locked()
    _persist_update_jobs()


def _active_update_job_payload() -> dict[str, Any] | None:
    with UPDATE_JOB_LOCK:
        active_jobs = [
            _public_update_job(job)
            for job in UPDATE_JOBS.values()
            if job["status"] in {"queued", "running"}
        ]
    if not active_jobs:
        return None
    active_jobs.sort(key=lambda item: item["created_at"], reverse=True)
    return active_jobs[0]


def _update_job_history_payload() -> list[dict[str, Any]]:
    with UPDATE_JOB_LOCK:
        jobs = [
            _public_update_job(job)
            for job in sorted(
                UPDATE_JOBS.values(),
                key=lambda item: item["created_at"],
                reverse=True,
            )
        ]
    return jobs


def _safe_fetch_status(target_date: str) -> dict[str, Any] | None:
    try:
        return _fetch_status(date.fromisoformat(target_date))
    except Exception as exc:
        log.warning("Failed to capture update job status snapshot: %s", exc)
        return None


def _create_update_job(action: UpdateAction, target_date: str) -> dict[str, Any]:
    sequence = _update_job_sequence(action, target_date)
    job = {
        "job_id": uuid4().hex,
        "action": action,
        "label": _update_job_label(action),
        "target_date": target_date,
        "sequence": sequence,
        "status": "queued",
        "created_at": _utc_now_iso(),
        "started_at": None,
        "finished_at": None,
        "current_step": None,
        "completed_steps": 0,
        "total_steps": len(sequence),
        "steps": [],
        "error": None,
        "status_snapshot": None,
    }
    with UPDATE_JOB_LOCK:
        UPDATE_JOBS[job["job_id"]] = job
        _trim_finished_jobs_locked()
    _persist_pipeline_run(job)
    _persist_update_jobs()
    return _public_update_job(job)


def _get_update_job(job_id: str) -> dict[str, Any] | None:
    with UPDATE_JOB_LOCK:
        job = UPDATE_JOBS.get(job_id)
        return None if job is None else _public_update_job(job)


def _run_update_job_background(job_id: str) -> None:
    with UPDATE_JOB_LOCK:
        job = UPDATE_JOBS.get(job_id)
        if job is None:
            return
        job["status"] = "running"
        job["started_at"] = _utc_now_iso()
        _persist_pipeline_run(job)
    _persist_update_jobs()

    target_date = None
    sequence: list[tuple[str, list[str]]] = []
    with UPDATE_JOB_LOCK:
        job = UPDATE_JOBS.get(job_id)
        if job is None:
            return
        target_date = str(job["target_date"])
        action = job["action"]
        sequence = list(job["sequence"])

    blocker = _action_blocker(action, date.fromisoformat(target_date))
    if blocker is not None:
        with UPDATE_JOB_LOCK:
            job = UPDATE_JOBS.get(job_id)
            if job is None:
                return
            job["status"] = "failed"
            job["finished_at"] = _utc_now_iso()
            job["current_step"] = None
            job["error"] = blocker["message"]
            job["status_snapshot"] = _safe_fetch_status(target_date)
            _trim_finished_jobs_locked()
            _persist_pipeline_run(job)
        _persist_update_jobs()
        return

    for index, (module_name, args) in enumerate(sequence, start=1):
        with UPDATE_JOB_LOCK:
            job = UPDATE_JOBS.get(job_id)
            if job is None:
                return
            job["current_step"] = module_name
        step = _run_module(module_name, *args)
        _persist_pipeline_step(job_id, index, step)
        with UPDATE_JOB_LOCK:
            job = UPDATE_JOBS.get(job_id)
            if job is None:
                return
            job["steps"].append(step)
            job["completed_steps"] = len(job["steps"])
            if step["returncode"] != 0:
                job["status"] = "failed"
                job["finished_at"] = _utc_now_iso()
                job["current_step"] = None
                job["error"] = f"{module_name} exited with code {step['returncode']}"
                job["status_snapshot"] = _safe_fetch_status(target_date)
                _trim_finished_jobs_locked()
                _persist_pipeline_run(job)
                _persist_update_jobs()
                return
            _persist_pipeline_run(job)
        _persist_update_jobs()

    with UPDATE_JOB_LOCK:
        job = UPDATE_JOBS.get(job_id)
        if job is None:
            return
        job["status"] = "succeeded"
        job["finished_at"] = _utc_now_iso()
        job["current_step"] = None
        job["status_snapshot"] = _safe_fetch_status(target_date)
        _trim_finished_jobs_locked()
        _persist_pipeline_run(job)
    _persist_update_jobs()


def _launch_update_job(job_id: str) -> None:
    thread = threading.Thread(
        target=_run_update_job_background,
        args=(job_id,),
        name=f"update-job-{job_id[:8]}",
        daemon=True,
    )
    thread.start()


def _run_module_sequence(sequence: list[tuple[str, list[str]]]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    steps: list[dict[str, Any]] = []
    failed_step: dict[str, Any] | None = None
    for module_name, args in sequence:
        step = _run_module(module_name, *args)
        steps.append(step)
        if step["returncode"] != 0:
            failed_step = step
            break
    return steps, failed_step


def _publish_target_date_sequence(
    target_date: str,
    *,
    refresh_aggregates: bool,
    rebuild_features: bool,
    include_market_freeze: bool,
) -> list[tuple[str, list[str]]]:
    sequence: list[tuple[str, list[str]]] = []
    if refresh_aggregates:
        sequence.extend(
            [
                ("src.transforms.offense_daily", []),
                ("src.transforms.bullpens_daily", []),
            ]
        )
    if rebuild_features:
        if include_market_freeze:
            sequence.append(("src.transforms.freeze_markets", ["--target-date", target_date]))
        sequence.extend(
            [
                ("src.features.totals_builder", ["--target-date", target_date]),
                ("src.features.first5_totals_builder", ["--target-date", target_date]),
                ("src.features.hits_builder", ["--target-date", target_date]),
                ("src.features.strikeouts_builder", ["--target-date", target_date]),
            ]
        )
    sequence.extend(
        [
            ("src.models.predict_totals", ["--target-date", target_date]),
            ("src.models.predict_first5_totals", ["--target-date", target_date]),
            ("src.models.predict_hits", ["--target-date", target_date]),
            ("src.models.predict_strikeouts", ["--target-date", target_date]),
            ("src.transforms.product_surfaces", ["--target-date", target_date]),
        ]
    )
    return sequence


def _pipeline_sequence(target_date: str, refresh_aggregates: bool, rebuild_features: bool) -> list[tuple[str, list[str]]]:
    return _publish_target_date_sequence(
        target_date,
        refresh_aggregates=refresh_aggregates,
        rebuild_features=rebuild_features,
        include_market_freeze=True,
    )


def _results_refresh_sequence(target_date: str) -> list[tuple[str, list[str]]]:
    return [
        ("src.ingestors.boxscores", ["--target-date", target_date]),
        ("src.ingestors.player_batting", ["--target-date", target_date]),
        ("src.ingestors.weather", ["--target-date", target_date, "--mode", "observed"]),
        ("src.transforms.offense_daily", ["--target-date", target_date]),
        ("src.transforms.bullpens_daily", ["--target-date", target_date]),
        ("src.transforms.product_surfaces", ["--target-date", target_date]),
    ]


def _update_job_sequence(action: UpdateAction, target_date: str) -> list[tuple[str, list[str]]]:
    if action == "refresh_everything":
        return [
            ("src.ingestors.games", ["--target-date", target_date]),
            ("src.ingestors.starters", ["--target-date", target_date]),
            ("src.ingestors.prepare_slate_inputs", ["--target-date", target_date]),
            ("src.ingestors.lineups", ["--target-date", target_date]),
            ("src.ingestors.market_totals", ["--target-date", target_date]),
            ("src.ingestors.weather", ["--target-date", target_date]),
            ("src.transforms.freeze_markets", ["--target-date", target_date]),
            ("src.ingestors.validator", ["--target-date", target_date]),
            ("src.transforms.offense_daily", []),
            ("src.transforms.bullpens_daily", []),
            ("src.features.totals_builder", ["--target-date", target_date]),
            ("src.features.first5_totals_builder", ["--target-date", target_date]),
            ("src.features.hits_builder", ["--target-date", target_date]),
            ("src.features.strikeouts_builder", ["--target-date", target_date]),
            ("src.models.predict_totals", ["--target-date", target_date]),
            ("src.models.predict_first5_totals", ["--target-date", target_date]),
            ("src.models.predict_hits", ["--target-date", target_date]),
            ("src.models.predict_strikeouts", ["--target-date", target_date]),
            ("src.transforms.product_surfaces", ["--target-date", target_date]),
        ]
    if action == "prepare_slate":
        return [
            ("src.ingestors.games", ["--target-date", target_date]),
            ("src.ingestors.starters", ["--target-date", target_date]),
            ("src.ingestors.prepare_slate_inputs", ["--target-date", target_date]),
            ("src.ingestors.lineups", ["--target-date", target_date]),
            ("src.ingestors.market_totals", ["--target-date", target_date]),
            ("src.ingestors.weather", ["--target-date", target_date]),
            ("src.transforms.freeze_markets", ["--target-date", target_date]),
            ("src.ingestors.validator", ["--target-date", target_date]),
            *_publish_target_date_sequence(
                target_date,
                refresh_aggregates=False,
                rebuild_features=True,
                include_market_freeze=False,
            ),
        ]
    if action == "import_manual_inputs":
        return [
            ("src.ingestors.lineups", ["--target-date", target_date]),
            ("src.ingestors.market_totals", ["--target-date", target_date]),
            ("src.transforms.freeze_markets", ["--target-date", target_date]),
            ("src.ingestors.validator", ["--target-date", target_date]),
            *_publish_target_date_sequence(
                target_date,
                refresh_aggregates=False,
                rebuild_features=True,
                include_market_freeze=False,
            ),
        ]
    if action == "refresh_results":
        return _results_refresh_sequence(target_date)
    if action == "grade_predictions":
        return _results_refresh_sequence(target_date)
    return _pipeline_sequence(target_date, refresh_aggregates=False, rebuild_features=True)


def _update_job_label(action: UpdateAction) -> str:
    return {
        "refresh_everything": "Refresh Everything",
        "prepare_slate": "Prepare Slate",
        "import_manual_inputs": "Update Lineups & Markets",
        "refresh_results": "Refresh Daily Results",
        "rebuild_predictions": "Rebuild Predictions",
        "grade_predictions": "Grade Predictions",
    }[action]


def _json_response(payload: dict[str, Any], status_code: int = 200) -> JSONResponse:
    return JSONResponse(content=jsonable_encoder(payload), status_code=status_code)


@app.get("/health")
@app.get("/api/health")
def health() -> JSONResponse:
    return _json_response(_fetch_status(date.today()))


@app.get("/")
def index() -> FileResponse:
    return _html_file_response(INDEX_FILE)


@app.get("/favicon.ico", include_in_schema=False)
@app.get("/favicon.svg", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(FAVICON_FILE, media_type="image/svg+xml")


@app.get("/hot-hittes")
@app.get("/hot-hittes/")
@app.get("/hot-hitters/")
@app.get("/hot-hitters")
def hot_hitters_page() -> FileResponse:
    return _html_file_response(HOT_HITTERS_FILE)


@app.get("/results/")
@app.get("/results")
def results_page() -> FileResponse:
    return _html_file_response(RESULTS_FILE)


@app.get("/doctor/")
@app.get("/doctor")
def doctor_page() -> FileResponse:
    return _html_file_response(DOCTOR_FILE)


@app.get("/totals/")
@app.get("/totals")
def totals_page() -> FileResponse:
    return _html_file_response(TOTALS_FILE)


@app.get("/pitchers/")
@app.get("/pitchers")
def pitchers_page() -> FileResponse:
    return _html_file_response(PITCHERS_FILE)


@app.get("/game/")
@app.get("/game")
def game_page() -> FileResponse:
    return _html_file_response(GAME_FILE)


@app.get("/api/status")
def status(target_date: date = Query(default_factory=date.today)) -> JSONResponse:
    return _json_response(_fetch_status(target_date))


@app.get("/api/doctor")
def doctor_status(
    target_date: date = Query(default_factory=date.today),
    source_health_hours: int = Query(default=24, ge=1, le=168),
    pipeline_limit: int = Query(default=5, ge=1, le=20),
    update_history_limit: int = Query(default=5, ge=1, le=20),
) -> JSONResponse:
    return _json_response(
        _doctor_payload(
            target_date,
            source_health_hours=source_health_hours,
            pipeline_limit=pipeline_limit,
            update_history_limit=update_history_limit,
        )
    )


@app.get("/api/predictions/totals")
def totals_predictions(target_date: date = Query(default_factory=date.today)) -> JSONResponse:
    rows = _fetch_totals_predictions(target_date)
    return _json_response({"target_date": target_date.isoformat(), "rows": rows})


@app.get("/api/totals/board")
def totals_board(target_date: date = Query(default_factory=date.today)) -> JSONResponse:
    return _json_response({"target_date": target_date.isoformat(), **_fetch_totals_board(target_date)})


@app.get("/api/predictions/hits")
def hit_predictions(
    target_date: date = Query(default_factory=date.today),
    limit: int = Query(default=40, ge=1, le=200),
    min_probability: float = Query(default=0.0, ge=0.0, le=1.0),
    confirmed_only: bool = Query(default=False),
    include_inferred: bool = Query(default=True),
) -> JSONResponse:
    rows = _fetch_hit_predictions(target_date, limit, min_probability, confirmed_only, include_inferred)
    return _json_response({"target_date": target_date.isoformat(), "rows": rows})


@app.get("/api/hot-hitters")
def hot_hitters(
    target_date: date = Query(default_factory=date.today),
    limit: int = Query(default=60, ge=1, le=200),
    min_probability: float = Query(default=0.35, ge=0.0, le=1.0),
    confirmed_only: bool = Query(default=False),
    include_inferred: bool = Query(default=True),
) -> JSONResponse:
    payload = _fetch_hot_hitters(target_date, min_probability, confirmed_only, limit, include_inferred)
    return _json_response({"target_date": target_date.isoformat(), **payload})


@app.get("/api/results/daily")
def daily_results(
    target_date: date = Query(default_factory=date.today),
    hit_min_probability: float = Query(default=0.5, ge=0.0, le=1.0),
) -> JSONResponse:
    return _json_response(
        {
            "target_date": target_date.isoformat(),
            **_fetch_daily_results(target_date, hit_min_probability),
        }
    )


@app.get("/api/model-scorecards")
def model_scorecards(target_date: date = Query(default_factory=date.today), window_days: int = Query(default=14, ge=1, le=60)) -> JSONResponse:
    return _json_response({"target_date": target_date.isoformat(), **_fetch_model_scorecards(target_date, window_days)})


@app.get("/api/trends/players/{player_id}")
def player_trend(player_id: int, target_date: date = Query(default_factory=date.today), limit: int = Query(default=10, ge=1, le=30)) -> JSONResponse:
    return _json_response({"target_date": target_date.isoformat(), "player_id": player_id, "rows": _fetch_player_trend(player_id, target_date, limit)})


@app.get("/api/trends/pitchers/{pitcher_id}")
def pitcher_trend(pitcher_id: int, target_date: date = Query(default_factory=date.today), limit: int = Query(default=10, ge=1, le=30)) -> JSONResponse:
    return _json_response({"target_date": target_date.isoformat(), "pitcher_id": pitcher_id, "rows": _fetch_pitcher_trend(pitcher_id, target_date, limit)})


@app.get("/api/pitchers/{pitcher_id}/recent-starts")
def pitcher_recent_starts(pitcher_id: int, target_date: date = Query(default_factory=date.today), limit: int = Query(default=5, ge=1, le=15)) -> JSONResponse:
    return _json_response(
        {
            "target_date": target_date.isoformat(),
            "pitcher_id": pitcher_id,
            "rows": _fetch_pitcher_recent_starts(pitcher_id, target_date, limit),
        }
    )


@app.get("/api/games/board")
def games_board(
    target_date: date = Query(default_factory=date.today),
    hit_limit_per_team: int = Query(default=4, ge=1, le=9),
    min_probability: float = Query(default=0.0, ge=0.0, le=1.0),
    confirmed_only: bool = Query(default=False),
    include_inferred: bool = Query(default=True),
) -> JSONResponse:
    rows = _fetch_game_board(target_date, hit_limit_per_team, min_probability, confirmed_only, include_inferred)
    return _json_response(
        {
            "target_date": target_date.isoformat(),
            "summary": _summarize_board_rows(rows, target_date),
            "games": rows,
        }
    )


@app.get("/api/games/{game_id}/detail")
def game_detail(
    game_id: int,
    target_date: date = Query(default_factory=date.today),
    include_inferred: bool = Query(default=True),
) -> JSONResponse:
    payload = _fetch_game_detail(game_id, target_date, include_inferred=include_inferred)
    if payload is None:
        return _json_response({"target_date": target_date.isoformat(), "game": None}, status_code=404)
    return _json_response({"target_date": target_date.isoformat(), "game": payload})


@app.post("/api/pipeline/run")
def run_pipeline(request: PipelineRunRequest) -> JSONResponse:
    blocker = _pipeline_blocker(request.target_date)
    if blocker is not None:
        return _json_response(_blocked_pipeline_payload(request.target_date, blocker), status_code=409)

    target_date = request.target_date.isoformat()
    steps, failed_step = _run_module_sequence(
        _pipeline_sequence(target_date, request.refresh_aggregates, request.rebuild_features)
    )
    if failed_step is not None:
        return _json_response({"ok": False, "steps": steps}, status_code=500)
    return _json_response(
        {
            "ok": True,
            "target_date": target_date,
            "steps": steps,
            "status": _fetch_status(request.target_date),
        }
    )


@app.post("/api/update-jobs/run")
def run_update_job(request: UpdateJobRunRequest) -> JSONResponse:
    blocker = _action_blocker(request.action, request.target_date)
    if blocker is not None:
        return _json_response(_blocked_update_payload(request.action, request.target_date, blocker), status_code=409)

    target_date = request.target_date.isoformat()
    steps, failed_step = _run_module_sequence(_update_job_sequence(request.action, target_date))
    payload = {
        "ok": failed_step is None,
        "action": request.action,
        "label": _update_job_label(request.action),
        "target_date": target_date,
        "steps": steps,
        "status": _fetch_status(request.target_date),
    }
    if failed_step is not None:
        return _json_response(payload, status_code=500)
    return _json_response(payload)


@app.post("/api/update-jobs/start")
def start_update_job(request: UpdateJobRunRequest) -> JSONResponse:
    blocker = _action_blocker(request.action, request.target_date)
    if blocker is not None:
        return _json_response(
            {
                "ok": False,
                "message": blocker["message"],
                "blocker": blocker,
                "status": _fetch_status(request.target_date),
            },
            status_code=409,
        )

    active_job = _active_update_job_payload()
    if active_job is not None:
        return _json_response(
            {
                "ok": False,
                "message": "Another update job is already running.",
                "active_job": active_job,
            },
            status_code=409,
        )

    job = _create_update_job(request.action, request.target_date.isoformat())
    _launch_update_job(job["job_id"])
    created_job = _get_update_job(job["job_id"])
    return _json_response({"ok": True, "job": created_job}, status_code=202)


@app.get("/api/update-jobs/active")
def active_update_job() -> JSONResponse:
    return _json_response({"job": _active_update_job_payload()})


@app.get("/api/update-jobs/history")
def update_job_history() -> JSONResponse:
    return _json_response({"jobs": _update_job_history_payload()})


@app.get("/api/update-jobs/{job_id}")
def update_job_status(job_id: str) -> JSONResponse:
    job = _get_update_job(job_id)
    if job is None:
        return _json_response({"job": None}, status_code=404)
    return _json_response({"job": job})


@app.get("/api/pipeline-runs")
def pipeline_run_history(limit: int = 20) -> JSONResponse:
    """Return recent pipeline runs from the DB (not in-memory)."""
    return _json_response({"runs": _fetch_pipeline_runs(limit)})


@app.get("/api/pipeline-runs/{job_id}/steps")
def pipeline_run_steps(job_id: str) -> JSONResponse:
    """Return steps for a specific pipeline run from the DB."""
    if not _table_exists("pipeline_run_steps"):
        return _json_response({"steps": []})
    steps_frame = _safe_frame(
        """
        SELECT step_index, module_name, returncode, stdout, stderr,
               started_at, finished_at
        FROM pipeline_run_steps
        WHERE job_id = :job_id
        ORDER BY step_index
        """,
        {"job_id": job_id},
    )
    return _json_response({"steps": _frame_records(steps_frame)})


@app.get("/api/game-readiness")
def game_readiness(target_date: date | None = None) -> JSONResponse:
    """Return the latest validation readiness for each game on the target date."""
    return _json_response(_fetch_game_readiness_payload(target_date))


@app.get("/api/review/top-misses")
def review_top_misses(
    target_date: date = Query(default_factory=date.today),
    market: str = "totals",
    limit: int = Query(default=6, ge=1, le=20),
) -> JSONResponse:
    """Return the biggest misses and best calls for the selected date."""
    return _json_response(_fetch_top_review_payload(target_date, market=market, limit=limit))


@app.get("/api/review/clv")
def review_clv(
    start_date: date | None = None,
    end_date: date | None = None,
    market: str = "totals",
    limit: int = Query(default=8, ge=1, le=25),
) -> JSONResponse:
    """Return best and worst closing-line-value rows over a date range."""
    sd = start_date or (date.today() - timedelta(days=7))
    ed = end_date or date.today()
    return _json_response(_fetch_clv_review_payload(sd, ed, market=market, limit=limit))


@app.get("/api/source-health")
def source_health(hours: int = 24) -> JSONResponse:
    """Return recent source health checks grouped by source."""
    return _json_response(_fetch_source_health_payload(hours))


@app.get("/api/calibration")
def calibration_bins(
    market: str = "totals",
    start_date: date | None = None,
    end_date: date | None = None,
) -> JSONResponse:
    """Return calibration bins for a market over a date range."""
    if not _table_exists("prediction_calibration_bins"):
        return _json_response({"bins": []})
    sd = str(start_date or (date.today() - timedelta(days=30)))
    ed = str(end_date or date.today())
    frame = _safe_frame(
        """
        SELECT bin_label, bin_lower, bin_upper,
               SUM(count) AS count,
               CASE WHEN SUM(count) > 0
                    THEN CAST(SUM(actual_hit_rate * count) AS DOUBLE PRECISION) / SUM(count)
                    ELSE NULL END AS actual_hit_rate,
               CASE WHEN SUM(count) > 0
                    THEN CAST(SUM(mean_predicted_prob * count) AS DOUBLE PRECISION) / SUM(count)
                    ELSE NULL END AS mean_predicted_prob,
               SUM(brier_score_sum) AS brier_score_sum
        FROM prediction_calibration_bins
        WHERE market = :market
          AND score_date BETWEEN :start_date AND :end_date
        GROUP BY bin_label, bin_lower, bin_upper
        ORDER BY bin_lower
        """,
        {"market": market, "start_date": sd, "end_date": ed},
    )
    return _json_response({"bins": _frame_records(frame), "market": market, "start_date": sd, "end_date": ed})


@app.get("/api/recommendations/history")
def recommendation_history_endpoint(
    market: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    graded_only: bool = False,
    limit: int = 50,
) -> JSONResponse:
    """Return recommendation history with optional filters."""
    if not _table_exists("recommendation_history"):
        return _json_response({"recommendations": []})
    sd = str(start_date or (date.today() - timedelta(days=7)))
    ed = str(end_date or date.today())
    conditions = ["game_date BETWEEN :start_date AND :end_date"]
    params: dict[str, Any] = {"start_date": sd, "end_date": ed, "lim": max(1, min(limit, 200))}
    if market:
        conditions.append("market = :market")
        params["market"] = market
    if graded_only:
        conditions.append("graded = TRUE")
    where = " AND ".join(conditions)
    frame = _safe_frame(
        f"""
        SELECT game_date, game_id, market, entity_type, entity_id, player_id,
               team, away_team, home_team, model_name, model_version,
               recommended_side, probability, market_line, predicted_value,
               entry_market_sportsbook, closing_market_sportsbook, closing_market_same_sportsbook,
               closing_market_line, clv_line_delta, clv_side_value, beat_closing_line,
               actual_value, actual_side, graded, success, edge
        FROM recommendation_history
        WHERE {where}
        ORDER BY game_date DESC, game_id
        LIMIT :lim
        """,
        params,
    )
    records = _frame_records(frame)
    summary = {
        "total": len(records),
        "graded": sum(1 for r in records if r.get("graded")),
        "wins": sum(1 for r in records if r.get("success") is True),
        "losses": sum(1 for r in records if r.get("success") is False),
    }
    return _json_response({"recommendations": records, "summary": summary})


_load_persisted_update_jobs()