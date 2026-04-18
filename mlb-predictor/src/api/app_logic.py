from __future__ import annotations

import contextlib
import importlib
import io
import json
import math
import os
import subprocess
import sys
import threading
import traceback
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from fastapi import Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse

from src.api.constants import (
    DOCTOR_FILE,
    EXPERIMENTS_FILE,
    FAVICON_FILE,
    GAME_BOARD_UI_DEFAULT_CONFIRMED_ONLY,
    GAME_BOARD_UI_DEFAULT_HIT_LIMIT_PER_TEAM,
    GAME_BOARD_UI_DEFAULT_INCLUDE_INFERRED,
    GAME_BOARD_UI_DEFAULT_MIN_HIT_PROBABILITY,
    GAME_FILE,
    HOT_HITTERS_FILE,
    HTML_SHELL_HEADERS,
    INDEX_FILE,
    MATCHUP_H2H_ADEQUATE_MIN_GAMES,
    MATCHUP_H2H_STRONG_MIN_GAMES,
    PITCHERS_FILE,
    RESULTS_FILE,
    TOTALS_FILE,
    UPDATE_MODULE_MAINS,
)
from src.api.dialect_sql import (
    DB_DIALECT,
    _sql_bind_list,
    _sql_boolean,
    _sql_integer,
    _sql_json_text,
    _sql_order_nulls_last,
    _sql_ratio,
    _sql_real,
    _sql_year,
    _sql_year_param,
)
from src.api.schemas import (
    PipelineRunRequest,
    UpdateAction,
)
from src.api.update_job_sequences import (
    UPDATE_JOB_ACTION_KEYS,
    build_pipeline_run_sequence,
    build_update_job_sequence,
    label_for_update_action,
)
from src.utils import best_bets as best_bets_utils
from src.utils.bvp_lookup import fetch_batter_vs_pitcher_map as _fetch_batter_vs_pitcher_map_core
from src.utils.hr_source_recs_db import load_hr_source_recs_for_date
from src.utils.slugger_hr_selection import (
    SLUGGER_DAILY_RESULTS_MAX_CARDS,
    SLUGGER_HR_PER_GAME,
    SLUGGER_HOT_HITTERS_PAGE_MAX_CARDS,
    iter_slugger_tracked_cards,
)
from src.utils.hitter_form import HOT_HITTER_PAGE_FORM_KEYS, classify_hitter_form
from src.utils.top_ev_pick import collect_top_ev_candidates, select_top_weighted_ev_pick
from src.utils.input_trust import input_trust_from_certainty as _input_trust_from_certainty
from src.utils.pregame_lock import is_before_scheduled_first_pitch, is_pregame_ingest_locked
from src.utils.matchup_keys import team_abbr_to_opponent_id
from src.utils.db import get_dialect_name, query_df, run_sql, table_exists, upsert_rows
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)
settings = get_settings()


def _pipeline_step_timeout_seconds() -> float | None:
    """Per update-job subprocess step. Default 7200s (2h). Set MLB_PIPELINE_STEP_TIMEOUT_SEC=0 for no limit."""
    raw = (os.environ.get("MLB_PIPELINE_STEP_TIMEOUT_SEC") or "7200").strip()
    if raw == "0":
        return None
    try:
        return float(raw)
    except ValueError:
        log.warning("Invalid MLB_PIPELINE_STEP_TIMEOUT_SEC=%r — using default 7200", raw)
        return 7200.0


_update_job_sequence = build_update_job_sequence
_pipeline_sequence = build_pipeline_run_sequence


def _update_job_label(action: UpdateAction) -> str:
    return label_for_update_action(action)

SUPPLEMENTAL_GAME_MARKET_TYPES = best_bets_utils.SUPPLEMENTAL_GAME_MARKET_TYPES
BEST_BET_MARKET_KEYS = best_bets_utils.BEST_BET_MARKET_KEYS
BEST_BET_SELECTION_LIMIT_PER_GAME = best_bets_utils.BEST_BET_SELECTION_LIMIT_PER_GAME
BEST_BET_THRESHOLD_MAP = best_bets_utils.BEST_BET_THRESHOLD_MAP
MARKET_SIM_MAX_RUNS = best_bets_utils.MARKET_SIM_MAX_RUNS
EXPERIMENTAL_CAROUSEL_MARKETS = ("nrfi", "yrfi")
# Lanes from ``prediction_outcomes_daily`` surfaced in Daily Results “best picks” (green / watchlist).
# Hitter props (``hits`` lane, 1+ hits, HR) are not tracked there — use matchup / hitter sections instead.
# ``totals`` / ``first5`` are legacy market strings from ``_build_prediction_outcomes``; best-bet snapshots
# use ``game_total`` / ``first_five_*`` keys — both families can coexist in the same table.
TRACKED_RECOMMENDATION_MARKETS = (
    BEST_BET_MARKET_KEYS
    + EXPERIMENTAL_CAROUSEL_MARKETS
    + (
        "pitcher_strikeouts",
        "totals",
        "first5",
    )
)


def _daily_results_excluded_from_team_best_picks(market: str | None) -> bool:
    """True for markets that must not appear in Daily Results green / watchlist pick lists."""
    m = str(market or "")
    if m == "hits":
        return True
    return best_bets_utils.excluded_from_team_best_pick_board(m)


def _html_file_response(path: Path) -> FileResponse:
    return FileResponse(path, headers=HTML_SHELL_HEADERS)


# Re-entrant: ``_persist_update_jobs`` may run under an existing update-job critical section
# (e.g. sequence rebuild path); a plain ``Lock`` deadlocks forever (0 steps, job stuck running).
UPDATE_JOB_LOCK = threading.RLock()
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
            "Desktop historical data is incomplete. Rebuild predictions is blocked because the "
            f"SQLite database has no prior rows in {missing_labels} ({missing_tables}). "
            "Run 'Refresh Everything' first to populate the required history tables."
        ),
        "missing": missing,
        "row_counts": row_counts,
    }


def _action_blocker(action: UpdateAction, target_date: date) -> dict[str, Any] | None:
    if action not in {"rebuild_predictions", "retrain_models"}:
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
        WITH latest_snapshots AS (
            SELECT
                game_id,
                team,
                player_id,
                DENSE_RANK() OVER (
                    PARTITION BY game_id, team
                    ORDER BY snapshot_ts DESC
                ) AS snapshot_rank
            FROM lineups
            WHERE game_date = :target_date
              AND game_id IS NOT NULL
              AND player_id IS NOT NULL
        )
        SELECT DISTINCT game_id, player_id
        FROM latest_snapshots
        WHERE snapshot_rank = 1
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

    def _sort_lineup_records(lineup_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            list(lineup_records),
            key=lambda record: (
                record.get("lineup_slot") is None,
                record.get("lineup_slot") if record.get("lineup_slot") is not None else 999,
                str(record.get("player_name") or ""),
            ),
        )

    if confirmed_records:
        displayed_records = _sort_lineup_records(confirmed_records)
        scope = "confirmed"
    elif snapshot_records:
        displayed_records = _sort_lineup_records(snapshot_records)
        scope = "snapshot"
    elif records:
        displayed_records = _sort_lineup_records(records)
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


def _fetch_boxscore_primary_starter_map(
    target_date: date,
    *,
    game_id: int | None = None,
) -> dict[tuple[int, str], dict[str, Any]]:
    """(game_id, team) -> starter row from box score (``is_starter``), with ``pitcher_starts`` joined when present."""
    if not _table_exists("player_game_pitching"):
        return {}
    game_filter = ""
    params: dict[str, Any] = {"target_date": target_date}
    if game_id is not None:
        game_filter = " AND p.game_id = :game_id"
        params["game_id"] = int(game_id)

    ps_join = ""
    if _table_exists("pitcher_starts"):
        ps_join = """
            LEFT JOIN pitcher_starts s
              ON s.game_id = p.game_id
             AND s.game_date = p.game_date
             AND s.pitcher_id = p.player_id
        """
    prob_sql = "COALESCE(s.is_probable, FALSE)" if ps_join else "FALSE"
    rest_sql = "s.days_rest" if ps_join else "NULL"
    ip_sql = "COALESCE(s.ip, p.innings_pitched)" if ps_join else "p.innings_pitched"
    k_sql = "COALESCE(s.strikeouts, p.strikeouts)" if ps_join else "p.strikeouts"
    bb_sql = "COALESCE(s.walks, p.walks)" if ps_join else "p.walks"
    pc_sql = "COALESCE(s.pitch_count, p.pitches_thrown)" if ps_join else "p.pitches_thrown"
    xw_sql = "COALESCE(s.xwoba_against, p.xwoba_allowed)" if ps_join else "p.xwoba_allowed"
    csw_sql = "s.csw_pct" if ps_join else "NULL"
    velo_sql = "s.avg_fb_velo" if ps_join else "NULL"
    whiff_sql = "s.whiff_pct" if ps_join else "NULL"

    starter_filter = "COALESCE(CAST(p.is_starter AS INTEGER), 0) != 0"
    if DB_DIALECT == "sqlite":
        starter_filter = "(p.is_starter = 1 OR p.is_starter = TRUE OR p.is_starter = 'true')"

    frame = _safe_frame(
        f"""
        WITH prim AS (
            SELECT
                p.game_id,
                p.team,
                p.player_id AS pitcher_id,
                COALESCE(dp.full_name, CAST(p.player_id AS TEXT)) AS pitcher_name,
                dp.throws,
                {prob_sql} AS is_probable,
                {rest_sql} AS days_rest,
                {ip_sql} AS ip,
                {k_sql} AS strikeouts,
                {bb_sql} AS walks,
                {pc_sql} AS pitch_count,
                {xw_sql} AS xwoba_against,
                {csw_sql} AS csw_pct,
                {velo_sql} AS avg_fb_velo,
                {whiff_sql} AS whiff_pct,
                ROW_NUMBER() OVER (
                    PARTITION BY p.game_id, p.team
                    ORDER BY COALESCE(p.innings_pitched, 0) DESC, p.player_id
                ) AS row_rank
            FROM player_game_pitching p
            LEFT JOIN dim_players dp ON dp.player_id = p.player_id
            {ps_join}
            WHERE p.game_date = :target_date
              AND {starter_filter}
              {game_filter}
        )
        SELECT
            game_id,
            team,
            pitcher_id,
            pitcher_name,
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
        FROM prim
        WHERE row_rank = 1
        """,
        params,
    )
    out: dict[tuple[int, str], dict[str, Any]] = {}
    for row in _frame_records(frame):
        gid = row.get("game_id")
        team = row.get("team")
        if gid is None or team is None:
            continue
        out[(int(gid), str(team).strip())] = row
    return out


def _starter_records_prefer_boxscore(
    ranked_records: list[dict[str, Any]],
    target_date: date,
    *,
    game_id: int | None = None,
) -> list[dict[str, Any]]:
    """When box score rows exist for ``is_starter``, use that pitcher instead of probable-only ranking."""
    box_map = _fetch_boxscore_primary_starter_map(target_date, game_id=game_id)
    if not box_map:
        return ranked_records
    merged: list[dict[str, Any]] = []
    for row in ranked_records:
        key = (int(row["game_id"]), str(row.get("team") or "").strip())
        if key in box_map:
            box_row = box_map[key]
            merged.append({**row, **box_row})
        else:
            merged.append(row)
    return merged


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


def _live_hit_streak(hit_history: list[dict[str, Any]]) -> int:
    """Compute current consecutive-game hit streak from live batting history.

    ``hit_history`` must be ordered most-recent-first (the default from
    ``_fetch_recent_hit_history_map``).
    """
    streak = 0
    for entry in hit_history:
        if int(entry.get("hits") or 0) > 0:
            streak += 1
        else:
            break
    return streak


def _row_hit_streak_value(row: dict[str, Any]) -> int:
    """Aligned with hot-hitters UI: max of recent game-log chain and streak fields."""
    hist = row.get("recent_hit_history") or []
    from_hist = _live_hit_streak(hist) if hist else 0
    from_api = max(
        int(row.get("streak_len") or 0),
        int(row.get("streak_len_capped") or 0),
    )
    return max(from_hist, from_api)


def _overlay_live_batting_stats(
    record: dict[str, Any],
    hit_history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Replace potentially-stale feature-payload stats with live values.

    Overrides ``streak_len``, ``streak_len_capped``, ``hit_rate_7``, and
    ``hit_rate_30`` using the live ``hit_history`` (already fetched from
    ``player_game_batting`` for display) and the ``games_last7`` /
    ``hit_games_last7`` columns from the ``recent_batting`` CTE.
    """
    if hit_history:
        live_streak = _live_hit_streak(hit_history)
        record["streak_len"] = live_streak
        record["streak_len_capped"] = min(live_streak, 5)

    games_last7 = int(record.get("games_last7") or 0)
    hit_games_last7 = int(record.get("hit_games_last7") or 0)
    if games_last7 > 0:
        record["hit_rate_7"] = hit_games_last7 / games_last7

    # Approximate 30-game hit rate from the full hit_history when available.
    if hit_history:
        games_with_hits = sum(1 for e in hit_history if int(e.get("hits") or 0) > 0)
        record["hit_rate_30"] = games_with_hits / len(hit_history)

    return record


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


def _fetch_batter_vs_pitcher_map(
    target_date: date,
    matchups: list[tuple[int, int]],
) -> dict[tuple[int, int], dict[str, Any]]:
    """Thin wrapper — implementation lives in ``src.utils.bvp_lookup``."""
    return _fetch_batter_vs_pitcher_map_core(target_date, matchups)


def _compact_bvp_vs_pitcher_payload(
    stats: dict[str, Any] | None,
    *,
    pitcher_name: str,
) -> dict[str, Any]:
    """Shape BvP stats for game detail cards (vs today's opposing starter)."""
    payload: dict[str, Any] = {
        "opposing_pitcher_name": pitcher_name or None,
        "has_sample": False,
        "source": None,
    }
    if not stats:
        return payload
    ab = int(stats.get("at_bats") or 0)
    payload["source"] = stats.get("source")
    if ab <= 0:
        return payload
    payload["has_sample"] = True
    payload["at_bats"] = ab
    payload["hits"] = int(stats.get("hits") or 0)
    payload["home_runs"] = int(stats.get("home_runs") or 0)
    payload["strikeouts"] = int(stats.get("strikeouts") or 0)
    payload["rbi"] = int(stats.get("rbi") or 0)
    payload["batting_avg"] = stats.get("batting_avg")
    return payload


def _fetch_pitcher_vs_team_matchup_row(
    pitcher_id: int | None,
    opponent_team_abbr: str,
) -> dict[str, Any] | None:
    """Career pitcher-vs-opponent-team row from matchup_splits (StatMuse)."""
    if pitcher_id is None or not str(opponent_team_abbr).strip() or not _table_exists("matchup_splits"):
        return None
    oid = team_abbr_to_opponent_id(opponent_team_abbr)
    frame = _safe_frame(
        """
        SELECT games, era, strikeouts, innings_pitched, earned_runs, walks, hits, whip, k_per_9
        FROM matchup_splits
        WHERE player_id = :pid
          AND split_type = 'pitcher_vs_team'
          AND season = 0
          AND opponent_id = :oid
        LIMIT 1
        """,
        {"pid": int(pitcher_id), "oid": oid},
    )
    if frame.empty:
        return None
    row = _frame_records(frame)[0]
    return {
        "opponent_team_abbr": str(opponent_team_abbr).strip(),
        "games": row.get("games"),
        "era": row.get("era"),
        "strikeouts": row.get("strikeouts"),
        "innings_pitched": row.get("innings_pitched"),
        "walks": row.get("walks"),
        "hits": row.get("hits"),
        "whip": row.get("whip"),
        "k_per_9": row.get("k_per_9"),
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
                b.runs,
                b.rbi,
                b.walks,
                b.stolen_bases,
                b.strikeouts,
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
            runs,
            rbi,
            walks,
            stolen_bases,
            strikeouts,
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
                "runs": int(row.get("runs") or 0),
                "rbi": int(row.get("rbi") or 0),
                "walks": int(row.get("walks") or 0),
                "stolen_bases": int(row.get("stolen_bases") or 0),
                "strikeouts": int(row.get("strikeouts") or 0),
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


PLAYER_STATUS_DEFAULTS = {
    "availability_bucket": None,
    "is_active_roster": None,
    "is_available": None,
    "is_injured": None,
    "roster_status_code": None,
    "roster_status_description": None,
    "roster_status_note": None,
    "roster_snapshot_ts": None,
}


def _fetch_player_status_map(
    target_date: date,
    rows: list[dict[str, Any]],
) -> dict[tuple[int, str], dict[str, Any]]:
    if not rows or not _table_exists("player_status_daily"):
        return {}

    params: dict[str, Any] = {"target_date": target_date}
    player_ids = sorted(
        {
            int(row["player_id"])
            for row in rows
            if row.get("player_id") is not None
        }
    )
    teams = sorted(
        {
            str(row["team"]).strip().upper()
            for row in rows
            if row.get("team")
        }
    )
    if not player_ids or not teams:
        return {}

    player_placeholders = _sql_bind_list("status_player_id", player_ids, params)
    team_placeholders = _sql_bind_list("status_team", teams, params)
    frame = _safe_frame(
        f"""
        WITH ranked AS (
            SELECT
                ps.player_id,
                UPPER(ps.team) AS team,
                ps.availability_bucket,
                ps.is_active_roster,
                ps.is_available,
                ps.is_injured,
                ps.status_code,
                ps.status_description,
                ps.status_note,
                ps.snapshot_ts,
                ROW_NUMBER() OVER (
                    PARTITION BY ps.player_id, UPPER(ps.team)
                    ORDER BY ps.snapshot_ts DESC
                ) AS row_rank
            FROM player_status_daily ps
            WHERE ps.game_date = :target_date
              AND ps.player_id IN ({player_placeholders})
              AND UPPER(ps.team) IN ({team_placeholders})
        )
        SELECT
            player_id,
            team,
            availability_bucket,
            is_active_roster,
            is_available,
            is_injured,
            status_code,
            status_description,
            status_note,
            snapshot_ts
        FROM ranked
        WHERE row_rank = 1
        """,
        params,
    )
    if frame.empty:
        return {}

    status_map: dict[tuple[int, str], dict[str, Any]] = {}
    for record in _frame_records(frame):
        status_map[(int(record["player_id"]), str(record["team"]).strip().upper())] = {
            "availability_bucket": record.get("availability_bucket"),
            "is_active_roster": None if record.get("is_active_roster") is None else bool(record.get("is_active_roster")),
            "is_available": None if record.get("is_available") is None else bool(record.get("is_available")),
            "is_injured": None if record.get("is_injured") is None else bool(record.get("is_injured")),
            "roster_status_code": record.get("status_code"),
            "roster_status_description": record.get("status_description"),
            "roster_status_note": record.get("status_note"),
            "roster_snapshot_ts": record.get("snapshot_ts"),
        }
    return status_map


def _attach_player_status_context(
    player: dict[str, Any],
    status_map: dict[tuple[int, str], dict[str, Any]],
) -> dict[str, Any]:
    player_id = player.get("player_id")
    team = str(player.get("team") or "").strip().upper()
    status = None
    if player_id is not None and team:
        status = status_map.get((int(player_id), team))
    if status is None and player_id is not None:
        matching = [
            record
            for (status_player_id, _), record in status_map.items()
            if status_player_id == int(player_id)
        ]
        if len(matching) == 1:
            status = matching[0]
    player.update(status or PLAYER_STATUS_DEFAULTS)
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
        consensus_line = round(float(pd.Series(line_values).median()), 2) if line_values else None
        # Prices must come from rows at the *same* K line as ``consensus_line``. Otherwise
        # ``best_*_price`` was max across alternate lines (e.g. +235 at 4.5 K) while the UI
        # shows median line 5.5 — mismatched vs a single book like FanDuel.
        line_tol = 0.05
        if consensus_line is not None and line_values:
            line_rows = [
                row
                for row in rows
                if row.get("line_value") is not None
                and abs(float(row["line_value"]) - float(consensus_line)) <= line_tol
            ]
        else:
            line_rows = list(rows)
        if not line_rows:
            line_rows = list(rows)
        over_prices = [int(row["over_price"]) for row in line_rows if row.get("over_price") is not None]
        under_prices = [int(row["under_price"]) for row in line_rows if row.get("under_price") is not None]
        market_map[key] = {
            "market_type": "pitcher_strikeouts",
            "player_name": next((row.get("player_name") for row in rows if row.get("player_name")), None),
            "team": next((row.get("team") for row in rows if row.get("team")), None),
            "consensus_line": consensus_line,
            "line_min": min(line_values) if line_values else None,
            "line_max": max(line_values) if line_values else None,
            "best_over_price": max(over_prices) if over_prices else None,
            "best_under_price": max(under_prices) if under_prices else None,
            "sportsbook_count": len(line_rows),
            "sportsbooks": [str(row.get("sportsbook")) for row in line_rows if row.get("sportsbook")],
            "source_names": sorted({str(row.get("source_name")) for row in rows if row.get("source_name")}),
            "latest_snapshot_ts": max((row.get("snapshot_ts") for row in rows if row.get("snapshot_ts") is not None), default=None),
        }
    return market_map


def _primary_strikeout_projection_for_market(projection: dict[str, Any]) -> float | None:
    """K total used for model-vs-line delta and ``display_projected_strikeouts``.

    Order matches ``pitchers.html`` (public → fundamentals → calibrated). We take the
    first finite **non-negative** value so a bad calibrated row does not override a
    sane fundamentals/public number or produce nonsense deltas.
    """
    for key in (
        "public_projected_strikeouts",
        "projected_strikeouts_fundamentals",
        "projected_strikeouts",
    ):
        value = _to_float(projection.get(key))
        if value is None or not math.isfinite(value):
            continue
        if value >= 0:
            return float(value)
    return None


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
    primary_k = _primary_strikeout_projection_for_market(merged)
    merged["display_projected_strikeouts"] = primary_k
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
        "projection_delta": None if primary_k is None or market_line is None else round(primary_k - market_line, 2),
    }
    return merged


def _strikeout_probability_sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _derive_strikeout_probability_std(
    calibrated_projection: float | None,
    market_line: float | None,
    over_probability: float | None,
    under_probability: float | None,
) -> float | None:
    cal = _to_float(calibrated_projection)
    line = _to_float(market_line)
    over = _to_float(over_probability)
    under = _to_float(under_probability)
    if cal is None or line is None:
        return None
    if over is None and under is not None:
        over = 1.0 - under
    if over is None or not 0.0 < over < 1.0:
        return None
    delta = cal - line
    if abs(delta) < 1e-9:
        return None
    clipped = min(max(over, 1e-4), 1.0 - 1e-4)
    logit = math.log(clipped / (1.0 - clipped))
    if abs(logit) < 1e-9:
        return None
    std = delta / logit
    if not math.isfinite(std) or std <= 0:
        return None
    return float(std)


def _detect_public_strikeout_slate_collapse(
    prediction_map: dict[tuple[int, int], dict[str, Any]],
) -> str | None:
    paired: list[tuple[float, float, float]] = []
    for projection in prediction_map.values():
        calibrated = _to_float(projection.get("projected_strikeouts"))
        fundamentals = _to_float(projection.get("projected_strikeouts_fundamentals"))
        market_line = _to_float((projection.get("market") or {}).get("consensus_line"))
        if calibrated is None or fundamentals is None or market_line is None:
            continue
        paired.append((calibrated, fundamentals, market_line))

    if len(paired) < 8:
        return None

    calibrated_values = [calibrated for calibrated, _, _ in paired]
    fundamentals_values = [fundamentals for _, fundamentals, _ in paired]
    calibrated_sides = ["over" if calibrated >= market_line else "under" for calibrated, _, market_line in paired]
    fundamentals_sides = ["over" if fundamentals >= market_line else "under" for _, fundamentals, market_line in paired]
    calibrated_dominant_fraction = max(calibrated_sides.count("over"), calibrated_sides.count("under")) / len(calibrated_sides)
    fundamentals_dominant_fraction = max(fundamentals_sides.count("over"), fundamentals_sides.count("under")) / len(fundamentals_sides)
    rounded_counts = pd.Series([round(value, 2) for value in calibrated_values]).value_counts()
    largest_bucket = int(rounded_counts.iloc[0]) if not rounded_counts.empty else 0
    unique_bucket_count = int(len(rounded_counts))
    calibrated_std = _stddev(calibrated_values)
    fundamentals_std = _stddev(fundamentals_values)

    if (
        calibrated_dominant_fraction >= 0.9
        and fundamentals_dominant_fraction < calibrated_dominant_fraction
        and largest_bucket >= 3
        and unique_bucket_count <= max(6, len(paired) // 2)
    ):
        return (
            "public_fallback_one_sided_"
            f"{calibrated_dominant_fraction:.0%}_bucket_{largest_bucket}"
        )

    if (
        calibrated_dominant_fraction >= 0.95
        and calibrated_std < max(0.55, fundamentals_std * 0.55)
    ):
        return (
            "public_fallback_low_variance_"
            f"std_{calibrated_std:.2f}_vs_{fundamentals_std:.2f}"
        )

    return None


def _apply_public_strikeout_projection_fallback(
    prediction_map: dict[tuple[int, int], dict[str, Any]],
) -> dict[tuple[int, int], dict[str, Any]]:
    collapse_reason = _detect_public_strikeout_slate_collapse(prediction_map)
    if not collapse_reason:
        return prediction_map

    log.warning("Public strikeout slate fallback activated: %s", collapse_reason)
    default_std = 1.35
    for projection in prediction_map.values():
        fundamentals = _to_float(projection.get("projected_strikeouts_fundamentals"))
        calibrated = _to_float(projection.get("projected_strikeouts"))
        market = dict(projection.get("market") or {})
        market_line = _to_float(market.get("consensus_line"))
        if fundamentals is None:
            continue

        projection["public_projection_mode"] = "fundamentals_fallback"
        projection["public_projection_reason"] = collapse_reason
        projection["public_projected_strikeouts"] = fundamentals

        if market_line is None:
            continue

        effective_std = _derive_strikeout_probability_std(
            calibrated,
            market_line,
            projection.get("over_probability"),
            projection.get("under_probability"),
        )
        effective_std = max(float(effective_std or default_std), 0.35)
        public_over_probability = _strikeout_probability_sigmoid((fundamentals - market_line) / effective_std)
        projection["public_over_probability"] = public_over_probability
        projection["public_under_probability"] = 1.0 - public_over_probability

    return prediction_map


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
            CAST(NULLIF(f.feature_payload ->> 'throws', '') AS TEXT) AS throws,
            CAST(NULLIF(f.feature_payload ->> 'season_starts', '') AS INTEGER) AS season_starts,
            CAST(NULLIF(f.feature_payload ->> 'season_innings', '') AS DOUBLE PRECISION) AS season_innings,
            CAST(NULLIF(f.feature_payload ->> 'season_strikeouts', '') AS INTEGER) AS season_strikeouts,
            CAST(NULLIF(f.feature_payload ->> 'season_k_per_start', '') AS DOUBLE PRECISION) AS season_k_per_start,
            CAST(NULLIF(f.feature_payload ->> 'season_k_per_batter', '') AS DOUBLE PRECISION) AS season_k_per_batter,
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
            p.predicted_strikeouts_fundamentals,
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
            "throws": row.get("throws"),
            "prediction_ts": row.get("prediction_ts"),
            "model_name": row.get("model_name"),
            "model_version": row.get("model_version"),
            "projected_strikeouts": row.get("predicted_strikeouts"),
            "projected_strikeouts_fundamentals": row.get("predicted_strikeouts_fundamentals"),
            "baseline_strikeouts": row.get("baseline_strikeouts"),
            "season_context": {
                "starts": row.get("season_starts"),
                "innings": row.get("season_innings"),
                "strikeouts": row.get("season_strikeouts"),
                "strikeouts_per_start": row.get("season_k_per_start"),
                "strikeouts_per_batter": row.get("season_k_per_batter"),
            },
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
    return _apply_public_strikeout_projection_fallback(prediction_map)


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
    hr_count = 0
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
    if _table_exists("predictions_player_hr"):
        hr_rows = _safe_frame(
            "SELECT COUNT(*) AS row_count FROM predictions_player_hr WHERE game_date = :target_date",
            {"target_date": target_date},
        )
        if not hr_rows.empty:
            hr_count = int(hr_rows.iloc[0]["row_count"])
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
        "hr_artifact_ready": _artifact_ready("hr"),
        "strikeouts_artifact_ready": _artifact_ready("strikeouts"),
        "totals_predictions": totals_count,
        "hits_predictions": hits_count,
        "hr_predictions": hr_count,
        "strikeouts_predictions": strikeouts_count,
        "rebuild_blocker": rebuild_blocker,
        "tables": {
            "games": _table_exists("games"),
            "game_features_totals": _table_exists("game_features_totals"),
            "player_features_hits": _table_exists("player_features_hits"),
            "player_features_hr": _table_exists("player_features_hr"),
            "game_features_pitcher_strikeouts": _table_exists("game_features_pitcher_strikeouts"),
            "predictions_totals": _table_exists("predictions_totals"),
            "predictions_player_hits": _table_exists("predictions_player_hits"),
            "predictions_player_hr": _table_exists("predictions_player_hr"),
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


def _doctor_experimental_markets_snapshot(target_date: date) -> dict[str, Any]:
    """Summarize NRFI/YRFI rows in ``game_markets`` for Doctor (first-inning Odds API ingest)."""
    snapshot: dict[str, Any] = {
        "table_present": bool(_table_exists("game_markets")),
        "games_on_slate": 0,
        "distinct_games_with_lines": 0,
        "total_market_rows": 0,
        "by_market_type": {},
        "note": (
            "NRFI/YRFI rows are stored when the Odds API returns totals_1st_1_innings on the "
            "player-prop / first-five market pull (see src/ingestors/market_totals._extract_odds_api_event_first5_rows). "
            "They are experimental tracking lines, not green-board best bets."
        ),
    }
    if not snapshot["table_present"]:
        return snapshot

    if _table_exists("games"):
        gframe = _safe_frame(
            "SELECT COUNT(*) AS n FROM games WHERE game_date = :d",
            {"d": target_date.isoformat()},
        )
        if not gframe.empty:
            snapshot["games_on_slate"] = int(gframe.iloc[0]["n"] or 0)

    placeholders = ", ".join(f":m{i}" for i in range(len(EXPERIMENTAL_CAROUSEL_MARKETS)))
    params: dict[str, Any] = {"d": target_date.isoformat()}
    for i, key in enumerate(EXPERIMENTAL_CAROUSEL_MARKETS):
        params[f"m{i}"] = key

    mframe = _safe_frame(
        f"""
        SELECT market_type, COUNT(*) AS row_count, COUNT(DISTINCT game_id) AS games_with_line
        FROM game_markets
        WHERE game_date = :d AND market_type IN ({placeholders})
        GROUP BY market_type
        """,
        params,
    )
    total_rows = 0
    for row in _frame_records(mframe):
        mt = str(row.get("market_type") or "")
        rc = int(row.get("row_count") or 0)
        gw = int(row.get("games_with_line") or 0)
        total_rows += rc
        snapshot["by_market_type"][mt] = {"rows": rc, "games_with_line": gw}
    snapshot["total_market_rows"] = total_rows

    dframe = _safe_frame(
        f"""
        SELECT COUNT(DISTINCT game_id) AS n
        FROM game_markets
        WHERE game_date = :d AND market_type IN ({placeholders})
        """,
        params,
    )
    if not dframe.empty:
        snapshot["distinct_games_with_lines"] = int(dframe.iloc[0]["n"] or 0)

    return snapshot


def _doctor_board_action_score_payload() -> dict[str, Any]:
    """Learning overlay (``train_board_action_score``): Act % on cards; does not gate green-strip inclusion."""
    try:
        from src.models.board_action_score import artifact_paths
    except Exception as exc:
        return {"artifact_exists": False, "error": str(exc)}
    path, meta_path = artifact_paths()
    out: dict[str, Any] = {
        "artifact_path": str(path),
        "meta_path": str(meta_path),
        "artifact_exists": bool(path.exists()),
        "model_label": "board_action_logistic_v1",
        "role": (
            "Adds action_score to team best-bet cards in JSON/UI when the joblib loads. "
            "Strip membership still uses EV + soft gates (best_bets.qualifies_board_green_strip)."
        ),
    }
    if path.exists():
        try:
            st = path.stat()
            out["artifact_mtime_utc"] = (
                datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat()
            )
            out["artifact_bytes"] = int(st.st_size)
        except OSError as exc:
            out["stat_error"] = str(exc)
    return out


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
    experimental_snapshot = _doctor_experimental_markets_snapshot(target_date)
    board_action_score = _doctor_board_action_score_payload()
    games_on_slate = int(experimental_snapshot.get("games_on_slate") or 0)
    experimental_rows = int(experimental_snapshot.get("total_market_rows") or 0)
    experimental_ok = games_on_slate == 0 or experimental_rows > 0
    if games_on_slate == 0:
        experimental_detail = "No games on the slate for this date."
    elif experimental_rows > 0:
        experimental_detail = (
            f"Found {experimental_rows} NRFI/YRFI row(s) in game_markets covering "
            f"{int(experimental_snapshot.get('distinct_games_with_lines') or 0)} distinct game(s)."
        )
    else:
        experimental_detail = (
            "Slate has games but no nrfi/yrfi rows in game_markets—Odds API may not return "
            "totals_1st_1_innings for these events, or the market ingest has not run successfully."
        )

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
        _doctor_check(
            "experimental_first_inning_markets",
            "NRFI/YRFI market lines",
            experimental_ok,
            "warning",
            experimental_detail,
        ),
        _doctor_check(
            "board_action_score",
            "Board action score (learning overlay)",
            bool(board_action_score.get("artifact_exists")),
            "warning",
            (
                f"Artifact present ({board_action_score.get('artifact_bytes')} bytes, mtime "
                f"{board_action_score.get('artifact_mtime_utc')}). Cards may show Act % / Action %."
                if board_action_score.get("artifact_exists")
                else (
                    "No action_classifier.joblib — train with Retrain Models or "
                    "`python -m src.models.train_board_action_score`. Strip picks still use EV gates only."
                )
            ),
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
        "experimental_markets": experimental_snapshot,
        "board_action_score": board_action_score,
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


_MIN_START_ROWS_FOR_START_ONLY_RECENT_FORM = 3


def _recent_outings_from_player_game_pitching(
    pitcher_id: int,
    target_date: date,
    limit: int,
) -> list[dict[str, Any]]:
    if not _table_exists("player_game_pitching"):
        return []
    history_frame = _safe_frame(
        """
        SELECT
            p.game_date,
            p.game_id,
            p.team,
            p.innings_pitched AS ip,
            p.earned_runs,
            p.strikeouts,
            p.walks,
            p.pitches_thrown AS pitch_count,
            CASE
                WHEN g.home_team = p.team THEN g.away_team
                WHEN g.away_team = p.team THEN g.home_team
                ELSE NULL
            END AS opponent
        FROM player_game_pitching p
        LEFT JOIN games g
          ON g.game_id = p.game_id
         AND g.game_date = p.game_date
        WHERE p.player_id = :pitcher_id
          AND p.game_date < :target_date
        ORDER BY p.game_date DESC, p.game_id DESC
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
            "earned_runs": int(row.get("earned_runs") or 0)
            if row.get("earned_runs") is not None
            else None,
            "strikeouts": int(row.get("strikeouts") or 0),
            "pitch_count": int(row.get("pitch_count") or 0)
            if row.get("pitch_count") is not None
            else None,
        }
        for row in _frame_records(history_frame)
    ]


def _fetch_pitcher_recent_starts(
    pitcher_id: int | None,
    target_date: date,
    limit: int = 5,
    *,
    outing_source: str | None = None,
) -> list[dict[str, Any]]:
    """Return recent pitching lines. ``outing_source`` ``\"starts\"`` / ``\"appearances\"`` forces a table; ``None`` uses starts then falls back to all appearances."""
    if pitcher_id is None:
        return []

    if outing_source == "appearances":
        return _recent_outings_from_player_game_pitching(int(pitcher_id), target_date, limit)

    if _table_exists("pitcher_starts"):
        history_frame = _safe_frame(
            """
            SELECT
                ps.game_date,
                ps.game_id,
                ps.team,
                ps.ip,
                ps.earned_runs,
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
        rows = [
            {
                "game_date": row.get("game_date"),
                "game_id": row.get("game_id"),
                "team": row.get("team"),
                "opponent": row.get("opponent"),
                "ip": _coerce_float(row.get("ip")),
                "earned_runs": int(row.get("earned_runs") or 0)
                if row.get("earned_runs") is not None
                else None,
                "strikeouts": int(row.get("strikeouts") or 0),
                "pitch_count": int(row.get("pitch_count") or 0)
                if row.get("pitch_count") is not None
                else None,
            }
            for row in _frame_records(history_frame)
        ]
        if rows or outing_source == "starts":
            return rows

    return _recent_outings_from_player_game_pitching(int(pitcher_id), target_date, limit)


def _baseball_ip_to_outs(ip_value: Any) -> int:
    innings = _coerce_float(ip_value)
    if innings is None or math.isnan(innings):
        return 0
    whole_innings = int(innings)
    partial_innings = round(innings - whole_innings, 1)
    partial_outs = 0
    if abs(partial_innings - 0.1) < 0.05:
        partial_outs = 1
    elif abs(partial_innings - 0.2) < 0.05:
        partial_outs = 2
    return max((whole_innings * 3) + partial_outs, 0)


def _baseball_ip_from_outs(outs: Any) -> float:
    outs_value = int(_coerce_float(outs) or 0)
    whole_innings, partial_outs = divmod(max(outs_value, 0), 3)
    return float(f"{whole_innings}.{partial_outs}")


def _era_from_pitcher_history(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    outs_recorded = pd.to_numeric(frame.get("ip"), errors="coerce").apply(_baseball_ip_to_outs).sum()
    if outs_recorded <= 0:
        return None
    earned_runs = pd.to_numeric(frame.get("earned_runs"), errors="coerce").fillna(0).sum()
    return float(earned_runs) * 27.0 / float(outs_recorded)


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
            r.predicted_total_fundamentals,
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
        ORDER BY {game_start_order}, away_team, home_team, r.game_id
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
          AND UPPER(COALESCE(dp.position, '')) NOT IN ('P', 'SP', 'RP', 'CP')
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
            g.venue_id,
            v.latitude AS venue_latitude,
            v.longitude AS venue_longitude,
            COALESCE(v.venue_name, g.venue_name) AS venue_name,
            v.city AS venue_city,
            v.state AS venue_state,
            v.roof_type,
            p.model_name,
            p.model_version,
            p.prediction_ts,
            p.predicted_total_runs,
            p.predicted_total_fundamentals,
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
            CAST(f.feature_payload ->> 'away_bullpen_late_innings_last3' AS DOUBLE PRECISION) AS away_bullpen_late_innings_last3,
            CAST(f.feature_payload ->> 'home_bullpen_late_innings_last3' AS DOUBLE PRECISION) AS home_bullpen_late_innings_last3,
            CAST(f.feature_payload ->> 'away_bullpen_late_runs_allowed_last3' AS DOUBLE PRECISION) AS away_bullpen_late_runs_allowed_last3,
            CAST(f.feature_payload ->> 'home_bullpen_late_runs_allowed_last3' AS DOUBLE PRECISION) AS home_bullpen_late_runs_allowed_last3,
            CAST(f.feature_payload ->> 'away_bullpen_late_earned_runs_last3' AS DOUBLE PRECISION) AS away_bullpen_late_earned_runs_last3,
            CAST(f.feature_payload ->> 'home_bullpen_late_earned_runs_last3' AS DOUBLE PRECISION) AS home_bullpen_late_earned_runs_last3,
            CAST(f.feature_payload ->> 'away_bullpen_late_hits_allowed_last3' AS DOUBLE PRECISION) AS away_bullpen_late_hits_allowed_last3,
            CAST(f.feature_payload ->> 'home_bullpen_late_hits_allowed_last3' AS DOUBLE PRECISION) AS home_bullpen_late_hits_allowed_last3,
            CAST(f.feature_payload ->> 'away_bullpen_late_era_last3' AS DOUBLE PRECISION) AS away_bullpen_late_era_last3,
            CAST(f.feature_payload ->> 'home_bullpen_late_era_last3' AS DOUBLE PRECISION) AS home_bullpen_late_era_last3,
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
            CAST(f.feature_payload ->> 'precipitation_pct' AS DOUBLE PRECISION) AS precipitation_pct,
            CAST(f.feature_payload ->> 'cloud_cover_pct' AS DOUBLE PRECISION) AS cloud_cover_pct,
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
        ORDER BY {game_start_order}, g.away_team, g.home_team, g.game_id
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
    starter_records = _starter_records_prefer_boxscore(
        _frame_records(starters_frame),
        target_date,
    )

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
    player_status_map = _fetch_player_status_map(target_date, hit_records)
    hit_split_map = _fetch_hitter_pitch_hand_splits(
        target_date,
        [int(hit["player_id"]) for hit in hit_records if hit.get("player_id") is not None],
    )
    lineup_handedness_by_game = _fetch_lineup_handedness_by_game(target_date)
    pitcher_k_market_map = _fetch_pitcher_strikeout_market_map(target_date)
    pitcher_k_prediction_map = _fetch_pitcher_strikeout_prediction_map(target_date)
    first5_totals_map = _fetch_first5_totals_map(target_date)
    latest_game_market_rows = _fetch_latest_game_market_rows(target_date)
    supplemental_markets_map = _aggregate_supplemental_market_rows(latest_game_market_rows)
    market_rows_by_game: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for market_row in latest_game_market_rows:
        game_id = market_row.get("game_id")
        market_type = str(market_row.get("market_type") or "")
        if game_id is None or not market_type:
            continue
        market_rows_by_game[int(game_id)][market_type].append(market_row)
    umpire_map = _fetch_umpire_map(target_date)
    tb_prediction_map = _fetch_total_bases_prediction_map(target_date)
    pitcher_prop_map = _fetch_pitcher_prop_map(target_date)
    bullpen_context_map = _fetch_bullpen_context_map(target_date)
    recent_bullpen_map = _fetch_recent_bullpen_map(target_date)

    games_by_id: dict[int, dict[str, Any]] = {}
    for record in game_records:
        game_id = int(record["game_id"])
        is_final = _is_final_game_status(record.get("status"))
        away_recent_bullpen = recent_bullpen_map.get(str(record["away_team"]) or "", {}) or {}
        home_recent_bullpen = recent_bullpen_map.get(str(record["home_team"]) or "", {}) or {}
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
                "venue_id": record.get("venue_id"),
                "name": record["venue_name"],
                "city": record["venue_city"],
                "state": record["venue_state"],
                "roof_type": record["roof_type"],
                "latitude": record.get("venue_latitude"),
                "longitude": record.get("venue_longitude"),
            },
            "weather": {
                "temperature_f": record["temperature_f"],
                "wind_speed_mph": record["wind_speed_mph"],
                "wind_direction_deg": record["wind_direction_deg"],
                "humidity_pct": record["humidity_pct"],
                "precipitation_pct": record.get("precipitation_pct"),
                "cloud_cover_pct": record.get("cloud_cover_pct"),
            },
            "totals": {
                "model_name": record["model_name"],
                "model_version": record["model_version"],
                "prediction_ts": record["prediction_ts"],
                "predicted_total_runs": record["predicted_total_runs"],
                "predicted_total_fundamentals": record.get("predicted_total_fundamentals"),
                "market_total": record["market_total"],
                "over_probability": record["over_probability"],
                "under_probability": record["under_probability"],
                "edge": record["edge"],
                "confidence_level": record.get("confidence_level"),
                "suppress_reason": record.get("suppress_reason"),
                "lane_status": record.get("lane_status", "research_only"),
                "away_expected_runs": record["away_expected_runs"],
                "home_expected_runs": record["home_expected_runs"],
                "away_bullpen_pitches_last3": away_recent_bullpen.get("pitches_last3", record["away_bullpen_pitches_last3"]),
                "home_bullpen_pitches_last3": home_recent_bullpen.get("pitches_last3", record["home_bullpen_pitches_last3"]),
                "away_bullpen_innings_last3": away_recent_bullpen.get("innings_last3", record["away_bullpen_innings_last3"]),
                "home_bullpen_innings_last3": home_recent_bullpen.get("innings_last3", record["home_bullpen_innings_last3"]),
                "away_bullpen_b2b": away_recent_bullpen.get("b2b", record["away_bullpen_b2b"]),
                "home_bullpen_b2b": home_recent_bullpen.get("b2b", record["home_bullpen_b2b"]),
                "away_bullpen_runs_allowed_last3": away_recent_bullpen.get("runs_allowed_last3", record["away_bullpen_runs_allowed_last3"]),
                "home_bullpen_runs_allowed_last3": home_recent_bullpen.get("runs_allowed_last3", record["home_bullpen_runs_allowed_last3"]),
                "away_bullpen_earned_runs_last3": away_recent_bullpen.get("earned_runs_last3", record["away_bullpen_earned_runs_last3"]),
                "home_bullpen_earned_runs_last3": home_recent_bullpen.get("earned_runs_last3", record["home_bullpen_earned_runs_last3"]),
                "away_bullpen_hits_allowed_last3": away_recent_bullpen.get("hits_allowed_last3", record["away_bullpen_hits_allowed_last3"]),
                "home_bullpen_hits_allowed_last3": home_recent_bullpen.get("hits_allowed_last3", record["home_bullpen_hits_allowed_last3"]),
                "away_bullpen_era_last3": away_recent_bullpen.get("era_last3", record["away_bullpen_era_last3"]),
                "home_bullpen_era_last3": home_recent_bullpen.get("era_last3", record["home_bullpen_era_last3"]),
                "away_bullpen_late_innings_last3": away_recent_bullpen.get("late_innings_last3", record["away_bullpen_late_innings_last3"]),
                "home_bullpen_late_innings_last3": home_recent_bullpen.get("late_innings_last3", record["home_bullpen_late_innings_last3"]),
                "away_bullpen_late_runs_allowed_last3": away_recent_bullpen.get("late_runs_allowed_last3", record["away_bullpen_late_runs_allowed_last3"]),
                "home_bullpen_late_runs_allowed_last3": home_recent_bullpen.get("late_runs_allowed_last3", record["home_bullpen_late_runs_allowed_last3"]),
                "away_bullpen_late_earned_runs_last3": away_recent_bullpen.get("late_earned_runs_last3", record["away_bullpen_late_earned_runs_last3"]),
                "home_bullpen_late_earned_runs_last3": home_recent_bullpen.get("late_earned_runs_last3", record["home_bullpen_late_earned_runs_last3"]),
                "away_bullpen_late_hits_allowed_last3": away_recent_bullpen.get("late_hits_allowed_last3", record["away_bullpen_late_hits_allowed_last3"]),
                "home_bullpen_late_hits_allowed_last3": home_recent_bullpen.get("late_hits_allowed_last3", record["home_bullpen_late_hits_allowed_last3"]),
                "away_bullpen_late_era_last3": away_recent_bullpen.get("late_era_last3", record["away_bullpen_late_era_last3"]),
                "home_bullpen_late_era_last3": home_recent_bullpen.get("late_era_last3", record["home_bullpen_late_era_last3"]),
                "away_bullpen_season_era": (bullpen_context_map.get(str(record["away_team"]) or "", {}) or {}).get("season_era"),
                "home_bullpen_season_era": (bullpen_context_map.get(str(record["home_team"]) or "", {}) or {}).get("season_era"),
                "away_bullpen_late_season_era": (bullpen_context_map.get(str(record["away_team"]) or "", {}) or {}).get("late_season_era"),
                "home_bullpen_late_season_era": (bullpen_context_map.get(str(record["home_team"]) or "", {}) or {}).get("late_season_era"),
                "away_bullpen_season_games": (bullpen_context_map.get(str(record["away_team"]) or "", {}) or {}).get("season_games"),
                "home_bullpen_season_games": (bullpen_context_map.get(str(record["home_team"]) or "", {}) or {}).get("season_games"),
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
            "certainty": _build_certainty_payload(
                starter_certainty=record["starter_certainty_score"],
                lineup_certainty=record["lineup_certainty_score"],
                weather_freshness=record["weather_freshness_score"],
                market_freshness=record["market_freshness_score"],
                bullpen_completeness=record["bullpen_completeness_score"],
                missing_fallback_count=record["missing_fallback_count"],
                board_state=record["board_state"],
            ),
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
            "supplemental_markets": supplemental_markets_map.get(game_id, {}),
            "umpire": umpire_map.get(game_id),
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
            # Box / merged pitching line (present after ingest; needed for Top EV + board parity with detail)
            "ip": starter.get("ip"),
            "strikeouts": starter.get("strikeouts"),
            "walks": starter.get("walks"),
            "pitch_count": starter.get("pitch_count"),
            "recent_form": _fetch_starter_recent_form(starter["pitcher_id"], target_date),
            "pitcher_props": pitcher_prop_map.get(
                (int(game["game_id"]), int(starter["pitcher_id"])), {}
            ) if starter.get("pitcher_id") is not None else {},
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
        player_payload = _attach_player_status_context(
            {
            "player_id": hit["player_id"],
            "player_name": hit["player_name"],
            "bats": hit["bats"],
            "position": hit["position"],
            "team": team,
            "opponent": hit["opponent"],
            "lineup_slot": hit["lineup_slot"],
            "is_confirmed_lineup": hit["is_confirmed_lineup"],
            "is_inferred_lineup": bool(hit.get("is_inferred_lineup")),
            "projected_plate_appearances": hit["projected_plate_appearances"],
            "streak_len_capped": hit["streak_len_capped"],
            "streak_len": hit.get("streak_len") or hit["streak_len_capped"],
            "predicted_hit_probability": hit["predicted_hit_probability"],
            "fair_price": hit["fair_price"],
            "market_price": hit["market_price"],
            "edge": hit["edge"],
            "hit_rate_blended": hit["hit_rate_blended"],
            "hit_rate_7": hit.get("hit_rate_7"),
            "hit_rate_30": hit.get("hit_rate_30"),
            "hard_hit_pct_14": hit.get("hard_hit_pct_14"),
            "hit_games_last7": hit.get("hit_games_last7"),
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
            "total_bases": tb_prediction_map.get(
                (int(hit["game_id"]), int(hit["player_id"]))
            ),
            **actual_meta,
            },
            player_status_map,
        )
        merged_hitter = _attach_hitter_matchup_context(
            player_payload,
            opposing_starter["throws"] if opposing_starter else None,
            hit_split_map,
        )
        merged_hitter["form"] = classify_hitter_form(merged_hitter)
        game["hit_targets"][team].append(merged_hitter)

    # --- Freshness / confidence badges ---
    ingest_freshness = _fetch_ingest_freshness(target_date)
    market_freezes = _fetch_market_freeze_map(target_date)
    readiness_map = _fetch_game_readiness_map(target_date)
    for game in games_by_id.values():
        total_freeze = market_freezes.get((int(game["game_id"]), "total"), {})
        _apply_market_freeze_payload(game["totals"], total_freeze)
        _apply_market_freeze_payload(
            game["first5_totals"],
            market_freezes.get((int(game["game_id"]), "first_five_total"), {}),
        )
        game["data_quality"] = _compute_data_quality_badge(game, ingest_freshness, readiness_map)
        market_cards, best_bets = _build_market_cards_for_game(
            game,
            market_rows_by_game.get(int(game["game_id"]), {}),
        )
        game["market_cards"] = _drop_first_five_team_total_cards(market_cards)
        game["best_bets"] = _drop_first_five_team_total_best_bets(best_bets)

    board_rows_ordered = [games_by_id[gid] for gid in games_by_id]
    _maybe_insert_board_top_ev_run_snapshots(target_date, board_rows_ordered)
    _maybe_insert_board_top_ev_snapshots(target_date, board_rows_ordered)
    lock_top_ev: dict[int, dict[str, Any]] = {}
    run_top_ev: dict[int, dict[str, Any]] = {}
    if get_settings().board_top_ev_snapshot_enabled and _table_exists("board_top_ev_snapshots"):
        lock_top_ev = _fetch_board_top_ev_snapshots_map(target_date)
    if get_settings().board_top_ev_run_snapshot_enabled and _table_exists("board_top_ev_run_snapshots"):
        run_top_ev = _fetch_board_top_ev_run_snapshots_map(target_date)
    for game in games_by_id.values():
        gid = int(game["game_id"])
        game["top_ev_pick"] = _resolve_top_ev_pick_with_snapshots(
            game,
            market_rows_by_game.get(gid, {}),
            lock_top_ev,
            run_top_ev,
        )

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


def _scale_expected_run_split(
    predicted_total: Any,
    away_weight: Any,
    home_weight: Any,
) -> tuple[float | None, float | None]:
    return best_bets_utils.scale_expected_run_split(predicted_total, away_weight, home_weight)


def _apply_market_freeze_payload(payload: dict[str, Any], freeze_row: dict[str, Any] | None) -> dict[str, Any]:
    freeze = freeze_row or {}
    frozen_line_value = freeze.get("frozen_line_value")
    payload["market_locked"] = bool(freeze)
    payload["locked_sportsbook"] = freeze.get("frozen_sportsbook")
    payload["locked_snapshot_ts"] = freeze.get("frozen_snapshot_ts")
    payload["locked_line_value"] = frozen_line_value
    if frozen_line_value is not None:
        payload["market_total"] = frozen_line_value
        payload["market_backed"] = True

    # Use stats-only (fundamentals) as primary; fall back to blended.
    fundamentals_total = _coerce_float(payload.get("predicted_total_fundamentals"))
    blended_total = _coerce_float(payload.get("predicted_total_runs"))
    predicted_total = fundamentals_total if fundamentals_total is not None else blended_total
    market_total = _coerce_float(payload.get("market_total"))
    actual_total = _coerce_float(payload.get("actual_total_runs"))
    if "recommended_side" in payload:
        payload["recommended_side"] = _recommended_side(predicted_total, market_total)
    if "actual_side" in payload:
        payload["actual_side"] = _actual_side(actual_total, market_total) if actual_total is not None else None
    if "result" in payload:
        payload["result"] = _graded_pick_result(
            payload.get("recommended_side"),
            payload.get("actual_side"),
            actual_total is not None,
        )
    if "delta_vs_market" in payload:
        payload["delta_vs_market"] = (
            None if predicted_total is None or market_total is None else round(predicted_total - market_total, 2)
        )
    return payload


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
    away_team = str(detail.get("away_team") or "Away")
    home_team = str(detail.get("home_team") or "Home")
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
    if predicted_total is None:
        predicted_total = _coerce_float(totals.get("predicted_total_fundamentals"))
    market_total = _coerce_float(totals.get("market_total"))
    edge_delta = None if predicted_total is None or market_total is None else predicted_total - market_total
    away_expected = _coerce_float(totals.get("away_expected_runs"))
    home_expected = _coerce_float(totals.get("home_expected_runs"))
    venue_run_factor = _coerce_float(totals.get("venue_run_factor"))
    temperature_f = _coerce_float(weather.get("temperature_f"))
    wind_speed_mph = _coerce_float(weather.get("wind_speed_mph"))
    line_movement = _coerce_float(totals.get("line_movement"))
    away_top5 = _coerce_float(totals.get("away_lineup_top5_xwoba"))
    home_top5 = _coerce_float(totals.get("home_lineup_top5_xwoba"))
    away_lineup_k = _coerce_float(totals.get("away_lineup_k_pct"))
    home_lineup_k = _coerce_float(totals.get("home_lineup_k_pct"))
    away_late_era = _coerce_float(totals.get("away_bullpen_late_era_last3"))
    home_late_era = _coerce_float(totals.get("home_bullpen_late_era_last3"))
    away_late_season_era = _coerce_float(totals.get("away_bullpen_late_season_era"))
    home_late_season_era = _coerce_float(totals.get("home_bullpen_late_season_era"))
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

    starter_profiles: list[dict[str, Any]] = []
    for side_key, team_name in (("away", away_team), ("home", home_team)):
        starter = starters.get(side_key) or {}
        recent = starter.get("recent_form") or {}
        starter_profiles.append(
            {
                "team": team_name,
                "xwoba_against": _coerce_float(starter.get("xwoba_against")),
                "csw_pct": _coerce_float(starter.get("csw_pct")),
                "whiff_pct": _coerce_float(recent.get("whiff_pct") or starter.get("whiff_pct")),
                "avg_ip": _coerce_float(recent.get("avg_ip")),
            }
        )

    def _best_profile(metric_name: str, *, reverse: bool = False) -> dict[str, Any] | None:
        candidates = [profile for profile in starter_profiles if profile.get(metric_name) is not None]
        if not candidates:
            return None
        return sorted(candidates, key=lambda profile: float(profile.get(metric_name) or 0.0), reverse=reverse)[0]

    if edge_delta is not None:
        signals.append(f"Model total sits {abs(edge_delta):.1f} runs {direction or 'away from'} the market.")
    if direction == "over":
        high_expected_teams = [
            f"{team} ({expected_runs:.1f})"
            for team, expected_runs in ((away_team, away_expected), (home_team, home_expected))
            if expected_runs is not None and expected_runs >= 4.6
        ]
        if high_expected_teams:
            signals.append(f"{', '.join(high_expected_teams)} are carrying strong implied team-run pressure.")
        if venue_run_factor is not None and venue_run_factor >= 1.03:
            signals.append(f"Park factor {venue_run_factor:.2f} points to a friendlier run environment.")
        if temperature_f is not None and temperature_f >= 78:
            signals.append(f"Warm {temperature_f:.0f}° weather boosts carry and run scoring.")
        if wind_speed_mph is not None and wind_speed_mph >= 12:
            signals.append(f"{wind_speed_mph:.0f} mph wind adds extra batted-ball volatility.")
        if max(value or 0.0 for value in [away_top5, home_top5]) >= 0.34:
            signals.append("At least one top-of-order group brings premium xwOBA form.")
        vulnerable_starter = _best_profile("xwoba_against", reverse=True)
        if vulnerable_starter is not None and (vulnerable_starter.get("xwoba_against") or 0.0) >= 0.325:
            signals.append(
                f"{vulnerable_starter['team']} starter contact quality is loose ({_format_metric(vulnerable_starter.get('xwoba_against'))} xwOBA allowed)."
            )
        short_starter = _best_profile("avg_ip")
        if short_starter is not None and (short_starter.get("avg_ip") or 9.0) <= 5.0:
            signals.append(
                f"{short_starter['team']} is getting only {_format_metric(short_starter.get('avg_ip'), 1)} IP lately from its starter, exposing relief earlier."
            )
        shaky_late_pen = max(
            (
                (away_team, away_late_era, away_late_season_era),
                (home_team, home_late_era, home_late_season_era),
            ),
            key=lambda item: float(item[1] or item[2] or -999.0),
        )
        if (shaky_late_pen[1] or shaky_late_pen[2] or 0.0) >= 4.40:
            recent_text = _format_metric(shaky_late_pen[1], 2)
            season_text = _format_metric(shaky_late_pen[2], 2)
            signals.append(f"{shaky_late_pen[0]} late bullpen ERA is stretched (L3 {recent_text} / season {season_text}).")
        if bullpen_burden >= 165 or b2b_bullpens >= 1:
            signals.append("Recent bullpen workload suggests thinner late-inning coverage.")
        if line_movement is not None and line_movement > 0.15:
            signals.append("The market has already drifted upward toward the over.")
    elif direction == "under":
        muted_offenses = [team for team, expected_runs in ((away_team, away_expected), (home_team, home_expected)) if expected_runs is not None and expected_runs <= 4.0]
        if len(muted_offenses) == 2:
            signals.append(f"Neither lineup is projected above 4.0 runs ({away_expected:.1f} / {home_expected:.1f}).")
        if venue_run_factor is not None and venue_run_factor <= 0.97:
            signals.append(f"Park factor {venue_run_factor:.2f} suppresses baseline scoring.")
        if temperature_f is not None and temperature_f <= 60:
            signals.append(f"Cooler {temperature_f:.0f}° weather leans run-suppressive.")
        if wind_speed_mph is not None and wind_speed_mph >= 12 and temperature_f is not None and temperature_f <= 68:
            signals.append(f"Cool air plus {wind_speed_mph:.0f} mph wind points to lower-quality run conditions.")
        if max(value or 0.0 for value in [away_lineup_k, home_lineup_k]) >= 0.24:
            signals.append("Strikeout-heavy lineup shape trims ball-in-play volume.")
        if max(value or 0.0 for value in [away_top5, home_top5]) <= 0.315:
            signals.append("Neither projected top-of-order group rates as especially dangerous.")
        suppressive_starter = _best_profile("xwoba_against")
        if suppressive_starter is not None and (suppressive_starter.get("xwoba_against") or 9.0) <= 0.295:
            signals.append(
                f"{suppressive_starter['team']} starter is suppressing contact ({_format_metric(suppressive_starter.get('xwoba_against'))} xwOBA allowed)."
            )
        bat_missing_starter = _best_profile("csw_pct", reverse=True)
        if bat_missing_starter is not None and (bat_missing_starter.get("csw_pct") or 0.0) >= 0.30:
            whiff_text = _format_rate(bat_missing_starter.get("whiff_pct"))
            signals.append(
                f"{bat_missing_starter['team']} starter brings bat-missing form ({_format_rate(bat_missing_starter.get('csw_pct'))} CSW, {whiff_text} whiff)."
            )
        stable_pens = [
            team
            for team, recent_era, season_era in (
                (away_team, away_late_era, away_late_season_era),
                (home_team, home_late_era, home_late_season_era),
            )
            if (recent_era is not None and recent_era <= 3.7) or (season_era is not None and season_era <= 3.8)
        ]
        if len(stable_pens) == 2:
            signals.append("Both bullpens bring stable late-inning run prevention.")
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

    signals = list(dict.fromkeys(signals))
    risks = list(dict.fromkeys(risks))

    headline = signals[0] if signals else (
        f"Core projection leans {direction}." if direction else "No strong totals rationale is available yet."
    )
    return {
        "direction": direction,
        "confidence": confidence,
        "headline": headline,
        "signals": signals[:6],
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


def _best_bet_market_display_name(market: str | None) -> str:
    mapping = {
        "moneyline": "Moneyline",
        "run_line": "Run Line",
        "game_total": "Game Total (Runs)",
        "totals": "Game Total (Runs)",
        "first5": "First 5 Totals",
        "player_hits": "1+ Hits",
        "away_team_total": "Away Team Total",
        "home_team_total": "Home Team Total",
        "first_five_moneyline": "First 5 Moneyline",
        "first_five_total": "First 5 Combined Total",
        "first_five_spread": "First 5 Run Line",
        "first_five_team_total_away": "First 5 Away Team Total",
        "first_five_team_total_home": "First 5 Home Team Total",
    }
    return mapping.get(str(market or ""), str(market or "Best Bet"))


def _best_bet_selection_label(
    market: str | None,
    recommended_side: str | None,
    away_team: str | None,
    home_team: str | None,
    line_value: float | None,
) -> str:
    market_key = str(market or "")
    side = str(recommended_side or "")
    away = away_team or "Away"
    home = home_team or "Home"
    if market_key == "moneyline":
        return f"{away if side == 'away' else home} ML"
    if market_key == "first_five_moneyline":
        return f"F5 {away if side == 'away' else home} ML"
    if market_key in ("game_total", "totals"):
        suffix = f" {line_value:.1f} runs" if line_value is not None else ""
        return f"Game total {str(side).title()}{suffix}".strip()
    if market_key in ("first_five_total", "first5"):
        suffix = f" {line_value:.1f}" if line_value is not None else ""
        return f"F5 {str(side).title()}{suffix}".strip()
    if market_key == "first_five_team_total_away":
        suffix = f" {line_value:.1f}" if line_value is not None else ""
        return f"F5 {away} TT {str(side).title()}{suffix}"
    if market_key == "first_five_team_total_home":
        suffix = f" {line_value:.1f}" if line_value is not None else ""
        return f"F5 {home} TT {str(side).title()}{suffix}"
    if market_key == "away_team_total":
        suffix = f" {line_value:.1f}" if line_value is not None else ""
        return f"{away} TT {str(side).title()}{suffix}"
    if market_key == "home_team_total":
        suffix = f" {line_value:.1f}" if line_value is not None else ""
        return f"{home} TT {str(side).title()}{suffix}"
    if market_key == "first_five_spread":
        team = away if side == "away" else home
        if line_value is None:
            return f"F5 {team} Run Line"
        display_line = line_value
        if side == "away" and display_line > 0:
            display_line = -display_line
        if side == "home" and display_line < 0:
            display_line = abs(display_line)
        return f"F5 {team} {display_line:+.1f}"
    if market_key == "run_line":
        team = away if side == "away" else home
        if line_value is None:
            return f"{team} Run Line"
        display_line = line_value
        if side == "away" and display_line > 0:
            display_line = -display_line
        if side == "home" and display_line < 0:
            display_line = abs(display_line)
        return f"{team} {display_line:+.1f}"
    return _best_bet_market_display_name(market_key)


def _fetch_best_bet_history_payload(
    target_date: date,
    *,
    window_days: int = 14,
    limit: int = 12,
    graded_only: bool = False,
) -> dict[str, Any]:
    if not _table_exists("prediction_outcomes_daily"):
        return {
            "target_date": str(target_date),
            "start_date": str(target_date),
            "end_date": str(target_date),
            "window_days": window_days,
            "summary": {},
            "by_market": [],
            "by_input_trust": [],
            "by_input_trust_market": [],
            "monotonicity": {
                "metric": "win_rate",
                "status": "insufficient_sample",
                "interpretation": "No prediction outcomes table — run product surfaces after games grade.",
            },
            "rows": [],
        }

    start_date = target_date - timedelta(days=max(window_days - 1, 0))
    params: dict[str, Any] = {
        "start_date": str(start_date),
        "target_date": str(target_date),
    }
    market_conditions: list[str] = []
    for index, market_key in enumerate(BEST_BET_MARKET_KEYS):
        param_name = f"market_{index}"
        params[param_name] = market_key
        market_conditions.append(f"market = :{param_name}")
    params["legacy_totals"] = "totals"
    params["legacy_first5"] = "first5"
    market_conditions.append("market = :legacy_totals")
    market_conditions.append("market = :legacy_first5")

    conditions = [
        "game_date BETWEEN :start_date AND :target_date",
        "entity_type = 'game'",
        "recommended_side IS NOT NULL",
        f"({' OR '.join(market_conditions)})",
    ]
    if graded_only:
        conditions.append("graded = TRUE")
    where = " AND ".join(conditions)

    frame = _safe_frame(
        f"""
        SELECT game_date, game_id, market, recommended_side, actual_side,
               graded, success, beat_market, probability, predicted_value,
               market_line, actual_value, entry_market_sportsbook,
               closing_market_sportsbook, closing_market_same_sportsbook,
               closing_market_line, clv_line_delta, clv_side_value,
               beat_closing_line, meta_payload
        FROM prediction_outcomes_daily
        WHERE {where}
        ORDER BY game_date DESC, game_id DESC, market
        """,
        params,
    )

    rows: list[dict[str, Any]] = []
    for record in _frame_records(frame):
        meta = _review_meta_payload(record.get("meta_payload"))
        away_team = meta.get("away_team")
        home_team = meta.get("home_team")
        market = str(record.get("market") or "")
        line_value = _coerce_float(meta.get("line_value"))
        rows.append(
            {
                "game_date": record.get("game_date"),
                "game_id": int(record["game_id"]),
                "market": market,
                "market_label": meta.get("market_label") or _best_bet_market_display_name(market),
                "selection_label": meta.get("selection_label")
                or _best_bet_selection_label(
                    market,
                    record.get("recommended_side"),
                    away_team,
                    home_team,
                    line_value,
                ),
                "away_team": away_team,
                "home_team": home_team,
                "recommended_side": record.get("recommended_side"),
                "actual_side": record.get("actual_side"),
                "result": _graded_pick_result(
                    record.get("recommended_side"),
                    record.get("actual_side"),
                    bool(record.get("graded")),
                ),
                "graded": bool(record.get("graded")),
                "success": record.get("success"),
                "beat_market": record.get("beat_market"),
                "model_probability": _coerce_float(record.get("probability")),
                "predicted_value": _coerce_float(record.get("predicted_value")),
                "no_vig_probability": _coerce_float(record.get("market_line")),
                "actual_value": _coerce_float(record.get("actual_value")),
                "line_value": line_value,
                "sportsbook": meta.get("sportsbook") or record.get("entry_market_sportsbook"),
                "price": meta.get("price"),
                "opposing_price": meta.get("opposing_price"),
                "probability_edge": _coerce_float(meta.get("probability_edge")),
                "weighted_ev": _coerce_float(meta.get("weighted_ev")),
                "entry_market_sportsbook": record.get("entry_market_sportsbook"),
                "closing_market_sportsbook": record.get("closing_market_sportsbook"),
                "closing_market_same_sportsbook": record.get("closing_market_same_sportsbook"),
                "closing_market_line": _coerce_float(record.get("closing_market_line")),
                "clv_line_delta": _coerce_float(record.get("clv_line_delta")),
                "clv_side_value": _coerce_float(record.get("clv_side_value")),
                "beat_closing_line": record.get("beat_closing_line"),
                "input_trust_grade": meta.get("input_trust_grade"),
                "input_trust_score": _coerce_float(meta.get("input_trust_score")),
                "promotion_tier": meta.get("promotion_tier"),
            }
        )

    def _history_summary(history_rows: list[dict[str, Any]]) -> dict[str, Any]:
        wins = sum(1 for row in history_rows if row.get("result") == "won")
        losses = sum(1 for row in history_rows if row.get("result") == "lost")
        pushes = sum(1 for row in history_rows if row.get("result") == "push")
        pending = sum(1 for row in history_rows if row.get("result") == "pending")
        decisions = wins + losses
        weighted_evs = [
            float(row["weighted_ev"])
            for row in history_rows
            if row.get("weighted_ev") is not None
        ]
        probability_edges = [
            float(row["probability_edge"])
            for row in history_rows
            if row.get("probability_edge") is not None
        ]
        clv_values = [
            float(row["clv_side_value"])
            for row in history_rows
            if row.get("clv_side_value") is not None
        ]
        beat_close_rows = [
            row for row in history_rows if row.get("beat_closing_line") is not None
        ]
        return {
            "total": len(history_rows),
            "graded": wins + losses + pushes,
            "pending": pending,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate": None if decisions <= 0 else float(wins / decisions),
            "avg_weighted_ev": None if not weighted_evs else float(sum(weighted_evs) / len(weighted_evs)),
            "avg_probability_edge": None if not probability_edges else float(sum(probability_edges) / len(probability_edges)),
            "avg_clv_side_value": None if not clv_values else float(sum(clv_values) / len(clv_values)),
            "positive_clv_rate": None
            if not beat_close_rows
            else float(sum(1 for row in beat_close_rows if row.get("beat_closing_line")) / len(beat_close_rows)),
        }

    def _bucket_rows_by_input_trust(history_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grades = ("A", "B", "C", "D", "unknown")
        buckets: list[dict[str, Any]] = []
        for grade in grades:
            if grade == "unknown":
                sub = [
                    row
                    for row in history_rows
                    if str(row.get("input_trust_grade") or "").strip().upper() not in {"A", "B", "C", "D"}
                ]
            else:
                sub = [
                    row
                    for row in history_rows
                    if str(row.get("input_trust_grade") or "").strip().upper() == grade
                ]
            if not sub:
                continue
            buckets.append({"input_trust_grade": grade, **_history_summary(sub)})
        return buckets

    def _by_input_trust_and_market(history_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        keys: set[tuple[str, str]] = set()
        for row in history_rows:
            g = str(row.get("input_trust_grade") or "").strip().upper()
            if g not in {"A", "B", "C", "D"}:
                g = "unknown"
            keys.add((g, str(row.get("market") or "")))
        out: list[dict[str, Any]] = []
        for grade, market_key in sorted(keys, key=lambda t: (t[0], t[1])):
            if not market_key:
                continue
            sub = [
                row
                for row in history_rows
                if str(row.get("market") or "") == market_key
                and (
                    str(row.get("input_trust_grade") or "").strip().upper() == grade
                    if grade != "unknown"
                    else str(row.get("input_trust_grade") or "").strip().upper() not in {"A", "B", "C", "D"}
                )
            ]
            if not sub:
                continue
            out.append(
                {
                    "input_trust_grade": grade,
                    "market": market_key,
                    "market_label": _best_bet_market_display_name(market_key),
                    **_history_summary(sub),
                }
            )
        return out

    def _monotonicity_win_rate(history_rows: list[dict[str, Any]]) -> dict[str, Any]:
        min_decisions = 3
        order = ["A", "B", "C", "D"]
        series: list[dict[str, Any]] = []
        for grade in order:
            sub = [
                row
                for row in history_rows
                if str(row.get("input_trust_grade") or "").strip().upper() == grade
            ]
            chunk = _history_summary(sub)
            decisions = int(chunk.get("wins") or 0) + int(chunk.get("losses") or 0)
            series.append(
                {
                    "grade": grade,
                    "graded_decisions": decisions,
                    "win_rate": chunk.get("win_rate"),
                    "sample_size": chunk.get("total"),
                }
            )
        rated = [
            item
            for item in series
            if item["graded_decisions"] >= min_decisions and item["win_rate"] is not None
        ]
        if len(rated) < 2:
            return {
                "metric": "win_rate",
                "status": "insufficient_sample",
                "min_graded_decisions_per_grade": min_decisions,
                "series": series,
                "interpretation": (
                    "Need at least two trust grades with enough graded decisions to judge whether "
                    "win rate improves from D toward A."
                ),
            }
        ordered = sorted(rated, key=lambda x: order.index(x["grade"]))
        rates = [float(x["win_rate"]) for x in ordered]
        mono = all(rates[i] >= rates[i + 1] for i in range(len(rates) - 1))
        return {
            "metric": "win_rate",
            "status": "monotonic" if mono else "non_monotonic",
            "min_graded_decisions_per_grade": min_decisions,
            "series": series,
            "interpretation": (
                "In this window, higher input-trust grades show higher or equal win rates — trust stratification is supported."
                if mono
                else "Win rate is not monotonic from A through D — review EV thresholds or trust calibration."
            ),
        }

    summary = _history_summary(rows)
    by_input_trust = _bucket_rows_by_input_trust(rows)
    by_input_trust_market = _by_input_trust_and_market(rows)
    monotonicity = _monotonicity_win_rate(rows)
    by_market = []
    for market_key in BEST_BET_MARKET_KEYS:
        market_rows = [row for row in rows if row.get("market") == market_key]
        if not market_rows:
            continue
        by_market.append(
            {
                "market": market_key,
                "market_label": _best_bet_market_display_name(market_key),
                **_history_summary(market_rows),
            }
        )

    by_market.sort(
        key=lambda row: (
            -(int(row.get("graded") or 0)),
            -(int(row.get("total") or 0)),
            str(row.get("market_label") or ""),
        )
    )
    return {
        "target_date": str(target_date),
        "start_date": str(start_date),
        "end_date": str(target_date),
        "window_days": window_days,
        "summary": summary,
        "by_market": by_market,
        "by_input_trust": by_input_trust,
        "by_input_trust_market": by_input_trust_market,
        "monotonicity": monotonicity,
        "rows": rows[: max(1, min(limit, 100))],
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
        "input_trust": cert.get("input_trust"),
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
        "slate_context": {
            "avg_input_trust_score": None,
            "avg_certainty_weight": None,
            "games_with_hit_targets": 0,
            "games_with_any_inferred_hitter": 0,
            "slate_note": None,
        },
    }

    model_errors: list[float] = []
    market_errors: list[float] = []
    model_biases: list[float] = []
    comparable_games = 0
    trust_scores: list[float] = []
    certainty_weights: list[float] = []
    games_with_targets = 0
    games_with_inferred = 0

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

        cert = game.get("certainty") if isinstance(game.get("certainty"), dict) else {}
        it = cert.get("input_trust") if isinstance(cert.get("input_trust"), dict) else {}
        ts = _to_float(it.get("score"))
        if ts is not None:
            trust_scores.append(float(ts))
        cw = best_bets_utils.game_certainty_weight(cert)
        certainty_weights.append(float(cw))

        hit_map = game.get("hit_targets") if isinstance(game.get("hit_targets"), dict) else {}
        any_target = False
        any_inferred = False
        for players in hit_map.values():
            for player in players or []:
                if not isinstance(player, dict):
                    continue
                any_target = True
                if player.get("is_inferred_lineup"):
                    any_inferred = True
        if any_target:
            games_with_targets += 1
            if any_inferred:
                games_with_inferred += 1

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

    slate = summary["slate_context"]
    if trust_scores:
        slate["avg_input_trust_score"] = round(sum(trust_scores) / len(trust_scores), 4)
    if certainty_weights:
        slate["avg_certainty_weight"] = round(sum(certainty_weights) / len(certainty_weights), 4)
    slate["games_with_hit_targets"] = games_with_targets
    slate["games_with_any_inferred_hitter"] = games_with_inferred

    slate_notes: list[str] = []
    total_games = int(summary["total_games"] or 0)
    if (
        games_with_targets
        and games_with_inferred >= max(2, int(0.5 * games_with_targets))
        and total_games >= 2
    ):
        slate_notes.append(
            "Many hit-board lineups are still projected — strict greens may stay sparse."
        )
    avg_ts = slate.get("avg_input_trust_score")
    if avg_ts is not None and float(avg_ts) < 0.52 and total_games >= 2:
        slate_notes.append("Slate-wide input trust is thin — expect fewer strict green edges.")
    if slate_notes:
        slate["slate_note"] = " ".join(slate_notes)

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


def _mean_numeric(values: list[Any], *, decimals: int = 2) -> float | None:
    nums: list[float] = []
    for v in values:
        if v is None:
            continue
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            continue
    if not nums:
        return None
    return round(sum(nums) / len(nums), decimals)


def _compute_team_recent_game_rollups(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Averages over the displayed recent-game sample for one team."""
    if not rows:
        return {
            "games": 0,
            "avg_team_runs": None,
            "avg_opponent_runs": None,
            "avg_game_total_runs": None,
            "avg_first5_combined_runs": None,
            "avg_six_plus_combined_runs": None,
            "avg_team_first5_runs": None,
            "avg_team_six_plus_runs": None,
        }
    return {
        "games": len(rows),
        "avg_team_runs": _mean_numeric([r.get("team_runs") for r in rows]),
        "avg_opponent_runs": _mean_numeric([r.get("opponent_runs") for r in rows]),
        "avg_game_total_runs": _mean_numeric([r.get("total_runs") for r in rows]),
        "avg_first5_combined_runs": _mean_numeric([r.get("first5_total_runs") for r in rows]),
        "avg_six_plus_combined_runs": _mean_numeric([r.get("six_plus_combined_runs") for r in rows]),
        "avg_team_first5_runs": _mean_numeric([r.get("team_runs_first5") for r in rows]),
        "avg_team_six_plus_runs": _mean_numeric([r.get("team_runs_late") for r in rows]),
    }


def _recent_games_f5_column_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """How many recent games have boxscore-backed F5 columns (games.total_runs_first5, etc.)."""
    n = len(rows)
    if not n:
        return {
            "games": 0,
            "with_game_total": 0,
            "with_combined_f5": 0,
            "with_six_plus_combined": 0,
            "with_team_f5": 0,
            "with_team_late": 0,
        }
    return {
        "games": n,
        "with_game_total": sum(1 for r in rows if r.get("total_runs") is not None),
        "with_combined_f5": sum(1 for r in rows if r.get("first5_total_runs") is not None),
        "with_six_plus_combined": sum(1 for r in rows if r.get("six_plus_combined_runs") is not None),
        "with_team_f5": sum(1 for r in rows if r.get("team_runs_first5") is not None),
        "with_team_late": sum(1 for r in rows if r.get("team_runs_late") is not None),
    }


def _fetch_team_last_n_final_games(
    team: str,
    *,
    target_date: date,
    exclude_game_id: int,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Last N completed games for a team before (or excluding) the current game — scores and F5 totals."""
    if not team or not _table_exists("games"):
        return []
    game_start_desc_order = _sql_order_nulls_last("g.game_start_ts", "DESC")
    frame = _safe_frame(
        f"""
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
            g.away_runs_first5,
            g.home_runs_first5,
            g.total_runs_first5 AS first5_total_runs,
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
                WHEN g.home_team = :team THEN 1
                ELSE 0
            END AS was_home
        FROM games g
        WHERE (g.home_team = :team OR g.away_team = :team)
          AND g.total_runs IS NOT NULL
          AND (
            g.game_date < :target_date
            OR (g.game_date = :target_date AND g.game_id <> :exclude_game_id)
          )
          AND (
              LOWER(COALESCE(g.status, '')) LIKE '%final%'
              OR LOWER(COALESCE(g.status, '')) LIKE '%completed%'
              OR LOWER(COALESCE(g.status, '')) LIKE '%game over%'
              OR LOWER(COALESCE(g.status, '')) LIKE '%closed%'
          )
        ORDER BY g.game_date DESC, {game_start_desc_order}
        LIMIT :limit
        """,
        {
            "team": team,
            "target_date": target_date,
            "exclude_game_id": int(exclude_game_id),
            "limit": int(limit),
        },
    )
    if frame.empty:
        return []
    out: list[dict[str, Any]] = []
    for row in _frame_records(frame):
        tr = _to_float(row.get("team_runs"))
        opp = _to_float(row.get("opponent_runs"))
        result = None
        if tr is not None and opp is not None:
            if tr > opp:
                result = "W"
            elif tr < opp:
                result = "L"
            else:
                result = "T"
        f5 = _to_float(row.get("first5_total_runs"))
        tot = _to_float(row.get("total_runs"))
        away_f5 = _to_float(row.get("away_runs_first5"))
        home_f5 = _to_float(row.get("home_runs_first5"))
        team_f5: float | None = None
        if str(row.get("away_team") or "") == team:
            team_f5 = away_f5
        elif str(row.get("home_team") or "") == team:
            team_f5 = home_f5
        team_late: int | None = None
        if tr is not None and team_f5 is not None:
            team_late = int(round(float(tr) - float(team_f5)))
        six_plus_combined: int | None = None
        if tot is not None and f5 is not None:
            six_plus_combined = int(round(float(tot) - float(f5)))
        out.append(
            {
                "game_id": row.get("game_id"),
                "game_date": row.get("game_date"),
                "game_start_ts": row.get("game_start_ts"),
                "away_team": row.get("away_team"),
                "home_team": row.get("home_team"),
                "opponent": row.get("opponent"),
                "was_home": bool(row.get("was_home")),
                "team_runs": int(tr) if tr is not None else None,
                "opponent_runs": int(opp) if opp is not None else None,
                "total_runs": int(tot) if tot is not None else None,
                "first5_total_runs": int(f5) if f5 is not None else None,
                "six_plus_combined_runs": six_plus_combined,
                "team_runs_first5": int(team_f5) if team_f5 is not None else None,
                "team_runs_late": team_late,
                "result": result,
            }
        )
    return out


def _fetch_first5_totals_map(target_date: date) -> dict[int, dict[str, Any]]:
    prediction_created_order = _sql_order_nulls_last("p.created_at", "DESC")
    feature_cutoff_order = _sql_order_nulls_last("f.feature_cutoff_ts", "DESC")
    market_line_order = _sql_order_nulls_last("gm.line_value", "DESC")
    market_freezes = _fetch_market_freeze_map(target_date)
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
            p.predicted_total_fundamentals,
            COALESCE(p.market_total, market.market_total) AS market_total,
            p.over_probability,
            p.under_probability,
            p.edge,
            CAST(f.feature_payload ->> 'away_runs_rate_blended' AS DOUBLE PRECISION) AS away_context_runs,
            CAST(f.feature_payload ->> 'home_runs_rate_blended' AS DOUBLE PRECISION) AS home_context_runs
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
        predicted_total_fundamentals = _to_float(row.get("predicted_total_fundamentals"))
        # Use stats-only as primary; fall back to blended
        primary_total = predicted_total_fundamentals if predicted_total_fundamentals is not None else predicted_total
        market_total = _to_float(row.get("market_total"))
        actual_total = _to_float(row.get("total_runs_first5"))
        away_expected_runs, home_expected_runs = _scale_expected_run_split(
            primary_total,
            row.get("away_context_runs"),
            row.get("home_context_runs"),
        )
        supported = any(
            value is not None
            for value in (
                primary_total,
                market_total,
                actual_total,
                away_expected_runs,
                home_expected_runs,
            )
        )
        recommended_side = _recommended_side(primary_total, market_total)
        actual_side = _actual_side(actual_total, market_total) if actual_total is not None else None
        payload = {
            "supported": supported,
            "model_name": row.get("model_name"),
            "model_version": row.get("model_version"),
            "prediction_ts": row.get("prediction_ts"),
            "predicted_total_runs": row.get("predicted_total_runs"),
            "predicted_total_fundamentals": row.get("predicted_total_fundamentals"),
            "market_total": row.get("market_total"),
            "over_probability": row.get("over_probability"),
            "under_probability": row.get("under_probability"),
            "edge": row.get("edge"),
            "market_backed": market_total is not None,
            "away_expected_runs": away_expected_runs,
            "home_expected_runs": home_expected_runs,
            "away_runs": row.get("away_runs_first5"),
            "home_runs": row.get("home_runs_first5"),
            "actual_total_runs": row.get("total_runs_first5"),
            "recommended_side": recommended_side,
            "actual_side": actual_side,
            "result": _graded_pick_result(recommended_side, actual_side, actual_total is not None),
            "delta_vs_market": None if primary_total is None or market_total is None else round(primary_total - market_total, 2),
        }
        payload_by_game[int(game_id)] = _apply_market_freeze_payload(
            payload,
            market_freezes.get((int(game_id), "first_five_total")),
        )
    return payload_by_game


def _fetch_supplemental_markets_map(target_date: date) -> dict[int, dict[str, Any]]:
    """Team totals, moneyline, run line, F5 moneyline keyed by game_id."""
    rows = _fetch_latest_game_market_rows(target_date)
    return _aggregate_supplemental_market_rows(rows)


def _fetch_latest_game_market_rows(
    target_date: date,
    game_id: int | None = None,
    market_types: tuple[str, ...] = SUPPLEMENTAL_GAME_MARKET_TYPES,
) -> list[dict[str, Any]]:
    if not _table_exists("game_markets"):
        return []

    market_list = ", ".join(f"'{market_type}'" for market_type in market_types)
    filters = ["gm.game_date = :target_date", f"gm.market_type IN ({market_list})"]
    params: dict[str, Any] = {"target_date": target_date}
    if game_id is not None:
        filters.append("gm.game_id = :game_id")
        params["game_id"] = game_id

    frame = _safe_frame(
        f"""
        WITH ranked AS (
            SELECT
                gm.game_id,
                gm.market_type,
                gm.sportsbook,
                gm.line_value,
                gm.over_price,
                gm.under_price,
                gm.snapshot_ts,
                gm.source_name,
                ROW_NUMBER() OVER (
                    PARTITION BY gm.game_id, gm.market_type, COALESCE(gm.sportsbook, '')
                    ORDER BY gm.snapshot_ts DESC
                ) AS row_rank
            FROM game_markets gm
            WHERE {' AND '.join(filters)}
        )
        SELECT
            game_id,
            market_type,
            sportsbook,
            line_value,
            over_price,
            under_price,
            snapshot_ts,
            source_name
        FROM ranked
        WHERE row_rank = 1
        ORDER BY game_id, market_type, sportsbook
        """,
        params,
    )
    return _frame_records(frame)


def _market_focus_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float | None]:
    return best_bets_utils.market_focus_rows(rows)


def _aggregate_supplemental_market_rows(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return best_bets_utils.aggregate_supplemental_market_rows(rows)


def _format_price_text(price: Any) -> str:
    value = _to_float(price)
    if value is None:
        return "-"
    rounded = int(round(value))
    return f"+{rounded}" if rounded > 0 else str(rounded)


def _format_market_line_text(value: Any) -> str:
    line = _to_float(value)
    if line is None:
        return "-"
    return f"{line:+.1f}" if abs(line) >= 0.05 else "0.0"


def _american_implied_probability(price: Any) -> float | None:
    value = _to_float(price)
    if value is None or abs(value) < 1e-9:
        return None
    if value > 0:
        return 100.0 / (value + 100.0)
    return abs(value) / (abs(value) + 100.0)


def _american_profit_per_unit(price: Any) -> float | None:
    value = _to_float(price)
    if value is None or abs(value) < 1e-9:
        return None
    if value > 0:
        return value / 100.0
    return 100.0 / abs(value)


def _no_vig_pair(price_a: Any, price_b: Any) -> tuple[float | None, float | None]:
    implied_a = _american_implied_probability(price_a)
    implied_b = _american_implied_probability(price_b)
    if implied_a is None or implied_b is None:
        return None, None
    total = implied_a + implied_b
    if total <= 0:
        return None, None
    return implied_a / total, implied_b / total


def _poisson_distribution(mean_runs: Any, max_runs: int = MARKET_SIM_MAX_RUNS) -> list[float] | None:
    mean_value = _to_float(mean_runs)
    if mean_value is None or mean_value < 0:
        return None
    probabilities = [math.exp(-mean_value)]
    for run_total in range(1, max_runs + 1):
        probabilities.append(probabilities[-1] * mean_value / float(run_total))
    total_probability = sum(probabilities)
    if total_probability <= 0:
        return None
    if total_probability < 1.0:
        probabilities[-1] += 1.0 - total_probability
    else:
        probabilities = [probability / total_probability for probability in probabilities]
    return probabilities


def _joint_run_distribution(away_mean: Any, home_mean: Any) -> dict[str, float] | None:
    away_probs = _poisson_distribution(away_mean)
    home_probs = _poisson_distribution(home_mean)
    if away_probs is None or home_probs is None:
        return None
    away_win = 0.0
    home_win = 0.0
    tie = 0.0
    for away_runs, away_probability in enumerate(away_probs):
        for home_runs, home_probability in enumerate(home_probs):
            joint_probability = away_probability * home_probability
            if away_runs > home_runs:
                away_win += joint_probability
            elif home_runs > away_runs:
                home_win += joint_probability
            else:
                tie += joint_probability
    total_probability = away_win + home_win + tie
    if total_probability <= 0:
        return None
    scale = 1.0 / total_probability
    return {
        "away_win": away_win * scale,
        "home_win": home_win * scale,
        "tie": tie * scale,
    }


def _team_total_side_probabilities(team_mean: Any, line_value: Any) -> dict[str, dict[str, float]] | None:
    distribution = _poisson_distribution(team_mean)
    line = _to_float(line_value)
    if distribution is None or line is None:
        return None
    over_win = 0.0
    under_win = 0.0
    push = 0.0
    for runs, probability in enumerate(distribution):
        if runs > line:
            over_win += probability
        elif runs < line:
            under_win += probability
        else:
            push += probability
    return {
        "over": {
            "win": over_win,
            "loss": under_win,
            "push": push,
        },
        "under": {
            "win": under_win,
            "loss": over_win,
            "push": push,
        },
    }


def _moneyline_side_probabilities(
    away_mean: Any,
    home_mean: Any,
    *,
    push_on_tie: bool,
) -> dict[str, dict[str, float]] | None:
    joint = _joint_run_distribution(away_mean, home_mean)
    if joint is None:
        return None
    if push_on_tie:
        return {
            "away": {
                "win": joint["away_win"],
                "loss": joint["home_win"],
                "push": joint["tie"],
            },
            "home": {
                "win": joint["home_win"],
                "loss": joint["away_win"],
                "push": joint["tie"],
            },
        }
    away_mean_value = _to_float(away_mean) or 0.0
    home_mean_value = _to_float(home_mean) or 0.0
    total_mean = away_mean_value + home_mean_value
    home_tie_share = 0.5 if total_mean <= 0 else home_mean_value / total_mean
    away_tie_share = 1.0 - home_tie_share
    return {
        "away": {
            "win": joint["away_win"] + (joint["tie"] * away_tie_share),
            "loss": joint["home_win"] + (joint["tie"] * home_tie_share),
            "push": 0.0,
        },
        "home": {
            "win": joint["home_win"] + (joint["tie"] * home_tie_share),
            "loss": joint["away_win"] + (joint["tie"] * away_tie_share),
            "push": 0.0,
        },
    }


def _run_line_side_probabilities(
    away_mean: Any,
    home_mean: Any,
    home_line_value: Any,
) -> dict[str, dict[str, float]] | None:
    joint = _joint_run_distribution(away_mean, home_mean)
    home_line = _to_float(home_line_value)
    away_probs = _poisson_distribution(away_mean)
    home_probs = _poisson_distribution(home_mean)
    if joint is None or home_line is None or away_probs is None or home_probs is None:
        return None
    home_cover = 0.0
    away_cover = 0.0
    push = 0.0
    for away_runs, away_probability in enumerate(away_probs):
        for home_runs, home_probability in enumerate(home_probs):
            joint_probability = away_probability * home_probability
            adjusted_home = float(home_runs) + home_line
            if adjusted_home > float(away_runs):
                home_cover += joint_probability
            elif adjusted_home < float(away_runs):
                away_cover += joint_probability
            else:
                push += joint_probability
    total_probability = home_cover + away_cover + push
    if total_probability <= 0:
        return None
    scale = 1.0 / total_probability
    home_cover *= scale
    away_cover *= scale
    push *= scale
    return {
        "home": {"win": home_cover, "loss": away_cover, "push": push},
        "away": {"win": away_cover, "loss": home_cover, "push": push},
    }


def _build_certainty_payload(
    *,
    starter_certainty: Any,
    lineup_certainty: Any,
    weather_freshness: Any,
    market_freshness: Any,
    bullpen_completeness: Any,
    missing_fallback_count: Any,
    board_state: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "starter_certainty": starter_certainty,
        "lineup_certainty": lineup_certainty,
        "weather_freshness": weather_freshness,
        "market_freshness": market_freshness,
        "bullpen_completeness": bullpen_completeness,
        "missing_fallback_count": missing_fallback_count,
        "board_state": board_state,
    }
    payload["input_trust"] = _input_trust_from_certainty(payload)
    return payload


def _market_thresholds(market_key: str) -> dict[str, float]:
    return BEST_BET_THRESHOLD_MAP.get(
        market_key,
        {
            "weighted_ev": 0.015,
            "probability_edge": 0.025,
            "certainty_weight": 0.80,
            "model_probability": 0.57,
        },
    )


def _passes_best_bet_thresholds(
    market_key: str,
    *,
    weighted_ev: float,
    probability_edge: float,
    certainty_weight: float,
    model_probability: float,
) -> bool:
    thresholds = _market_thresholds(market_key)
    return bool(
        weighted_ev >= float(thresholds["weighted_ev"])
        and probability_edge >= float(thresholds["probability_edge"])
        and certainty_weight >= float(thresholds["certainty_weight"])
        and model_probability >= float(thresholds["model_probability"])
    )


def _build_market_candidate(
    *,
    game: dict[str, Any],
    market_key: str,
    market_label: str,
    selection_label: str,
    bet_side: str,
    sportsbook: Any,
    line_value: Any,
    price: Any,
    opposing_price: Any,
    model_probability: float | None,
    model_loss_probability: float | None,
    push_probability: float | None,
    certainty_weight: float,
    market_summary: str,
    model_summary: str,
) -> dict[str, Any] | None:
    fair_probability, _ = _no_vig_pair(price, opposing_price)
    profit_per_unit = _american_profit_per_unit(price)
    if fair_probability is None or profit_per_unit is None or model_probability is None:
        return None
    loss_probability = model_loss_probability
    push_probability = 0.0 if push_probability is None else max(0.0, push_probability)
    if loss_probability is None:
        loss_probability = max(0.0, 1.0 - model_probability - push_probability)
    raw_ev = (model_probability * profit_per_unit) - loss_probability
    probability_edge = model_probability - fair_probability
    weighted_ev = raw_ev * certainty_weight
    positive = _passes_best_bet_thresholds(
        market_key,
        weighted_ev=weighted_ev,
        probability_edge=probability_edge,
        certainty_weight=certainty_weight,
        model_probability=model_probability,
    )
    return {
        "game_id": int(game.get("game_id") or 0),
        "away_team": game.get("away_team"),
        "home_team": game.get("home_team"),
        "market_key": market_key,
        "market_label": market_label,
        "selection_label": selection_label,
        "bet_side": bet_side,
        "sportsbook": sportsbook,
        "line_value": _to_float(line_value),
        "price": None if price is None else int(price),
        "opposing_price": None if opposing_price is None else int(opposing_price),
        "model_probability": round(float(model_probability), 4),
        "no_vig_probability": round(float(fair_probability), 4),
        "probability_edge": round(float(probability_edge), 4),
        "raw_ev": round(float(raw_ev), 4),
        "weighted_ev": round(float(weighted_ev), 4),
        "certainty_weight": round(float(certainty_weight), 4),
        "push_probability": round(float(push_probability), 4),
        "positive": positive,
        "market_summary": market_summary,
        "model_summary": model_summary,
    }


def _fallback_market_card(
    *,
    game: dict[str, Any],
    market_key: str,
    market_label: str,
    market_summary: str,
    model_summary: str,
) -> dict[str, Any]:
    return {
        "game_id": int(game.get("game_id") or 0),
        "away_team": game.get("away_team"),
        "home_team": game.get("home_team"),
        "market_key": market_key,
        "market_label": market_label,
        "selection_label": None,
        "bet_side": None,
        "sportsbook": None,
        "line_value": None,
        "price": None,
        "opposing_price": None,
        "model_probability": None,
        "no_vig_probability": None,
        "probability_edge": None,
        "raw_ev": None,
        "weighted_ev": None,
        "certainty_weight": None,
        "push_probability": None,
        "positive": False,
        "market_summary": market_summary,
        "model_summary": model_summary,
    }


def _best_market_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda candidate: (
            float(candidate.get("weighted_ev") or -999.0),
            float(candidate.get("probability_edge") or -999.0),
            float(candidate.get("raw_ev") or -999.0),
            float(candidate.get("model_probability") or -999.0),
        ),
    )


def _build_market_cards_for_game(
    game: dict[str, Any],
    market_rows_by_type: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return best_bets_utils.build_market_cards_for_game(game, market_rows_by_type)


_F5_TEAM_TOTAL_MARKET_KEYS = frozenset({"first_five_team_total_away", "first_five_team_total_home"})


def _drop_first_five_team_total_cards(
    cards: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Hide F5 *per-team* O/U only. Combined ``first_five_total`` (both teams) stays in the product."""
    return [c for c in cards if str(c.get("market_key") or "") not in _F5_TEAM_TOTAL_MARKET_KEYS]


def _drop_first_five_team_total_best_bets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if str(r.get("market_key") or "") not in _F5_TEAM_TOTAL_MARKET_KEYS]


def _build_top_ev_pick_for_game_detail(
    detail: dict[str, Any],
    game_market_rows_by_type: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """Highest weighted-EV priced line among team markets, 1+ hit, and pitcher K O/U for this game."""
    return select_top_weighted_ev_pick(
        collect_top_ev_candidates(detail, game_market_rows_by_type),
    )


def _is_game_board_green_locked(game_row: dict[str, Any]) -> bool:
    """True once the game is inside the pregame ingest lock window (includes post first-pitch)."""
    ts = game_row.get("game_start_ts")
    if ts is None:
        return False
    lock_m = int(get_settings().pregame_ingest_lock_minutes or 0)
    if lock_m <= 0:
        return False
    return bool(is_pregame_ingest_locked(ts, lock_minutes=lock_m))


def _is_game_top_ev_snapshot_lock_active(game_row: dict[str, Any]) -> bool:
    """True inside the Top EV snapshot window — may use a different cutoff than ingest (see settings)."""
    ts = game_row.get("game_start_ts")
    if ts is None:
        return False
    settings = get_settings()
    inherited = int(settings.pregame_ingest_lock_minutes or 0)
    explicit = settings.board_top_ev_snapshot_lock_minutes
    lock_m = inherited if explicit is None else int(explicit)
    if lock_m <= 0:
        return False
    return bool(is_pregame_ingest_locked(ts, lock_minutes=lock_m))


def _fetch_board_green_snapshots_map(target_date: date) -> dict[int, dict[str, Any]]:
    if not table_exists("board_green_snapshots"):
        return {}
    frame = _safe_frame(
        """
        SELECT game_id, snapshot_payload
        FROM board_green_snapshots
        WHERE game_date = :gd
        """,
        {"gd": target_date},
    )
    if frame.empty:
        return {}
    out: dict[int, dict[str, Any]] = {}
    for row in _frame_records(frame):
        gid = int(row.get("game_id") or 0)
        raw = row.get("snapshot_payload")
        if not gid or raw is None:
            continue
        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict):
            out[gid] = payload
    return out


def _insert_board_green_snapshot_row(target_date: date, game_id: int, payload: dict[str, Any]) -> None:
    raw = json.dumps(jsonable_encoder(payload), default=str)
    frozen_at = datetime.now(timezone.utc)
    dia = get_dialect_name()
    params = {
        "game_id": game_id,
        "game_date": target_date.isoformat() if dia == "sqlite" else target_date,
        "snapshot_payload": raw,
        "frozen_at": frozen_at,
    }
    if dia == "sqlite":
        run_sql(
            """
            INSERT OR IGNORE INTO board_green_snapshots (game_id, game_date, snapshot_payload, frozen_at)
            VALUES (:game_id, :game_date, :snapshot_payload, :frozen_at)
            """,
            params,
        )
    else:
        run_sql(
            """
            INSERT INTO board_green_snapshots (game_id, game_date, snapshot_payload, frozen_at)
            VALUES (:game_id, :game_date, :snapshot_payload, :frozen_at)
            ON CONFLICT (game_id, game_date) DO NOTHING
            """,
            params,
        )


def _fetch_board_green_run_snapshots_map(target_date: date) -> dict[int, dict[str, Any]]:
    if not table_exists("board_green_run_snapshots"):
        return {}
    frame = _safe_frame(
        """
        SELECT game_id, snapshot_payload
        FROM board_green_run_snapshots
        WHERE game_date = :gd
        """,
        {"gd": target_date},
    )
    if frame.empty:
        return {}
    out: dict[int, dict[str, Any]] = {}
    for row in _frame_records(frame):
        gid = int(row.get("game_id") or 0)
        raw = row.get("snapshot_payload")
        if not gid or raw is None:
            continue
        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict):
            out[gid] = payload
    return out


def _insert_board_green_run_snapshot_row(target_date: date, game_id: int, payload: dict[str, Any]) -> None:
    raw = json.dumps(jsonable_encoder(payload), default=str)
    frozen_at = datetime.now(timezone.utc)
    dia = get_dialect_name()
    params = {
        "game_id": game_id,
        "game_date": target_date.isoformat() if dia == "sqlite" else target_date,
        "snapshot_payload": raw,
        "frozen_at": frozen_at,
    }
    if dia == "sqlite":
        run_sql(
            """
            INSERT OR IGNORE INTO board_green_run_snapshots (game_id, game_date, snapshot_payload, frozen_at)
            VALUES (:game_id, :game_date, :snapshot_payload, :frozen_at)
            """,
            params,
        )
    else:
        run_sql(
            """
            INSERT INTO board_green_run_snapshots (game_id, game_date, snapshot_payload, frozen_at)
            VALUES (:game_id, :game_date, :snapshot_payload, :frozen_at)
            ON CONFLICT (game_id, game_date) DO NOTHING
            """,
            params,
        )


def _maybe_insert_board_green_run_snapshots(
    target_date: date,
    rows: list[dict[str, Any]],
    live_green: list[dict[str, Any]],
) -> None:
    if not get_settings().board_green_run_snapshot_enabled:
        return
    if not table_exists("board_green_run_snapshots"):
        return
    row_by = {int(r["game_id"]): r for r in rows if r.get("game_id") is not None}
    existing = _fetch_board_green_run_snapshots_map(target_date)
    for card in live_green:
        gid = int(card.get("game_id") or 0)
        if not gid or gid in existing:
            continue
        gr = row_by.get(gid)
        if gr is None or not is_before_scheduled_first_pitch(gr.get("game_start_ts")):
            continue
        _insert_board_green_run_snapshot_row(target_date, gid, card)


def _maybe_insert_board_green_snapshots(
    target_date: date,
    rows: list[dict[str, Any]],
    live_green: list[dict[str, Any]],
) -> None:
    if not get_settings().board_green_snapshot_enabled:
        return
    if not table_exists("board_green_snapshots"):
        return
    row_by = {int(r["game_id"]): r for r in rows if r.get("game_id") is not None}
    existing = _fetch_board_green_snapshots_map(target_date)
    for card in live_green:
        gid = int(card.get("game_id") or 0)
        if not gid or gid in existing:
            continue
        gr = row_by.get(gid)
        if gr is None or not _is_game_board_green_locked(gr):
            continue
        if not is_before_scheduled_first_pitch(gr.get("game_start_ts")):
            continue
        _insert_board_green_snapshot_row(target_date, gid, card)


def _merge_board_green_snapshots_into_live(
    target_date: date,
    rows: list[dict[str, Any]],
    live_green: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    settings = get_settings()
    row_by = {int(r["game_id"]): r for r in rows if r.get("game_id") is not None}
    lock_snaps: dict[int, dict[str, Any]] = {}
    if settings.board_green_snapshot_enabled and table_exists("board_green_snapshots"):
        lock_snaps = _fetch_board_green_snapshots_map(target_date)
    run_snaps: dict[int, dict[str, Any]] = {}
    if settings.board_green_run_snapshot_enabled and table_exists("board_green_run_snapshots"):
        run_snaps = _fetch_board_green_run_snapshots_map(target_date)
    if not lock_snaps and not run_snaps:
        return live_green

    def _frozen_green_payload(gid: int) -> tuple[dict[str, Any], str] | None:
        gr = row_by.get(gid)
        if gr is None:
            return None
        if lock_snaps and gid in lock_snaps and _is_game_board_green_locked(gr):
            return dict(lock_snaps[gid]), "lock"
        if run_snaps and gid in run_snaps and is_before_scheduled_first_pitch(gr.get("game_start_ts")):
            return dict(run_snaps[gid]), "run"
        return None

    out: list[dict[str, Any]] = []
    seen: set[int] = set()
    for c in live_green:
        gid = int(c.get("game_id") or 0)
        resolved = _frozen_green_payload(gid) if gid else None
        if resolved:
            fc, kind = resolved
            fc["board_green_frozen"] = True
            fc["board_green_snapshot_kind"] = kind
            out.append(fc)
        else:
            out.append(c)
        if gid:
            seen.add(gid)

    for gid in set(lock_snaps) | set(run_snaps):
        if gid in seen:
            continue
        resolved = _frozen_green_payload(gid)
        if not resolved:
            continue
        fc, kind = resolved
        fc["board_green_frozen"] = True
        fc["board_green_snapshot_kind"] = kind
        out.append(fc)

    out.sort(
        key=lambda x: (
            float(x.get("weighted_ev") or -999.0),
            float(x.get("probability_edge") or -999.0),
        ),
        reverse=True,
    )
    return out


def _fetch_board_top_ev_snapshots_map(target_date: date) -> dict[int, dict[str, Any]]:
    if not _table_exists("board_top_ev_snapshots"):
        return {}
    frame = _safe_frame(
        """
        SELECT game_id, snapshot_payload
        FROM board_top_ev_snapshots
        WHERE game_date = :gd
        """,
        {"gd": target_date},
    )
    if frame.empty:
        return {}
    out: dict[int, dict[str, Any]] = {}
    for row in _frame_records(frame):
        gid = int(row.get("game_id") or 0)
        raw = row.get("snapshot_payload")
        if not gid or raw is None:
            continue
        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict):
            out[gid] = payload
    return out


def _fetch_board_top_ev_snapshots_for_range(start_date: date, end_date: date) -> dict[tuple[date, int], dict[str, Any]]:
    """All frozen Top EV payloads between ``start_date`` and ``end_date`` (inclusive)."""
    if not _table_exists("board_top_ev_snapshots"):
        return {}
    frame = _safe_frame(
        """
        SELECT game_date, game_id, snapshot_payload
        FROM board_top_ev_snapshots
        WHERE game_date BETWEEN :start_date AND :end_date
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    if frame.empty:
        return {}
    out: dict[tuple[date, int], dict[str, Any]] = {}
    for row in _frame_records(frame):
        gd = row.get("game_date")
        if gd is None:
            continue
        try:
            gday = date.fromisoformat(str(gd)[:10])
        except ValueError:
            continue
        gid = int(row.get("game_id") or 0)
        raw = row.get("snapshot_payload")
        if not gid or raw is None:
            continue
        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict):
            out[(gday, gid)] = payload
    return out


def _insert_board_top_ev_snapshot_row(target_date: date, game_id: int, payload: dict[str, Any]) -> None:
    raw = json.dumps(jsonable_encoder(payload), default=str)
    frozen_at = datetime.now(timezone.utc)
    dia = get_dialect_name()
    params = {
        "game_id": game_id,
        "game_date": target_date.isoformat() if dia == "sqlite" else target_date,
        "snapshot_payload": raw,
        "frozen_at": frozen_at,
    }
    if dia == "sqlite":
        run_sql(
            """
            INSERT OR IGNORE INTO board_top_ev_snapshots (game_id, game_date, snapshot_payload, frozen_at)
            VALUES (:game_id, :game_date, :snapshot_payload, :frozen_at)
            """,
            params,
        )
    else:
        run_sql(
            """
            INSERT INTO board_top_ev_snapshots (game_id, game_date, snapshot_payload, frozen_at)
            VALUES (:game_id, :game_date, :snapshot_payload, :frozen_at)
            ON CONFLICT (game_id, game_date) DO NOTHING
            """,
            params,
        )


def _fetch_board_top_ev_run_snapshots_map(target_date: date) -> dict[int, dict[str, Any]]:
    if not _table_exists("board_top_ev_run_snapshots"):
        return {}
    frame = _safe_frame(
        """
        SELECT game_id, snapshot_payload
        FROM board_top_ev_run_snapshots
        WHERE game_date = :gd
        """,
        {"gd": target_date},
    )
    if frame.empty:
        return {}
    out: dict[int, dict[str, Any]] = {}
    for row in _frame_records(frame):
        gid = int(row.get("game_id") or 0)
        raw = row.get("snapshot_payload")
        if not gid or raw is None:
            continue
        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict):
            out[gid] = payload
    return out


def _fetch_board_top_ev_run_snapshots_for_range(start_date: date, end_date: date) -> dict[tuple[date, int], dict[str, Any]]:
    if not _table_exists("board_top_ev_run_snapshots"):
        return {}
    frame = _safe_frame(
        """
        SELECT game_date, game_id, snapshot_payload
        FROM board_top_ev_run_snapshots
        WHERE game_date BETWEEN :start_date AND :end_date
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    if frame.empty:
        return {}
    out: dict[tuple[date, int], dict[str, Any]] = {}
    for row in _frame_records(frame):
        gd = row.get("game_date")
        if gd is None:
            continue
        try:
            gday = date.fromisoformat(str(gd)[:10])
        except ValueError:
            continue
        gid = int(row.get("game_id") or 0)
        raw = row.get("snapshot_payload")
        if not gid or raw is None:
            continue
        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict):
            out[(gday, gid)] = payload
    return out


def _insert_board_top_ev_run_snapshot_row(target_date: date, game_id: int, payload: dict[str, Any]) -> None:
    raw = json.dumps(jsonable_encoder(payload), default=str)
    frozen_at = datetime.now(timezone.utc)
    dia = get_dialect_name()
    params = {
        "game_id": game_id,
        "game_date": target_date.isoformat() if dia == "sqlite" else target_date,
        "snapshot_payload": raw,
        "frozen_at": frozen_at,
    }
    if dia == "sqlite":
        run_sql(
            """
            INSERT OR IGNORE INTO board_top_ev_run_snapshots (game_id, game_date, snapshot_payload, frozen_at)
            VALUES (:game_id, :game_date, :snapshot_payload, :frozen_at)
            """,
            params,
        )
    else:
        run_sql(
            """
            INSERT INTO board_top_ev_run_snapshots (game_id, game_date, snapshot_payload, frozen_at)
            VALUES (:game_id, :game_date, :snapshot_payload, :frozen_at)
            ON CONFLICT (game_id, game_date) DO NOTHING
            """,
            params,
        )


def _maybe_insert_board_top_ev_run_snapshots(target_date: date, board_rows: list[dict[str, Any]]) -> None:
    """Persist the first Top EV pick per game on the earliest pregame board build (no lock window)."""
    if not get_settings().board_top_ev_run_snapshot_enabled:
        return
    if not _table_exists("board_top_ev_run_snapshots"):
        return
    row_by = {int(r["game_id"]): r for r in board_rows if r.get("game_id") is not None}
    existing = _fetch_board_top_ev_run_snapshots_map(target_date)
    market_rows = _fetch_latest_game_market_rows(target_date)
    market_rows_by_game: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for market_row in market_rows:
        gid_raw = market_row.get("game_id")
        mt = str(market_row.get("market_type") or "")
        if gid_raw is None or not mt:
            continue
        gid = int(gid_raw)
        market_rows_by_game.setdefault(gid, {}).setdefault(mt, []).append(market_row)

    for game in board_rows:
        gid = int(game.get("game_id") or 0)
        if not gid or gid in existing:
            continue
        gr = row_by.get(gid)
        if gr is None or not is_before_scheduled_first_pitch(gr.get("game_start_ts")):
            continue
        pick = _top_ev_pick_for_board_row(gr, market_rows_by_game.get(gid, {}))
        if not pick:
            continue
        _insert_board_top_ev_run_snapshot_row(target_date, gid, pick)


def _maybe_insert_board_top_ev_snapshots(target_date: date, board_rows: list[dict[str, Any]]) -> None:
    """Persist the first Top EV pick per game when the slate crosses the pregame ingest lock window."""
    if not get_settings().board_top_ev_snapshot_enabled:
        return
    if not _table_exists("board_top_ev_snapshots"):
        return
    row_by = {int(r["game_id"]): r for r in board_rows if r.get("game_id") is not None}
    existing = _fetch_board_top_ev_snapshots_map(target_date)
    market_rows = _fetch_latest_game_market_rows(target_date)
    market_rows_by_game: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for market_row in market_rows:
        gid_raw = market_row.get("game_id")
        mt = str(market_row.get("market_type") or "")
        if gid_raw is None or not mt:
            continue
        gid = int(gid_raw)
        market_rows_by_game.setdefault(gid, {}).setdefault(mt, []).append(market_row)

    for game in board_rows:
        gid = int(game.get("game_id") or 0)
        if not gid or gid in existing:
            continue
        gr = row_by.get(gid)
        if gr is None or not _is_game_top_ev_snapshot_lock_active(gr):
            continue
        if not is_before_scheduled_first_pitch(gr.get("game_start_ts")):
            continue
        pick = _top_ev_pick_for_board_row(gr, market_rows_by_game.get(gid, {}))
        if not pick:
            continue
        _insert_board_top_ev_snapshot_row(target_date, gid, pick)


def _flatten_best_bets(
    rows: list[dict[str, Any]],
    limit: int | None = None,
    *,
    target_date: date | None = None,
) -> list[dict[str, Any]]:
    live = best_bets_utils.flatten_best_bets(rows, limit)
    if target_date is None:
        return live
    _maybe_insert_board_top_ev_run_snapshots(target_date, rows)
    _maybe_insert_board_green_run_snapshots(target_date, rows, live)
    _maybe_insert_board_green_snapshots(target_date, rows, live)
    _maybe_insert_board_top_ev_snapshots(target_date, rows)
    return _merge_board_green_snapshots_into_live(target_date, rows, live)


def _flatten_watchlist_markets(
    rows: list[dict[str, Any]],
    limit: int = best_bets_utils.BOARD_WATCHLIST_LIMIT,
    *,
    secondary_lines_only: bool = False,
) -> list[dict[str, Any]]:
    return best_bets_utils.flatten_watchlist_markets(
        rows, limit, secondary_lines_only=secondary_lines_only
    )


def _fetch_bullpen_context_map(target_date: date, team: str | None = None) -> dict[str, dict[str, Any]]:
    if not _table_exists("bullpens_daily"):
        return {}
    filters = ["game_date < :target_date", "season = :season"]
    params: dict[str, Any] = {"target_date": target_date, "season": target_date.year}
    if team is not None:
        filters.append("team = :team")
        params["team"] = team
    frame = _safe_frame(
        f"""
        SELECT
            team,
            game_date,
            innings_pitched,
            earned_runs,
            late_innings_pitched,
            late_earned_runs
        FROM bullpens_daily
        WHERE {' AND '.join(filters)}
        ORDER BY team, game_date
        """,
        params,
    )
    if frame.empty:
        return {}

    context_map: dict[str, dict[str, Any]] = {}
    for team_name, team_frame in frame.groupby("team", dropna=False):
        outs = pd.to_numeric(team_frame.get("innings_pitched"), errors="coerce").apply(_baseball_ip_to_outs).sum()
        late_outs = pd.to_numeric(team_frame.get("late_innings_pitched"), errors="coerce").apply(_baseball_ip_to_outs).sum()
        earned_runs = pd.to_numeric(team_frame.get("earned_runs"), errors="coerce").fillna(0).sum()
        late_earned_runs = pd.to_numeric(team_frame.get("late_earned_runs"), errors="coerce").fillna(0).sum()
        season_era = None if outs <= 0 else float(earned_runs) * 27.0 / float(outs)
        late_season_era = None if late_outs <= 0 else float(late_earned_runs) * 27.0 / float(late_outs)
        context_map[str(team_name)] = {
            "season_games": int(len(team_frame.index)),
            "season_era": season_era,
            "late_season_era": late_season_era,
        }
    return context_map


def _fetch_h2h_totals_for_game(game_id: int) -> dict[str, Any]:
    """Same-calendar-year head-to-head scoring rollup (aligned with GET /matchups head_to_head)."""
    y_g = _sql_year("g.game_date")
    y_this = _sql_year("(SELECT game_date FROM this_game)")
    frame = _safe_frame(
        f"""
        WITH this_game AS (
            SELECT home_team, away_team, game_date
            FROM games WHERE game_id = :game_id
        ),
        prior AS (
            SELECT g.game_id, g.game_date,
                   g.home_team, g.away_team,
                   g.home_runs, g.away_runs,
                   (g.home_runs + g.away_runs) AS total_runs,
                   gm.line_value AS market_total
            FROM games g
            JOIN this_game tg
              ON ((g.home_team = tg.home_team AND g.away_team = tg.away_team)
               OR (g.home_team = tg.away_team AND g.away_team = tg.home_team))
            LEFT JOIN game_markets gm
              ON gm.game_id = g.game_id AND gm.market_type = 'total'
            WHERE g.status = 'final'
              AND g.game_date < tg.game_date
              AND {y_g} = {y_this}
        )
        SELECT
            COUNT(*)                                     AS games_played,
            SUM(CASE WHEN total_runs IS NOT NULL
                     THEN total_runs ELSE 0 END) * 1.0
                / NULLIF(COUNT(*), 0)                    AS avg_total_runs,
            SUM(CASE WHEN market_total IS NOT NULL
                          AND total_runs > market_total
                     THEN 1 ELSE 0 END) * 1.0
                / NULLIF(SUM(CASE WHEN market_total IS NOT NULL
                                  THEN 1 ELSE 0 END), 0) AS over_pct,
            (SELECT home_team FROM this_game)             AS home_team,
            (SELECT away_team FROM this_game)             AS away_team,
            (SELECT {_sql_year('game_date')} FROM this_game) AS season_year
        FROM prior
        """,
        {"game_id": game_id},
    )
    rows = _frame_records(frame)
    if not rows:
        return {}
    h2h = dict(rows[0])
    gp = h2h.get("games_played")
    gp_int = int(gp) if gp is not None else 0
    if gp_int < MATCHUP_H2H_ADEQUATE_MIN_GAMES:
        tier = "low"
    elif gp_int < MATCHUP_H2H_STRONG_MIN_GAMES:
        tier = "adequate"
    else:
        tier = "strong"
    h2h["sample_tier"] = tier
    h2h["h2h_window"] = "same_calendar_year"
    return h2h


def _fetch_recent_bullpen_map(target_date: date, team: str | None = None) -> dict[str, dict[str, Any]]:
    if not _table_exists("bullpens_daily"):
        return {}
    filters = ["game_date < :target_date"]
    params: dict[str, Any] = {"target_date": target_date}
    if team is not None:
        filters.append("team = :team")
        params["team"] = team
    frame = _safe_frame(
        f"""
        SELECT
            team,
            game_date,
            innings_pitched,
            pitches_thrown,
            earned_runs,
            hits_allowed,
            late_innings_pitched,
            late_runs_allowed,
            late_earned_runs,
            late_hits_allowed
        FROM bullpens_daily
        WHERE {' AND '.join(filters)}
        ORDER BY team, game_date
        """,
        params,
    )
    if frame.empty:
        return {}

    snapshot_map: dict[str, dict[str, Any]] = {}
    previous_day = target_date - timedelta(days=1)
    for team_name, team_frame in frame.groupby("team", dropna=False):
        ordered = team_frame.sort_values("game_date")
        last3 = ordered.tail(3).copy()
        last5 = ordered.tail(5).copy()
        last10 = ordered.tail(10).copy()
        outs = pd.to_numeric(last3.get("innings_pitched"), errors="coerce").apply(_baseball_ip_to_outs).sum()
        late_outs = pd.to_numeric(last3.get("late_innings_pitched"), errors="coerce").apply(_baseball_ip_to_outs).sum()
        earned_runs = pd.to_numeric(last3.get("earned_runs"), errors="coerce").fillna(0).sum()
        hits_allowed = pd.to_numeric(last3.get("hits_allowed"), errors="coerce").fillna(0).sum()
        late_runs_allowed = pd.to_numeric(last3.get("late_runs_allowed"), errors="coerce").fillna(0).sum()
        late_earned_runs = pd.to_numeric(last3.get("late_earned_runs"), errors="coerce").fillna(0).sum()
        late_hits_allowed = pd.to_numeric(last3.get("late_hits_allowed"), errors="coerce").fillna(0).sum()
        innings_decimal = outs / 3 if outs else 0
        late_innings_decimal = late_outs / 3 if late_outs else 0
        late_outs_5 = (
            int(pd.to_numeric(last5.get("late_innings_pitched"), errors="coerce").apply(_baseball_ip_to_outs).sum())
            if not last5.empty
            else 0
        )
        late_runs_5 = pd.to_numeric(last5.get("late_runs_allowed"), errors="coerce").fillna(0).sum() if not last5.empty else 0
        late_er_5 = pd.to_numeric(last5.get("late_earned_runs"), errors="coerce").fillna(0).sum() if not last5.empty else 0
        late_hits_5 = pd.to_numeric(last5.get("late_hits_allowed"), errors="coerce").fillna(0).sum() if not last5.empty else 0
        late_inn_dec_5 = late_outs_5 / 3 if late_outs_5 else 0
        late_era_5 = None if late_inn_dec_5 <= 0 else float(late_er_5) * 9.0 / float(late_inn_dec_5)
        late_outs_10 = (
            int(pd.to_numeric(last10.get("late_innings_pitched"), errors="coerce").apply(_baseball_ip_to_outs).sum())
            if not last10.empty
            else 0
        )
        late_er_10 = pd.to_numeric(last10.get("late_earned_runs"), errors="coerce").fillna(0).sum() if not last10.empty else 0
        late_hits_10 = pd.to_numeric(last10.get("late_hits_allowed"), errors="coerce").fillna(0).sum() if not last10.empty else 0
        late_runs_10 = pd.to_numeric(last10.get("late_runs_allowed"), errors="coerce").fillna(0).sum() if not last10.empty else 0
        late_inn_dec_10 = late_outs_10 / 3 if late_outs_10 else 0
        late_era_10 = None if late_inn_dec_10 <= 0 else float(late_er_10) * 9.0 / float(late_inn_dec_10)
        game_dates = pd.to_datetime(ordered.get("game_date"), errors="coerce").dt.date
        snapshot_map[str(team_name)] = {
            "pitches_last3": int(pd.to_numeric(last3.get("pitches_thrown"), errors="coerce").fillna(0).sum()),
            "innings_last3": _baseball_ip_from_outs(outs),
            "b2b": int((game_dates == previous_day).any()) if not ordered.empty else 0,
            "runs_allowed_last3": int(earned_runs),
            "earned_runs_last3": int(earned_runs),
            "hits_allowed_last3": int(hits_allowed),
            "era_last3": None if innings_decimal <= 0 else float(earned_runs) * 9.0 / float(innings_decimal),
            "late_innings_last3": _baseball_ip_from_outs(late_outs),
            "late_runs_allowed_last3": int(late_runs_allowed),
            "late_earned_runs_last3": int(late_earned_runs),
            "late_hits_allowed_last3": int(late_hits_allowed),
            "late_era_last3": None if late_innings_decimal <= 0 else float(late_earned_runs) * 9.0 / float(late_innings_decimal),
            "late_innings_last5": _baseball_ip_from_outs(late_outs_5),
            "late_runs_allowed_last5": int(late_runs_5),
            "late_earned_runs_last5": int(late_er_5),
            "late_hits_allowed_last5": int(late_hits_5),
            "late_era_last5": late_era_5,
            "late_bullpen_games_in_sample": int(len(last5.index)),
            "late_innings_last10": _baseball_ip_from_outs(late_outs_10),
            "late_runs_allowed_last10": int(late_runs_10),
            "late_earned_runs_last10": int(late_er_10),
            "late_hits_allowed_last10": int(late_hits_10),
            "late_era_last10": late_era_10,
            "late_bullpen_games_in_sample_10": int(len(last10.index)),
        }
    return snapshot_map


def _fetch_umpire_map(target_date: date) -> dict[int, dict[str, Any]]:
    """Home plate umpire keyed by game_id."""
    if not _table_exists("umpire_assignments"):
        return {}
    frame = _safe_frame(
        """
        SELECT ua.game_id, ua.umpire_name, ua.umpire_id
        FROM umpire_assignments ua
        INNER JOIN (
            SELECT game_id, MAX(snapshot_ts) AS max_ts
            FROM umpire_assignments
            WHERE game_date = :target_date
            GROUP BY game_id
        ) latest ON ua.game_id = latest.game_id AND ua.snapshot_ts = latest.max_ts
        WHERE ua.game_date = :target_date
        """,
        {"target_date": target_date},
    )
    return {
        int(r["game_id"]): {"umpire_name": r.get("umpire_name"), "umpire_id": r.get("umpire_id")}
        for r in _frame_records(frame)
    }


def _fetch_total_bases_prediction_map(target_date: date) -> dict[tuple[int, int], dict[str, Any]]:
    """TB predictions keyed by (game_id, player_id)."""
    if not _table_exists("predictions_total_bases"):
        return {}
    frame = _safe_frame(
        """
        SELECT ptb.game_id, ptb.player_id, ptb.player_name, ptb.team,
               ptb.predicted_tb, ptb.over_probability, ptb.under_probability,
               ptb.market_line, ptb.market_over_price, ptb.market_under_price, ptb.edge
        FROM predictions_total_bases ptb
        INNER JOIN (
            SELECT game_id, player_id, MAX(prediction_ts) AS max_ts
            FROM predictions_total_bases
            WHERE game_date = :target_date
            GROUP BY game_id, player_id
        ) latest ON ptb.game_id = latest.game_id
               AND ptb.player_id = latest.player_id
               AND ptb.prediction_ts = latest.max_ts
        WHERE ptb.game_date = :target_date
        """,
        {"target_date": target_date},
    )
    return {(int(r["game_id"]), int(r["player_id"])): r for r in _frame_records(frame)}


def _fetch_hr_prediction_map(target_date: date) -> dict[tuple[int, int], dict[str, Any]]:
    """Latest HR yes-line prediction per (game_id, player_id)."""
    if not _table_exists("predictions_player_hr"):
        return {}
    frame = _safe_frame(
        """
        SELECT ph.game_id, ph.player_id,
               ph.predicted_hr_probability, ph.fair_price, ph.market_price, ph.edge
        FROM predictions_player_hr ph
        INNER JOIN (
            SELECT game_id, player_id, MAX(prediction_ts) AS max_ts
            FROM predictions_player_hr
            WHERE game_date = :target_date
            GROUP BY game_id, player_id
        ) latest ON ph.game_id = latest.game_id
               AND ph.player_id = latest.player_id
               AND ph.prediction_ts = latest.max_ts
        WHERE ph.game_date = :target_date
        """,
        {"target_date": target_date},
    )
    return {(int(r["game_id"]), int(r["player_id"])): dict(r) for r in _frame_records(frame)}


def _fetch_pitcher_prop_map(target_date: date) -> dict[tuple[int, int], dict[str, Any]]:
    """Pitcher props (hits_allowed, earned_runs, walks) keyed by (game_id, player_id)."""
    if not _table_exists("player_prop_markets"):
        return {}
    frame = _safe_frame(
        """
        SELECT ppm.game_id, ppm.player_id, ppm.market_type,
               ppm.line_value, ppm.over_price, ppm.under_price, ppm.sportsbook
        FROM player_prop_markets ppm
        INNER JOIN (
            SELECT game_id, player_id, market_type, MAX(snapshot_ts) AS max_ts
            FROM player_prop_markets
            WHERE game_date = :target_date
              AND market_type IN ('pitcher_hits_allowed', 'pitcher_earned_runs', 'pitcher_walks')
            GROUP BY game_id, player_id, market_type
        ) latest ON ppm.game_id = latest.game_id
               AND ppm.player_id = latest.player_id
               AND ppm.market_type = latest.market_type
               AND ppm.snapshot_ts = latest.max_ts
        WHERE ppm.game_date = :target_date
        """,
        {"target_date": target_date},
    )
    result: dict[tuple[int, int], dict] = {}
    for row in _frame_records(frame):
        key = (int(row["game_id"]), int(row["player_id"]))
        if key not in result:
            result[key] = {}
        mt = str(row.get("market_type") or "")
        result[key][mt] = {
            "line": _to_float(row.get("line_value")),
            "over_price": row.get("over_price"),
            "under_price": row.get("under_price"),
        }
    return result


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
        # Use stats-only (fundamentals) as primary; fall back to blended.
        fundamentals_total = _to_float(totals.get("predicted_total_fundamentals"))
        blended_total = _to_float(totals.get("predicted_total_runs"))
        predicted_total = fundamentals_total if fundamentals_total is not None else blended_total
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
        "avg_earned_runs": None,
        "avg_pitch_count": None,
        "xwoba_against": None,
        "csw_pct": None,
        "whiff_pct": None,
        "avg_fb_velo": None,
        "last_start_date": None,
        "season_starts": 0,
        "season_era": None,
        "era_last3": None,
        "era_last5": None,
        "recent_starts": [],
        "form_basis": "none",
    }
    if pitcher_id is None:
        return default
    has_starts_tbl = _table_exists("pitcher_starts")
    has_pgp_tbl = _table_exists("player_game_pitching")
    if not has_starts_tbl and not has_pgp_tbl:
        return default

    starts_frame = pd.DataFrame()
    if has_starts_tbl:
        starts_frame = _safe_frame(
            """
            SELECT
                game_date,
                ip,
                earned_runs,
                strikeouts,
                walks,
                pitch_count,
                xwoba_against,
                csw_pct,
                whiff_pct,
                avg_fb_velo
            FROM pitcher_starts
            WHERE pitcher_id = :pitcher_id
              AND game_date < :target_date
            ORDER BY game_date DESC, game_id DESC
            """,
            {"pitcher_id": pitcher_id, "target_date": target_date},
        )
    appearances_frame = pd.DataFrame()
    if has_pgp_tbl:
        appearances_frame = _safe_frame(
            """
            SELECT
                game_date,
                innings_pitched AS ip,
                earned_runs,
                strikeouts,
                walks,
                pitches_thrown AS pitch_count,
                xwoba_allowed AS xwoba_against
            FROM player_game_pitching
            WHERE player_id = :pitcher_id
              AND game_date < :target_date
            ORDER BY game_date DESC, game_id DESC
            """,
            {"pitcher_id": pitcher_id, "target_date": target_date},
        )

    n_starts = 0 if starts_frame.empty else int(len(starts_frame.index))
    n_app = 0 if appearances_frame.empty else int(len(appearances_frame.index))

    if n_starts == 0 and n_app == 0:
        return default

    if n_starts >= _MIN_START_ROWS_FOR_START_ONLY_RECENT_FORM:
        history = starts_frame.copy()
        form_basis = "starts"
        outing_source: str | None = "starts"
    elif n_app > 0:
        history = appearances_frame.copy()
        for col in ("csw_pct", "whiff_pct", "avg_fb_velo"):
            history[col] = float("nan")
        form_basis = "appearances"
        outing_source = "appearances"
    else:
        history = starts_frame.copy()
        form_basis = "starts"
        outing_source = "starts"

    history["game_date"] = pd.to_datetime(history["game_date"])
    recent_frame = history.head(5).copy()
    recent3_frame = recent_frame.head(3).copy()
    season_frame = history[history["game_date"].dt.year == int(target_date.year)].copy()

    def _mean(column: str) -> float | None:
        if recent_frame.empty or column not in recent_frame.columns:
            return None
        value = pd.to_numeric(recent_frame[column], errors="coerce").mean()
        return None if pd.isna(value) else float(value)

    record = {
        "sample_starts": int(len(recent_frame.index)),
        "avg_ip": _mean("ip"),
        "avg_strikeouts": _mean("strikeouts"),
        "avg_walks": _mean("walks"),
        "avg_earned_runs": _mean("earned_runs"),
        "avg_pitch_count": _mean("pitch_count"),
        "xwoba_against": _mean("xwoba_against"),
        "csw_pct": _mean("csw_pct"),
        "whiff_pct": _mean("whiff_pct"),
        "avg_fb_velo": _mean("avg_fb_velo"),
        "last_start_date": None if recent_frame.empty else recent_frame.iloc[0]["game_date"].date().isoformat(),
        "season_starts": int(len(season_frame.index)),
        "season_era": _era_from_pitcher_history(season_frame),
        "era_last3": _era_from_pitcher_history(recent3_frame),
        "era_last5": _era_from_pitcher_history(recent_frame),
        "form_basis": form_basis,
    }
    recent_starts = _fetch_pitcher_recent_starts(
        pitcher_id,
        target_date,
        limit=5,
        outing_source=outing_source,
    )
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
            g.venue_id,
            v.latitude AS venue_latitude,
            v.longitude AS venue_longitude,
            COALESCE(v.venue_name, g.venue_name) AS venue_name,
            v.city AS venue_city,
            v.state AS venue_state,
            v.roof_type,
            p.model_name,
            p.model_version,
            p.prediction_ts,
            p.predicted_total_runs,
            p.predicted_total_fundamentals,
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
            CAST(f.feature_payload ->> 'away_bullpen_late_innings_last3' AS DOUBLE PRECISION) AS away_bullpen_late_innings_last3,
            CAST(f.feature_payload ->> 'home_bullpen_late_innings_last3' AS DOUBLE PRECISION) AS home_bullpen_late_innings_last3,
            CAST(f.feature_payload ->> 'away_bullpen_late_runs_allowed_last3' AS DOUBLE PRECISION) AS away_bullpen_late_runs_allowed_last3,
            CAST(f.feature_payload ->> 'home_bullpen_late_runs_allowed_last3' AS DOUBLE PRECISION) AS home_bullpen_late_runs_allowed_last3,
            CAST(f.feature_payload ->> 'away_bullpen_late_earned_runs_last3' AS DOUBLE PRECISION) AS away_bullpen_late_earned_runs_last3,
            CAST(f.feature_payload ->> 'home_bullpen_late_earned_runs_last3' AS DOUBLE PRECISION) AS home_bullpen_late_earned_runs_last3,
            CAST(f.feature_payload ->> 'away_bullpen_late_hits_allowed_last3' AS DOUBLE PRECISION) AS away_bullpen_late_hits_allowed_last3,
            CAST(f.feature_payload ->> 'home_bullpen_late_hits_allowed_last3' AS DOUBLE PRECISION) AS home_bullpen_late_hits_allowed_last3,
            CAST(f.feature_payload ->> 'away_bullpen_late_era_last3' AS DOUBLE PRECISION) AS away_bullpen_late_era_last3,
            CAST(f.feature_payload ->> 'home_bullpen_late_era_last3' AS DOUBLE PRECISION) AS home_bullpen_late_era_last3,
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
            CAST(f.feature_payload ->> 'precipitation_pct' AS DOUBLE PRECISION) AS precipitation_pct,
            CAST(f.feature_payload ->> 'cloud_cover_pct' AS DOUBLE PRECISION) AS cloud_cover_pct,
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
    supplemental_market_rows = _fetch_latest_game_market_rows(target_date, game_id=int(game_id))
    supplemental_markets_map = _aggregate_supplemental_market_rows(supplemental_market_rows)
    game_market_rows_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for market_row in supplemental_market_rows:
        market_type = str(market_row.get("market_type") or "")
        if market_type:
            game_market_rows_by_type[market_type].append(market_row)
    bullpen_context_map = _fetch_bullpen_context_map(target_date)
    recent_bullpen_map = _fetch_recent_bullpen_map(target_date)

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
    starter_records = _starter_records_prefer_boxscore(
        _frame_records(starters_frame),
        target_date,
        game_id=int(game_id),
    )

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
                    DENSE_RANK() OVER (
                        PARTITION BY l.game_id, l.team
                        ORDER BY l.snapshot_ts DESC
                    ) AS snapshot_rank,
                    ROW_NUMBER() OVER (
                        PARTITION BY l.game_id, l.team, l.player_id
                        ORDER BY l.snapshot_ts DESC, l.lineup_slot ASC, l.player_id
                    ) AS row_rank
                FROM lineups l
                WHERE l.game_id = :game_id
                  AND l.game_date = :target_date
            ),
            selected_players AS (
                SELECT DISTINCT game_id, player_id
                FROM ranked_lineups
                WHERE snapshot_rank = 1
                  AND row_rank = 1
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
                    AND rl.snapshot_rank = 1
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

    market_freezes = _fetch_market_freeze_map(target_date)
    total_freeze = market_freezes.get((int(game_id), "total"), {})

    detail = {
        "game_id": int(game["game_id"]),
        "game_date": game["game_date"],
        "status": game["status"],
        "away_team": game["away_team"],
        "home_team": game["home_team"],
        "game_start_ts": game["game_start_ts"],
        "venue": {
            "venue_id": game.get("venue_id"),
            "name": game["venue_name"],
            "city": game["venue_city"],
            "state": game["venue_state"],
            "roof_type": game["roof_type"],
            "latitude": game.get("venue_latitude"),
            "longitude": game.get("venue_longitude"),
        },
        "weather": {
            "temperature_f": game["temperature_f"],
            "wind_speed_mph": game["wind_speed_mph"],
            "wind_direction_deg": game["wind_direction_deg"],
            "humidity_pct": game["humidity_pct"],
            "precipitation_pct": game.get("precipitation_pct"),
            "cloud_cover_pct": game.get("cloud_cover_pct"),
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
            "predicted_total_fundamentals": game.get("predicted_total_fundamentals"),
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
            "away_bullpen_pitches_last3": (recent_bullpen_map.get(str(game["away_team"]) or "", {}) or {}).get("pitches_last3", game["away_bullpen_pitches_last3"]),
            "home_bullpen_pitches_last3": (recent_bullpen_map.get(str(game["home_team"]) or "", {}) or {}).get("pitches_last3", game["home_bullpen_pitches_last3"]),
            "away_bullpen_innings_last3": (recent_bullpen_map.get(str(game["away_team"]) or "", {}) or {}).get("innings_last3", game["away_bullpen_innings_last3"]),
            "home_bullpen_innings_last3": (recent_bullpen_map.get(str(game["home_team"]) or "", {}) or {}).get("innings_last3", game["home_bullpen_innings_last3"]),
            "away_bullpen_b2b": (recent_bullpen_map.get(str(game["away_team"]) or "", {}) or {}).get("b2b", game["away_bullpen_b2b"]),
            "home_bullpen_b2b": (recent_bullpen_map.get(str(game["home_team"]) or "", {}) or {}).get("b2b", game["home_bullpen_b2b"]),
            "away_bullpen_runs_allowed_last3": (recent_bullpen_map.get(str(game["away_team"]) or "", {}) or {}).get("runs_allowed_last3", game["away_bullpen_runs_allowed_last3"]),
            "home_bullpen_runs_allowed_last3": (recent_bullpen_map.get(str(game["home_team"]) or "", {}) or {}).get("runs_allowed_last3", game["home_bullpen_runs_allowed_last3"]),
            "away_bullpen_earned_runs_last3": (recent_bullpen_map.get(str(game["away_team"]) or "", {}) or {}).get("earned_runs_last3", game["away_bullpen_earned_runs_last3"]),
            "home_bullpen_earned_runs_last3": (recent_bullpen_map.get(str(game["home_team"]) or "", {}) or {}).get("earned_runs_last3", game["home_bullpen_earned_runs_last3"]),
            "away_bullpen_hits_allowed_last3": (recent_bullpen_map.get(str(game["away_team"]) or "", {}) or {}).get("hits_allowed_last3", game["away_bullpen_hits_allowed_last3"]),
            "home_bullpen_hits_allowed_last3": (recent_bullpen_map.get(str(game["home_team"]) or "", {}) or {}).get("hits_allowed_last3", game["home_bullpen_hits_allowed_last3"]),
            "away_bullpen_era_last3": (recent_bullpen_map.get(str(game["away_team"]) or "", {}) or {}).get("era_last3", game["away_bullpen_era_last3"]),
            "home_bullpen_era_last3": (recent_bullpen_map.get(str(game["home_team"]) or "", {}) or {}).get("era_last3", game["home_bullpen_era_last3"]),
            "away_bullpen_late_innings_last3": (recent_bullpen_map.get(str(game["away_team"]) or "", {}) or {}).get("late_innings_last3", game["away_bullpen_late_innings_last3"]),
            "home_bullpen_late_innings_last3": (recent_bullpen_map.get(str(game["home_team"]) or "", {}) or {}).get("late_innings_last3", game["home_bullpen_late_innings_last3"]),
            "away_bullpen_late_runs_allowed_last3": (recent_bullpen_map.get(str(game["away_team"]) or "", {}) or {}).get("late_runs_allowed_last3", game["away_bullpen_late_runs_allowed_last3"]),
            "home_bullpen_late_runs_allowed_last3": (recent_bullpen_map.get(str(game["home_team"]) or "", {}) or {}).get("late_runs_allowed_last3", game["home_bullpen_late_runs_allowed_last3"]),
            "away_bullpen_late_earned_runs_last3": (recent_bullpen_map.get(str(game["away_team"]) or "", {}) or {}).get("late_earned_runs_last3", game["away_bullpen_late_earned_runs_last3"]),
            "home_bullpen_late_earned_runs_last3": (recent_bullpen_map.get(str(game["home_team"]) or "", {}) or {}).get("late_earned_runs_last3", game["home_bullpen_late_earned_runs_last3"]),
            "away_bullpen_late_hits_allowed_last3": (recent_bullpen_map.get(str(game["away_team"]) or "", {}) or {}).get("late_hits_allowed_last3", game["away_bullpen_late_hits_allowed_last3"]),
            "home_bullpen_late_hits_allowed_last3": (recent_bullpen_map.get(str(game["home_team"]) or "", {}) or {}).get("late_hits_allowed_last3", game["home_bullpen_late_hits_allowed_last3"]),
            "away_bullpen_late_era_last3": (recent_bullpen_map.get(str(game["away_team"]) or "", {}) or {}).get("late_era_last3", game["away_bullpen_late_era_last3"]),
            "home_bullpen_late_era_last3": (recent_bullpen_map.get(str(game["home_team"]) or "", {}) or {}).get("late_era_last3", game["home_bullpen_late_era_last3"]),
            "away_bullpen_late_innings_last5": (recent_bullpen_map.get(str(game["away_team"]) or "", {}) or {}).get("late_innings_last5"),
            "home_bullpen_late_innings_last5": (recent_bullpen_map.get(str(game["home_team"]) or "", {}) or {}).get("late_innings_last5"),
            "away_bullpen_late_runs_allowed_last5": (recent_bullpen_map.get(str(game["away_team"]) or "", {}) or {}).get("late_runs_allowed_last5"),
            "home_bullpen_late_runs_allowed_last5": (recent_bullpen_map.get(str(game["home_team"]) or "", {}) or {}).get("late_runs_allowed_last5"),
            "away_bullpen_late_earned_runs_last5": (recent_bullpen_map.get(str(game["away_team"]) or "", {}) or {}).get("late_earned_runs_last5"),
            "home_bullpen_late_earned_runs_last5": (recent_bullpen_map.get(str(game["home_team"]) or "", {}) or {}).get("late_earned_runs_last5"),
            "away_bullpen_late_hits_allowed_last5": (recent_bullpen_map.get(str(game["away_team"]) or "", {}) or {}).get("late_hits_allowed_last5"),
            "home_bullpen_late_hits_allowed_last5": (recent_bullpen_map.get(str(game["home_team"]) or "", {}) or {}).get("late_hits_allowed_last5"),
            "away_bullpen_late_era_last5": (recent_bullpen_map.get(str(game["away_team"]) or "", {}) or {}).get("late_era_last5"),
            "home_bullpen_late_era_last5": (recent_bullpen_map.get(str(game["home_team"]) or "", {}) or {}).get("late_era_last5"),
            "away_bullpen_late_innings_last10": (recent_bullpen_map.get(str(game["away_team"]) or "", {}) or {}).get("late_innings_last10"),
            "home_bullpen_late_innings_last10": (recent_bullpen_map.get(str(game["home_team"]) or "", {}) or {}).get("late_innings_last10"),
            "away_bullpen_late_runs_allowed_last10": (recent_bullpen_map.get(str(game["away_team"]) or "", {}) or {}).get("late_runs_allowed_last10"),
            "home_bullpen_late_runs_allowed_last10": (recent_bullpen_map.get(str(game["home_team"]) or "", {}) or {}).get("late_runs_allowed_last10"),
            "away_bullpen_late_earned_runs_last10": (recent_bullpen_map.get(str(game["away_team"]) or "", {}) or {}).get("late_earned_runs_last10"),
            "home_bullpen_late_earned_runs_last10": (recent_bullpen_map.get(str(game["home_team"]) or "", {}) or {}).get("late_earned_runs_last10"),
            "away_bullpen_late_hits_allowed_last10": (recent_bullpen_map.get(str(game["away_team"]) or "", {}) or {}).get("late_hits_allowed_last10"),
            "home_bullpen_late_hits_allowed_last10": (recent_bullpen_map.get(str(game["home_team"]) or "", {}) or {}).get("late_hits_allowed_last10"),
            "away_bullpen_late_era_last10": (recent_bullpen_map.get(str(game["away_team"]) or "", {}) or {}).get("late_era_last10"),
            "home_bullpen_late_era_last10": (recent_bullpen_map.get(str(game["home_team"]) or "", {}) or {}).get("late_era_last10"),
            "away_bullpen_late_games_sample": (recent_bullpen_map.get(str(game["away_team"]) or "", {}) or {}).get("late_bullpen_games_in_sample"),
            "home_bullpen_late_games_sample": (recent_bullpen_map.get(str(game["home_team"]) or "", {}) or {}).get("late_bullpen_games_in_sample"),
            "away_bullpen_late_games_sample_10": (recent_bullpen_map.get(str(game["away_team"]) or "", {}) or {}).get("late_bullpen_games_in_sample_10"),
            "home_bullpen_late_games_sample_10": (recent_bullpen_map.get(str(game["home_team"]) or "", {}) or {}).get("late_bullpen_games_in_sample_10"),
            "away_bullpen_season_era": (bullpen_context_map.get(str(game["away_team"]) or "", {}) or {}).get("season_era"),
            "home_bullpen_season_era": (bullpen_context_map.get(str(game["home_team"]) or "", {}) or {}).get("season_era"),
            "away_bullpen_late_season_era": (bullpen_context_map.get(str(game["away_team"]) or "", {}) or {}).get("late_season_era"),
            "home_bullpen_late_season_era": (bullpen_context_map.get(str(game["home_team"]) or "", {}) or {}).get("late_season_era"),
            "away_bullpen_season_games": (bullpen_context_map.get(str(game["away_team"]) or "", {}) or {}).get("season_games"),
            "home_bullpen_season_games": (bullpen_context_map.get(str(game["home_team"]) or "", {}) or {}).get("season_games"),
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
        "certainty": _build_certainty_payload(
            starter_certainty=game["starter_certainty_score"],
            lineup_certainty=game["lineup_certainty_score"],
            weather_freshness=game["weather_freshness_score"],
            market_freshness=game["market_freshness_score"],
            bullpen_completeness=game["bullpen_completeness_score"],
            missing_fallback_count=game["missing_fallback_count"],
            board_state=game["board_state"],
        ),
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
        "supplemental_markets": supplemental_markets_map.get(int(game_id), {}),
        "h2h_totals": _fetch_h2h_totals_for_game(int(game["game_id"])),
    }

    for starter in starter_records:
        side = None
        if starter["team"] == detail["away_team"]:
            side = "away"
        elif starter["team"] == detail["home_team"]:
            side = "home"
        if side is None:
            continue
        recent_form = _fetch_starter_recent_form(starter["pitcher_id"], target_date)
        xwoba_source = (
            "current_start_row"
            if starter.get("xwoba_against") is not None
            else "recent_average"
            if recent_form.get("xwoba_against") is not None
            else "missing"
        )
        csw_source = (
            "current_start_row"
            if starter.get("csw_pct") is not None
            else "recent_average"
            if recent_form.get("csw_pct") is not None
            else "missing"
        )
        velo_source = (
            "current_start_row"
            if starter.get("avg_fb_velo") is not None
            else "recent_average"
            if recent_form.get("avg_fb_velo") is not None
            else "missing"
        )
        whiff_source = (
            "current_start_row"
            if starter.get("whiff_pct") is not None
            else "recent_average"
            if recent_form.get("whiff_pct") is not None
            else "missing"
        )
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
            "xwoba_against": starter["xwoba_against"] if starter.get("xwoba_against") is not None else recent_form.get("xwoba_against"),
            "csw_pct": starter["csw_pct"] if starter.get("csw_pct") is not None else recent_form.get("csw_pct"),
            "avg_fb_velo": starter["avg_fb_velo"] if starter.get("avg_fb_velo") is not None else recent_form.get("avg_fb_velo"),
            "whiff_pct": starter["whiff_pct"] if starter.get("whiff_pct") is not None else recent_form.get("whiff_pct"),
            "advanced_stats_source": {
                "xwoba_against": xwoba_source,
                "csw_pct": csw_source,
                "avg_fb_velo": velo_source,
                "whiff_pct": whiff_source,
            },
            "recent_form": recent_form,
        }

    for side in ("away", "home"):
        st_side = detail["starters"][side]
        if not st_side or st_side.get("pitcher_id") is None:
            continue
        opp_team = detail["home_team"] if side == "away" else detail["away_team"]
        pvt_row = _fetch_pitcher_vs_team_matchup_row(int(st_side["pitcher_id"]), str(opp_team))
        if pvt_row:
            st_side["pitcher_vs_opponent_team"] = pvt_row

    for player in lineup_records:
        player.update(_build_hit_actual_meta(player.get("actual_hits"), is_final))
        player_id = player.get("player_id")
        live_history = [] if player_id is None else hit_history_map.get(int(player_id), [])
        _overlay_live_batting_stats(player, live_history)
        player["recent_hit_history"] = live_history[:10]
        side = "away" if player["team"] == detail["away_team"] else "home"
        opposing_starter = detail["starters"]["home"] if side == "away" else detail["starters"]["away"]
        _attach_hitter_matchup_context(
            player,
            opposing_starter["throws"] if opposing_starter else None,
            lineup_split_map,
        )
        player["form"] = classify_hitter_form(player)
        detail["teams"][side]["lineup"].append(player)

    bvp_pairs: list[tuple[int, int]] = []
    for side in ("away", "home"):
        opp_side = "home" if side == "away" else "away"
        st_o = detail["starters"].get(opp_side)
        if not st_o or st_o.get("pitcher_id") is None:
            continue
        pid_o = int(st_o["pitcher_id"])
        for pl in detail["teams"][side]["lineup"]:
            if pl.get("player_id") is not None:
                bvp_pairs.append((int(pl["player_id"]), pid_o))
    bvp_agg = _fetch_batter_vs_pitcher_map(target_date, bvp_pairs) if bvp_pairs else {}
    for side in ("away", "home"):
        opp_side = "home" if side == "away" else "away"
        st_o = detail["starters"].get(opp_side)
        if not st_o or st_o.get("pitcher_id") is None:
            continue
        pid_o = int(st_o["pitcher_id"])
        pname = str(st_o.get("pitcher_name") or "")
        for pl in detail["teams"][side]["lineup"]:
            bid = pl.get("player_id")
            if bid is None:
                continue
            pl["bvp_vs_today_starter"] = _compact_bvp_vs_pitcher_payload(
                bvp_agg.get((int(bid), pid_o)),
                pitcher_name=pname,
            )

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

    first5_totals_map = _fetch_first5_totals_map(target_date)
    detail["first5_totals"] = first5_totals_map.get(int(game_id)) or {"supported": False}
    _apply_market_freeze_payload(detail["totals"], total_freeze)
    _apply_market_freeze_payload(
        detail["first5_totals"],
        market_freezes.get((int(game_id), "first_five_total"), {}),
    )
    market_cards, best_bets = _build_market_cards_for_game(detail, game_market_rows_by_type)
    detail["market_cards"] = _drop_first_five_team_total_cards(market_cards)
    detail["best_bets"] = _drop_first_five_team_total_best_bets(best_bets)
    detail["top_ev_pick"] = _build_top_ev_pick_for_game_detail(detail, game_market_rows_by_type)

    gid_int = int(detail["game_id"])
    away_recent = _fetch_team_last_n_final_games(
        str(detail["away_team"] or ""),
        target_date=target_date,
        exclude_game_id=gid_int,
        limit=10,
    )
    home_recent = _fetch_team_last_n_final_games(
        str(detail["home_team"] or ""),
        target_date=target_date,
        exclude_game_id=gid_int,
        limit=10,
    )
    detail["team_recent_games"] = {
        "away": away_recent,
        "home": home_recent,
        "away_summary": _compute_team_recent_game_rollups(away_recent),
        "home_summary": _compute_team_recent_game_rollups(home_recent),
        "f5_coverage": {
            "away": _recent_games_f5_column_coverage(away_recent),
            "home": _recent_games_f5_column_coverage(home_recent),
        },
    }

    return detail


def _fetch_hr_source_recs_for_date(
    target_date: date,
    min_probability: float,
) -> list[dict[str, Any]]:
    """Latest HR prediction per batter; rows shaped for ``build_player_hr_board_card`` (+ daily extras)."""
    if not _table_exists("predictions_player_hr"):
        return []
    return load_hr_source_recs_for_date(target_date, min_probability)


def _fetch_slugger_hr_bets(
    target_date: date,
    min_probability: float,
    *,
    max_cards: int = 8,
) -> list[dict[str, Any]]:
    """HR-only board strip: best batters per game, then global cap. ``min_probability=0`` on the board."""
    source = _fetch_hr_source_recs_for_date(target_date, min_probability)
    if not source:
        return []
    return iter_slugger_tracked_cards(
        source,
        per_game=SLUGGER_HR_PER_GAME,
        max_cards=max_cards if max_cards > 0 else None,
    )


def _fetch_daily_home_run_rows(target_date: date) -> list[dict[str, Any]]:
    """Slugger HR picks (same ranking as the board), slate-capped for Daily Results, graded vs box score."""
    rows = _fetch_hr_source_recs_for_date(target_date, 0.0)
    if not rows:
        return []
    tracked = iter_slugger_tracked_cards(
        rows,
        per_game=SLUGGER_HR_PER_GAME,
        max_cards=SLUGGER_DAILY_RESULTS_MAX_CARDS,
    )
    rec_by_pair = {
        (int(r["game_id"] or 0), int(r["player_id"] or 0)): r
        for r in rows
    }
    out: list[dict[str, Any]] = []
    for card in tracked:
        gid = int(card.get("game_id") or 0)
        pid = int(card.get("player_id") or 0)
        rec = rec_by_pair.get((gid, pid))
        if rec is None:
            continue
        rank = int(card.get("slugger_rank_in_game") or 0)
        mp = _to_float(rec.get("predicted_hr_probability"))
        phr_disp = (
            best_bets_utils.format_hr_probability_pct_display(float(mp))
            if mp is not None
            else "—"
        )
        fair_raw = rec.get("fair_price")
        fair_disp = best_bets_utils._sanitize_hr_fair_american_for_display(fair_raw)
        is_final = _is_final_game_status(rec.get("game_status"))
        actual_hr = rec.get("actual_home_runs")
        if is_final and actual_hr is None:
            result = "missing"
        elif not is_final:
            result = "pending"
        elif float(actual_hr) > 0:
            result = "hit"
        else:
            result = "no_hit"
        out.append(
            {
                "game_id": gid,
                "game_date": rec.get("game_date"),
                "game_start_ts": rec.get("game_start_ts"),
                "game_status": rec.get("game_status"),
                "is_final": is_final,
                "player_id": pid,
                "player_name": rec.get("player_name"),
                "team": rec.get("team"),
                "opponent": rec.get("opponent"),
                "slugger_rank_in_game": rank,
                "predicted_hr_probability": mp,
                "hr_probability_display": phr_disp,
                "fair_price": fair_disp if fair_disp is not None else None,
                "market_price": rec.get("market_price"),
                "edge": rec.get("edge"),
                "actual_home_runs": actual_hr,
                "result": result,
                "market_backed": rec.get("market_price") is not None,
            }
        )

    def _sort_key(r: dict[str, Any]) -> tuple[Any, ...]:
        return (
            r.get("game_start_ts") or "",
            int(r.get("game_id") or 0),
            int(r.get("slugger_rank_in_game") or 99),
            str(r.get("player_name") or ""),
        )

    out.sort(key=_sort_key)
    return out


def _fetch_hot_hitters(
    target_date: date,
    min_probability: float,
    confirmed_only: bool,
    limit: int,
    include_inferred: bool,
    streak_only: bool = False,
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
            "hitter_best_bets": [],
            "slugger_hr_bets": [],
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
            SELECT DISTINCT rf.player_id
            FROM ranked_features rf
            LEFT JOIN dim_players dp ON dp.player_id = rf.player_id
            WHERE rf.row_rank = 1
              AND UPPER(COALESCE(dp.position, '')) NOT IN ('P', 'SP', 'RP', 'CP')
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
            ORDER BY {game_start_order}, team, {lineup_slot_order}, player_name, f.game_id
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
    tb_pred_map = _fetch_total_bases_prediction_map(target_date)
    hr_pred_map = _fetch_hr_prediction_map(target_date)
    bvp_matchups = [
        (int(record["player_id"]), int(record["opposing_pitcher_id"]))
        for record in records
        if record.get("player_id") is not None and record.get("opposing_pitcher_id") is not None
    ]
    bvp_map = _fetch_batter_vs_pitcher_map(target_date, bvp_matchups)
    player_status_map = _fetch_player_status_map(target_date, records)
    hit_history_map = _fetch_recent_hit_history_map(
        target_date,
        [int(record["player_id"]) for record in records if record.get("player_id") is not None],
        limit=30,
    )

    all_hot_rows: list[dict[str, Any]] = []
    for record in records:
        player_id = int(record["player_id"]) if record.get("player_id") is not None else None
        pitcher_id = int(record["opposing_pitcher_id"]) if record.get("opposing_pitcher_id") is not None else None
        live_history = [] if player_id is None else hit_history_map.get(player_id, [])
        _overlay_live_batting_stats(record, live_history)
        form = classify_hitter_form(record)
        if (form.get("form_key") or "") not in HOT_HITTER_PAGE_FORM_KEYS:
            continue
        bvp_stats = bvp_map.get((player_id, pitcher_id)) if player_id and pitcher_id else None
        actual_meta = _build_hit_actual_meta(record.get("actual_hits"), _is_final_game_status(record.get("game_status")))
        game_id_int = int(record["game_id"]) if record.get("game_id") is not None else None
        enriched = _attach_hitter_matchup_context(
            _attach_player_status_context(
                {
                **record,
                **actual_meta,
                "form": form,
                "recent_hit_history": live_history[:10],
                "bvp": bvp_stats,
                "total_bases": tb_pred_map.get((game_id_int, player_id)) if game_id_int and player_id else None,
                "hr_prediction": hr_pred_map.get((game_id_int, player_id)) if game_id_int and player_id else None,
            },
                player_status_map,
            ),
            record.get("opposing_pitcher_throws"),
            hit_split_map,
        )
        all_hot_rows.append(enriched)

    if min_probability > 0:
        hot_rows_after_prob = [
            row
            for row in all_hot_rows
            if (_to_float(row.get("predicted_hit_probability")) or 0.0) >= min_probability
        ]
    else:
        hot_rows_after_prob = list(all_hot_rows)

    if streak_only:
        hot_rows = [
            row
            for row in hot_rows_after_prob
            if _row_hit_streak_value(row) >= 2
        ]
        total_hot_count_summary = len(hot_rows)
    else:
        hot_rows = list(hot_rows_after_prob)
        total_hot_count_summary = len(all_hot_rows)

    if streak_only:
        hot_rows.sort(
            key=lambda row: (
                -_row_hit_streak_value(row),
                -float((row.get("form") or {}).get("heat_score") or 0.0),
                -float(_to_float(row.get("predicted_hit_probability")) or 0.0),
                -float(_to_float(row.get("edge")) or 0.0),
                int(row.get("lineup_slot") or 99),
                str(row.get("player_name") or ""),
            )
        )
    else:
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

    hitter_best_bets: list[dict[str, Any]] = []
    for row in hot_rows:
        card = best_bets_utils.build_player_hits_board_card(row)
        if card:
            hitter_best_bets.append(best_bets_utils.annotate_market_card_for_display(card))
    hitter_best_bets.sort(
        key=lambda c: (
            float(c.get("weighted_ev") or -999.0),
            float(c.get("probability_edge") or -999.0),
        ),
        reverse=True,
    )

    slugger_hr_bets = _fetch_slugger_hr_bets(
        target_date,
        min_probability,
        max_cards=SLUGGER_HOT_HITTERS_PAGE_MAX_CARDS,
    )

    probabilities = [_to_float(row.get("predicted_hit_probability")) for row in hot_rows]
    valid_probabilities = [value for value in probabilities if value is not None]
    return {
        "rows": hot_rows,
        "hitter_best_bets": hitter_best_bets,
        "slugger_hr_bets": slugger_hr_bets,
        "summary": {
            "count": len(hot_rows),
            "total_hot_count": total_hot_count_summary,
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


def _fetch_season_leaderboards(target_date: date, limit: int = 10) -> dict[str, Any]:
    season_start = date(int(target_date.year), 1, 1)

    pitcher_rows = _safe_frame(
        """
        WITH season_rows AS (
            SELECT
                ps.pitcher_id,
                COALESCE(dp.full_name, CAST(ps.pitcher_id AS TEXT)) AS player_name,
                ps.team,
                ps.game_date,
                COALESCE(ps.strikeouts, 0) AS strikeouts,
                COALESCE(ps.batters_faced, 0) AS batters_faced
            FROM pitcher_starts ps
            LEFT JOIN dim_players dp ON dp.player_id = ps.pitcher_id
            WHERE ps.game_date >= :season_start
              AND ps.game_date < :target_date
        ),
        latest_team AS (
            SELECT
                pitcher_id,
                team,
                ROW_NUMBER() OVER (
                    PARTITION BY pitcher_id
                    ORDER BY game_date DESC, pitcher_id
                ) AS row_rank
            FROM season_rows
        )
        SELECT
            sr.pitcher_id,
            MAX(sr.player_name) AS player_name,
            lt.team,
            COUNT(*) AS starts,
            SUM(sr.strikeouts) AS strikeouts,
            CASE WHEN COUNT(*) <= 0 THEN NULL ELSE SUM(sr.strikeouts) * 1.0 / COUNT(*) END AS strikeouts_per_start,
            CASE WHEN SUM(sr.batters_faced) <= 0 THEN NULL ELSE SUM(sr.strikeouts) * 1.0 / SUM(sr.batters_faced) END AS strikeouts_per_batter
        FROM season_rows sr
        LEFT JOIN latest_team lt ON lt.pitcher_id = sr.pitcher_id AND lt.row_rank = 1
        GROUP BY sr.pitcher_id, lt.team
        HAVING COUNT(*) > 0
        ORDER BY strikeouts DESC, strikeouts_per_start DESC, player_name ASC
        LIMIT :limit
        """,
        {"season_start": season_start, "target_date": target_date, "limit": limit},
    )

    hitter_rows = _safe_frame(
        """
        WITH season_rows AS (
            SELECT
                pgb.player_id,
                COALESCE(dp.full_name, CAST(pgb.player_id AS TEXT)) AS player_name,
                pgb.team,
                pgb.game_date,
                COALESCE(pgb.hits, 0) AS hits,
                pgb.game_id
            FROM player_game_batting pgb
            LEFT JOIN dim_players dp ON dp.player_id = pgb.player_id
            WHERE pgb.game_date >= :season_start
              AND pgb.game_date < :target_date
        ),
        latest_team AS (
            SELECT
                player_id,
                team,
                ROW_NUMBER() OVER (
                    PARTITION BY player_id
                    ORDER BY game_date DESC, player_id
                ) AS row_rank
            FROM season_rows
        )
        SELECT
            sr.player_id,
            MAX(sr.player_name) AS player_name,
            lt.team,
            COUNT(DISTINCT sr.game_id) AS games,
            SUM(sr.hits) AS hits,
            CASE WHEN COUNT(DISTINCT sr.game_id) <= 0 THEN NULL ELSE SUM(sr.hits) * 1.0 / COUNT(DISTINCT sr.game_id) END AS hits_per_game
        FROM season_rows sr
        LEFT JOIN latest_team lt ON lt.player_id = sr.player_id AND lt.row_rank = 1
        GROUP BY sr.player_id, lt.team
        HAVING COUNT(DISTINCT sr.game_id) > 0
        ORDER BY hits DESC, hits_per_game DESC, player_name ASC
        LIMIT :limit
        """,
        {"season_start": season_start, "target_date": target_date, "limit": limit},
    )

    team_runs_rows = _safe_frame(
        """
        WITH season_rows AS (
            SELECT away_team AS team, away_runs AS runs, game_id
            FROM games
            WHERE game_date >= :season_start AND game_date < :target_date AND away_runs IS NOT NULL
            UNION ALL
            SELECT home_team AS team, home_runs AS runs, game_id
            FROM games
            WHERE game_date >= :season_start AND game_date < :target_date AND home_runs IS NOT NULL
        )
        SELECT
            team,
            COUNT(DISTINCT game_id) AS games,
            SUM(runs) AS runs,
            CASE WHEN COUNT(DISTINCT game_id) <= 0 THEN NULL ELSE SUM(runs) * 1.0 / COUNT(DISTINCT game_id) END AS runs_per_game
        FROM season_rows
        GROUP BY team
        HAVING COUNT(DISTINCT game_id) > 0
        ORDER BY runs DESC, runs_per_game DESC, team ASC
        LIMIT :limit
        """,
        {"season_start": season_start, "target_date": target_date, "limit": limit},
    )

    team_strikeout_rows = _safe_frame(
        """
        SELECT
            team,
            COUNT(DISTINCT game_id) AS games,
            SUM(COALESCE(strikeouts, 0)) AS strikeouts,
            CASE WHEN COUNT(DISTINCT game_id) <= 0 THEN NULL ELSE SUM(COALESCE(strikeouts, 0)) * 1.0 / COUNT(DISTINCT game_id) END AS strikeouts_per_game
        FROM player_game_batting
        WHERE game_date >= :season_start
          AND game_date < :target_date
        GROUP BY team
        HAVING COUNT(DISTINCT game_id) > 0
        ORDER BY strikeouts DESC, strikeouts_per_game DESC, team ASC
        LIMIT :limit
        """,
        {"season_start": season_start, "target_date": target_date, "limit": limit},
    )

    pitcher_era_rows = _safe_frame(
        """
        WITH season_rows AS (
            SELECT
                ps.pitcher_id,
                COALESCE(dp.full_name, CAST(ps.pitcher_id AS TEXT)) AS player_name,
                ps.team,
                ps.game_date,
                COALESCE(ps.ip, 0) AS ip,
                COALESCE(ps.earned_runs, 0) AS earned_runs
            FROM pitcher_starts ps
            LEFT JOIN dim_players dp ON dp.player_id = ps.pitcher_id
            WHERE ps.game_date >= :season_start
              AND ps.game_date < :target_date
        ),
        latest_team AS (
            SELECT
                pitcher_id,
                team,
                ROW_NUMBER() OVER (
                    PARTITION BY pitcher_id
                    ORDER BY game_date DESC, pitcher_id
                ) AS row_rank
            FROM season_rows
        )
        SELECT
            sr.pitcher_id,
            MAX(sr.player_name) AS player_name,
            lt.team,
            COUNT(*) AS starts,
            SUM(sr.ip) AS innings_pitched,
            SUM(sr.earned_runs) AS earned_runs,
            CASE
                WHEN SUM(sr.ip) > 0 THEN 9.0 * SUM(sr.earned_runs) * 1.0 / SUM(sr.ip)
                ELSE NULL
            END AS era
        FROM season_rows sr
        LEFT JOIN latest_team lt ON lt.pitcher_id = sr.pitcher_id AND lt.row_rank = 1
        GROUP BY sr.pitcher_id, lt.team
        HAVING SUM(sr.ip) >= 3
        ORDER BY era ASC, innings_pitched DESC, player_name ASC
        LIMIT :limit
        """,
        {"season_start": season_start, "target_date": target_date, "limit": limit},
    )

    pitcher_whip_rows = _safe_frame(
        """
        WITH season_rows AS (
            SELECT
                ps.pitcher_id,
                COALESCE(dp.full_name, CAST(ps.pitcher_id AS TEXT)) AS player_name,
                ps.team,
                ps.game_date,
                COALESCE(ps.ip, 0) AS ip,
                COALESCE(ps.hits_allowed, 0) AS hits_allowed,
                COALESCE(ps.walks, 0) AS walks
            FROM pitcher_starts ps
            LEFT JOIN dim_players dp ON dp.player_id = ps.pitcher_id
            WHERE ps.game_date >= :season_start
              AND ps.game_date < :target_date
        ),
        latest_team AS (
            SELECT
                pitcher_id,
                team,
                ROW_NUMBER() OVER (
                    PARTITION BY pitcher_id
                    ORDER BY game_date DESC, pitcher_id
                ) AS row_rank
            FROM season_rows
        )
        SELECT
            sr.pitcher_id,
            MAX(sr.player_name) AS player_name,
            lt.team,
            COUNT(*) AS starts,
            SUM(sr.ip) AS innings_pitched,
            CASE
                WHEN SUM(sr.ip) > 0 THEN (SUM(sr.hits_allowed) + SUM(sr.walks)) * 1.0 / SUM(sr.ip)
                ELSE NULL
            END AS whip
        FROM season_rows sr
        LEFT JOIN latest_team lt ON lt.pitcher_id = sr.pitcher_id AND lt.row_rank = 1
        GROUP BY sr.pitcher_id, lt.team
        HAVING SUM(sr.ip) >= 3
        ORDER BY whip ASC, innings_pitched DESC, player_name ASC
        LIMIT :limit
        """,
        {"season_start": season_start, "target_date": target_date, "limit": limit},
    )

    hitter_home_run_rows = _safe_frame(
        """
        WITH season_rows AS (
            SELECT
                pgb.player_id,
                COALESCE(dp.full_name, CAST(pgb.player_id AS TEXT)) AS player_name,
                pgb.team,
                pgb.game_date,
                COALESCE(pgb.home_runs, 0) AS home_runs,
                pgb.game_id
            FROM player_game_batting pgb
            LEFT JOIN dim_players dp ON dp.player_id = pgb.player_id
            WHERE pgb.game_date >= :season_start
              AND pgb.game_date < :target_date
        ),
        latest_team AS (
            SELECT
                player_id,
                team,
                ROW_NUMBER() OVER (
                    PARTITION BY player_id
                    ORDER BY game_date DESC, player_id
                ) AS row_rank
            FROM season_rows
        )
        SELECT
            sr.player_id,
            MAX(sr.player_name) AS player_name,
            lt.team,
            COUNT(DISTINCT sr.game_id) AS games,
            SUM(sr.home_runs) AS home_runs,
            CASE
                WHEN COUNT(DISTINCT sr.game_id) <= 0 THEN NULL
                ELSE SUM(sr.home_runs) * 1.0 / COUNT(DISTINCT sr.game_id)
            END AS home_runs_per_game
        FROM season_rows sr
        LEFT JOIN latest_team lt ON lt.player_id = sr.player_id AND lt.row_rank = 1
        GROUP BY sr.player_id, lt.team
        HAVING COUNT(DISTINCT sr.game_id) > 0
        ORDER BY home_runs DESC, home_runs_per_game DESC, player_name ASC
        LIMIT :limit
        """,
        {"season_start": season_start, "target_date": target_date, "limit": limit},
    )

    team_home_run_rows = _safe_frame(
        """
        WITH season_rows AS (
            SELECT
                team,
                game_id,
                SUM(COALESCE(home_runs, 0)) AS hr_game
            FROM player_game_batting
            WHERE game_date >= :season_start
              AND game_date < :target_date
            GROUP BY team, game_id
        )
        SELECT
            team,
            COUNT(DISTINCT game_id) AS games,
            SUM(hr_game) AS home_runs,
            CASE
                WHEN COUNT(DISTINCT game_id) <= 0 THEN NULL
                ELSE SUM(hr_game) * 1.0 / COUNT(DISTINCT game_id)
            END AS home_runs_per_game
        FROM season_rows
        GROUP BY team
        HAVING COUNT(DISTINCT game_id) > 0
        ORDER BY home_runs DESC, home_runs_per_game DESC, team ASC
        LIMIT :limit
        """,
        {"season_start": season_start, "target_date": target_date, "limit": limit},
    )

    return {
        "season": int(target_date.year),
        "season_start": season_start.isoformat(),
        "through_date": target_date.isoformat(),
        "pitcher_strikeouts": _frame_records(pitcher_rows),
        "pitcher_era": _frame_records(pitcher_era_rows),
        "pitcher_whip": _frame_records(pitcher_whip_rows),
        "hitter_hits": _frame_records(hitter_rows),
        "hitter_home_runs": _frame_records(hitter_home_run_rows),
        "team_runs": _frame_records(team_runs_rows),
        "team_strikeouts": _frame_records(team_strikeout_rows),
        "team_home_runs": _frame_records(team_home_run_rows),
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


def _daily_results_market_display_name(market: str | None) -> str:
    market_key = str(market or "")
    if market_key in BEST_BET_MARKET_KEYS:
        return _best_bet_market_display_name(market_key)
    if market_key in EXPERIMENTAL_CAROUSEL_MARKETS:
        return _experimental_market_display_name(market_key)
    mapping = {
        "totals": "Game Totals",
        "hits": "1+ Hits",
        "pitcher_strikeouts": "Pitcher Strikeouts",
        "first5": "First 5 Totals",
        "top_ev": "Top EV",
        best_bets_utils.PLAYER_HOME_RUN_MARKET_KEY: "To hit a HR",
    }
    return mapping.get(market_key, market_key.replace("_", " ").title() or "AI Pick")


def _daily_results_pick_label(
    market: str,
    recommended_side: str | None,
    market_line: float | None,
    line_value: float | None,
    away_team: str | None,
    home_team: str | None,
    meta: dict[str, Any],
) -> str:
    if market in BEST_BET_MARKET_KEYS:
        selection_label = meta.get("selection_label")
        if selection_label:
            return str(selection_label)
        return _best_bet_selection_label(market, recommended_side, away_team, home_team, line_value)
    if market in EXPERIMENTAL_CAROUSEL_MARKETS:
        selection_label = meta.get("selection_label") or meta.get("market_label")
        if selection_label:
            return str(selection_label)
        return _experimental_market_display_name(market)

    side = str(recommended_side or "")
    if market == "hits":
        return "1+ Hit"
    if market == best_bets_utils.PLAYER_HOME_RUN_MARKET_KEY:
        return "HR · Yes"
    if market == "totals":
        return f"{side.title()} {market_line:.1f}" if market_line is not None else side.title() or "No line"
    if market == "first5":
        if market_line is not None:
            return f"F5 {side.title()} {market_line:.1f}"
        return f"F5 {side.title()}".strip()
    if market == "pitcher_strikeouts":
        return f"{side.title()} {market_line:.1f} Ks" if market_line is not None else f"{side.title()} Ks".strip()
    return _daily_results_market_display_name(market)


def _daily_results_actual_display(
    record: dict[str, Any],
    market: str,
    away_team: str | None,
    home_team: str | None,
    meta: dict[str, Any],
) -> str:
    if market == "top_ev":
        um = str(meta.get("underlying_market_key") or "")
        if um == best_bets_utils.PLAYER_HITS_MARKET_KEY:
            um = "hits"
        if um:
            return _daily_results_actual_display(record, um, away_team, home_team, meta)
        return "Pending"

    is_final = record.get("game_final")
    if is_final is None:
        is_final = bool(record.get("graded"))
    result = _graded_pick_result(
        record.get("recommended_side"),
        record.get("actual_side"),
        bool(is_final),
    )
    if result == "pending":
        return "Pending"
    if result == "missing":
        return "Missing"

    actual_side = str(record.get("actual_side") or "")
    actual_value = _coerce_float(record.get("actual_value"))
    actual_measure = _coerce_float(meta.get("actual_measure"))
    display_value = actual_measure if actual_measure is not None else actual_value
    if market == "hits":
        actual_hits = _coerce_float(record.get("actual_hits"))
        actual_at_bats = _coerce_float(record.get("actual_at_bats"))
        if actual_hits is not None:
            hits_text = f"{int(actual_hits)} hit" + ("" if int(actual_hits) == 1 else "s")
            if actual_at_bats is not None:
                return f"{hits_text} · {int(actual_at_bats)} AB"
            return hits_text
        if actual_side == "yes":
            return "1+ hit"
        if actual_side == "no":
            return "No hit"
        return "Pending"

    if market == best_bets_utils.PLAYER_HOME_RUN_MARKET_KEY:
        actual_hr = _coerce_float(record.get("actual_home_runs"))
        if actual_hr is not None:
            hr_n = int(actual_hr)
            return f"{hr_n} HR" + ("" if hr_n == 1 else "s")
        if actual_side == "yes":
            return "Homered"
        if actual_side == "no":
            return "No HR"
        return "Pending"

    if market in {"totals", "first5", "first_five_total", "game_total"} and display_value is not None:
        if actual_side:
            return f"{display_value:.0f} runs · {actual_side.upper()}"
        return f"{display_value:.0f} runs"

    if market in {"nrfi", "yrfi"} and display_value is not None:
        if actual_side:
            return f"{display_value:.0f} runs (1st inn) · {actual_side.upper()}"
        return f"{display_value:.0f} runs (1st inn)"

    # Pitcher K actuals: ``grade_best_bet_pick`` sets ``actual_value`` to 1/0 (win/loss), not K count.
    # The real total is ``actual_measure`` (passed through ``meta`` from grading).
    if market == "pitcher_strikeouts":
        ks_count = _coerce_float(meta.get("actual_measure"))
        if ks_count is None:
            ks_count = _coerce_float(record.get("actual_strikeouts"))
        if ks_count is not None:
            if actual_side:
                return f"{ks_count:.0f} Ks · {actual_side.upper()}"
            return f"{ks_count:.0f} Ks"
        if result not in ("pending", "missing") and actual_side:
            return actual_side.upper()
        return "Pending"

    if market in {"moneyline", "first_five_moneyline"}:
        if actual_side == "away":
            return f"{away_team or 'Away'} won"
        if actual_side == "home":
            return f"{home_team or 'Home'} won"
        if actual_side == "push":
            return "Push"

    if market == "run_line":
        if actual_side == "away":
            return f"{away_team or 'Away'} covered"
        if actual_side == "home":
            return f"{home_team or 'Home'} covered"
        if actual_side == "push":
            return "Push"

    if market == "first_five_spread":
        if actual_side == "away":
            return f"F5 {away_team or 'Away'} covered"
        if actual_side == "home":
            return f"F5 {home_team or 'Home'} covered"
        if actual_side == "push":
            return "Push"

    if market in {"away_team_total", "home_team_total"}:
        if display_value is not None and actual_side:
            return f"{display_value:.0f} runs · {actual_side.upper()}"
        if actual_side == "push":
            return "Push"

    if market in {"first_five_team_total_away", "first_five_team_total_home"}:
        if display_value is not None and actual_side:
            return f"{display_value:.0f} runs · {actual_side.upper()}"
        if actual_side == "push":
            return "Push"

    if actual_side == "push":
        return "Push"
    return actual_side.replace("_", " ").title() or "Pending"


def _daily_results_confidence(record: dict[str, Any], market: str) -> float | None:
    if market == "top_ev":
        return _coerce_float(record.get("probability"))
    recommended_side = str(record.get("recommended_side") or "")
    probability = _coerce_float(record.get("probability"))
    opposite_probability = _coerce_float(record.get("opposite_probability"))
    if market in BEST_BET_MARKET_KEYS or market == "hits" or recommended_side == "yes":
        return probability
    if recommended_side == "under":
        if opposite_probability is not None:
            return opposite_probability
        if probability is not None:
            return max(0.0, min(1.0, 1.0 - probability))
    return probability


def _daily_results_edge_value(record: dict[str, Any], market: str, meta: dict[str, Any]) -> float | None:
    if market == "top_ev":
        weighted_ev = _coerce_float(meta.get("weighted_ev"))
        if weighted_ev is not None:
            return weighted_ev
        return _coerce_float(meta.get("probability_edge"))
    if market in BEST_BET_MARKET_KEYS:
        weighted_ev = _coerce_float(meta.get("weighted_ev"))
        if weighted_ev is not None:
            return weighted_ev
        return _coerce_float(meta.get("probability_edge"))
    if market == "hits":
        return _coerce_float(meta.get("edge"))
    if market == best_bets_utils.PLAYER_HOME_RUN_MARKET_KEY:
        return _coerce_float(meta.get("edge"))
    predicted_value = _coerce_float(record.get("predicted_value"))
    market_line = _coerce_float(record.get("market_line"))
    if predicted_value is None or market_line is None:
        return None
    return float(predicted_value) - float(market_line)


def _daily_results_model_display(record: dict[str, Any], market: str, confidence: float | None) -> str:
    predicted_value = _coerce_float(record.get("predicted_value"))
    if market == "top_ev":
        return "-" if confidence is None else f"{confidence * 100:.1f}% model"
    if market == "hits":
        return f"{confidence * 100:.0f}% yes" if confidence is not None else "-"
    if market == best_bets_utils.PLAYER_HOME_RUN_MARKET_KEY:
        return f"{confidence * 100:.0f}% HR" if confidence is not None else "-"
    if market in BEST_BET_MARKET_KEYS:
        return f"{confidence * 100:.0f}%" if confidence is not None else "-"
    if market in EXPERIMENTAL_CAROUSEL_MARKETS:
        return "-"
    if market in {"totals", "first5"}:
        return "-" if predicted_value is None else f"{predicted_value:.2f} runs"
    if market == "pitcher_strikeouts":
        return "-" if predicted_value is None else f"{predicted_value:.2f} Ks"
    return "-" if predicted_value is None else f"{predicted_value:.2f}"


def _daily_results_notes_display(
    record: dict[str, Any],
    market: str,
    market_label: str,
    confidence: float | None,
    edge_value: float | None,
    meta: dict[str, Any],
) -> str:
    details: list[str] = [market_label]
    sportsbook = meta.get("sportsbook") or record.get("entry_market_sportsbook")
    if sportsbook:
        details.append(str(sportsbook))

    if market in BEST_BET_MARKET_KEYS:
        weighted_ev = _coerce_float(meta.get("weighted_ev"))
        probability_edge = _coerce_float(meta.get("probability_edge"))
        price = meta.get("price")
        if price is not None:
            details.append(f"Price {_format_price_text(price)}")
        if weighted_ev is not None:
            details.append(f"EV {weighted_ev * 100:+.1f}%")
        if probability_edge is not None:
            details.append(f"No-vig {probability_edge * 100:+.1f} pts")
        return " · ".join(details)

    if market == "top_ev":
        weighted_ev = _coerce_float(meta.get("weighted_ev"))
        probability_edge = _coerce_float(meta.get("probability_edge"))
        price = meta.get("price")
        if price is not None:
            details.append(f"Price {_format_price_text(price)}")
        if weighted_ev is not None:
            details.append(f"EV {weighted_ev * 100:+.1f}%")
        if probability_edge is not None:
            details.append(f"No-vig {probability_edge * 100:+.1f} pts")
        n = meta.get("top_ev_candidate_count")
        if n is not None:
            details.append(f"{n} priced candidates")
        return " · ".join(details)

    if market in EXPERIMENTAL_CAROUSEL_MARKETS:
        price = meta.get("price")
        if price is not None:
            details.append(f"Price {_format_price_text(price)}")
        experimental_reason = meta.get("experimental_reason")
        if experimental_reason:
            details.append(str(experimental_reason))
        else:
            details.append("Tracked separately from the green board")
        return " · ".join(details)

    if market == "hits":
        market_price = meta.get("market_price")
        fair_price = meta.get("fair_price")
        if market_price is not None:
            details.append(f"Market {_format_price_text(market_price)}")
        if fair_price is not None:
            details.append(f"Fair {_format_price_text(fair_price)}")
        if edge_value is not None:
            details.append(f"Edge {edge_value:+.3f}")
        return " · ".join(details)

    if market == best_bets_utils.PLAYER_HOME_RUN_MARKET_KEY:
        market_price = meta.get("market_price")
        fair_price = meta.get("fair_price")
        if market_price is not None:
            details.append(f"Market {_format_price_text(market_price)}")
        if fair_price is not None:
            details.append(f"Fair {_format_price_text(fair_price)}")
        if edge_value is not None:
            details.append(f"Edge {edge_value:+.3f}")
        return " · ".join(details)

    if confidence is not None:
        details.append(f"Pick {confidence * 100:.0f}%")
    if edge_value is not None:
        details.append(f"Edge {edge_value:+.2f}")
    return " · ".join(details)


def _recommendation_result_sort_key(row: dict[str, Any], rank_field: str | None = None) -> tuple[float, float, float, str, int]:
    rank_value = _coerce_float(row.get(rank_field)) if rank_field else None
    edge_value = _coerce_float(row.get("edge"))
    confidence = _coerce_float(row.get("confidence"))
    start_ts = str(row.get("game_start_ts") or "")
    game_id = int(row.get("game_id") or 0)
    return (
        9999.0 if rank_value is None else -float(rank_value),
        -999.0 if edge_value is None else float(edge_value),
        -999.0 if confidence is None else float(confidence),
        start_ts,
        -game_id,
    )


def _fetch_game_recommendation_results(
    target_date: date,
    *,
    bucket: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if not _table_exists("prediction_outcomes_daily"):
        return []

    frame = _safe_frame(
        """
        SELECT
            o.game_date,
            o.game_id,
            o.market,
            o.entity_type,
            o.player_id,
            o.pitcher_id,
            o.prediction_ts,
            o.team,
            o.opponent,
            o.recommended_side,
            o.actual_side,
            o.graded,
            o.success,
            o.beat_market,
            o.predicted_value,
            o.actual_value,
            o.market_line,
            o.probability,
            o.opposite_probability,
            o.entry_market_sportsbook,
            o.meta_payload,
            g.game_start_ts,
            g.status AS game_status,
            g.away_team,
            g.home_team,
            batter.hits AS actual_hits,
            batter.at_bats AS actual_at_bats,
            batter.home_runs AS actual_home_runs,
            dp.full_name AS player_name,
            pitcher.full_name AS pitcher_name
        FROM prediction_outcomes_daily o
        LEFT JOIN games g
            ON g.game_id = o.game_id
           AND g.game_date = o.game_date
        LEFT JOIN player_game_batting batter
            ON batter.game_id = o.game_id
           AND batter.player_id = o.player_id
           AND batter.game_date = o.game_date
        LEFT JOIN dim_players dp ON dp.player_id = o.player_id
        LEFT JOIN dim_players pitcher ON pitcher.player_id = o.pitcher_id
        WHERE o.game_date = :target_date
          AND o.recommended_side IS NOT NULL
          AND (
              o.entity_type = 'game'
              OR (o.entity_type = 'pitcher' AND o.market = 'pitcher_strikeouts')
          )
        ORDER BY
            COALESCE(g.game_start_ts, o.prediction_ts),
            o.game_id,
            CASE o.market
                WHEN 'totals' THEN 0
                WHEN 'first5' THEN 1
                WHEN 'pitcher_strikeouts' THEN 2
                ELSE 3
            END,
            COALESCE(dp.full_name, pitcher.full_name, '')
        """,
        {"target_date": str(target_date)},
    )

    rows: list[dict[str, Any]] = []
    for record in _frame_records(frame):
        record = dict(record)
        record["game_final"] = _is_final_game_status(record.get("game_status"))
        meta = _review_meta_payload(record.get("meta_payload"))
        market = str(record.get("market") or "")
        if market not in TRACKED_RECOMMENDATION_MARKETS:
            continue
        away_team = meta.get("away_team") or record.get("away_team")
        home_team = meta.get("home_team") or record.get("home_team")
        matchup = None
        if away_team or home_team:
            matchup = f"{away_team or '?'} at {home_team or '?'}"

        line_value = _coerce_float(meta.get("line_value"))
        market_line = _coerce_float(record.get("market_line"))
        market_label = _daily_results_market_display_name(market)
        confidence = _daily_results_confidence(record, market)
        edge_value = _daily_results_edge_value(record, market, meta)
        result = _graded_pick_result(
            record.get("recommended_side"),
            record.get("actual_side"),
            bool(record.get("game_final")),
        )
        is_green_pick = bool(meta.get("is_board_green_pick") or meta.get("is_green_pick"))
        is_watchlist_pick = bool(
            meta.get("is_board_watchlist_pick") or meta.get("is_hr_slate_pick")
        )
        is_experimental_pick = bool(meta.get("is_experimental_pick")) or market in EXPERIMENTAL_CAROUSEL_MARKETS
        green_rank = meta.get("board_green_pick_rank") or meta.get("green_pick_rank")
        watchlist_rank = meta.get("board_watchlist_rank")
        recommendation_reason = meta.get("green_reason")
        if not recommendation_reason and is_green_pick:
            recommendation_reason = "Green board pick"
        if not recommendation_reason and is_watchlist_pick:
            recommendation_reason = str(meta.get("hr_slate_reason") or "Watchlist pick")
        if not recommendation_reason and is_experimental_pick:
            recommendation_reason = str(meta.get("experimental_reason") or "Experimental first-inning tracking")
        price = meta.get("price")
        market_price = meta.get("market_price")
        market_backed = any(
            value is not None
            for value in (
                record.get("entry_market_sportsbook"),
                market_line,
                line_value,
                price,
                market_price,
            )
        )

        if market == "hits":
            subject_label = record.get("player_name") or meta.get("player_name") or f"Player {record.get('player_id') or '?'}"
            subject_subtitle = f"{record.get('team') or '?'} vs {record.get('opponent') or 'TBD'}"
        elif market == best_bets_utils.PLAYER_HOME_RUN_MARKET_KEY:
            subject_label = record.get("player_name") or meta.get("player_name") or f"Player {record.get('player_id') or '?'}"
            subject_subtitle = f"{record.get('team') or '?'} vs {record.get('opponent') or 'TBD'}"
        elif market == "pitcher_strikeouts":
            subject_label = record.get("pitcher_name") or meta.get("pitcher_name") or f"Pitcher {record.get('pitcher_id') or '?'}"
            subject_subtitle = f"{record.get('team') or '?'} vs {record.get('opponent') or 'TBD'}"
        else:
            subject_label = matchup or market_label
            subject_subtitle = market_label

        rows.append(
            {
                "game_id": record.get("game_id"),
                "game_date": record.get("game_date"),
                "game_start_ts": record.get("game_start_ts"),
                "market": market,
                "market_label": market_label,
                "subject_label": subject_label,
                "subject_subtitle": subject_subtitle,
                "pick_label": _daily_results_pick_label(
                    market,
                    record.get("recommended_side"),
                    market_line,
                    line_value,
                    away_team,
                    home_team,
                    meta,
                ),
                "model_display": _daily_results_model_display(record, market, confidence),
                "actual_display": _daily_results_actual_display(record, market, away_team, home_team, meta),
                "notes_display": _daily_results_notes_display(
                    record,
                    market,
                    market_label,
                    confidence,
                    edge_value,
                    meta,
                ),
                "confidence": confidence,
                "edge": edge_value,
                "result": result,
                "market_backed": market_backed,
                "is_green_pick": is_green_pick,
                "is_watchlist_pick": is_watchlist_pick,
                "is_experimental_pick": is_experimental_pick,
                "green_rank": green_rank,
                "watchlist_rank": watchlist_rank,
                "green_reason": recommendation_reason,
                "input_trust_grade": meta.get("input_trust_grade"),
                "promotion_tier": meta.get("promotion_tier"),
                "green_strip_tier": meta.get("green_strip_tier"),
            }
        )

    if bucket == "experimental":
        rows = [row for row in rows if row.get("is_experimental_pick")]
        rows.sort(key=lambda row: _recommendation_result_sort_key(row))
        return rows if limit is None else rows[:limit]

    # Green / watchlist must never use NRFI/YRFI (experimental-only). Those markets share
    # prediction_outcomes_daily with board lanes; without this filter the edge fallback can
    # surface nrfi as the only "pick" per game when snapshot flags are missing.
    # Team best picks also exclude hitter lanes (hits / 1+ / HR); those are not tracked in this UI.
    board_rows = [
        row
        for row in rows
        if str(row.get("market") or "").lower() not in EXPERIMENTAL_CAROUSEL_MARKETS
        and not _daily_results_excluded_from_team_best_picks(str(row.get("market") or ""))
    ]

    has_snapshot_flags = any(
        row.get("is_green_pick") or row.get("is_watchlist_pick") for row in board_rows
    )
    if has_snapshot_flags:
        if bucket == "green":
            rows = [row for row in board_rows if row.get("is_green_pick")]
            rows.sort(key=lambda row: _recommendation_result_sort_key(row, "green_rank"))
        elif bucket == "watchlist":
            rows = [row for row in board_rows if row.get("is_watchlist_pick")]
            rows.sort(key=lambda row: _recommendation_result_sort_key(row, "watchlist_rank"))
        else:
            rows = sorted(board_rows, key=lambda row: _recommendation_result_sort_key(row))
        return rows if limit is None else rows[:limit]

    if bucket == "watchlist":
        return []

    if limit is None:
        limit = _green_pick_board_limit(target_date)

    def _row_rank_key(row: dict[str, Any]) -> tuple[float, float, str, int]:
        edge_value = _coerce_float(row.get("edge"))
        confidence = _coerce_float(row.get("confidence"))
        start_ts = str(row.get("game_start_ts") or "")
        game_id = int(row.get("game_id") or 0)
        return (
            -999.0 if edge_value is None else float(edge_value),
            -999.0 if confidence is None else float(confidence),
            start_ts,
            -game_id,
        )

    best_by_game: dict[int, dict[str, Any]] = {}
    for row in board_rows:
        game_id = row.get("game_id")
        if game_id is None:
            continue
        game_id_int = int(game_id)
        existing = best_by_game.get(game_id_int)
        if existing is None or _row_rank_key(row) > _row_rank_key(existing):
            best_by_game[game_id_int] = row

    ranked_rows = sorted(best_by_game.values(), key=_row_rank_key, reverse=True)
    if limit is None:
        return ranked_rows
    return ranked_rows[:limit]


def _fetch_ai_pick_results(target_date: date, limit: int | None = None) -> list[dict[str, Any]]:
    return _fetch_game_recommendation_results(target_date, bucket="green", limit=limit)


def _fetch_watchlist_pick_results(target_date: date, limit: int | None = None) -> list[dict[str, Any]]:
    return _fetch_game_recommendation_results(target_date, bucket="watchlist", limit=limit)


def _fetch_experimental_pick_results(target_date: date, limit: int | None = None) -> list[dict[str, Any]]:
    return _fetch_game_recommendation_results(target_date, bucket="experimental", limit=limit)


def _lineup_player_by_id(detail: dict[str, Any], player_id: int) -> dict[str, Any] | None:
    for side in ("away", "home"):
        for pl in (detail.get("teams") or {}).get(side, {}).get("lineup") or []:
            if pl.get("player_id") is not None and int(pl["player_id"]) == int(player_id):
                return pl
    return None


def _starter_by_pitcher_id(detail: dict[str, Any], pitcher_id: int) -> dict[str, Any] | None:
    for side in ("away", "home"):
        st = (detail.get("starters") or {}).get(side)
        if st and st.get("pitcher_id") is not None and int(st["pitcher_id"]) == int(pitcher_id):
            return st
    return None


def _grade_top_ev_pick_for_daily_results(
    detail: dict[str, Any],
    pick: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    market = str(pick.get("market_key") or "")
    away_team = detail.get("away_team")
    home_team = detail.get("home_team")
    line_value = _coerce_float(pick.get("line_value"))
    actual_result = dict(detail.get("actual_result") or {})
    f5 = detail.get("first5_totals") or {}
    first5_result = {
        "away_runs": f5.get("away_runs"),
        "home_runs": f5.get("home_runs"),
        "total_runs": f5.get("actual_total_runs"),
    }
    pid = pick.get("player_id")
    player_prop_actuals = None
    pitcher_ks: float | None = None
    if pid is not None:
        pid_i = int(pid)
        if market in (best_bets_utils.PLAYER_HITS_MARKET_KEY, best_bets_utils.PLAYER_HOME_RUN_MARKET_KEY):
            pl = _lineup_player_by_id(detail, pid_i)
            if pl is not None:
                player_prop_actuals = {"hits": pl.get("actual_hits"), "home_runs": pl.get("actual_home_runs")}
        elif market == best_bets_utils.PITCHER_STRIKEOUTS_MARKET_KEY:
            st = _starter_by_pitcher_id(detail, pid_i)
            if st is not None:
                pitcher_ks = best_bets_utils.to_float(st.get("strikeouts"))

    card_for_grade = {
        "market_key": market,
        "bet_side": pick.get("bet_side"),
        "line_value": line_value,
        "away_team": away_team,
        "home_team": home_team,
    }
    grade = best_bets_utils.grade_best_bet_pick(
        card_for_grade,
        actual_result=actual_result,
        first5_result=first5_result,
        player_prop_actuals=player_prop_actuals,
        pitcher_strikeouts_actual=pitcher_ks,
    )
    meta: dict[str, Any] = {
        "selection_label": pick.get("selection_label"),
        "market_label": pick.get("market_label"),
        "weighted_ev": pick.get("weighted_ev"),
        "probability_edge": pick.get("probability_edge"),
        "price": pick.get("price"),
        "sportsbook": pick.get("sportsbook"),
        "away_team": away_team,
        "home_team": home_team,
        "top_ev_candidate_count": pick.get("top_ev_candidate_count"),
        "underlying_market_key": market,
    }
    meta["actual_measure"] = grade.get("actual_measure")
    gres = str(grade.get("result") or "pending")
    pick_result = gres if gres in ("won", "lost", "push") else "pending"
    if not actual_result.get("is_final"):
        pick_result = "pending"
    record = {
        "recommended_side": str(pick.get("bet_side") or ""),
        "probability": pick.get("model_probability"),
        "opposite_probability": None,
        "graded": bool(grade.get("graded")),
        "actual_side": grade.get("actual_side"),
        "actual_value": grade.get("actual_value"),
        "game_final": bool(actual_result.get("is_final")),
    }
    if market == best_bets_utils.PLAYER_HITS_MARKET_KEY and pid is not None:
        pl = _lineup_player_by_id(detail, int(pid))
        if pl is not None:
            record["actual_hits"] = pl.get("actual_hits")
            record["actual_at_bats"] = pl.get("actual_at_bats")
    if market == best_bets_utils.PLAYER_HOME_RUN_MARKET_KEY and pid is not None:
        pl = _lineup_player_by_id(detail, int(pid))
        if pl is not None:
            record["actual_home_runs"] = pl.get("actual_home_runs")
    return record, pick_result, meta


def _detail_for_top_ev_from_board_row(game: dict[str, Any]) -> dict[str, Any]:
    """Shape a main-board row into the ``detail`` subset ``collect_top_ev_candidates`` / grading need."""
    away = game.get("away_team")
    home = game.get("home_team")
    ht = game.get("hit_targets") or {}
    return {
        "game_id": game.get("game_id"),
        "away_team": away,
        "home_team": home,
        "game_date": game.get("game_date"),
        "certainty": game.get("certainty"),
        "totals": dict(game.get("totals") or {}),
        "first5_totals": dict(game.get("first5_totals") or {}),
        "teams": {
            "away": {"lineup": list(ht.get(away) or [])},
            "home": {"lineup": list(ht.get(home) or [])},
        },
        "starters": game.get("starters") or {"away": None, "home": None},
        "actual_result": dict(game.get("actual_result") or {}),
    }


def _top_ev_pick_for_board_row(
    game: dict[str, Any],
    market_rows_by_type: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """Same weighted-EV selector as game detail and Daily Results, using the board row shape."""
    detail = _detail_for_top_ev_from_board_row(game)
    return select_top_weighted_ev_pick(
        collect_top_ev_candidates(detail, market_rows_by_type),
    )


def _resolve_top_ev_pick_with_snapshots(
    game: dict[str, Any],
    market_rows_by_type: dict[str, list[dict[str, Any]]],
    lock_by_gid: dict[int, dict[str, Any]],
    run_by_gid: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    """Prefer lock-time Top EV snapshot, then first-run snapshot, else live weighted-EV winner (main board)."""
    gid = int(game.get("game_id") or 0)
    live = _top_ev_pick_for_board_row(game, market_rows_by_type)
    snap_lock = lock_by_gid.get(gid) if gid else None
    if isinstance(snap_lock, dict) and snap_lock:
        out = dict(snap_lock)
        out["top_ev_snapshot_kind"] = "lock"
        out["top_ev_frozen"] = True
        return out
    snap_run = run_by_gid.get(gid) if gid else None
    if isinstance(snap_run, dict) and snap_run:
        out = dict(snap_run)
        out["top_ev_snapshot_kind"] = "run"
        out["top_ev_frozen"] = True
        return out
    if live:
        out = dict(live)
        out["top_ev_snapshot_kind"] = None
        out["top_ev_frozen"] = False
        return out
    return None


def _live_top_ev_rows_for_daily_results(
    target_date: date,
    *,
    board_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """One Top EV row per game — same selector as game detail, using main-board row shape (fast path).

    Avoids one ``_fetch_game_detail`` call per game (that made Daily Results time out).
    Uses the same board defaults as the green strip so EV pool matches the main board.

    Pass ``board_rows`` from ``_fetch_daily_results`` to avoid a duplicate ``_fetch_game_board`` call.
    """
    if board_rows is None:
        if not _table_exists("games"):
            return []
        board_rows = _fetch_game_board(
            target_date,
            hit_limit_per_team=GAME_BOARD_UI_DEFAULT_HIT_LIMIT_PER_TEAM,
            min_probability=GAME_BOARD_UI_DEFAULT_MIN_HIT_PROBABILITY,
            confirmed_only=GAME_BOARD_UI_DEFAULT_CONFIRMED_ONLY,
            include_inferred=GAME_BOARD_UI_DEFAULT_INCLUDE_INFERRED,
        )
    market_rows_by_game: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for market_row in _fetch_latest_game_market_rows(target_date):
        gid_raw = market_row.get("game_id")
        mt = str(market_row.get("market_type") or "")
        if gid_raw is None or not mt:
            continue
        gid = int(gid_raw)
        market_rows_by_game.setdefault(gid, {}).setdefault(mt, []).append(market_row)

    frozen_lock = _fetch_board_top_ev_snapshots_map(target_date)
    frozen_run = _fetch_board_top_ev_run_snapshots_map(target_date)
    out: list[dict[str, Any]] = []
    for game in board_rows:
        gid_raw = game.get("game_id")
        if gid_raw is None:
            continue
        game_id = int(gid_raw)
        detail = _detail_for_top_ev_from_board_row(game)
        pick: dict[str, Any] | None = None
        frozen = False
        snap_kind: str | None = None
        snap_lock = frozen_lock.get(game_id)
        snap_run = frozen_run.get(game_id)
        if isinstance(snap_lock, dict) and snap_lock:
            pick = dict(snap_lock)
            frozen = True
            snap_kind = "lock"
        elif isinstance(snap_run, dict) and snap_run:
            pick = dict(snap_run)
            frozen = True
            snap_kind = "run"
        if pick is None:
            candidates = collect_top_ev_candidates(
                detail,
                market_rows_by_game.get(game_id, {}),
            )
            pick = select_top_weighted_ev_pick(candidates)
        if not pick:
            continue
        away_team = game.get("away_team")
        home_team = game.get("home_team")
        matchup = f"{away_team or '?'} at {home_team or '?'}" if away_team or home_team else None
        line_value = _coerce_float(pick.get("line_value"))
        record, pick_result, meta = _grade_top_ev_pick_for_daily_results(detail, pick)
        market_label = "Top EV"
        confidence = _daily_results_confidence(record, "top_ev")
        edge_value = _daily_results_edge_value(record, "top_ev", meta)
        if frozen and snap_kind == "lock":
            reason = "Top EV pick (frozen at pregame lock — same weighted-EV winner as the board at lock)"
        elif frozen and snap_kind == "run":
            reason = "Top EV pick (frozen at first pregame board run — before lock window)"
        else:
            reason = "Top EV pick (largest weighted EV among priced candidates for this game)"

        out.append(
            {
                "game_id": game_id,
                "game_date": target_date.isoformat(),
                "game_start_ts": game.get("game_start_ts"),
                "market": "top_ev",
                "market_label": market_label,
                "subject_label": matchup or market_label,
                "subject_subtitle": str(meta.get("underlying_market_key") or ""),
                "pick_label": pick.get("selection_label") or pick.get("pick_label") or market_label,
                "model_display": _daily_results_model_display(record, "top_ev", confidence),
                "actual_display": _daily_results_actual_display(record, "top_ev", away_team, home_team, meta),
                "notes_display": _daily_results_notes_display(
                    record,
                    "top_ev",
                    market_label,
                    confidence,
                    edge_value,
                    meta,
                ),
                "confidence": confidence,
                "edge": edge_value,
                "result": pick_result,
                "market_backed": bool(
                    meta.get("sportsbook") is not None
                    or line_value is not None
                    or meta.get("price") is not None
                ),
                "is_green_pick": False,
                "is_watchlist_pick": False,
                "is_experimental_pick": False,
                "is_top_ev_pick": True,
                "green_rank": None,
                "watchlist_rank": None,
                "green_reason": reason,
                "top_ev_frozen": frozen,
                "top_ev_snapshot_kind": snap_kind,
                "input_trust_grade": (pick.get("input_trust") or {}).get("grade"),
                "promotion_tier": pick.get("promotion_tier"),
                "green_strip_tier": None,
            }
        )
    out.sort(key=lambda row: _recommendation_result_sort_key(row, "green_rank"))
    return out


def _grade_live_best_bet_card_for_daily_results(
    card: dict[str, Any],
    game: dict[str, Any],
    base_meta: dict[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Use boxscore-backed scores from the board payload to grade green/watchlist mirror rows."""
    away_team = base_meta.get("away_team") or game.get("away_team")
    home_team = base_meta.get("home_team") or game.get("home_team")
    market = str(card.get("market_key") or "")
    line_value = _coerce_float(card.get("line_value"))
    actual_result = dict(game.get("actual_result") or {})
    f5 = game.get("first5_totals") or {}
    first5_result = {
        "away_runs": f5.get("away_runs"),
        "home_runs": f5.get("home_runs"),
        "total_runs": f5.get("actual_total_runs"),
    }
    card_for_grade = {
        "market_key": market,
        "bet_side": card.get("bet_side"),
        "line_value": line_value,
        "away_team": away_team,
        "home_team": home_team,
    }
    pitcher_ks: float | None = None
    pid_raw = card.get("player_id")
    if market == best_bets_utils.PITCHER_STRIKEOUTS_MARKET_KEY and pid_raw is not None:
        st = _starter_by_pitcher_id(
            {"starters": game.get("starters") or {}},
            int(pid_raw),
        )
        if st is not None:
            pitcher_ks = best_bets_utils.to_float(st.get("strikeouts"))
    grade = best_bets_utils.grade_best_bet_pick(
        card_for_grade,
        actual_result=actual_result,
        first5_result=first5_result,
        pitcher_strikeouts_actual=pitcher_ks,
    )
    meta = dict(base_meta)
    meta["actual_measure"] = grade.get("actual_measure")
    gres = str(grade.get("result") or "pending")
    pick_result = gres if gres in ("won", "lost", "push") else "pending"
    record = {
        "recommended_side": str(card.get("bet_side") or ""),
        "probability": card.get("model_probability"),
        "opposite_probability": None,
        "graded": bool(grade.get("graded")),
        "actual_side": grade.get("actual_side"),
        "actual_value": grade.get("actual_value"),
        "game_final": bool(actual_result.get("is_final")),
    }
    return record, pick_result, meta


def _live_green_board_rows_for_daily_results(
    target_date: date,
    *,
    board_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build Daily Results green-board rows from the same board inputs as the main UI ``best_bets`` strip.

    Uses ``GAME_BOARD_UI_*`` defaults so results match a fresh load of the game board (index.html filters).
    """
    if board_rows is None:
        board_rows = _fetch_game_board(
            target_date,
            hit_limit_per_team=GAME_BOARD_UI_DEFAULT_HIT_LIMIT_PER_TEAM,
            min_probability=GAME_BOARD_UI_DEFAULT_MIN_HIT_PROBABILITY,
            confirmed_only=GAME_BOARD_UI_DEFAULT_CONFIRMED_ONLY,
            include_inferred=GAME_BOARD_UI_DEFAULT_INCLUDE_INFERRED,
        )
    cards = _flatten_best_bets(board_rows, target_date=target_date)
    game_by_id: dict[int, dict[str, Any]] = {}
    for row in board_rows:
        gid = row.get("game_id")
        if gid is not None:
            game_by_id[int(gid)] = row

    out: list[dict[str, Any]] = []
    for card in cards:
        gid = int(card.get("game_id") or 0)
        game = game_by_id.get(gid, {})
        market = str(card.get("market_key") or "")
        if market not in BEST_BET_MARKET_KEYS:
            continue
        away_team = card.get("away_team") or game.get("away_team")
        home_team = card.get("home_team") or game.get("home_team")
        matchup = f"{away_team or '?'} at {home_team or '?'}" if away_team or home_team else None

        line_value = _coerce_float(card.get("line_value"))
        meta = {
            "selection_label": card.get("selection_label"),
            "market_label": card.get("market_label"),
            "weighted_ev": card.get("weighted_ev"),
            "probability_edge": card.get("probability_edge"),
            "price": card.get("price"),
            "sportsbook": card.get("sportsbook"),
            "away_team": away_team,
            "home_team": home_team,
        }
        record, pick_result, meta = _grade_live_best_bet_card_for_daily_results(card, game, meta)
        market_label = _daily_results_market_display_name(market)
        confidence = _daily_results_confidence(record, market)
        edge_value = _daily_results_edge_value(record, market, meta)

        out.append(
            {
                "game_id": gid,
                "game_date": target_date.isoformat(),
                "game_start_ts": game.get("game_start_ts"),
                "market": market,
                "market_label": market_label,
                "subject_label": matchup or market_label,
                "subject_subtitle": market_label,
                "pick_label": _daily_results_pick_label(
                    market,
                    record.get("recommended_side"),
                    line_value,
                    line_value,
                    away_team,
                    home_team,
                    meta,
                ),
                "model_display": _daily_results_model_display(record, market, confidence),
                "actual_display": _daily_results_actual_display(record, market, away_team, home_team, meta),
                "notes_display": _daily_results_notes_display(
                    record,
                    market,
                    market_label,
                    confidence,
                    edge_value,
                    meta,
                ),
                "confidence": confidence,
                "edge": edge_value,
                "result": pick_result,
                "market_backed": bool(
                    meta.get("sportsbook") is not None
                    or line_value is not None
                    or meta.get("price") is not None
                ),
                "is_green_pick": True,
                "is_watchlist_pick": False,
                "is_experimental_pick": False,
                "green_rank": None,
                "watchlist_rank": None,
                "green_reason": "Green board pick (live slate — results when games go final)",
                "input_trust_grade": None,
                "promotion_tier": card.get("promotion_tier"),
                "green_strip_tier": card.get("green_strip_tier")
                or ("strict" if card.get("positive") else "soft"),
            }
        )

    out.sort(key=lambda row: _recommendation_result_sort_key(row, "green_rank"))
    return out


def _daily_results_row_identity(row: dict[str, Any]) -> tuple[int, str, str]:
    return (
        int(row.get("game_id") or 0),
        str(row.get("market") or ""),
        str(row.get("pick_label") or ""),
    )


def _live_watchlist_board_rows_for_daily_results(
    target_date: date,
    *,
    board_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Mirror ``GET /api/games/board`` ``watchlist_markets`` for dates when archived outcomes are incomplete."""
    if board_rows is None:
        board_rows = _fetch_game_board(
            target_date,
            hit_limit_per_team=GAME_BOARD_UI_DEFAULT_HIT_LIMIT_PER_TEAM,
            min_probability=GAME_BOARD_UI_DEFAULT_MIN_HIT_PROBABILITY,
            confirmed_only=GAME_BOARD_UI_DEFAULT_CONFIRMED_ONLY,
            include_inferred=GAME_BOARD_UI_DEFAULT_INCLUDE_INFERRED,
        )
    cards = _flatten_watchlist_markets(board_rows, secondary_lines_only=True)
    game_by_id: dict[int, dict[str, Any]] = {}
    for row in board_rows:
        gid = row.get("game_id")
        if gid is not None:
            game_by_id[int(gid)] = row

    out: list[dict[str, Any]] = []
    for card in cards:
        gid = int(card.get("game_id") or 0)
        game = game_by_id.get(gid, {})
        market = str(card.get("market_key") or "")
        if market not in BEST_BET_MARKET_KEYS:
            continue
        away_team = card.get("away_team") or game.get("away_team")
        home_team = card.get("home_team") or game.get("home_team")
        matchup = f"{away_team or '?'} at {home_team or '?'}" if away_team or home_team else None

        line_value = _coerce_float(card.get("line_value"))
        meta = {
            "selection_label": card.get("selection_label"),
            "market_label": card.get("market_label"),
            "weighted_ev": card.get("weighted_ev"),
            "probability_edge": card.get("probability_edge"),
            "price": card.get("price"),
            "sportsbook": card.get("sportsbook"),
            "away_team": away_team,
            "home_team": home_team,
        }
        record, pick_result, meta = _grade_live_best_bet_card_for_daily_results(card, game, meta)
        market_label = _daily_results_market_display_name(market)
        confidence = _daily_results_confidence(record, market)
        edge_value = _daily_results_edge_value(record, market, meta)

        out.append(
            {
                "game_id": gid,
                "game_date": target_date.isoformat(),
                "game_start_ts": game.get("game_start_ts"),
                "market": market,
                "market_label": market_label,
                "subject_label": matchup or market_label,
                "subject_subtitle": market_label,
                "pick_label": _daily_results_pick_label(
                    market,
                    record.get("recommended_side"),
                    line_value,
                    line_value,
                    away_team,
                    home_team,
                    meta,
                ),
                "model_display": _daily_results_model_display(record, market, confidence),
                "actual_display": _daily_results_actual_display(record, market, away_team, home_team, meta),
                "notes_display": _daily_results_notes_display(
                    record,
                    market,
                    market_label,
                    confidence,
                    edge_value,
                    meta,
                ),
                "confidence": confidence,
                "edge": edge_value,
                "result": pick_result,
                "market_backed": bool(
                    meta.get("sportsbook") is not None
                    or line_value is not None
                    or meta.get("price") is not None
                ),
                "is_green_pick": False,
                "is_watchlist_pick": True,
                "is_experimental_pick": False,
                "green_rank": None,
                "watchlist_rank": None,
                "green_reason": "Watchlist pick (live slate — same as main board)",
                "input_trust_grade": None,
                "promotion_tier": card.get("promotion_tier"),
                "green_strip_tier": None,
            }
        )

    out.sort(key=lambda row: _recommendation_result_sort_key(row, "watchlist_rank"))
    return out


def _merge_watchlist_daily_results(
    archived_rows: list[dict[str, Any]],
    live_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    """Prefer archived (graded) rows; add live board rows not already present."""
    seen = {_daily_results_row_identity(r) for r in archived_rows}
    merged = list(archived_rows)
    supplemental = False
    for row in live_rows:
        ident = _daily_results_row_identity(row)
        if ident in seen:
            continue
        merged.append(row)
        seen.add(ident)
        supplemental = True
    merged.sort(key=lambda r: _recommendation_result_sort_key(r, "watchlist_rank"))
    return merged, supplemental


def _experimental_market_display_name(market: str | None) -> str:
    mapping = {
        "nrfi": "NRFI",
        "yrfi": "YRFI",
    }
    return mapping.get(str(market or "").lower(), str(market or "Experimental"))


def _american_to_implied_probability(price: Any) -> float | None:
    """Implied probability from American odds for the priced side (0–1)."""
    p = _coerce_float(price)
    if p is None or p == 0:
        return None
    if p > 0:
        return round(100.0 / (p + 100.0), 4)
    ap = abs(p)
    return round(ap / (ap + 100.0), 4)


def _experimental_carousel_price(market: str, row: dict[str, Any]) -> float | None:
    """Price shown for the tracked side: NRFI → under, YRFI → over."""
    m = str(market or "").lower()
    if m == "nrfi":
        u = _coerce_float(row.get("under_price"))
        if u is not None:
            return float(u)
    elif m == "yrfi":
        o = _coerce_float(row.get("over_price"))
        if o is not None:
            return float(o)
    return _coerce_float(row.get("over_price")) or _coerce_float(row.get("under_price"))


def _pick_experimental_nrfi_yrfi_card(
    *,
    nrfi_row: dict[str, Any] | None,
    yrfi_row: dict[str, Any] | None,
    model_recommended: str | None,
) -> tuple[str, dict[str, Any], str]:
    """One experimental card per game: classifier side when available; else a single fallback row."""
    if nrfi_row and not yrfi_row:
        return "nrfi", nrfi_row, "nrfi_only"
    if yrfi_row and not nrfi_row:
        return "yrfi", yrfi_row, "yrfi_only"
    if nrfi_row and yrfi_row:
        side = str(model_recommended or "").lower()
        if side == "nrfi":
            return "nrfi", nrfi_row, "model_pick"
        if side == "yrfi":
            return "yrfi", yrfi_row, "model_pick"
        return "nrfi", nrfi_row, "no_model_both_posted_fallback"
    raise ValueError("experimental NRFI/YRFI pick requires at least one market row")


def _experimental_first_inning_reasoning_notes(
    *,
    market: str,
    pick_reason: str,
    row: dict[str, Any],
    model_nrfi_probability: float | None = None,
    model_confidence: str | None = None,
) -> str:
    lines: list[str] = []
    mkt = str(market or "").lower()
    if model_nrfi_probability is not None:
        p = float(model_nrfi_probability)
        lines.append(
            f"Trained inning-1 model: P(NRFI)≈{p * 100:.0f}% · P(YRFI)≈{(1.0 - p) * 100:.0f}% (experimental; not the green board)."
        )
        pr = str(pick_reason or "")
        if pr == "model_pick":
            lines.append(
                "This carousel card follows the classifier: it shows the NRFI or YRFI book line that matches the model’s favored side."
            )
        elif pr.startswith("model_only"):
            lines.append(
                "No NRFI/YRFI row from books in our feed for this game — this card is the classifier pick only "
                "(add first-inning market ingest to attach prices / implied odds)."
            )
        elif mkt == "nrfi":
            lines.append("Shown line: NRFI (no run in the 1st) from the book.")
        elif mkt == "yrfi":
            lines.append("Shown line: YRFI (≥1 run in the 1st) from the book.")
        if model_confidence:
            lines.append(f"Model input confidence: {model_confidence}.")
    else:
        lines.append(
            "Experimental 1st-inning prop — not part of the green board. "
            "Without a trained inning-1 artifact, this row tracks posted odds (not a projection pick). "
            "Run feature build + train_inning1_nrfi + predict_inning1_nrfi to enable the classifier."
        )
    if pick_reason == "nrfi_only":
        lines.append("Only an NRFI line appears in our market feed for this game (no YRFI row).")
    elif pick_reason == "yrfi_only":
        lines.append("Only a YRFI line appears in our market feed for this game (no NRFI row).")
    elif pick_reason == "no_model_both_posted_fallback":
        lines.append(
            "Both NRFI and YRFI are posted, but there is no inning-1 prediction row for this game yet — "
            "showing the NRFI book row until predict_inning1_nrfi runs (then the model will pick the side)."
        )

    pprice = _experimental_carousel_price(market, row)
    ip = _american_to_implied_probability(pprice)
    if ip is not None and pprice is not None:
        side = "no run in the 1st" if str(market).lower() == "nrfi" else "≥1 run in the 1st"
        lines.append(f"Posted odds {pprice:+.0f} on this side → ~{ip * 100:.0f}% implied {side}.")

    return "\n".join(lines)


def _fmt_pct_or_raw(value: float | None, *, as_probability: bool = True) -> str | None:
    if value is None:
        return None
    v = float(value)
    if as_probability:
        if 0 <= v <= 1.0:
            return f"{v * 100:.0f}%"
        if 1.0 < v <= 100.0:
            return f"{v:.0f}%"
    return f"{v:.3f}"


def _format_nrfi_experimental_context_block(ctx: dict[str, Any] | None) -> str:
    """Human-readable inning-1 *context* (starters, top-of-order, park) — not a calibrated probability."""
    if not ctx:
        return ""
    lines: list[str] = [
        "What usually matters for inning 1 (heuristic snapshot from our board inputs — not a NRFI/YRFI model):",
    ]
    away_team = str(ctx.get("away_team") or "Away")
    home_team = str(ctx.get("home_team") or "Home")
    a5 = _coerce_float(ctx.get("away_lineup_top5_xwoba"))
    h5 = _coerce_float(ctx.get("home_lineup_top5_xwoba"))
    if a5 is not None or h5 is not None:
        a_s = _fmt_pct_or_raw(a5, as_probability=False) if a5 is not None else "—"
        h_s = _fmt_pct_or_raw(h5, as_probability=False) if h5 is not None else "—"
        lines.append(
            f"• Top-of-order quality (model xwOBA, top slots): {away_team} {a_s} · {home_team} {h_s}."
        )

    starters = ctx.get("starters") or {}
    for side, team_label in (("away", away_team), ("home", home_team)):
        sp = starters.get(side)
        if not sp:
            lines.append(f"• {team_label} starter: not on the board yet (lineups/starters ingest).")
            continue
        name = str(sp.get("pitcher_name") or "SP")
        bits: list[str] = []
        xw = _coerce_float(sp.get("xwoba_against"))
        if xw is not None:
            bits.append(f"xwOBA allowed {_fmt_pct_or_raw(xw, as_probability=False)}")
        csw = _coerce_float(sp.get("csw_pct"))
        if csw is not None:
            bits.append(f"CSW {_fmt_pct_or_raw(csw)}")
        wf = _coerce_float(sp.get("whiff_pct"))
        if wf is not None:
            bits.append(f"Whiff {_fmt_pct_or_raw(wf)}")
        aw = _coerce_float(sp.get("avg_walks"))
        if aw is not None:
            bits.append(f"~{aw:.1f} BB/start recent")
        ak = _coerce_float(sp.get("avg_strikeouts"))
        if ak is not None:
            bits.append(f"~{ak:.1f} K/start recent")
        sample = sp.get("sample_starts")
        tail = f" (last {sample} starts)" if sample else ""
        line = f"• {team_label} SP {name}"
        if bits:
            line += ": " + " · ".join(bits) + tail + "."
        else:
            line += ": detail pending."
        lines.append(line)

    env_bits: list[str] = []
    vr = _coerce_float(ctx.get("venue_run_factor"))
    if vr is not None:
        env_bits.append(f"park run factor {vr:.2f}")
    roof = str(ctx.get("roof_type") or "").strip()
    if roof:
        env_bits.append(roof)
    tf = _coerce_float(ctx.get("temperature_f"))
    if tf is not None:
        env_bits.append(f"{tf:.0f}°F")
    wf = _coerce_float(ctx.get("wind_speed_mph"))
    if wf is not None:
        env_bits.append(f"{wf:.0f} mph wind")
    if env_bits:
        lines.append("• Environment: " + " · ".join(env_bits) + ".")

    return "\n".join(lines)


def _fetch_nrfi_experimental_context_by_game(target_date: date) -> dict[int, dict[str, Any]]:
    """Per game_id: top-of-order xwOBA, both SPs (contact/walk/K recent), park/weather — for copy only."""
    if not _table_exists("games"):
        return {}
    game_start_order = _sql_order_nulls_last("g.game_start_ts")
    slim = _safe_frame(
        f"""
        WITH ranked_features AS (
            SELECT
                f.*,
                ROW_NUMBER() OVER (PARTITION BY f.game_id ORDER BY f.prediction_ts DESC) AS row_rank
            FROM game_features_totals f
            WHERE f.game_date = :target_date
        )
        SELECT
            g.game_id,
            g.away_team,
            g.home_team,
            COALESCE(v.roof_type, '') AS roof_type,
            CAST(rf.feature_payload ->> 'away_lineup_top5_xwoba' AS DOUBLE PRECISION) AS away_lineup_top5_xwoba,
            CAST(rf.feature_payload ->> 'home_lineup_top5_xwoba' AS DOUBLE PRECISION) AS home_lineup_top5_xwoba,
            CAST(rf.feature_payload ->> 'venue_run_factor' AS DOUBLE PRECISION) AS venue_run_factor,
            CAST(rf.feature_payload ->> 'temperature_f' AS DOUBLE PRECISION) AS temperature_f,
            CAST(rf.feature_payload ->> 'wind_speed_mph' AS DOUBLE PRECISION) AS wind_speed_mph
        FROM games g
        LEFT JOIN dim_venues v ON v.venue_id = g.venue_id
        LEFT JOIN ranked_features rf ON rf.game_id = g.game_id AND rf.row_rank = 1
        WHERE g.game_date = :target_date
        ORDER BY {game_start_order}, g.away_team, g.home_team, g.game_id
        """,
        {"target_date": target_date},
    )
    out: dict[int, dict[str, Any]] = {}
    for row in _frame_records(slim):
        gid = int(row["game_id"])
        out[gid] = {
            "away_team": row.get("away_team"),
            "home_team": row.get("home_team"),
            "away_lineup_top5_xwoba": _coerce_float(row.get("away_lineup_top5_xwoba")),
            "home_lineup_top5_xwoba": _coerce_float(row.get("home_lineup_top5_xwoba")),
            "venue_run_factor": _coerce_float(row.get("venue_run_factor")),
            "roof_type": str(row.get("roof_type") or "").strip() or None,
            "temperature_f": _coerce_float(row.get("temperature_f")),
            "wind_speed_mph": _coerce_float(row.get("wind_speed_mph")),
            "starters": {},
        }

    if not _table_exists("pitcher_starts"):
        return out

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
    starter_records = _starter_records_prefer_boxscore(_frame_records(starters_frame), target_date)
    recent_cache: dict[int, dict[str, Any]] = {}

    def _recent(pid: int) -> dict[str, Any]:
        if pid not in recent_cache:
            recent_cache[pid] = _fetch_starter_recent_form(pid, target_date) or {}
        return recent_cache[pid]

    for st in starter_records:
        gid = int(st["game_id"])
        base = out.get(gid)
        if not base:
            continue
        team = str(st.get("team") or "").strip()
        away_t = str(base.get("away_team") or "").strip()
        home_t = str(base.get("home_team") or "").strip()
        if team == away_t:
            side = "away"
        elif team == home_t:
            side = "home"
        else:
            continue
        pid = st.get("pitcher_id")
        rf = _recent(int(pid)) if pid is not None else {}
        base["starters"][side] = {
            "pitcher_name": st.get("pitcher_name"),
            "throws": st.get("throws"),
            "xwoba_against": _coerce_float(st.get("xwoba_against")),
            "csw_pct": _coerce_float(st.get("csw_pct")),
            "whiff_pct": _coerce_float(st.get("whiff_pct")),
            "avg_walks": _coerce_float(rf.get("avg_walks")),
            "avg_strikeouts": _coerce_float(rf.get("avg_strikeouts")),
            "sample_starts": rf.get("sample_starts"),
        }
    return out


def _fetch_inning1_nrfi_predictions_map(target_date: date) -> dict[int, dict[str, Any]]:
    """Latest inning-1 NRFI model row per game (trained classifier), if table exists."""
    if not _table_exists("predictions_inning1_nrfi"):
        return {}
    frame = _safe_frame(
        """
        WITH ranked AS (
            SELECT
                game_id,
                predicted_nrfi_probability,
                predicted_yrfi_probability,
                recommended_side,
                confidence_level,
                suppress_reason,
                model_version,
                ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY prediction_ts DESC) AS rn
            FROM predictions_inning1_nrfi
            WHERE game_date = :target_date
        )
        SELECT game_id, predicted_nrfi_probability, predicted_yrfi_probability, recommended_side,
               confidence_level, suppress_reason, model_version
        FROM ranked
        WHERE rn = 1
        """,
        {"target_date": str(target_date)},
    )
    if frame.empty:
        return {}
    out: dict[int, dict[str, Any]] = {}
    for row in _frame_records(frame):
        gid = row.get("game_id")
        if gid is None:
            continue
        out[int(gid)] = dict(row)
    return out


def _infer_inning1_experimental_market_from_model(model_row: dict[str, Any]) -> tuple[str, str]:
    """Return (market_key, pick_reason) for a classifier row when there is no book row."""
    rec = str(model_row.get("recommended_side") or "").strip().lower()
    if rec in ("nrfi", "yrfi"):
        return rec, "model_only_classifier"
    p = _coerce_float(model_row.get("predicted_nrfi_probability"))
    if p is not None:
        return ("nrfi", "model_only_infer_probability") if float(p) >= 0.5 else ("yrfi", "model_only_infer_probability")
    return "nrfi", "model_only_default_nrfi"


def _fetch_games_metadata_for_ids(target_date: date, game_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not game_ids or not _table_exists("games"):
        return {}
    unique_ids = sorted({int(g) for g in game_ids})
    placeholders = ", ".join(f":gid{i}" for i in range(len(unique_ids)))
    params: dict[str, Any] = {"d": str(target_date)}
    for i, gid in enumerate(unique_ids):
        params[f"gid{i}"] = int(gid)
    frame = _safe_frame(
        f"""
        SELECT game_id, away_team, home_team, game_start_ts
        FROM games
        WHERE game_date = :d AND game_id IN ({placeholders})
        """,
        params,
    )
    out: dict[int, dict[str, Any]] = {}
    for row in _frame_records(frame):
        gid = row.get("game_id")
        if gid is None:
            continue
        out[int(gid)] = dict(row)
    return out


def _inning1_experimental_card_from_model_row(
    *,
    game_id: int,
    model_row: dict[str, Any],
    game_row: dict[str, Any],
    ctx_map: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    market_key, pick_reason = _infer_inning1_experimental_market_from_model(model_row)
    model_p = _coerce_float(model_row.get("predicted_nrfi_probability"))
    model_conf = str(model_row.get("confidence_level") or "") if model_row else None
    model_rec: str | None = None
    if str(model_row.get("recommended_side") or "").lower() in ("nrfi", "yrfi"):
        model_rec = str(model_row.get("recommended_side")).lower()
    away_team = game_row.get("away_team") or "Away"
    home_team = game_row.get("home_team") or "Home"
    synthetic = {
        "away_team": away_team,
        "home_team": home_team,
        "game_start_ts": game_row.get("game_start_ts"),
        "over_price": None,
        "under_price": None,
        "sportsbook": None,
        "book_row_present": False,
    }
    market_label = _experimental_market_display_name(market_key)
    inning1_ctx = ctx_map.get(int(game_id))
    notes = _experimental_first_inning_reasoning_notes(
        market=market_key,
        pick_reason=pick_reason,
        row=synthetic,
        model_nrfi_probability=model_p,
        model_confidence=model_conf or None,
    )
    extra = _format_nrfi_experimental_context_block(inning1_ctx)
    if extra:
        notes = f"{notes}\n\n{extra}"
    return {
        "game_id": int(game_id),
        "game_start_ts": game_row.get("game_start_ts"),
        "market": market_key,
        "market_label": market_label,
        "subject_label": f"{away_team} at {home_team}",
        "pick_label": f"Pick: {market_label}",
        "sportsbook": None,
        "price": None,
        "experimental_pick_reason": pick_reason,
        "inning1_context": inning1_ctx,
        "notes_display": notes,
        "model_nrfi_probability": model_p,
        "model_confidence": model_conf or None,
        "model_recommended_side": model_rec,
        "model_agrees_with_card": (
            True
            if model_rec and str(market_key).lower() == model_rec
            else (False if model_rec else None)
        ),
        "inning1_model_only": True,
    }


def _build_inning1_model_only_experimental_cards(
    target_date: date,
    ctx_map: dict[int, dict[str, Any]],
    inning1_model_map: dict[int, dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """When ``game_markets`` has no NRFI/YRFI rows, still surface classifier output per game."""
    if not inning1_model_map:
        return []
    meta = _fetch_games_metadata_for_ids(target_date, list(inning1_model_map.keys()))
    cards: list[dict[str, Any]] = []
    for game_id in sorted(inning1_model_map.keys()):
        g = meta.get(int(game_id))
        model_row = inning1_model_map.get(int(game_id))
        if not g or not model_row:
            continue
        cards.append(
            _inning1_experimental_card_from_model_row(
                game_id=int(game_id),
                model_row=model_row,
                game_row=g,
                ctx_map=ctx_map,
            )
        )
    cards.sort(
        key=lambda card: (
            str(card.get("game_start_ts") or ""),
            str(card.get("subject_label") or ""),
            str(card.get("market_label") or ""),
        )
    )
    return cards[: max(1, min(limit, 48))]


def _fetch_experimental_market_cards(target_date: date, limit: int = 24) -> list[dict[str, Any]]:
    ctx_map = _fetch_nrfi_experimental_context_by_game(target_date)
    inning1_model_map = _fetch_inning1_nrfi_predictions_map(target_date)

    if not _table_exists("game_markets"):
        return _build_inning1_model_only_experimental_cards(
            target_date,
            ctx_map,
            inning1_model_map,
            limit,
        )

    params: dict[str, Any] = {"target_date": str(target_date)}
    market_conditions: list[str] = []
    for index, market_key in enumerate(EXPERIMENTAL_CAROUSEL_MARKETS):
        param_name = f"experimental_market_{index}"
        params[param_name] = market_key
        market_conditions.append(f"gm.market_type = :{param_name}")

    frame = _safe_frame(
        f"""
        WITH ranked AS (
            SELECT
                gm.game_id,
                gm.market_type,
                gm.sportsbook,
                gm.over_price,
                gm.under_price,
                gm.snapshot_ts,
                g.away_team,
                g.home_team,
                g.game_start_ts,
                ROW_NUMBER() OVER (
                    PARTITION BY gm.game_id, gm.market_type, COALESCE(gm.sportsbook, '')
                    ORDER BY gm.snapshot_ts DESC
                ) AS row_rank
            FROM game_markets gm
            JOIN games g ON g.game_id = gm.game_id AND g.game_date = gm.game_date
            WHERE gm.game_date = :target_date
              AND ({' OR '.join(market_conditions)})
              AND (gm.over_price IS NOT NULL OR gm.under_price IS NOT NULL)
        )
        SELECT game_id, market_type, sportsbook, over_price, under_price, snapshot_ts, away_team, home_team, game_start_ts
        FROM ranked
        WHERE row_rank = 1
        ORDER BY game_start_ts, game_id, market_type, sportsbook
        """,
        params,
    )
    per_game: dict[int, dict[str, dict[str, Any]]] = {}
    if not frame.empty:
        grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
        for record in _frame_records(frame):
            game_id = record.get("game_id")
            market = str(record.get("market_type") or "").lower()
            if game_id is None or not market:
                continue
            grouped.setdefault((int(game_id), market), []).append(record)

        for (game_id, market), market_rows in grouped.items():
            chosen = max(
                market_rows,
                key=lambda row: (
                    -999999
                    if _coerce_float(row.get("over_price")) is None and _coerce_float(row.get("under_price")) is None
                    else max(
                        _coerce_float(row.get("over_price")) or -999999,
                        _coerce_float(row.get("under_price")) or -999999,
                    ),
                    str(row.get("snapshot_ts") or ""),
                ),
            )
            per_game.setdefault(int(game_id), {})[market] = chosen

    cards: list[dict[str, Any]] = []
    for game_id in sorted(per_game.keys()):
        markets = per_game[game_id]
        nrfi = markets.get("nrfi")
        yrfi = markets.get("yrfi")
        if not nrfi and not yrfi:
            continue
        model_row = inning1_model_map.get(int(game_id))
        model_p = _coerce_float(model_row.get("predicted_nrfi_probability")) if model_row else None
        model_conf = str(model_row.get("confidence_level") or "") if model_row else None
        model_rec: str | None = None
        if model_row and str(model_row.get("recommended_side") or "").lower() in ("nrfi", "yrfi"):
            model_rec = str(model_row.get("recommended_side")).lower()
        try:
            market_key, chosen, pick_reason = _pick_experimental_nrfi_yrfi_card(
                nrfi_row=nrfi,
                yrfi_row=yrfi,
                model_recommended=model_rec,
            )
        except ValueError:
            continue
        price = _experimental_carousel_price(market_key, chosen)
        market_label = _experimental_market_display_name(market_key)
        away_team = chosen.get("away_team") or "Away"
        home_team = chosen.get("home_team") or "Home"
        inning1_ctx = ctx_map.get(int(game_id))
        notes = _experimental_first_inning_reasoning_notes(
            market=market_key,
            pick_reason=pick_reason,
            row=chosen,
            model_nrfi_probability=model_p,
            model_confidence=model_conf or None,
        )
        extra = _format_nrfi_experimental_context_block(inning1_ctx)
        if extra:
            notes = f"{notes}\n\n{extra}"
        cards.append(
            {
                "game_id": game_id,
                "game_start_ts": chosen.get("game_start_ts"),
                "market": market_key,
                "market_label": market_label,
                "subject_label": f"{away_team} at {home_team}",
                "pick_label": market_label,
                "sportsbook": chosen.get("sportsbook"),
                "price": price,
                "experimental_pick_reason": pick_reason,
                "inning1_context": inning1_ctx,
                "notes_display": notes,
                "model_nrfi_probability": model_p,
                "model_confidence": model_conf or None,
                "model_recommended_side": model_rec,
                "model_agrees_with_card": (
                    True
                    if model_rec and str(market_key).lower() == model_rec
                    else (False if model_rec else None)
                ),
                "inning1_model_only": False,
            }
        )

    seen_game_ids = {int(c["game_id"]) for c in cards}
    missing_ids = [int(gid) for gid in sorted(inning1_model_map.keys()) if int(gid) not in seen_game_ids]
    if missing_ids:
        meta_missing = _fetch_games_metadata_for_ids(target_date, missing_ids)
        for gid in missing_ids:
            model_row = inning1_model_map.get(int(gid))
            g = meta_missing.get(int(gid))
            if not model_row or not g:
                continue
            cards.append(
                _inning1_experimental_card_from_model_row(
                    game_id=int(gid),
                    model_row=model_row,
                    game_row=g,
                    ctx_map=ctx_map,
                )
            )

    cards.sort(
        key=lambda card: (
            str(card.get("game_start_ts") or ""),
            str(card.get("subject_label") or ""),
            str(card.get("market_label") or ""),
        )
    )
    return cards[: max(1, min(limit, 48))]


def _fetch_experiment_summary(target_date: date, window_days: int = 14) -> dict[str, Any]:
    """Compare fundamentals-only vs market-calibrated vs market line accuracy."""
    from datetime import timedelta

    start_date = target_date - timedelta(days=window_days - 1)
    totals_daily: list[dict[str, Any]] = []
    strikeouts_daily: list[dict[str, Any]] = []

    if _table_exists("predictions_totals") and _table_exists("games"):
        frame = _safe_frame(
            """
            WITH ranked AS (
                SELECT p.*, ROW_NUMBER() OVER (
                    PARTITION BY p.game_id ORDER BY p.prediction_ts DESC
                ) AS rn
                FROM predictions_totals p
                WHERE p.game_date BETWEEN :start_date AND :end_date
            )
            SELECT
                p.game_date,
                p.predicted_total_runs,
                p.predicted_total_fundamentals,
                p.market_total,
                g.total_runs AS actual
            FROM ranked p
            INNER JOIN games g ON g.game_id = p.game_id AND g.game_date = p.game_date
            WHERE p.rn = 1
              AND g.total_runs IS NOT NULL
            ORDER BY p.game_date
            """,
            {"start_date": start_date, "end_date": target_date},
        )
        by_date: dict[str, list[dict]] = {}
        for row in _frame_records(frame):
            d = str(row.get("game_date", ""))[:10]
            by_date.setdefault(d, []).append(row)
        for d in sorted(by_date):
            rows = by_date[d]
            cal_errors, fund_errors, mkt_errors = [], [], []
            cal_preds, fund_preds, mkt_preds, actuals = [], [], [], []
            fund_won, fund_lost, fund_push = 0, 0, 0
            cal_won, cal_lost, cal_push = 0, 0, 0
            for r in rows:
                actual = _to_float(r.get("actual"))
                cal = _to_float(r.get("predicted_total_runs"))
                fund = _to_float(r.get("predicted_total_fundamentals"))
                mkt = _to_float(r.get("market_total"))
                if actual is not None:
                    actuals.append(actual)
                if cal is not None:
                    cal_preds.append(cal)
                if fund is not None:
                    fund_preds.append(fund)
                if mkt is not None:
                    mkt_preds.append(mkt)
                if actual is not None and cal is not None:
                    cal_errors.append(abs(actual - cal))
                if actual is not None and fund is not None:
                    fund_errors.append(abs(actual - fund))
                if actual is not None and mkt is not None:
                    mkt_errors.append(abs(actual - mkt))
                # W-L record: did the model's predicted side match the actual side?
                if actual is not None and mkt is not None:
                    actual_side = "over" if actual > mkt else ("under" if actual < mkt else "push")
                    if fund is not None:
                        fund_side = "over" if fund >= mkt else "under"
                        if actual_side == "push":
                            fund_push += 1
                        elif fund_side == actual_side:
                            fund_won += 1
                        else:
                            fund_lost += 1
                    if cal is not None:
                        cal_side = "over" if cal >= mkt else "under"
                        if actual_side == "push":
                            cal_push += 1
                        elif cal_side == actual_side:
                            cal_won += 1
                        else:
                            cal_lost += 1
            totals_daily.append({
                "date": d,
                "games": len(rows),
                "actual_avg": round(sum(actuals) / len(actuals), 2) if actuals else None,
                "calibrated_mae": round(sum(cal_errors) / len(cal_errors), 3) if cal_errors else None,
                "fundamentals_mae": round(sum(fund_errors) / len(fund_errors), 3) if fund_errors else None,
                "market_mae": round(sum(mkt_errors) / len(mkt_errors), 3) if mkt_errors else None,
                "calibrated_avg": round(sum(cal_preds) / len(cal_preds), 2) if cal_preds else None,
                "fundamentals_avg": round(sum(fund_preds) / len(fund_preds), 2) if fund_preds else None,
                "market_avg": round(sum(mkt_preds) / len(mkt_preds), 2) if mkt_preds else None,
                "fund_won": fund_won, "fund_lost": fund_lost, "fund_push": fund_push,
                "cal_won": cal_won, "cal_lost": cal_lost, "cal_push": cal_push,
            })

    if _table_exists("predictions_pitcher_strikeouts") and _table_exists("pitcher_starts"):
        frame = _safe_frame(
            """
            WITH ranked AS (
                SELECT p.*, ROW_NUMBER() OVER (
                    PARTITION BY p.game_id, p.pitcher_id ORDER BY p.prediction_ts DESC
                ) AS rn
                FROM predictions_pitcher_strikeouts p
                WHERE p.game_date BETWEEN :start_date AND :end_date
            )
            SELECT
                p.game_date,
                p.predicted_strikeouts,
                p.predicted_strikeouts_fundamentals,
                p.market_line,
                ps.strikeouts AS actual
            FROM ranked p
            INNER JOIN pitcher_starts ps
                ON ps.game_id = p.game_id AND ps.pitcher_id = p.pitcher_id AND ps.game_date = p.game_date
            WHERE p.rn = 1
              AND ps.strikeouts IS NOT NULL
            ORDER BY p.game_date
            """,
            {"start_date": start_date, "end_date": target_date},
        )
        by_date_k: dict[str, list[dict]] = {}
        for row in _frame_records(frame):
            d = str(row.get("game_date", ""))[:10]
            by_date_k.setdefault(d, []).append(row)
        for d in sorted(by_date_k):
            rows = by_date_k[d]
            cal_errors, fund_errors, mkt_errors = [], [], []
            cal_preds, fund_preds, mkt_preds, actuals = [], [], [], []
            fund_won, fund_lost, fund_push = 0, 0, 0
            cal_won, cal_lost, cal_push = 0, 0, 0
            for r in rows:
                actual = _to_float(r.get("actual"))
                cal = _to_float(r.get("predicted_strikeouts"))
                fund = _to_float(r.get("predicted_strikeouts_fundamentals"))
                mkt = _to_float(r.get("market_line"))
                if actual is not None:
                    actuals.append(actual)
                if cal is not None:
                    cal_preds.append(cal)
                if fund is not None:
                    fund_preds.append(fund)
                if mkt is not None:
                    mkt_preds.append(mkt)
                if actual is not None and cal is not None:
                    cal_errors.append(abs(actual - cal))
                if actual is not None and fund is not None:
                    fund_errors.append(abs(actual - fund))
                if actual is not None and mkt is not None:
                    mkt_errors.append(abs(actual - mkt))
                if actual is not None and mkt is not None:
                    actual_side = "over" if actual > mkt else ("under" if actual < mkt else "push")
                    if fund is not None:
                        fund_side = "over" if fund >= mkt else "under"
                        if actual_side == "push":
                            fund_push += 1
                        elif fund_side == actual_side:
                            fund_won += 1
                        else:
                            fund_lost += 1
                    if cal is not None:
                        cal_side = "over" if cal >= mkt else "under"
                        if actual_side == "push":
                            cal_push += 1
                        elif cal_side == actual_side:
                            cal_won += 1
                        else:
                            cal_lost += 1
            strikeouts_daily.append({
                "date": d,
                "pitchers": len(rows),
                "actual_avg": round(sum(actuals) / len(actuals), 2) if actuals else None,
                "calibrated_mae": round(sum(cal_errors) / len(cal_errors), 3) if cal_errors else None,
                "fundamentals_mae": round(sum(fund_errors) / len(fund_errors), 3) if fund_errors else None,
                "market_mae": round(sum(mkt_errors) / len(mkt_errors), 3) if mkt_errors else None,
                "calibrated_avg": round(sum(cal_preds) / len(cal_preds), 2) if cal_preds else None,
                "fundamentals_avg": round(sum(fund_preds) / len(fund_preds), 2) if fund_preds else None,
                "market_avg": round(sum(mkt_preds) / len(mkt_preds), 2) if mkt_preds else None,
                "fund_won": fund_won, "fund_lost": fund_lost, "fund_push": fund_push,
                "cal_won": cal_won, "cal_lost": cal_lost, "cal_push": cal_push,
            })

    def _agg(daily: list[dict], count_key: str) -> dict[str, Any]:
        total_count = sum(d.get(count_key, 0) for d in daily)
        actual_vals = [d["actual_avg"] for d in daily if d.get("actual_avg") is not None]
        cal_vals = [d["calibrated_mae"] for d in daily if d.get("calibrated_mae") is not None]
        fund_vals = [d["fundamentals_mae"] for d in daily if d.get("fundamentals_mae") is not None]
        mkt_vals = [d["market_mae"] for d in daily if d.get("market_mae") is not None]
        cal_avgs = [d["calibrated_avg"] for d in daily if d.get("calibrated_avg") is not None]
        fund_avgs = [d["fundamentals_avg"] for d in daily if d.get("fundamentals_avg") is not None]
        mkt_avgs = [d["market_avg"] for d in daily if d.get("market_avg") is not None]
        return {
            "days": len(daily),
            "total_count": total_count,
            "actual_avg": round(sum(actual_vals) / len(actual_vals), 2) if actual_vals else None,
            "calibrated_mae": round(sum(cal_vals) / len(cal_vals), 3) if cal_vals else None,
            "fundamentals_mae": round(sum(fund_vals) / len(fund_vals), 3) if fund_vals else None,
            "market_mae": round(sum(mkt_vals) / len(mkt_vals), 3) if mkt_vals else None,
            "calibrated_avg": round(sum(cal_avgs) / len(cal_avgs), 2) if cal_avgs else None,
            "fundamentals_avg": round(sum(fund_avgs) / len(fund_avgs), 2) if fund_avgs else None,
            "market_avg": round(sum(mkt_avgs) / len(mkt_avgs), 2) if mkt_avgs else None,
            "fund_won": sum(d.get("fund_won", 0) for d in daily),
            "fund_lost": sum(d.get("fund_lost", 0) for d in daily),
            "fund_push": sum(d.get("fund_push", 0) for d in daily),
            "cal_won": sum(d.get("cal_won", 0) for d in daily),
            "cal_lost": sum(d.get("cal_lost", 0) for d in daily),
            "cal_push": sum(d.get("cal_push", 0) for d in daily),
        }

    first5_daily: list[dict[str, Any]] = []

    if _table_exists("predictions_first5_totals") and _table_exists("games"):
        frame = _safe_frame(
            """
            WITH ranked AS (
                SELECT p.*, ROW_NUMBER() OVER (
                    PARTITION BY p.game_id ORDER BY p.prediction_ts DESC
                ) AS rn
                FROM predictions_first5_totals p
                WHERE p.game_date BETWEEN :start_date AND :end_date
            )
            SELECT
                p.game_date,
                p.predicted_total_runs,
                p.predicted_total_fundamentals,
                p.market_total,
                g.total_runs_first5 AS actual
            FROM ranked p
            INNER JOIN games g ON g.game_id = p.game_id AND g.game_date = p.game_date
            WHERE p.rn = 1
              AND g.total_runs_first5 IS NOT NULL
            ORDER BY p.game_date
            """,
            {"start_date": start_date, "end_date": target_date},
        )
        by_date_f5: dict[str, list[dict]] = {}
        for row in _frame_records(frame):
            d = str(row.get("game_date", ""))[:10]
            by_date_f5.setdefault(d, []).append(row)
        for d in sorted(by_date_f5):
            rows = by_date_f5[d]
            cal_errors, fund_errors, mkt_errors = [], [], []
            cal_preds, fund_preds, mkt_preds, actuals = [], [], [], []
            fund_won, fund_lost, fund_push = 0, 0, 0
            cal_won, cal_lost, cal_push = 0, 0, 0
            for r in rows:
                actual = _to_float(r.get("actual"))
                cal = _to_float(r.get("predicted_total_runs"))
                fund = _to_float(r.get("predicted_total_fundamentals"))
                mkt = _to_float(r.get("market_total"))
                if actual is not None:
                    actuals.append(actual)
                if cal is not None:
                    cal_preds.append(cal)
                if fund is not None:
                    fund_preds.append(fund)
                if mkt is not None:
                    mkt_preds.append(mkt)
                if actual is not None and cal is not None:
                    cal_errors.append(abs(actual - cal))
                if actual is not None and fund is not None:
                    fund_errors.append(abs(actual - fund))
                if actual is not None and mkt is not None:
                    mkt_errors.append(abs(actual - mkt))
                if actual is not None and mkt is not None:
                    actual_side = "over" if actual > mkt else ("under" if actual < mkt else "push")
                    if fund is not None:
                        fund_side = "over" if fund >= mkt else "under"
                        if actual_side == "push":
                            fund_push += 1
                        elif fund_side == actual_side:
                            fund_won += 1
                        else:
                            fund_lost += 1
                    if cal is not None:
                        cal_side = "over" if cal >= mkt else "under"
                        if actual_side == "push":
                            cal_push += 1
                        elif cal_side == actual_side:
                            cal_won += 1
                        else:
                            cal_lost += 1
            first5_daily.append({
                "date": d,
                "games": len(rows),
                "actual_avg": round(sum(actuals) / len(actuals), 2) if actuals else None,
                "calibrated_mae": round(sum(cal_errors) / len(cal_errors), 3) if cal_errors else None,
                "fundamentals_mae": round(sum(fund_errors) / len(fund_errors), 3) if fund_errors else None,
                "market_mae": round(sum(mkt_errors) / len(mkt_errors), 3) if mkt_errors else None,
                "calibrated_avg": round(sum(cal_preds) / len(cal_preds), 2) if cal_preds else None,
                "fundamentals_avg": round(sum(fund_preds) / len(fund_preds), 2) if fund_preds else None,
                "market_avg": round(sum(mkt_preds) / len(mkt_preds), 2) if mkt_preds else None,
                "fund_won": fund_won, "fund_lost": fund_lost, "fund_push": fund_push,
                "cal_won": cal_won, "cal_lost": cal_lost, "cal_push": cal_push,
            })

    return {
        "target_date": target_date.isoformat(),
        "window_days": window_days,
        "start_date": start_date.isoformat(),
        "totals": {"daily": totals_daily, "aggregate": _agg(totals_daily, "games")},
        "strikeouts": {"daily": strikeouts_daily, "aggregate": _agg(strikeouts_daily, "pitchers")},
        "first5": {"daily": first5_daily, "aggregate": _agg(first5_daily, "games")},
    }


def _fetch_experiment_daily_detail(target_date: date) -> dict[str, Any]:
    """Per-game detail for a single date: totals + strikeouts."""
    totals_games: list[dict[str, Any]] = []
    strikeouts_games: list[dict[str, Any]] = []

    if _table_exists("predictions_totals") and _table_exists("games"):
        frame = _safe_frame(
            """
            WITH ranked AS (
                SELECT p.*, ROW_NUMBER() OVER (
                    PARTITION BY p.game_id ORDER BY p.prediction_ts DESC
                ) AS rn
                FROM predictions_totals p
                WHERE p.game_date = :target_date
            )
            SELECT
                p.game_id,
                g.away_team || ' @ ' || g.home_team AS matchup,
                p.predicted_total_runs   AS calibrated,
                p.predicted_total_fundamentals AS fundamentals,
                p.market_total           AS market,
                g.total_runs             AS actual
            FROM ranked p
            INNER JOIN games g ON g.game_id = p.game_id AND g.game_date = p.game_date
            WHERE p.rn = 1
            ORDER BY g.away_team
            """,
            {"target_date": target_date},
        )
        for r in _frame_records(frame):
            actual = _to_float(r.get("actual"))
            cal = _to_float(r.get("calibrated"))
            fund = _to_float(r.get("fundamentals"))
            mkt = _to_float(r.get("market"))
            totals_games.append({
                "game_id": r.get("game_id"),
                "matchup": r.get("matchup", ""),
                "calibrated": round(cal, 2) if cal is not None else None,
                "fundamentals": round(fund, 2) if fund is not None else None,
                "market": round(mkt, 2) if mkt is not None else None,
                "actual": int(actual) if actual is not None else None,
                "cal_error": round(abs(actual - cal), 2) if actual is not None and cal is not None else None,
                "fund_error": round(abs(actual - fund), 2) if actual is not None and fund is not None else None,
                "mkt_error": round(abs(actual - mkt), 2) if actual is not None and mkt is not None else None,
            })

    if _table_exists("predictions_pitcher_strikeouts") and _table_exists("pitcher_starts"):
        frame = _safe_frame(
            """
            WITH ranked AS (
                SELECT p.*, ROW_NUMBER() OVER (
                    PARTITION BY p.game_id, p.pitcher_id ORDER BY p.prediction_ts DESC
                ) AS rn
                FROM predictions_pitcher_strikeouts p
                WHERE p.game_date = :target_date
            )
            SELECT
                p.game_id,
                COALESCE(dp.full_name, 'ID ' || p.pitcher_id) AS pitcher_name,
                p.team,
                p.predicted_strikeouts         AS calibrated,
                p.predicted_strikeouts_fundamentals AS fundamentals,
                p.market_line                  AS market,
                ps.strikeouts                  AS actual
            FROM ranked p
            LEFT JOIN dim_players dp ON dp.player_id = p.pitcher_id
            LEFT JOIN pitcher_starts ps
                ON ps.game_id = p.game_id AND ps.pitcher_id = p.pitcher_id AND ps.game_date = p.game_date
            WHERE p.rn = 1
            ORDER BY p.team, pitcher_name
            """,
            {"target_date": target_date},
        )
        for r in _frame_records(frame):
            actual = _to_float(r.get("actual"))
            cal = _to_float(r.get("calibrated"))
            fund = _to_float(r.get("fundamentals"))
            mkt = _to_float(r.get("market"))
            strikeouts_games.append({
                "game_id": r.get("game_id"),
                "pitcher": r.get("pitcher_name", ""),
                "team": r.get("team", ""),
                "calibrated": round(cal, 2) if cal is not None else None,
                "fundamentals": round(fund, 2) if fund is not None else None,
                "market": round(mkt, 2) if mkt is not None else None,
                "actual": int(actual) if actual is not None else None,
                "cal_error": round(abs(actual - cal), 2) if actual is not None and cal is not None else None,
                "fund_error": round(abs(actual - fund), 2) if actual is not None and fund is not None else None,
                "mkt_error": round(abs(actual - mkt), 2) if actual is not None and mkt is not None else None,
            })

    first5_games: list[dict[str, Any]] = []

    if _table_exists("predictions_first5_totals") and _table_exists("games"):
        frame = _safe_frame(
            """
            WITH ranked AS (
                SELECT p.*, ROW_NUMBER() OVER (
                    PARTITION BY p.game_id ORDER BY p.prediction_ts DESC
                ) AS rn
                FROM predictions_first5_totals p
                WHERE p.game_date = :target_date
            )
            SELECT
                p.game_id,
                g.away_team || ' @ ' || g.home_team AS matchup,
                p.predicted_total_runs   AS calibrated,
                p.predicted_total_fundamentals AS fundamentals,
                p.market_total           AS market,
                g.total_runs_first5      AS actual
            FROM ranked p
            INNER JOIN games g ON g.game_id = p.game_id AND g.game_date = p.game_date
            WHERE p.rn = 1
            ORDER BY g.away_team
            """,
            {"target_date": target_date},
        )
        for r in _frame_records(frame):
            actual = _to_float(r.get("actual"))
            cal = _to_float(r.get("calibrated"))
            fund = _to_float(r.get("fundamentals"))
            mkt = _to_float(r.get("market"))
            first5_games.append({
                "game_id": r.get("game_id"),
                "matchup": r.get("matchup", ""),
                "calibrated": round(cal, 2) if cal is not None else None,
                "fundamentals": round(fund, 2) if fund is not None else None,
                "market": round(mkt, 2) if mkt is not None else None,
                "actual": int(actual) if actual is not None else None,
                "cal_error": round(abs(actual - cal), 2) if actual is not None and cal is not None else None,
                "fund_error": round(abs(actual - fund), 2) if actual is not None and fund is not None else None,
                "mkt_error": round(abs(actual - mkt), 2) if actual is not None and mkt is not None else None,
            })

    return {
        "target_date": target_date.isoformat(),
        "totals": totals_games,
        "strikeouts": strikeouts_games,
        "first5": first5_games,
    }


def _rank_and_cap_hitter_rows_by_probability(
    rows: list[dict[str, Any]],
    *,
    top_n: int,
) -> list[dict[str, Any]]:
    """Highest model P(hit) first; optional cap for a short daily review list."""

    def sort_key(row: dict[str, Any]) -> float:
        v = _to_float(row.get("predicted_hit_probability"))
        return float(v) if v is not None else float("-inf")

    ordered = sorted(rows, key=sort_key, reverse=True)
    if top_n <= 0 or len(ordered) <= top_n:
        return ordered
    return ordered[:top_n]


def _fetch_daily_results(
    target_date: date,
    hit_min_probability: float = 0.35,
    hitter_top_n: int = 24,
) -> dict[str, Any]:
    # Always mirror the main board ``best_bets`` (same defaults as index.html) so Daily Results matches the game board.
    # Single board fetch for green + watchlist + Top EV (was 3× identical work per request).
    daily_results_board_rows = _fetch_game_board(
        target_date,
        hit_limit_per_team=GAME_BOARD_UI_DEFAULT_HIT_LIMIT_PER_TEAM,
        min_probability=GAME_BOARD_UI_DEFAULT_MIN_HIT_PROBABILITY,
        confirmed_only=GAME_BOARD_UI_DEFAULT_CONFIRMED_ONLY,
        include_inferred=GAME_BOARD_UI_DEFAULT_INCLUDE_INFERRED,
    )
    ai_pick_rows = _live_green_board_rows_for_daily_results(
        target_date,
        board_rows=daily_results_board_rows,
    )
    live_green_fallback = True
    watchlist_archived = _fetch_watchlist_pick_results(target_date)
    live_watchlist = _live_watchlist_board_rows_for_daily_results(
        target_date,
        board_rows=daily_results_board_rows,
    )
    watchlist_rows, live_watchlist_fallback = _merge_watchlist_daily_results(
        watchlist_archived,
        live_watchlist,
    )
    experimental_rows = _fetch_experimental_pick_results(target_date)
    _maybe_insert_board_top_ev_run_snapshots(target_date, daily_results_board_rows)
    _maybe_insert_board_top_ev_snapshots(target_date, daily_results_board_rows)
    top_ev_rows = _live_top_ev_rows_for_daily_results(
        target_date,
        board_rows=daily_results_board_rows,
    )
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
                p.predicted_total_fundamentals,
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
            # Use stats-only (fundamentals) as the primary grading model; fall back
            # to blended if fundamentals is unavailable.
            fundamentals_total = _to_float(row.get("predicted_total_fundamentals"))
            blended_total = _to_float(row.get("predicted_total_runs"))
            primary_total = fundamentals_total if fundamentals_total is not None else blended_total
            market_total_val = _to_float(row.get("market_total"))
            recommended_side = _recommended_side(primary_total, market_total_val)
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
                    "predicted_total_fundamentals": row.get("predicted_total_fundamentals"),
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
                        ORDER BY g.game_start_ts, p.predicted_hit_probability DESC, {lineup_slot_order}, player_name, p.game_id, p.player_id
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
        hitter_rows = _rank_and_cap_hitter_rows_by_probability(
            hitter_rows,
            top_n=hitter_top_n,
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
                p.predicted_strikeouts_fundamentals,
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
            ORDER BY g.game_start_ts, CASE WHEN p.market_line IS NULL THEN 1 ELSE 0 END, p.team, pitcher_name, p.game_id, p.pitcher_id
            """,
            {"target_date": target_date},
        )
        for row in _frame_records(strikeout_frame):
            is_final = _is_final_game_status(row.get("game_status"))
            # Use stats-only (fundamentals) as the primary grading model; fall back
            # to blended if fundamentals is unavailable.
            fund_ks = row.get("predicted_strikeouts_fundamentals")
            blended_ks = row.get("predicted_strikeouts")
            primary_ks = fund_ks if fund_ks is not None else blended_ks
            recommended_side = _recommended_side(primary_ks, row.get("market_line"))
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
                    "predicted_strikeouts_fundamentals": row.get("predicted_strikeouts_fundamentals"),
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

    first5_rows: list[dict[str, Any]] = []
    home_run_rows: list[dict[str, Any]] = []
    if _table_exists("predictions_player_hr"):
        home_run_rows = _fetch_daily_home_run_rows(target_date)

    if _table_exists("predictions_first5_totals") and _table_exists("games"):
        first5_frame = _safe_frame(
            """
            WITH ranked_predictions AS (
                SELECT
                    p.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY p.game_id
                        ORDER BY p.prediction_ts DESC
                    ) AS row_rank
                FROM predictions_first5_totals p
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
                p.predicted_total_fundamentals,
                p.market_total,
                p.over_probability,
                p.under_probability,
                p.edge,
                g.total_runs_first5 AS actual_total_runs
            FROM ranked_predictions p
            INNER JOIN games g
                ON g.game_id = p.game_id
               AND g.game_date = p.game_date
            WHERE p.row_rank = 1
            ORDER BY g.game_start_ts, p.game_id
            """,
            {"target_date": target_date},
        )
        for row in _frame_records(first5_frame):
            is_final = _is_final_game_status(row.get("status"))
            fundamentals_total = _to_float(row.get("predicted_total_fundamentals"))
            blended_total = _to_float(row.get("predicted_total_runs"))
            predicted_total = fundamentals_total if fundamentals_total is not None else blended_total
            recommended_side = _recommended_side(predicted_total, row.get("market_total"))
            actual_side = _actual_side(row.get("actual_total_runs"), row.get("market_total")) if is_final else None
            first5_rows.append(
                {
                    "game_id": row.get("game_id"),
                    "game_date": row.get("game_date"),
                    "game_start_ts": row.get("game_start_ts"),
                    "game_status": row.get("status"),
                    "is_final": is_final,
                    "away_team": row.get("away_team"),
                    "home_team": row.get("home_team"),
                    "predicted_total_runs": row.get("predicted_total_runs"),
                    "predicted_total_fundamentals": row.get("predicted_total_fundamentals"),
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

    final_games = sum(1 for row in totals_rows if row.get("is_final"))
    total_games = len(totals_rows)
    ai_pick_summary = _summarize_category(ai_pick_rows)
    ai_pick_summary["green_picks"] = len(ai_pick_rows)
    ai_pick_summary["live_board_fallback"] = live_green_fallback
    watchlist_summary = _summarize_category(watchlist_rows)
    watchlist_summary["live_board_supplement"] = live_watchlist_fallback
    experimental_summary = _summarize_category(experimental_rows)
    top_ev_summary = _summarize_category(top_ev_rows)
    hitter_summary = _summarize_category(hitter_rows)
    hitter_summary["review_cap"] = hitter_top_n
    hitter_summary["min_hit_probability"] = hit_min_probability
    return {
        "summary": {
            "final_games": final_games,
            "pending_games": max(total_games - final_games, 0),
            "total_games": total_games,
            "live_green_board_fallback": live_green_fallback,
            "live_watchlist_board_supplement": live_watchlist_fallback,
            "ai_picks": ai_pick_summary,
            "watchlist": watchlist_summary,
            "experimental": experimental_summary,
            "top_ev": top_ev_summary,
            "totals": _summarize_category(totals_rows),
            "hitters": hitter_summary,
            "strikeouts": _summarize_category(strikeout_rows),
            "first5": _summarize_category(first5_rows),
            "home_runs": _summarize_category(home_run_rows),
        },
        "ai_picks": ai_pick_rows,
        "watchlist": watchlist_rows,
        "experimental": experimental_rows,
        "top_ev": top_ev_rows,
        "totals": totals_rows,
        "hitters": hitter_rows,
        "strikeouts": strikeout_rows,
        "first5": first5_rows,
        "home_runs": home_run_rows,
    }


def _child_process_env() -> dict[str, str]:
    """Ensure ``python -m src....`` subprocesses resolve the project root (notably on Windows)."""
    env = os.environ.copy()
    root = str(settings.base_dir.resolve())
    sep = os.pathsep
    existing = [p.strip() for p in (env.get("PYTHONPATH") or "").split(sep) if p.strip()]
    if root not in existing:
        env["PYTHONPATH"] = root if not existing else sep.join([root, *existing])
    return env


def _format_failed_step_message(module_name: str, step: dict[str, Any]) -> str:
    rc = step.get("returncode")
    stderr = str(step.get("stderr") or "").strip()
    stdout = str(step.get("stdout") or "").strip()
    lines = [f"{module_name} exited with code {rc}"]
    detail = stderr
    if not detail and "Traceback" in stdout:
        detail = stdout
    if detail:
        tail = detail[-2500:] if len(detail) > 2500 else detail
        lines.append(tail)
    return "\n".join(lines)


def _run_module(module_name: str, *args: str) -> dict[str, Any]:
    if getattr(sys, "frozen", False):
        return _run_module_in_process(module_name, *args)

    command = [sys.executable, "-m", module_name, *args]
    timeout_sec = _pipeline_step_timeout_seconds()
    try:
        completed = subprocess.run(
            command,
            cwd=settings.base_dir,
            capture_output=True,
            text=True,
            env=_child_process_env(),
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        log.error("Pipeline step timed out: %s (limit %ss)", module_name, timeout_sec)
        out = (exc.stdout or "").strip() if isinstance(exc.stdout, str) else ""
        err = (exc.stderr or "").strip() if isinstance(exc.stderr, str) else ""
        msg = (
            f"Subprocess exceeded MLB_PIPELINE_STEP_TIMEOUT_SEC ({int(timeout_sec)}s). "
            "Increase the limit or run heavy ingestors from the CLI. "
            f"Module {module_name}."
        )
        return {
            "module": module_name,
            "command": command,
            "returncode": 124,
            "stdout": out[-8000:] if out else "",
            "stderr": "\n".join(part for part in (msg, err) if part),
        }
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
    """Write update-job history to disk. Must never raise — a failed write must not kill the pipeline thread."""
    try:
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
    except OSError as exc:
        log.warning("Failed to persist update job history to %s: %s", UPDATE_JOB_STORE_PATH, exc)
    except (TypeError, ValueError) as exc:
        log.warning("Failed to serialize update job history: %s", exc)


def _hydrate_persisted_job(payload: dict[str, Any]) -> dict[str, Any] | None:
    job_id = str(payload.get("job_id") or "").strip()
    action = payload.get("action")
    target_date = str(payload.get("target_date") or "").strip()
    created_at = str(payload.get("created_at") or "").strip()
    if not job_id or action not in UPDATE_JOB_ACTION_KEYS:
        return None
    if not target_date or not created_at:
        return None
    steps = payload.get("steps") if isinstance(payload.get("steps"), list) else []
    status = str(payload.get("status") or "failed").strip().lower()
    error = payload.get("error")
    if status in {"queued", "running"}:
        status = "failed"
        interrupted = str(payload.get("current_step") or "").strip()
        completed_recorded = int(payload.get("completed_steps") or len(steps))
        if not error:
            if interrupted:
                error = (
                    "The API process restarted or exited before this update job finished "
                    f"({completed_recorded} step(s) completed; was running {interrupted} when the process ended). "
                    "Common causes: "
                    "(1) `uvicorn --reload` — saving or touching a `.py` file restarts the server and aborts in-flight jobs; "
                    "for Retrain Models and other long tasks use `make start-app-stable` (no reload); "
                    "(2) if you start uvicorn manually with `--reload`, add `--reload-exclude \"data/**\" --reload-exclude \"db/**\"` "
                    "so pipeline writes under data/ and db/ cannot trigger a reload on some setups; "
                    "(3) the desktop app or API process was closed or crashed; "
                    "(4) multiple uvicorn workers (in-memory jobs require a single worker). "
                    "Run the pipeline again with reload disabled and one worker."
                )
            else:
                error = (
                    "The API process restarted or exited before this update job finished (no active step was recorded). "
                    "Common causes: (1) `uvicorn --reload` and a `.py` file changed; "
                    "(2) the process was closed; (3) multiple workers. "
                    "Use `make start-app-stable` for long jobs."
                )
        if interrupted and "lineups" in interrupted and "start-app-stable" not in str(error):
            error = (
                str(error)
                + "\n\nTip: If you use `make start-app`, uvicorn runs with `--reload` — saving any `.py` file "
                "restarts the API and aborts in-flight jobs. For long updates, run `make start-app-stable` "
                "(no reload), or run the first step manually: "
                "`python -m src.ingestors.lineups --target-date "
                + target_date
                + "` then retry Update Lineups & Markets."
            )
        elif interrupted and (
            interrupted.startswith("src.models.train_")
            or interrupted == "src.models.retrain_models"
            or action == "retrain_models"
        ) and "start-app-stable" not in str(error):
            error = (
                str(error)
                + "\n\nTip: Retrain Models spends a long time on early steps (e.g. `train_totals`). "
                "Run the API with `make start-app-stable` so IDE or formatter auto-save on `.py` files cannot restart mid-job."
            )
        elif interrupted and action == "refresh_everything" and "start-app-stable" not in str(error):
            error = (
                str(error)
                + "\n\nTip: Refresh Everything runs 33 steps; it starts with `src.ingestors.games` (schedule ingest), "
                "which can take 1–2+ minutes — the counter stays at 0/N until that subprocess returns. "
                "Use `make start-app-stable` (no `--reload`) so the server is not restarted mid-run. "
                "To verify the first step: from the project root, "
                f"`python -m src.ingestors.games --target-date {target_date}`."
            )
    rebuilt_sequence: list[tuple[str, list[str]]] = []
    try:
        rebuilt_sequence = _update_job_sequence(str(action), target_date)
    except Exception as exc:
        log.warning("Could not rebuild update job sequence for %s: %s", job_id, exc)
    job = {
        "job_id": job_id,
        "action": action,
        "label": _update_job_label(action),
        "target_date": target_date,
        "sequence": rebuilt_sequence,
        "status": status,
        "created_at": created_at,
        "started_at": payload.get("started_at"),
        "finished_at": payload.get("finished_at") or _utc_now_iso(),
        "current_step": None,
        "completed_steps": int(payload.get("completed_steps") or len(steps)),
        "total_steps": int(payload.get("total_steps") or len(rebuilt_sequence) or len(steps)),
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
    action_key = ""
    sequence: list[tuple[str, list[str]]] = []
    with UPDATE_JOB_LOCK:
        job = UPDATE_JOBS.get(job_id)
        if job is None:
            return
        target_date = str(job["target_date"])
        action_key = str(job.get("action") or "")
        sequence = list(job.get("sequence") or ())

    if not sequence and action_key and target_date:
        try:
            sequence = _update_job_sequence(action_key, target_date)
            with UPDATE_JOB_LOCK:
                job = UPDATE_JOBS.get(job_id)
                if job is not None:
                    job["sequence"] = sequence
                    job["total_steps"] = len(sequence)
        except Exception as exc:
            log.warning("Update job %s could not rebuild sequence: %s", job_id, exc)
            sequence = []
        else:
            _persist_update_jobs()

    if not sequence:
        log.error(
            "Update job %s has no pipeline sequence (action=%r target_date=%r)",
            job_id,
            action_key,
            target_date,
        )
        with UPDATE_JOB_LOCK:
            job = UPDATE_JOBS.get(job_id)
            if job is None:
                return
            job["status"] = "failed"
            job["finished_at"] = _utc_now_iso()
            job["current_step"] = None
            job["error"] = (
                "Internal error: empty update job sequence. "
                "Try Refresh Everything again; if this repeats, check server logs."
            )
            _trim_finished_jobs_locked()
            _persist_pipeline_run(job)
        _persist_update_jobs()
        return

    try:
        for index, (module_name, args) in enumerate(sequence, start=1):
            with UPDATE_JOB_LOCK:
                job = UPDATE_JOBS.get(job_id)
                if job is None:
                    return
                job["current_step"] = module_name
            _persist_update_jobs()
            log.info(
                "Update job %s pipeline step %s/%s: %s %s",
                job_id[:12],
                index,
                len(sequence),
                module_name,
                " ".join(args),
            )
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
                    job["error"] = _format_failed_step_message(module_name, step)
                    job["status_snapshot"] = _safe_fetch_status(target_date)
                    _trim_finished_jobs_locked()
                    _persist_pipeline_run(job)
                else:
                    _persist_pipeline_run(job)
            _persist_update_jobs()
            if step["returncode"] != 0:
                return
    except Exception as exc:
        log.exception("Update job %s crashed", job_id)
        detail = traceback.format_exc()
        tail = detail[-3500:] if len(detail) > 3500 else detail
        with UPDATE_JOB_LOCK:
            job = UPDATE_JOBS.get(job_id)
            if job is None:
                return
            job["status"] = "failed"
            job["finished_at"] = _utc_now_iso()
            job["current_step"] = None
            job["error"] = (
                "Pipeline thread crashed before finishing a step (no subprocess result recorded). "
                f"{type(exc).__name__}: {exc}\n{tail}"
            )
            job["status_snapshot"] = _safe_fetch_status(target_date)
            _trim_finished_jobs_locked()
            _persist_pipeline_run(job)
        _persist_update_jobs()
        return

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


def _json_response(payload: dict[str, Any], status_code: int = 200) -> JSONResponse:
    return JSONResponse(content=jsonable_encoder(payload), status_code=status_code)


def _green_pick_board_limit(_target_date: date) -> int | None:
    """Return ``None`` so DB-backed recommendation fallbacks are not sliced to a fixed N."""
    return None


__all__ = tuple(
    k
    for k in globals().keys()
    if k != "__all__" and not (k.startswith("__") and k.endswith("__"))
)

_load_persisted_update_jobs()
