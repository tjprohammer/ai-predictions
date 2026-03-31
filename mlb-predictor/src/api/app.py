from __future__ import annotations

import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from src.utils.db import query_df
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)
settings = get_settings()
STATIC_DIR = Path(__file__).with_name("static")
INDEX_FILE = STATIC_DIR / "index.html"
HOT_HITTERS_FILE = STATIC_DIR / "hot-hitters.html"
FAVICON_FILE = STATIC_DIR / "favicon.svg"

app = FastAPI(title="MLB Predictor", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PipelineRunRequest(BaseModel):
    target_date: date = Field(default_factory=date.today)
    refresh_aggregates: bool = True
    rebuild_features: bool = True


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
    frame = _safe_frame(
        """
        SELECT 1 AS present
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = :table_name
        LIMIT 1
        """,
        {"table_name": table_name},
    )
    return not frame.empty


def _artifact_ready(lane: str) -> bool:
    return any((settings.model_dir / lane).glob("*.pkl"))


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
        + ((int(player.get("streak_len_capped") or 0)) * 0.04)
    )


def _classify_hitter_form(player: dict[str, Any]) -> dict[str, Any]:
    hit_rate_7 = _to_float(player.get("hit_rate_7"))
    hit_rate_30 = _to_float(player.get("hit_rate_30"))
    xwoba_14 = _to_float(player.get("xwoba_14"))
    hard_hit_pct_14 = _to_float(player.get("hard_hit_pct_14"))
    streak = int(player.get("streak_len_capped") or 0)
    hit_delta = None if hit_rate_7 is None or hit_rate_30 is None else hit_rate_7 - hit_rate_30

    evidence = [
        f"7G {_format_rate(hit_rate_7)} vs 30G {_format_rate(hit_rate_30)}"
        + (f" ({hit_delta * 100:+.0f} pts)" if hit_delta is not None else ""),
        f"xwOBA14 {_format_metric(xwoba_14)}",
        f"HH14 {_format_rate(hard_hit_pct_14)}",
        f"Streak {streak}",
    ]

    hot_reasons: list[str] = []
    cold_reasons: list[str] = []
    if hit_delta is not None and hit_delta >= 0.12:
        hot_reasons.append(f"7G hit rate {_format_rate(hit_rate_7)} is {hit_delta * 100:+.0f} points above the 30G baseline")
    if xwoba_14 is not None and xwoba_14 >= 0.38:
        hot_reasons.append(f"xwOBA over the last 14 games is {_format_metric(xwoba_14)}")
    if streak >= 4:
        hot_reasons.append(f"Riding a {streak}-game hit streak")
    if hard_hit_pct_14 is not None and hard_hit_pct_14 >= 0.45:
        hot_reasons.append(f"Hard-hit rate over the last 14 games is {_format_rate(hard_hit_pct_14)}")

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

    frame = _safe_frame(
        """
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
        ORDER BY game_id, team, lineup_slot ASC NULLS LAST, player_id
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

    frame = _safe_frame(
        """
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
            CASE
                WHEN SUM(b.at_bats) = 0 THEN NULL
                ELSE SUM(b.hits)::DOUBLE PRECISION / SUM(b.at_bats)
            END AS split_batting_avg,
            AVG(b.xwoba) AS split_xwoba,
            AVG(b.hard_hit_pct) AS split_hard_hit_pct
        FROM player_game_batting b
        INNER JOIN ranked_starters rs
            ON rs.game_id = b.game_id
           AND rs.team = b.opponent
           AND rs.row_rank = 1
        WHERE b.game_date < :target_date
          AND b.player_id = ANY(:player_ids)
          AND rs.throws IN ('R', 'L')
        GROUP BY b.player_id, rs.throws
        """,
        {"target_date": target_date, "player_ids": list(player_ids)},
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
                "trailing_success_rate": _weighted_average("success_rate", graded_weight),
                "trailing_avg_absolute_error": _weighted_average("avg_absolute_error", graded_weight),
                "trailing_brier_score": _weighted_average("brier_score", graded_weight),
                "trailing_beat_market_rate": _weighted_average("beat_market_rate", graded_weight),
            }
        )
    latest_score_date = max(record["latest_score_date"] for record in rows if record.get("latest_score_date")) if rows else None
    return {"latest_score_date": latest_score_date, "window_days": window_days, "rows": rows}


def _fetch_totals_predictions(target_date: date) -> list[dict[str, Any]]:
    if not _table_exists("predictions_totals"):
        return []
    frame = _safe_frame(
        """
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
            r.edge
        FROM ranked r
        LEFT JOIN games g ON g.game_id = r.game_id
        WHERE r.row_rank = 1
        ORDER BY g.game_start_ts NULLS LAST, away_team, home_team
        """,
        {"target_date": target_date},
    )
    return _frame_records(frame)


def _fetch_hit_predictions(target_date: date, limit: int, min_probability: float, confirmed_only: bool) -> list[dict[str, Any]]:
    if not _table_exists("predictions_player_hits"):
        return []
    frame = _safe_frame(
        """
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
            CAST(NULLIF(f.feature_payload ->> 'is_confirmed_lineup', '') AS BOOLEAN) AS is_confirmed_lineup,
            CAST(NULLIF(f.feature_payload ->> 'projected_plate_appearances', '') AS DOUBLE PRECISION) AS projected_plate_appearances,
            CAST(NULLIF(f.feature_payload ->> 'streak_len_capped', '') AS SMALLINT) AS streak_len_capped,
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
                                 CAST(NULLIF(f.feature_payload ->> 'is_confirmed_lineup', '') AS BOOLEAN) DESC NULLS LAST,
                                 CAST(NULLIF(f.feature_payload ->> 'projected_plate_appearances', '') AS DOUBLE PRECISION) DESC NULLS LAST,
                                 CAST(NULLIF(f.feature_payload ->> 'streak_len_capped', '') AS SMALLINT) DESC NULLS LAST,
                                 player_name
        LIMIT :limit
        """,
        {
            "target_date": target_date,
            "limit": limit,
            "min_probability": min_probability,
        },
    )
    if confirmed_only and not frame.empty and "is_confirmed_lineup" in frame.columns:
        frame = frame[frame["is_confirmed_lineup"] == True].copy()
    return _frame_records(frame)


def _fetch_game_board(
    target_date: date,
    hit_limit_per_team: int,
    min_probability: float,
    confirmed_only: bool,
) -> list[dict[str, Any]]:
    if not _table_exists("games"):
        return []

    games_frame = _safe_frame(
        """
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
            CAST(f.feature_payload ->> 'away_runs_rate_blended' AS DOUBLE PRECISION) AS away_expected_runs,
            CAST(f.feature_payload ->> 'home_runs_rate_blended' AS DOUBLE PRECISION) AS home_expected_runs,
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
            CAST(f.feature_payload ->> 'line_movement' AS DOUBLE PRECISION) AS line_movement
        FROM games g
        LEFT JOIN dim_venues v ON v.venue_id = g.venue_id
        LEFT JOIN ranked_predictions p ON p.game_id = g.game_id AND p.row_rank = 1
        LEFT JOIN ranked_features f ON f.game_id = g.game_id AND f.row_rank = 1
        WHERE g.game_date = :target_date
        ORDER BY g.game_start_ts NULLS LAST, g.away_team, g.home_team
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
        """
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
                CASE
                    WHEN SUM(recent.at_bats) = 0 THEN NULL
                    ELSE SUM(recent.hits)::DOUBLE PRECISION / SUM(recent.at_bats)
                END AS batting_avg_last7
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
                CASE
                    WHEN SUM(b.at_bats) = 0 THEN NULL
                    ELSE SUM(b.hits)::DOUBLE PRECISION / SUM(b.at_bats)
                END AS batting_avg_season
            FROM player_game_batting b
            INNER JOIN selected_players sp ON sp.player_id = b.player_id
            WHERE b.game_date < :target_date
              AND EXTRACT(YEAR FROM b.game_date) = EXTRACT(YEAR FROM CAST(:target_date AS DATE))
            GROUP BY b.player_id
        ),
        joined AS (
            SELECT
                p.game_id,
                p.game_date,
                p.player_id,
                COALESCE(f.feature_payload ->> 'player_name', dp.full_name, CAST(p.player_id AS TEXT)) AS player_name,
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
                CAST(NULLIF(f.feature_payload ->> 'lineup_slot', '') AS SMALLINT) AS lineup_slot,
                CAST(NULLIF(f.feature_payload ->> 'is_confirmed_lineup', '') AS BOOLEAN) AS is_confirmed_lineup,
                CAST(NULLIF(f.feature_payload ->> 'projected_plate_appearances', '') AS DOUBLE PRECISION) AS projected_plate_appearances,
                CAST(NULLIF(f.feature_payload ->> 'streak_len_capped', '') AS SMALLINT) AS streak_len_capped,
                p.predicted_hit_probability,
                p.fair_price,
                p.market_price,
                p.edge,
                CAST(f.feature_payload ->> 'hit_rate_blended' AS DOUBLE PRECISION) AS hit_rate_blended,
                CAST(f.feature_payload ->> 'xwoba_14' AS DOUBLE PRECISION) AS xwoba_14,
                CAST(f.feature_payload ->> 'opposing_starter_xwoba' AS DOUBLE PRECISION) AS opposing_starter_xwoba,
                CAST(f.feature_payload ->> 'opposing_starter_csw' AS DOUBLE PRECISION) AS opposing_starter_csw,
                CAST(f.feature_payload ->> 'team_run_environment' AS DOUBLE PRECISION) AS team_run_environment,
                CAST(f.feature_payload ->> 'park_hr_factor' AS DOUBLE PRECISION) AS park_hr_factor,
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
                             j.projected_plate_appearances DESC NULLS LAST,
                             j.lineup_slot ASC NULLS LAST,
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
        ORDER BY game_id, team, predicted_hit_probability DESC, lineup_slot ASC NULLS LAST, player_name
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
    hit_split_map = _fetch_hitter_pitch_hand_splits(
        target_date,
        [int(hit["player_id"]) for hit in hit_records if hit.get("player_id") is not None],
    )
    lineup_handedness_by_game = _fetch_lineup_handedness_by_game(target_date)
    pitcher_k_market_map = _fetch_pitcher_strikeout_market_map(target_date)
    pitcher_k_prediction_map = _fetch_pitcher_strikeout_prediction_map(target_date)

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
                "away_expected_runs": record["away_expected_runs"],
                "home_expected_runs": record["home_expected_runs"],
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
            "starters": {
                "away": None,
                "home": None,
            },
            "lineup_handedness": lineup_handedness_by_game.get(game_id, {}),
            "hit_targets": {
                record["away_team"]: [],
                record["home_team"]: [],
            },
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

    return [games_by_id[game_id] for game_id in games_by_id]


def _fetch_full_hit_review(target_date: date) -> dict[str, Any]:
    default = {
        "total_targets": 0,
        "confirmed_targets": 0,
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
    frame = _safe_frame(
        """
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
        ORDER BY game_date DESC, game_start_ts DESC NULLS LAST
        LIMIT 1
        """,
        {"team": team, "target_date": target_date},
    )
    if frame.empty:
        return None
    return _frame_records(frame)[0]


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
    return {**default, **record}


def _fetch_game_detail(game_id: int, target_date: date) -> dict[str, Any] | None:
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
            CAST(f.feature_payload ->> 'away_runs_rate_blended' AS DOUBLE PRECISION) AS away_expected_runs,
            CAST(f.feature_payload ->> 'home_runs_rate_blended' AS DOUBLE PRECISION) AS home_expected_runs,
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
            CAST(f.feature_payload ->> 'line_movement' AS DOUBLE PRECISION) AS line_movement
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
        lineup_frame = _safe_frame(
            """
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
                    CASE
                        WHEN SUM(at_bats) = 0 THEN NULL
                        ELSE SUM(hits)::DOUBLE PRECISION / SUM(at_bats)
                    END AS batting_avg_last7,
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
                    CASE
                        WHEN SUM(b.at_bats) = 0 THEN NULL
                        ELSE SUM(b.hits)::DOUBLE PRECISION / SUM(b.at_bats)
                    END AS batting_avg_season
                FROM player_game_batting b
                INNER JOIN selected_players sp ON sp.player_id = b.player_id
                WHERE b.game_date < :target_date
                  AND EXTRACT(YEAR FROM b.game_date) = EXTRACT(YEAR FROM CAST(:target_date AS DATE))
                GROUP BY b.player_id
            )
            SELECT
                f.player_id,
                COALESCE(f.feature_payload ->> 'player_name', dp.full_name, CAST(f.player_id AS TEXT)) AS player_name,
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
                CAST(NULLIF(f.feature_payload ->> 'lineup_slot', '') AS SMALLINT) AS lineup_slot,
                CAST(NULLIF(f.feature_payload ->> 'is_confirmed_lineup', '') AS BOOLEAN) AS is_confirmed_lineup,
                CAST(NULLIF(f.feature_payload ->> 'projected_plate_appearances', '') AS DOUBLE PRECISION) AS projected_plate_appearances,
                CAST(NULLIF(f.feature_payload ->> 'streak_len_capped', '') AS SMALLINT) AS streak_len_capped,
                CAST(NULLIF(f.feature_payload ->> 'hit_rate_7', '') AS DOUBLE PRECISION) AS hit_rate_7,
                CAST(NULLIF(f.feature_payload ->> 'hit_rate_14', '') AS DOUBLE PRECISION) AS hit_rate_14,
                CAST(NULLIF(f.feature_payload ->> 'hit_rate_30', '') AS DOUBLE PRECISION) AS hit_rate_30,
                CAST(NULLIF(f.feature_payload ->> 'hit_rate_blended', '') AS DOUBLE PRECISION) AS hit_rate_blended,
                CAST(NULLIF(f.feature_payload ->> 'xba_14', '') AS DOUBLE PRECISION) AS xba_14,
                CAST(NULLIF(f.feature_payload ->> 'xwoba_14', '') AS DOUBLE PRECISION) AS xwoba_14,
                CAST(NULLIF(f.feature_payload ->> 'hard_hit_pct_14', '') AS DOUBLE PRECISION) AS hard_hit_pct_14,
                CAST(NULLIF(f.feature_payload ->> 'k_pct_14', '') AS DOUBLE PRECISION) AS k_pct_14,
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
                recent_batting.hard_hit_pct_last7
            FROM ranked_features f
            LEFT JOIN ranked_predictions p
                ON p.game_id = f.game_id
               AND p.player_id = f.player_id
               AND p.row_rank = 1
            LEFT JOIN dim_players dp ON dp.player_id = f.player_id
            LEFT JOIN games g ON g.game_id = f.game_id
                LEFT JOIN season_batting ON season_batting.player_id = f.player_id
            LEFT JOIN player_game_batting actual
                ON actual.game_id = f.game_id
               AND actual.player_id = f.player_id
            LEFT JOIN recent_batting ON recent_batting.player_id = f.player_id
            WHERE f.row_rank = 1
            ORDER BY
                CASE
                    WHEN COALESCE(f.team, dp.team_abbr) = g.away_team THEN 0
                    WHEN COALESCE(f.team, dp.team_abbr) = g.home_team THEN 1
                    ELSE 2
                END,
                CAST(NULLIF(f.feature_payload ->> 'lineup_slot', '') AS SMALLINT) ASC NULLS LAST,
                player_name
            """,
            {"game_id": game_id, "target_date": target_date},
        )
        lineup_records = _frame_records(lineup_frame)
        lineup_split_map = _fetch_hitter_pitch_hand_splits(
            target_date,
            [int(player["player_id"]) for player in lineup_records if player.get("player_id") is not None],
        )
    else:
        lineup_split_map = {}

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
            "over_probability": game["over_probability"],
            "under_probability": game["under_probability"],
            "edge": game["edge"],
            "away_expected_runs": game["away_expected_runs"],
            "home_expected_runs": game["home_expected_runs"],
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
        side = "away" if player["team"] == detail["away_team"] else "home"
        opposing_starter = detail["starters"]["home"] if side == "away" else detail["starters"]["away"]
        _attach_hitter_matchup_context(
            player,
            opposing_starter["throws"] if opposing_starter else None,
            lineup_split_map,
        )
        detail["teams"][side]["lineup"].append(player)

    for side in ("away", "home"):
        team_lineup = detail["teams"][side]["lineup"]
        detail["teams"][side]["confirmed_lineup"] = any(player.get("is_confirmed_lineup") for player in team_lineup)
        detail["teams"][side]["lineup_handedness"] = _summarize_lineup_handedness(
            team_lineup,
            confirmed_key="is_confirmed_lineup",
        )

    detail["lineup_handedness"] = {
        detail["away_team"]: detail["teams"]["away"]["lineup_handedness"],
        detail["home_team"]: detail["teams"]["home"]["lineup_handedness"],
    }

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


def _fetch_hot_hitters(target_date: date, min_probability: float, confirmed_only: bool, limit: int) -> dict[str, Any]:
    empty = {
        "rows": [],
        "summary": {
            "count": 0,
            "total_hot_count": 0,
            "confirmed_count": 0,
            "games_count": 0,
            "average_hit_probability": None,
            "latest_prediction_ts": None,
        },
    }
    if not _table_exists("player_features_hits") or not _table_exists("games"):
        return empty

    frame = _safe_frame(
        """
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
                CASE
                    WHEN SUM(at_bats) = 0 THEN NULL
                    ELSE SUM(hits)::DOUBLE PRECISION / SUM(at_bats)
                END AS batting_avg_last7,
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
                CASE
                    WHEN SUM(b.at_bats) = 0 THEN NULL
                    ELSE SUM(b.hits)::DOUBLE PRECISION / SUM(b.at_bats)
                END AS batting_avg_season
            FROM player_game_batting b
            INNER JOIN selected_players sp ON sp.player_id = b.player_id
            WHERE b.game_date < :target_date
              AND EXTRACT(YEAR FROM b.game_date) = EXTRACT(YEAR FROM CAST(:target_date AS DATE))
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
            COALESCE(f.feature_payload ->> 'player_name', dp.full_name, CAST(f.player_id AS TEXT)) AS player_name,
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
            CAST(NULLIF(f.feature_payload ->> 'lineup_slot', '') AS SMALLINT) AS lineup_slot,
            CAST(NULLIF(f.feature_payload ->> 'is_confirmed_lineup', '') AS BOOLEAN) AS is_confirmed_lineup,
            CAST(NULLIF(f.feature_payload ->> 'projected_plate_appearances', '') AS DOUBLE PRECISION) AS projected_plate_appearances,
            CAST(NULLIF(f.feature_payload ->> 'streak_len_capped', '') AS SMALLINT) AS streak_len_capped,
            CAST(NULLIF(f.feature_payload ->> 'hit_rate_7', '') AS DOUBLE PRECISION) AS hit_rate_7,
            CAST(NULLIF(f.feature_payload ->> 'hit_rate_14', '') AS DOUBLE PRECISION) AS hit_rate_14,
            CAST(NULLIF(f.feature_payload ->> 'hit_rate_30', '') AS DOUBLE PRECISION) AS hit_rate_30,
            CAST(NULLIF(f.feature_payload ->> 'hit_rate_blended', '') AS DOUBLE PRECISION) AS hit_rate_blended,
            CAST(NULLIF(f.feature_payload ->> 'xba_14', '') AS DOUBLE PRECISION) AS xba_14,
            CAST(NULLIF(f.feature_payload ->> 'xwoba_14', '') AS DOUBLE PRECISION) AS xwoba_14,
            CAST(NULLIF(f.feature_payload ->> 'hard_hit_pct_14', '') AS DOUBLE PRECISION) AS hard_hit_pct_14,
            CAST(NULLIF(f.feature_payload ->> 'k_pct_14', '') AS DOUBLE PRECISION) AS k_pct_14,
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
        ORDER BY g.game_start_ts NULLS LAST, team, lineup_slot ASC NULLS LAST, player_name
        """,
        {"target_date": target_date},
    )
    if frame.empty:
        return empty

    records = _frame_records(frame)
    if confirmed_only:
        records = [record for record in records if record.get("is_confirmed_lineup")]
    if not records:
        return empty

    hit_split_map = _fetch_hitter_pitch_hand_splits(
        target_date,
        [int(record["player_id"]) for record in records if record.get("player_id") is not None],
    )

    all_hot_rows: list[dict[str, Any]] = []
    for record in records:
        form = _classify_hitter_form(record)
        if form["label"] != "Hot":
            continue
        actual_meta = _build_hit_actual_meta(record.get("actual_hits"), _is_final_game_status(record.get("game_status")))
        enriched = _attach_hitter_matchup_context(
            {
                **record,
                **actual_meta,
                "form": form,
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
            "games_count": len({int(row["game_id"]) for row in hot_rows if row.get("game_id") is not None}),
            "average_hit_probability": round(sum(valid_probabilities) / len(valid_probabilities), 4) if valid_probabilities else None,
            "latest_prediction_ts": max((row.get("prediction_ts") for row in hot_rows if row.get("prediction_ts") is not None), default=None),
        },
    }


def _run_module(module_name: str, *args: str) -> dict[str, Any]:
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


def _json_response(payload: dict[str, Any], status_code: int = 200) -> JSONResponse:
    return JSONResponse(content=jsonable_encoder(payload), status_code=status_code)


@app.get("/health")
@app.get("/api/health")
def health() -> JSONResponse:
    return _json_response(_fetch_status(date.today()))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_FILE)


@app.get("/favicon.ico", include_in_schema=False)
@app.get("/favicon.svg", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(FAVICON_FILE, media_type="image/svg+xml")


@app.get("/hot-hittes")
@app.get("/hot-hittes/")
@app.get("/hot-hitters/")
@app.get("/hot-hitters")
def hot_hitters_page() -> FileResponse:
    return FileResponse(HOT_HITTERS_FILE)


@app.get("/api/status")
def status(target_date: date = Query(default_factory=date.today)) -> JSONResponse:
    return _json_response(_fetch_status(target_date))


@app.get("/api/predictions/totals")
def totals_predictions(target_date: date = Query(default_factory=date.today)) -> JSONResponse:
    rows = _fetch_totals_predictions(target_date)
    return _json_response({"target_date": target_date.isoformat(), "rows": rows})


@app.get("/api/predictions/hits")
def hit_predictions(
    target_date: date = Query(default_factory=date.today),
    limit: int = Query(default=40, ge=1, le=200),
    min_probability: float = Query(default=0.0, ge=0.0, le=1.0),
    confirmed_only: bool = Query(default=False),
) -> JSONResponse:
    rows = _fetch_hit_predictions(target_date, limit, min_probability, confirmed_only)
    return _json_response({"target_date": target_date.isoformat(), "rows": rows})


@app.get("/api/hot-hitters")
def hot_hitters(
    target_date: date = Query(default_factory=date.today),
    limit: int = Query(default=60, ge=1, le=200),
    min_probability: float = Query(default=0.35, ge=0.0, le=1.0),
    confirmed_only: bool = Query(default=False),
) -> JSONResponse:
    payload = _fetch_hot_hitters(target_date, min_probability, confirmed_only, limit)
    return _json_response({"target_date": target_date.isoformat(), **payload})


@app.get("/api/model-scorecards")
def model_scorecards(target_date: date = Query(default_factory=date.today), window_days: int = Query(default=14, ge=1, le=60)) -> JSONResponse:
    return _json_response({"target_date": target_date.isoformat(), **_fetch_model_scorecards(target_date, window_days)})


@app.get("/api/trends/players/{player_id}")
def player_trend(player_id: int, target_date: date = Query(default_factory=date.today), limit: int = Query(default=10, ge=1, le=30)) -> JSONResponse:
    return _json_response({"target_date": target_date.isoformat(), "player_id": player_id, "rows": _fetch_player_trend(player_id, target_date, limit)})


@app.get("/api/trends/pitchers/{pitcher_id}")
def pitcher_trend(pitcher_id: int, target_date: date = Query(default_factory=date.today), limit: int = Query(default=10, ge=1, le=30)) -> JSONResponse:
    return _json_response({"target_date": target_date.isoformat(), "pitcher_id": pitcher_id, "rows": _fetch_pitcher_trend(pitcher_id, target_date, limit)})


@app.get("/api/games/board")
def games_board(
    target_date: date = Query(default_factory=date.today),
    hit_limit_per_team: int = Query(default=4, ge=1, le=9),
    min_probability: float = Query(default=0.0, ge=0.0, le=1.0),
    confirmed_only: bool = Query(default=False),
) -> JSONResponse:
    rows = _fetch_game_board(target_date, hit_limit_per_team, min_probability, confirmed_only)
    return _json_response(
        {
            "target_date": target_date.isoformat(),
            "summary": _summarize_board_rows(rows, target_date),
            "games": rows,
        }
    )


@app.get("/api/games/{game_id}/detail")
def game_detail(game_id: int, target_date: date = Query(default_factory=date.today)) -> JSONResponse:
    payload = _fetch_game_detail(game_id, target_date)
    if payload is None:
        return _json_response({"target_date": target_date.isoformat(), "game": None}, status_code=404)
    return _json_response({"target_date": target_date.isoformat(), "game": payload})


@app.post("/api/pipeline/run")
def run_pipeline(request: PipelineRunRequest) -> JSONResponse:
    target_date = request.target_date.isoformat()
    steps: list[dict[str, Any]] = []
    if request.refresh_aggregates:
        steps.append(_run_module("src.transforms.offense_daily"))
        if steps[-1]["returncode"] != 0:
            return _json_response({"ok": False, "steps": steps}, status_code=500)
        steps.append(_run_module("src.transforms.bullpens_daily"))
        if steps[-1]["returncode"] != 0:
            return _json_response({"ok": False, "steps": steps}, status_code=500)
    if request.rebuild_features:
        steps.append(_run_module("src.features.totals_builder", "--target-date", target_date))
        if steps[-1]["returncode"] != 0:
            return _json_response({"ok": False, "steps": steps}, status_code=500)
        steps.append(_run_module("src.features.hits_builder", "--target-date", target_date))
        if steps[-1]["returncode"] != 0:
            return _json_response({"ok": False, "steps": steps}, status_code=500)
        steps.append(_run_module("src.features.strikeouts_builder", "--target-date", target_date))
        if steps[-1]["returncode"] != 0:
            return _json_response({"ok": False, "steps": steps}, status_code=500)
    steps.append(_run_module("src.models.predict_totals", "--target-date", target_date))
    if steps[-1]["returncode"] != 0:
        return _json_response({"ok": False, "steps": steps}, status_code=500)
    steps.append(_run_module("src.models.predict_hits", "--target-date", target_date))
    if steps[-1]["returncode"] != 0:
        return _json_response({"ok": False, "steps": steps}, status_code=500)
    steps.append(_run_module("src.models.predict_strikeouts", "--target-date", target_date))
    if steps[-1]["returncode"] != 0:
        return _json_response({"ok": False, "steps": steps}, status_code=500)
    steps.append(_run_module("src.transforms.product_surfaces", "--target-date", target_date))
    if steps[-1]["returncode"] != 0:
        return _json_response({"ok": False, "steps": steps}, status_code=500)
    return _json_response(
        {
            "ok": True,
            "target_date": target_date,
            "steps": steps,
            "status": _fetch_status(request.target_date),
        }
    )