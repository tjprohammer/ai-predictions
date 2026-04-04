from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import requests

from src.utils.db import table_exists, upsert_rows
from src.utils.logging import get_logger


BASE_URL = "https://statsapi.mlb.com"
TIMEOUT_SECONDS = 30
_audit_log = get_logger(__name__)


def statsapi_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = requests.get(f"{BASE_URL}{path}", params=params, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
    except Exception as exc:
        record_source_health(
            source_name="mlb_statsapi",
            is_available=False,
            response_time_ms=int((time.perf_counter() - started) * 1000),
            error_message=str(exc),
        )
        raise
    record_source_health(
        source_name="mlb_statsapi",
        is_available=True,
        response_time_ms=int((time.perf_counter() - started) * 1000),
    )
    return response.json()


def iter_schedule_games(start_date: str, end_date: str) -> list[dict[str, Any]]:
    payload = statsapi_get(
        "/api/v1/schedule",
        params={
            "sportId": 1,
            "startDate": start_date,
            "endDate": end_date,
            "hydrate": "probablePitcher,linescore,team,venue",
        },
    )
    games: list[dict[str, Any]] = []
    for date_row in payload.get("dates", []):
        games.extend(date_row.get("games", []))
    return games


@lru_cache(maxsize=64)
def get_team_details(team_id: int) -> dict[str, Any]:
    payload = statsapi_get(f"/api/v1/teams/{team_id}")
    return payload.get("teams", [{}])[0]


@lru_cache(maxsize=512)
def get_person_details(player_id: int) -> dict[str, Any]:
    payload = statsapi_get(f"/api/v1/people/{player_id}")
    return payload.get("people", [{}])[0]


@lru_cache(maxsize=128)
def get_venue_details(venue_id: int) -> dict[str, Any]:
    payload = statsapi_get(
        f"/api/v1/venues/{venue_id}",
        params={"hydrate": "location,timezone,fieldInfo"},
    )
    return payload.get("venues", [{}])[0]


def team_dimension_row(team_id: int) -> dict[str, Any]:
    team = get_team_details(team_id)
    league = (team.get("league") or {}).get("abbreviation")
    division = (team.get("division") or {}).get("name", "")
    return {
        "team_abbr": team.get("abbreviation") or team.get("fileCode", "")[:3].upper(),
        "team_name": team.get("name") or team.get("teamName") or str(team_id),
        "league": league,
        "division": division.replace(" Division", "") or None,
    }


def player_dimension_row(
    player_id: int,
    *,
    full_name_override: str | None = None,
    team_abbr_override: str | None = None,
    position_override: str | None = None,
) -> dict[str, Any]:
    person = get_person_details(player_id)
    position = (person.get("primaryPosition") or {}).get("abbreviation")
    team = (person.get("currentTeam") or {}).get("id")
    team_abbr = team_dimension_row(team)["team_abbr"] if team else None
    return {
        "player_id": person.get("id", player_id),
        "full_name": full_name_override or person.get("fullName") or str(player_id),
        "first_name": person.get("firstName"),
        "last_name": person.get("lastName"),
        "bats": (person.get("batSide") or {}).get("code"),
        "throws": (person.get("pitchHand") or {}).get("code"),
        "position": position_override or position,
        "team_abbr": team_abbr_override or team_abbr,
        "active": person.get("active", True),
    }


def venue_dimension_row(venue_id: int) -> dict[str, Any]:
    venue = get_venue_details(venue_id)
    location = venue.get("location") or {}
    coords = location.get("defaultCoordinates") or {}
    timezone = venue.get("timeZone") or {}
    field_info = venue.get("fieldInfo") or {}
    return {
        "venue_id": venue.get("id", venue_id),
        "venue_name": venue.get("name") or str(venue_id),
        "city": location.get("city"),
        "state": location.get("stateAbbrev") or location.get("state"),
        "latitude": coords.get("latitude"),
        "longitude": coords.get("longitude"),
        "elevation_ft": location.get("elevation"),
        "roof_type": field_info.get("roofType"),
        "timezone_name": timezone.get("id"),
    }


def compute_payload_hash(rows: list[dict[str, Any]]) -> str:
    """Deterministic hash of the row payload for change detection."""
    canonical = json.dumps(rows, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def record_ingest_event(
    *,
    source_name: str,
    ingestor_module: str,
    target_date: str,
    row_count: int,
    game_id: int | None = None,
    parse_status: str = "ok",
    validation_status: str = "ok",
    payload_hash: str | None = None,
    warning_flags: str | None = None,
    error_message: str | None = None,
    duration_seconds: float | None = None,
) -> None:
    """Write one audit row to raw_ingest_events. Silently skips if table is missing."""
    try:
        if not table_exists("raw_ingest_events"):
            return
        upsert_rows(
            "raw_ingest_events",
            [
                {
                    "source_name": source_name,
                    "ingestor_module": ingestor_module,
                    "target_date": str(target_date),
                    "game_id": game_id,
                    "row_count": row_count,
                    "parse_status": parse_status,
                    "validation_status": validation_status,
                    "payload_hash": payload_hash,
                    "warning_flags": warning_flags,
                    "error_message": error_message,
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                    "duration_seconds": round(duration_seconds, 2) if duration_seconds else None,
                }
            ],
            ["event_id"],
        )
    except Exception as exc:
        _audit_log.warning("Failed to record ingest event for %s: %s", ingestor_module, exc)


def record_source_health(
    *,
    source_name: str,
    is_available: bool,
    response_time_ms: int | None = None,
    error_message: str | None = None,
) -> None:
    """Append one source health record if the table exists."""
    try:
        if not table_exists("source_health"):
            return
        upsert_rows(
            "source_health",
            [
                {
                    "source_name": source_name,
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                    "is_available": is_available,
                    "response_time_ms": response_time_ms,
                    "error_message": error_message,
                }
            ],
            ["health_id"],
        )
    except Exception as exc:
        _audit_log.warning("Failed to record source health for %s: %s", source_name, exc)