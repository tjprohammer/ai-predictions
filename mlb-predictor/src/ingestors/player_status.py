from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from typing import Any

from src.ingestors.common import compute_payload_hash, iter_schedule_games, record_ingest_event, statsapi_get, team_dimension_row
from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.db import upsert_rows
from src.utils.logging import get_logger


log = get_logger(__name__)

ROSTER_TYPE = "40Man"
INACTIVE_STATUS_KEYWORDS = (
    "inactive",
    "bereavement",
    "paternity",
    "restricted",
    "suspended",
    "temporarily inactive",
)


def _official_game_date(game: dict[str, Any]) -> date:
    official_date = game.get("officialDate")
    if official_date:
        return date.fromisoformat(str(official_date))
    game_date = str(game.get("gameDate") or "")[:10]
    return date.fromisoformat(game_date)


def _status_flags(
    status_code: Any,
    status_description: Any,
    note: Any,
) -> dict[str, Any]:
    code = str(status_code or "").strip().upper()
    description = str(status_description or "").strip()
    description_lower = description.lower()
    note_text = str(note or "").strip()

    is_active_roster = code == "A" or description_lower == "active"
    is_injured = "injured" in description_lower
    if is_active_roster:
        availability_bucket = "active"
    elif is_injured:
        availability_bucket = "injured"
    elif any(keyword in description_lower for keyword in INACTIVE_STATUS_KEYWORDS) or code:
        availability_bucket = "inactive"
    else:
        availability_bucket = "unknown"

    return {
        "availability_bucket": availability_bucket,
        "is_active_roster": is_active_roster,
        "is_available": availability_bucket == "active",
        "is_injured": is_injured,
        "status_note": note_text or None,
    }


def _fetch_team_player_status_rows(
    team_id: int,
    team_abbr: str,
    target_date: date,
    snapshot_ts: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = statsapi_get(
        f"/api/v1/teams/{team_id}/roster",
        params={"date": target_date.isoformat(), "rosterType": ROSTER_TYPE},
    )
    roster = payload.get("roster") or []
    source_url = (
        f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
        f"?date={target_date.isoformat()}&rosterType={ROSTER_TYPE}"
    )

    status_rows: list[dict[str, Any]] = []
    player_rows: list[dict[str, Any]] = []
    for entry in roster:
        person = entry.get("person") or {}
        player_id = person.get("id")
        if player_id is None:
            continue
        status = entry.get("status") or {}
        flags = _status_flags(status.get("code"), status.get("description"), entry.get("note"))
        player_name = person.get("fullName") or str(player_id)
        position = (entry.get("position") or {}).get("abbreviation")

        player_rows.append(
            {
                "player_id": int(player_id),
                "full_name": player_name,
                "position": position,
                "team_abbr": team_abbr,
                "active": flags["is_available"],
            }
        )
        status_rows.append(
            {
                "game_date": target_date,
                "team_id": int(team_id),
                "team": team_abbr,
                "player_id": int(player_id),
                "player_name": player_name,
                "position": position,
                "jersey_number": entry.get("jerseyNumber"),
                "roster_type": ROSTER_TYPE,
                "status_code": status.get("code"),
                "status_description": status.get("description"),
                "status_note": flags["status_note"],
                "availability_bucket": flags["availability_bucket"],
                "is_active_roster": flags["is_active_roster"],
                "is_available": flags["is_available"],
                "is_injured": flags["is_injured"],
                "source_name": "mlb_statsapi_roster",
                "source_url": source_url,
                "snapshot_ts": snapshot_ts,
                "raw_payload": entry,
            }
        )
    return status_rows, player_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest MLB roster status snapshots for slate teams")
    add_date_range_args(parser)
    args = parser.parse_args()
    start_date, end_date = resolve_date_range(args)

    games = iter_schedule_games(start_date.isoformat(), end_date.isoformat())
    if not games:
        log.info("No schedule rows returned for %s to %s", start_date, end_date)
        return 0

    team_keys = sorted(
        {
            (_official_game_date(game), int(game["teams"][side]["team"]["id"]))
            for game in games
            for side in ("away", "home")
            if game.get("teams")
        }
    )
    snapshot_ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    all_status_rows: list[dict[str, Any]] = []
    all_player_rows: list[dict[str, Any]] = []

    for target_date, team_id in team_keys:
        team_abbr = team_dimension_row(team_id)["team_abbr"]
        try:
            status_rows, player_rows = _fetch_team_player_status_rows(
                team_id,
                team_abbr,
                target_date,
                snapshot_ts,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Roster status fetch failed for %s on %s: %s",
                team_abbr,
                target_date,
                exc,
            )
            continue
        all_status_rows.extend(status_rows)
        all_player_rows.extend(player_rows)

    if not all_status_rows:
        log.info("No roster status rows were returned for %s to %s", start_date, end_date)
        return 0

    upsert_rows("dim_players", all_player_rows, ["player_id"])
    inserted = upsert_rows(
        "player_status_daily",
        all_status_rows,
        ["game_date", "team", "player_id", "roster_type", "source_name", "snapshot_ts"],
    )
    log.info("Imported %s player status rows for %s to %s", inserted, start_date, end_date)
    record_ingest_event(
        source_name="mlb_statsapi_roster",
        ingestor_module="src.ingestors.player_status",
        target_date=start_date.isoformat(),
        row_count=inserted,
        payload_hash=compute_payload_hash(all_status_rows),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())