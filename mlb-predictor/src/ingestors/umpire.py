"""Ingest home plate umpire assignments for MLB games via the MLB Stats API.

Endpoint: GET /api/v1/schedule?sportId=1&date=YYYY-MM-DD&hydrate=officials

The ingestor stores (game_id, game_date, umpire_name, umpire_id) for the home
plate umpire of each game. Historical K-rate and run-value adjustments are
computed at feature-build time in strikeouts_builder / totals_builder by
joining this table against pitcher_starts and games.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone

import requests

from src.ingestors.common import record_ingest_event
from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.db import query_df, table_exists, upsert_rows
from src.utils.logging import get_logger


log = get_logger(__name__)

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
_FIELDS = (
    "dates,games,gamePk,officials,official,id,fullName,officialType"
)


def _fetch_assignments_for_date(target_date: date) -> list[dict]:
    """Return a list of home-plate umpire rows for every game on *target_date*."""
    try:
        resp = requests.get(
            MLB_SCHEDULE_URL,
            params={
                "sportId": 1,
                "date": target_date.isoformat(),
                "hydrate": "officials",
                "fields": _FIELDS,
            },
            timeout=20,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("Umpire fetch failed for %s: %s", target_date, exc)
        return []

    data = resp.json()
    snapshot_ts = datetime.now(timezone.utc)
    rows: list[dict] = []

    for date_block in data.get("dates", []) or []:
        for game in date_block.get("games", []) or []:
            game_pk = game.get("gamePk")
            if not game_pk:
                continue
            for official in game.get("officials", []) or []:
                if str(official.get("officialType", "")).strip() != "Home Plate":
                    continue
                person = official.get("official") or {}
                umpire_name = str(person.get("fullName") or "").strip()
                umpire_id = person.get("id")
                # Skip placeholder entries the API returns for unassigned slots
                if not umpire_name or umpire_name in {"HP Umpire", "NO UMPIRE"}:
                    continue
                rows.append(
                    {
                        "game_id": int(game_pk),
                        "game_date": target_date,
                        "umpire_name": umpire_name,
                        "umpire_id": int(umpire_id) if umpire_id else None,
                        "snapshot_ts": snapshot_ts,
                    }
                )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest MLB home plate umpire assignments")
    add_date_range_args(parser)
    args = parser.parse_args()
    start_date, end_date = resolve_date_range(args)

    if not table_exists("umpire_assignments"):
        log.warning("umpire_assignments table not found — run DB migrations first")
        return 0

    all_rows: list[dict] = []
    current = start_date if isinstance(start_date, date) else date.fromisoformat(str(start_date))
    end = end_date if isinstance(end_date, date) else date.fromisoformat(str(end_date))

    while current <= end:
        day_rows = _fetch_assignments_for_date(current)
        log.info("Umpire fetch %s: %d assignment(s)", current, len(day_rows))
        all_rows.extend(day_rows)
        current += timedelta(days=1)

    target_str = str(end_date)
    if not all_rows:
        log.info("No umpire assignments found for %s – %s", start_date, end_date)
        record_ingest_event(
            source_name="mlb_stats_api",
            ingestor_module="src.ingestors.umpire",
            target_date=target_str,
            row_count=0,
            parse_status="no_data",
        )
        return 0

    inserted = upsert_rows(
        "umpire_assignments",
        all_rows,
        conflict_columns=["game_id", "snapshot_ts"],
    )
    log.info("Upserted %d umpire assignment row(s)", inserted)
    record_ingest_event(
        source_name="mlb_stats_api",
        ingestor_module="src.ingestors.umpire",
        target_date=target_str,
        row_count=inserted,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
