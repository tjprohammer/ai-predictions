"""Backfill the ``lineups`` table from existing ``player_game_batting`` rows.

``player_batting`` ingest already stores ``lineup_slot`` from the MLB live feed, but
``matchup_splits`` and other features read the canonical ``lineups`` table. This
ingestor copies batting-order rows (slots 1-9) into ``lineups`` so slate-style
joins see full coverage without re-hitting the API.

Run::

    python -m src.ingestors.lineups_backfill --start-date 2025-04-01 --end-date 2025-10-01

Idempotent: uses ``source_name='derived_player_game_batting'`` and one
``snapshot_ts`` per game (from ``games.game_start_ts``, or noon UTC on
``game_date`` if missing).
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, time, timezone

import pandas as pd

from src.ingestors.common import record_ingest_event
from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.db import get_engine, query_df, upsert_rows
from src.utils.logging import get_logger


log = get_logger(__name__)

SOURCE_NAME = "derived_player_game_batting"


def _coerce_game_date(gd: object) -> date:
    if isinstance(gd, datetime):
        return gd.date()
    if isinstance(gd, date):
        return gd
    if hasattr(gd, "date") and callable(gd.date):
        return gd.date()  # pandas.Timestamp
    return date.fromisoformat(str(gd)[:10])


def _snapshot_ts(game_start_ts: object, game_date: date) -> datetime:
    if game_start_ts is not None and not pd.isna(game_start_ts):
        ts = pd.to_datetime(game_start_ts, utc=True)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.to_pydatetime()
    return datetime.combine(game_date, time(12, 0), tzinfo=timezone.utc)


def backfill_lineups_from_player_batting(
    start_date: date,
    end_date: date,
    *,
    engine=None,
) -> int:
    """Insert/update ``lineups`` rows from ``player_game_batting``. Returns row count."""
    active_engine = engine or get_engine()
    frame = query_df(
        """
        SELECT
            pgb.game_id,
            pgb.game_date,
            pgb.team,
            pgb.player_id,
            COALESCE(dp.full_name, CAST(pgb.player_id AS TEXT)) AS player_name,
            pgb.lineup_slot,
            dp.position AS field_position,
            g.game_start_ts
        FROM player_game_batting pgb
        INNER JOIN games g
          ON g.game_id = pgb.game_id
         AND g.game_date = pgb.game_date
        LEFT JOIN dim_players dp
          ON dp.player_id = pgb.player_id
        WHERE pgb.game_date BETWEEN :start_date AND :end_date
          AND pgb.lineup_slot IS NOT NULL
          AND pgb.lineup_slot BETWEEN 1 AND 9
        """,
        {"start_date": start_date, "end_date": end_date},
        engine=active_engine,
    )
    if frame.empty:
        log.info(
            "No player_game_batting rows with lineup_slot 1-9 for %s to %s",
            start_date,
            end_date,
        )
        return 0

    lineup_rows: list[dict] = []
    for row in frame.itertuples(index=False):
        gd = _coerce_game_date(row.game_date)
        snap = _snapshot_ts(row.game_start_ts, gd)
        lineup_rows.append(
            {
                "game_id": int(row.game_id),
                "game_date": gd,
                "team": row.team,
                "player_id": int(row.player_id),
                "player_name": row.player_name,
                "lineup_slot": int(row.lineup_slot),
                "field_position": row.field_position,
                "batting_order": int(row.lineup_slot),
                "is_confirmed": True,
                "source_name": SOURCE_NAME,
                "source_url": None,
                "snapshot_ts": snap,
            }
        )

    inserted = upsert_rows(
        "lineups",
        lineup_rows,
        ["game_id", "player_id", "source_name", "snapshot_ts"],
        engine=active_engine,
    )
    log.info(
        "Backfilled %s lineup rows from player_game_batting (%s to %s)",
        inserted,
        start_date,
        end_date,
    )
    return inserted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill lineups table from player_game_batting (lineup slots 1-9)"
    )
    add_date_range_args(parser)
    args = parser.parse_args()
    start_date, end_date = resolve_date_range(args)

    row_count = backfill_lineups_from_player_batting(start_date, end_date)
    record_ingest_event(
        source_name="player_game_batting_derived",
        ingestor_module="src.ingestors.lineups_backfill",
        target_date=start_date.isoformat(),
        row_count=row_count,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
