from __future__ import annotations

import argparse

from sqlalchemy import text

from src.ingestors.common import player_dimension_row
from src.utils.db import get_engine, upsert_rows
from src.utils.logging import get_logger


log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill dim_players metadata from MLB player records")
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Only refresh players missing bats or throws metadata",
    )
    args = parser.parse_args()

    query = "SELECT player_id FROM dim_players"
    if args.missing_only:
        query += " WHERE bats IS NULL OR throws IS NULL"
    query += " ORDER BY player_id"

    engine = get_engine()
    with engine.begin() as connection:
        player_ids = [int(row[0]) for row in connection.execute(text(query)).fetchall()]

    if not player_ids:
        log.info("No dim_players rows matched the requested backfill scope")
        return 0

    player_rows = [player_dimension_row(player_id) for player_id in player_ids]
    inserted = upsert_rows("dim_players", player_rows, ["player_id"], engine=engine)
    log.info("Backfilled %s dim_players rows", inserted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())