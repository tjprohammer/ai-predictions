from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd

from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.db import query_df, table_exists, upsert_rows
from src.utils.logging import get_logger

log = get_logger(__name__)


def _source_priority(value: object) -> int:
    raw = str(value or "").lower()
    if "manual" in raw or "csv" in raw:
        return 2
    if "odds" in raw or "covers" in raw:
        return 1
    return 0


def _freeze_rows(start_date, end_date) -> list[dict[str, object]]:
    if not table_exists("game_markets") or not table_exists("market_selection_freezes"):
        return []
    frame = query_df(
        """
        SELECT gm.game_id, gm.market_type, gm.sportsbook, gm.line_value,
               gm.snapshot_ts, gm.source_name, gm.over_price, gm.under_price,
               g.game_start_ts
        FROM game_markets gm
        JOIN games g ON g.game_id = gm.game_id AND g.game_date = gm.game_date
        WHERE gm.game_date BETWEEN :start_date AND :end_date
          AND gm.market_type IN ('total', 'first_five_total')
          AND gm.line_value IS NOT NULL
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    if frame.empty:
        return []

    frame["snapshot_ts"] = pd.to_datetime(frame["snapshot_ts"], errors="coerce", utc=True)
    frame["game_start_ts"] = pd.to_datetime(frame["game_start_ts"], errors="coerce", utc=True)
    frame = frame[frame["snapshot_ts"].notna()].copy()
    frame = frame[frame["game_start_ts"].isna() | (frame["snapshot_ts"] <= frame["game_start_ts"])].copy()
    if frame.empty:
        return []

    now = datetime.now(timezone.utc)
    frame["effective_cutoff"] = frame["game_start_ts"].fillna(now)
    frame["seconds_to_start"] = (frame["effective_cutoff"] - frame["snapshot_ts"]).dt.total_seconds()
    frame["price_complete"] = frame[["over_price", "under_price"]].notna().all(axis=1).astype(int)
    frame["source_priority"] = frame["source_name"].map(_source_priority)

    rows: list[dict[str, object]] = []
    for (game_id, market_type), group in frame.groupby(["game_id", "market_type"]):
        chosen = group.sort_values(
            ["snapshot_ts", "price_complete", "source_priority", "sportsbook"],
            ascending=[False, False, False, True],
        ).iloc[0]
        rows.append(
            {
                "game_id": int(game_id),
                "market_type": str(market_type),
                "frozen_sportsbook": str(chosen["sportsbook"]),
                "frozen_line_value": chosen["line_value"],
                "frozen_snapshot_ts": chosen["snapshot_ts"],
                "reason": "pregame_latest",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze pregame market selections")
    add_date_range_args(parser)
    args = parser.parse_args()
    start_date, end_date = resolve_date_range(args)

    rows = _freeze_rows(start_date, end_date)
    if not rows:
        log.info("No market freeze rows available for %s to %s", start_date, end_date)
        return 0
    written = upsert_rows("market_selection_freezes", rows, ["game_id", "market_type"])
    log.info("Frozen %s pregame market rows for %s to %s", written, start_date, end_date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
