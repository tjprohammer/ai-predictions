"""Backfill batter Statcast summaries (xwOBA, xBA, hard-hit%, barrel%, exit-velo, launch-angle)
into the existing player_game_batting rows.

Uses pybaseball.statcast() to pull pitch-level data in date chunks, then
aggregates per batter per game and UPDATEs the corresponding rows.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone

import pandas as pd
from pybaseball import statcast
from sqlalchemy import text

from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.db import get_engine, query_df
from src.utils.logging import get_logger

log = get_logger(__name__)

# pybaseball recommends chunks of ≤5 days to avoid Baseball Savant timeouts
_CHUNK_DAYS = 5


def _safe_mean(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return round(float(clean.mean()), 4)


def _fraction(mask: pd.Series) -> float | None:
    clean = mask.dropna()
    if clean.empty:
        return None
    return round(float(clean.mean()), 4)


def _date_chunks(start: date, end: date, chunk_size: int = _CHUNK_DAYS):
    """Yield (chunk_start, chunk_end) pairs spanning the full range."""
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_size - 1), end)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def _aggregate_batter_game(frame: pd.DataFrame) -> list[dict]:
    """Aggregate pitch-level Statcast data into per-batter, per-game summaries."""
    if frame.empty:
        return []

    # Ensure required columns exist
    for col in ("batter", "game_date", "game_pk"):
        if col not in frame.columns:
            return []

    rows = []
    grouped = frame.groupby(["batter", "game_date", "game_pk"])
    for (batter_id, game_date_val, _game_pk), group in grouped:
        # Batted-ball events (have launch data)
        batted = group[pd.to_numeric(group.get("launch_speed", pd.Series(dtype=float)), errors="coerce").notna()].copy()

        ev_avg = _safe_mean(batted.get("launch_speed", pd.Series(dtype=float)))
        la_avg = _safe_mean(batted.get("launch_angle", pd.Series(dtype=float)))

        launch_speed = pd.to_numeric(batted.get("launch_speed", pd.Series(dtype=float)), errors="coerce")
        hard_hit = _fraction(launch_speed >= 95) if not batted.empty else None

        launch_speed_angle = pd.to_numeric(batted.get("launch_speed_angle", pd.Series(dtype=float)), errors="coerce")
        barrel = _fraction(launch_speed_angle == 6) if not batted.empty else None

        xba = _safe_mean(group.get("estimated_ba_using_speedangle", pd.Series(dtype=float)))
        xwoba = _safe_mean(group.get("estimated_woba_using_speedangle", pd.Series(dtype=float)))

        # Skip if we got nothing useful
        if all(v is None for v in (ev_avg, la_avg, hard_hit, barrel, xba, xwoba)):
            continue

        rows.append({
            "player_id": int(batter_id),
            "game_date": str(game_date_val)[:10],
            "exit_velocity_avg": ev_avg,
            "launch_angle_avg": la_avg,
            "hard_hit_pct": hard_hit,
            "barrel_pct": barrel,
            "xba": xba,
            "xwoba": xwoba,
        })

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill batter Statcast summaries via pybaseball")
    add_date_range_args(parser)
    args = parser.parse_args()
    start_date, end_date = resolve_date_range(args)

    # Find which rows still need Statcast data
    missing = query_df(
        """
        SELECT DISTINCT game_date
        FROM player_game_batting
        WHERE game_date BETWEEN :start_date AND :end_date
          AND xwoba IS NULL
        ORDER BY game_date
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    if missing.empty:
        log.info("All player_game_batting rows already have Statcast data for the requested range")
        return 0

    dates_needed = sorted(set(str(d)[:10] for d in missing["game_date"]))
    range_start = date.fromisoformat(dates_needed[0])
    range_end = date.fromisoformat(dates_needed[-1])
    log.info(
        "Fetching batter Statcast for %d dates (%s to %s)",
        len(dates_needed), range_start, range_end,
    )

    total_updated = 0
    updated_at = datetime.now(timezone.utc)

    statement = text(
        """
        UPDATE player_game_batting
        SET exit_velocity_avg = :exit_velocity_avg,
            launch_angle_avg = :launch_angle_avg,
            hard_hit_pct = :hard_hit_pct,
            barrel_pct = :barrel_pct,
            xba = :xba,
            xwoba = :xwoba,
            updated_at = :updated_at
        WHERE player_id = :player_id
          AND game_date = :game_date
        """
    )

    for chunk_start, chunk_end in _date_chunks(range_start, range_end):
        log.info("  Pulling Statcast chunk %s to %s ...", chunk_start, chunk_end)
        try:
            frame = statcast(
                start_dt=chunk_start.isoformat(),
                end_dt=chunk_end.isoformat(),
                verbose=False,
                parallel=True,
            )
        except Exception:
            log.warning("  Statcast fetch failed for %s to %s, skipping", chunk_start, chunk_end, exc_info=True)
            continue

        if frame is None or frame.empty:
            log.info("  No data returned for %s to %s", chunk_start, chunk_end)
            continue

        chunk_updates = _aggregate_batter_game(frame)
        log.info("  Got %d batter-game summaries from chunk", len(chunk_updates))

        if not chunk_updates:
            continue

        for row in chunk_updates:
            row["updated_at"] = updated_at

        with get_engine().begin() as connection:
            connection.execute(statement, chunk_updates)

        total_updated += len(chunk_updates)
        log.info("  Committed chunk — running total: %d rows updated", total_updated)

    if total_updated == 0:
        log.info("No Statcast batter summaries were available for the requested range")
        return 0

    log.info("Updated Statcast summaries for %d batter-game rows total", total_updated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
