"""Pregame ingest lock: skip mutating markets, lineups, and starters inside a window before first pitch."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd

from src.utils.db import query_df
from src.utils.logging import get_logger

log = get_logger(__name__)


def is_pregame_ingest_locked(
    game_start_ts: object,
    *,
    lock_minutes: int,
    now: datetime | None = None,
) -> bool:
    """Return True when ``now`` is within ``lock_minutes`` before scheduled first pitch (or after)."""
    if lock_minutes <= 0:
        return False
    parsed = _parse_utc_datetime(game_start_ts)
    if parsed is None:
        return False
    now_utc = now or datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    cutoff = parsed - timedelta(minutes=lock_minutes)
    return now_utc >= cutoff


def _parse_utc_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.to_pydatetime()
    except Exception:
        return None


def locked_game_ids_from_db(start_date: date, end_date: date, lock_minutes: int) -> set[int]:
    """Game IDs on ``[start_date, end_date]`` whose start time is inside the lock window."""
    if lock_minutes <= 0:
        return set()
    frame = query_df(
        """
        SELECT game_id, game_start_ts
        FROM games
        WHERE game_date BETWEEN :start_date AND :end_date
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    if frame.empty:
        return set()
    out: set[int] = set()
    for row in frame.itertuples(index=False):
        if is_pregame_ingest_locked(row.game_start_ts, lock_minutes=lock_minutes):
            out.add(int(row.game_id))
    return out


def filter_games_dataframe_pregame_unlocked(
    games: pd.DataFrame,
    lock_minutes: int,
    *,
    log_name: str = "games",
) -> pd.DataFrame:
    """Drop rows for games in the pregame lock window (preserve DB state for those games)."""
    if lock_minutes <= 0 or games.empty:
        return games
    if "game_start_ts" not in games.columns:
        return games
    before = len(games)
    keep: list[bool] = []
    for val in games["game_start_ts"]:
        keep.append(not is_pregame_ingest_locked(val, lock_minutes=lock_minutes))
    out = games.loc[keep].copy()
    skipped = before - len(out)
    if skipped:
        log.info(
            "Pregame ingest lock (%s min before first pitch): skipping %s %s (preserving existing data)",
            lock_minutes,
            skipped,
            log_name,
        )
    return out


def filter_row_dicts_by_game_id(
    rows: list[dict[str, Any]],
    locked_ids: set[int],
    *,
    label: str,
) -> list[dict[str, Any]]:
    if not locked_ids or not rows:
        return rows
    out: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        gid = row.get("game_id")
        try:
            gid_int = int(gid)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            out.append(row)
            continue
        if gid_int in locked_ids:
            skipped += 1
            continue
        out.append(row)
    if skipped:
        log.info(
            "Pregame ingest lock: dropped %s %s row(s) for locked game(s)",
            skipped,
            label,
        )
    return out
