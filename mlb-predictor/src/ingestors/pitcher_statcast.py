from __future__ import annotations

import argparse
from datetime import date, datetime, timezone

import pandas as pd
from pybaseball import statcast_pitcher
from sqlalchemy import text

from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.db import get_engine, query_df
from src.utils.logging import get_logger


log = get_logger(__name__)


SWINGING_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked"}
CALLED_DESCRIPTIONS = {"called_strike"}
FASTBALL_TYPES = {"FF", "FA", "FC", "SI"}


def _coerce_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        raise TypeError("game_date is required")
    if "T" in text or " " in text:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    return date.fromisoformat(text)


def _safe_mean(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def _fraction(mask: pd.Series) -> float | None:
    if len(mask) == 0:
        return None
    return float(mask.mean())


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill starter Statcast summaries via pybaseball")
    add_date_range_args(parser)
    args = parser.parse_args()
    start_date, end_date = resolve_date_range(args)

    starts = query_df(
        """
        SELECT game_id, game_date, pitcher_id
        FROM pitcher_starts
        WHERE game_date BETWEEN :start_date AND :end_date
          AND xwoba_against IS NULL
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    if starts.empty:
        log.info("All pitcher_starts rows already have Statcast data for the requested range")
        return 0

    log.info("Fetching Statcast for %d pitcher starts ...", len(starts))
    statement = text(
        """
        UPDATE pitcher_starts
        SET xwoba_against = :xwoba_against,
            xslg_against = :xslg_against,
            csw_pct = :csw_pct,
            avg_fb_velo = :avg_fb_velo,
            barrel_pct = :barrel_pct,
            hard_hit_pct = :hard_hit_pct,
            whiff_pct = :whiff_pct,
            updated_at = :updated_at
        WHERE game_id = :game_id
          AND pitcher_id = :pitcher_id
        """
    )

    total_updated = 0
    updated_at = datetime.now(timezone.utc)
    for i, row in enumerate(starts.itertuples(index=False), 1):
        game_date = _coerce_date(row.game_date)
        try:
            frame = statcast_pitcher(start_dt=game_date.isoformat(), end_dt=game_date.isoformat(), player_id=int(row.pitcher_id))
        except Exception:
            log.warning("  Statcast fetch failed for pitcher %s on %s, skipping", row.pitcher_id, game_date, exc_info=True)
            continue
        if frame is None or frame.empty:
            continue
        descriptions = frame.get("description", pd.Series(dtype=str)).fillna("")
        called = descriptions.isin(CALLED_DESCRIPTIONS)
        swinging = descriptions.isin(SWINGING_DESCRIPTIONS)
        batted = frame[pd.to_numeric(frame.get("launch_speed"), errors="coerce").notna()].copy()
        fb_frame = frame[frame.get("pitch_type", pd.Series(dtype=str)).isin(FASTBALL_TYPES)].copy()
        launch_speed_angle = pd.to_numeric(batted.get("launch_speed_angle"), errors="coerce")
        update_row = {
            "game_id": int(row.game_id),
            "pitcher_id": int(row.pitcher_id),
            "xwoba_against": _safe_mean(frame.get("estimated_woba_using_speedangle", pd.Series(dtype=float))),
            "xslg_against": _safe_mean(frame.get("estimated_slg_using_speedangle", pd.Series(dtype=float))),
            "csw_pct": _fraction(called | swinging),
            "avg_fb_velo": _safe_mean(fb_frame.get("release_speed", pd.Series(dtype=float))),
            "barrel_pct": _fraction(launch_speed_angle == 6),
            "hard_hit_pct": _fraction(pd.to_numeric(batted.get("launch_speed"), errors="coerce") >= 95),
            "whiff_pct": _fraction(swinging),
            "updated_at": updated_at,
        }

        with get_engine().begin() as connection:
            connection.execute(statement, [update_row])
        total_updated += 1

        if i % 50 == 0:
            log.info("  Progress: %d / %d starts processed (%d updated)", i, len(starts), total_updated)

    if total_updated == 0:
        log.info("No Statcast summaries were available for the requested range")
        return 0

    log.info("Updated Statcast summaries for %d starter rows", total_updated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())