#!/usr/bin/env python3
"""Summarize pitching data coverage: starters vs relievers per team-game, and spot-check aggregates.

Relief/bullpen rollups in this repo come from:

- ``player_game_pitching`` — one row per pitcher who appears in the MLB StatsAPI box score
  (``src.ingestors.boxscores``). ``is_starter`` is set from ``gamesStarted > 0`` in the feed.
- ``bullpens_daily`` — built by aggregating **non-starter** rows in ``player_game_pitching``
  (``src.transforms.bullpens_daily``). Run it after boxscores for the same date range.

Starter **probables** live in ``pitcher_starts`` from ``src.ingestors.starters``; after a game,
``src.ingestors.boxscores`` upserts actual starter lines into ``pitcher_starts`` for players
with ``gamesStarted``.

Usage (from ``mlb-predictor``):

  python scripts/validate_pitching_coverage.py --target-date 2026-04-15
  python scripts/validate_pitching_coverage.py --start-date 2026-04-01 --end-date 2026-04-15

Environment: ``DATABASE_URL`` / ``.env`` as for the rest of the app.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.db import get_dialect_name, query_df, table_exists


def _is_starter_expr(alias: str = "p") -> str:
    """SQL fragment true when row is a starter (SQLite + PostgreSQL)."""
    col = f"{alias}.is_starter"
    if get_dialect_name() == "sqlite":
        return f"({col} = 1 OR {col} = TRUE OR {col} = 'true')"
    return f"COALESCE({col}, FALSE) IS TRUE"


def pitching_team_game_summary(start: date, end: date) -> None:
    st = _is_starter_expr("p")
    frame = query_df(
        f"""
        WITH sides AS (
            SELECT g.game_id, g.game_date, g.status, g.away_team AS team, 'away' AS which
            FROM games g
            WHERE g.game_date BETWEEN :start_date AND :end_date
              AND g.game_type = 'R'
            UNION ALL
            SELECT g.game_id, g.game_date, g.status, g.home_team, 'home'
            FROM games g
            WHERE g.game_date BETWEEN :start_date AND :end_date
              AND g.game_type = 'R'
        )
        SELECT
            s.game_id,
            s.game_date,
            s.status,
            s.which,
            s.team,
            COUNT(p.player_id) AS n_pitchers,
            SUM(CASE WHEN {st} THEN 1 ELSE 0 END) AS n_starters,
            SUM(CASE WHEN NOT ({st}) THEN 1 ELSE 0 END) AS n_relievers
        FROM sides s
        LEFT JOIN player_game_pitching p
          ON p.game_id = s.game_id
         AND p.game_date = s.game_date
         AND p.team = s.team
        GROUP BY s.game_id, s.game_date, s.status, s.which, s.team
        ORDER BY s.game_date, s.game_id, s.which
        """,
        {"start_date": start, "end_date": end},
    )
    if frame.empty:
        print("No regular-season games in range (or DB empty).")
        return

    final_mask = frame["status"].astype(str).str.lower().isin(("final", "completed", "game over"))
    finals = frame[final_mask].copy()
    missing_pitchers = finals[finals["n_pitchers"] == 0]
    missing_starters = finals[finals["n_starters"] == 0]
    multi_starters = finals[finals["n_starters"] > 1]

    print("=== Per team-side (regular season) ===")
    print(frame.to_string(index=False))
    print()
    print("=== Flags (status in final/completed/game over) ===")
    print(f"  Team-sides with **no** pitching rows: {len(missing_pitchers)}")
    if not missing_pitchers.empty:
        print(missing_pitchers[["game_id", "game_date", "team", "status"]].to_string(index=False))
    print(f"  Team-sides with **no** starter flag: {len(missing_starters)} (openers/bulk issues or missing boxscore)")
    if not missing_starters.empty:
        print(missing_starters[["game_id", "game_date", "team", "n_pitchers", "n_relievers"]].to_string(index=False))
    print(f"  Team-sides with **>1** starter (opener/bulk): {len(multi_starters)}")
    if not multi_starters.empty:
        print(multi_starters[["game_id", "game_date", "team", "n_starters", "n_relievers"]].to_string(index=False))


def bullpens_daily_sanity(start: date, end: date) -> None:
    if not table_exists("bullpens_daily"):
        print("Table bullpens_daily not present; skip aggregate check.")
        return
    st = _is_starter_expr("p")
    cmp_frame = query_df(
        f"""
        SELECT
            p.game_date,
            p.team,
            COUNT(DISTINCT p.player_id) AS relievers_from_pgp,
            MAX(b.relievers_used) AS relievers_in_bullpens_daily
        FROM player_game_pitching p
        LEFT JOIN bullpens_daily b
          ON b.game_date = p.game_date AND b.team = p.team
        WHERE p.game_date BETWEEN :start_date AND :end_date
          AND NOT ({st})
        GROUP BY p.game_date, p.team
        ORDER BY p.game_date, p.team
        """,
        {"start_date": start, "end_date": end},
    )
    if cmp_frame.empty:
        print("No reliever rows in player_game_pitching for range (or no games).")
        return
    mismatch = cmp_frame[
        (cmp_frame["relievers_in_bullpens_daily"].notna())
        & (cmp_frame["relievers_from_pgp"] != cmp_frame["relievers_in_bullpens_daily"])
    ]
    print("=== bullpens_daily.relievers_used vs count(distinct relievers) in player_game_pitching ===")
    print(cmp_frame.to_string(index=False))
    if not mismatch.empty:
        print("\n**Mismatch rows** (rebuild bullpens_daily for these dates?):")
        print(mismatch.to_string(index=False))
    else:
        print("\nCounts match where bullpens_daily row exists.")


def pitcher_starts_vs_box_starters(start: date, end: date) -> None:
    """Probable rows (is_probable) may differ from actual starter in player_game_pitching."""
    st = _is_starter_expr("p")
    frame = query_df(
        f"""
        SELECT
            ps.game_id,
            ps.game_date,
            ps.team,
            ps.pitcher_id,
            ps.is_probable,
            p.is_starter AS pgp_is_starter
        FROM pitcher_starts ps
        LEFT JOIN player_game_pitching p
          ON p.game_id = ps.game_id
         AND p.game_date = ps.game_date
         AND p.player_id = ps.pitcher_id
        WHERE ps.game_date BETWEEN :start_date AND :end_date
        ORDER BY ps.game_date, ps.game_id, ps.team
        """,
        {"start_date": start, "end_date": end},
    )
    if frame.empty:
        print("No pitcher_starts rows in range.")
        return
    # Probable who never appeared as pitcher in box
    prob_no_line = frame[frame["is_probable"].fillna(False).astype(bool) & frame["pgp_is_starter"].isna()]
    # Appeared in box but not marked starter when probable still true (pregame row not reconciled)
    weird = frame[frame["pgp_is_starter"].notna() & ~frame["pgp_is_starter"].fillna(False).astype(bool) & frame["is_probable"].fillna(False).astype(bool)]

    print("=== pitcher_starts vs player_game_pitching (same game_id, pitcher_id) ===")
    print(
        "  Probable row but **no** pitching line in box: "
        f"{len(prob_no_line)} (scratch / wrong ID / box not run yet)"
    )
    if not prob_no_line.empty and len(prob_no_line) <= 40:
        print(prob_no_line.to_string(index=False))
    print(
        "  Probable + pitching row but **not** starter (pregame stale?): "
        f"{len(weird)}"
    )
    if not weird.empty and len(weird) <= 40:
        print(weird.to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate pitching / relief coverage in the database")
    add_date_range_args(parser)
    parser.add_argument(
        "--skip-bullpens",
        action="store_true",
        help="Do not compare bullpens_daily to player_game_pitching reliever counts",
    )
    parser.add_argument(
        "--skip-starters",
        action="store_true",
        help="Do not compare pitcher_starts to box rows",
    )
    args = parser.parse_args()
    start, end = resolve_date_range(args, default_days_back=0)
    if not table_exists("player_game_pitching"):
        print("player_game_pitching table missing.", file=sys.stderr)
        return 1

    print(f"Date range: {start} .. {end}\n")
    pitching_team_game_summary(start, end)
    print()
    if not args.skip_bullpens:
        bullpens_daily_sanity(start, end)
        print()
    if not args.skip_starters:
        pitcher_starts_vs_box_starters(start, end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
