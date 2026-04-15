"""Report matchup_splits coverage vs. lineups + starters implied pairs.

Run from repo root:
  python scripts/audit_matchup_data.py

Uses DATABASE_URL / project SQLite like the rest of the app.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text

from src.utils.db import get_engine


THEORETICAL_BVP_SQL = """
WITH game_teams AS (
  SELECT game_id, home_team, away_team FROM games
),
lineup_unique AS (
  SELECT DISTINCT l.game_id, l.player_id, l.team
  FROM lineups l
  WHERE l.lineup_slot IS NOT NULL AND l.lineup_slot BETWEEN 1 AND 9
),
starters AS (
  SELECT ps.game_id, ps.pitcher_id, ps.side
  FROM pitcher_starts ps
),
starter_by_game AS (
  SELECT game_id,
    MAX(CASE WHEN side = 'home' THEN pitcher_id END) AS home_pid,
    MAX(CASE WHEN side = 'away' THEN pitcher_id END) AS away_pid
  FROM starters
  GROUP BY game_id
),
pairs AS (
  SELECT DISTINCT
    lu.player_id AS batter_id,
    CASE
      WHEN lu.team = gt.home_team THEN sb.away_pid
      WHEN lu.team = gt.away_team THEN sb.home_pid
    END AS pitcher_id
  FROM lineup_unique lu
  JOIN game_teams gt ON lu.game_id = gt.game_id
  JOIN starter_by_game sb ON lu.game_id = sb.game_id
  WHERE (lu.team = gt.home_team OR lu.team = gt.away_team)
    AND sb.home_pid IS NOT NULL AND sb.away_pid IS NOT NULL
)
SELECT COUNT(*) FROM pairs;
"""

THEORETICAL_PVT_SQL = """
WITH game_teams AS (
  SELECT game_id, home_team, away_team FROM games
),
starters AS (
  SELECT ps.game_id, ps.pitcher_id, ps.team, ps.side
  FROM pitcher_starts ps
),
pairs AS (
  SELECT DISTINCT ps.pitcher_id,
    CASE WHEN ps.side = 'home' THEN gt.away_team ELSE gt.home_team END AS opp_team
  FROM starters ps
  JOIN game_teams gt ON ps.game_id = gt.game_id
)
SELECT COUNT(*) FROM pairs;
"""


def main() -> int:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT split_type, season, COUNT(*) AS n
                FROM matchup_splits
                GROUP BY split_type, season
                ORDER BY split_type, season
                """
            )
        ).fetchall()
        print("=== matchup_splits (cached) ===")
        for r in rows:
            print(f"  {r[0]:<18} season={r[1]!s:>5}  rows={r[2]}")

        g = conn.execute(
            text(
                """
                SELECT COUNT(*), MIN(game_date), MAX(game_date)
                FROM games
                """
            )
        ).fetchone()
        print("\n=== games ===")
        print(f"  count={g[0]}  min_date={g[1]}  max_date={g[2]}")

        lu_dates = conn.execute(
            text(
                """
                SELECT COUNT(DISTINCT game_date)
                FROM lineups
                WHERE lineup_slot IS NOT NULL AND lineup_slot BETWEEN 1 AND 9
                """
            )
        ).scalar()
        lu_games = conn.execute(
            text(
                """
                SELECT COUNT(DISTINCT game_id)
                FROM lineups
                WHERE lineup_slot IS NOT NULL AND lineup_slot BETWEEN 1 AND 9
                """
            )
        ).scalar()
        games_both_starters = conn.execute(
            text(
                """
                SELECT COUNT(*) FROM (
                  SELECT game_id
                  FROM pitcher_starts
                  GROUP BY game_id
                  HAVING COUNT(DISTINCT side) >= 2
                ) t
                """
            )
        ).scalar()
        print("\n=== lineups & starters (limits in-DB game-log BvP / joins) ===")
        print(f"  distinct game_date with lineup slot 1-9: {lu_dates}")
        print(f"  distinct game_id with lineup rows: {lu_games}")
        print(f"  games with starter row on home + away: {games_both_starters}")

        bvp_theory = conn.execute(text(THEORETICAL_BVP_SQL)).scalar()
        pvt_theory = conn.execute(text(THEORETICAL_PVT_SQL)).scalar()
        bvp_cached = conn.execute(
            text(
                "SELECT COUNT(*) FROM matchup_splits WHERE split_type = 'bvp' AND season = 0"
            )
        ).scalar()
        pvt_cached = conn.execute(
            text(
                """
                SELECT COUNT(*) FROM matchup_splits
                WHERE split_type = 'pitcher_vs_team' AND season = 0
                """
            )
        ).scalar()

        print("\n=== StatMuse cache (career rows, season=0) ===")
        print(f"  bvp rows: {bvp_cached}")
        print(f"  pitcher_vs_team rows: {pvt_cached}")
        print(
            "  (Grows from daily ingest on slates you run; not tied 1:1 to games row count.)"
        )

        print("\n=== Pairs derivable from current DB (lineups + both starters) ===")
        print(
            f"  distinct batter-vs-pitcher pairs (same logic as ingest slate): {bvp_theory}"
        )
        print(
            f"  distinct pitcher-vs-opponent-team pairs from starter rows: {pvt_theory}"
        )
        print(
            "  Cache can exceed these counts when ingest ran on dates with richer slates"
            " than historical rows preserved in games/lineups/pitcher_starts."
        )

        print("\n=== backfill options ===")
        print(
            "  1) StatMuse: run ingest over a date range (rate-limited; fills matchup_splits):\n"
            "       python -m src.ingestors.matchup_splits --start-date YYYY-MM-DD --end-date YYYY-MM-DD"
        )
        print(
            "  2) In-DB: backfill lineups + pitcher_starts for past games so game-log BvP /"
            " rollups can join more events."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
