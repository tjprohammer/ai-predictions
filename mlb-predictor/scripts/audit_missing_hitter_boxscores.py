"""List hitter prop rows where the game is final but player_game_batting has no hits row.

Usage (from mlb-predictor/):
  python scripts/audit_missing_hitter_boxscores.py
  python scripts/audit_missing_hitter_boxscores.py --date 2026-04-03
  python scripts/audit_missing_hitter_boxscores.py --start-date 2026-03-26 --end-date 2026-04-12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy import create_engine, text

from src.utils.settings import get_settings

FINAL_PRED = """(
    LOWER(TRIM(g.status)) LIKE '%final%'
    OR LOWER(TRIM(g.status)) LIKE '%completed%'
    OR LOWER(TRIM(g.status)) LIKE '%game over%'
    OR LOWER(TRIM(g.status)) LIKE '%closed%'
)"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Single game_date (YYYY-MM-DD)")
    parser.add_argument("--start-date", default="2026-03-26")
    parser.add_argument("--end-date", default="2026-04-12")
    parser.add_argument("--min-probability", type=float, default=0.5)
    args = parser.parse_args()

    engine = create_engine(get_settings().database_url)
    minp = args.min_probability

    base_from = f"""
    WITH ranked_predictions AS (
        SELECT p.*, ROW_NUMBER() OVER (
            PARTITION BY p.game_id, p.player_id ORDER BY p.prediction_ts DESC
        ) AS row_rank
        FROM predictions_player_hits p
        WHERE p.game_date = :d
    )
    SELECT p.game_id, p.player_id, g.status, g.game_start_ts,
           g.away_team || ' @ ' || g.home_team AS matchup,
           COALESCE(dp.full_name, CAST(p.player_id AS TEXT)) AS player_name,
           p.team, p.predicted_hit_probability
    FROM ranked_predictions p
    JOIN games g ON g.game_id = p.game_id AND g.game_date = p.game_date
    LEFT JOIN player_game_batting b ON b.game_id = p.game_id AND b.player_id = p.player_id
    LEFT JOIN dim_players dp ON dp.player_id = p.player_id
    WHERE p.row_rank = 1
      AND p.predicted_hit_probability >= :minp
      AND b.hits IS NULL
      AND {FINAL_PRED}
    ORDER BY g.game_start_ts, p.game_id, player_name
    """

    rollup_sql = f"""
    WITH ranked_predictions AS (
        SELECT p.*, ROW_NUMBER() OVER (
            PARTITION BY p.game_id, p.player_id ORDER BY p.prediction_ts DESC
        ) AS row_rank
        FROM predictions_player_hits p
        WHERE p.game_date BETWEEN :d1 AND :d2
    )
    SELECT CAST(p.game_date AS TEXT) AS game_date, COUNT(*) AS missing_count
    FROM ranked_predictions p
    JOIN games g ON g.game_id = p.game_id AND g.game_date = p.game_date
    LEFT JOIN player_game_batting b ON b.game_id = p.game_id AND b.player_id = p.player_id
    WHERE p.row_rank = 1
      AND p.predicted_hit_probability >= :minp
      AND b.hits IS NULL
      AND {FINAL_PRED}
    GROUP BY p.game_date
    ORDER BY p.game_date
    """

    with engine.connect() as conn:
        if args.date:
            rows = conn.execute(
                text(base_from), {"d": args.date, "minp": minp}
            ).mappings().all()
            print(f"date={args.date}  missing_rows={len(rows)}  (min_prob={minp})\n")
            for r in rows:
                print(dict(r))
            return 0

        d1, d2 = args.start_date, args.end_date
        roll = conn.execute(
            text(rollup_sql), {"d1": d1, "d2": d2, "minp": minp}
        ).mappings().all()
        total = sum(int(r["missing_count"]) for r in roll)
        print(f"Missing hitter box rows per date ({d1} .. {d2}), min_prob={minp}")
        print(f"{'game_date':<12} {'missing':>8}")
        for r in roll:
            print(f"{r['game_date']!s:<12} {int(r['missing_count']):>8}")
        print(f"{'TOTAL':<12} {total:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
