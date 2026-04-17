"""
Walk-forward HR backtest: run ``predict_hr`` per calendar day, then grade vs ``player_game_batting``.

Requires HR feature parquet for each day, a trained artifact, and (for edges) ``player_prop_markets`` rows.

Example::

    python -m src.models.train_hr --train-end-date 2026-04-01
    python scripts/simulate_hr_backtest.py --start-date 2026-03-20 --end-date 2026-04-15
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.db import query_df


def _daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Day-by-day HR prediction vs outcomes")
    parser.add_argument("--start-date", required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, metavar="YYYY-MM-DD")
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Only grade existing predictions_player_hr rows (do not invoke predict_hr)",
    )
    args = parser.parse_args()
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    if end < start:
        print("end-date must be >= start-date", file=sys.stderr)
        return 1

    graded_green = {"won": 0, "lost": 0, "pending": 0}
    all_brier: list[float] = []

    for d in _daterange(start, end):
        if not args.skip_predict:
            subprocess.run(
                [sys.executable, "-m", "src.models.predict_hr", "--target-date", d.isoformat()],
                cwd=ROOT,
                check=False,
            )
        frame = query_df(
            """
            WITH ranked AS (
                SELECT
                    p.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY p.game_id, p.player_id
                        ORDER BY p.prediction_ts DESC
                    ) AS row_rank
                FROM predictions_player_hr p
                WHERE p.game_date = :d
            )
            SELECT
                r.player_id,
                r.predicted_hr_probability,
                r.edge,
                r.market_price,
                b.home_runs AS actual_home_runs,
                g.status AS game_status
            FROM ranked r
            INNER JOIN games g ON g.game_id = r.game_id AND g.game_date = r.game_date
            LEFT JOIN player_game_batting b
                ON b.game_id = r.game_id AND b.player_id = r.player_id AND b.game_date = r.game_date
            WHERE r.row_rank = 1
            """,
            {"d": d},
        )
        if frame.empty:
            print(f"{d}: no predictions")
            continue

        final_markers = ("final", "completed", "game over", "closed")

        def _is_final(status: object) -> bool:
            s = str(status or "").strip().lower()
            return bool(s) and any(m in s for m in final_markers)

        day_green = {"won": 0, "lost": 0, "pending": 0}
        for _, row in frame.iterrows():
            mkt = row["market_price"]
            edge = row["edge"]
            if mkt is None or pd.isna(mkt) or edge is None or pd.isna(edge) or float(edge) <= 0:
                continue
            if not _is_final(row.get("game_status")):
                day_green["pending"] += 1
                graded_green["pending"] += 1
                continue
            hr_ct = row["actual_home_runs"]
            if hr_ct is None or (isinstance(hr_ct, float) and pd.isna(hr_ct)):
                day_green["pending"] += 1
                graded_green["pending"] += 1
                continue
            hit = float(hr_ct) > 0
            key = "won" if hit else "lost"
            day_green[key] += 1
            graded_green[key] += 1

        for _, row in frame.iterrows():
            if not _is_final(row.get("game_status")):
                continue
            hr_ct = row["actual_home_runs"]
            if hr_ct is None or (isinstance(hr_ct, float) and pd.isna(hr_ct)):
                continue
            y = 1.0 if float(hr_ct) > 0 else 0.0
            p = float(row["predicted_hr_probability"])
            all_brier.append((p - y) ** 2)

        print(
            f"{d}: n={len(frame)} | day green W/L {day_green['won']}/{day_green['lost']}"
            f" (green pending {day_green['pending']})"
        )

    if all_brier:
        mean_brier = sum(all_brier) / len(all_brier)
        print(f"Mean Brier (final games, all rows): {mean_brier:.5f} over {len(all_brier)} player-games")
    print("Green picks (edge>0, priced) cumulative:", graded_green)

    top = query_df(
        """
        WITH ranked AS (
            SELECT p.*, ROW_NUMBER() OVER (
                PARTITION BY p.game_id, p.player_id ORDER BY p.prediction_ts DESC
            ) AS rn
            FROM predictions_player_hr p
            WHERE p.game_date = :end_d
        )
        SELECT r.predicted_hr_probability, r.edge, r.market_price,
               COALESCE(dp.full_name, CAST(r.player_id AS TEXT)) AS player_name, r.team
        FROM ranked r
        LEFT JOIN dim_players dp ON dp.player_id = r.player_id
        WHERE r.rn = 1 AND r.market_price IS NOT NULL
        ORDER BY r.predicted_hr_probability DESC NULLS LAST
        LIMIT 15
        """,
        {"end_d": end},
    )
    if not top.empty:
        print(f"\nTop priced HR candidates for {end}:")
        for _, row in top.iterrows():
            print(
                f"  {row.get('player_name')} ({row.get('team')}): "
                f"P={float(row['predicted_hr_probability']):.3f} edge={row.get('edge')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
