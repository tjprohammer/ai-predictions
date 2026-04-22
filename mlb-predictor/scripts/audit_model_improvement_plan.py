"""Print Phase 0 metrics for docs/MODEL_IMPROVEMENT_PLAN.md (DB + feature parquets)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.db import query_df
from src.utils.settings import get_settings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_date_range_args(parser)
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=540,
        help="If no range given, end=yesterday and start=end-lookback (default 540)",
    )
    args = parser.parse_args()
    if getattr(args, "target_date", None) or (args.start_date and args.end_date):
        start_date, end_date = resolve_date_range(args)
    else:
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=max(1, args.lookback_days))

    print("=== MODEL_IMPROVEMENT_PLAN Phase 0 audit ===")
    print(f"Window: {start_date} .. {end_date}")
    print()

    df = query_df(
        """
        SELECT
            COUNT(*) AS games,
            SUM(CASE WHEN status = 'final' THEN 1 ELSE 0 END) AS final_games,
            SUM(CASE WHEN total_runs_inning1 IS NOT NULL THEN 1 ELSE 0 END) AS with_inning1,
            SUM(CASE WHEN status = 'final' AND total_runs_inning1 IS NULL THEN 1 ELSE 0 END) AS final_missing_inning1
        FROM games
        WHERE game_date BETWEEN :s AND :e
        """,
        {"s": start_date, "e": end_date},
    )
    print("games / inning-1 labels:")
    print(df.to_string(index=False))
    print()

    try:
        df2 = query_df(
            """
            SELECT market_type, COUNT(DISTINCT game_id) AS games_with_line
            FROM game_markets
            WHERE game_date BETWEEN :s AND :e
              AND market_type IN ('total', 'first_five_total')
              AND line_value IS NOT NULL
            GROUP BY market_type
            ORDER BY market_type
            """,
            {"s": start_date, "e": end_date},
        )
        print("game_markets (distinct games with line_value):")
        print(df2.to_string(index=False))
    except Exception as exc:
        print("game_markets:", exc)
    print()

    settings = get_settings()
    lane = Path(settings.feature_dir) / "first5_totals"
    files = sorted(lane.glob("*.parquet"))
    if not files:
        print(f"first5_totals: no parquet under {lane}")
    else:
        n, m = 0, 0
        for p in files:
            f = pd.read_parquet(p)
            n += len(f)
            if "market_total" in f.columns:
                m += int(f["market_total"].notna().sum())
        pct = 100.0 * m / n if n else 0.0
        print(f"first5_totals parquets: {len(files)} files, {n} rows, {m} non-null market_total ({pct:.1f}%)")

    report_dir = Path(settings.report_dir) / "totals"
    reports = sorted(report_dir.glob("totals_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    print()
    if not reports:
        print(f"totals reports: none under {report_dir} (train totals to generate)")
    else:
        latest = reports[0]
        try:
            payload = json.loads(latest.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"totals reports: failed to read {latest}: {exc}")
        else:
            m = payload.get("metrics") or {}
            b = payload.get("baselines") or {}
            vm = payload.get("validation_metrics") or {}
            print(f"Latest totals report: {latest.name}")
            if m and payload.get("model_name"):
                mn = payload["model_name"]
                if mn in m:
                    print(f"  val MAE (fundamentals, {mn}): {m[mn].get('mae')}")
            team = b.get("team_average") or {}
            if team.get("mae") is not None:
                print(f"  val MAE (team_average baseline): {team['mae']}")
            if vm.get("post_calibration_val_mae") is not None:
                print(f"  val MAE (post calibration): {vm['post_calibration_val_mae']}")
            if payload.get("market_shrink") is not None:
                print(f"  market_shrink: {payload['market_shrink']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
