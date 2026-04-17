"""
Simulate **Daily Results–style** slugger HR picks over a date range.

Uses the same data loader and ``iter_slugger_tracked_cards`` as the API (round-robin cap,
per-game ranking). Optional day-by-day ``predict_hr`` to fill ``predictions_player_hr``.

Example::

    python -m src.features.hr_builder --start-date 2026-03-20 --end-date 2026-04-15
    python -m src.models.train_hr
    python scripts/simulate_slugger_hr_daily.py --start-date 2026-04-01 --end-date 2026-04-15

With predictions already in DB::

    python scripts/simulate_slugger_hr_daily.py --start-date 2026-04-10 --end-date 2026-04-15 --skip-predict
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.cli import add_date_range_args, date_range, resolve_date_range
from src.utils.hr_source_recs_db import load_hr_source_recs_for_date
from src.utils.slugger_hr_selection import SLUGGER_DAILY_RESULTS_MAX_CARDS, iter_slugger_tracked_cards

_FINAL_MARKERS = ("final", "completed", "game over", "closed")


def _is_final(status: object) -> bool:
    s = str(status or "").strip().lower()
    return bool(s) and any(m in s for m in _FINAL_MARKERS)


def _grade(*, game_status: object, actual: object) -> str:
    if not _is_final(game_status):
        return "pending"
    if actual is None:
        return "missing"
    try:
        return "hit" if float(actual) > 0 else "no_hit"
    except (TypeError, ValueError):
        return "missing"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Historic slugger HR picks (Daily Results logic) vs outcomes",
    )
    add_date_range_args(parser)
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Do not run predict_hr per day (use existing predictions_player_hr)",
    )
    parser.add_argument(
        "--max-cards",
        type=int,
        default=SLUGGER_DAILY_RESULTS_MAX_CARDS,
        metavar="N",
        help=f"Global cap after round-robin (default: {SLUGGER_DAILY_RESULTS_MAX_CARDS}, same as Daily Results)",
    )
    parser.add_argument(
        "--csv",
        metavar="PATH",
        help="Append one row per slugger pick per day to this CSV file",
    )
    args = parser.parse_args()
    start, end = resolve_date_range(args)
    if end < start:
        print("end-date must be >= start-date", file=sys.stderr)
        return 1

    csv_path = Path(args.csv) if args.csv else None
    csv_file = None
    csv_writer: csv.DictWriter | None = None
    if csv_path is not None:
        new_file = not csv_path.exists()
        csv_file = csv_path.open("a", newline="", encoding="utf-8")
        fieldnames = [
            "game_date",
            "game_id",
            "player_id",
            "player_name",
            "team",
            "opponent",
            "slugger_rank_in_game",
            "predicted_hr_probability",
            "edge",
            "market_price",
            "result",
            "actual_home_runs",
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if new_file:
            csv_writer.writeheader()

    total_hits = 0
    total_no = 0
    total_pending = 0
    total_missing = 0
    days_with_picks = 0

    try:
        for d in date_range(start, end):
            if not args.skip_predict:
                subprocess.run(
                    [sys.executable, "-m", "src.models.predict_hr", "--target-date", d.isoformat()],
                    cwd=ROOT,
                    check=False,
                )
            source = load_hr_source_recs_for_date(d, 0.0)
            if not source:
                print(f"{d}: no HR source rows (features + predictions missing?)")
                continue
            tracked = iter_slugger_tracked_cards(
                source,
                max_cards=args.max_cards,
            )
            if not tracked:
                print(f"{d}: slugger selection produced no cards")
                continue
            days_with_picks += 1
            rec_by = {(int(r["game_id"] or 0), int(r["player_id"] or 0)): r for r in source}
            day_lines: list[str] = []
            for card in tracked:
                gid = int(card.get("game_id") or 0)
                pid = int(card.get("player_id") or 0)
                rec = rec_by.get((gid, pid)) or {}
                status = rec.get("game_status")
                actual = rec.get("actual_home_runs")
                gr = _grade(game_status=status, actual=actual)
                if gr == "hit":
                    total_hits += 1
                elif gr == "no_hit":
                    total_no += 1
                elif gr == "missing":
                    total_missing += 1
                else:
                    total_pending += 1
                mp = card.get("model_probability")
                edge = card.get("probability_edge")
                if edge is None:
                    edge = rec.get("edge")
                mkt = rec.get("market_price")
                if mkt is None:
                    mkt = card.get("price")
                rank = card.get("slugger_rank_in_game")
                name = card.get("player_name") or rec.get("player_name")
                team = card.get("team") or rec.get("team")
                opp = rec.get("opponent") or ""
                line = (
                    f"  #{rank} {name} ({team}) vs {opp} | P(HR)={mp} edge={edge} mkt={mkt} -> {gr}"
                    f"{'' if actual is None else f' (HR={actual})'}"
                )
                day_lines.append(line)
                if csv_writer is not None:
                    csv_writer.writerow(
                        {
                            "game_date": d.isoformat(),
                            "game_id": gid,
                            "player_id": pid,
                            "player_name": name,
                            "team": team,
                            "opponent": opp,
                            "slugger_rank_in_game": rank,
                            "predicted_hr_probability": mp,
                            "edge": edge,
                            "market_price": mkt,
                            "result": gr,
                            "actual_home_runs": actual,
                        }
                    )
            print(f"{d}: {len(tracked)} slugger picks")
            for ln in day_lines:
                print(ln)
    finally:
        if csv_file is not None:
            csv_file.close()

    print(
        f"\nSummary ({start} .. {end}): days with picks={days_with_picks} | "
        f"graded hit={total_hits} no_hit={total_no} missing_box={total_missing} pending={total_pending}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
