"""Print best-bet trust bucket stats and monotonicity (CLI helper for calibration review).

Run from repo root with the app database configured, e.g.:

  python scripts/report_best_bet_trust.py --target-date 2026-04-12 --window-days 14 --graded-only

Uses the same logic as GET /api/recommendations/best-bets-history.
"""
from __future__ import annotations

import argparse
import json
from datetime import date

import src.api.app_logic as app_logic


def _parse_date(raw: str) -> date:
    return date.fromisoformat(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Best-bet trust calibration summary")
    parser.add_argument(
        "--target-date",
        default=None,
        help="End of trailing window (ISO date). Default: today (local date).",
    )
    parser.add_argument("--window-days", type=int, default=14, help="Trailing window length")
    parser.add_argument(
        "--graded-only",
        action="store_true",
        help="Only include graded tickets (same as API graded_only=true)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON (summary, by_input_trust, monotonicity only)",
    )
    args = parser.parse_args()

    if args.target_date:
        target = _parse_date(args.target_date)
    else:
        target = date.today()

    payload = app_logic._fetch_best_bet_history_payload(
        target,
        window_days=max(1, min(args.window_days, 60)),
        limit=500,
        graded_only=bool(args.graded_only),
    )

    slim = {
        "target_date": payload.get("target_date"),
        "start_date": payload.get("start_date"),
        "end_date": payload.get("end_date"),
        "window_days": payload.get("window_days"),
        "summary": payload.get("summary"),
        "by_input_trust": payload.get("by_input_trust"),
        "monotonicity": payload.get("monotonicity"),
    }

    if args.json:
        print(json.dumps(slim, indent=2, default=str))
        return

    print("Best Bet trust — trailing window")
    print(f"  End date: {slim['end_date']}  |  days: {slim['window_days']}  |  graded_only: {args.graded_only}")
    s = slim.get("summary") or {}
    print(
        f"  All tickets: {s.get('total', 0)}  |  graded: {s.get('graded', 0)}  |  "
        f"record {s.get('wins', 0)}-{s.get('losses', 0)}-{s.get('pushes', 0)}"
    )
    if s.get("win_rate") is not None:
        print(f"  Win rate (decisions): {float(s['win_rate']):.3f}")

    print("\nBy input trust grade:")
    for row in slim.get("by_input_trust") or []:
        g = row.get("input_trust_grade")
        wr = row.get("win_rate")
        wr_s = f"{float(wr):.3f}" if wr is not None else "—"
        print(
            f"  {g}: n={row.get('total', 0)} graded={row.get('graded', 0)} "
            f"wins={row.get('wins', 0)} win_rate={wr_s}"
        )

    mono = slim.get("monotonicity") or {}
    print("\nMonotonicity (win rate A→D):")
    print(f"  status: {mono.get('status')}")
    print(f"  {mono.get('interpretation', '')}")


if __name__ == "__main__":
    main()
