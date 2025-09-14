"""Backfill prediction tracking for a recent date range.

Uses existing API endpoints:
  POST /api/prediction-tracking/record/{date}
  POST /api/prediction-tracking/update-results/{date}

Default behavior:
  - Iterates from (today - N days) up to yesterday
  - Records predictions (if not already recorded)
  - Updates results (to capture actual totals & correctness)

Optional:
  --include-today : also record today's predictions (no results yet)
  --days N        : number of trailing days (default 7)

Run:
  python mlb/tracking/monitoring/backfill_prediction_tracking.py --days 7
"""

from __future__ import annotations
import argparse
import requests
from datetime import datetime, timedelta
from typing import List, Dict

API_BASE = "http://localhost:8000"

def post(url: str) -> Dict:
    try:
        r = requests.post(url, timeout=30)
        try:
            data = r.json()
        except Exception:
            data = {"raw": r.text}
        return {"status": r.status_code, "data": data}
    except requests.RequestException as e:
        return {"status": None, "error": str(e)}

def backfill(days: int, include_today: bool) -> None:
    today = datetime.utcnow().date()
    # Range: today - days .. (today if include_today else yesterday)
    start_date = today - timedelta(days=days)
    end_date = today if include_today else today - timedelta(days=1)

    print(f"🔁 Backfilling prediction tracking from {start_date} to {end_date} (include_today={include_today})")

    dates: List[str] = []
    cur = start_date
    while cur <= end_date:
        dates.append(cur.strftime('%Y-%m-%d'))
        cur += timedelta(days=1)

    summary = []
    for d in dates:
        print(f"\n📅 Date: {d}")
        record_resp = post(f"{API_BASE}/api/prediction-tracking/record/{d}")
        if record_resp.get("status") == 200:
            recorded = record_resp["data"].get("recorded")
            print(f"  ✅ Record: {recorded} predictions")
        else:
            print(f"  ❌ Record failed: {record_resp}")
        # Skip updating today's results (games not done yet)
        if d == today.strftime('%Y-%m-%d') and not include_today:
            continue
        update_resp = post(f"{API_BASE}/api/prediction-tracking/update-results/{d}")
        if update_resp.get("status") == 200:
            print(f"  🔄 Update: OK")
        else:
            print(f"  ❌ Update failed: {update_resp}")
        summary.append({
            "date": d,
            "record_status": record_resp.get("status"),
            "update_status": update_resp.get("status")
        })

    print("\n=== Summary ===")
    for row in summary:
        print(f"{row['date']}: record={row['record_status']} update={row['update_status']}")
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Backfill prediction tracking data")
    parser.add_argument('--days', type=int, default=7, help='Trailing days to backfill (default 7)')
    parser.add_argument('--include-today', action='store_true', help='Also record today (results will remain pending)')
    args = parser.parse_args()
    backfill(args.days, args.include_today)

if __name__ == '__main__':
    main()
