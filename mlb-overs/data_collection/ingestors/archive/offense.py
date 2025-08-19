from __future__ import annotations
"""
ESPN totals ingestor (unofficial endpoints)
- Pulls MLB scoreboard JSON for a given date and extracts over/under lines when present.
- Upserts into `market_moves` with book='espn'.

Usage:
  python -m ingestors.espn_market_lines --date 2025-04-01 [--as-close]

Notes:
- ESPN's public/hidden endpoints can change without notice. This script is best-effort.
- Endpoint: https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates=YYYYMMDD
"""
import argparse
import os
from datetime import datetime, timezone
import requests
import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = os.environ.get('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
eng = create_engine(DB_URL)

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"


def fetch_espn_totals(date_str: str) -> pd.DataFrame:
    d = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d")
    params = {"dates": d}
    headers = {
        "User-Agent": "mlb-overs/1.0 (+https://example.com)"
    }
    r = requests.get(SCOREBOARD, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()
    rows = []
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    for ev in data.get('events', []):
        gid = str(ev.get('id'))
        comps = ev.get('competitions') or []
        if not comps:
            continue
        comp = comps[0]
        # odds object can be at comp['odds'][0]
        odds = comp.get('odds') or []
        total = None
        juice = None
        if odds:
            o = odds[0]
            total = o.get('overUnder')
            # some payloads include details like 'O/U 8.5'
            # juice not always present; leave None
        # fallback: check pickcenter if present
        if total is None:
            pc = comp.get('pickcenter') or []
            if pc:
                total = pc[0].get('overUnder') or pc[0].get('total')
        if total is None:
            continue
        rows.append({
            'game_id': gid,
            'ts': now,
            'total': total,
            'juice': juice,
            'book': 'espn',
            'is_close': False,
        })
    return pd.DataFrame(rows)


def upsert(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    with eng.begin() as cx:
        df.to_sql('tmp_espn_odds', cx, index=False, if_exists='replace')
        cols = ','.join(df.columns)
        cx.execute(text(f"""
            INSERT INTO market_moves ({cols})
            SELECT {cols} FROM tmp_espn_odds
            ON CONFLICT (game_id, ts, book) DO UPDATE SET
              total = EXCLUDED.total,
              juice = EXCLUDED.juice,
              is_close = EXCLUDED.is_close
        """))
        cx.execute(text("DROP TABLE tmp_espn_odds"))
    return len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=True, help='YYYY-MM-DD')
    ap.add_argument('--as-close', action='store_true', help='mark snapshot as closing line')
    args = ap.parse_args()

    df = fetch_espn_totals(args.date)
    if args.as_close and not df.empty:
        df['is_close'] = True
    n = upsert(df)
    print('upsert espn totals:', n)

if __name__ == '__main__':
    main()

"""
Team rolling hitting metrics.
"""

def load_offense(date=None):
    return None

if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser()
    p.add_argument("--start")
    p.add_argument("--end")
    args = p.parse_args()
    print(f"[ingestors.offense_daily] stub ingest start={args.start} end={args.end}")
    sys.exit(0)
