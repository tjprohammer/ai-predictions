#!/usr/bin/env python3
"""
Historical MLB totals backfill using The Odds API snapshots.

Usage examples:
  $env:THE_ODDS_API_KEY="YOURKEY"
  python historical_totals_backfill.py --start 2025-07-15 --end 2025-08-14
  python historical_totals_backfill.py --date  2025-08-14 --book-order pinnacle,fanduel,draftkings

Notes
- Requires a paid plan that enables the /v4/historical endpoint.
- We query a snapshot at (game_time_utc - 60s) to approximate *closing*.
- We update enhanced_games.market_total, over_odds, under_odds.
- Optional archive table `market_lines` (see comment) if you want to keep a history.
"""

import os
import time
import argparse
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd
from sqlalchemy import create_engine, text

SPORT_KEY = "baseball_mlb"
ODDS_API = "https://api.the-odds-api.com/v4/historical/sports/{sport}/odds"

DEFAULT_BOOK_ORDER = ["pinnacle", "fanduel", "draftkings"]

def get_engine():
    url = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
    return create_engine(url, pool_pre_ping=True)

def iso(dt: datetime) -> str:
    # Expect dt to be UTC
    return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

def load_games_for_day(engine, ds: str) -> pd.DataFrame:
    q = text("""
        SELECT game_id, home_team, away_team, game_time_utc
        FROM enhanced_games
        WHERE "date" = :d
        ORDER BY game_time_utc NULLS LAST, game_id
    """)
    df = pd.read_sql(q, engine, params={"d": ds})
    # normalize names to improve match robustness
    for col in ("home_team","away_team"):
        df[col + "_norm"] = df[col].str.strip().str.lower()
    return df

def fetch_snapshot(api_key: str, snap_dt_utc: datetime, regions="us", markets="totals", odds_format="american") -> dict:
    params = {
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "apiKey": api_key,
        "date": iso(snap_dt_utc),
    }
    url = ODDS_API.format(sport=SPORT_KEY)
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def choose_book(markets: List[dict], book_order: List[str]) -> Optional[Tuple[float, Optional[int], Optional[int], str]]:
    """
    From a list of bookmaker dicts, pick the first in book_order that has a 'totals' market.
    Return (total_points, over_price, under_price, book_key)
    """
    for pref in book_order:
        for bk in markets:
            if bk.get("key") != pref:
                continue
            for mkt in bk.get("markets", []):
                if mkt.get("key") != "totals":
                    continue
                outs = mkt.get("outcomes", []) or []
                # find 'Over' and 'Under' with a 'point'
                total = None
                over_price = under_price = None
                for o in outs:
                    nm = (o.get("name") or "").lower()
                    if o.get("point") is not None:
                        total = float(o["point"])
                    if nm == "over":
                        over_price = o.get("price")
                        if isinstance(over_price, str):
                            try: over_price = int(over_price)
                            except: over_price = None
                    elif nm == "under":
                        under_price = o.get("price")
                        if isinstance(under_price, str):
                            try: under_price = int(under_price)
                            except: under_price = None
                if total is not None:
                    return total, over_price, under_price, pref
    return None

def match_event_to_game(event: dict, games: pd.DataFrame, time_tolerance_min: int = 15) -> Optional[str]:
    """
    Match a historical event to our game_id by normalized names and commence_time proximity.
    """
    ev_home = (event.get("home_team") or "").strip().lower()
    ev_away = (event.get("away_team") or "").strip().lower()
    try:
        ev_time = datetime.fromisoformat(event["commence_time"].replace("Z","+00:00"))
    except Exception:
        return None

    # Filter candidate rows by team names
    cand = games[(games["home_team_norm"] == ev_home) & (games["away_team_norm"] == ev_away)].copy()
    if cand.empty:
        return None

    # pick closest by start time if available
    if "game_time_utc" in cand and cand["game_time_utc"].notna().any():
        cand["diff_min"] = cand["game_time_utc"].apply(
            lambda t: abs((t - ev_time).total_seconds())/60 if pd.notna(t) else 1e9
        )
        cand = cand.sort_values("diff_min")
        best = cand.iloc[0]
        if best["diff_min"] <= time_tolerance_min:
            return str(best["game_id"])
        return None
    else:
        # If we lack start times, fall back to first match on names
        return str(cand.iloc[0]["game_id"])

def backfill_day(engine, ds: str, api_key: str, book_order: List[str], sleep_s: float = 0.3) -> int:
    games = load_games_for_day(engine, ds)
    if games.empty:
        print(f"{ds}: no games found.")
        return 0

    # unique snapshot instants to query (closing ~ 1 minute before first pitch)
    snaps = []
    for _, row in games.iterrows():
        if pd.isna(row["game_time_utc"]):
            continue
        snaps.append(row["game_time_utc"] - timedelta(seconds=60))
    # de-dup minute resolution to batch calls
    snaps = sorted({ dt.replace(second=0, microsecond=0) for dt in snaps })

    updates = []  # (game_id, total, over, under, book)
    for snap_dt in snaps:
        try:
            payload = fetch_snapshot(api_key, snap_dt)
        except requests.HTTPError as e:
            print(f"{ds} snapshot {snap_dt.isoformat()} HTTP {e.response.status_code}: {e}")
            # If plan doesn’t have access, bail out early
            if e.response is not None and e.response.status_code in (401,403):
                break
            continue
        except Exception as e:
            print(f"{ds} snapshot {snap_dt.isoformat()} error: {e}")
            continue

        data = payload.get("data", [])
        # Build an index for faster lookups by (home,away)
        for ev in data:
            gid = match_event_to_game(ev, games)
            if not gid:
                continue
            pick = choose_book(ev.get("bookmakers", []), book_order)
            if not pick:
                continue
            total, over_price, under_price, book_key = pick
            updates.append((gid, total, over_price, under_price, book_key))

        time.sleep(sleep_s)

    # Deduplicate by game_id keeping the *latest* update we collected
    latest = {}
    for gid, total, over_price, under_price, book_key in updates:
        latest[gid] = (total, over_price, under_price, book_key)

    if not latest:
        print(f"{ds}: no matches found in snapshots.")
        return 0

    with engine.begin() as conn:
        # OPTIONAL: ensure archive table (uncomment if you want to keep line history)
        conn.execute(text("""
          CREATE TABLE IF NOT EXISTS market_lines(
            game_id varchar NOT NULL,
            "date" date NOT NULL,
            source text NOT NULL,
            market_total numeric,
            over_odds integer,
            under_odds integer,
            captured_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (game_id, captured_at)
          );"""))

        n = 0
        for gid, (tot, o, u, bk) in latest.items():
            # archive (optional)
            conn.execute(text("""
              INSERT INTO market_lines(game_id,"date",source,market_total,over_odds,under_odds)
              VALUES (:gid, :d, :src, :tot, :o, :u)
            """), {"gid": gid, "d": ds, "src": bk, "tot": tot, "o": o, "u": u})

            # update current row
            res = conn.execute(text("""
                UPDATE enhanced_games
                   SET market_total = :tot,
                       over_odds     = :o,
                       under_odds    = :u
                 WHERE game_id = :gid AND "date" = :d
            """), {"tot": tot, "o": o, "u": u, "gid": gid, "d": ds})
            n += res.rowcount
    print(f"{ds}: updated {n} games with historical totals.")
    return n

def verify_days(engine, start: str, end: str):
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT "date",
                   COUNT(*) AS games,
                   COUNT(market_total) AS markets,
                   COUNT(predicted_total) AS preds,
                   COUNT(total_runs) AS finals
              FROM enhanced_games
             WHERE "date" BETWEEN :s AND :e
             GROUP BY 1
             ORDER BY 1
        """), {"s": start, "e": end}).fetchall()
    for r in rows:
        print(f"{r[0]}: games={r[1]:2d} markets={r[2]:2d} preds={r[3]:2d} finals={r[4]:2d}")

def parse_args():
    ap = argparse.ArgumentParser()
    one = ap.add_mutually_exclusive_group(required=True)
    one.add_argument("--date", help="Single day (YYYY-MM-DD)")
    one.add_argument("--start", help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", help="End date (YYYY-MM-DD) when --start is used")
    ap.add_argument("--book-order", default=",".join(DEFAULT_BOOK_ORDER),
                    help="Comma list of bookmaker keys by preference (e.g. pinnacle,fanduel,draftkings)")
    ap.add_argument("--sleep", type=float, default=0.3, help="Seconds between snapshot calls")
    return ap.parse_args()

def main():
    args = parse_args()
    api_key = os.getenv("THE_ODDS_API_KEY")
    if not api_key:
        print("ERROR: THE_ODDS_API_KEY not set.")
        raise SystemExit(2)

    engine = get_engine()
    books = [b.strip().lower() for b in args.book_order.split(",") if b.strip()]

    if args.date:
        backfill_day(engine, args.date, api_key, books, args.sleep)
        verify_days(engine, args.date, args.date)
    else:
        if not args.end:
            print("ERROR: --end is required when using --start")
            raise SystemExit(2)
        start = datetime.strptime(args.start, "%Y-%m-%d").date()
        end   = datetime.strptime(args.end,   "%Y-%m-%d").date()
        if end < start:
            print("ERROR: end < start")
            raise SystemExit(2)
        d = start
        while d <= end:
            ds = d.strftime("%Y-%m-%d")
            backfill_day(engine, ds, api_key, books, args.sleep)
            d += timedelta(days=1)
        verify_days(engine, args.start, args.end)

if __name__ == "__main__":
    main()
