#!/usr/bin/env python3
"""
Fetch real historical MLB totals from The Odds API snapshots and write them into:
  - betting_lines (one row per game/bookmaker/snapshot used)
  - enhanced_games.market_total, over_odds, under_odds (closing, by bookmaker)

Usage:
  $env:THE_ODDS_API_KEY="..."    # Windows PowerShell
  python historical_markets_backfill.py --start 2025-05-01 --end 2025-08-15 --bookmaker fanduel --sleep 0.8

Notes:
- Requires a paid plan (historical endpoint).
- We pick the snapshot at/just before each game's first pitch (game_time_utc).
- We match games by (date, home_team, away_team) using normalized names.
"""

import os, sys, time, argparse, math
import requests
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine, text

SPORT_KEY = "baseball_mlb"  # The Odds API sport key
REGION    = "us"
MARKETS   = "totals"        # we only need totals
API_URL   = "https://api.the-odds-api.com/v4/historical/sports/{sport}/odds"

def norm(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())

TEAM_SYNONYM = {
    # add any mismatches if you see them in logs
    "oaklandathletics":"athletics",
    "losangelesangels":"losangelesangels",  # sometimes "la angels" appears; add if needed
    "losangelesdodgers":"losangelesdodgers",
    "stlouiscardinals":"stlouiscardinals",
    "chicagowhitesox":"chicagowhitesox",
    "arizonadiamondbacks":"arizonadiamondbacks",
}

def same_team(a, b):
    na, nb = norm(a), norm(b)
    na = TEAM_SYNONYM.get(na, na)
    nb = TEAM_SYNONYM.get(nb, nb)
    return na == nb

def get_engine():
    url = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
    return create_engine(url, pool_pre_ping=True)

def fetch_snapshot(api_key: str, iso_ts: str):
    params = {
        "regions": REGION,
        "oddsFormat": "american",
        "markets": MARKETS,
        "apiKey": api_key,
        "date": iso_ts,
    }
    url = API_URL.format(sport=SPORT_KEY)
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    # top-level wrapper with snapshot metadata
    return r.json()  # { "timestamp":..., "previous_timestamp":..., "next_timestamp":..., "data":[...] }

def choose_totals(record: dict, bookmaker_key: str):
    """Extract totals line (point, over price, under price) for a single event for a given bookmaker."""
    for bk in record.get("bookmakers", []):
        if bk.get("key") != bookmaker_key:
            continue
        for m in bk.get("markets", []):
            if m.get("key") != "totals":
                continue
            point = None
            over_price = under_price = None
            for out in m.get("outcomes", []):
                # Outcomes for totals have "name": "Over"/"Under" and "point": 9.5
                if "point" in out:
                    point = float(out["point"])
                if out.get("name","").lower() == "over":
                    over_price = int(out["price"])
                elif out.get("name","").lower() == "under":
                    under_price = int(out["price"])
            if point is not None:
                return point, over_price, under_price, bk.get("last_update")
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)  # YYYY-MM-DD
    ap.add_argument("--end", required=True)    # YYYY-MM-DD
    ap.add_argument("--bookmaker", default="fanduel", help="bookmaker key (e.g., fanduel, draftkings, caesars)")
    ap.add_argument("--sleep", type=float, default=0.8, help="seconds between API calls to be kind to rate limits")
    args = ap.parse_args()

    api_key = os.getenv("THE_ODDS_API_KEY")
    if not api_key:
        print("ERROR: THE_ODDS_API_KEY not set")
        sys.exit(2)

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end   = datetime.strptime(args.end, "%Y-%m-%d").date()

    eng = get_engine()
    updated = inserted = 0
    days = (end - start).days + 1

    with eng.begin() as conn:
        # Create betting_lines minimally if missing
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS betting_lines (
            id BIGSERIAL PRIMARY KEY,
            game_id VARCHAR NULL,
            date DATE NOT NULL,
            bookmaker TEXT NOT NULL,
            market_key TEXT NOT NULL,
            total NUMERIC(4,1) NULL,
            over_odds INT NULL,
            under_odds INT NULL,
            snapshot_ts TIMESTAMPTZ NULL,
            home_team TEXT NULL,
            away_team TEXT NULL
        )"""))

    for i in range(days):
        d = start + timedelta(days=i)
        ds = d.strftime("%Y-%m-%d")

        # Pull the day's games from EG to know teams & start times
        with eng.begin() as conn:
            games = conn.execute(text("""
                SELECT game_id, home_team, away_team, game_time_utc
                FROM enhanced_games
                WHERE "date" = :d
                ORDER BY game_time_utc NULLS LAST, game_id
            """), {"d": ds}).fetchall()

        if not games:
            print(f"{ds}: no games in enhanced_games; skipping")
            continue

        # We'll fetch one snapshot right before EACH game's first pitch.
        for g in games:
            gid, home, away, gts = g
            # If game_time_utc missing, take 23:59Z of that date (rough close proxy)
            if gts is None:
                snap_dt = datetime(d.year, d.month, d.day, 23, 59, tzinfo=timezone.utc)
            else:
                # "At or before first pitch": five minutes before first pitch to be safe
                snap_dt = gts - timedelta(minutes=5)
                if snap_dt.tzinfo is None:
                    snap_dt = snap_dt.replace(tzinfo=timezone.utc)

            iso_ts = snap_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            try:
                snap = fetch_snapshot(api_key, iso_ts)
            except requests.HTTPError as e:
                print(f"{ds} {gid}: HTTP {e.response.status_code} – {e.response.text[:200]}")
                continue
            except Exception as e:
                print(f"{ds} {gid}: snapshot fetch error: {e}")
                continue

            data = snap.get("data", [])
            # Match the event by team names
            match = None
            for rec in data:
                home_api = rec.get("home_team","")
                away_api = rec.get("away_team","")
                if same_team(home_api, home) and same_team(away_api, away):
                    match = rec
                    break
                if same_team(away_api, home) and same_team(home_api, away):
                    # safety net in case API flips fields
                    match = rec
                    break
            if not match:
                # Not found at this snapshot; try the 'previous_timestamp' snapshot if provided
                prev = snap.get("previous_timestamp")
                if prev:
                    try:
                        snap2 = fetch_snapshot(api_key, prev)
                        for rec in snap2.get("data", []):
                            home_api = rec.get("home_team","")
                            away_api = rec.get("away_team","")
                            if same_team(home_api, home) and same_team(away_api, away):
                                match = rec
                                snap = snap2
                                break
                    except Exception:
                        pass

            if not match:
                print(f"{ds} {gid}: no odds match for {away} @ {home} at snapshot {iso_ts}")
                time.sleep(args.sleep)
                continue

            tot = choose_totals(match, args.bookmaker)
            if not tot:
                print(f"{ds} {gid}: no totals market for {away} @ {home} ({args.bookmaker})")
                time.sleep(args.sleep)
                continue

            total, over_odds, under_odds, last_update = tot
            snap_ts = snap.get("timestamp")

            with eng.begin() as conn:
                # Insert a record into betting_lines
                conn.execute(text("""
                    INSERT INTO betting_lines (game_id, date, bookmaker, market_key, total, over_odds, under_odds, snapshot_ts, home_team, away_team)
                    VALUES (:game_id, :date, :book, 'totals', :total, :over, :under, :snapts, :home, :away)
                """), {
                    "game_id": gid, "date": d, "book": args.bookmaker,
                    "total": total, "over": over_odds, "under": under_odds,
                    "snapts": snap_ts, "home": home, "away": away
                })
                inserted += 1

                # Update enhanced_games to reflect the closing line for that game
                r = conn.execute(text("""
                    UPDATE enhanced_games
                       SET market_total = :total,
                           over_odds = :over,
                           under_odds = :under
                     WHERE game_id = :gid AND "date" = :date
                """), {"total": total, "over": over_odds, "under": under_odds, "gid": gid, "date": d})
                updated += r.rowcount

            print(f"{ds} {gid}: {away} @ {home} → total {total} (O:{over_odds}/U:{under_odds}) [{args.bookmaker}]")
            time.sleep(args.sleep)

    print(f"\nDone. betting_lines inserts: {inserted} | enhanced_games updated: {updated}")

if __name__ == "__main__":
    main()
