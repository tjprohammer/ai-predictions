#!/usr/bin/env python3
"""
Improved Market Totals Collector (ESPN + fallback)
- Uses ESPN odds for a specific date
- Falls back to DB-based estimates when odds are missing
- Updates enhanced_games.market_total
"""

import os, argparse
from datetime import datetime
import requests
import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')

def get_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True)

def espn_odds_for_date(target_date: str) -> dict:
    """
    Return dict keyed by (home_name, away_name) -> total (float)
    ESPN uses 'displayName' which matches your DB's team names.
    """
    yyyymmdd = target_date.replace('-', '')
    url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={yyyymmdd}"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[ESPN] Failed to fetch odds: {e}")
        return {}

    out = {}
    for ev in data.get("events", []):
        comps = ev.get("competitions") or []
        if not comps: 
            continue
        comp = comps[0]
        comps_list = comp.get("competitors") or []
        home, away = None, None
        for c in comps_list:
            name = (c.get("team") or {}).get("displayName")
            if not name: 
                continue
            if c.get("homeAway") == "home":
                home = name
            else:
                away = name
        if not home or not away:
            continue

        total = None
        for o in comp.get("odds") or []:
            ou = o.get("overUnder")
            if ou:
                try:
                    total = float(ou)
                    break
                except Exception:
                    pass
        if total is not None:
            out[(home, away)] = total
            print(f"[ESPN] {away} @ {home}: O/U {total}")
    print(f"[ESPN] Found odds for {len(out)} games on {target_date}")
    return out

def load_games_for_date(engine, target_date: str) -> pd.DataFrame:
    sql = text("""
        SELECT game_id, home_team, away_team, venue_name
        FROM enhanced_games
        WHERE "date" = :d
        ORDER BY game_id
    """)
    return pd.read_sql(sql, engine, params={"d": target_date})

def estimate_total(engine, target_date: str, home_team: str, away_team: str, venue_name: str | None) -> float:
    """
    Estimate using last-30 team scoring and last-60 venue totals (before target_date).
    """
    with engine.begin() as conn:
        # last 30 home games for this home team
        home_q = text("""
            SELECT AVG(home_team_runs::float) AS avg_runs
            FROM (
                SELECT home_team_runs
                FROM enhanced_games
                WHERE home_team = :home
                  AND total_runs IS NOT NULL
                  AND "date" < :d
                ORDER BY "date" DESC
                LIMIT 30
            ) s
        """)
        away_q = text("""
            SELECT AVG(away_team_runs::float) AS avg_runs
            FROM (
                SELECT away_team_runs
                FROM enhanced_games
                WHERE away_team = :away
                  AND total_runs IS NOT NULL
                  AND "date" < :d
                ORDER BY "date" DESC
                LIMIT 30
            ) s
        """)
        ven_q = text("""
            SELECT AVG((home_team_runs + away_team_runs)::float) AS avg_total
            FROM (
                SELECT home_team_runs, away_team_runs
                FROM enhanced_games
                WHERE (:venue IS NOT NULL AND venue_name = :venue)
                  AND total_runs IS NOT NULL
                  AND "date" < :d
                ORDER BY "date" DESC
                LIMIT 60
            ) s
        """)

        home_avg = conn.execute(home_q, {"home": home_team, "d": target_date}).scalar() or 4.5
        away_avg = conn.execute(away_q, {"away": away_team, "d": target_date}).scalar() or 4.5
        venue_avg = None
        if venue_name:
            venue_avg = conn.execute(ven_q, {"venue": venue_name, "d": target_date}).scalar()
        if venue_avg is None:
            venue_avg = 8.5

    est = 0.4 * (home_avg + away_avg) + 0.6 * venue_avg
    # round to the nearest 0.5
    return round(est * 2) / 2

def update_market_totals(engine, target_date: str, rows: list[dict]) -> int:
    """
    rows: [{game_id, market_total, source}, ...]
    """
    if not rows:
        return 0
    upd = text("""UPDATE enhanced_games
                  SET market_total = :total
                  WHERE game_id = :game_id""")
    n = 0
    with engine.begin() as conn:
        for r in rows:
            n += conn.execute(upd, {"total": r["market_total"], "game_id": r["game_id"]}).rowcount
    print(f"Updated market_total for {n} games on {target_date}")
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD (defaults to today)")
    ap.add_argument("--no-emoji", action="store_true")
    args = ap.parse_args()

    target_date = args.date or datetime.now().strftime("%Y-%m-%d")
    print(f"Collecting market totals for {target_date}")

    engine = get_engine()
    games = load_games_for_date(engine, target_date)
    if games.empty:
        print("No games found in enhanced_games for that date.")
        return

    espn = espn_odds_for_date(target_date)
    out_rows = []
    real_ct = 0
    est_ct = 0

    for _, g in games.iterrows():
        key = (g["home_team"], g["away_team"])
        if key in espn:
            mt = float(espn[key])
            real_ct += 1
            source = "espn"
        else:
            mt = estimate_total(engine, target_date, g["home_team"], g["away_team"], g.get("venue_name"))
            est_ct += 1
            source = "estimate"
        out_rows.append({"game_id": g["game_id"], "market_total": mt, "source": source})

    updated = update_market_totals(engine, target_date, out_rows)
    print(f"Done. Real odds: {real_ct} | Estimated: {est_ct} | Updated rows: {updated}")

if __name__ == "__main__":
    main()
