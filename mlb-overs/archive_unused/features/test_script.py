#!/usr/bin/env python3
import argparse, math
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import os

LEAGUE_ERA = 4.20

def build_engine():
    url = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
    return create_engine(url)

def latest_markets_date(eng):
    q = text("SELECT MAX(date) AS d FROM markets_totals")
    d = pd.read_sql(q, eng)["d"].iloc[0]
    return None if pd.isna(d) else pd.to_datetime(d).date()

def get_lines(eng, d):
    q = text("""
    WITH snaps AS (
      SELECT game_id, k_total AS total, snapshot_ts,
             ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY snapshot_ts DESC) rn
      FROM markets_totals
      WHERE date=:d AND market_type='snapshot'
    ),
    closes AS (
      SELECT game_id, close_total AS total, 1 AS rn
      FROM markets_totals
      WHERE date=:d AND market_type='close'
    )
    SELECT s.game_id, s.total, 'snapshot' AS src FROM snaps s WHERE s.rn=1
    UNION ALL
    SELECT c.game_id, c.total, 'close'    AS src FROM closes c
      WHERE c.game_id NOT IN (SELECT game_id FROM snaps)
    """)
    return pd.read_sql(q, eng, params={"d": d})

def starters_df(eng, cutoff_date):
    q = text("""
      SELECT pitcher_id::text AS pitcher_id, date, ip, er
      FROM pitchers_starts
      WHERE date < :d
    """)
    df = pd.read_sql(q, eng, params={"d": cutoff_date})
    if df.empty: return df
    df["ip"] = pd.to_numeric(df["ip"], errors="coerce")
    df["er"] = pd.to_numeric(df["er"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df

def era_from_rows(df):
    if df.empty: return None
    ip = df["ip"].sum(skipna=True)
    er = df["er"].sum(skipna=True)
    return float(er*9/ip) if ip and ip > 0 else None

def era_lastN(df, pid, N):
    sub = (df[df["pitcher_id"]==pid].sort_values("date").tail(N))
    return era_from_rows(sub)

def era_season(df, pid):
    sub = df[df["pitcher_id"]==pid]
    return era_from_rows(sub)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="YYYY-MM-DD (default: tomorrow if lines exist, else latest lines date)")
    args = ap.parse_args()

    eng = build_engine()

    # choose target date
    if args.date:
        target = pd.to_datetime(args.date).date()
    else:
        target = (datetime.utcnow() + timedelta(days=1)).date()

    lines = get_lines(eng, target)
    if lines.empty:
        # fall back to most recent date with lines
        fallback = latest_markets_date(eng)
        if fallback:
            print(f"No lines for {target}; falling back to latest lines date {fallback}")
            target = fallback
            lines = get_lines(eng, target)

    # games for target
    games = pd.read_sql(text("""
        SELECT game_id, date, home_team, away_team,
               home_sp_id::text AS home_pid,
               away_sp_id::text AS away_pid
        FROM games WHERE date=:d
        ORDER BY game_id
    """), eng, params={"d": target})

    if games.empty:
        print(f"No games found for {target}")
        return

    # attach line (k_close) if available
    if not lines.empty:
        lines["game_id"] = pd.to_numeric(lines["game_id"], errors="coerce").astype("Int64")
        games["game_id"] = pd.to_numeric(games["game_id"], errors="coerce").astype("Int64")
        games = games.merge(lines.rename(columns={"total":"k_close"}), on="game_id", how="left")
    else:
        games["k_close"] = pd.NA

    starts = starters_df(eng, target)

    print(f"\nFound {len(games)} games for {target}:")
    for _, r in games.iterrows():
        hpid, apid = r["home_pid"], r["away_pid"]
        hs = era_season(starts, hpid) if hpid else None
        as_ = era_season(starts, apid) if apid else None
        hl5 = era_lastN(starts, hpid, 5) if hpid else None
        al5 = era_lastN(starts, apid, 5) if apid else None

        # fallbacks to league if missing
        hs = hs if hs is not None else LEAGUE_ERA
        as_ = as_ if as_ is not None else LEAGUE_ERA
        hl5 = hl5 if hl5 is not None else hs
        al5 = al5 if al5 is not None else as_

        era_impact = (
            -0.25*(hs - LEAGUE_ERA)
            -0.25*(as_ - LEAGUE_ERA)
            -0.15*(hl5 - hs)
            -0.15*(al5 - as_)
        )
        adj_total = (float(r["k_close"]) + era_impact) if pd.notna(r["k_close"]) else None

        print(f"{r['away_team']} @ {r['home_team']}  "
              f"O/U: {r['k_close'] if pd.notna(r['k_close']) else '—'}")
        print(f"  Home SP ERA: season {hs:.2f}, L5 {hl5:.2f}")
        print(f"  Away SP ERA: season {as_:.2f}, L5 {al5:.2f}")
        print(f"  ERA impact: {era_impact:+.2f}"
              + (f"  → adjusted total: {adj_total:.1f}" if adj_total is not None else " (no market line)"))

if __name__ == "__main__":
    main()
