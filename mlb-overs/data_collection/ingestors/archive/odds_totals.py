# ingestors/markets_totals.py
from __future__ import annotations
import os, argparse, datetime as dt, requests, pandas as pd
from sqlalchemy import text
from ingestors.util import get_engine

SPORT = "baseball_mlb"   # The Odds API sport key

def team_key(name: str) -> str:
    s = (name or "").strip().lower()
    if "white sox" in s: return "white sox"
    if "red sox" in s: return "red sox"
    if "blue jays" in s: return "blue jays"
    toks = s.split()
    return toks[-1] if toks else s

def fetch_totals(date: str) -> pd.DataFrame:
    key = os.getenv("THE_ODDS_API_KEY")
    if not key:
        print("THE_ODDS_API_KEY not set"); return pd.DataFrame()

    url = (
        "https://api.the-odds-api.com/v4/sports/{sport}/odds"
        "?regions=us&markets=totals&oddsFormat=american&dateFormat=iso"
    ).format(sport=SPORT)

    r = requests.get(url, params={"apiKey": key}, timeout=20)
    r.raise_for_status()
    data = r.json()

    rows = []
    snap_ts = pd.Timestamp.utcnow()
    for ev in data:
        away = ev.get("away_team"); home = ev.get("home_team")
        comps = ev.get("bookmakers") or []
        for b in comps:
            book = b.get("key")
            for m in (b.get("markets") or []):
                if m.get("key") != "totals":
                    continue
                for o in (m.get("outcomes") or []):
                    pt = o.get("point")
                    if pt is None:
                        continue
                    rows.append({
                        "home_key": team_key(home),
                        "away_key": team_key(away),
                        "book": book,
                        "k_total": float(pt),
                        "snapshot_ts": snap_ts,
                        "date": pd.to_datetime(date).date(),
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Collapse per book per matchup per day: median helps smooth tiny diffs
    agg = (
        df.groupby(["home_key","away_key","book","date"], as_index=False)
          .agg(k_total=("k_total","median"),
               snapshot_ts=("snapshot_ts","max"))
    )
    return agg

def attach_game_ids(df: pd.DataFrame, eng) -> pd.DataFrame:
    if df.empty: return df
    games = pd.read_sql(
        text("""
            SELECT game_id, date, home_team, away_team
            FROM games
            WHERE date = :d
        """),
        eng, params={"d": df["date"].iloc[0]}
    )
    if games.empty:
        return pd.DataFrame()

    games = games.assign(
        home_key=games["home_team"].apply(team_key),
        away_key=games["away_team"].apply(team_key),
    )

    out = df.merge(
        games[["game_id","home_key","away_key","date"]],
        on=["home_key","away_key","date"], how="inner"
    )
    return out

def upsert_markets(df: pd.DataFrame, eng) -> int:
    """
    Insert snapshot rows into markets_totals. We rely on the unique index
    (game_id,date,book,market_type,COALESCE(snapshot_ts,'epoch')) to avoid dupes.
    """
    if df.empty: return 0
    df = df.copy()
    df["market_type"] = "snapshot"

    # in upsert_markets() of ingestors/odds_totals.py
    with eng.begin() as cx:
        df.to_sql("tmp_markets_totals", cx, if_exists="replace", index=False)
        cx.execute(text("""
            INSERT INTO markets_totals (game_id, date, book, market_type, k_total, snapshot_ts)
            SELECT game_id, date, book, market_type, k_total, snapshot_ts
            FROM tmp_markets_totals
            ON CONFLICT (game_id, date, book, market_type, snapshot_ts) DO NOTHING
        """))
        cx.execute(text("DROP TABLE tmp_markets_totals"))

        return len(df)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=dt.date.today().isoformat())
    args = ap.parse_args()

    eng = get_engine()
    df = fetch_totals(args.date)
    if df.empty:
        print("[odds_totals] no rows"); return
    df = attach_game_ids(df, eng)
    if df.empty:
        print("[odds_totals] no matching games for that date"); return
    n = upsert_markets(df, eng)
    print(f"[odds_totals] upserted snapshots: {n}")

if __name__ == "__main__":
    main()
