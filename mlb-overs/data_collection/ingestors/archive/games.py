# ingestors/games.py
from __future__ import annotations
import argparse, datetime as dt
import pandas as pd
import statsapi
from sqlalchemy import inspect, text
from util import get_engine, upsert_df

def probable_pitchers_for(date: dt.date) -> pd.DataFrame:
    # returns DataFrame(game_id, home_sp_id, away_sp_id)
    sched = statsapi.schedule(start_date=date.isoformat(), end_date=date.isoformat())
    rows = []
    for g in sched:
        gid = g.get("game_id") or g.get("gamePk")  # depending on statsapi version
        if not gid:
            continue
        home = g.get("home_name")
        away = g.get("away_name")
        # statsapi keys commonly: probablePitcher* or prob* fields
        home_sp = (g.get("home_probable_pitcher_id") or g.get("home_probable_pitcher"))
        away_sp = (g.get("away_probable_pitcher_id") or g.get("away_probable_pitcher"))
        # normalize to ints or None
        home_sp = int(home_sp) if pd.notna(home_sp) and str(home_sp).isdigit() else None
        away_sp = int(away_sp) if pd.notna(away_sp) and str(away_sp).isdigit() else None
        rows.append({"game_id": int(gid), "home_sp_id": home_sp, "away_sp_id": away_sp})
    return pd.DataFrame(rows)

def upsert_probables(eng, date: dt.date):
    df = probable_pitchers_for(date)
    if df.empty:
        return 0
    with eng.begin() as cx:
        df.to_sql("tmp_probables", cx, if_exists="replace", index=False)
        cx.execute(text("""
            UPDATE games g
            SET home_sp_id = COALESCE(g.home_sp_id, t.home_sp_id),
                away_sp_id = COALESCE(g.away_sp_id, t.away_sp_id)
            FROM tmp_probables t
            WHERE g.game_id::bigint = t.game_id AND g.date = :d
        """), {"d": date})
        cx.execute(text("DROP TABLE tmp_probables"))
    return len(df)


def table_columns(eng, table: str) -> set[str]:
    insp = inspect(eng)
    return {c["name"] if isinstance(c, dict) else c.name for c in insp.get_columns(table)}  # SA 2.x/1.4 safety


def _date_iter(s: dt.date, e: dt.date):
    d = s
    while d <= e:
        yield d
        d += dt.timedelta(days=1)

def fetch_schedule(start: str, end: str) -> pd.DataFrame:
    s, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    rows = []
    for d in _date_iter(s, e):
        sched = statsapi.schedule(date=d.strftime("%m/%d/%Y")) or []
        for g in sched:
            gid = int(g.get("gamePk") or g.get("game_id"))
            rows.append({
                "game_id": gid,
                "date": d,
                "status": (g.get("status") or "").lower(),
                "home_team_id": g.get("home_id"),
                "away_team_id": g.get("away_id"),
                "home_team": g.get("home_name"),
                "away_team": g.get("away_name"),
                "home_sp_id": g.get("home_pitcher_id"),
                "away_sp_id": g.get("away_pitcher_id"),
                "venue_id": g.get("venue_id"),
                "venue_name": g.get("venue_name"),
                # placeholders you can expand later
                "umpire_hp": None,
                "total_runs": None,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # clean types
    for c in ["home_sp_id","away_sp_id","home_team_id","away_team_id","venue_id"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()

    df = fetch_schedule(args.start, args.end)
    if df.empty:
        print("no games returned"); return

    eng = get_engine()

    # If total_runs exists, make it numeric (or drop it if your table doesn't have it)
    if "total_runs" in df.columns:
        df["total_runs"] = pd.to_numeric(df["total_runs"], errors="coerce").astype("Int64")

    # Keep only columns that exist in the DB schema
    cols = table_columns(eng, "games")
    df = df[[c for c in df.columns if c in cols]]

    n = upsert_df(df, "games", pk=["game_id"])
    print(f"upsert games: {n}")

if __name__ == "__main__":
    main()
