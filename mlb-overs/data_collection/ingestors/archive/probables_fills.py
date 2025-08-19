# ingestors/probables_fill.py
from __future__ import annotations
import argparse
import pandas as pd
import statsapi
from sqlalchemy import text
from ingestors.util import get_engine

def nick(name: str) -> str:
    s = (name or "").lower().strip()
    if "white sox" in s: return "white sox"
    if "red sox"   in s: return "red sox"
    if "blue jays" in s: return "blue jays"
    toks = s.split()
    return toks[-1] if toks else s

def _to_int(x):
    try:
        return int(x) if x is not None and str(x).strip() != "" else None
    except:
        return None

def _lookup_id_from_name(name: str) -> int | None:
    if not name:
        return None
    try:
        # statsapi.lookup_player returns list of dicts
        res = statsapi.lookup_player(name) or []
        if not res:
            return None
        pid = res[0].get("id") or res[0].get("playerid") or res[0].get("mlb_id")
        return _to_int(pid)
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    args = ap.parse_args()
    d = pd.to_datetime(args.date).date()

    eng = get_engine()
    games = pd.read_sql(
        text("SELECT game_id, home_team, away_team FROM games WHERE date=:d"),
        eng, params={"d": d}
    )
    if games.empty:
        print(f"[probables_fill] no games for {d}")
        return

    # MLB schedule for that date
    sched = statsapi.schedule(start_date=d.isoformat(), end_date=d.isoformat()) or []

    rows = []
    found_by_id = 0
    found_by_name = 0

    for g in sched:
        # team names
        home_name = g.get('home_name') or ((g.get('teams') or {}).get('home') or {}).get('team',{}).get('name')
        away_name = g.get('away_name') or ((g.get('teams') or {}).get('away') or {}).get('team',{}).get('name')
        if not home_name or not away_name:
            continue
        hn, an = nick(home_name), nick(away_name)

        # try IDs first
        h_id = _to_int(g.get('home_probable_pitcher_id'))
        a_id = _to_int(g.get('away_probable_pitcher_id'))

        # if IDs missing, try resolving from probable pitcher names
        if h_id is None:
            h_name = g.get('home_probable_pitcher')
            h_id = _lookup_id_from_name(h_name) if h_name else None
            if h_id: found_by_name += 1
        else:
            found_by_id += 1

        if a_id is None:
            a_name = g.get('away_probable_pitcher')
            a_id = _lookup_id_from_name(a_name) if a_name else None
            if a_id: found_by_name += 1
        else:
            found_by_id += 1

        rows.append({"home_key": hn, "away_key": an, "home_sp_id": h_id, "away_sp_id": a_id})

    df = pd.DataFrame(rows)
    if df.empty:
        print("[probables_fill] schedule returned no matchable rows")
        return

    # join to games by nickname
    games = games.assign(home_key=games['home_team'].apply(nick),
                         away_key=games['away_team'].apply(nick))
    m = games.merge(df, on=['home_key','away_key'], how='left')

    # upsert only where we actually found values
    updated = 0
    with eng.begin() as cx:
        for _, r in m.iterrows():
            h = r.get('home_sp_id')
            a = r.get('away_sp_id')
            if pd.notna(h) or pd.notna(a):
                cx.execute(text("""
                    UPDATE games
                       SET home_sp_id = COALESCE(:h, home_sp_id),
                           away_sp_id = COALESCE(:a, away_sp_id)
                     WHERE game_id = :gid
                """), {"h": int(h) if pd.notna(h) else None,
                       "a": int(a) if pd.notna(a) else None,
                       "gid": r['game_id']})
                updated += 1

    print(f"[probables_fill] updated rows: {updated} (found_by_id={found_by_id}, found_by_name={found_by_name})")

if __name__ == "__main__":
    main()
