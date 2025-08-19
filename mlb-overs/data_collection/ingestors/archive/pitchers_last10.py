# ingestors/pitchers_last10.py
from __future__ import annotations
import argparse, datetime as dt
import pandas as pd
from sqlalchemy import text
from pybaseball import statcast_pitcher, cache
import statsapi
from ingestors.util import get_engine, upsert_df

cache.enable()


def innings_to_float(ip) -> float | None:
    """Convert innings pitched string to float"""
    if ip is None: 
        return None
    s = str(ip).strip()
    if not s: 
        return None
    # handle formats like "5.1" (5 and 1/3 innings)
    if "." in s:
        try:
            whole, frac = s.split(".")
            return float(whole) + float(frac) / 3.0
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None


def get_pitcher_game_stats(pitcher_id: int, game_date) -> dict:
    """Get detailed pitcher stats for a specific game date"""
    try:
        # Use statsapi to get pitcher game logs - don't pass season for gameLog type
        stats = statsapi.player_stat_data(pitcher_id, group="[pitching]", type="gameLog")
        
        if not stats or 'stats' not in stats:
            return {}
            
        # Find the game matching our date
        target_date = pd.to_datetime(game_date).strftime('%Y-%m-%d')
        
        for stat_group in stats['stats']:
            if 'splits' not in stat_group:
                continue
                
            for game in stat_group['splits']:
                game_date_str = game.get('date', '')
                if game_date_str == target_date:
                    stats_data = game.get('stat', {})
                    
                    # Extract key stats
                    ip_str = stats_data.get('inningsPitched', '0')
                    er = int(stats_data.get('earnedRuns', 0))
                    ip_float = innings_to_float(ip_str)
                    
                    return {
                        'ip': ip_float,
                        'er': er,
                        'h': int(stats_data.get('hits', 0)),
                        'bb': int(stats_data.get('baseOnBalls', 0)),
                        'k': int(stats_data.get('strikeOuts', 0)),
                        'hr': int(stats_data.get('homeRuns', 0)),
                        'r': int(stats_data.get('runs', 0)),
                        'bf': int(stats_data.get('battersFaced', 0)),
                        'pitches': int(stats_data.get('numberOfPitches', 0)),
                        'era_game': (er * 9.0 / ip_float) if ip_float and ip_float > 0 else None,
                    }
        return {}
    except Exception as e:
        # Silently return empty dict to avoid spam - many games won't have detailed stats
        return {}


def _lookup_player_id_by_name(name: str) -> int | None:
    try:
        if not name:
            return None
        res = statsapi.lookup_player(name)
        if not res:
            return None
        # prefer exact case-insensitive match
        name_lower = name.strip().lower()
        for r in res:
            full = (r.get('fullName') or r.get('full_name') or '').strip().lower()
            if full == name_lower:
                return int(r.get('id') or r.get('player_id'))
        # else first result
        return int(res[0].get('id') or res[0].get('player_id'))
    except Exception:
        return None


def _pid_from_sched_game(g, side: str):
    # many possible shapes for probables in schedule JSON
    # 1) direct numeric ids
    pid = g.get(f"{side}_pitcher_id")
    if pid:
        return pid
    pid = g.get(f"{side}_probable_pitcher_id")
    if pid:
        return pid

    # 2) probablePitchers sub-dict
    pp = g.get("probablePitchers") or {}
    if isinstance(pp, dict):
        s = pp.get(side) or {}
        pid = s.get("id") or s.get("playerId") or s.get("player_id")
        if pid:
            return pid
        # sometimes just a name here
        name = s.get('fullName') or s.get('full_name') or s.get('name')
        if name:
            pid = _lookup_player_id_by_name(name)
            if pid:
                return pid

    # 3) side_probable_pitcher often a string name or dict with id
    sp = g.get(f"{side}_probable_pitcher")
    if isinstance(sp, dict):
        pid = sp.get('id') or sp.get('player_id')
        if pid:
            return pid
        name = sp.get('fullName') or sp.get('full_name') or sp.get('name')
        if name:
            pid = _lookup_player_id_by_name(name)
            if pid:
                return pid
    elif isinstance(sp, str) and sp.strip():
        pid = _lookup_player_id_by_name(sp)
        if pid:
            return pid

    return None


def _pid_from_boxscore(game_pk: int, side: str):
    """
    For live/started/final games, identify the actual starter:
      - first pitcher with gamesStarted > 0, else
      - pitcher with max innings pitched
    """
    try:
        box = statsapi.boxscore_data(game_pk)
    except Exception:
        return None
    side_data = box.get(side) or {}
    players = (side_data.get("players") or {})
    if not players:
        return None

    pitchers = []
    for _, pdata in players.items():
        pstats = (pdata.get("stats") or {}).get("pitching") or {}
        if not pstats:
            continue
        pid = (pdata.get("person") or {}).get("id")
        if not pid:
            continue
        # inningsPitched like "1.2" -> float-ish
        ip_raw = pstats.get("inningsPitched")
        try:
            ip_val = float(ip_raw) if ip_raw not in (None, "") else 0.0
        except Exception:
            ip_val = 0.0
        pitchers.append({
            "pid": int(pid),
            "isGS": int(pstats.get("gamesStarted") or 0),
            "ip": ip_val,
        })

    if not pitchers:
        return None
    # Prefer true starter, else max IP
    starters = [p for p in pitchers if p["isGS"] > 0]
    if starters:
        return starters[0]["pid"]
    pitchers.sort(key=lambda r: r["ip"], reverse=True)
    return pitchers[0]["pid"]


def probable_ids_for_range(eng, start: dt.date, end: dt.date) -> pd.DataFrame:
    # Base rows from DB to keep game_id/date (ok if SPs are null)
    df = pd.read_sql(
        text("SELECT game_id, date, home_sp_id, away_sp_id FROM games WHERE date BETWEEN :s AND :e"),
        eng, params={"s": start, "e": end}
    )
    # If DB has no games, synthesize from schedule
    if df.empty:
        rows = []
        d = start
        while d <= end:
            for g in (statsapi.schedule(date=d.strftime("%m/%d/%Y")) or []):
                rows.append({"game_id": int(g.get("gamePk") or g.get("game_id")), "date": d,
                             "home_sp_id": None, "away_sp_id": None})
            d += dt.timedelta(days=1)
        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["game_id","date","home_sp_id","away_sp_id"])

    # Overlay probables from schedule + fallback to boxscore starters
    need_dates = sorted(pd.Series(df["date"]).dropna().unique())
    sched_map = {}
    for d in need_dates:
        for g in (statsapi.schedule(date=pd.to_datetime(d).strftime("%m/%d/%Y")) or []):
            gid = str(g.get("gamePk") or g.get("game_id"))
            sched_map[gid] = {
                "home": _pid_from_sched_game(g, "home"),
                "away": _pid_from_sched_game(g, "away"),
            }

    def resolve_pid(row, side):
        # 1) DB value if present
        dbv = row[f"{side}_sp_id"]
        if pd.notna(dbv):
            try: return int(dbv)
            except: pass
        gid = str(row["game_id"])
        # 2) schedule probable
        sched_pid = (sched_map.get(gid) or {}).get("home" if side=="home" else "away")
        if sched_pid:
            try: return int(sched_pid)
            except: pass
        # 3) boxscore (actual starter) if game is underway or final
        try:
            return _pid_from_boxscore(int(row["game_id"]), side)
        except Exception:
            return None

    df["home_pid"] = df.apply(lambda r: resolve_pid(r, "home"), axis=1)
    df["away_pid"] = df.apply(lambda r: resolve_pid(r, "away"), axis=1)
    return df[["game_id","date","home_pid","away_pid"]]


def statcast_last_120_days(pid: int, up_to_date: dt.date) -> pd.DataFrame:
    start_dt = (pd.to_datetime(up_to_date) - pd.Timedelta(days=120)).date().strftime("%Y-%m-%d")
    end_dt   = up_to_date.strftime("%Y-%m-%d")
    try:
        sc = statcast_pitcher(start_dt=start_dt, end_dt=end_dt, player_id=int(pid))
    except Exception:
        return pd.DataFrame()
    if sc is None or sc.empty: return pd.DataFrame()
    sc = sc.copy()
    sc["game_date"] = pd.to_datetime(sc["game_date"]).dt.date
    g = sc.groupby("game_date", as_index=False).agg(
        pitches=("pitch_number","count"),
        csw=("description", lambda s: (s.astype(str).str.lower()
               .isin({"called_strike","swinging_strike","swinging_strike_blocked"})).sum()),
        xwoba_allowed=("estimated_woba_using_speedangle","mean"),
        xslg_allowed=("estimated_slg_using_speedangle","mean"),
        avg_ev_allowed=("launch_speed","mean"),
        velo_fb=("release_speed", "mean"),
    )
    if g.empty: return g
    g["csw_pct"] = g["csw"] / g["pitches"]
    g["pitcher_id"] = int(pid)
    return g.sort_values("game_date")


def build_last10_rows(prob: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in prob.iterrows():
        game_day = r["date"]
        for pid in [r.get("home_pid"), r.get("away_pid")]:
            if pid is None or pd.isna(pid):
                continue
            starts = statcast_last_120_days(int(pid), game_day)
            if starts.empty:
                continue
            last10 = starts[starts["game_date"] < game_day].tail(10)
            if last10.empty:
                continue
            for _, s in last10.iterrows():
                game_date = s["game_date"]
                
                rows.append({
                    "start_id": f"{int(pid)}_{game_date}",
                    "game_id": None,
                    "pitcher_id": int(pid),
                    "team": None, "opp_team": None, "is_home": None,
                    "date": game_date,
                    "ip": None, "h": None, "bb": None, "k": None, "hr": None,
                    "r": None, "er": None, "bf": None,
                    "pitches": int(s.get("pitches") or 0),
                    "csw_pct": float(s.get("csw_pct") or 0.0),
                    "velo_fb": float(s.get("velo_fb") or 0.0),
                    "velo_delta_3g": None,
                    "hh_pct_allowed": None, "barrel_pct_allowed": None,
                    "avg_ev_allowed": float(s.get("avg_ev_allowed") or 0.0),
                    "xwoba_allowed": float(s.get("xwoba_allowed") or 0.0),
                    "xslg_allowed": float(s.get("xslg_allowed") or 0.0),
                    "era_game": None, "fip_game": None, "xfip_game": None, "siera_game": None,
                    "opp_lineup_l_pct": None, "opp_lineup_r_pct": None,
                    "days_rest": None, "tto": None,
                    "pitch_count_prev1": None, "pitch_count_prev2": None,
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()

    eng = get_engine()
    s, e = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    prob = probable_ids_for_range(eng, s, e)
    if prob.empty:
        print("No probables found."); return

    df = build_last10_rows(prob)
    if df.empty:
        print("No pitcher starts found."); return

    # --- cleanup to match DB types ---
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    if "is_home" in df.columns:
        df["is_home"] = pd.Series([None] * len(df), dtype="boolean")
    num_cols = [
        "ip","h","bb","k","hr","r","er","bf","pitches","csw_pct","velo_fb",
        "velo_delta_3g","hh_pct_allowed","barrel_pct_allowed","avg_ev_allowed",
        "xwoba_allowed","xslg_allowed","era_game","fip_game","xfip_game","siera_game",
        "opp_lineup_l_pct","opp_lineup_r_pct","days_rest","tto",
        "pitch_count_prev1","pitch_count_prev2",
    ]
    for c in [c for c in num_cols if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n = upsert_df(df, "pitchers_starts", pk=["start_id"])
    print("upsert pitchers_starts:", n)

if __name__ == "__main__":
    main()
