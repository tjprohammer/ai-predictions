from __future__ import annotations
import os, datetime as dt
import pandas as pd
from sqlalchemy import create_engine, text
import statsapi
from tqdm import tqdm


DB_URL = os.environ.get('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
eng = create_engine(DB_URL)

def innings_to_float(ip) -> float | None:
    if ip is None: return None
    s = str(ip).strip()
    if not s: return None
    try:
        whole, dot, outs = s.partition('.')
        w = int(whole)
        o = int(outs) if outs else 0
        if o not in (0,1,2): o = 0
        return float(w + o/3.0)
    except Exception:
        return None

def team_key(team_id: int) -> str:
    try:
        rec = (statsapi.lookup_team(team_id) or [])[0]
        return rec.get('teamName') or rec.get('name') or str(team_id)
    except Exception:
        return str(team_id)

def extract_ip_er(pitching: dict) -> tuple[float|None, float|None]:
    # primary keys
    ip = innings_to_float(pitching.get("inningsPitched"))
    er = pitching.get("earnedRuns")
    # alternates that occasionally show up
    if er is None:
        er = pitching.get("er") or pitching.get("earned")
    # normalize er to float
    try:
        er = None if er is None else float(er)
    except Exception:
        er = None
    return ip, er

def fetch_teams_any(game_pk: int) -> tuple[dict, dict | None]:
    """
    Return (teams_dict, live) where teams_dict looks like:
    {"home": {"team": {...}, "players": {...}}, "away": {...}}
    live is the liveFeed payload if we needed it (else None).
    """
    # try 1: boxscore_data
    try:
        box = statsapi.boxscore_data(game_pk) or {}
        teams = box.get("teams") or {}
        if teams:
            return teams, None
    except Exception:
        pass

    # try 2: raw boxscore
    try:
        raw = statsapi.get("game_boxscore", {"gamePk": game_pk}) or {}
        teams = raw.get("teams") or {}
        if teams:
            return teams, None
    except Exception:
        pass

    # try 3: liveFeed (most reliable)
    live = None
    try:
        live = statsapi.get("game", {"gamePk": game_pk}) or {}
        live_teams = (((live.get("liveData") or {}).get("boxscore") or {}).get("teams") or {})
        if live_teams:
            # synthesize the same structure: include players dicts
            # team ids/names live under gameData.teams
            gmeta = statsapi.get("game", {"gamePk": game_pk}) or {}
            game_teams = ((gmeta.get("gameData") or {}).get("teams") or {})
            return {
                "home": {
                    "team": game_teams.get("home", {}),
                    "players": (live_teams.get("home", {}) or {}).get("players", {}) or {}
                },
                "away": {
                    "team": game_teams.get("away", {}),
                    "players": (live_teams.get("away", {}) or {}).get("players", {}) or {}
                }
            }, live
    except Exception:
        pass

    return {}, None


def backfill_pitcher_lines(start_date: str, end_date: str):
    with eng.begin() as cx:
        games = pd.read_sql(text("""
            SELECT game_id, date, home_team, away_team
            FROM games
            WHERE date BETWEEN :s AND :e
            ORDER BY date
        """), cx, params={"s": start_date, "e": end_date})

    if games.empty:
        print("no games in window")
        return

    total_boxes = 0
    empty_boxes = 0
    no_pitch_stats = 0
    rows = []

    pbar = tqdm(total=len(games), desc="Backfilling pitcher lines", unit="game")


    for _, g in games.iterrows():
        gid_str = str(g["game_id"])
        try:
            gid = int(gid_str)
        except Exception:
            continue

        # optional: skip non-final games during historical backfill
        try:
            gmeta = statsapi.get("game", {"gamePk": gid}) or {}
            status = ((gmeta.get("gameData") or {}).get("status") or {}).get("abstractGameState")
            if status not in ("Final", "Completed Early", "Game Over"):
                continue
        except Exception:
            pass

        teams, live = fetch_teams_any(gid)
        total_boxes += 1
        if not teams:
            empty_boxes += 1
            pbar.update(1); pbar.set_postfix(boxes=total_boxes, empty=empty_boxes,
                                             no_pitch=no_pitch_stats, rows=len(rows))
            continue

        added_any_for_game = False

        for side in ("home","away"):
            t = teams.get(side) or {}
            players = t.get("players") or {}
            team_id = (t.get("team") or {}).get("id")
            opp_side = "home" if side == "away" else "away"
            opp_team_id = (teams.get(opp_side) or {}).get("team", {}).get("id")

            for pid_key, pdata in players.items():
                pos = (pdata.get("position") or {}).get("abbreviation")
                if pos != "P":
                    continue

                # 1) primary stats
                pitching = (pdata.get("stats") or {}).get("pitching") or {}
                ip_f, er_v = extract_ip_er(pitching)

                # 2) raw boxscore fallback
                if ip_f is None and er_v is None:
                    try:
                        raw = statsapi.get("boxscore", {"gamePk": gid}) or {}
                        p2 = (raw.get("teams", {}).get(side, {}).get("players", {}) or {}).get(pid_key, {})
                        pitching2 = (p2.get("stats") or {}).get("pitching") or {}
                        ip_f, er_v = extract_ip_er(pitching2)
                    except Exception:
                        pass

                # 3) liveFeed per-player fallback
                if ip_f is None and er_v is None and live is not None:
                    try:
                        live_player = ((((live.get("liveData") or {})
                                        .get("boxscore") or {})
                                        .get("teams") or {})
                                        .get(side, {}) or {}) \
                                        .get("players", {}) \
                                        .get(pid_key, {}) or {}
                        pitching3 = (live_player.get("stats") or {}).get("pitching") or {}
                        ip_f, er_v = extract_ip_er(pitching3)
                    except Exception:
                        pass

                if ip_f is None and er_v is None:
                    continue

                era_game = float(9.0 * er_v / ip_f) if (ip_f and ip_f > 0 and er_v is not None) else None
                pid = (pdata.get("person") or {}).get("id")

                rows.append({
                    "start_id": f"{gid}_{pid}",
                    "game_id": gid,
                    "pitcher_id": pid,
                    "team": team_key(team_id) if team_id else None,
                    "opp_team": team_key(opp_team_id) if opp_team_id else None,
                    "is_home": (side == "home"),
                    "date": pd.to_datetime(g["date"]).date(),
                    "ip": ip_f,
                    "er": er_v,
                    "era_game": era_game,
                })
                added_any_for_game = True

        if not added_any_for_game:
            no_pitch_stats += 1


        pbar.update(1)
        pbar.set_postfix(boxes=total_boxes, empty=empty_boxes,
                         no_pitch=no_pitch_stats, rows=len(rows))

    pbar.close()

    if not rows:
        print(f"no pitcher rows to upsert (boxes fetched={total_boxes}, empty_boxes={empty_boxes}, games_with_no_pitch_stats={no_pitch_stats})")
        return

    df = pd.DataFrame(rows).drop_duplicates(["start_id"])

    with eng.begin() as cx:
        df.to_sql("tmp_ps_up", cx, index=False, if_exists="replace")
        cx.execute(text("""
            INSERT INTO pitchers_starts (start_id, game_id, pitcher_id, team, opp_team, is_home, date, ip, er, era_game)
            SELECT start_id, game_id, CAST(pitcher_id AS TEXT), team, opp_team, is_home, date, ip, er, era_game
            FROM tmp_ps_up
            ON CONFLICT (start_id) DO UPDATE SET
              team = EXCLUDED.team,
              opp_team = EXCLUDED.opp_team,
              is_home = EXCLUDED.is_home,
              ip = COALESCE(EXCLUDED.ip, pitchers_starts.ip),
              er = COALESCE(EXCLUDED.er, pitchers_starts.er),
              era_game = COALESCE(EXCLUDED.era_game, pitchers_starts.era_game)
        """))
        cx.execute(text("DROP TABLE tmp_ps_up"))
    print(f"upserted/updated {len(df)} pitcher lines")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()
    backfill_pitcher_lines(args.start, args.end)
