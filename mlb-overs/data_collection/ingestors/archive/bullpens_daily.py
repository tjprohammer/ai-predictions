# ingestors/bullpens_daily.py
from __future__ import annotations
"""
Bullpens daily ingestor
- Builds team-day bullpen form + availability from MLB StatsAPI box scores
- Upserts into `bullpens_daily` (PK: team,date)

Adds:
- Correct IP math for 'inningsPitched' (e.g., "1.2" = 1 + 2/3)
- relief_ip, relief_pitches, relievers_used (per day)
- relief_pitches_d1 (yesterday)
- any_b2b_reliever (someone pitched on d-1 and d-2)

Also exposes `bullpen_recent_usage()` to produce last-3-day usage features.
"""

import argparse
import datetime as dt
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
import statsapi

from .util import get_engine, upsert_df


# ---------- utilities ----------

def _date_iter(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def ip_str_to_float(ip_str: str | None) -> float:
    """
    Convert MLB 'inningsPitched' string like "1.2" to decimal innings:
      "1.0" -> 1.0
      "1.1" -> 1 + 1/3  = 1.3333
      "1.2" -> 1 + 2/3  = 1.6667
    """
    if not ip_str:
        return 0.0
    s = str(ip_str).strip()
    if "." not in s:
        try:
            return float(int(s))
        except Exception:
            return 0.0
    whole, frac = s.split(".", 1)
    try:
        w = int(whole)
    except Exception:
        w = 0
    outs = 0
    if frac == "1":
        outs = 1
    elif frac == "2":
        outs = 2
    return w + outs / 3.0


# ---------- boxscore parsing ----------

def _team_relief_stats_for_game(game_pk: int) -> List[Dict]:
    """
    Return one record per side with relief details:
      [{
        "team_id": <int>, "team": <name>,
        "relievers": [ {player_id,..., ip, er, hr, bb, k, pitches}, ... ],
        "relief_ip": <float>, "relief_pitches": <int>, "reliever_ids": [ ... ],
      }, ...]
    """
    try:
        box = statsapi.boxscore_data(game_pk)
    except Exception:
        return []

    out: List[Dict] = []
    for side in ("home", "away"):
        side_data = box.get(side) or {}
        team_meta = (side_data.get("team") or {})
        team_id = int(team_meta.get("id") or 0)
        team_name = (
            team_meta.get("teamName")
            or team_meta.get("name")
            or team_meta.get("team_name")
            or team_meta.get("abbreviation")
            or (str(team_id) if team_id else "")
        )
        if not (team_id or team_name):
            continue

        players = (side_data.get("players") or {})
        if not players:
            continue

        pitchers = []
        for _, pdata in players.items():
            pstats = (pdata.get("stats") or {}).get("pitching") or {}
            if not pstats:
                continue
            ip_val = ip_str_to_float(pstats.get("inningsPitched"))
            pid = int((pdata.get("person") or {}).get("id") or 0)
            pitchers.append({
                "player_id": pid,
                "isGS": int(pstats.get("gamesStarted") or 0),
                "ip": float(ip_val),
                "er": int(pstats.get("earnedRuns") or 0),
                "hr": int(pstats.get("homeRuns") or 0),
                "bb": int(pstats.get("baseOnBalls") or 0) + int(pstats.get("intentionalWalks") or 0),
                "k": int(pstats.get("strikeOuts") or 0),
                "pitches": int(pstats.get("numberOfPitches") or 0),
            })

        if not pitchers:
            continue

        # Identify starter: GS>0 else max IP
        pitchers.sort(key=lambda r: (r["isGS"], r["ip"]), reverse=True)
        relievers = pitchers[1:] if len(pitchers) > 1 else []

        out.append({
            "team_id": team_id,
            "team": team_name,
            "relievers": relievers,
            "relief_ip": sum(r["ip"] for r in relievers),
            "relief_pitches": sum(r["pitches"] for r in relievers),
            "reliever_ids": [r["player_id"] for r in relievers],
        })
    return out


# ---------- daily builder ----------

def build_bullpen_daily(start: str, end: str) -> pd.DataFrame:
    s, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    rows = []

    # Track who pitched & how many pitches per day for b2b & yesterday volume
    pitched_map = defaultdict(lambda: defaultdict(set))   # pitched_map[team_id][date] -> {player_ids}
    pitches_map = defaultdict(lambda: defaultdict(int))   # pitches_map[team_id][date] -> total relief pitches

    for d in _date_iter(s, e):
        sched = statsapi.schedule(date=d.strftime('%m/%d/%Y')) or []
        team_agg = defaultdict(lambda: {"er":0, "ip":0.0, "hr":0, "bb":0, "k":0, "relief_ip":0.0, "relief_pitches":0, "team":None})

        for g in sched:
            game_pk = g.get("gamePk") or g.get("game_id")
            if not game_pk:
                continue
            game_pk = int(game_pk)
            for t in _team_relief_stats_for_game(game_pk):
                team_id = t["team_id"]; team_name = t["team"]
                relievers = t["relievers"]
                agg = team_agg[team_id]
                agg["team"] = team_name
                for r in relievers:
                    agg["er"] += r["er"]
                    agg["ip"] += r["ip"]
                    agg["hr"] += r["hr"]
                    agg["bb"] += r["bb"]
                    agg["k"]  += r["k"]
                agg["relief_ip"]      += t["relief_ip"]
                agg["relief_pitches"] += t["relief_pitches"]
                pitched_map[team_id][d].update(t["reliever_ids"])
                pitches_map[team_id][d] += t["relief_pitches"]

        # finalize rows per team/day
        for team_id, agg in team_agg.items():
            team_name = agg["team"] or str(team_id)
            ip = max(0.0, agg["ip"])
            er, hr, bb, k = agg["er"], agg["hr"], agg["bb"], agg["k"]
            era  = (er * 9.0) / ip if ip > 0 else None
            fip  = ((13*hr) + (3*bb) - (2*k)) / ip if ip > 0 else None
            kbb_pct = (k - bb) / max(1, (k + bb))
            hr9  = (hr * 9.0) / ip if ip > 0 else None

            d1 = d - dt.timedelta(days=1)
            d2 = d - dt.timedelta(days=2)

            relievers_y = pitched_map[team_id].get(d1, set())
            relievers_d2 = pitched_map[team_id].get(d2, set())
            total_pitches_y = pitches_map[team_id].get(d1, 0)

            # any reliever who pitched on both d-1 and d-2
            any_b2b = bool(relievers_y & relievers_d2)

            rows.append({
                "team": team_name,               # keep name as key for merges you already use
                "date": d,
                "bp_era": era,
                "bp_fip": fip,
                "bp_kbb_pct": kbb_pct,
                "bp_hr9": hr9,
                "relief_ip": agg["relief_ip"],
                "relief_pitches": agg["relief_pitches"],
                "relievers_used": len(pitched_map[team_id].get(d, set())),
                "relief_pitches_d1": total_pitches_y,
                "any_b2b_reliever": any_b2b,
            })

    return pd.DataFrame(rows)


# ---------- feature helper (for your features script) ----------

def bullpen_recent_usage(eng, games_df: pd.DataFrame, window_days: int = 3) -> pd.DataFrame:
    """
    Returns one row per game_id with last-N-day bullpen usage:
      home_bp_ip_{N}d, home_bp_pitches_{N}d, home_bp_b2b_flag, home_bp_fatigued
      away_...
    """
    if games_df.empty:
        return pd.DataFrame()

    gmin = pd.to_datetime(games_df["date"]).min()
    need_start = (gmin - pd.Timedelta(days=window_days+5)).date()  # small buffer

    bp = pd.read_sql("""
        SELECT team, date, relief_ip, relief_pitches, any_b2b_reliever
        FROM bullpens_daily
        WHERE date >= %(s)s
    """, eng, params={"s": need_start})
    if bp.empty:
        return pd.DataFrame()
    bp["date"] = pd.to_datetime(bp["date"])

    by_team = {t: df.sort_values("date") for t, df in bp.groupby("team", as_index=False)}

    rows = []
    for _, r in games_df.iterrows():
        gid = r["game_id"]; gd = pd.to_datetime(r["date"])
        for side, team_col, prefix in [("home","home_team","home_bp"), ("away","away_team","away_bp")]:
            team = r.get(team_col)
            if not isinstance(team, str):
                continue
            tdf = by_team.get(team)
            if tdf is None:
                continue
            win_start = gd - pd.Timedelta(days=window_days)
            recent = tdf[(tdf["date"] < gd) & (tdf["date"] >= win_start)]
            ip_sum = float(recent["relief_ip"].sum()) if "relief_ip" in recent else 0.0
            pitch_sum = float(recent["relief_pitches"].sum()) if "relief_pitches" in recent else 0.0
            b2b = bool(recent["any_b2b_reliever"].any()) if "any_b2b_reliever" in recent else False
            rows.append({
                "game_id": gid,
                f"{prefix}_ip_{window_days}d": ip_sum,
                f"{prefix}_pitches_{window_days}d": pitch_sum,
                f"{prefix}_b2b_flag": b2b,
                # crude fatigue flag; tune later
                f"{prefix}_fatigued": bool(ip_sum >= 12.0 or pitch_sum >= 150.0),
            })
    if not rows:
        return pd.DataFrame()

    # collapse to one row per game
    wide = {}
    for d in rows:
        gid = d.pop("game_id")
        acc = wide.setdefault(gid, {"game_id": gid})
        acc.update(d)
    return pd.DataFrame(list(wide.values()))


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    args = ap.parse_args()

    df = build_bullpen_daily(args.start, args.end)
    if df.empty:
        print('no bullpen rows built'); return
    n = upsert_df(df, 'bullpens_daily', pk=['team','date'])
    print(f"upsert bullpens_daily: {n}")


if __name__ == '__main__':
    main()




