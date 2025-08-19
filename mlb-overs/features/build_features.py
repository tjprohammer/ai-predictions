# features/build_features.py
from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, inspect
from datetime import datetime

# Fix import paths for running as module
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ingestors.bullpens_daily import bullpen_recent_usage  # we use this helper
    from ingestors.era_ingestor import backfill_todays_pitchers, get_pitcher_era_stats  # ERA functionality
except ImportError:
    # Try relative import for direct execution
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from ingestors.bullpens_daily import bullpen_recent_usage
    from ingestors.era_ingestor import backfill_todays_pitchers, get_pitcher_era_stats

# Check if we should use enhanced historical data
def should_use_enhanced_data():
    """Check if enhanced historical data exists and is recent"""
    enhanced_path = Path(__file__).parent.parent / "data" / "enhanced_historical_games_2025.parquet"
    if not enhanced_path.exists():
        return False
    
    try:
        df = pd.read_parquet(enhanced_path)
        if len(df) < 1000:  # Minimum threshold
            return False
        
        # Check if data is recent (last 30 days)
        max_date = pd.to_datetime(df['date']).max()
        days_old = (datetime.now() - max_date).days
        
        if days_old <= 30:
            print(f"✅ Using enhanced historical data: {len(df)} games, {days_old} days old")
            return True
        else:
            print(f"⚠️  Enhanced data is {days_old} days old, using database instead")
            return False
            
    except Exception as e:
        print(f"⚠️  Error reading enhanced data: {e}")
        return False

# --- nickname / abbrev normalizer ---
ABBR_FIX = {
    "ATH":"athletics","OAK":"athletics",
    "AZ":"diamondbacks","ARI":"diamondbacks",
    "SD":"padres","SDP":"padres",
    "TB":"rays","TBR":"rays",
    "SF":"giants","SFG":"giants",
    "WSH":"nationals","WSN":"nationals",
    "CWS":"white sox",
    "NYY":"yankees","NYM":"mets",
    "KC":"royals","LAD":"dodgers","LAA":"angels",
    "ATL":"braves","BAL":"orioles","BOS":"red sox","CHC":"cubs",
    "CIN":"reds","CLE":"guardians","COL":"rockies","DET":"tigers",
    "HOU":"astros","MIA":"marlins","MIL":"brewers","MIN":"twins",
    "PHI":"phillies","PIT":"pirates","SEA":"mariners",
    "STL":"cardinals","TEX":"rangers","TOR":"blue jays",
}

def team_key_from_name(name: str) -> str:
    if not isinstance(name,str) or not name: return ""
    s = name.lower().strip()
    if "white sox" in s: return "white sox"
    if "red sox" in s: return "red sox"
    if "blue jays" in s: return "blue jays"
    return s.split()[-1]

def team_key_from_any(x: str) -> str:
    if not isinstance(x,str) or not x: return ""
    x = x.strip()
    if x.upper() in ABBR_FIX: return ABBR_FIX[x.upper()]
    return team_key_from_name(x)

# ---------- Starter form (last 3/5/10) ----------
def starter_form_for_games(eng, games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame()

    gmax = pd.to_datetime(games_df["date"]).max()
    lookback_start = (gmax - pd.Timedelta(days=150)).date()

    ps = pd.read_sql("""
        SELECT pitcher_id, date, pitches, csw_pct, velo_fb, avg_ev_allowed, xwoba_allowed, xslg_allowed
        FROM pitchers_starts
        WHERE date >= %(s)s
    """, eng, params={"s": lookback_start})

    if ps.empty:
        return pd.DataFrame()

    ps["date"] = pd.to_datetime(ps["date"])
    ps = ps.sort_values(["pitcher_id","date"]).reset_index(drop=True)
    by_pid = {pid: df for pid, df in ps.groupby("pitcher_id", as_index=False)}

    def roll_feats(df_pitcher, cutoff_date, N):
        sub = df_pitcher.loc[df_pitcher["date"] < cutoff_date].tail(N)
        if sub.empty:
            return {
                f"csw{N}": np.nan, f"velo{N}": np.nan,
                f"xwoba_allowed{N}": np.nan, f"xslg_allowed{N}": np.nan
            }
        return {
            f"csw{N}": sub["csw_pct"].mean(skipna=True),
            f"velo{N}": sub["velo_fb"].mean(skipna=True),
            f"xwoba_allowed{N}": sub["xwoba_allowed"].mean(skipna=True),
            f"xslg_allowed{N}": sub["xslg_allowed"].mean(skipna=True),
        }

    rows = []
    for _, r in games_df.iterrows():
        gid = r["game_id"]
        gd = pd.to_datetime(r["date"])
        for pid_col, prefix in [("home_sp_id","home_sp"), ("away_sp_id","away_sp")]:
            pid = r.get(pid_col)
            if pid is None or pd.isna(pid):
                continue
            try:
                pid = int(pid)
            except:
                continue
            pdf = by_pid.get(pid)
            if pdf is None:
                continue
            out = {"game_id": gid}
            for N in (3,5,10):
                rolled = roll_feats(pdf, gd, N)
                out.update({f"{prefix}_{k}": v for k,v in rolled.items()})
            rows.append(out)

    if not rows:
        return pd.DataFrame()

    # collapse to one row per game
    wide = {}
    for d in rows:
        gid = d.pop("game_id")
        acc = wide.setdefault(gid, {"game_id": gid})
        acc.update(d)
    return pd.DataFrame(list(wide.values()))

# ---------- Main feature build ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--database-url", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    eng = create_engine(args.database_url)
    insp = inspect(eng)

    # Most recent game date snapshot (include park_id so we can join parks)
    games = pd.read_sql("""
        WITH d AS (SELECT MAX(date) AS d FROM games)
        SELECT game_id, date, home_team, away_team, home_sp_id, away_sp_id, park_id, total_runs
        FROM games JOIN d ON games.date = d.d
        ORDER BY game_id
    """, eng)

    if games.empty:
        pd.DataFrame().to_parquet(args.out, index=False)
        print("no games to build features")
        return

    # normalize keys and types early
    games["home_key"] = games["home_team"].apply(team_key_from_any)
    games["away_key"] = games["away_team"].apply(team_key_from_any)
    games["game_id"] = games["game_id"].astype(str)  # keep everything string keyed

    # Latest market total (prefer latest snapshot, else close)
    mkt = pd.read_sql("""
        WITH s AS (
          SELECT game_id, k_total, snapshot_ts,
                 ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY snapshot_ts DESC NULLS LAST) rn
          FROM markets_totals
          WHERE date = (SELECT MAX(date) FROM games) AND market_type='snapshot'
        ),
        c AS (
          SELECT game_id, close_total FROM markets_totals
          WHERE date = (SELECT MAX(date) FROM games) AND market_type='close'
        )
        SELECT COALESCE(s.game_id, c.game_id) AS game_id,
               COALESCE(s.k_total, c.close_total) AS k_close
        FROM s FULL OUTER JOIN c USING (game_id)
        WHERE (s.rn = 1 OR s.rn IS NULL)
    """, eng).drop_duplicates("game_id")
    if not mkt.empty:
        mkt["game_id"] = mkt["game_id"].astype(str)

    # Offense daily
    off = pd.read_sql("""
      SELECT team, date, xwoba, iso, bb_pct, k_pct
      FROM teams_offense_daily
      WHERE date = (SELECT MAX(date) FROM games)
    """, eng)
    off["off_key"] = off["team"].astype(str).apply(team_key_from_any)

    # Bullpen daily (form + usage flags)
    bp = pd.read_sql("""
      SELECT team, date,
             bp_era, bp_fip, bp_kbb_pct, bp_hr9,
             closer_pitches_d1, setup1_pitches_d1, setup2_pitches_d1, closer_back2back_flag
      FROM bullpens_daily
      WHERE date = (SELECT MAX(date) FROM games)
    """, eng)
    if not bp.empty:
        bp["bp_key"] = bp["team"].astype(str).apply(team_key_from_any)

    # Start assembling feature frame
    df = games.merge(mkt, on="game_id", how="left")

    # Ensure ERA data is available for today's pitchers (backfill if needed)
    print("[build_features] Ensuring ERA data availability...")
    try:
        target_date = pd.to_datetime(games["date"].iloc[0]).date()
        backfill_results = backfill_todays_pitchers(target_date, args.database_url)
        if backfill_results:
            total_backfilled = sum(backfill_results.values())
            print(f"[build_features] Backfilled {total_backfilled} pitcher starts")
    except Exception as e:
        print(f"[build_features] ERA backfill failed: {e}")

    # Extract ERA features for starting pitchers
    print("[build_features] Extracting pitcher ERA features...")
    era_features = []
    for _, game in games.iterrows():
        game_date = pd.to_datetime(game["date"]).date()
        
        # Home pitcher ERA stats (including vs opponent)
        era_row = {"game_id": game["game_id"]}
        if pd.notna(game["home_sp_id"]):
            # Regular ERA stats
            home_era = get_pitcher_era_stats(str(int(game["home_sp_id"])), game_date, args.database_url)
            for key, value in home_era.items():
                era_row[f"home_sp_{key}"] = value
            
            # ERA vs opponent team  
            away_team_key = team_key_from_any(game["away_team"])
            home_era_vs_opp = get_pitcher_era_stats(str(int(game["home_sp_id"])), game_date, 
                                                   args.database_url, opponent_team=away_team_key)
            for key, value in home_era_vs_opp.items():
                if key.startswith('era_vs_') or key.startswith('games_vs_'):
                    era_row[f"home_sp_{key}"] = value
        
        # Away pitcher ERA stats (including vs opponent)
        if pd.notna(game["away_sp_id"]):
            # Regular ERA stats
            away_era = get_pitcher_era_stats(str(int(game["away_sp_id"])), game_date, args.database_url)
            for key, value in away_era.items():
                era_row[f"away_sp_{key}"] = value
                
            # ERA vs opponent team
            home_team_key = team_key_from_any(game["home_team"])
            away_era_vs_opp = get_pitcher_era_stats(str(int(game["away_sp_id"])), game_date,
                                                   args.database_url, opponent_team=home_team_key)
            for key, value in away_era_vs_opp.items():
                if key.startswith('era_vs_') or key.startswith('games_vs_'):
                    era_row[f"away_sp_{key}"] = value
        
        era_features.append(era_row)
    
    if era_features:
        era_df = pd.DataFrame(era_features)
        era_df["game_id"] = era_df["game_id"].astype(str)
        df = df.merge(era_df, on="game_id", how="left")

    # Offense joins
    home_off = off.add_prefix("home_")
    away_off = off.add_prefix("away_")
    df = df.merge(
        home_off[["home_off_key","home_xwoba","home_iso","home_bb_pct","home_k_pct"]],
        left_on="home_key", right_on="home_off_key", how="left"
    ).drop(columns=["home_off_key"])
    df = df.merge(
        away_off[["away_off_key","away_xwoba","away_iso","away_bb_pct","away_k_pct"]],
        left_on="away_key", right_on="away_off_key", how="left"
    ).drop(columns=["away_off_key"])

    # Bullpen joins (form + usage flags)
    if not bp.empty:
        bp_home = bp.add_prefix("home_")
        bp_away = bp.add_prefix("away_")
        df = df.merge(
            bp_home[["home_bp_key","home_bp_era","home_bp_fip","home_bp_kbb_pct","home_bp_hr9",
                     "home_closer_pitches_d1","home_setup1_pitches_d1","home_setup2_pitches_d1","home_closer_back2back_flag"]],
            left_on="home_key", right_on="home_bp_key", how="left"
        ).drop(columns=["home_bp_key"])
        df = df.merge(
            bp_away[["away_bp_key","away_bp_era","away_bp_fip","away_bp_kbb_pct","away_bp_hr9",
                     "away_closer_pitches_d1","away_setup1_pitches_d1","away_setup2_pitches_d1","away_closer_back2back_flag"]],
            left_on="away_key", right_on="away_bp_key", how="left"
        ).drop(columns=["away_bp_key"])

    # Recent bullpen usage (3d window)
    bp_recent = bullpen_recent_usage(eng, games, window_days=3)
    if not bp_recent.empty:
        bp_recent["game_id"] = bp_recent["game_id"].astype(str)
        df = df.merge(bp_recent, on="game_id", how="left")

    # Starter rolling form
    sp_form = starter_form_for_games(eng, games)
    if not sp_form.empty:
        sp_form["game_id"] = sp_form["game_id"].astype(str)
        df = df.merge(sp_form, on="game_id", how="left")

    # --- Parks join (park factors / roof) ---
    parks = pd.read_sql("""
        SELECT
          park_id::text AS park_id,
          pf_runs_3y,
          pf_hr_3y,
          altitude_ft AS park_altitude_ft,
          roof_type
        FROM parks
    """, eng)
    # ensure we have park_id in df from games already
    df = df.merge(parks, on="park_id", how="left")

    # --- Weather join (scheduled game-time hour) ---
    wx = pd.read_sql("""
      SELECT
        game_id::bigint AS gid_big,
        temp_f, humidity_pct, wind_mph, wind_dir_deg, precip_prob,
        altitude_ft, air_density_idx, wind_out_mph, wind_cross_mph
      FROM weather_game
    """, eng)
    if not wx.empty:
        wx = wx.rename(columns={"gid_big":"game_id"})
        wx["game_id"] = wx["game_id"].astype(str)
        df = df.merge(wx, on="game_id", how="left")
    else:
        # create empty columns if table has no rows
        # ensure weather cols are numeric (avoid abs(None) crash)
        for c in ["temp_f","humidity_pct","wind_mph","wind_dir_deg","precip_prob",
                "altitude_ft","air_density_idx","wind_out_mph","wind_cross_mph"]:
            if c not in df.columns:
                df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors="coerce")


    # Derived weather helper (optional)
    df["wx_run_boost"] = (
        df["wind_out_mph"].fillna(0.0) * 0.05
        + (df["temp_f"] - 60.0).fillna(0.0) * 0.01
        + df["wind_cross_mph"].fillna(0.0).abs() * 0.01
        + (df["altitude_ft"].fillna(0.0) / 1000.0) * 0.15
    ).clip(-1.5, 1.5)

    # Final column order (keep core stuff; the rest can ride along)
    keep = [
        "game_id","date","home_team","away_team","k_close","total_runs",  # Added total_runs for training
        "home_xwoba","home_iso","home_bb_pct","home_k_pct",
        "away_xwoba","away_iso","away_bb_pct","away_k_pct",
        "home_bp_era","home_bp_fip","home_bp_kbb_pct","home_bp_hr9",
        "away_bp_era","away_bp_fip","away_bp_kbb_pct","away_bp_hr9",
        "home_closer_pitches_d1","home_setup1_pitches_d1","home_setup2_pitches_d1","home_closer_back2back_flag",
        "away_closer_pitches_d1","away_setup1_pitches_d1","away_setup2_pitches_d1","away_closer_back2back_flag",
        # bullpen recent
        "home_bp_ip_3d","home_bp_pitches_3d","home_bp_b2b_flag","home_bp_fatigued",
        "away_bp_ip_3d","away_bp_pitches_3d","away_bp_b2b_flag","away_bp_fatigued",
        # starter form
        "home_sp_csw3","home_sp_velo3","home_sp_xwoba_allowed3","home_sp_xslg_allowed3",
        "away_sp_csw3","away_sp_velo3","away_sp_xwoba_allowed3","away_sp_xslg_allowed3",
        "home_sp_csw5","away_sp_csw5",
        "home_sp_csw10","away_sp_csw10",
        # pitcher ERA features (new)
        "home_sp_era_season","away_sp_era_season",
        "home_sp_era_l3","away_sp_era_l3",
        "home_sp_era_l5","away_sp_era_l5", 
        "home_sp_era_l10","away_sp_era_l10",
        "home_sp_games_l3","away_sp_games_l3",
        "home_sp_games_l5","away_sp_games_l5",
        "home_sp_games_l10","away_sp_games_l10",
        # vs opponent ERA features
        "home_sp_era_vs_opp","away_sp_era_vs_opp",
        "home_sp_games_vs_opp","away_sp_games_vs_opp",
        # parks
        "pf_runs_3y","pf_hr_3y","park_altitude_ft","roof_type",
        # weather
        "temp_f","humidity_pct","wind_mph","wind_dir_deg","precip_prob",
        "altitude_ft","air_density_idx","wind_out_mph","wind_cross_mph","wx_run_boost",
    ]

    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    df = df.drop_duplicates("game_id")[keep]
    
    # Add target_y for training (rename total_runs to target_y for consistency)
    if "total_runs" in df.columns:
        df["target_y"] = df["total_runs"]

    df.to_parquet(args.out, index=False)
    print(f"wrote features: {len(df)} rows -> {args.out}")

if __name__ == "__main__":
    main()
