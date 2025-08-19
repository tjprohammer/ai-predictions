# models/infer.py
from __future__ import annotations
import argparse, os, math, datetime as dt
import pandas as pd
from sqlalchemy import create_engine, text
from joblib import load
from pathlib import Path

# Import our ERA ingestor
import sys
sys.path.append(str(Path(__file__).parent.parent))
from ingestors.era_ingestor import backfill_todays_pitchers, get_pitcher_era_stats

# --- team key normalizer (match features/build_features.py behavior) ---
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


def ewmean(s, span=10):
    s = pd.to_numeric(s, errors="coerce")
    return s.ewm(span=span, min_periods=max(3, span//3)).mean().iloc[-1] if len(s) else None

def rolling_team(off_df, team_key, end_date, window):
    d = pd.to_datetime(end_date)
    start = (d - pd.Timedelta(days=window)).date()
    sub = off_df[(off_df["off_key"]==team_key) & (off_df["date"]>=start) & (off_df["date"]<d.date())].sort_values("date")
    if sub.empty: return {}
    return {
        f"xwoba{window}": sub["xwoba"].astype(float).mean(skipna=True),
        f"iso{window}": sub["iso"].astype(float).mean(skipna=True),
        f"bb{window}": sub["bb_pct"].astype(float).mean(skipna=True),
        f"k{window}": sub["k_pct"].astype(float).mean(skipna=True),
    }

def last10_sp_agg(starts, pid, end_date):
    sub = starts[(starts["pitcher_id"]==pid) & (starts["date"]<end_date)].sort_values("date").tail(10)
    if sub.empty: return {}
    return {
        "sp_xwoba_allowed": pd.to_numeric(sub["xwoba_allowed"], errors="coerce").mean(),
        "sp_csw": pd.to_numeric(sub["csw_pct"], errors="coerce").mean(),
        "sp_velo": pd.to_numeric(sub["velo_fb"], errors="coerce").mean(),
        "sp_pitches": pd.to_numeric(sub["pitches"], errors="coerce").mean(),
    }

def _era_from_rows(sub: pd.DataFrame) -> float | None:
    if sub.empty:
        return None
    er = pd.to_numeric(sub.get("er"), errors="coerce").fillna(0).sum()
    ip = pd.to_numeric(sub.get("ip"), errors="coerce").fillna(0).sum()
    return float(er * 9.0 / ip) if ip and ip > 0 else None

def era_season(starts_df: pd.DataFrame, pid: int | None, end_date) -> float | None:
    if not pid:
        return None
    sub = starts_df[(starts_df["pitcher_id"] == str(pid)) & (starts_df["date"] < end_date)]
    return _era_from_rows(sub)

def era_lastN(starts_df: pd.DataFrame, pid: int | None, end_date, N: int) -> float | None:
    if not pid:
        return None
    sub = (starts_df[(starts_df["pitcher_id"] == str(pid)) & (starts_df["date"] < end_date)]
           .sort_values("date").tail(N))
    return _era_from_rows(sub)


def get_pitcher_era_stats_infer(eng, pitcher_id, end_date, opponent_team=None):
    """Get comprehensive ERA statistics for a pitcher using the ERA ingestor"""
    if not pitcher_id:
        return {}
    
    # Convert SQLAlchemy engine to psycopg2 connection string
    # Extract components to rebuild URL with password
    url_parts = eng.url
    database_url = f"postgresql://{url_parts.username}:{url_parts.password}@{url_parts.host}:{url_parts.port}/{url_parts.database}"
    
    from ingestors.era_ingestor import get_pitcher_era_stats as ingestor_era_stats
    return ingestor_era_stats(str(pitcher_id), end_date, database_url, opponent_team)

def sigmoid(x, alpha=1.1):
    try:
        return 1.0 / (1.0 + math.exp(-alpha * float(x)))
    except OverflowError:
        return 1.0 if x > 0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb"))
    ap.add_argument("--date", default=dt.date.today().isoformat(), help="Date to infer (YYYY-MM-DD)")
    ap.add_argument("--out", default="predictions_today.parquet")
    ap.add_argument("--model", default="models/model_totals.joblib")
    args = ap.parse_args()

    # Load the trained ML model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return
    
    model = load(model_path)
    print(f"✅ Loaded ML model: {model_path}")

    eng = create_engine(args.database_url)

    target_date = pd.to_datetime(args.date).date()

    # games for the target date
    games = pd.read_sql(text("SELECT game_id, date, home_team, away_team, home_sp_id, away_sp_id FROM games WHERE date=:d"),
                        eng, params={"d": target_date})
    if games.empty:
        print("no games to infer"); 
        pd.DataFrame(columns=["game_id","k_close","y_pred","p_over_cal","p_under_cal","edge"]).to_parquet(args.out, index=False)
        return

    # normalize team keys for joins
    games["home_key"] = games["home_team"].astype(str).apply(team_key_from_any)
    games["away_key"] = games["away_team"].astype(str).apply(team_key_from_any)

    # markets: prefer latest snapshot else close
    mkt = pd.read_sql(text("""
        WITH s AS (
          SELECT game_id, k_total, snapshot_ts,
                 ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY snapshot_ts DESC NULLS LAST) rn
          FROM markets_totals
          WHERE date = :d AND market_type='snapshot'
        ),
        c AS (
          SELECT game_id, close_total FROM markets_totals
          WHERE date = :d AND market_type='close'
        )
        SELECT COALESCE(s.game_id, c.game_id) AS game_id,
               COALESCE(s.k_total, c.close_total) AS k_close
        FROM s FULL OUTER JOIN c USING (game_id)
        WHERE (s.rn = 1 OR s.rn IS NULL)
    """), eng, params={"d": target_date})

    df = games.merge(mkt, on="game_id", how="left")
    df["k_close"] = pd.to_numeric(df["k_close"], errors="coerce")
    before = len(df)
    df = df[df["k_close"].notna()].copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"[infer] dropped {dropped} games without k_close (no sportsbook total)")

    # load team offense & bullpen & sp starts
    off = pd.read_sql(text("SELECT team, date, xwoba, iso, bb_pct, k_pct FROM teams_offense_daily"), eng)
    off["date"] = pd.to_datetime(off["date"]).dt.date
    off["off_key"] = off["team"].astype(str).apply(team_key_from_any)

    bp = pd.read_sql(text("SELECT team, date, bp_fip, closer_back2back_flag FROM bullpens_daily"), eng)
    bp["date"] = pd.to_datetime(bp["date"]).dt.date
    bp["bp_key"] = bp["team"].astype(str).apply(team_key_from_any)

    sp = pd.read_sql(text("SELECT pitcher_id, date, xwoba_allowed, csw_pct, velo_fb, pitches FROM pitchers_starts"), eng)
    sp["date"] = pd.to_datetime(sp["date"]).dt.date

    # ERA inputs (ER/IP). Keep separate from `sp` so we don't disturb other logic.
    starts_era = pd.read_sql(text("""
        SELECT pitcher_id, date, ip, er
        FROM pitchers_starts
    """), eng)
    if not starts_era.empty:
        starts_era["date"] = pd.to_datetime(starts_era["date"]).dt.date
        starts_era["pitcher_id"] = starts_era["pitcher_id"].astype(str)
        for c in ("ip", "er"):
            starts_era[c] = pd.to_numeric(starts_era[c], errors="coerce")
    else:
        starts_era = pd.DataFrame(columns=["pitcher_id","date","ip","er"])


    rows = []
    for _, g in df.iterrows():
        d = g["date"]
        home = g["home_team"]; away = g["away_team"]
        hkey = g["home_key"]; akey = g["away_key"]
        hsp = int(g["home_sp_id"]) if pd.notna(g["home_sp_id"]) else None
        asp = int(g["away_sp_id"]) if pd.notna(g["away_sp_id"]) else None

        # offense windows
        ho14 = rolling_team(off, hkey, d, 14); ho30 = rolling_team(off, hkey, d, 30)
        ao14 = rolling_team(off, akey, d, 14); ao30 = rolling_team(off, akey, d, 30)

        # bullpen yesterday (simple)
        d1 = (pd.to_datetime(d) - pd.Timedelta(days=1)).date()
        hbp = bp[(bp["bp_key"]==hkey) & (bp["date"]==d1)].tail(1)
        abp = bp[(bp["bp_key"]==akey) & (bp["date"]==d1)].tail(1)
        hbp_fip = float(hbp["bp_fip"].iloc[0]) if not hbp.empty and pd.notna(hbp["bp_fip"].iloc[0]) else None
        abp_fip = float(abp["bp_fip"].iloc[0]) if not abp.empty and pd.notna(abp["bp_fip"].iloc[0]) else None
        hbp_b2b = int(hbp["closer_back2back_flag"].iloc[0]) if not hbp.empty and pd.notna(hbp["closer_back2back_flag"].iloc[0]) else 0
        abp_b2b = int(abp["closer_back2back_flag"].iloc[0]) if not abp.empty and pd.notna(abp["closer_back2back_flag"].iloc[0]) else 0

        # starters last-10 and ERA stats
        hsp10 = last10_sp_agg(sp, hsp, d) if hsp else {}
        asp10 = last10_sp_agg(sp, asp, d) if asp else {}
        
        # pitcher ERA statistics (including vs opponent)
        hsp_era = get_pitcher_era_stats_infer(eng, hsp, d) if hsp else {}
        asp_era = get_pitcher_era_stats_infer(eng, asp, d) if asp else {}
        
        # ERA vs opponent stats
        hsp_era_vs_opp = get_pitcher_era_stats_infer(eng, hsp, d, opponent_team=akey) if hsp else {}
        asp_era_vs_opp = get_pitcher_era_stats_infer(eng, asp, d, opponent_team=hkey) if asp else {}
        
        # Debug: Print what we actually got from ERA function
        if hsp and hsp_era:
            print(f"    Home SP {hsp} ERA data: {hsp_era}")
        if asp and asp_era:
            print(f"    Away SP {asp} ERA data: {asp_era}")

        # baselines
        # baselines
        k_close = float(g["k_close"])
        league_xwoba = 0.320
        league_era = 4.20

        # ERA from database rows (ER/IP), with sensible fallbacks
        hsp_season_era = era_season(starts_era, hsp, d) if hsp else None
        asp_season_era = era_season(starts_era, asp, d) if asp else None
        hsp_l5_era     = era_lastN(starts_era, hsp, d, 5) if hsp else None
        asp_l5_era     = era_lastN(starts_era, asp, d, 5) if asp else None

        hsp_season_era = hsp_season_era if hsp_season_era is not None else league_era
        asp_season_era = asp_season_era if asp_season_era is not None else league_era
        hsp_l5_era     = hsp_l5_era     if hsp_l5_era     is not None else hsp_season_era
        asp_l5_era     = asp_l5_era     if asp_l5_era     is not None else asp_season_era

        
        # assemble signals (fillna with sensible priors)
        hx14 = float(ho14.get("xwoba14", league_xwoba) or league_xwoba)
        ax14 = float(ao14.get("xwoba14", league_xwoba) or league_xwoba)
        hiso14 = float(ho14.get("iso14", 0.160) or 0.160)
        aiso14 = float(ao14.get("iso14", 0.160) or 0.160)
        hbb14 = float(ho14.get("bb14", 0.085) or 0.085)
        abb14 = float(ao14.get("bb14", 0.085) or 0.085)

        hxsp = float(hsp10.get("sp_xwoba_allowed", 0.300) or 0.300)
        axsp = float(asp10.get("sp_xwoba_allowed", 0.300) or 0.300)
        
        # pitcher ERA metrics with fallbacks
        hsp_season_era = float(hsp_era.get("era_season", league_era) or league_era)
        asp_season_era = float(asp_era.get("era_season", league_era) or league_era)
        hsp_l5_era = float(hsp_era.get("era_l5", hsp_season_era) or hsp_season_era)
        asp_l5_era = float(asp_era.get("era_l5", asp_season_era) or asp_season_era)
        
        # vs opponent ERA metrics with fallbacks
        hsp_vs_opp_era = float(hsp_era_vs_opp.get("era_vs_opp", hsp_season_era) or hsp_season_era)
        asp_vs_opp_era = float(asp_era_vs_opp.get("era_vs_opp", asp_season_era) or asp_season_era)

        hbp_fip = 4.10 if hbp_fip is None else hbp_fip
        abp_fip = 4.10 if abp_fip is None else abp_fip

        # Get 30-day stats (default to 14-day if missing)
        hx30 = float(ho30.get("xwoba30", hx14) or hx14)
        ax30 = float(ao30.get("xwoba30", ax14) or ax14)
        hiso30 = float(ho30.get("iso30", hiso14) or hiso14)
        aiso30 = float(ao30.get("iso30", aiso14) or aiso14)
        hbb30 = float(ho30.get("bb30", hbb14) or hbb14)
        abb30 = float(ao30.get("bb30", abb14) or abb14)

        # Get additional ERA stats (use season if missing)
        hsp_l3_era = float(hsp_era.get("era_l3", hsp_season_era) or hsp_season_era)
        asp_l3_era = float(asp_era.get("era_l3", asp_season_era) or asp_season_era)
        hsp_l10_era = float(hsp_era.get("era_l10", hsp_season_era) or hsp_season_era)
        asp_l10_era = float(asp_era.get("era_l10", asp_season_era) or asp_season_era)

        # Build feature vector matching EXACT training features from DEF_FEATURES
        # Must match the order in models/train.py DEF_FEATURES exactly
        feature_dict = {
            "k_close": k_close,
            "hxwoba14": hx14, "axwoba14": ax14, "hiso14": hiso14, "aiso14": aiso14, "hbb14": hbb14, "abb14": abb14,
            "hxwoba30": hx30, "axwoba30": ax30, "hiso30": hiso30, "aiso30": aiso30, "hbb30": hbb30, "abb30": abb30,
            "hbp_fip_yday": hbp_fip, "abp_fip_yday": abp_fip, "hbp_b2b": hbp_b2b, "abp_b2b": abp_b2b,
            "hsp_xwoba10": hxsp, "asp_xwoba10": axsp, "hsp_csw10": 0.27, "asp_csw10": 0.27, "hsp_velo10": 94.0, "asp_velo10": 94.0,
            "home_sp_era_season": hsp_season_era, "away_sp_era_season": asp_season_era,
            "home_sp_era_l3": hsp_l3_era, "away_sp_era_l3": asp_l3_era,
            "home_sp_era_l5": hsp_l5_era, "away_sp_era_l5": asp_l5_era,
            "home_sp_era_l10": hsp_l10_era, "away_sp_era_l10": asp_l10_era,
            "home_sp_era_vs_opp": hsp_vs_opp_era, "away_sp_era_vs_opp": asp_vs_opp_era,
            "home_xwoba": hx14, "away_xwoba": ax14, "home_iso": hiso14, "away_iso": aiso14,
            "home_bp_era": hbp_fip, "away_bp_era": abp_fip, "home_bp_fip": hbp_fip, "away_bp_fip": abp_fip,
            "temp_f": 72.0, "wind_mph": 8.0, "pf_runs_3y": 1.0, "park_altitude_ft": 500.0,
            "wx_run_boost": 0.0, "home_bp_b2b_flag": hbp_b2b, "away_bp_b2b_flag": abp_b2b,
            "home_sp_xwoba_allowed3": hxsp, "away_sp_xwoba_allowed3": axsp,
            "home_sp_csw3": 0.27, "away_sp_csw3": 0.27, "home_sp_velo3": 94.0, "away_sp_velo3": 94.0,
        }
        
        # Convert to dataframe with proper feature names for sklearn
        feature_df = pd.DataFrame([feature_dict])
        
        # Use the ML model that was trained on 1,868 games
        y_pred = model.predict(feature_df)[0]

        edge = y_pred - k_close
        p_over = sigmoid(edge, alpha=1.1)
        p_under = 1.0 - p_over
        conf = max(p_over, p_under)

        rows.append({
            "game_id": g["game_id"],
            "date": d,
            "home_team": home,
            "away_team": away,
            "matchup": f"{away} @ {home}",
            "home_starting_pitcher": str(g["home_sp_name"]) if pd.notna(g.get("home_sp_name")) else "Unknown",
            "away_starting_pitcher": str(g["away_sp_name"]) if pd.notna(g.get("away_sp_name")) else "Unknown",
            "home_starting_pitcher_id": hsp if hsp else None,
            "away_starting_pitcher_id": asp if asp else None,
            "k_close": k_close,
            "y_pred": float(y_pred),
            "edge": float(edge),
            "p_over_cal": float(p_over),
            "p_under_cal": float(p_under),
            "conf": float(conf),
            "home_xwoba_14": round(hx14, 3),
            "away_xwoba_14": round(ax14, 3),
            "home_xwoba_allowed": round(hxsp, 3),
            "away_xwoba_allowed": round(axsp, 3),
            "home_pitcher_era_season": round(hsp_season_era, 2),
            "away_pitcher_era_season": round(asp_season_era, 2),
            "home_pitcher_era_l5": round(hsp_l5_era, 2),
            "away_pitcher_era_l5": round(asp_l5_era, 2),
            "home_pitcher_era_vs_opp": round(hsp_vs_opp_era, 2),
            "away_pitcher_era_vs_opp": round(asp_vs_opp_era, 2),
        })

    out = pd.DataFrame(rows)
    out.to_parquet(args.out, index=False)
    print(f"wrote predictions: {len(out)} rows -> {args.out}")

if __name__ == "__main__":
    main()
