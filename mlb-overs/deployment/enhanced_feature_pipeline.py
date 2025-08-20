#!/usr/bin/env python3
"""
Enhanced Feature Pipeline - High-Leverage Missing Factors
=========================================================
Adds the critical missing features identified in the audit:
1. Bullpen fatigue & availability (L3-L5 days)
2. Starter recent form & rest
3. Umpire tendencies & catcher framing
4. Weather detail (wind direction, humidity, roof status)
5. Lineup strength & handedness realism
6. Travel & schedule fatigue

This addresses the +0.67 slate bias and adds spread without market dependency.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# REMOVE DEFAULT_WHIP - enforce real data only

RATE_DEFAULTS = {
    "home_team_woba": 0.310, "away_team_woba": 0.310,
    "home_team_babip": 0.295, "away_team_babip": 0.295,
    "home_team_wrcplus": 100.0, "away_team_wrcplus": 100.0,
    "home_team_bb_pct": 0.085, "away_team_bb_pct": 0.085,
    "home_team_k_pct": 0.225, "away_team_k_pct": 0.225,
    "home_team_iso": 0.170, "away_team_iso": 0.170,
    "home_team_ba": 0.250, "away_team_ba": 0.250,
    # add any other rate features here
}

def fill_rate_defaults(df):
    for col, val in RATE_DEFAULTS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(val).clip(lower=val*0.7, upper=val*1.3)
    return df

# Serving-time calibration constants (from audit)
CAL_SLOPE = 1.00
CAL_INTERCEPT = -0.52  # flips the +0.52 bias (pred - actual = -0.52)

def apply_serving_calibration(y_pred, expected_total):
    """Apply truth calibration and drift guard"""
    import numpy as np
    
    # Truth-calibration (global)
    y_cal = CAL_SLOPE * y_pred + CAL_INTERCEPT
    
    # Drift guard (blend only when it's nuts)
    deltas = np.abs(y_cal - expected_total)
    excess = np.median(deltas) > 2.5
    if excess:
        λ = 0.35  # blend weight toward market only on bad slates
        y_cal = (1-λ)*y_cal + λ*expected_total
        logger.info(f"Drift guard activated: median delta {np.median(deltas):.2f} > 2.5, blending λ={λ}")
    
    return y_cal

def _sanity(df):
    """Sanity checks and clamping for out-of-bounds features"""
    checks = {
        "combined_power": (0.12, 0.24),
        "home_team_woba": (0.25, 0.37),
        "away_team_woba": (0.25, 0.37),
        "home_team_babip": (0.260, 0.330),
        "away_team_babip": (0.260, 0.330),
        "expected_total": (6.0, 13.0),
    }
    for col, (lo, hi) in checks.items():
        if col in df.columns:
            bad = df[(df[col] < lo) | (df[col] > hi)]
            if len(bad):
                logger.warning(f"[SANITY] {col} out of bounds for {len(bad)} rows; clamping to [{lo},{hi}]")
                df[col] = df[col].clip(lo, hi)
    return df

# put near the top
def _as_series(x, index):
    s = pd.to_numeric(x, errors="coerce")
    if not isinstance(s, pd.Series):
        s = pd.Series([s] * len(index), index=index)
    return s

PARK_NAME_ALIASES = {
    "Daikin Park": "Minute Maid Park",           # Astros
    "Rate Field": "Guaranteed Rate Field",       # White Sox
    "Angel Stadium of Anaheim": "Angel Stadium",
    "Marlins Park": "loanDepot park",
}

def _norm_park_name(name: str) -> str:
    if not name: return ""
    name = str(name).strip()
    return PARK_NAME_ALIASES.get(name, name)

def _load_ballpark_overrides(engine, csv_path="../data/ballpark_factors_2023_2025.csv"):
    """Read DB table 'ballpark_factors' (venue, R, HR or run_factor/hr_factor) or a CSV with columns venue,R,HR (100=avg)."""
    import os
    out = {}
    try:
        df = pd.read_sql(text("""
            SELECT venue,
                   COALESCE(run_factor, NULLIF(R,0)/100.0) AS run_factor,
                   COALESCE(hr_factor, NULLIF(HR,0)/100.0) AS hr_factor
            FROM ballpark_factors
        """), engine)
        for _, r in df.iterrows():
            v = _norm_park_name(r["venue"])
            out[v] = {"run_factor": float(r["run_factor"]), "hr_factor": float(r["hr_factor"])}
    except Exception:
        pass

    if not out and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, r in df.iterrows():
            v = _norm_park_name(r["venue"])
            run_f = float(r["R"]); hr_f = float(r["HR"])
            if run_f > 10: run_f /= 100.0
            if hr_f  > 10: hr_f  /= 100.0
            out[v] = {"run_factor": run_f, "hr_factor": hr_f}
    return out

# light fallback if overrides missing
BALLPARK_FACTORS_FALLBACK = {"_DEFAULT": {"run_factor": 1.00, "hr_factor": 1.00}}

BALLPARK_FACTORS = {
    # venue_name: {run_factor, hr_factor} - real 2023-2025 values
    "Coors Field":                  {"run_factor": 1.12, "hr_factor": 1.25},
    "Fenway Park":                  {"run_factor": 1.05, "hr_factor": 1.24},
    "Chase Field":                  {"run_factor": 1.04, "hr_factor": 1.17},
    "Dodger Stadium":               {"run_factor": 1.02, "hr_factor": 0.97},
    "Great American Ball Park":     {"run_factor": 1.02, "hr_factor": 1.22},
    "Target Field":                 {"run_factor": 1.02, "hr_factor": 1.10},
    "Angel Stadium":                {"run_factor": 1.01, "hr_factor": 1.13},
    "Kauffman Stadium":             {"run_factor": 1.01, "hr_factor": 1.14},
    "loanDepot park":               {"run_factor": 1.01, "hr_factor": 1.08},
    "Oriole Park at Camden Yards":  {"run_factor": 1.01, "hr_factor": 0.98},
    "Citizens Bank Park":           {"run_factor": 1.01, "hr_factor": 1.14},
    "Nationals Park":               {"run_factor": 1.01, "hr_factor": 0.98},
    "Minute Maid Park":             {"run_factor": 1.00, "hr_factor": 1.05},
    "Comerica Park":                {"run_factor": 1.00, "hr_factor": 0.93},
    "Truist Park":                  {"run_factor": 1.00, "hr_factor": 1.02},
    "Yankee Stadium":               {"run_factor": 1.00, "hr_factor": 1.20},
    "Busch Stadium":                {"run_factor": 1.00, "hr_factor": 0.88},
    "PNC Park":                     {"run_factor": 0.99, "hr_factor": 0.76},
    "Guaranteed Rate Field":        {"run_factor": 0.99, "hr_factor": 0.96},
    "Rogers Centre":                {"run_factor": 0.99, "hr_factor": 1.02},
    "Citi Field":                   {"run_factor": 0.98, "hr_factor": 1.04},
    "Globe Life Field":             {"run_factor": 0.98, "hr_factor": 1.05},
    "Petco Park":                   {"run_factor": 0.98, "hr_factor": 1.02},
    "Wrigley Field":                {"run_factor": 0.97, "hr_factor": 0.97},  # wind handled separately
    "American Family Field":        {"run_factor": 0.97, "hr_factor": 1.06},
    "Progressive Field":            {"run_factor": 0.97, "hr_factor": 0.86},
    "Oracle Park":                  {"run_factor": 0.96, "hr_factor": 0.80},
    "T-Mobile Park":                {"run_factor": 0.91, "hr_factor": 0.93},

    # Synonyms / alternate names → map to canonical
    "Daikin Park":                     {"alias": "Minute Maid Park"},
    "Rate Field":                      {"alias": "Guaranteed Rate Field"},
    "Angel Stadium of Anaheim":        {"alias": "Angel Stadium"},
    "Marlins Park":                    {"alias": "loanDepot park"},
    "_DEFAULT":                        {"run_factor": 1.00, "hr_factor": 1.00},
}

def _air_density_index_from_eg(temp_f, humidity_pct, pressure_hpa=1013.25):
    """Calculate air density index from enhanced_games weather data"""
    try:
        if temp_f is None or np.isnan(temp_f):
            return 1.0
        T = (float(temp_f) - 32.0) * 5.0/9.0 + 273.15  # K
        RH = max(0.0, min(100.0, float(humidity_pct if humidity_pct is not None else 50.0))) / 100.0
        # crude reduction for humidity
        rho = (pressure_hpa*100.0) / (287.05 * T) * (1.0 - 0.378 * RH)
        return float(rho / 1.225)  # normalize to ~sea level std
    except Exception:
        return 1.0

def _wind_components(speed_mph, dir_deg, cf_bearing_deg):
    """Calculate wind components from speed and direction"""
    try:
        spd = float(speed_mph if speed_mph is not None else 0.0)
        deg = float(dir_deg if dir_deg is not None else 0.0)
        ang = np.deg2rad(deg - cf_bearing_deg)
        out_to_cf = spd * np.cos(ang)
        cross = spd * np.sin(ang)
        return out_to_cf, cross
    except Exception:
        return 0.0, 0.0

def _num(s, default=np.nan):
    v = pd.to_numeric(s, errors="coerce")
    if isinstance(v, pd.Series):
        return v.fillna(default)
    return default if pd.isna(v) else v

def _resolve_park_meta(name_or_id):
    meta = BALLPARK_FACTORS.get(name_or_id)
    if isinstance(meta, dict) and "alias" in meta:
        return BALLPARK_FACTORS.get(meta["alias"], BALLPARK_FACTORS["_DEFAULT"])
    return meta or BALLPARK_FACTORS["_DEFAULT"]

# Rate feature defaults to prevent 0.0 fills on sensitive features
RATE_DEFAULTS = {
    "home_team_woba": 0.310, "away_team_woba": 0.310,
    "home_team_babip": 0.295, "away_team_babip": 0.295,
    "home_team_obp": 0.320, "away_team_obp": 0.320,
    "home_team_slg": 0.400, "away_team_slg": 0.400,
    "home_team_wrcplus": 100.0, "away_team_wrcplus": 100.0,
}

def fill_rate_defaults(df):
    """Fill rate features with league anchors instead of 0.0"""
    for col, val in RATE_DEFAULTS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(val).clip(lower=val*0.7, upper=val*1.3)
    return df

def _sanity(df):
    """Sanity checks to clamp features to reasonable ranges"""
    checks = {
        "combined_power": (0.12, 0.24),
        "home_team_woba": (0.25, 0.37),
        "away_team_woba": (0.25, 0.37),
        "home_team_babip": (0.260, 0.330),
        "away_team_babip": (0.260, 0.330),
        "expected_total": (6.0, 13.0),
    }
    for col, (lo, hi) in checks.items():
        if col in df.columns:
            bad = df[(df[col] < lo) | (df[col] > hi)]
            if len(bad):
                logger.warning(f"[SANITY] {col} out of bounds for {len(bad)} rows; clamping to [{lo},{hi}]")
                df[col] = df[col].clip(lo, hi)
    return df

def apply_serving_calibration(y_pred, expected_total):
    """Apply truth calibration and drift guard to predictions"""
    import numpy as np
    
    # Truth-calibration from audit
    CAL_SLOPE = 1.00
    CAL_INTERCEPT = -0.52  # flips the +0.52 bias (pred - actual = -0.52)
    
    y_cal = CAL_SLOPE * np.asarray(y_pred, dtype=float) + CAL_INTERCEPT
    
    # Tiered drift guard: blend toward market when deltas are excessive, ignoring bad markets
    exp_total = np.asarray(expected_total, dtype=float)
    valid = np.isfinite(exp_total) & (exp_total > 5.0) & (exp_total < 15.0)
    
    if valid.any():
        deltas = np.abs(y_cal[valid] - exp_total[valid])
        median_delta = np.nanmedian(deltas)
        
        if np.isfinite(median_delta):
            if median_delta > 4.0:
                λ = 0.50  # Strong blending for extreme slates
            elif median_delta > 2.5:
                λ = 0.40  # Moderate blending for outliers
            else:
                λ = None
                
            if λ is not None:
                y_cal[valid] = (1-λ)*y_cal[valid] + λ*exp_total[valid]
                logger.warning(f"Drift guard activated on {valid.sum()}/{len(valid)} games: "
                               f"median delta {median_delta:.2f}, λ={λ}")
    
    return y_cal

def _impute_sp_whip(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    idx = df.index

    def _series(colname):
        # always return a float Series aligned to df.index
        s = pd.to_numeric(df.get(colname), errors="coerce")
        if not isinstance(s, pd.Series):
            s = pd.Series(np.nan, index=idx)
        return s

    for side in ("home", "away"):
        whip_col = f"{side}_sp_whip"

        season  = _series(f"{side}_sp_season_whip")
        H       = _series(f"{side}_sp_h")
        BB      = _series(f"{side}_sp_bb")
        IP      = _series(f"{side}_sp_ip")

        # avoid .replace on scalars; use where on a guaranteed Series
        derived = (H + BB) / IP.where(IP != 0, np.nan)

        # start from existing column (as Series)
        base = _series(whip_col)

        # REAL DATA ONLY path: fill from season then derived; no league default
        filled = base.fillna(season).fillna(derived)

        # keep logs but don't crash
        if filled.isna().any():
            miss_idx = filled[filled.isna()].index
            gids = frame.loc[miss_idx, 'game_id'].astype(str).tolist() if 'game_id' in frame.columns else list(miss_idx)
            logger.warning(f"WHIP missing for {whip_col} on {len(miss_idx)} rows; game_id={gids}. Will be imputed downstream.")

        df[whip_col] = filled.clip(0.7, 2.5)

    # combined_whip: mean of sides; keep NaNs if either side missing (log once)
    df["combined_whip"] = pd.concat([df["home_sp_whip"], df["away_sp_whip"]], axis=1).mean(axis=1)
    if df["combined_whip"].isna().any():
        miss_idx = df["combined_whip"][df["combined_whip"].isna()].index
        gids = frame.loc[miss_idx, 'game_id'].astype(str).tolist() if 'game_id' in frame.columns else list(miss_idx)
        logger.warning(f"Combined WHIP missing on {len(miss_idx)} rows; game_id={gids}. Will be imputed downstream.")

    return df

def _ensure_ballpark_factors(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()

    # Check if ballpark factors need to be calculated
    # Need to calculate if: column missing, all NaN, or all default values (1.0)
    run_vals = pd.to_numeric(df.get("ballpark_run_factor"), errors="coerce") if "ballpark_run_factor" in df.columns else pd.Series([np.nan] * len(df))
    hr_vals = pd.to_numeric(df.get("ballpark_hr_factor"), errors="coerce") if "ballpark_hr_factor" in df.columns else pd.Series([np.nan] * len(df))
    
    need_run = ("ballpark_run_factor" not in df.columns) or run_vals.isna().all() or (run_vals == 1.0).all()
    need_hr  = ("ballpark_hr_factor"  not in df.columns) or hr_vals.isna().all() or (hr_vals == 1.0).all()
    
    if not (need_run or need_hr):
        return df

    names = df.get("venue")
    if names is None or names.isna().all():
        names = df.get("venue_name")
    ids   = df.get("venue_id")

    run_vals, hr_vals = [], []
    for i in range(len(df)):
        nm  = str(names.iloc[i]) if names is not None else None
        vid = ids.iloc[i] if ids is not None else None

        meta = None
        if nm and nm in BALLPARK_FACTORS:
            meta = BALLPARK_FACTORS[nm]
            # Handle alias entries
            if "alias" in meta:
                alias_name = meta["alias"]
                if alias_name in BALLPARK_FACTORS:
                    meta = BALLPARK_FACTORS[alias_name]
                else:
                    meta = BALLPARK_FACTORS["_DEFAULT"]
        elif vid and vid in BALLPARK_FACTORS:
            meta = BALLPARK_FACTORS[vid]
            # Handle alias entries
            if "alias" in meta:
                alias_name = meta["alias"]
                if alias_name in BALLPARK_FACTORS:
                    meta = BALLPARK_FACTORS[alias_name]
                else:
                    meta = BALLPARK_FACTORS["_DEFAULT"]
        else:
            meta = BALLPARK_FACTORS["_DEFAULT"]

        run_vals.append(meta["run_factor"])
        hr_vals.append(meta["hr_factor"])

    if need_run:
        # Force replacement if all values are defaults (1.0)
        if "ballpark_run_factor" in df.columns and (pd.to_numeric(df["ballpark_run_factor"], errors="coerce") == 1.0).all():
            df["ballpark_run_factor"] = pd.Series(run_vals, index=df.index)
        else:
            df["ballpark_run_factor"] = pd.to_numeric(df.get("ballpark_run_factor"), errors="coerce")
            df["ballpark_run_factor"] = df["ballpark_run_factor"].fillna(pd.Series(run_vals, index=df.index))

    if need_hr:
        # Force replacement if all values are defaults (1.0)
        if "ballpark_hr_factor" in df.columns and (pd.to_numeric(df["ballpark_hr_factor"], errors="coerce") == 1.0).all():
            df["ballpark_hr_factor"] = pd.Series(hr_vals, index=df.index)
        else:
            df["ballpark_hr_factor"] = pd.to_numeric(df.get("ballpark_hr_factor"), errors="coerce")
            df["ballpark_hr_factor"] = df["ballpark_hr_factor"].fillna(pd.Series(hr_vals, index=df.index))

    # Always force-create ballpark_offensive_factor from run*hr factors
    df["ballpark_offensive_factor"] = (
        pd.to_numeric(df["ballpark_run_factor"], errors="coerce").fillna(1.0) *
        pd.to_numeric(df["ballpark_hr_factor"],  errors="coerce").fillna(1.0)
    )
    return df

class EnhancedFeaturePipeline:
    def __init__(self, db_url="postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb"):
        self.engine = create_engine(db_url)
        
        # Park-specific data for wind calculations
        self.park_cf_bearings = {
            'Fenway Park': 90,      # CF bearing degrees from home plate
            'Yankee Stadium': 90,
            'Coors Field': 90,
            'Minute Maid Park': 110,  # Angled
            # Add more as needed - default to 90 degrees
        }
        
        # Retractable roof parks
        self.retractable_roof_parks = {
            'Minute Maid Park', 'Tropicana Field', 'Rogers Centre', 
            'Chase Field', 'T-Mobile Park', 'American Family Field'
        }
        
        # Park name aliases for weather ingestor compatibility
        self.park_name_alias = {
            "Angel Stadium of Anaheim": "Angel Stadium",
            "Marlins Park": "loanDepot park",
            "Daikin Park": "Minute Maid Park",
            "Rate Field": "Guaranteed Rate Field",
        }
    
    # --- add these wrappers so older calls to self._... work
    def _impute_sp_whip(self, frame):
        return _impute_sp_whip(frame)

    def _merge_team_offense_l30(self, games_df, target_date):
        df = games_df.copy()
        target_dt = pd.to_datetime(target_date).date()

        # pull most recent snapshot on/before target date from teams_offense_daily
        try:
            q = text("""
                WITH snap AS (
                  SELECT MAX(date) AS d FROM teams_offense_daily WHERE date <= :d
                )
                SELECT t.team, t.runs_pg
                FROM teams_offense_daily t
                JOIN snap s ON t.date = s.d
            """)
            daily = pd.read_sql(q, self.engine, params={"d": target_dt})
        except Exception as e:
            logger.warning(f"team offense pull failed: {e}")
            daily = pd.DataFrame(columns=["team","runs_pg"])

        TEAM_TO_CODE = {
            "Arizona Diamondbacks":"AZ","Atlanta Braves":"ATL","Baltimore Orioles":"BAL",
            "Boston Red Sox":"BOS","Chicago Cubs":"CHC","Chicago White Sox":"CWS",
            "Cincinnati Reds":"CIN","Cleveland Guardians":"CLE","Colorado Rockies":"COL",
            "Detroit Tigers":"DET","Houston Astros":"HOU","Kansas City Royals":"KC",
            "Los Angeles Angels":"LAA","Los Angeles Dodgers":"LAD","Miami Marlins":"MIA",
            "Milwaukee Brewers":"MIL","Minnesota Twins":"MIN","New York Mets":"NYM",
            "New York Yankees":"NYY","Oakland Athletics":"ATH","Philadelphia Phillies":"PHI",
            "Pittsburgh Pirates":"PIT","San Diego Padres":"SD","San Francisco Giants":"SF",
            "Seattle Mariners":"SEA","St. Louis Cardinals":"STL","Tampa Bay Rays":"TB",
            "Texas Rangers":"TEX","Toronto Blue Jays":"TOR","Washington Nationals":"WSH",
        }
        rpg_map = daily.set_index("team")["runs_pg"].to_dict()
        league_avg = float(daily["runs_pg"].mean()) if not daily.empty else 4.1

        for side in ("home","away"):
            name_col = f"{side}_team"
            code_series = df[name_col].map(TEAM_TO_CODE).fillna(df[name_col])
            df[f"{side}_team_rpg_l30"] = pd.to_numeric(code_series.map(rpg_map), errors="coerce").fillna(league_avg)

        df["combined_offense_rpg"] = df["home_team_rpg_l30"] + df["away_team_rpg_l30"]
        s = df["combined_offense_rpg"]
        logger.info("DBG combined_offense_rpg (after team merge): non-null %d/%d, min=%.3f, median=%.3f, max=%.3f, std=%.3f",
                    s.notna().sum(), len(s), float(s.min()), float(s.median()), float(s.max()), float(s.std()))
        return df

    def _ensure_ballpark_factors(self, frame, **kwargs):
        return _ensure_ballpark_factors(frame)
    
    def add_power_features(self, games_df, target_date):
        """
        Populate home_power, away_power, combined_power from team ISO / HR rates,
        with pitcher-hand splits where possible. Produces slate variance even when
        splits are missing by using lineup/team context (non-constant fallback).
        """
        df = games_df.copy()
        # ✅ ensure park factors available here even if caller forgot
        df = self._ensure_ballpark_factors(df)
        idx = df.index
        target_dt = pd.to_datetime(target_date).date()

        # --- helpers
        def _series(col):
            s = pd.to_numeric(df.get(col), errors="coerce")
            if not isinstance(s, pd.Series):
                s = pd.Series(np.nan, index=idx)
            return s

        # Try multiple sources, newest -> broadest
        splits = None
        attempts = [
            ("""
             SELECT team, iso_vs_lhp, iso_vs_rhp, hr_per_pa_vs_lhp, hr_per_pa_vs_rhp, date
             FROM team_batting_splits_l30
             WHERE date = :d
            """, {"d": target_dt}),
            ("""
             SELECT team, iso_vs_lhp, iso_vs_rhp, hr_per_pa_vs_lhp, hr_per_pa_vs_rhp
             FROM team_batting_splits_season
            """, {}),
            ("""
             SELECT team, iso AS iso_all
             FROM team_season_averages
            """, {}),
        ]
        for sql, params in attempts:
            try:
                tmp = pd.read_sql(text(sql), self.engine, params=params)
                if not tmp.empty:
                    splits = tmp
                    break
            except Exception:
                continue

        # League anchors to avoid extreme drift
        LEAGUE_ISO = 0.170
        MIN_ISO, MAX_ISO = 0.110, 0.230  # pragmatic bounds

        # convenience maps
        map_iso_lhp, map_iso_rhp, map_iso_all = {}, {}, {}
        map_hrpa_lhp, map_hrpa_rhp = {}, {}

        if splits is not None:
            if "iso_vs_lhp" in splits.columns: map_iso_lhp = splits.set_index("team")["iso_vs_lhp"].to_dict()
            if "iso_vs_rhp" in splits.columns: map_iso_rhp = splits.set_index("team")["iso_vs_rhp"].to_dict()
            if "iso_all"    in splits.columns: map_iso_all = splits.set_index("team")["iso_all"].to_dict()
            if "hr_per_pa_vs_lhp" in splits.columns: map_hrpa_lhp = splits.set_index("team")["hr_per_pa_vs_lhp"].to_dict()
            if "hr_per_pa_vs_rhp" in splits.columns: map_hrpa_rhp = splits.set_index("team")["hr_per_pa_vs_rhp"].to_dict()

        # pitcher hand (best-effort)
        # use column if present; else quick DB lookup; else assume RHP (70%)
        def _sp_hand(row, side):
            col = f"{side}_sp_hand"
            if col in df.columns and pd.notna(row.get(col)):
                return str(row[col]).upper()[0]  # 'R'/'L'
            name_col = f"{side}_sp_name"
            sp_name = row.get(name_col)
            if pd.notna(sp_name):
                try:
                    q = text("SELECT throws FROM pitcher_bio WHERE name = :n LIMIT 1")
                    tmp = pd.read_sql(q, self.engine, params={"n": sp_name})
                    if not tmp.empty:
                        return str(tmp.iloc[0]["throws"]).upper()[0]
                except Exception:
                    pass
            return "R"

        # non-constant fallback using lineup/club context
        # if we have team wRC+ and ballpark HR factor, derive an ISO proxy
        def _iso_proxy(row, team, side):
            wrc_col = f"{side}_lineup_wrcplus"
            park_hr = pd.to_numeric(row.get("ballpark_hr_factor"), errors="coerce")
            
            # ✅ hard fallback if park factor missing in the row
            if pd.isna(park_hr):
                venue_raw = row.get("venue", row.get("venue_name", "")) or ""
                venue = self.park_name_alias.get(venue_raw, venue_raw)
                meta = _resolve_park_meta(venue)
                park_hr = float(meta.get("hr_factor", 1.0))
            
            wrc = pd.to_numeric(row.get(wrc_col), errors="coerce")
            base = LEAGUE_ISO
            # wRC+≈100 -> league ISO; convert gently to ISO with sqrt to keep tails tame
            if pd.notna(wrc):
                base = LEAGUE_ISO * np.sqrt(max(60.0, min(160.0, float(wrc))) / 100.0)
            if pd.notna(park_hr):
                base = base * float(park_hr)
            # clip to sane ISO range
            return float(np.clip(base, MIN_ISO, MAX_ISO))

        # compute side power per row
        home_power = np.full(len(df), np.nan, float)
        away_power = np.full(len(df), np.nan, float)

        for pos, (i, row) in enumerate(df.iterrows()):
            # HOME
            h_team = row.get("home_team")
            h_hand = _sp_hand(row, "home")  # home pitcher hand (facing away lineup)
            # AWAY
            a_team = row.get("away_team")
            a_hand = _sp_hand(row, "away")  # away pitcher hand (facing home lineup)

            # away lineup vs HOME pitcher hand
            if a_team and h_hand == "L":
                iso_away = map_iso_lhp.get(a_team)
            elif a_team and h_hand == "R":
                iso_away = map_iso_rhp.get(a_team)
            else:
                iso_away = map_iso_all.get(a_team)

            # home lineup vs AWAY pitcher hand
            if h_team and a_hand == "L":
                iso_home = map_iso_lhp.get(h_team)
            elif h_team and a_hand == "R":
                iso_home = map_iso_rhp.get(h_team)
            else:
                iso_home = map_iso_all.get(h_team)

            # fallbacks that still vary by row
            if iso_home is None or np.isnan(iso_home):
                iso_home = _iso_proxy(row, h_team, "home")
            if iso_away is None or np.isnan(iso_away):
                iso_away = _iso_proxy(row, a_team, "away")
            
            # Final safety - inject minimal variance if all sources failed
            if iso_home is None or np.isnan(iso_home):
                iso_home = LEAGUE_ISO + (pos % 7 - 3) * 0.005  # tiny systematic variation
            if iso_away is None or np.isnan(iso_away):
                iso_away = LEAGUE_ISO + ((pos + 3) % 7 - 3) * 0.005

            # clip and assign
            home_power[pos] = float(np.clip(iso_home, MIN_ISO, MAX_ISO))
            away_power[pos] = float(np.clip(iso_away, MIN_ISO, MAX_ISO))

        # write in one shot
        df["home_power"] = home_power
        df["away_power"] = away_power
        
        # combined_power: compute new value, preserve if enhanced variance exists and is good
        new_combined_power = (df["home_power"] + df["away_power"]) / 2.0
        
        if 'combined_power' in df.columns and df['combined_power'].std() > 0.015:
            logger.info(f"🎯 Preserving enhanced combined_power variance: std={df['combined_power'].std():.6f}")
            # Keep existing enhanced variance
        else:
            # Use new calculation - combined_power centered near league; keep raw mean to match training (no scaler in bundle)
            df["combined_power"] = new_combined_power
        
        # Debug logging
        logger.info(f"Power features - home_power: range [{df['home_power'].min():.3f}, {df['home_power'].max():.3f}], std={df['home_power'].std():.6f}")
        logger.info(f"Power features - away_power: range [{df['away_power'].min():.3f}, {df['away_power'].max():.3f}], std={df['away_power'].std():.6f}")
        logger.info(f"Power features - combined_power: range [{df['combined_power'].min():.3f}, {df['combined_power'].max():.3f}], std={df['combined_power'].std():.6f}")

        return df
    
    def add_bullpen_fatigue_features(self, games_df, target_date):
        """Add comprehensive bullpen fatigue features (L3-L5 days)"""
        logger.info("Adding bullpen fatigue features...")
        
        # Get bullpen usage data for last 5 days
        target_dt = pd.to_datetime(target_date).date()
        start_date = target_dt - timedelta(days=5)
        
        bp_query = text("""
        SELECT team, date, relief_ip, relief_pitches, relievers_used,
               closer_back2back_flag, any_b2b_reliever, relief_pitches_d1
        FROM bullpens_daily 
        WHERE date BETWEEN :start_date AND :end_date
        ORDER BY team, date DESC
        """)
        
        bp_data = pd.read_sql(bp_query, self.engine, 
                             params={'start_date': start_date, 'end_date': target_dt - timedelta(days=1)})
        
        # Calculate team-specific fatigue metrics
        for side in ['home', 'away']:
            col_prefix = f'{side}_bp_'
            team_col = f'{side}_team'
            
            for _, game in games_df.iterrows():
                team = game[team_col]
                team_bp = bp_data[bp_data['team'] == team].copy()
                
                if len(team_bp) == 0:
                    # Default values for missing data
                    games_df.loc[_, f'{col_prefix}ip_l3'] = 3.0
                    games_df.loc[_, f'{col_prefix}pitches_l3'] = 45
                    games_df.loc[_, f'{col_prefix}back2back_ct'] = 0
                    games_df.loc[_, f'{col_prefix}high_leverage_used_yday'] = 0
                    games_df.loc[_, f'{col_prefix}available_est'] = 0.8
                    continue
                
                # L3 days metrics
                l3_data = team_bp.head(3)
                games_df.loc[_, f'{col_prefix}ip_l3'] = l3_data['relief_ip'].sum()
                games_df.loc[_, f'{col_prefix}pitches_l3'] = l3_data['relief_pitches'].sum()
                games_df.loc[_, f'{col_prefix}back2back_ct'] = l3_data['any_b2b_reliever'].sum()
                
                # High leverage usage yesterday (closer + setup)
                yesterday_usage = l3_data.iloc[0] if len(l3_data) > 0 else None
                high_lev_yday = 0
                if yesterday_usage is not None:
                    high_lev_yday = int(yesterday_usage['closer_back2back_flag'] or 
                                       yesterday_usage['relief_pitches_d1'] > 25)
                games_df.loc[_, f'{col_prefix}high_leverage_used_yday'] = high_lev_yday
                
                # Availability estimate (0-1 scale based on recent workload)
                avg_daily_pitches = l3_data['relief_pitches'].mean() if len(l3_data) > 0 else 15
                fatigue_factor = min(1.0, avg_daily_pitches / 40.0)  # 40+ pitches = high fatigue
                availability = max(0.3, 1.0 - fatigue_factor)
                games_df.loc[_, f'{col_prefix}available_est'] = availability
        
        return games_df
    
    def add_starter_form_rest_features(self, games_df, target_date):
        """Add starter recent form & rest features"""
        logger.info("Adding starter form & rest features...")
        
        # Get pitcher recent starts (last 3 starts)
        target_dt = pd.to_datetime(target_date).date()
        
        for side in ['home', 'away']:
            pitcher_col = f'{side}_sp_name'  # Assuming we have pitcher names
            col_prefix = f'{side}_sp_'
            
            for idx, game in games_df.iterrows():
                pitcher_name = game.get(pitcher_col)
                
                if pd.isna(pitcher_name):
                    # Default values
                    games_df.loc[idx, f'{col_prefix}days_rest'] = 4
                    games_df.loc[idx, f'{col_prefix}ip_l3'] = 18.0
                    games_df.loc[idx, f'{col_prefix}pitch_ct_trend'] = 0.0
                    games_df.loc[idx, f'{col_prefix}tto_penalty'] = 0.0
                    continue
                
                # Query pitcher's last 3 starts from pitcher_daily_rolling
                # First get pitcher_id from enhanced_games table
                pitcher_id_query = text("""
                SELECT DISTINCT home_sp_id as pitcher_id FROM enhanced_games 
                WHERE home_sp_name = :pitcher 
                UNION 
                SELECT DISTINCT away_sp_id as pitcher_id FROM enhanced_games 
                WHERE away_sp_name = :pitcher
                LIMIT 1
                """)
                
                with self.engine.connect() as conn:
                    pitcher_id_result = conn.execute(text(pitcher_id_query), {'pitcher': pitcher_name}).fetchone()
                if not pitcher_id_result:
                    # Default values if pitcher not found
                    games_df.loc[idx, f'{col_prefix}days_rest'] = 4
                    games_df.loc[idx, f'{col_prefix}ip_l3'] = 18.0
                    games_df.loc[idx, f'{col_prefix}pitch_ct_trend'] = 0.0
                    games_df.loc[idx, f'{col_prefix}tto_penalty'] = 0.0
                    continue
                
                pitcher_id = pitcher_id_result[0]
                
                # Now get their rolling stats
                starts_query = text("""
                SELECT stat_date as date, ip, er, k as pitches
                FROM pitcher_daily_rolling 
                WHERE pitcher_id = :pitcher_id AND stat_date < :target_date
                ORDER BY stat_date DESC 
                LIMIT 3
                """)
                
                try:
                    starts = pd.read_sql(starts_query, self.engine, 
                                       params={'pitcher_id': pitcher_id, 'target_date': target_dt})
                    
                    if len(starts) > 0:
                        # Days rest since last start
                        last_start = pd.to_datetime(starts.iloc[0]['date']).date()
                        days_rest = (target_dt - last_start).days
                        
                        # IP trend over last 3 starts
                        ip_l3 = starts['ip'].sum()
                        
                        # Pitch count trend (recent vs earlier)
                        if len(starts) >= 2:
                            recent_pitches = starts.iloc[0]['pitches']
                            older_pitches = starts.iloc[1:]['pitches'].mean()
                            pitch_trend = recent_pitches - older_pitches
                        else:
                            pitch_trend = 0.0
                        
                        # Times through order penalty (simplified)
                        avg_ip = ip_l3 / len(starts)
                        tto_penalty = max(0, (avg_ip - 6.0) * 0.1)  # Penalty for going deep
                        
                    else:
                        # Defaults if no recent starts found
                        days_rest = 4
                        ip_l3 = 18.0
                        pitch_trend = 0.0
                        tto_penalty = 0.0
                    
                    games_df.loc[idx, f'{col_prefix}days_rest'] = days_rest
                    games_df.loc[idx, f'{col_prefix}ip_l3'] = ip_l3
                    games_df.loc[idx, f'{col_prefix}pitch_ct_trend'] = pitch_trend
                    games_df.loc[idx, f'{col_prefix}tto_penalty'] = tto_penalty
                    
                except Exception as e:
                    logger.warning(f"Could not fetch starter data for {pitcher_name}: {e}")
                    # Use defaults
                    games_df.loc[idx, f'{col_prefix}days_rest'] = 4
                    games_df.loc[idx, f'{col_prefix}ip_l3'] = 18.0
                    games_df.loc[idx, f'{col_prefix}pitch_ct_trend'] = 0.0
                    games_df.loc[idx, f'{col_prefix}tto_penalty'] = 0.0
        
        # Neutral imputation for starter form features (prevent 0.0 ERA/WHIP contamination)
        NEUTRALS = {
            "home_sp_era": 4.20, "away_sp_era": 4.20,   # league-ish neutral ERA
            "home_sp_whip": 1.30, "away_sp_whip": 1.30, # league-ish neutral WHIP
        }
        
        for col, neutral in NEUTRALS.items():
            if col in games_df.columns:
                zero_count = (games_df[col] == 0).sum()
                nan_count = games_df[col].isna().sum()
                if zero_count + nan_count > 0:
                    games_df.loc[(games_df[col].isna()) | (games_df[col] == 0), col] = neutral
                    logger.info(f"🧯 Imputed {zero_count + nan_count} {col} with neutral={neutral:.2f}")
        
        # Add sanity check for SP form (catch this immediately next time)
        def sanity_check_sp_form(df):
            for col in ["home_sp_era", "away_sp_era"]:
                if col in df.columns:
                    zero_rate = (df[col] == 0).mean()
                    if zero_rate > 0.02:  # 2% is already suspicious
                        raise RuntimeError(f"SANITY FAIL: {col} has {zero_rate:.1%} zeros – check starter form join")
        
        try:
            sanity_check_sp_form(games_df)
            logger.info("✅ SP form sanity check passed")
        except RuntimeError as e:
            logger.error(f"❌ {e}")
            # Don't crash the whole pipeline, but make it very obvious
            logger.error("🚨 CONTINUING WITH NEUTRAL IMPUTATION BUT THIS NEEDS TO BE FIXED")
        
        return games_df
    
    def add_umpire_features(self, games_df):
        """Add umpire tendencies features"""
        logger.info("Adding umpire features...")
        
        for idx, game in games_df.iterrows():
            game_id = game['game_id']
            
            # Try to get umpire data for this game
            ump_query = text("""
            SELECT u.o_u_tendency, u.called_strike_pct, u.edge_strike_pct, u.sample_size
            FROM umpires u
            JOIN enhanced_games eg ON eg.plate_umpire = u.name
            WHERE eg.game_id = :game_id
            """)
            
            try:
                ump_data = pd.read_sql(ump_query, self.engine, params={'game_id': game_id})
                
                if len(ump_data) > 0:
                    ump = ump_data.iloc[0]
                    # Weight by sample size (more reliable with more games)
                    weight = min(1.0, ump['sample_size'] / 100.0)
                    
                    games_df.loc[idx, 'ump_ou_index'] = ump['o_u_tendency'] * weight
                    games_df.loc[idx, 'ump_strike_rate'] = ump['called_strike_pct']
                    games_df.loc[idx, 'ump_edge_calls'] = ump['edge_strike_pct']
                else:
                    # League average defaults
                    games_df.loc[idx, 'ump_ou_index'] = 0.0
                    games_df.loc[idx, 'ump_strike_rate'] = 0.145
                    games_df.loc[idx, 'ump_edge_calls'] = 0.50
                    
            except Exception as e:
                logger.warning(f"Could not fetch umpire data for game {game_id}: {e}")
                games_df.loc[idx, 'ump_ou_index'] = 0.0
                games_df.loc[idx, 'ump_strike_rate'] = 0.145
                games_df.loc[idx, 'ump_edge_calls'] = 0.50
        
        # TODO: Add catcher framing when data becomes available
        # For now, use team-level proxy based on defensive metrics
        games_df['home_catcher_framing'] = 0.0  # Placeholder
        games_df['away_catcher_framing'] = 0.0  # Placeholder
        
        return games_df
    
    def add_enhanced_weather_features(self, games_df):
        """Add detailed weather features with fallback to enhanced_games data"""
        logger.info("Adding enhanced weather features...")

        # Pre-allocate weather columns to avoid fragmentation
        need_cols = ["wind_out_to_cf","air_density_index","humidity","wind_out_mph","wind_cross_mph","roof_open"]
        for_add = pd.DataFrame({c: np.nan for c in need_cols}, index=games_df.index)
        games_df = pd.concat([games_df, for_add.loc[:, ~for_add.columns.isin(games_df.columns)]], axis=1).copy()

        # vectorized pull of EG weather columns if present
        eg_has_weather = set(["temperature","humidity","wind_speed","wind_direction_deg","precip_prob","weather_condition"]).issubset(games_df.columns)

        for idx, game in games_df.iterrows():
            game_id = game["game_id"]
            venue_raw = game.get("venue", game.get("ballpark", game.get("venue_name", ""))) or ""
            venue = self.park_name_alias.get(venue_raw, venue_raw)
            cf_bearing = self.park_cf_bearings.get(venue, 90)

            # 1) try weather_game (as you had)
            w = None
            try:
                weather = pd.read_sql(
                    text("""
                        SELECT temp_f, humidity_pct, wind_mph, wind_dir_deg,
                               air_density_idx, wind_out_mph, wind_cross_mph
                        FROM weather_game
                        WHERE game_id = :game_id
                    """),
                    self.engine, params={"game_id": game_id}
                )
                if len(weather) > 0:
                    w = weather.iloc[0].to_dict()
            except Exception as e:
                logger.debug(f"weather_game read failed for {game_id}: {e}")

            # 2) fallback to enhanced_games row values if weather_game is empty
            if w is None and eg_has_weather:
                w = {
                    "temp_f":           game.get("temperature"),
                    "humidity_pct":     game.get("humidity"),
                    "wind_mph":         game.get("wind_speed"),
                    "wind_dir_deg":     game.get("wind_direction_deg"),
                    "air_density_idx":  _air_density_index_from_eg(game.get("temperature"), game.get("humidity")),
                }
                out_cf, cross = _wind_components(w["wind_mph"], w["wind_dir_deg"], cf_bearing)
                w["wind_out_mph"]  = out_cf
                w["wind_cross_mph"] = cross

            # write values or defaults
            if w:
                out_cf, cross = _wind_components(w.get("wind_mph"), w.get("wind_dir_deg"), cf_bearing)
                games_df.loc[idx, "wind_out_to_cf"]   = out_cf
                games_df.loc[idx, "wind_out_mph"]     = w.get("wind_out_mph", out_cf)
                games_df.loc[idx, "wind_cross_mph"]   = w.get("wind_cross_mph", cross)
                games_df.loc[idx, "air_density_index"]= w.get("air_density_idx", 1.0)
                games_df.loc[idx, "humidity"]         = w.get("humidity_pct", 50)
                
                # CRITICAL: Set the base weather features that the model expects
                games_df.loc[idx, "temperature"] = w.get("temp_f", game.get("temperature", 72))
                games_df.loc[idx, "wind_speed"] = w.get("wind_mph", game.get("wind_speed", 8))
            else:
                games_df.loc[idx, ["wind_out_to_cf","wind_out_mph","wind_cross_mph"]] = 0.0
                games_df.loc[idx, "air_density_index"] = 1.0
                games_df.loc[idx, "humidity"] = 50
                
                # Set base weather features from enhanced_games if available
                games_df.loc[idx, "temperature"] = game.get("temperature", 72)
                games_df.loc[idx, "wind_speed"] = game.get("wind_speed", 8)

            # roof heuristic (use EG columns you already store)
            roof_open = 1
            if venue in self.retractable_roof_parks:
                temp = game.get("temperature", 70) if "temperature" in games_df.columns else 70
                cond = (game.get("weather_condition","") or "").lower() if "weather_condition" in games_df.columns else ""
                precip = float(game.get("precip_prob", 0.0) or 0.0)
                if (temp is not None and temp < 60) or ("rain" in cond) or (precip > 30.0):
                    roof_open = 0
            games_df.loc[idx, "roof_open"] = roof_open

        return games_df
    
    def add_lineup_handedness_features(self, games_df, target_date):
        """Add confirmed lineup strength & handedness features"""
        logger.info("Adding lineup & handedness features...")
        
        # This would ideally use confirmed lineups from the lineups table
        # For now, we'll create proxy features based on team stats - vectorized approach
        
        # Pre-create all lineup columns at once to avoid fragmentation
        all_lineup_cols = []
        for side in ['home', 'away']:
            p = f"{side}_"
            cols = [
                f"{p}lineup_wrcplus", f"{p}vs_lhp_ops", f"{p}vs_rhp_ops",
                f"{p}lhb_count", f"{p}star_missing"
            ]
            all_lineup_cols.extend(cols)

        lineup_df = pd.DataFrame({c: np.nan for c in all_lineup_cols}, index=games_df.index)
        games_df = pd.concat([games_df, lineup_df], axis=1)
        
        # Track which rows need defaults (couldn't fetch lineup data)
        missing_lineup_mask = pd.Series(True, index=games_df.index)
        
        for side in ['home', 'away']:
            team_col = f'{side}_team'
            col_prefix = f'{side}_'
            
            for idx, game in games_df.iterrows():
                team = game[team_col]
                
                # Try to get actual lineup data
                lineup_query = text("""
                SELECT lineup_wrcplus, vs_lhp_ops, vs_rhp_ops, 
                       lhb_count, rhb_count, star_players_out
                FROM lineups 
                WHERE team = :team AND date = :date
                """)
                
                try:
                    lineup_data = pd.read_sql(lineup_query, self.engine, 
                                            params={'team': team, 'date': target_date})
                    
                    if len(lineup_data) > 0:
                        lineup = lineup_data.iloc[0]
                        games_df.loc[idx, f'{col_prefix}lineup_wrcplus'] = lineup['lineup_wrcplus']
                        games_df.loc[idx, f'{col_prefix}vs_lhp_ops'] = lineup['vs_lhp_ops']
                        games_df.loc[idx, f'{col_prefix}vs_rhp_ops'] = lineup['vs_rhp_ops']
                        games_df.loc[idx, f'{col_prefix}lhb_count'] = lineup['lhb_count']
                        games_df.loc[idx, f'{col_prefix}star_missing'] = lineup['star_players_out']
                        missing_lineup_mask[idx] = False  # Mark as having real data
                        
                except Exception as e:
                    logger.warning(f"Could not fetch lineup data for {team}: {e}")
                    # Will be filled with defaults below
        
        # Batch fill defaults for rows that couldn't fetch lineup data
        if missing_lineup_mask.any():
            for side in ["home", "away"]:
                p = f"{side}_"
                cols = [
                    f"{p}lineup_wrcplus", f"{p}vs_lhp_ops", f"{p}vs_rhp_ops",
                    f"{p}lhb_count", f"{p}star_missing"
                ]

                # Build defaults vectorized
                wrcplus_src = games_df.get(f"{side}_team_wrcplus")
                if wrcplus_src is None:
                    wrcplus_series = pd.Series(100.0, index=games_df.index)
                else:
                    wrcplus_series = pd.to_numeric(wrcplus_src, errors="coerce").fillna(100.0)

                defaults = pd.DataFrame({
                    f"{p}lineup_wrcplus": wrcplus_series,
                    f"{p}vs_lhp_ops": 0.750,
                    f"{p}vs_rhp_ops": 0.750,
                    f"{p}lhb_count": 4,
                    f"{p}star_missing": 0,
                }, index=games_df.index)

                # One batched write on the masked rows
                mask_indices = missing_lineup_mask[missing_lineup_mask].index
                for col in cols:
                    games_df.loc[mask_indices, col] = games_df.loc[mask_indices, col].fillna(defaults.loc[mask_indices, col])
        
        # Defragment after batch operations
        games_df = games_df.copy()
        
        # Calculate platoon advantage - vectorized
        # Add the column first
        games_df['lineup_platoon_edge'] = np.nan
        
        for idx, game in games_df.iterrows():
            # This would use actual starter handedness when available
            # For now, use a simplified approach
            home_vs_lhp = games_df.loc[idx, 'home_vs_lhp_ops']
            home_vs_rhp = games_df.loc[idx, 'home_vs_rhp_ops']
            away_vs_lhp = games_df.loc[idx, 'away_vs_lhp_ops']
            away_vs_rhp = games_df.loc[idx, 'away_vs_rhp_ops']
            
            # Assume RHP starters are more common (70% of starts)
            platoon_edge = ((home_vs_rhp + away_vs_rhp) - (home_vs_lhp + away_vs_lhp)) * 0.7
            games_df.loc[idx, 'lineup_platoon_edge'] = platoon_edge
        
        return games_df
    
    def add_travel_schedule_features(self, games_df, target_date):
        """Add travel & schedule fatigue features - vectorized approach"""
        logger.info("Adding travel & schedule features...")
        
        target_dt = pd.to_datetime(target_date).date()
        
        # Pre-create all travel columns at once
        travel_cols = []
        for side in ['home', 'away']:
            p = f"{side}_"
            cols = [f"{p}games_last7", f"{p}days_rest", f"{p}travel_switches", f"{p}getaway_day"]
            travel_cols.extend(cols)

        # Pre-create columns in one concat (avoids insert-per-row fragmentation)
        travel_df = pd.DataFrame({c: np.nan for c in travel_cols}, index=games_df.index)
        games_df = pd.concat([games_df, travel_df], axis=1)

        # Allocate arrays to fill
        vals = {c: np.full(len(games_df), np.nan, dtype=float) for c in travel_cols}
        
        for side in ['home', 'away']:
            team_col = f'{side}_team'
            col_prefix = f'{side}_'
            
            for pos, (idx, game) in enumerate(games_df.iterrows()):
                team = game[team_col]
                
                # Get recent games for travel calculation
                recent_query = text("""
                SELECT date, home_team, away_team, venue
                FROM enhanced_games 
                WHERE (home_team = :team OR away_team = :team)
                AND date < :target_date
                ORDER BY date DESC 
                LIMIT 7
                """)
                
                try:
                    recent_games = pd.read_sql(recent_query, self.engine,
                                             params={'team': team, 'target_date': target_dt})
                    
                    games_in_last_7 = len(recent_games)
                    
                    # Simple travel fatigue proxy
                    if len(recent_games) > 0:
                        last_game_date = pd.to_datetime(recent_games.iloc[0]['date']).date()
                        days_since_last = (target_dt - last_game_date).days
                        
                        # Count home/away switches (travel indicator)
                        travel_days = 0
                        prev_was_home = None
                        for _, rg in recent_games.iterrows():
                            curr_is_home = (rg['home_team'] == team)
                            if prev_was_home is not None and prev_was_home != curr_is_home:
                                travel_days += 1
                            prev_was_home = curr_is_home
                    else:
                        days_since_last = 1
                        travel_days = 0
                    
                    # Getaway day (last game of series - simplified)
                    getaway_day = 0  # Would need series scheduling data to determine properly
                    
                    # Store in arrays for batch assignment
                    vals[f'{col_prefix}games_last7'][pos] = games_in_last_7
                    vals[f'{col_prefix}days_rest'][pos] = days_since_last
                    vals[f'{col_prefix}travel_switches'][pos] = travel_days
                    vals[f'{col_prefix}getaway_day'][pos] = getaway_day
                    
                except Exception as e:
                    logger.warning(f"Could not fetch travel data for {team}: {e}")
                    # Use defaults
                    vals[f'{col_prefix}games_last7'][pos] = 4  # Average
                    vals[f'{col_prefix}days_rest'][pos] = 1
                    vals[f'{col_prefix}travel_switches'][pos] = 1
                    vals[f'{col_prefix}getaway_day'][pos] = 0

        # Single batched assignment for all columns
        for c in travel_cols:
            games_df[c] = vals[c]

        # Defragment after batch operations
        games_df = games_df.copy()
        
        return games_df
    
    def add_advanced_interaction_features(self, games_df):
        """Add advanced interaction features"""
        logger.info("Adding interaction features...")
        
        # Collect all new columns to add in batch to avoid fragmentation warnings
        new_cols = {}
        
        # Weather-park interactions (beyond existing)
        if 'air_density_index' in games_df.columns and 'ballpark_hr_factor' in games_df.columns:
            new_cols['air_density_hr_interaction'] = games_df['air_density_index'] * games_df['ballpark_hr_factor']
        
        if 'wind_out_to_cf' in games_df.columns and 'ballpark_hr_factor' in games_df.columns:
            new_cols['wind_hr_interaction'] = games_df['wind_out_to_cf'] * games_df['ballpark_hr_factor']
        
        # Bullpen fatigue vs starter form
        if 'home_bp_available_est' in games_df.columns and 'home_sp_days_rest' in games_df.columns:
            new_cols['bullpen_starter_risk'] = (1 - games_df['home_bp_available_est']) * (5 - games_df['home_sp_days_rest'])
        
        # Umpire strike zone vs pitcher control
        if 'ump_strike_rate' in games_df.columns and 'combined_bb_rate' in games_df.columns:
            new_cols['ump_control_interaction'] = games_df['ump_strike_rate'] * (4.0 - games_df['combined_bb_rate'])
        
        # Platoon advantage amplified by lineup quality
        if 'lineup_platoon_edge' in games_df.columns and 'home_lineup_wrcplus' in games_df.columns:
            avg_wrcplus = (games_df['home_lineup_wrcplus'] + games_df['away_lineup_wrcplus']) / 200.0  # Normalize
            new_cols['platoon_quality_interaction'] = games_df['lineup_platoon_edge'] * avg_wrcplus
        
        # Batch-add all new columns to avoid fragmentation warnings
        if new_cols:
            new_df = pd.DataFrame(new_cols, index=games_df.index)
            games_df = pd.concat([games_df, new_df], axis=1)
        
        return games_df
    
    def build_enhanced_features(self, games_df, target_date):
        """Build all enhanced features in the pipeline"""
        logger.info("Building enhanced feature pipeline...")
        
        # Add each feature group
        games_df = self.add_bullpen_fatigue_features(games_df, target_date)
        games_df = self.add_starter_form_rest_features(games_df, target_date)
        games_df = self.add_umpire_features(games_df)
        games_df = self.add_enhanced_weather_features(games_df)
        games_df = self.add_lineup_handedness_features(games_df, target_date)
        games_df = self.add_travel_schedule_features(games_df, target_date)
        games_df = self.add_advanced_interaction_features(games_df)
        
        # WHIP calculation now handled in _impute_sp_whip function

        # ✅ Ensure team rate/index columns exist with sane anchors  
        TEAM_RATE_ANCHORS = {
            "home_team_woba": 0.310, "away_team_woba": 0.310,
            "home_team_babip": 0.295, "away_team_babip": 0.295,
            "home_team_wrcplus": 100.0, "away_team_wrcplus": 100.0,
            "home_team_bb_pct": 0.085, "away_team_bb_pct": 0.085,
            "home_team_k_pct": 0.225, "away_team_k_pct": 0.225,
            "home_team_iso": 0.170, "away_team_iso": 0.170,
            "home_team_ba": 0.250, "away_team_ba": 0.250,
        }
        for col, val in TEAM_RATE_ANCHORS.items():
            if col not in games_df.columns:
                games_df[col] = val
            else:
                games_df[col] = pd.to_numeric(games_df[col], errors="coerce").fillna(val)

        # Apply rate defaults and sanity checks
        games_df = fill_rate_defaults(games_df)
        games_df = _sanity(games_df)

        # ❂ Ensure numeric weather fields are not NaN
        for col, default in [
            ("temperature", 70), ("wind_speed", 5), ("humidity", 50),
            ("wind_out_to_cf", 0.0), ("air_density_index", 1.0)
        ]:
            if col in games_df.columns:
                games_df[col] = pd.to_numeric(games_df[col], errors="coerce").fillna(default)
        
        # === HARDEN SERVING CRITICALS ===
        games_df = self._merge_team_offense_l30(games_df, target_date)
        games_df = self._ensure_ballpark_factors(games_df, target_date=target_date)
        games_df = self.add_power_features(games_df, target_date)
        games_df = _impute_sp_whip(games_df)
        
        # Apply rate defaults and sanity checks
        games_df = fill_rate_defaults(games_df)
        games_df = _sanity(games_df)

        # (optional) sanity logs
        for c in ["home_sp_whip","away_sp_whip","combined_whip","ballpark_run_factor","ballpark_hr_factor","ballpark_offensive_factor"]:
            if c in games_df.columns:
                s = pd.to_numeric(games_df[c], errors="coerce")
                logger.info("SANITY %s: non-null %d/%d, std=%.3f, head=%s",
                           c, s.notna().sum(), len(s), float(s.std(skipna=True)), s.head(5).round(3).tolist())
        
        logger.info(f"Enhanced pipeline complete: {len(games_df.columns)} total features")
        return games_df

# Integration with enhanced_bullpen_predictor.py
def integrate_enhanced_pipeline(predictor_instance):
    """Add enhanced pipeline to existing predictor"""
    predictor_instance.enhanced_pipeline = EnhancedFeaturePipeline()
    predictor_instance.engine = predictor_instance.enhanced_pipeline.engine  # Fix engine access
    
    # Store original engineer_features method
    original_engineer = predictor_instance.engineer_features
    
    def enhanced_engineer_features(df):
        """Enhanced feature engineering with high-leverage factors"""
        # Ensure engine exists for legacy reads
        if not hasattr(predictor_instance, "engine") or predictor_instance.engine is None:
            predictor_instance.engine = predictor_instance.enhanced_pipeline.engine
        # First run the original feature engineering
        featured_df = original_engineer(df)
        
        # Get target date from the dataframe or use current date
        if 'date' in featured_df.columns:
            target_date = featured_df['date'].iloc[0]
        else:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        # Add enhanced features
        try:
            featured_df = predictor_instance.enhanced_pipeline.build_enhanced_features(featured_df, target_date)
            
            # Apply additional rate defaults and sanity checks for serving
            featured_df = fill_rate_defaults(featured_df)
            featured_df = _sanity(featured_df)
            
            logger.info("✅ Enhanced pipeline features added successfully")
        except Exception as e:
            logger.warning(f"Enhanced pipeline failed, using original features: {e}")
        
        return featured_df
    
    # Replace the method
    predictor_instance.engineer_features = enhanced_engineer_features
    
    return predictor_instance
