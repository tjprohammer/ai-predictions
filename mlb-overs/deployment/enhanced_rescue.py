# enhanced_rescue.py
import numpy as np
import pandas as pd
import zlib

NEUTRAL_BP_ERA = 4.20

def _team_prior(team: str, base=NEUTRAL_BP_ERA, spread=0.8) -> float:
    """
    Deterministic, per-team prior around league average.
    +/- spread/2 (default +/-0.4). Stable across runs.
    """
    if not isinstance(team, str) or not team:
        return base
    h = zlib.adler32(team.encode("utf-8")) % 10000
    u = h / 10000.0  # [0,1)
    return base + (u - 0.5) * spread  # base +/- spread/2

FEATURE_ALIASES = {
    "home_team_runs_pg": "home_team_rpg_l30",
    "away_team_runs_pg": "away_team_rpg_l30", 
    "combined_team_offense": "combined_offense_rpg",
}

NEUTRALS = {
    "home_bp_era": 4.20, "away_bp_era": 4.20,
    "home_bp_fip": 4.20, "away_bp_fip": 4.20,
    "home_sp_era": 4.20, "away_sp_era": 4.20,
    "home_sp_whip": 1.30, "away_sp_whip": 1.30,
    "home_sp_k_per_9": 8.0, "away_sp_k_per_9": 8.0,
    "home_sp_bb_per_9": 3.0, "away_sp_bb_per_9": 3.0,
    "ballpark_run_factor": 1.00, "temperature": 78.0, "wind_speed": 8.0,
}

def rescue_serving_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rescue high-importance features that are constant/missing at serving time."""
    df = df.copy()
    
    # Apply feature aliases (training name -> serving name mapping)
    for dst, src in FEATURE_ALIASES.items():
        if dst in df.columns and src in df.columns:
            if df[dst].isna().all() or df[dst].nunique(dropna=True) <= 1:
                df[dst] = df[src]
    
    # --- bullpen priors with team-specific variance ---
    for side in ("home","away"):
        era_col, fip_col, team_col = f"{side}_bp_era", f"{side}_bp_fip", f"{side}_team"
        # if we have FIP use it; else team prior around 4.20
        if era_col not in df.columns or df[era_col].isna().all():
            if fip_col in df.columns and pd.to_numeric(df[fip_col], errors="coerce").notna().any():
                df[era_col] = pd.to_numeric(df[fip_col], errors="coerce")
            else:
                # deterministic per-team prior
                if team_col in df.columns:
                    df[era_col] = df[team_col].astype(str).apply(_team_prior)
                else:
                    df[era_col] = NEUTRAL_BP_ERA

    # small, deterministic micro-jitter to avoid exact ties if still constant
    for side in ("home","away"):
        era_col, team_col = f"{side}_bp_era", f"{side}_team"
        s = pd.to_numeric(df[era_col], errors="coerce")
        if s.nunique(dropna=True) <= 1:
            if team_col in df.columns:
                jit = df[team_col].astype(str).apply(lambda t: (_team_prior(t, 0.0, 0.10)))
                df[era_col] = (s.fillna(NEUTRAL_BP_ERA) + jit).clip(3.0, 6.5)

    # cheap FIP proxy if needed (keeps model columns non-empty)
    for side in ("home","away"):
        era_col, fip_col = f"{side}_bp_era", f"{side}_bp_fip"
        if fip_col in df.columns and df[fip_col].isna().all():
            df[fip_col] = pd.to_numeric(df[era_col], errors="coerce") - 0.15
    
    # existing SP quality signal (unchanged)
    for side in ("home","away"):
        sp_era = pd.to_numeric(df.get(f"{side}_sp_era", 4.2), errors="coerce").fillna(4.2)
        bp_era = pd.to_numeric(df.get(f"{side}_bp_era", 4.2), errors="coerce").fillna(4.2)
        k9     = pd.to_numeric(df.get(f"{side}_sp_k_per_9", 8.0), errors="coerce").fillna(8.0)
        bb9    = pd.to_numeric(df.get(f"{side}_sp_bb_per_9", 3.0), errors="coerce").fillna(3.0)
        df[f"{side}_pitcher_quality"] = (-0.6*sp_era) + (-0.4*bp_era) + 0.08*k9 - 0.06*bb9

    # recompute bullpen quality with actual/prior ERA
    home_bp = pd.to_numeric(df["home_bp_era"], errors="coerce")
    away_bp = pd.to_numeric(df["away_bp_era"], errors="coerce")
    df["combined_bullpen_quality"] = -0.5*(home_bp + away_bp)
    
    if "combined_bullpen_era" in df.columns and df["combined_bullpen_era"].isna().all():
        df["combined_bullpen_era"] = 0.5 * (
            pd.to_numeric(df["home_bp_era"], errors='coerce').fillna(NEUTRALS["home_bp_era"]) +
            pd.to_numeric(df["away_bp_era"], errors='coerce').fillna(NEUTRALS["away_bp_era"])
        )
    
    if "era_differential" in df.columns and df["era_differential"].isna().all():
        if "home_sp_era" in df.columns and "away_sp_era" in df.columns:
            df["era_differential"] = (
                pd.to_numeric(df["home_sp_era"], errors='coerce').fillna(NEUTRALS["home_sp_era"]) -
                pd.to_numeric(df["away_sp_era"], errors='coerce').fillna(NEUTRALS["away_sp_era"])
            )
    
    # Nice-to-have: reduce "All-NaN features" count
    for col, base in {
        "home_team_power":  df.get("home_power", 0.17),
        "away_team_power":  df.get("away_power", 0.17),
        "combined_team_power": (pd.to_numeric(df.get("home_power", 0.17), errors="coerce").fillna(0.17) +
                                pd.to_numeric(df.get("away_power", 0.17), errors="coerce").fillna(0.17))/2.0,
    }.items():
        if col in df.columns and df[col].isna().all():
            df[col] = base

    # environmental stuff (unchanged)
    run  = pd.to_numeric(df.get("ballpark_run_factor", 1.00), errors="coerce").fillna(1.00)
    temp = pd.to_numeric(df.get("temperature", 78.0), errors="coerce").fillna(78.0)
    wind = pd.to_numeric(df.get("wind_speed", 8.0), errors="coerce").fillna(8.0)
    temp_factor = 1.0 + 0.015*(temp - 75.0)
    wind_factor = 1.0 + 0.02*(wind/10.0)
    df["expected_offensive_environment"] = (run * temp_factor * wind_factor).clip(0.80, 1.30)
    df["temp_park_interaction"] = (temp_factor - 1.0) * run

    # global "pitching_dominance" (unchanged)
    df["pitching_dominance"] = (
        -0.5*(pd.to_numeric(df.get("home_sp_whip", 1.3), errors="coerce").fillna(1.3) +
              pd.to_numeric(df.get("away_sp_whip", 1.3), errors="coerce").fillna(1.3))
        + 0.15*(pd.to_numeric(df.get("home_sp_k_per_9", 8.0), errors="coerce").fillna(8.0) +
                pd.to_numeric(df.get("away_sp_k_per_9", 8.0), errors="coerce").fillna(8.0))
        - 0.12*(pd.to_numeric(df.get("home_sp_bb_per_9", 3.0), errors="coerce").fillna(3.0) +
                pd.to_numeric(df.get("away_sp_bb_per_9", 3.0), errors="coerce").fillna(3.0))
    )

    return df

def backfill_and_alias_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """Fill the 25 placeholder-only features that show up as NaN at serving time."""
    df = df.copy()
    
    # === ALIASES ===
    # Park offensive factor alias
    if 'ballpark_offensive_factor' in df.columns and 'park_offensive_factor' in df.columns:
        park_val = df['park_offensive_factor']
        ballpark_val = df['ballpark_offensive_factor']
        # Only fillna if park_offensive_factor has NaN values
        if park_val.isna().any():
            df['park_offensive_factor'] = park_val.fillna(ballpark_val)
    
    # === WIND & ENVIRONMENT ===
    wind_factor = pd.to_numeric(df.get('wind_factor', 1.0), errors='coerce').fillna(1.0)
    df['wind_out'] = (wind_factor > 1.0).astype(float)
    df['wind_in'] = (wind_factor < 1.0).astype(float)
    
    wind_speed = pd.to_numeric(df.get('wind_speed', 0), errors='coerce').fillna(0)
    ballpark_hr = pd.to_numeric(df.get('ballpark_hr_factor', 1.0), errors='coerce').fillna(1.0)
    df['wind_park_interaction'] = wind_speed * (ballpark_hr - 1.0)
    
    air_density = pd.to_numeric(df.get('air_density_index', 1.0), errors='coerce').fillna(1.0)
    df['air_density_hr_interaction'] = air_density * (ballpark_hr - 1.0)
    
    wind_out_mph = pd.to_numeric(df.get('wind_out_mph', 0), errors='coerce').fillna(0)
    wind_in_mph = pd.to_numeric(df.get('wind_in_mph', 0), errors='coerce').fillna(0)
    df['wind_hr_interaction'] = wind_out_mph - wind_in_mph
    
    roof_open = pd.to_numeric(df.get('roof_open', 1.0), errors='coerce').fillna(1.0)
    df['is_dome'] = (1.0 - roof_open.clip(0, 1))
    
    # Simple rain proxy (no precip data, so default to 0)
    df['is_rain'] = 0.0
    
    # === BULLPEN ===
    home_bp_fip = pd.to_numeric(df.get('home_bp_fip', 4.2), errors='coerce').fillna(4.2)
    away_bp_fip = pd.to_numeric(df.get('away_bp_fip', 4.2), errors='coerce').fillna(4.2)
    df['combined_bullpen_fip'] = (home_bp_fip + away_bp_fip) / 2
    
    home_bp_era = pd.to_numeric(df.get('home_bp_era', 4.2), errors='coerce').fillna(4.2)
    away_bp_era = pd.to_numeric(df.get('away_bp_era', 4.2), errors='coerce').fillna(4.2)
    df['bullpen_era_advantage'] = home_bp_era - away_bp_era
    df['bullpen_era_differential'] = df['bullpen_era_advantage']  # Keep both names for compatibility
    
    # Bullpen fatigue calculation
    for side in ['home', 'away']:
        ip_l3 = pd.to_numeric(df.get(f'{side}_bp_ip_l3', 0), errors='coerce').fillna(0)
        pitches_l3 = pd.to_numeric(df.get(f'{side}_bp_pitches_l3', 0), errors='coerce').fillna(0)
        back2back = pd.to_numeric(df.get(f'{side}_bp_back2back_ct', 0), errors='coerce').fillna(0)
        high_lev = pd.to_numeric(df.get(f'{side}_bp_high_leverage_used_yday', 0), errors='coerce').fillna(0)
        available = pd.to_numeric(df.get(f'{side}_bp_available_est', 0.3), errors='coerce').fillna(0.3)
        
        fatigue = (ip_l3 * 1.0 + pitches_l3 * 0.4 + back2back * 0.8 + high_lev * 0.6 - available * 1.0)
        df[f'{side}_bullpen_fatigue'] = fatigue
    
    df['combined_bullpen_fatigue'] = (df['home_bullpen_fatigue'] + df['away_bullpen_fatigue']) / 2
    
    # Bullpen reliability from ERA std or IP proxy
    home_era_std = pd.to_numeric(df.get('home_bp_era_std', 0.5), errors='coerce').fillna(0.5)
    away_era_std = pd.to_numeric(df.get('away_bp_era_std', 0.5), errors='coerce').fillna(0.5)
    df['combined_bullpen_reliability'] = 1.0 / (1e-6 + (home_era_std + away_era_std) / 2)
    
    # === WEIGHTED PITCHING ===
    # Simple weighted ERA (2/3 starter, 1/3 bullpen)
    home_sp_era = pd.to_numeric(df.get('home_sp_era', 4.2), errors='coerce').fillna(4.2)
    away_sp_era = pd.to_numeric(df.get('away_sp_era', 4.2), errors='coerce').fillna(4.2)
    
    df['home_weighted_pitching_era'] = (2/3) * home_sp_era + (1/3) * home_bp_era
    df['away_weighted_pitching_era'] = (2/3) * away_sp_era + (1/3) * away_bp_era
    df['combined_weighted_pitching_era'] = (df['home_weighted_pitching_era'] + df['away_weighted_pitching_era']) / 2
    
    # === OFFENSE ===
    home_woba = pd.to_numeric(df.get('home_team_woba', 0.31), errors='coerce').fillna(0.31)
    away_woba = pd.to_numeric(df.get('away_team_woba', 0.31), errors='coerce').fillna(0.31)
    df['combined_team_woba'] = (home_woba + away_woba) / 2
    
    home_wrcplus = pd.to_numeric(df.get('home_team_wrcplus', 100.0), errors='coerce').fillna(100.0)
    away_wrcplus = pd.to_numeric(df.get('away_team_wrcplus', 100.0), errors='coerce').fillna(100.0)
    df['combined_team_wrcplus'] = (home_wrcplus + away_wrcplus) / 2
    
    df['offensive_balance'] = abs(home_wrcplus - away_wrcplus)
    
    # Team discipline (walk/K rates)
    for side in ['home', 'away']:
        bb_pct = pd.to_numeric(df.get(f'{side}_team_bb_pct', 0.085), errors='coerce').fillna(0.085)
        k_pct = pd.to_numeric(df.get(f'{side}_team_k_pct', 0.23), errors='coerce').fillna(0.23)
        df[f'{side}_team_discipline'] = bb_pct - k_pct  # Positive = more disciplined
    
    df['combined_team_discipline'] = (df['home_team_discipline'] + df['away_team_discipline']) / 2
    
    # === MARKET COMPARISON ===
    # Simple market vs team total (if both exist)
    market_total = pd.to_numeric(df.get('market_total', np.nan), errors='coerce')
    team_rpg = pd.to_numeric(df.get('combined_offense_rpg', np.nan), errors='coerce')
    df['market_vs_team_total'] = (market_total - team_rpg).fillna(0.0)
    
    # Replace any remaining infs with 0
    df = df.replace([np.inf, -np.inf], 0.0)
    
    return df
