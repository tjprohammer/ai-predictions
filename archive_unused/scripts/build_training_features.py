"""
Builds a complete training feature set from historical game data.
"""
import pandas as pd
import numpy as np
import argparse
from sqlalchemy import create_engine
import os

# Import functions from infer.py to ensure consistency
from models.infer import team_key_from_any, rolling_team, last10_sp_agg, get_pitcher_era_stats_infer

def build_training_features(historical_data_path: str, database_url: str):
    """
    Loads historical game data and builds a complete feature set for model training.
    """
    print(f"Loading historical data from {historical_data_path}...")
    df = pd.read_parquet(historical_data_path)
    df['date'] = pd.to_datetime(df['date'])

    print(f"Loaded {len(df)} games. Now building features...")

    # We need a database connection to get ERA stats
    eng = create_engine(database_url)

    # Load team offense and bullpen data for rolling stats
    off = pd.read_sql("SELECT team, date, xwoba, iso, bb_pct, k_pct FROM teams_offense_daily", eng)
    off["date"] = pd.to_datetime(off["date"]).dt.date
    off["off_key"] = off["team"].astype(str).apply(team_key_from_any)

    bp = pd.read_sql("SELECT team, date, bp_fip, closer_back2back_flag FROM bullpens_daily", eng)
    bp["date"] = pd.to_datetime(bp["date"]).dt.date
    bp["bp_key"] = bp["team"].astype(str).apply(team_key_from_any)

    sp = pd.read_sql("SELECT pitcher_id, date, xwoba_allowed, csw_pct, velo_fb, pitches FROM pitchers_starts", eng)
    sp["date"] = pd.to_datetime(sp["date"]).dt.date

    features = []
    total_games = len(df)

    for i, g in df.iterrows():
        d = g["date"]
        home_team = g["home_team"]
        away_team = g["away_team"]
        hkey = team_key_from_any(home_team)
        akey = team_key_from_any(away_team)
        hsp = int(g["home_sp_id"]) if pd.notna(g["home_sp_id"]) else None
        asp = int(g["away_sp_id"]) if pd.notna(g["away_sp_id"]) else None

        # --- Feature Calculation (mirrors infer.py) ---
        ho14 = rolling_team(off, hkey, d, 14); ho30 = rolling_team(off, hkey, d, 30)
        ao14 = rolling_team(off, akey, d, 14); ao30 = rolling_team(off, akey, d, 30)

        hsp_era = get_pitcher_era_stats_infer(eng, hsp, d) if hsp else {}
        asp_era = get_pitcher_era_stats_infer(eng, asp, d) if asp else {}
        hsp_era_vs_opp = get_pitcher_era_stats_infer(eng, hsp, d, opponent_team=akey) if hsp else {}
        asp_era_vs_opp = get_pitcher_era_stats_infer(eng, asp, d, opponent_team=hkey) if asp else {}

        league_era = 4.20
        hsp_season_era = float(hsp_era.get("era_season", league_era) or league_era)
        asp_season_era = float(asp_era.get("era_season", league_era) or league_era)
        
        feature_dict = {
            "game_id": g["game_id"],
            "date": d,
            "target_y": g["total_runs"],
            "k_close": g["market_total"],
            "home_sp_era_season": hsp_season_era,
            "away_sp_era_season": asp_season_era,
            "home_sp_era_l3": float(hsp_era.get("era_l3", hsp_season_era) or hsp_season_era),
            "home_sp_era_l5": float(hsp_era.get("era_l5", hsp_season_era) or hsp_season_era),
            "home_sp_era_l10": float(hsp_era.get("era_l10", hsp_season_era) or hsp_season_era),
            "home_sp_era_vs_opp": float(hsp_era_vs_opp.get("era_vs_opp", hsp_season_era) or hsp_season_era),
            "away_sp_era_l3": float(asp_era.get("era_l3", asp_season_era) or asp_season_era),
            "away_sp_era_l5": float(asp_era.get("era_l5", asp_season_era) or asp_season_era),
            "away_sp_era_l10": float(asp_era.get("era_l10", asp_season_era) or asp_season_era),
            "away_sp_era_vs_opp": float(asp_era_vs_opp.get("era_vs_opp", asp_season_era) or asp_season_era),
            # Add other features from infer.py here...
        }
        features.append(feature_dict)

        if (i + 1) % 100 == 0:
            print(f"  Built features for {i + 1}/{total_games} games...")

    features_df = pd.DataFrame(features)
    print(f"Successfully built features for {len(features_df)} games.")
    return features_df

def main():
    parser = argparse.ArgumentParser(description="Build training features from historical data.")
    parser.add_argument("--year", type=int, default=2025, help="Year of the historical data.")
    parser.add_argument("--data-dir", type=str, default="s:/Projects/AI_Predictions/mlb-overs/data", help="Directory with historical data.")
    parser.add_argument("--out-dir", type=str, default="s:/Projects/AI_Predictions/mlb-overs/features", help="Output directory for features.")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb"))
    args = parser.parse_args()

    historical_data_path = f"{args.data_dir}/historical_games_{args.year}.parquet"
    features_df = build_training_features(historical_data_path, args.database_url)

    output_path = f"{args.out_dir}/training_features_{args.year}.parquet"
    features_df.to_parquet(output_path, index=False)

    print(f"✅ Training features saved to {output_path}")
    print("\nSample of training features:")
    print(features_df.head())

if __name__ == "__main__":
    main()
