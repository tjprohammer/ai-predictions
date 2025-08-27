import os
from sqlalchemy import create_engine, text
import pandas as pd

engine = create_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'))

# Check which model features are available in enhanced_games
expected_features = [
    'home_sp_k', 'home_sp_bb', 'home_sp_h', 'away_sp_k', 'away_sp_bb', 'away_sp_h',
    'home_team_hits', 'home_team_rbi', 'home_team_lob', 'away_team_hits', 'away_team_rbi', 'away_team_lob',
    'over_odds', 'under_odds', 'predicted_total', 'edge',
    'home_sp_season_k', 'away_sp_season_k', 'home_sp_season_bb', 'away_sp_season_bb',
    'home_bp_k', 'home_bp_bb', 'home_bp_h', 'away_bp_k', 'away_bp_bb', 'away_bp_h',
    'home_team_wrc_plus', 'away_team_wrc_plus', 'home_team_stolen_bases', 'away_team_stolen_bases',
    'home_team_plate_appearances', 'away_team_plate_appearances', 'pressure', 'feels_like_temp',
    'home_bullpen_whip_l30', 'away_bullpen_whip_l30', 'home_bullpen_usage_rate', 'away_bullpen_usage_rate',
    'home_team_ops_l14', 'away_team_ops_l14', 'home_team_form_rating', 'away_team_form_rating', 'home_team_ops_l20'
]

with engine.connect() as conn:
    # Check enhanced_games columns
    eg_result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'enhanced_games' ORDER BY column_name"))
    eg_cols = [row[0] for row in eg_result]
    
    print(f'Enhanced_games has {len(eg_cols)} columns')
    print('\nModel feature availability in enhanced_games:')
    
    available_features = []
    missing_features = []
    
    for feat in expected_features:
        is_available = feat in eg_cols
        if is_available:
            available_features.append(feat)
        else:
            missing_features.append(feat)
        print(f'  {feat}: {"✅" if is_available else "❌"}')
    
    print(f'\nSummary:')
    print(f'  Available: {len(available_features)} features')
    print(f'  Missing: {len(missing_features)} features')
    
    if available_features:
        print(f'\nAvailable features to add to query:')
        for feat in available_features:
            print(f'  eg.{feat},')
    
    if missing_features:
        print(f'\nMissing features (will need placeholders):')
        for feat in missing_features:
            print(f'  {feat}')
