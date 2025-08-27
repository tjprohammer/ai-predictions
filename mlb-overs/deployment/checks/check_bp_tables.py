import os
from sqlalchemy import create_engine, text
import pandas as pd

engine = create_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'))

# Check what columns exist in legitimate_game_features vs enhanced_games
with engine.connect() as conn:
    # Check legitimate_game_features columns
    lgf_result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'legitimate_game_features' ORDER BY column_name"))
    lgf_cols = [row[0] for row in lgf_result]
    print(f'legitimate_game_features has {len(lgf_cols)} columns')
    
    # Check enhanced_games columns
    eg_result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'enhanced_games' ORDER BY column_name"))
    eg_cols = [row[0] for row in eg_result]
    print(f'enhanced_games has {len(eg_cols)} columns')
    
    # Find BP columns in each
    lgf_bp = [col for col in lgf_cols if 'bp' in col.lower()]
    eg_bp = [col for col in eg_cols if 'bp' in col.lower()]
    
    print(f'\nBP columns in legitimate_game_features: {lgf_bp}')
    print(f'BP columns in enhanced_games: {eg_bp}')
    
    # Check which model features are missing
    model_features = ['home_bp_k', 'home_bp_bb', 'home_bp_h', 'away_bp_k', 'away_bp_bb', 'away_bp_h']
    print('\nModel feature availability:')
    for feat in model_features:
        in_lgf = feat in lgf_cols
        in_eg = feat in eg_cols
        print(f'  {feat}: LGF={in_lgf}, EG={in_eg}')
