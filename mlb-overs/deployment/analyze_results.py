#!/usr/bin/env python3
import pandas as pd
from sqlalchemy import create_engine, text
import os

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

with engine.begin() as conn:
    # Get today's probability predictions
    preds = pd.read_sql(text('''
        SELECT game_id, p_over, ev_over, ev_under, kelly_over, kelly_under, adj_edge
        FROM probability_predictions 
        WHERE game_date = '2025-08-17'
        ORDER BY game_id
    '''), conn)
    
    print('📊 Final filtered results:')
    print(f'Games analyzed: {len(preds)}')
    print(f'Prob range: {preds["p_over"].min():.3f} to {preds["p_over"].max():.3f}')
    print(f'EV range: {preds[["ev_over","ev_under"]].stack().min():.3f} to {preds[["ev_over","ev_under"]].stack().max():.3f}')
    print(f'Kelly range: {preds[["kelly_over","kelly_under"]].stack().min():.3f} to {preds[["kelly_over","kelly_under"]].stack().max():.3f}')
    print(f'Edge range: {preds["adj_edge"].min():.2f} to {preds["adj_edge"].max():.2f}')
    
    # Show actual recommendations
    print('\n🎯 Betting positions:')
    for _, row in preds.iterrows():
        best_ev = max(row['ev_over'], row['ev_under'])
        best_kelly = max(row['kelly_over'], row['kelly_under'])
        side = 'OVER' if row['ev_over'] > row['ev_under'] else 'UNDER'
        print(f'  {row["game_id"]}: {side} | EV={best_ev:+.3f} | Kelly={best_kelly:.3f} | Edge={row["adj_edge"]:+.2f}')
