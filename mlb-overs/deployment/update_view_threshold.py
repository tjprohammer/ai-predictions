#!/usr/bin/env python3
"""Update api_games_today view with threshold 2.0"""

import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
engine = create_engine(DATABASE_URL)

sql = text("""
CREATE OR REPLACE VIEW api_games_today AS
SELECT eg.game_id, eg.date, eg.home_team, eg.away_team, eg.game_time_utc,
       eg.venue_name, eg.roof_type, eg.market_total, eg.predicted_total,
       ROUND(eg.predicted_total - eg.market_total,2) AS edge,
       CASE
         WHEN eg.predicted_total IS NULL OR eg.market_total IS NULL THEN NULL
         WHEN (eg.predicted_total - eg.market_total) >=  2.0 THEN 'OVER'
         WHEN (eg.predicted_total - eg.market_total) <= -2.0 THEN 'UNDER'
         ELSE 'NO BET'
       END AS recommendation,
       eg.home_sp_name, eg.away_sp_name, eg.home_sp_season_era, eg.away_sp_season_era,
       eg.temperature, eg.wind_speed, eg.wind_direction, eg.humidity
FROM enhanced_games eg
WHERE eg.date = CURRENT_DATE AND eg.total_runs IS NULL
ORDER BY eg.game_time_utc NULLS LAST, eg.game_id
""")

with engine.begin() as conn:
    conn.execute(sql)
    
print('✅ Updated api_games_today view with threshold 2.0')
print('   - Recommends OVER/UNDER only for |edge| >= 2.0')
print('   - Expected: ~4 bets/day, 56.9% win rate, +8.5% ROI')
