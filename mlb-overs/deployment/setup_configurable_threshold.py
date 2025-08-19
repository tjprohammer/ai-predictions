#!/usr/bin/env python3
"""Setup configurable threshold system for production flexibility"""

import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
engine = create_engine(DATABASE_URL)

# Create config table
setup_sql = text("""
-- Create model configuration table
CREATE TABLE IF NOT EXISTS model_config (
    key text PRIMARY KEY, 
    value text NOT NULL,
    updated_at timestamp DEFAULT CURRENT_TIMESTAMP
);

-- Insert optimal threshold based on analysis
INSERT INTO model_config(key, value) VALUES ('edge_threshold', '2.0')
ON CONFLICT (key) DO UPDATE SET 
    value = EXCLUDED.value,
    updated_at = CURRENT_TIMESTAMP;

-- Create configurable view
CREATE OR REPLACE VIEW api_games_today AS
WITH cfg AS (
  SELECT CAST(value AS numeric) AS thr FROM model_config WHERE key='edge_threshold'
)
SELECT eg.game_id, eg.date, eg.home_team, eg.away_team, eg.game_time_utc,
       eg.venue_name, eg.roof_type, eg.market_total, eg.predicted_total,
       ROUND(eg.predicted_total - eg.market_total, 2) AS edge,
       CASE
         WHEN eg.predicted_total IS NULL OR eg.market_total IS NULL THEN NULL
         WHEN (eg.predicted_total - eg.market_total) >= (SELECT thr FROM cfg) THEN 'OVER'
         WHEN (eg.predicted_total - eg.market_total) <= -(SELECT thr FROM cfg) THEN 'UNDER'
         ELSE 'NO BET'
       END AS recommendation,
       eg.home_sp_name, eg.away_sp_name, eg.home_sp_season_era, eg.away_sp_season_era,
       eg.temperature, eg.wind_speed, eg.wind_direction, eg.humidity
FROM enhanced_games eg
WHERE eg.date = CURRENT_DATE AND eg.total_runs IS NULL
ORDER BY eg.game_time_utc NULLS LAST, eg.game_id;
""")

with engine.begin() as conn:
    conn.execute(setup_sql)
    
print('✅ Setup configurable threshold system')
print('   - Created model_config table')
print('   - Set initial threshold: 2.0')
print('   - Updated api_games_today view to use configurable threshold')
print('')
print('🔧 To change threshold in future:')
print("   UPDATE model_config SET value='2.25' WHERE key='edge_threshold';")
