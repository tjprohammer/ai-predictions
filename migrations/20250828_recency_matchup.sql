-- Recency + Matchup Features Migration
-- File: migrations/20250828_recency_matchup.sql
-- Adds pitcher last-start, handedness splits, team vs-hand rolling stats

-- Pitcher meta + last start (home/away)
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS home_sp_hand CHAR(1);
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS away_sp_hand CHAR(1);
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS home_sp_last_ip NUMERIC;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS home_sp_last_runs NUMERIC;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS home_sp_last_pitch_count INTEGER;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS home_sp_days_rest INTEGER;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS away_sp_last_ip NUMERIC;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS away_sp_last_runs NUMERIC;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS away_sp_last_pitch_count INTEGER;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS away_sp_days_rest INTEGER;

-- Team offense vs hand (rolling & season baselines)
-- HOME hitting vs R/L
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS home_team_wRCplus_vs_R_l7 NUMERIC;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS home_team_wRCplus_vs_R_l14 NUMERIC;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS home_team_wRCplus_vs_R_l30 NUMERIC;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS home_team_wRCplus_vs_R_season NUMERIC;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS home_team_PA_vs_R_l7 INTEGER;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS home_team_PA_vs_R_l14 INTEGER;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS home_team_PA_vs_R_l30 INTEGER;

ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS home_team_wRCplus_vs_L_l7 NUMERIC;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS home_team_wRCplus_vs_L_l14 NUMERIC;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS home_team_wRCplus_vs_L_l30 NUMERIC;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS home_team_wRCplus_vs_L_season NUMERIC;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS home_team_PA_vs_L_l7 INTEGER;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS home_team_PA_vs_L_l14 INTEGER;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS home_team_PA_vs_L_l30 INTEGER;

-- AWAY hitting vs R/L
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS away_team_wRCplus_vs_R_l7 NUMERIC;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS away_team_wRCplus_vs_R_l14 NUMERIC;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS away_team_wRCplus_vs_R_l30 NUMERIC;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS away_team_wRCplus_vs_R_season NUMERIC;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS away_team_PA_vs_R_l7 INTEGER;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS away_team_PA_vs_R_l14 INTEGER;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS away_team_PA_vs_R_l30 INTEGER;

ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS away_team_wRCplus_vs_L_l7 NUMERIC;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS away_team_wRCplus_vs_L_l14 NUMERIC;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS away_team_wRCplus_vs_L_l30 NUMERIC;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS away_team_wRCplus_vs_L_season NUMERIC;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS away_team_PA_vs_L_l7 INTEGER;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS away_team_PA_vs_L_l14 INTEGER;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS away_team_PA_vs_L_l30 INTEGER;

-- Lineup R/L mix (optional; safe defaults)
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS home_lineup_pct_R NUMERIC;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS away_lineup_pct_R NUMERIC;

-- Bullpen proxies (optional; filled by ingestor when available)
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS home_bullpen_era_l14 NUMERIC;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS away_bullpen_era_l14 NUMERIC;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_enhanced_games_sp_ids ON enhanced_games(home_sp_id, away_sp_id);
CREATE INDEX IF NOT EXISTS idx_enhanced_games_teams_date ON enhanced_games(home_team, away_team, date);

-- Log completion
INSERT INTO migration_log (migration_name, applied_at) 
VALUES ('20250828_recency_matchup', CURRENT_TIMESTAMP)
ON CONFLICT (migration_name) DO NOTHING;
