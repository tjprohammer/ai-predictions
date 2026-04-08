ALTER TABLE player_game_pitching ADD COLUMN IF NOT EXISTS late_innings_pitched NUMERIC(4,1);
ALTER TABLE player_game_pitching ADD COLUMN IF NOT EXISTS late_runs_allowed SMALLINT;
ALTER TABLE player_game_pitching ADD COLUMN IF NOT EXISTS late_earned_runs SMALLINT;
ALTER TABLE player_game_pitching ADD COLUMN IF NOT EXISTS late_hits_allowed SMALLINT;

ALTER TABLE bullpens_daily ADD COLUMN IF NOT EXISTS late_innings_pitched NUMERIC(4,1);
ALTER TABLE bullpens_daily ADD COLUMN IF NOT EXISTS late_relievers_used SMALLINT;
ALTER TABLE bullpens_daily ADD COLUMN IF NOT EXISTS late_runs_allowed SMALLINT;
ALTER TABLE bullpens_daily ADD COLUMN IF NOT EXISTS late_earned_runs SMALLINT;
ALTER TABLE bullpens_daily ADD COLUMN IF NOT EXISTS late_hits_allowed SMALLINT;
ALTER TABLE bullpens_daily ADD COLUMN IF NOT EXISTS late_era NUMERIC(6,2);