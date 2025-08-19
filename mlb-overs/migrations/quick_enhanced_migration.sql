-- Quick Enhanced Features Migration Script
-- ========================================
-- Simple ALTER TABLE commands to add missing columns

-- Enhanced Games Table - Add missing columns
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS plate_umpire TEXT;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS umpire_ou_tendency NUMERIC DEFAULT 0.0;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS venue TEXT;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS ballpark TEXT;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS roof_type TEXT DEFAULT 'open';
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS roof_status TEXT DEFAULT 'open';
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS humidity INTEGER;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS wind_direction_deg INTEGER;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS air_pressure NUMERIC;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS dew_point INTEGER;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS home_sp_hand CHAR(1);
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS away_sp_hand CHAR(1);
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS home_sp_days_rest INTEGER DEFAULT 4;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS away_sp_days_rest INTEGER DEFAULT 4;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS series_game INTEGER;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS getaway_day BOOLEAN DEFAULT FALSE;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS doubleheader BOOLEAN DEFAULT FALSE;
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS day_after_night BOOLEAN DEFAULT FALSE;

-- Lineups Table - Add missing columns
ALTER TABLE lineups ADD COLUMN IF NOT EXISTS date DATE;
ALTER TABLE lineups ADD COLUMN IF NOT EXISTS lineup_wrcplus INTEGER DEFAULT 100;
ALTER TABLE lineups ADD COLUMN IF NOT EXISTS vs_lhp_ops NUMERIC DEFAULT 0.750;
ALTER TABLE lineups ADD COLUMN IF NOT EXISTS vs_rhp_ops NUMERIC DEFAULT 0.750;
ALTER TABLE lineups ADD COLUMN IF NOT EXISTS lhb_count INTEGER DEFAULT 4;
ALTER TABLE lineups ADD COLUMN IF NOT EXISTS rhb_count INTEGER DEFAULT 5;
ALTER TABLE lineups ADD COLUMN IF NOT EXISTS star_players_out INTEGER DEFAULT 0;
ALTER TABLE lineups ADD COLUMN IF NOT EXISTS lineup_confirmed BOOLEAN DEFAULT FALSE;

-- Update venue column with existing venue_name data
UPDATE enhanced_games 
SET venue = venue_name 
WHERE venue IS NULL AND venue_name IS NOT NULL;

-- Set default roof types for known stadiums
UPDATE enhanced_games 
SET roof_type = CASE 
    WHEN venue_name ILIKE '%Tropicana%' THEN 'dome'
    WHEN venue_name ILIKE '%Minute Maid%' THEN 'retractable'
    WHEN venue_name ILIKE '%Rogers Centre%' THEN 'retractable'
    WHEN venue_name ILIKE '%Chase Field%' THEN 'retractable'
    WHEN venue_name ILIKE '%T-Mobile Park%' THEN 'retractable'
    WHEN venue_name ILIKE '%American Family Field%' THEN 'retractable'
    WHEN venue_name ILIKE '%Marlins Park%' THEN 'retractable'
    ELSE 'open'
END
WHERE roof_type = 'open' OR roof_type IS NULL;

-- Set default roof status
UPDATE enhanced_games 
SET roof_status = CASE 
    WHEN roof_type = 'dome' THEN 'closed'
    ELSE 'open'
END
WHERE roof_status = 'open' OR roof_status IS NULL;

-- Create supporting tables if they don't exist

-- Pitcher comprehensive stats table (enhanced)
CREATE TABLE IF NOT EXISTS pitcher_comprehensive_stats (
    id SERIAL PRIMARY KEY,
    pitcher_name TEXT NOT NULL,
    date DATE NOT NULL,
    game_id TEXT,
    team TEXT,
    opponent TEXT,
    ip NUMERIC,
    er INTEGER,
    h INTEGER,
    bb INTEGER,
    k INTEGER,
    hr INTEGER,
    pitches INTEGER,
    strikes INTEGER,
    balls INTEGER,
    game_score INTEGER,
    fip NUMERIC,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Catcher framing stats table
CREATE TABLE IF NOT EXISTS catcher_framing_stats (
    id SERIAL PRIMARY KEY,
    catcher_name TEXT NOT NULL,
    team TEXT,
    date DATE NOT NULL,
    framing_runs NUMERIC DEFAULT 0.0,
    strike_rate NUMERIC DEFAULT 0.5,
    called_strikes INTEGER DEFAULT 0,
    called_balls INTEGER DEFAULT 0,
    edge_calls INTEGER DEFAULT 0,
    edge_strikes INTEGER DEFAULT 0,
    csaa NUMERIC DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Team travel log table
CREATE TABLE IF NOT EXISTS team_travel_log (
    id SERIAL PRIMARY KEY,
    team TEXT NOT NULL,
    date DATE NOT NULL,
    venue TEXT,
    venue_city TEXT,
    venue_state TEXT,
    venue_timezone TEXT,
    travel_distance_miles INTEGER DEFAULT 0,
    timezone_change INTEGER DEFAULT 0,
    games_in_last_7 INTEGER DEFAULT 0,
    home_away_switch BOOLEAN DEFAULT FALSE,
    cross_country_travel BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create useful indexes
CREATE INDEX IF NOT EXISTS idx_enhanced_games_plate_umpire ON enhanced_games(plate_umpire);
CREATE INDEX IF NOT EXISTS idx_enhanced_games_venue ON enhanced_games(venue);
CREATE INDEX IF NOT EXISTS idx_enhanced_games_roof ON enhanced_games(roof_type, roof_status);
CREATE INDEX IF NOT EXISTS idx_enhanced_games_pitcher_names ON enhanced_games(home_sp_name, away_sp_name);
CREATE INDEX IF NOT EXISTS idx_lineups_team_date ON lineups(team, date DESC);
CREATE INDEX IF NOT EXISTS idx_lineups_game_team ON lineups(game_id, team);
CREATE INDEX IF NOT EXISTS idx_pitcher_stats_name_date ON pitcher_comprehensive_stats(pitcher_name, date DESC);
CREATE INDEX IF NOT EXISTS idx_pitcher_stats_game_id ON pitcher_comprehensive_stats(game_id);
CREATE INDEX IF NOT EXISTS idx_catcher_framing_name_date ON catcher_framing_stats(catcher_name, date DESC);
CREATE INDEX IF NOT EXISTS idx_travel_team_date ON team_travel_log(team, date DESC);
