-- Add market line timing columns to enhanced_games table
-- This allows us to track opening, closing, and live market lines

-- Add opening line columns
ALTER TABLE enhanced_games 
ADD COLUMN IF NOT EXISTS opening_total DECIMAL(3,1),
ADD COLUMN IF NOT EXISTS opening_over_odds INTEGER,
ADD COLUMN IF NOT EXISTS opening_under_odds INTEGER,
ADD COLUMN IF NOT EXISTS opening_captured_at TIMESTAMP;

-- Add live line columns
ALTER TABLE enhanced_games 
ADD COLUMN IF NOT EXISTS live_total DECIMAL(3,1),
ADD COLUMN IF NOT EXISTS live_over_odds INTEGER,
ADD COLUMN IF NOT EXISTS live_under_odds INTEGER,
ADD COLUMN IF NOT EXISTS live_captured_at TIMESTAMP;

-- Add closing line captured timestamp
ALTER TABLE enhanced_games 
ADD COLUMN IF NOT EXISTS closing_captured_at TIMESTAMP;

-- Add market source tracking
ALTER TABLE enhanced_games 
ADD COLUMN IF NOT EXISTS market_source VARCHAR(50) DEFAULT 'estimated';

-- Update existing records to mark as estimated
UPDATE enhanced_games 
SET market_source = 'estimated' 
WHERE market_source IS NULL;

-- Create index for efficient querying by capture time
CREATE INDEX IF NOT EXISTS idx_enhanced_games_opening_captured 
ON enhanced_games(opening_captured_at);

CREATE INDEX IF NOT EXISTS idx_enhanced_games_closing_captured 
ON enhanced_games(closing_captured_at);

-- Create view for real market data analysis
CREATE OR REPLACE VIEW real_market_games AS
SELECT 
    game_id,
    date,
    home_team,
    away_team,
    total_runs,
    predicted_total_original,
    predicted_total_learning,
    opening_total,
    market_total as closing_total,
    live_total,
    opening_captured_at,
    closing_captured_at,
    live_captured_at,
    market_source,
    -- Calculate line movements
    CASE 
        WHEN opening_total IS NOT NULL AND market_total IS NOT NULL 
        THEN market_total - opening_total 
        ELSE NULL 
    END as line_movement_open_to_close,
    
    CASE 
        WHEN market_total IS NOT NULL AND live_total IS NOT NULL 
        THEN live_total - market_total 
        ELSE NULL 
    END as line_movement_close_to_live,
    
    -- Mark games with real vs estimated data
    CASE 
        WHEN market_source = 'estimated' THEN 'estimated'
        WHEN opening_captured_at IS NOT NULL OR closing_captured_at IS NOT NULL THEN 'real'
        ELSE 'unknown'
    END as data_quality
FROM enhanced_games;

COMMENT ON VIEW real_market_games IS 'View showing games with real vs estimated market data for learning model training';
