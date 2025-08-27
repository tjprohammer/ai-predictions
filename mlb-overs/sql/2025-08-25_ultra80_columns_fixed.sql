-- ===============================
-- ULTRA80: schema additions (FIXED)
-- ===============================

-- --- enhanced_games: fields we read for leak-free training/prediction
ALTER TABLE enhanced_games
  ADD COLUMN IF NOT EXISTS scheduled_start_utc  timestamptz,
  ADD COLUMN IF NOT EXISTS created_at           timestamptz DEFAULT now(),
  ADD COLUMN IF NOT EXISTS temperature          numeric,
  ADD COLUMN IF NOT EXISTS wind_speed           numeric,
  ADD COLUMN IF NOT EXISTS venue_name           text,
  ADD COLUMN IF NOT EXISTS home_sp_id           bigint,
  ADD COLUMN IF NOT EXISTS away_sp_id           bigint;

-- Add start_ts as a regular column (not generated) to avoid immutability issues
ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS start_ts timestamptz;

-- Update start_ts with the computed value for existing rows
UPDATE enhanced_games 
SET start_ts = COALESCE(
  scheduled_start_utc,
  created_at,
  (date + INTERVAL '19 hours')
)
WHERE start_ts IS NULL;

-- Create a function to automatically update start_ts on insert/update
CREATE OR REPLACE FUNCTION update_start_ts()
RETURNS TRIGGER AS $$
BEGIN
  NEW.start_ts = COALESCE(
    NEW.scheduled_start_utc,
    NEW.created_at,
    (NEW.date + INTERVAL '19 hours')
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically update start_ts
DROP TRIGGER IF EXISTS trigger_update_start_ts ON enhanced_games;
CREATE TRIGGER trigger_update_start_ts
  BEFORE INSERT OR UPDATE ON enhanced_games
  FOR EACH ROW
  EXECUTE FUNCTION update_start_ts();

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_enhanced_games_start_ts ON enhanced_games (start_ts);
CREATE INDEX IF NOT EXISTS idx_enhanced_games_date      ON enhanced_games (date);
CREATE INDEX IF NOT EXISTS idx_enhanced_games_game_id   ON enhanced_games (game_id);

-- real_market_games is a view that already contains opening_total and closing_total
-- No changes needed for real_market_games
