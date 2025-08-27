-- ===============================
-- ULTRA80: schema additions
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

-- If you already store a local start time + tz elsewhere, consider normalizing
-- those into scheduled_start_utc during ingest, rather than storing local times.

-- Generated chronological key we can ORDER BY (UTC). Uses 7pm stub as last resort.
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name='enhanced_games' AND column_name='start_ts'
  ) THEN
    ALTER TABLE enhanced_games
      ADD COLUMN start_ts timestamptz
      GENERATED ALWAYS AS (
        COALESCE(
          scheduled_start_utc,
          created_at,
          (date + INTERVAL '19 hours')
        )
      ) STORED;
  END IF;
END$$;

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_enhanced_games_start_ts ON enhanced_games (start_ts);
CREATE INDEX IF NOT EXISTS idx_enhanced_games_date      ON enhanced_games (date);
CREATE INDEX IF NOT EXISTS idx_enhanced_games_game_id   ON enhanced_games (game_id);

-- real_market_games is a view that already contains opening_total and closing_total
-- No changes needed for real_market_games

-- Optional FK and constraints commented out since real_market_games is a view
-- ALTER TABLE real_market_games
--   ADD CONSTRAINT real_market_games_game_id_fkey
--   FOREIGN KEY (game_id) REFERENCES enhanced_games(game_id) ON DELETE CASCADE;

-- Optional: unique constraints if your data model guarantees one row per game
-- ALTER TABLE enhanced_games     ADD CONSTRAINT enhanced_games_game_id_key UNIQUE (game_id);
-- ALTER TABLE real_market_games  ADD CONSTRAINT real_market_games_game_id_key UNIQUE (game_id);
