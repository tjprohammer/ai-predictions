ALTER TABLE game_features_totals ADD COLUMN IF NOT EXISTS market_sportsbook TEXT;
ALTER TABLE game_features_first5_totals ADD COLUMN IF NOT EXISTS market_sportsbook TEXT;

ALTER TABLE predictions_totals ADD COLUMN IF NOT EXISTS market_sportsbook TEXT;
ALTER TABLE predictions_totals ADD COLUMN IF NOT EXISTS market_snapshot_ts TIMESTAMPTZ;

ALTER TABLE predictions_first5_totals ADD COLUMN IF NOT EXISTS market_sportsbook TEXT;
ALTER TABLE predictions_first5_totals ADD COLUMN IF NOT EXISTS market_snapshot_ts TIMESTAMPTZ;

ALTER TABLE prediction_outcomes_daily ADD COLUMN IF NOT EXISTS entry_market_sportsbook TEXT;
ALTER TABLE prediction_outcomes_daily ADD COLUMN IF NOT EXISTS entry_market_snapshot_ts TIMESTAMPTZ;
ALTER TABLE prediction_outcomes_daily ADD COLUMN IF NOT EXISTS closing_market_same_sportsbook BOOLEAN;

ALTER TABLE recommendation_history ADD COLUMN IF NOT EXISTS entry_market_sportsbook TEXT;
ALTER TABLE recommendation_history ADD COLUMN IF NOT EXISTS entry_market_snapshot_ts TIMESTAMPTZ;
ALTER TABLE recommendation_history ADD COLUMN IF NOT EXISTS closing_market_same_sportsbook BOOLEAN;
