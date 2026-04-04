ALTER TABLE prediction_outcomes_daily ADD COLUMN IF NOT EXISTS closing_market_sportsbook TEXT;
ALTER TABLE prediction_outcomes_daily ADD COLUMN IF NOT EXISTS closing_market_line NUMERIC(8,4);
ALTER TABLE prediction_outcomes_daily ADD COLUMN IF NOT EXISTS closing_market_snapshot_ts TIMESTAMPTZ;
ALTER TABLE prediction_outcomes_daily ADD COLUMN IF NOT EXISTS clv_line_delta NUMERIC(8,4);
ALTER TABLE prediction_outcomes_daily ADD COLUMN IF NOT EXISTS clv_side_value NUMERIC(8,4);
ALTER TABLE prediction_outcomes_daily ADD COLUMN IF NOT EXISTS beat_closing_line BOOLEAN;

ALTER TABLE recommendation_history ADD COLUMN IF NOT EXISTS closing_market_sportsbook TEXT;
ALTER TABLE recommendation_history ADD COLUMN IF NOT EXISTS closing_market_line NUMERIC(8,4);
ALTER TABLE recommendation_history ADD COLUMN IF NOT EXISTS clv_line_delta NUMERIC(8,4);
ALTER TABLE recommendation_history ADD COLUMN IF NOT EXISTS clv_side_value NUMERIC(8,4);
ALTER TABLE recommendation_history ADD COLUMN IF NOT EXISTS beat_closing_line BOOLEAN;

ALTER TABLE model_scorecards_daily ADD COLUMN IF NOT EXISTS avg_clv_side_value NUMERIC(8,4);
ALTER TABLE model_scorecards_daily ADD COLUMN IF NOT EXISTS positive_clv_rate NUMERIC(6,4);
ALTER TABLE model_scorecards_daily ADD COLUMN IF NOT EXISTS clv_count INTEGER DEFAULT 0;
