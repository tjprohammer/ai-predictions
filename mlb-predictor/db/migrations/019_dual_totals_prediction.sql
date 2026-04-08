-- Dual prediction: store fundamentals-only prediction alongside market-calibrated
ALTER TABLE predictions_totals ADD COLUMN IF NOT EXISTS predicted_total_fundamentals NUMERIC(6,3);

-- Strikeouts: store fundamentals-only prediction  
ALTER TABLE predictions_pitcher_strikeouts ADD COLUMN IF NOT EXISTS predicted_strikeouts_fundamentals NUMERIC(6,3);

-- Strikeouts: new differentiation features stored for diagnostics
ALTER TABLE game_features_pitcher_strikeouts ADD COLUMN IF NOT EXISTS pitcher_vs_team_k_rate NUMERIC(6,4);
ALTER TABLE game_features_pitcher_strikeouts ADD COLUMN IF NOT EXISTS opponent_lineup_k_pct_recent NUMERIC(6,4);
ALTER TABLE game_features_pitcher_strikeouts ADD COLUMN IF NOT EXISTS venue_k_factor NUMERIC(6,4);
