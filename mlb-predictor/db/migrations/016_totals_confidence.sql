-- Add confidence / suppression columns to full-game totals predictions.
-- All full-game totals are research-only until the lane proves signal;
-- the collapse detector suppresses entire slates when predictions are too flat.
ALTER TABLE predictions_totals ADD COLUMN IF NOT EXISTS confidence_level TEXT;
ALTER TABLE predictions_totals ADD COLUMN IF NOT EXISTS suppress_reason TEXT;
ALTER TABLE predictions_totals ADD COLUMN IF NOT EXISTS lane_status TEXT DEFAULT 'research_only';
