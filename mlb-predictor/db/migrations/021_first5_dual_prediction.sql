-- First-5 totals: store fundamentals-only prediction alongside market-calibrated
ALTER TABLE predictions_first5_totals ADD COLUMN IF NOT EXISTS predicted_total_fundamentals NUMERIC(6,3);
