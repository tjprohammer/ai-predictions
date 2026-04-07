-- Add confidence level and suppress reason to first5 predictions for publish/suppress logic
ALTER TABLE predictions_first5_totals ADD COLUMN IF NOT EXISTS confidence_level TEXT;
ALTER TABLE predictions_first5_totals ADD COLUMN IF NOT EXISTS suppress_reason TEXT;
