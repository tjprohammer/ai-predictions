-- Lane quality from training artifact (above_baseline / below_baseline); copied at predict time.
ALTER TABLE predictions_first5_totals ADD COLUMN IF NOT EXISTS lane_status TEXT;
