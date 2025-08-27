# ULTRA80 Database Migration Instructions

## 1. Migration File Created

The migration file has been saved as: `s:\Projects\AI_Predictions\mlb-overs\sql\2025-08-25_ultra80_columns.sql`

## 2. Running the Migration

Since you have your S:\Projects\AI_Predictions\mlb-overs\sql mounted to /SQL in the container, run:

```powershell
# From Windows PowerShell
docker ps  # find your postgres container name, e.g., ai_predictions_pg

# Run the migration
docker exec -i ai_predictions_pg psql -U mlbuser -d mlb -f /SQL/2025-08-25_ultra80_columns.sql
```

## 3. What This Migration Adds

### enhanced_games table:

- `scheduled_start_utc` (timestamptz) - UTC start time
- `created_at` (timestamptz) - Record creation timestamp
- `temperature` (numeric) - Game temperature
- `wind_speed` (numeric) - Wind speed
- `venue_name` (text) - Venue name
- `home_sp_id` (bigint) - Home starting pitcher ID
- `away_sp_id` (bigint) - Away starting pitcher ID
- `start_ts` (generated column) - Computed chronological timestamp for ordering

### real_market_games table:

- `opening_total` (numeric) - Pre-game market total
- `closing_total` (numeric) - Closing market total (optional)

### Indexes Added:

- `idx_enhanced_games_start_ts` - For chronological ordering
- `idx_enhanced_games_date` - For date filtering
- `idx_enhanced_games_game_id` - For joins
- `idx_real_market_games_game_id` - For joins

## 4. Python Code Updated

The `incremental_ultra_80_system.py` has been updated to use the new simplified query that leverages the `start_ts` generated column for proper chronological ordering.

## 5. Safe to Run Multiple Times

This migration is idempotent - you can run it multiple times safely. It uses `ADD COLUMN IF NOT EXISTS` and conditional logic to avoid errors.
