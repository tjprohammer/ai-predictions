ALTER TABLE prediction_outcomes_daily ADD COLUMN IF NOT EXISTS forecast_temperature_f NUMERIC(5,1);
ALTER TABLE prediction_outcomes_daily ADD COLUMN IF NOT EXISTS forecast_wind_speed_mph NUMERIC(5,1);
ALTER TABLE prediction_outcomes_daily ADD COLUMN IF NOT EXISTS forecast_humidity_pct NUMERIC(5,1);
ALTER TABLE prediction_outcomes_daily ADD COLUMN IF NOT EXISTS observed_temperature_f NUMERIC(5,1);
ALTER TABLE prediction_outcomes_daily ADD COLUMN IF NOT EXISTS observed_wind_speed_mph NUMERIC(5,1);
ALTER TABLE prediction_outcomes_daily ADD COLUMN IF NOT EXISTS observed_humidity_pct NUMERIC(5,1);
ALTER TABLE prediction_outcomes_daily ADD COLUMN IF NOT EXISTS weather_delta_temperature_f NUMERIC(5,1);
