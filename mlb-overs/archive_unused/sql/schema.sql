-- games
CREATE TABLE IF NOT EXISTS games (
  game_id TEXT PRIMARY KEY,
  date DATE NOT NULL,
  home_team TEXT,
  away_team TEXT,
  park_id TEXT,
  roof_status TEXT,
  ump_id TEXT,
  home_sp_id INTEGER,         -- INTEGER here
  away_sp_id INTEGER,         -- INTEGER here
  is_divisional BOOL,
  series_game_no INT,
  doubleheader_flag BOOL,
  home_travel_miles_72h INT,
  away_travel_miles_72h INT,
  tz_shift_hours INT,
  open_total NUMERIC,
  open_total_juice INT,
  curr_total NUMERIC,
  curr_total_juice INT,
  close_total NUMERIC,
  close_total_juice INT,
  total_runs INT
);

-- pitchers_starts
CREATE TABLE IF NOT EXISTS pitchers_starts (
  start_id TEXT PRIMARY KEY,
  game_id TEXT REFERENCES games(game_id),
  pitcher_id TEXT,
  team TEXT,
  opp_team TEXT,
  is_home BOOL,
  date DATE,
  ip NUMERIC,
  h INT,
  bb INT,
  k INT,
  hr INT,
  r INT,
  er INT,
  bf INT,
  pitches INT,
  csw_pct NUMERIC,
  velo_fb NUMERIC,
  velo_delta_3g NUMERIC,
  hh_pct_allowed NUMERIC,
  barrel_pct_allowed NUMERIC,
  avg_ev_allowed NUMERIC,
  xwoba_allowed NUMERIC,
  xslg_allowed NUMERIC,
  era_game NUMERIC,
  fip_game NUMERIC,
  xfip_game NUMERIC,
  siera_game NUMERIC,
  opp_lineup_l_pct NUMERIC,
  opp_lineup_r_pct NUMERIC,
  days_rest INT,
  tto INT,
  pitch_count_prev1 INT,
  pitch_count_prev2 INT
);

-- bullpens_daily
CREATE TABLE IF NOT EXISTS bullpens_daily (
  team TEXT,
  date DATE,
  bp_era NUMERIC,
  bp_fip NUMERIC,
  bp_kbb_pct NUMERIC,
  bp_hr9 NUMERIC,
  closer_pitches_d1 INT,
  setup1_pitches_d1 INT,
  setup2_pitches_d1 INT,
  closer_back2back_flag BOOL,
  PRIMARY KEY (team, date)
);

-- teams_offense_daily
CREATE TABLE IF NOT EXISTS teams_offense_daily (
  team TEXT,
  date DATE,
  wrcplus NUMERIC,
  woba NUMERIC,
  xwoba NUMERIC,
  iso NUMERIC,
  bb_pct NUMERIC,
  k_pct NUMERIC,
  babip NUMERIC,
  vs_rhp_xwoba NUMERIC,
  vs_lhp_xwoba NUMERIC,
  home_xwoba NUMERIC,
  away_xwoba NUMERIC,
  ba NUMERIC,          -- optional but nice if ingestor fills
  runs_pg NUMERIC,     -- optional but nice if ingestor fills
  PRIMARY KEY (team, date)
);

-- lineups
CREATE TABLE IF NOT EXISTS lineups (
  game_id TEXT,
  team TEXT,
  batter_id TEXT,
  order_spot INT,
  hand TEXT,
  proj_pa NUMERIC,
  xwoba_100 NUMERIC,
  xwoba_vs_hand_100 NUMERIC,
  iso_100 NUMERIC,
  k_pct_100 NUMERIC,
  bb_pct_100 NUMERIC
);

-- weather_game
CREATE TABLE IF NOT EXISTS weather_game (
  game_id TEXT PRIMARY KEY,
  temp_f INT,
  humidity_pct INT,
  wind_mph INT,
  wind_dir_deg INT,
  precip_prob INT,
  altitude_ft INT,
  air_density_idx NUMERIC
);

-- parks
CREATE TABLE IF NOT EXISTS parks (
  park_id TEXT PRIMARY KEY,
  name TEXT,
  pf_runs_3y NUMERIC,
  pf_hr_3y NUMERIC,
  altitude_ft INT,
  roof_type TEXT
);

-- umpires
CREATE TABLE IF NOT EXISTS umpires (
  ump_id TEXT PRIMARY KEY,
  name TEXT,
  called_strike_pct NUMERIC,
  edge_strike_pct NUMERIC,
  o_u_tendency NUMERIC,
  sample_size INT
);

-- injuries
CREATE TABLE IF NOT EXISTS injuries (
  team TEXT,
  player_id TEXT,
  status TEXT,
  expected_return DATE,
  impact_wrcplus_delta NUMERIC
);

-- market_moves (legacy, keep if you use it)
CREATE TABLE IF NOT EXISTS market_moves (
  game_id TEXT,
  ts TIMESTAMP,
  total NUMERIC,
  juice INT,
  book TEXT,
  is_close BOOL DEFAULT FALSE
);

-- markets_totals enum/table
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'market_type_enum') THEN
    CREATE TYPE market_type_enum AS ENUM ('snapshot','close');
  END IF;
END$$;

CREATE TABLE IF NOT EXISTS markets_totals (
  date DATE NOT NULL,
  game_id BIGINT NOT NULL,
  book TEXT NOT NULL,
  market_type market_type_enum NOT NULL,  -- 'snapshot' | 'close'
  k_total NUMERIC NULL,                   -- for snapshots
  close_total NUMERIC NULL,               -- for close
  snapshot_ts TIMESTAMPTZ NULL            -- for snapshots
);

-- unique + helpful indexes
CREATE UNIQUE INDEX IF NOT EXISTS uq_markets_snap
  ON markets_totals(game_id, date, book, market_type, COALESCE(snapshot_ts, 'epoch'::timestamptz));

CREATE INDEX IF NOT EXISTS idx_games_date ON games(date);
CREATE INDEX IF NOT EXISTS idx_starts_pid_date ON pitchers_starts(pitcher_id, date);
CREATE INDEX IF NOT EXISTS idx_offense_date_team ON teams_offense_daily(date, team);
CREATE INDEX IF NOT EXISTS idx_bullpens_date_team ON bullpens_daily(date, team);
CREATE INDEX IF NOT EXISTS idx_markets_date_game ON markets_totals(date, game_id);


ALTER TABLE teams_offense_daily
  ADD COLUMN IF NOT EXISTS ba NUMERIC,
  ADD COLUMN IF NOT EXISTS runs_pg NUMERIC;

ALTER TABLE parks
  ADD COLUMN IF NOT EXISTS lat NUMERIC,
  ADD COLUMN IF NOT EXISTS lon NUMERIC,
  ADD COLUMN IF NOT EXISTS cf_azimuth_deg INT;