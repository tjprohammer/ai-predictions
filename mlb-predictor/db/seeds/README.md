# Seed Files

This folder is for low-churn reference data and manual import templates.

Expected files:

- `park_factors.csv`
  - Columns: `season,team_abbr,venue_id,venue_name,run_factor,hr_factor,singles_factor,doubles_factor,triples_factor,source`
- `manual_lineups.csv`
  - Columns: `game_date,team,player_id,player_name,lineup_slot,position,confirmed,source_name,source_url,snapshot_ts`
- `manual_market_totals.csv`
  - Columns: `game_date,home_team,away_team,sportsbook,market_type,line_value,over_price,under_price,snapshot_ts,is_closing`

The initial rebuild treats lineups and market imports as optional manual overlays until the automated sources are locked down.

Current workflow:

- `python -m src.ingestors.park_factors` loads `park_factors.csv` when present and bootstraps the current season from the prior season or neutral values when the table is empty.
- `python -m src.ingestors.prepare_slate_inputs --target-date YYYY-MM-DD` creates editable lineup and market template CSV rows under `data/raw/` for the scheduled slate.
