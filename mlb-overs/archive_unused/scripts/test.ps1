# What offense rows do we have for the last 7 days?
@'
SELECT date, team, xwoba, iso, bb_pct, k_pct
FROM teams_offense_daily
WHERE date BETWEEN DATE '2025-08-04' AND DATE '2025-08-10'
ORDER BY date, team
'@ | docker compose exec -T db psql -U mlbuser -d mlb -f -

# What bullpen rows do we have for those dates?
@'
SELECT date, team, bp_era, bp_fip, bp_kbb_pct, bp_hr9
FROM bullpens_daily
WHERE date BETWEEN DATE '2025-08-04' AND DATE '2025-08-10'
ORDER BY date, team
'@ | docker compose exec -T db psql -U mlbuser -d mlb -f -

# What totals did we capture (only 6/15 so far)?
@'
SELECT game_id, book, market_type, COALESCE(close_total,k_total) AS total, snapshot_ts
FROM markets_totals
WHERE date = DATE '2025-08-10'
ORDER BY game_id, market_type, snapshot_ts NULLS FIRST
'@ | docker compose exec -T db psql -U mlbuser -d mlb -f -
