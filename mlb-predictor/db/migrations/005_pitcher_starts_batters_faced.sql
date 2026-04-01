ALTER TABLE pitcher_starts
ADD COLUMN IF NOT EXISTS batters_faced SMALLINT;

UPDATE pitcher_starts ps
SET batters_faced = pgp.batters_faced,
    updated_at = now()
FROM player_game_pitching pgp
WHERE ps.game_id = pgp.game_id
  AND ps.game_date = pgp.game_date
  AND ps.pitcher_id = pgp.player_id
  AND pgp.batters_faced IS NOT NULL
  AND ps.batters_faced IS DISTINCT FROM pgp.batters_faced;