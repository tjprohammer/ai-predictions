-- 1. odds event log (one row per change)
CREATE TABLE IF NOT EXISTS book_odds_events (
  game_id      TEXT NOT NULL,
  book_key     TEXT NOT NULL,        -- 'pinnacle','circa', etc.
  captured_at  TIMESTAMPTZ NOT NULL, -- provider event time or your crawl time
  market       TEXT NOT NULL,        -- 'totals'
  line_total   NUMERIC,              -- 8.5
  over_price   INT,                  -- -110
  under_price  INT,
  PRIMARY KEY (game_id, book_key, captured_at, market)
);
CREATE INDEX IF NOT EXISTS idx_boe_game_book ON book_odds_events(game_id, book_key, captured_at);

-- 2a. opening per book
CREATE MATERIALIZED VIEW IF NOT EXISTS opening_totals_v1 AS
SELECT z.game_id, z.book_key,
       z.line_total  AS opening_total,
       z.over_price  AS opening_over_odds,
       z.under_price AS opening_under_odds,
       z.captured_at AS opening_captured_at
FROM (
  SELECT e.*,
         ROW_NUMBER() OVER (PARTITION BY e.game_id, e.book_key ORDER BY e.captured_at) rn
  FROM book_odds_events e
  WHERE e.market = 'totals'
) z
WHERE z.rn = 1;

-- 2b. consensus opening (choose your books)
CREATE MATERIALIZED VIEW IF NOT EXISTS opening_totals_consensus_v1 AS
SELECT game_id,
       AVG(opening_total) AS opening_total_consensus,
       MIN(opening_captured_at) AS first_open_seen_at
FROM opening_totals_v1
WHERE book_key IN ('pinnacle','circa','betcris')
GROUP BY game_id;

-- 3. Add columns to enhanced_games
ALTER TABLE enhanced_games
  ADD COLUMN IF NOT EXISTS opening_total NUMERIC,
  ADD COLUMN IF NOT EXISTS opening_over_odds INT,
  ADD COLUMN IF NOT EXISTS opening_under_odds INT,
  ADD COLUMN IF NOT EXISTS opening_captured_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS opening_is_proxy BOOLEAN DEFAULT FALSE;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_enhanced_games_opening ON enhanced_games(opening_total) WHERE opening_total IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_enhanced_games_date_opening ON enhanced_games(date, opening_total);
