-- sql/market_totals.sql

CREATE TABLE IF NOT EXISTS markets_totals (
  game_id     TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
  date        DATE NOT NULL,
  book        TEXT,
  market_type TEXT NOT NULL CHECK (market_type IN ('open','close','snapshot')),
  open_total  NUMERIC(5,2),
  close_total NUMERIC(5,2),
  k_total     NUMERIC(5,2),
  snapshot_ts TIMESTAMPTZ,
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Partial unique indexes that ON CONFLICT can target
-- 1) Only one open or close row per (game, book, type)
CREATE UNIQUE INDEX IF NOT EXISTS ux_mk_openclose
  ON markets_totals (game_id, book, market_type)
  WHERE market_type IN ('open','close');

-- 2) Snapshots can repeat over time, but each snapshot_ts is unique
CREATE UNIQUE INDEX IF NOT EXISTS ux_mk_snap
  ON markets_totals (game_id, book, market_type, snapshot_ts)
  WHERE market_type = 'snapshot';

-- Helpful filter
CREATE INDEX IF NOT EXISTS ix_mk_date ON markets_totals (date);
