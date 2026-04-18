-- Persist Top EV pick per game (weighted-EV winner among priced candidates) at pregame lock
-- so Daily Results / prediction_outcomes_daily match what the board showed, not a next-day recompute.

CREATE TABLE IF NOT EXISTS board_top_ev_snapshots (
    game_id            BIGINT NOT NULL,
    game_date          DATE NOT NULL,
    snapshot_payload   TEXT NOT NULL,
    frozen_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (game_id, game_date)
);

CREATE INDEX IF NOT EXISTS idx_board_top_ev_snapshots_date ON board_top_ev_snapshots (game_date);
