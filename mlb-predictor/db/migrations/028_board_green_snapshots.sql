-- Persist green-strip card snapshots per game so early-window picks stay stable
-- after later refreshes when night-game data improves.

CREATE TABLE IF NOT EXISTS board_green_snapshots (
    game_id            BIGINT NOT NULL,
    game_date          DATE NOT NULL,
    snapshot_payload   TEXT NOT NULL,
    frozen_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (game_id, game_date)
);

CREATE INDEX IF NOT EXISTS idx_board_green_snapshots_date ON board_green_snapshots (game_date);
