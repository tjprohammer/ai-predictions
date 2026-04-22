-- First-run pick-of-the-day snapshots: per-game freeze so strict short-list picks do not drift on refresh
-- (mirrors board_green_run_snapshots / board_top_ev_run_snapshots).

CREATE TABLE IF NOT EXISTS board_pick_of_day_run_snapshots (
    game_id            BIGINT NOT NULL,
    game_date          DATE NOT NULL,
    snapshot_payload   TEXT NOT NULL,
    frozen_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (game_id, game_date)
);

CREATE INDEX IF NOT EXISTS idx_board_pick_of_day_run_snapshots_date
    ON board_pick_of_day_run_snapshots (game_date);
