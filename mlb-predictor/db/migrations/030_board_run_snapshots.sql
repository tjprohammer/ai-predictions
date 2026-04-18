-- First-run (pregame) snapshots: captured on the earliest board build before first pitch, without
-- requiring the pregame lock window. Complements lock-time tables 028/029 (pregame freeze).

CREATE TABLE IF NOT EXISTS board_green_run_snapshots (
    game_id            BIGINT NOT NULL,
    game_date          DATE NOT NULL,
    snapshot_payload   TEXT NOT NULL,
    frozen_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (game_id, game_date)
);

CREATE INDEX IF NOT EXISTS idx_board_green_run_snapshots_date ON board_green_run_snapshots (game_date);

CREATE TABLE IF NOT EXISTS board_top_ev_run_snapshots (
    game_id            BIGINT NOT NULL,
    game_date          DATE NOT NULL,
    snapshot_payload   TEXT NOT NULL,
    frozen_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (game_id, game_date)
);

CREATE INDEX IF NOT EXISTS idx_board_top_ev_run_snapshots_date ON board_top_ev_run_snapshots (game_date);
