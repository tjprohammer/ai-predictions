-- Matchup splits: batter-vs-pitcher, pitcher-vs-team, platoon, team H2H
-- Sourced from StatMuse natural language queries
CREATE TABLE IF NOT EXISTS matchup_splits (
    player_id           BIGINT      NOT NULL,
    opponent_id         BIGINT      NOT NULL,   -- pitcher_id for BvP, 0 for team/platoon splits
    split_type          VARCHAR(20) NOT NULL,   -- 'bvp', 'vs_team', 'platoon_lhp', 'platoon_rhp', 'pitcher_vs_team'
    season              SMALLINT    NOT NULL DEFAULT 0,  -- 0 = career, else year
    games               SMALLINT,
    plate_appearances   SMALLINT,
    at_bats             SMALLINT,
    hits                SMALLINT,
    home_runs           SMALLINT,
    walks               SMALLINT,
    strikeouts          SMALLINT,
    rbi                 SMALLINT,
    runs                SMALLINT,
    doubles             SMALLINT,
    triples             SMALLINT,
    batting_avg         NUMERIC(5,4),
    obp                 NUMERIC(5,4),
    slg                 NUMERIC(5,4),
    ops                 NUMERIC(5,4),
    -- Pitcher-specific columns (pitcher_vs_team rows)
    innings_pitched     NUMERIC(5,1),
    earned_runs         SMALLINT,
    era                 NUMERIC(5,2),
    whip                NUMERIC(5,3),
    k_per_9             NUMERIC(5,2),
    -- Metadata
    fetched_at          TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source_url          TEXT,
    PRIMARY KEY (player_id, opponent_id, split_type, season)
);
