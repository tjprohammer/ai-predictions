"""Tests for lineups backfill from player_game_batting."""
from __future__ import annotations

from datetime import date

from sqlalchemy import create_engine

from src.ingestors.lineups_backfill import SOURCE_NAME, backfill_lineups_from_player_batting
from src.utils.db import query_df
from src.utils.db_migrate import run_migrations


def test_backfill_lineups_from_player_batting_inserts_rows(tmp_path):
    db_path = tmp_path / "t.sqlite3"
    engine = create_engine(f"sqlite:///{db_path}", future=True, connect_args={"check_same_thread": False})
    run_migrations(engine=engine)

    with engine.begin() as conn:
        conn.exec_driver_sql(
            """
            INSERT INTO games (
                game_id, game_date, season, status, home_team, away_team, game_start_ts
            ) VALUES (
                999001, '2026-04-02', 2026, 'final', 'NYY', 'BOS', '2026-04-02T23:00:00Z'
            )
            """
        )
        conn.exec_driver_sql(
            """
            INSERT INTO dim_players (player_id, full_name, position)
            VALUES (111, 'Test Hitter', 'DH')
            """
        )
        conn.exec_driver_sql(
            """
            INSERT INTO player_game_batting (
                game_id, game_date, player_id, team, opponent, home_away,
                lineup_slot, plate_appearances, at_bats
            ) VALUES (
                999001, '2026-04-02', 111, 'NYY', 'BOS', 'H',
                3, 4, 3
            )
            """
        )

    n = backfill_lineups_from_player_batting(
        date(2026, 4, 2),
        date(2026, 4, 2),
        engine=engine,
    )
    assert n == 1

    rows = query_df(
        """
        SELECT game_id, player_id, team, lineup_slot, source_name
        FROM lineups
        WHERE game_id = 999001 AND player_id = 111
        """,
        engine=engine,
    )
    assert len(rows) == 1
    assert int(rows.iloc[0]["lineup_slot"]) == 3
    assert rows.iloc[0]["source_name"] == SOURCE_NAME


def test_snapshot_ts_fallback_without_game_start(tmp_path):
    db_path = tmp_path / "u.sqlite3"
    engine = create_engine(f"sqlite:///{db_path}", future=True, connect_args={"check_same_thread": False})
    run_migrations(engine=engine)

    with engine.begin() as conn:
        conn.exec_driver_sql(
            """
            INSERT INTO games (
                game_id, game_date, season, status, home_team, away_team, game_start_ts
            ) VALUES (
                999002, '2026-05-01', 2026, 'final', 'LAD', 'SDP', NULL
            )
            """
        )
        conn.exec_driver_sql(
            "INSERT INTO dim_players (player_id, full_name) VALUES (222, 'No Start Ts')"
        )
        conn.exec_driver_sql(
            """
            INSERT INTO player_game_batting (
                game_id, game_date, player_id, team, opponent, home_away,
                lineup_slot, plate_appearances, at_bats
            ) VALUES (
                999002, '2026-05-01', 222, 'LAD', 'SDP', 'H',
                1, 5, 4
            )
            """
        )

    backfill_lineups_from_player_batting(date(2026, 5, 1), date(2026, 5, 1), engine=engine)

    rows = query_df(
        "SELECT snapshot_ts FROM lineups WHERE game_id = 999002 AND player_id = 222",
        engine=engine,
    )
    assert len(rows) == 1
    assert rows.iloc[0]["snapshot_ts"] is not None
