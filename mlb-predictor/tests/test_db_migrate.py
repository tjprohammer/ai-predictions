from pathlib import Path

from sqlalchemy import create_engine, inspect

from src.utils.db import query_df
from src.utils.db_migrate import run_migrations


def test_run_migrations_builds_sqlite_schema(tmp_path):
    db_path = tmp_path / "mlb.sqlite3"
    engine = create_engine(f"sqlite:///{db_path}", future=True, connect_args={"check_same_thread": False})

    applied_count = run_migrations(engine=engine)
    inspector = inspect(engine)

    assert applied_count >= 14
    assert inspector.has_table("games") is True
    assert inspector.has_table("player_prop_markets") is True
    assert inspector.has_table("predictions_first5_totals") is True

    game_columns = {column["name"] for column in inspector.get_columns("games")}
    assert {"home_runs_first5", "away_runs_first5", "total_runs_first5"}.issubset(game_columns)

    pitcher_columns = {column["name"] for column in inspector.get_columns("pitcher_starts")}
    assert "batters_faced" in pitcher_columns

    prediction_columns = {column["name"]: str(column["type"]) for column in inspector.get_columns("predictions_totals")}
    assert {"market_sportsbook", "market_snapshot_ts"}.issubset(prediction_columns)
    assert prediction_columns["market_snapshot_ts"] == "TEXT"

    outcome_columns = {column["name"]: str(column["type"]) for column in inspector.get_columns("prediction_outcomes_daily")}
    assert {"entry_market_sportsbook", "entry_market_snapshot_ts", "closing_market_same_sportsbook"}.issubset(outcome_columns)
    assert outcome_columns["entry_market_snapshot_ts"] == "TEXT"


def test_run_migrations_supports_sqlite_inserts_after_schema_build(tmp_path):
    db_path = tmp_path / "writes.sqlite3"
    engine = create_engine(f"sqlite:///{db_path}", future=True, connect_args={"check_same_thread": False})

    run_migrations(engine=engine)
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "INSERT INTO games (game_id, game_date, home_team, away_team, status) VALUES (1, '2026-04-02', 'NYY', 'BOS', 'scheduled')"
        )

    frame = query_df("SELECT game_id, home_team, away_team FROM games", engine=engine)

    assert frame.to_dict(orient="records") == [{"game_id": 1, "home_team": "NYY", "away_team": "BOS"}]