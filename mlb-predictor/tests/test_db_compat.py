import pandas as pd
from sqlalchemy import create_engine

from src.utils import db as db_module
from src.utils.db import query_df, run_sql, table_exists, upsert_rows


def test_sqlite_upsert_rows_updates_existing_record(tmp_path):
    db_path = tmp_path / "compat.sqlite"
    engine = create_engine(f"sqlite:///{db_path}", future=True, connect_args={"check_same_thread": False})

    run_sql(
        """
        CREATE TABLE sample_rows (
            game_id INTEGER PRIMARY KEY,
            team TEXT,
            runs INTEGER,
            created_at TEXT
        )
        """,
        engine=engine,
    )

    upsert_rows(
        "sample_rows",
        [{"game_id": 1, "team": "NYY", "runs": 4}],
        ["game_id"],
        engine=engine,
    )
    upsert_rows(
        "sample_rows",
        [{"game_id": 1, "team": "NYY", "runs": 7}],
        ["game_id"],
        engine=engine,
    )

    frame = query_df("SELECT game_id, team, runs FROM sample_rows", engine=engine)

    assert table_exists("sample_rows", engine=engine) is True
    assert frame.to_dict(orient="records") == [{"game_id": 1, "team": "NYY", "runs": 7}]


def test_table_exists_returns_false_for_missing_table(tmp_path):
    db_path = tmp_path / "missing.sqlite"
    engine = create_engine(f"sqlite:///{db_path}", future=True, connect_args={"check_same_thread": False})

    assert table_exists("does_not_exist", engine=engine) is False


def test_sqlite_upsert_rows_serializes_dict_payloads(tmp_path):
    db_path = tmp_path / "json.sqlite"
    engine = create_engine(f"sqlite:///{db_path}", future=True, connect_args={"check_same_thread": False})

    run_sql(
        """
        CREATE TABLE feature_rows (
            game_id INTEGER PRIMARY KEY,
            feature_payload TEXT NOT NULL
        )
        """,
        engine=engine,
    )

    upsert_rows(
        "feature_rows",
        [{"game_id": 1, "feature_payload": {"market_total": 8.5, "confirmed": True}}],
        ["game_id"],
        engine=engine,
    )

    frame = query_df("SELECT game_id, feature_payload FROM feature_rows", engine=engine)

    assert frame.to_dict(orient="records") == [{"game_id": 1, "feature_payload": '{"market_total": 8.5, "confirmed": true}'}]


def test_sqlite_upsert_rows_coerces_timestamp_values_for_date_columns(tmp_path):
    db_path = tmp_path / "dates.sqlite"
    engine = create_engine(f"sqlite:///{db_path}", future=True, connect_args={"check_same_thread": False})

    run_sql(
        """
        CREATE TABLE scheduled_games (
            game_id INTEGER PRIMARY KEY,
            game_date DATE NOT NULL,
            game_start_ts DATETIME
        )
        """,
        engine=engine,
    )

    upsert_rows(
        "scheduled_games",
        [
            {
                "game_id": 1,
                "game_date": pd.Timestamp("2026-04-03T00:00:00Z"),
                "game_start_ts": pd.Timestamp("2026-04-03T19:05:00Z"),
            }
        ],
        ["game_id"],
        engine=engine,
    )

    frame = query_df("SELECT game_id, game_date, game_start_ts FROM scheduled_games", engine=engine)

    assert frame.loc[0, "game_id"] == 1
    assert str(frame.loc[0, "game_date"]).startswith("2026-04-03")
    assert "2026-04-03 19:05:00" in str(frame.loc[0, "game_start_ts"])


def test_sqlite_upsert_rows_chunks_large_batches(monkeypatch, tmp_path):
    db_path = tmp_path / "chunked.sqlite"
    engine = create_engine(f"sqlite:///{db_path}", future=True, connect_args={"check_same_thread": False})

    run_sql(
        """
        CREATE TABLE batch_rows (
            game_id INTEGER PRIMARY KEY,
            team TEXT,
            runs INTEGER,
            created_at TEXT
        )
        """,
        engine=engine,
    )

    monkeypatch.setattr(db_module, "SQLITE_SAFE_MAX_VARIABLES", 6)

    rows = [
        {"game_id": index, "team": f"T{index}", "runs": index % 10}
        for index in range(1, 6)
    ]
    upsert_rows("batch_rows", rows, ["game_id"], engine=engine)

    frame = query_df("SELECT game_id, team, runs FROM batch_rows ORDER BY game_id", engine=engine)

    assert len(frame) == 5
    assert frame.loc[0, "game_id"] == 1
    assert frame.loc[4, "game_id"] == 5