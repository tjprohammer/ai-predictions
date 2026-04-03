from sqlalchemy import create_engine

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