from pathlib import Path

from sqlalchemy import create_engine

from src.ingestors.park_factors import ensure_park_factors_seeded
from src.utils.db import query_df
from src.utils.db_migrate import run_migrations


def test_ensure_park_factors_seeded_imports_seed_csv_into_sqlite(tmp_path, monkeypatch):
    db_path = tmp_path / "bootstrap.sqlite3"
    engine = create_engine(f"sqlite:///{db_path}", future=True, connect_args={"check_same_thread": False})
    run_migrations(engine=engine)

    seed_csv = tmp_path / "park_factors.csv"
    seed_csv.write_text(
        "season,team_abbr,venue_id,venue_name,source_name,run_factor,hr_factor,singles_factor,doubles_factor,triples_factor\n"
        "2025,NYY,3313,Yankee Stadium,seed_csv,1.0,1.19,0.91,0.9,0.63\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("src.ingestors.park_factors.upsert_rows", lambda *args, **kwargs: __import__("src.utils.db", fromlist=["upsert_rows"]).upsert_rows(*args, engine=engine, **kwargs))
    monkeypatch.setattr("src.ingestors.park_factors.query_df", lambda query, params=None: query_df(query, params=params, engine=engine))

    result = ensure_park_factors_seeded(csv_path=seed_csv, skip_bootstrap=True)
    frame = query_df("SELECT season, team_abbr, venue_name FROM park_factors", engine=engine)

    assert result["imported"] == 1
    assert result["target_ready"] is True
    assert frame.to_dict(orient="records") == [{"season": 2025, "team_abbr": "NYY", "venue_name": "Yankee Stadium"}]


def test_ensure_park_factors_seeded_bootstraps_target_season_without_games(tmp_path, monkeypatch):
    db_path = tmp_path / "bootstrap_current.sqlite3"
    engine = create_engine(f"sqlite:///{db_path}", future=True, connect_args={"check_same_thread": False})
    run_migrations(engine=engine)

    seed_csv = tmp_path / "park_factors.csv"
    seed_csv.write_text(
        "season,team_abbr,venue_id,venue_name,source_name,run_factor,hr_factor,singles_factor,doubles_factor,triples_factor\n"
        "2025,NYY,3313,Yankee Stadium,seed_csv,1.0,1.19,0.91,0.9,0.63\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("src.ingestors.park_factors.upsert_rows", lambda *args, **kwargs: __import__("src.utils.db", fromlist=["upsert_rows"]).upsert_rows(*args, engine=engine, **kwargs))
    monkeypatch.setattr("src.ingestors.park_factors.query_df", lambda query, params=None: query_df(query, params=params, engine=engine))

    result = ensure_park_factors_seeded(
        csv_path=seed_csv,
        target_season=2026,
        fallback_season=2025,
        skip_bootstrap=False,
    )
    frame = query_df("SELECT season, team_abbr, source_name FROM park_factors ORDER BY season, team_abbr", engine=engine)

    assert result["imported"] == 1
    assert result["bootstrapped"] == 1
    assert result["target_ready"] is True
    assert frame.to_dict(orient="records") == [
        {"season": 2025, "team_abbr": "NYY", "source_name": "seed_csv"},
        {"season": 2026, "team_abbr": "NYY", "source_name": "bootstrap_2025"},
    ]