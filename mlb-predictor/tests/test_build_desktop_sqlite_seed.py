from pathlib import Path

from sqlalchemy import create_engine

from scripts.build_desktop_sqlite_seed import build_sqlite_seed, main as build_seed_main
import scripts.build_desktop_sqlite_seed as build_desktop_sqlite_seed
from src.utils.db import query_df
from src.utils.db_migrate import run_migrations


def _build_source_engine(source_path: Path):
    engine = create_engine(f"sqlite:///{source_path.as_posix()}", future=True, connect_args={"check_same_thread": False})
    run_migrations(engine=engine)
    return engine


def test_build_sqlite_seed_copies_expected_tables(tmp_path):
    source_engine = _build_source_engine(tmp_path / "source.sqlite3")
    with source_engine.begin() as connection:
        connection.exec_driver_sql(
            "INSERT INTO games (game_id, game_date, season, home_team, away_team, status) VALUES (1, '2026-04-02', 2026, 'NYY', 'BOS', 'scheduled')"
        )
        connection.exec_driver_sql(
            "INSERT INTO team_offense_daily (game_date, season, team, runs, hits) VALUES ('2026-04-01', 2026, 'NYY', 5, 9)"
        )
        connection.exec_driver_sql(
            "INSERT INTO bullpens_daily (game_date, season, team, innings_pitched, pitches_thrown) VALUES ('2026-04-01', 2026, 'NYY', 4.0, 62)"
        )

    output_path = tmp_path / "desktop.sqlite3"
    counts = build_sqlite_seed(
        source_engine,
        output_path,
        table_names=["games", "team_offense_daily", "bullpens_daily"],
    )
    destination_engine = create_engine(
        f"sqlite:///{output_path.as_posix()}",
        future=True,
        connect_args={"check_same_thread": False},
    )

    assert counts == {"games": 1, "team_offense_daily": 1, "bullpens_daily": 1}
    assert query_df("SELECT COUNT(*) AS c FROM games", engine=destination_engine).iloc[0, 0] == 1
    assert query_df("SELECT COUNT(*) AS c FROM team_offense_daily", engine=destination_engine).iloc[0, 0] == 1
    assert query_df("SELECT COUNT(*) AS c FROM bullpens_daily", engine=destination_engine).iloc[0, 0] == 1


def test_build_seed_cli_rejects_sqlite_source_without_override(monkeypatch, tmp_path):
    source_path = tmp_path / "source.sqlite3"
    _build_source_engine(source_path)
    output_path = tmp_path / "desktop.sqlite3"

    monkeypatch.setattr(
        "scripts.build_desktop_sqlite_seed.build_parser",
        lambda: type(
            "Parser",
            (),
            {
                "parse_args": staticmethod(
                    lambda: type(
                        "Args",
                        (),
                        {
                            "output": str(output_path),
                            "source_database_url": f"sqlite:///{source_path.as_posix()}",
                            "chunk_size": 100,
                            "allow_sqlite_source": False,
                        },
                    )()
                )
            },
        )()
    )

    assert build_seed_main() == 1


def test_resolve_source_database_url_falls_back_to_runtime_sqlite(monkeypatch, tmp_path):
    source_path = tmp_path / "runtime.sqlite3"
    _build_source_engine(source_path)

    monkeypatch.setattr(
        build_desktop_sqlite_seed,
        "get_settings",
        lambda: type("Settings", (), {"database_url": build_desktop_sqlite_seed.LEGACY_DEFAULT_DATABASE_URL})(),
    )
    monkeypatch.setattr(
        build_desktop_sqlite_seed,
        "_sqlite_source_candidates",
        lambda: [source_path],
    )
    monkeypatch.setattr(
        build_desktop_sqlite_seed,
        "_database_has_accessible_tables",
        lambda url: url.startswith("sqlite:///"),
    )

    resolved_url, auto_selected_sqlite = build_desktop_sqlite_seed._resolve_source_database_url(None)

    assert resolved_url == f"sqlite:///{source_path.resolve().as_posix()}"
    assert auto_selected_sqlite is True