from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import MetaData, Table, create_engine, inspect, select
from sqlalchemy.engine import Engine

from src.utils.db import get_dialect_name
from src.utils.db_migrate import run_migrations
from src.utils.settings import get_settings


DEFAULT_OUTPUT_PATH = ROOT / "db" / "mlb_predictor.sqlite3"
DEFAULT_CHUNK_SIZE = 2000
PREFERRED_TABLE_ORDER = (
    "dim_teams",
    "dim_players",
    "dim_venues",
    "dim_dates",
    "games",
    "game_weather",
    "game_markets",
    "player_prop_markets",
    "lineups",
    "player_game_batting",
    "player_game_pitching",
    "team_offense_daily",
    "bullpens_daily",
    "pitcher_starts",
    "park_factors",
    "game_features_totals",
    "player_features_hits",
    "predictions_totals",
    "predictions_player_hits",
    "backtest_totals",
    "backtest_player_hits",
    "player_trend_daily",
    "pitcher_trend_daily",
    "game_features_pitcher_strikeouts",
    "predictions_pitcher_strikeouts",
    "prediction_outcomes_daily",
    "model_scorecards_daily",
    "game_features_first5_totals",
    "predictions_first5_totals",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the desktop SQLite seed from the current source database")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Path to the desktop SQLite database file")
    parser.add_argument("--source-database-url", help="Override the source database URL used for the export")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Rows to copy per insert batch")
    parser.add_argument(
        "--allow-sqlite-source",
        action="store_true",
        help="Allow a SQLite source database. Intended for tests only.",
    )
    return parser


def _sqlite_ready_value(value: Any) -> Any:
    if isinstance(value, dict):
        return json.dumps({key: _sqlite_ready_value(inner_value) for key, inner_value in value.items()}, default=str)
    if isinstance(value, list):
        return json.dumps([_sqlite_ready_value(inner_value) for inner_value in value], default=str)
    if isinstance(value, tuple):
        return json.dumps([_sqlite_ready_value(inner_value) for inner_value in value], default=str)
    return value


def _resolve_table_order(source_tables: Iterable[str], destination_tables: Iterable[str]) -> list[str]:
    source_names = set(source_tables)
    destination_names = set(destination_tables)
    available = source_names & destination_names
    ordered = [table_name for table_name in PREFERRED_TABLE_ORDER if table_name in available]
    ordered.extend(sorted(table_name for table_name in available if table_name not in set(ordered)))
    return ordered


def _copy_table(source_engine: Engine, destination_engine: Engine, table_name: str, chunk_size: int) -> int:
    source_metadata = MetaData()
    destination_metadata = MetaData()
    source_table = Table(table_name, source_metadata, autoload_with=source_engine)
    destination_table = Table(table_name, destination_metadata, autoload_with=destination_engine)
    shared_columns = [column.name for column in destination_table.columns if column.name in source_table.c]
    if not shared_columns:
        return 0

    statement = select(*(source_table.c[column_name] for column_name in shared_columns))
    inserted = 0

    with source_engine.connect() as source_connection, destination_engine.begin() as destination_connection:
        destination_connection.execute(destination_table.delete())
        result = source_connection.execution_options(stream_results=True).execute(statement)
        while True:
            rows = result.fetchmany(chunk_size)
            if not rows:
                break
            payload = [
                {
                    column_name: _sqlite_ready_value(row._mapping[column_name])
                    for column_name in shared_columns
                }
                for row in rows
            ]
            destination_connection.execute(destination_table.insert(), payload)
            inserted += len(payload)

    return inserted


def build_sqlite_seed(
    source_engine: Engine,
    output_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    table_names: Sequence[str] | None = None,
) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    destination_engine = create_engine(
        f"sqlite:///{output_path.as_posix()}",
        future=True,
        connect_args={"check_same_thread": False},
    )
    run_migrations(engine=destination_engine)

    source_tables = inspect(source_engine).get_table_names()
    destination_tables = inspect(destination_engine).get_table_names()
    selected_tables = list(table_names) if table_names is not None else _resolve_table_order(source_tables, destination_tables)
    counts: dict[str, int] = {}
    for table_name in selected_tables:
        if table_name not in source_tables or table_name not in destination_tables:
            continue
        counts[table_name] = _copy_table(source_engine, destination_engine, table_name, chunk_size)

    raw_connection = destination_engine.raw_connection()
    try:
        raw_connection.execute("VACUUM")
        raw_connection.commit()
    finally:
        raw_connection.close()

    return counts


def _resolve_source_database_url(explicit_url: str | None) -> str:
    if explicit_url:
        return explicit_url.strip()
    return get_settings().database_url


def main() -> int:
    args = build_parser().parse_args()
    output_path = Path(args.output).resolve()
    source_database_url = _resolve_source_database_url(args.source_database_url)
    source_engine = create_engine(source_database_url, future=True)
    source_dialect = get_dialect_name(source_engine)
    if source_dialect == "sqlite" and not args.allow_sqlite_source:
        print(
            "Refusing to build the desktop seed from a SQLite source database. "
            "Clear DATABASE_URL or pass --source-database-url for the populated Postgres database."
        )
        return 1

    counts = build_sqlite_seed(source_engine, output_path, chunk_size=max(args.chunk_size, 1))
    total_rows = sum(counts.values())
    print(f"Built desktop SQLite seed: {output_path}")
    print(f"Source dialect: {source_dialect}")
    print(f"Copied {len(counts)} tables and {total_rows} rows")
    for table_name in sorted(counts):
        print(f"  {table_name}: {counts[table_name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())