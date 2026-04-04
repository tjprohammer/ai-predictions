from __future__ import annotations

import re
from pathlib import Path

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from src.utils.db import get_dialect_name, get_engine


MIGRATION_DIR = Path(__file__).resolve().parents[2] / "db" / "migrations"


def _sqlite_identity_sql() -> str:
    return "INTEGER PRIMARY KEY AUTOINCREMENT"


def _normalize_sqlite_sql(sql_text: str) -> str:
    normalized = sql_text
    normalized = re.sub(
        r"BIGINT\s+GENERATED\s+BY\s+DEFAULT\s+AS\s+IDENTITY\s+PRIMARY\s+KEY",
        _sqlite_identity_sql(),
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\bTIMESTAMPTZ\b", "TEXT", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bJSONB\b", "TEXT", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"DEFAULT\s+now\(\)", "DEFAULT CURRENT_TIMESTAMP", normalized, flags=re.IGNORECASE)
    normalized = re.sub(
        r"^\s*ALTER\s+TABLE\s+.+?ALTER\s+COLUMN\s+.+?TYPE\s+.+?;\s*$",
        "",
        normalized,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    return normalized


def _sqlite_add_column_if_missing(connection, table_name: str, column_name: str, column_type: str) -> None:
    existing_columns = {column["name"] for column in inspect(connection).get_columns(table_name)}
    if column_name in existing_columns:
        return
    connection.exec_driver_sql(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


def _apply_sqlite_migration_005(connection) -> None:
    _sqlite_add_column_if_missing(connection, "pitcher_starts", "batters_faced", "INTEGER")
    connection.exec_driver_sql(
        """
        UPDATE pitcher_starts
        SET batters_faced = (
                SELECT pgp.batters_faced
                FROM player_game_pitching pgp
                WHERE pitcher_starts.game_id = pgp.game_id
                  AND pitcher_starts.game_date = pgp.game_date
                  AND pitcher_starts.pitcher_id = pgp.player_id
                  AND pgp.batters_faced IS NOT NULL
                LIMIT 1
            ),
            updated_at = CURRENT_TIMESTAMP
        WHERE EXISTS (
            SELECT 1
            FROM player_game_pitching pgp
            WHERE pitcher_starts.game_id = pgp.game_id
              AND pitcher_starts.game_date = pgp.game_date
              AND pitcher_starts.pitcher_id = pgp.player_id
              AND pgp.batters_faced IS NOT NULL
              AND (pitcher_starts.batters_faced IS NULL OR pitcher_starts.batters_faced != pgp.batters_faced)
        )
        """
    )


def _apply_sqlite_migration_006(connection, sql_text: str) -> None:
    _sqlite_add_column_if_missing(connection, "games", "home_runs_first5", "INTEGER")
    _sqlite_add_column_if_missing(connection, "games", "away_runs_first5", "INTEGER")
    _sqlite_add_column_if_missing(connection, "games", "total_runs_first5", "INTEGER")
    create_sql = _normalize_sqlite_sql(
        re.sub(
            r"ALTER\s+TABLE\s+games\s+ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS.+?;\s*",
            "",
            sql_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    connection.connection.driver_connection.executescript(create_sql)


def _apply_sqlite_alter_add_columns(connection, sql_text: str) -> None:
    """Handle ALTER TABLE ... ADD COLUMN IF NOT EXISTS for SQLite."""
    for match in re.finditer(
        r"ALTER\s+TABLE\s+(\w+)\s+ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+(\w+)\s+([^;]+);",
        sql_text,
        flags=re.IGNORECASE,
    ):
        table_name, column_name, column_type = match.group(1), match.group(2), match.group(3).strip()
        column_type = re.sub(r"\bTIMESTAMPTZ\b", "TEXT", column_type, flags=re.IGNORECASE)
        column_type = re.sub(r"\bJSONB\b", "TEXT", column_type, flags=re.IGNORECASE)
        _sqlite_add_column_if_missing(connection, table_name, column_name, column_type)


def _apply_sqlite_migration(connection, migration_path: Path) -> None:
    sql_text = migration_path.read_text(encoding="utf-8")
    if migration_path.name == "004_pitcher_trend_precision.sql":
        return
    if migration_path.name == "005_pitcher_starts_batters_faced.sql":
        _apply_sqlite_migration_005(connection)
        return
    if migration_path.name == "006_first5_totals.sql":
        _apply_sqlite_migration_006(connection, sql_text)
        return

    # Migrations that are purely ALTER TABLE ADD COLUMN IF NOT EXISTS
    alter_only = re.sub(
        r"ALTER\s+TABLE\s+\w+\s+ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+\w+\s+[^;]+;",
        "",
        sql_text,
        flags=re.IGNORECASE,
    )
    has_alter_adds = alter_only.strip() != sql_text.strip()
    if has_alter_adds:
        _apply_sqlite_alter_add_columns(connection, sql_text)
        remaining = _normalize_sqlite_sql(alter_only)
        if remaining.strip():
            connection.connection.driver_connection.executescript(remaining)
        return

    normalized = _normalize_sqlite_sql(sql_text)
    if normalized.strip():
        connection.connection.driver_connection.executescript(normalized)


def run_migrations(engine: Engine | None = None) -> int:
    migration_files = sorted(MIGRATION_DIR.glob("*.sql"))
    if not migration_files:
        print("No migration files found")
        return 0

    active_engine = engine or get_engine()
    dialect_name = get_dialect_name(active_engine)
    for migration_path in migration_files:
        print(f"Applying {migration_path.name} ...")
        with active_engine.begin() as connection:
            if dialect_name == "sqlite":
                _apply_sqlite_migration(connection, migration_path)
            else:
                connection.execute(text(migration_path.read_text(encoding="utf-8")))
        print("  done")
    return len(migration_files)


def main() -> int:
    run_migrations()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())