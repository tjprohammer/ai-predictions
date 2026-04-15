from __future__ import annotations

from datetime import date as date_cls, datetime as datetime_cls
from functools import lru_cache
import json
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sqlalchemy import MetaData, Table, create_engine, delete, inspect, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.sql.sqltypes import Date, DateTime

from .settings import get_settings


SQLITE_SAFE_MAX_VARIABLES = 900


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()
    connect_args: dict[str, Any] = {}
    if settings.database_url.startswith("postgresql"):
        connect_args["connect_timeout"] = 3
    if settings.database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        future=True,
        connect_args=connect_args,
    )


def get_dialect_name(engine: Engine | None = None) -> str:
    active_engine = engine or get_engine()
    return str(active_engine.dialect.name).lower()


def table_exists(table_name: str, engine: Engine | None = None) -> bool:
    active_engine = engine or get_engine()
    return bool(inspect(active_engine).has_table(table_name))


def query_df(query: str, params: dict[str, Any] | None = None, engine: Engine | None = None) -> pd.DataFrame:
    active_engine = engine or get_engine()
    with active_engine.begin() as connection:
        return pd.read_sql_query(text(query), connection, params=params)


def run_sql(query: str, params: dict[str, Any] | None = None, engine: Engine | None = None) -> None:
    active_engine = engine or get_engine()
    with active_engine.begin() as connection:
        connection.execute(text(query), params or {})


def _normalize_db_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_db_value(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list):
        return [_normalize_db_value(inner_value) for inner_value in value]
    if isinstance(value, tuple):
        return tuple(_normalize_db_value(inner_value) for inner_value in value)
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.to_pydatetime()
    if isinstance(value, pd.Timedelta):
        return None if pd.isna(value) else value.to_pytimedelta()
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        return None if pd.isna(value) else value
    except TypeError:
        return value


def _serialize_sqlite_value(value: Any) -> Any:
    if isinstance(value, dict):
        return json.dumps({key: _serialize_sqlite_value(inner_value) for key, inner_value in value.items()}, default=str)
    if isinstance(value, list):
        return json.dumps([_serialize_sqlite_value(inner_value) for inner_value in value], default=str)
    if isinstance(value, tuple):
        return json.dumps([_serialize_sqlite_value(inner_value) for inner_value in value], default=str)
    return value


def _parse_temporal_string(value: str) -> pd.Timestamp | None:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(parsed) else parsed


def _coerce_value_for_column(value: Any, column: Any | None) -> Any:
    normalized = _normalize_db_value(value)
    if normalized is None or column is None:
        return normalized

    column_type = getattr(column, "type", None)
    if isinstance(column_type, DateTime):
        if isinstance(normalized, datetime_cls):
            return normalized
        if isinstance(normalized, date_cls):
            return datetime_cls.combine(normalized, datetime_cls.min.time())
        if isinstance(normalized, str):
            parsed = _parse_temporal_string(normalized)
            return parsed.to_pydatetime() if parsed is not None else normalized
        return normalized

    if isinstance(column_type, Date):
        if isinstance(normalized, datetime_cls):
            return normalized.date()
        if isinstance(normalized, date_cls):
            return normalized
        if isinstance(normalized, str):
            parsed = _parse_temporal_string(normalized)
            return parsed.date() if parsed is not None else normalized

    return normalized


def upsert_rows(
    table_name: str,
    rows: Iterable[dict[str, Any]],
    conflict_columns: list[str],
    engine: Engine | None = None,
) -> int:
    row_list = [dict(row) for row in rows]
    if not row_list:
        return 0

    active_engine = engine or get_engine()
    metadata = MetaData()
    table = Table(table_name, metadata, autoload_with=active_engine)
    column_map = {column.name: column for column in table.columns}
    # Only keys that exist on the table (skip fields when migration not applied yet).
    row_list = [
        {
            key: _coerce_value_for_column(value, column_map[key])
            for key, value in row.items()
            if key in column_map
        }
        for row in row_list
    ]

    # PostgreSQL rejects a single ON CONFLICT batch when the same key appears
    # multiple times inside the insert payload, so collapse to the last row.
    deduped_rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in row_list:
        deduped_rows[tuple(row.get(column) for column in conflict_columns)] = row
    row_list = list(deduped_rows.values())

    dialect_name = get_dialect_name(active_engine)
    if dialect_name == "sqlite":
        row_list = [{key: _serialize_sqlite_value(value) for key, value in row.items()} for row in row_list]

    def _execute_batch(connection, batch_rows: list[dict[str, Any]]) -> None:
        # Normalise: every row must have the same set of keys for the
        # multi-row VALUES clause to compile.
        all_keys = {k for row in batch_rows for k in row}
        batch_rows = [{k: row.get(k) for k in all_keys} for row in batch_rows]

        if dialect_name == "postgresql":
            insert_stmt = pg_insert(table).values(batch_rows)
        elif dialect_name == "sqlite":
            insert_stmt = sqlite_insert(table).values(batch_rows)
        else:
            raise NotImplementedError(f"upsert_rows does not support dialect '{dialect_name}' yet")
        provided_columns = {column_name for row in batch_rows for column_name in row.keys()}
        update_columns = {
            column.name: insert_stmt.excluded[column.name]
            for column in table.columns
            if column.name in provided_columns and column.name not in set(conflict_columns) and column.name != "created_at"
        }
        statement = insert_stmt.on_conflict_do_update(index_elements=conflict_columns, set_=update_columns)
        connection.execute(statement)

    with active_engine.begin() as connection:
        if dialect_name == "sqlite":
            max_columns = max(len(row.keys()) for row in row_list)
            batch_size = max(1, SQLITE_SAFE_MAX_VARIABLES // max(1, max_columns))
            for start in range(0, len(row_list), batch_size):
                _execute_batch(connection, row_list[start : start + batch_size])
        else:
            _execute_batch(connection, row_list)
    return len(row_list)


def delete_for_date_range(
    table_name: str,
    start_date: Any,
    end_date: Any,
    date_column: str = "game_date",
    engine: Engine | None = None,
) -> None:
    active_engine = engine or get_engine()
    metadata = MetaData()
    table = Table(table_name, metadata, autoload_with=active_engine)
    statement = delete(table).where(table.c[date_column] >= start_date, table.c[date_column] <= end_date)
    with active_engine.begin() as connection:
        connection.execute(statement)