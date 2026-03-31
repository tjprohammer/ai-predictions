from __future__ import annotations

from functools import lru_cache
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sqlalchemy import MetaData, Table, create_engine, delete, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from .settings import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()
    connect_args: dict[str, Any] = {}
    if settings.database_url.startswith("postgresql"):
        connect_args["connect_timeout"] = 3
    return create_engine(
        settings.database_url,
        pool_pre_ping=True,
        future=True,
        connect_args=connect_args,
    )


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


def upsert_rows(
    table_name: str,
    rows: Iterable[dict[str, Any]],
    conflict_columns: list[str],
    engine: Engine | None = None,
) -> int:
    row_list = [{key: _normalize_db_value(value) for key, value in row.items()} for row in rows]
    if not row_list:
        return 0

    # PostgreSQL rejects a single ON CONFLICT batch when the same key appears
    # multiple times inside the insert payload, so collapse to the last row.
    deduped_rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in row_list:
        deduped_rows[tuple(row.get(column) for column in conflict_columns)] = row
    row_list = list(deduped_rows.values())

    active_engine = engine or get_engine()
    metadata = MetaData()
    table = Table(table_name, metadata, autoload_with=active_engine)
    insert_stmt = pg_insert(table).values(row_list)
    update_columns = {
        column.name: insert_stmt.excluded[column.name]
        for column in table.columns
        if column.name not in set(conflict_columns) and column.name != "created_at"
    }
    statement = insert_stmt.on_conflict_do_update(index_elements=conflict_columns, set_=update_columns)
    with active_engine.begin() as connection:
        connection.execute(statement)
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