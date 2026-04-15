from __future__ import annotations

from typing import Any

from src.utils.db import get_dialect_name

# Resolved once at import; matches previous behavior in app.py.
DB_DIALECT = str(get_dialect_name()).lower()


def _sql_json_text(column: str, key: str, dialect: str | None = None) -> str:
    active_dialect = (dialect or DB_DIALECT).lower()
    if active_dialect == "sqlite":
        return f"json_extract({column}, '$.{key}')"
    return f"{column} ->> '{key}'"


def _sql_real(expression: str, dialect: str | None = None) -> str:
    active_dialect = (dialect or DB_DIALECT).lower()
    target_type = "REAL" if active_dialect == "sqlite" else "DOUBLE PRECISION"
    return f"CAST({expression} AS {target_type})"


def _sql_integer(expression: str, dialect: str | None = None) -> str:
    active_dialect = (dialect or DB_DIALECT).lower()
    target_type = "INTEGER" if active_dialect == "sqlite" else "SMALLINT"
    return f"CAST({expression} AS {target_type})"


def _sql_boolean(expression: str, dialect: str | None = None) -> str:
    active_dialect = (dialect or DB_DIALECT).lower()
    if active_dialect != "sqlite":
        return f"CAST({expression} AS BOOLEAN)"
    normalized = f"LOWER(TRIM(COALESCE({expression}, '')))"
    return (
        "CASE "
        f"WHEN {normalized} IN ('true', '1', 't', 'yes', 'y') THEN 1 "
        f"WHEN {normalized} IN ('false', '0', 'f', 'no', 'n') THEN 0 "
        "ELSE NULL END"
    )


def _sql_ratio(numerator_expression: str, denominator_expression: str) -> str:
    return (
        f"CASE WHEN SUM({denominator_expression}) = 0 THEN NULL "
        f"ELSE (1.0 * SUM({numerator_expression}) / SUM({denominator_expression})) END"
    )


def _sql_year(expression: str, dialect: str | None = None) -> str:
    active_dialect = (dialect or DB_DIALECT).lower()
    if active_dialect == "sqlite":
        return f"CAST(strftime('%Y', {expression}) AS INTEGER)"
    return f"EXTRACT(YEAR FROM {expression})"


def _sql_year_param(param_name: str, dialect: str | None = None) -> str:
    active_dialect = (dialect or DB_DIALECT).lower()
    if active_dialect == "sqlite":
        return _sql_year(f":{param_name}", active_dialect)
    return _sql_year(f"CAST(:{param_name} AS DATE)", active_dialect)


def _sql_order_nulls_last(expression: str, direction: str = "ASC") -> str:
    normalized_direction = direction.upper()
    return f"CASE WHEN {expression} IS NULL THEN 1 ELSE 0 END, {expression} {normalized_direction}"


def _sql_bind_list(prefix: str, values: list[Any], params: dict[str, Any]) -> str:
    placeholders: list[str] = []
    for index, value in enumerate(values):
        key = f"{prefix}_{index}"
        params[key] = value
        placeholders.append(f":{key}")
    if not placeholders:
        return "NULL"
    return ", ".join(placeholders)
