from __future__ import annotations
import os
import pandas as pd
from sqlalchemy import create_engine, text

_ENGINE = None

def get_engine():
    global _ENGINE
    if _ENGINE is None:
        url = os.environ.get(
            "DATABASE_URL",
            "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb"
        )
        _ENGINE = create_engine(url, pool_pre_ping=True)
    return _ENGINE

def upsert_df(df: pd.DataFrame, table: str, pk: list[str]) -> int:
    if df is None or df.empty:
        return 0
    eng = get_engine()
    df = df.copy()
    with eng.begin() as cx:
        tmp = f"tmp_{table}"
        df.to_sql(tmp, cx, if_exists="replace", index=False)
        cols = ",".join(df.columns)
        updates = ",".join([f"{c}=EXCLUDED.{c}" for c in df.columns if c not in pk])
        pkcols = ",".join(pk)
        cx.execute(text(f"""
            INSERT INTO {table} ({cols})
            SELECT {cols} FROM {tmp}
            ON CONFLICT ({pkcols}) DO UPDATE SET {updates}
        """))
        cx.execute(text(f"DROP TABLE {tmp}"))
    return len(df)

# Optional thin wrapper so other modules can call upsert(rows, pk=...)
def upsert(table: str, rows, eng=None, pk=()):
    df = pd.DataFrame(list(rows)) if not isinstance(rows, pd.DataFrame) else rows
    return upsert_df(df, table, list(pk))
