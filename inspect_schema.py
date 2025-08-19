"""Utility script to inspect key database tables/views for the MLB predictions API.

Prints column lists and a small sample plus recent date info for:
  - enhanced_games (table)
  - latest_probability_predictions (view)

Adjust the connection string via environment variables if needed:
  PG_USER, PG_PASS, PG_HOST, PG_PORT, PG_DB
Defaults assume local dev: mlbuser / mlbpass @ localhost:5432 / mlb
"""
from __future__ import annotations
import os
from datetime import date
from sqlalchemy import create_engine, text
import pandas as pd

USER = os.getenv("PG_USER", "mlbuser")
PASS = os.getenv("PG_PASS", "mlbpass")
HOST = os.getenv("PG_HOST", "localhost")
PORT = os.getenv("PG_PORT", "5432")
DB   = os.getenv("PG_DB", "mlb")

ENGINE_URL = f"postgresql+psycopg2://{USER}:{PASS}@{HOST}:{PORT}/{DB}"
engine = create_engine(ENGINE_URL)

def describe_relation(name: str):
    print("\n===", name, "===")
    # information_schema query (works for tables; for views columns also present)
    cols_q = text(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = current_schema() AND table_name = :t
        ORDER BY ordinal_position
        """
    )
    cols = pd.read_sql(cols_q, engine, params={"t": name})
    if cols.empty:
        print("(No columns found in current schema. Maybe different schema or permission issue.)")
        return
    print("Columns ({}):".format(len(cols)))
    print(", ".join(cols.column_name))

    # Try to infer date column name
    candidate_date_cols = [c for c in cols.column_name if c in {"date", "game_date", "dt", "gamedate"}]
    date_col = candidate_date_cols[0] if candidate_date_cols else None

    # Get recent rows
    limit_q = f"SELECT * FROM {name} ORDER BY 1 DESC LIMIT 5"  # naive; just to show structure
    try:
        sample = pd.read_sql(text(limit_q), engine)
        print("Sample rows (top 5 by first column desc):")
        print(sample.head().to_string(index=False)[:1000])
    except Exception as e:
        print("Could not fetch sample rows:", e)

    if date_col:
        stats_q = text(f"SELECT MIN({date_col}) AS min_date, MAX({date_col}) AS max_date, COUNT(*) AS cnt FROM {name}")
        try:
            stats = pd.read_sql(stats_q, engine).iloc[0]
            print(f"Date span via {date_col}: {stats.min_date} -> {stats.max_date} (rows={stats.cnt})")
            # Show today's rows if present
            today = date.today().isoformat()
            today_q = text(f"SELECT COUNT(*) AS n_today FROM {name} WHERE {date_col} = :d")
            n_today = pd.read_sql(today_q, engine, params={"d": today}).iloc[0].n_today
            print(f"Rows for today ({today}): {n_today}")
        except Exception as e:
            print("Date stats error:", e)
    else:
        print("No obvious date column among:", ", ".join(cols.column_name))

def main():
    for rel in ["enhanced_games", "latest_probability_predictions"]:
        try:
            describe_relation(rel)
        except Exception as e:
            print(f"Error describing {rel}: {e}")

if __name__ == "__main__":
    main()
