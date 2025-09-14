import os
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, text


def main(start_date: str, end_date: str):
    """
    Print per-date availability of predictions and results in enhanced_games for a range.
    Columns reported:
      - games: total rows
      - learn: non-null predicted_total (Learning Adaptive / primary)
      - ultra: non-null predicted_total_learning (Ultra 80 incremental)
      - finals: non-null total_runs (game finished)
    """
    # Validate dates
    _ = datetime.fromisoformat(start_date)
    _ = datetime.fromisoformat(end_date)

    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb",
    )
    engine = create_engine(db_url)

    q = text(
        """
        SELECT
            date::date AS d,
            COUNT(*) AS games,
            COUNT(predicted_total) AS learn,
            COUNT(predicted_total_learning) AS ultra,
            COUNT(total_runs) AS finals
        FROM enhanced_games
        WHERE date BETWEEN :s AND :e
        GROUP BY 1
        ORDER BY 1
        """
    )

    df = pd.read_sql(q, engine, params={"s": start_date, "e": end_date})
    if df.empty:
        print(f"No enhanced_games rows between {start_date} and {end_date}")
        return
    # Format
    df = df.sort_values("d")
    print(df.to_string(index=False))


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python mlb/utils/check_backfill_status.py START_DATE END_DATE")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
