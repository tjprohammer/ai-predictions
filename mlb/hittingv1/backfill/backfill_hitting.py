import argparse, os
from sqlalchemy import create_engine, text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()

    url = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
    engine = create_engine(url, pool_pre_ping=True)

    with engine.begin() as conn:
        for mv in ["mv_hitter_form","mv_bvp_agg","mv_pa_distribution"]:
            conn.execute(text(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {mv}"))

    print(f"[hitting-backfill] Refreshed materialized views for {args.start} → {args.end}")

if __name__ == "__main__":
    main()
