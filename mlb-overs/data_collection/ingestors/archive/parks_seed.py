from __future__ import annotations
import argparse, pandas as pd
from sqlalchemy import create_engine, text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--database-url", required=True)
    ap.add_argument("--csv", required=True)  # data/parks.csv
    args = ap.parse_args()
    eng = create_engine(args.database_url)

    df = pd.read_csv(args.csv)
    with eng.begin() as cx:
        df.to_sql("tmp_parks", cx, index=False, if_exists="replace")
        cx.execute(text("""
            INSERT INTO parks (park_id, name, pf_runs_3y, pf_hr_3y, altitude_ft, roof_type, lat, lon, cf_azimuth_deg)
            SELECT park_id::text, name, pf_runs_3y, pf_hr_3y, altitude_ft, roof_type, lat, lon, cf_azimuth_deg
            FROM tmp_parks
            ON CONFLICT (park_id) DO UPDATE SET
              name=EXCLUDED.name, pf_runs_3y=EXCLUDED.pf_runs_3y, pf_hr_3y=EXCLUDED.pf_hr_3y,
              altitude_ft=EXCLUDED.altitude_ft, roof_type=EXCLUDED.roof_type,
              lat=EXCLUDED.lat, lon=EXCLUDED.lon, cf_azimuth_deg=EXCLUDED.cf_azimuth_deg;
        """))
        cx.execute(text("DROP TABLE tmp_parks"))
    print(f"upserted {len(df)} parks")

if __name__ == "__main__":
    main()
