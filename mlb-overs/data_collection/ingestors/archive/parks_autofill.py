from __future__ import annotations
import argparse, time, pandas as pd
from sqlalchemy import create_engine, text
import statsapi

def get_venue_meta(venue_id: int) -> dict:
    v = statsapi.get("venues", {"venueIds": venue_id}) or {}
    v = (v.get("venues") or [{}])[0]
    loc = v.get("location") or {}
    # different payloads sometimes use 'coordinates' or 'location'
    lat = loc.get("latitude") or (v.get("coordinates") or {}).get("latitude")
    lon = loc.get("longitude") or (v.get("coordinates") or {}).get("longitude")
    roof = (v.get("roofType") or v.get("roof_type") or "").replace(" ", "_").lower() or None
    elev_m = (loc.get("elevation") or v.get("elevation"))  # meters if present
    alt_ft = int(round(float(elev_m)*3.28084)) if elev_m not in (None, "") else None
    return {
        "park_id": str(v.get("id") or venue_id),
        "name": v.get("name"),
        "lat": float(lat) if lat not in (None, "") else None,
        "lon": float(lon) if lon not in (None, "") else None,
        "altitude_ft": alt_ft,
        "roof_type": roof or None
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--database-url", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()
    eng = create_engine(args.database_url)

    # Pull games in window; fill missing park_id from MLB game meta
    games = pd.read_sql(text("""
        SELECT game_id::bigint AS game_id, date, park_id
        FROM games
        WHERE date BETWEEN :s AND :e
    """), eng, params={"s": args.start, "e": args.end})

    if games.empty:
        print("no games in range"); return

    # Collect venue ids (prefer existing park_id else MLB gameData.venue.id)
    venue_ids = set()
    for r in games.itertuples(index=False):
        if r.park_id:  # already have one
            try: venue_ids.add(int(r.park_id))
            except: pass
            continue
        g = statsapi.get("game", {"gamePk": int(r.game_id)}) or {}
        vid = (((g.get("gameData") or {}).get("venue") or {}).get("id"))
        if vid: venue_ids.add(int(vid))
        time.sleep(args.sleep)

    if not venue_ids:
        print("no venues discovered"); return

    rows = []
    for vid in sorted(venue_ids):
        try:
            rows.append(get_venue_meta(vid))
        except Exception:
            continue
        time.sleep(args.sleep)

    df = pd.DataFrame(rows)
    if df.empty:
        print("no venue metadata"); return

    with eng.begin() as cx:
        cx.execute(text("""
            ALTER TABLE parks
              ADD COLUMN IF NOT EXISTS lat NUMERIC,
              ADD COLUMN IF NOT EXISTS lon NUMERIC,
              ADD COLUMN IF NOT EXISTS cf_azimuth_deg INT
        """))
        df.to_sql("tmp_parks_auto", cx, index=False, if_exists="replace")
        cx.execute(text("""
            INSERT INTO parks (park_id, name, pf_runs_3y, pf_hr_3y, altitude_ft, roof_type, lat, lon, cf_azimuth_deg)
            SELECT park_id, name, NULL, NULL, altitude_ft, roof_type, lat, lon, NULL
            FROM tmp_parks_auto
            ON CONFLICT (park_id) DO UPDATE SET
              name=EXCLUDED.name,
              altitude_ft=COALESCE(EXCLUDED.altitude_ft, parks.altitude_ft),
              roof_type=COALESCE(EXCLUDED.roof_type, parks.roof_type),
              lat=COALESCE(EXCLUDED.lat, parks.lat),
              lon=COALESCE(EXCLUDED.lon, parks.lon);
        """))
        cx.execute(text("DROP TABLE tmp_parks_auto"))
    print(f"seeded/updated {len(df)} parks")

if __name__ == "__main__":
    main()
