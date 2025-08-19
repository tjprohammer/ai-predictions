# ingestors/weather_game.py
from __future__ import annotations

import argparse
import datetime as dt
import math
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
import requests
import statsapi
from sqlalchemy import create_engine, text

# Track venues we fail to geocode so we can plug gaps fast
MISSES: List[Tuple[str, str, str, int, str]] = []

FALLBACK_VENUE_COORDS = {
    "guaranteed rate field": (41.8299, -87.6339),
    "rate field": (41.8299, -87.6339),
    "minute maid park": (29.7573, -95.3555),
    "daikin park": (29.7573, -95.3555),
    "sutter health park": (38.5803, -121.5130),
    "petco park": (32.7076, -117.1569),
    "dodger stadium": (34.0739, -118.2400),
    "oracle park": (37.7786, -122.3893),
    "t-mobile park": (47.5914, -122.3325),
    "chase field": (33.4455, -112.0667),
    "busch stadium": (38.6226, -90.1928),
    "target field": (44.9817, -93.2789),
    "american family field": (43.0280, -87.9712),
    "pnc park": (40.4469, -80.0057),
    "oriole park at camden yards": (39.2839, -76.6217),
    "camden yards": (39.2839, -76.6217),
    "comerica park": (42.3390, -83.0485),
    "truist park": (33.8908, -84.4678),
    "yankee stadium": (40.8296, -73.9262),
    "great american ball park": (39.0975, -84.5066),
    "kauffman stadium": (39.0517, -94.4803),
    "angel stadium": (33.8003, -117.8827),
    "rogers centre": (43.6414, -79.3894),
    "citi field": (40.7571, -73.8458),
    "progressive field": (41.4962, -81.6852),
    "globe life field": (32.7473, -97.0843),
    "globe life park":  (32.7513, -97.0840),
}

def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def now_utc_date() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()

def parse_game_datetime_utc(g: Dict[str, Any]) -> Optional[dt.datetime]:
    try:
        iso = ((g.get("gameData") or {}).get("datetime") or {}).get("dateTime")
        if not iso:
            iso = (g.get("gameData") or {}).get("gameDate")
        if not iso:
            return None
        t = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
        # snap to nearest hour (00 or +1h) for hourly forecast alignment
        minute = t.minute
        t = t + dt.timedelta(minutes=(60 - minute)) if minute >= 30 else t - dt.timedelta(minutes=minute)
        return t.astimezone(dt.timezone.utc)
    except Exception:
        return None

def geocode_latlon_candidates(
    queries: List[str],
    allowed_cc = {"US","CA"}
) -> Tuple[Optional[float], Optional[float], str, str]:
    # Only try non-empty queries, keep to US/CA hits to avoid “Flushing, NL”
    for q in [q.strip() for q in queries if q and q.strip()]:
        try:
            r = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": q, "count": 1},
                timeout=15,
            )
            r.raise_for_status()
            d = r.json()
            if d.get("results"):
                it = d["results"][0]
                cc = (it.get("country_code") or "").upper()
                if allowed_cc and cc not in allowed_cc:
                    continue
                return float(it["latitude"]), float(it["longitude"]), q, cc
        except Exception:
            pass
    return None, None, "", ""

def fetch_venue_coords_http(vid: int, debug: bool=False) -> Tuple[Optional[float], Optional[float]]:
    # Try ?venueIds= first (list response), then /venues/{id} (object response)
    urls = [
        f"https://statsapi.mlb.com/api/v1/venues?venueIds={vid}",
        f"https://statsapi.mlb.com/api/v1/venues/{vid}",
    ]
    for u in urls:
        try:
            r = requests.get(u, timeout=15, headers={"User-Agent":"mlb-overs/1.0"})
            r.raise_for_status()
            d = r.json()
            if isinstance(d, dict) and "venues" in d and d["venues"]:
                v = d["venues"][0]
            else:
                v = d if isinstance(d, dict) else {}
            loc = v.get("coordinates") or v.get("location") or {}
            lat = loc.get("latitude"); lon = loc.get("longitude")
            if lat and lon:
                if debug: print(f"[coords_from_game] HTTP venues -> {lat},{lon}")
                return float(lat), float(lon)
        except Exception:
            continue
    return None, None

def coords_from_game(
    game_pk: int,
    debug: bool=False
) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[int], Dict[str, Any], str]:
    how = ""
    try:
        g = statsapi.get("game", {"gamePk": int(game_pk)}) or {}
    except Exception as e:
        MISSES.append(("?", "?", "?", int(game_pk), f"statsapi.get error: {e}"))
        return None, None, None, None, {}, f"statsapi.get error: {e}"

    v = ((g.get("gameData") or {}).get("venue") or {})
    coords = v.get("coordinates") or {}
    lat = coords.get("latitude")
    lon = coords.get("longitude")

    roof = (v.get("roofType") or v.get("roof_type") or "")
    roof = roof.replace(" ", "_").lower() or None

    altitude_ft = None
    loc = v.get("location") or {}
    elev_m = loc.get("elevation") or v.get("elevation")
    if elev_m not in (None, ""):
        try:
            altitude_ft = int(round(float(elev_m) * 3.28084))
        except Exception:
            pass

    vname = v.get("name") or ""
    city = loc.get("city") or ""
    state = loc.get("state") or loc.get("stateAbbrev") or ""

    if debug:
        print(f"[coords_from_game] gamePk={game_pk} venue='{vname}' city='{city}' state='{state}' latlon_from_game={lat},{lon}")

    if lat and lon:
        return float(lat), float(lon), roof, altitude_ft, g, "gameData.coordinates"

    vid = v.get("id")
    if vid:
        vlat, vlon = fetch_venue_coords_http(int(vid), debug=debug)
        if vlat and vlon:
            return vlat, vlon, roof, altitude_ft, g, "http.venues"

    qlist = [
        vname,
        ", ".join([p for p in [vname, city, state] if p]),
        ", ".join([p for p in [city, state] if p]),
    ]
    glat, glon, used_q, cc = geocode_latlon_candidates(qlist)
    if debug:
        print(f"[coords_from_game] geocode used='{used_q}' -> {glat},{glon} country={cc}")
    if glat is not None and glon is not None:
        # if the match was a bare city/state and we have a stadium pin, prefer the stadium
        if used_q == ", ".join([p for p in [city, state] if p]) and _norm(vname) in FALLBACK_VENUE_COORDS:
            latlon = FALLBACK_VENUE_COORDS[_norm(vname)]
            if debug:
                print("[coords_from_game] prefer static_map over city/state geocode")
            return latlon[0], latlon[1], roof, altitude_ft, g, "static_map_preferred"
        return glat, glon, roof, altitude_ft, g, f"geocode:{used_q}"

    key = _norm(vname)
    if key in FALLBACK_VENUE_COORDS:
        latlon = FALLBACK_VENUE_COORDS[key]
        if debug:
            print(f"[coords_from_game] static map for '{vname}' -> {latlon}")
        return latlon[0], latlon[1], roof, altitude_ft, g, "static_map"

    MISSES.append((vname or "?", city or "?", state or "?", int(game_pk), "no coords after all methods"))
    return None, None, roof, altitude_ft, g, "no coords after all methods"

def wind_components(
    speed_mph: Optional[float],
    wind_dir_deg: Optional[float],
    cf_azimuth_deg: Optional[int]
) -> Tuple[Optional[float], Optional[float]]:
    if speed_mph is None or wind_dir_deg is None or cf_azimuth_deg is None:
        return None, None
    theta_to = (float(wind_dir_deg) + 180.0) % 360.0
    delta = math.radians(theta_to - float(cf_azimuth_deg))
    out_comp = speed_mph * math.cos(delta)
    cross_comp = speed_mph * math.sin(delta)
    return round(out_comp, 2), round(cross_comp, 2)

def fetch_open_meteo_hour(lat: float, lon: float, when_utc: dt.datetime) -> Dict[str, Optional[float]]:
    day = when_utc.date().isoformat()
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation_probability"
        ]),
        "start_date": day,
        "end_date": day,
        "timezone": "UTC",
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        d = r.json()
        hourly = d.get("hourly") or {}
        times = hourly.get("time") or []
        if not times:
            return {}
        ts = [dt.datetime.fromisoformat(t.replace("Z", "+00:00")).astimezone(dt.timezone.utc) for t in times]
        target = when_utc.replace(minute=0, second=0, microsecond=0)
        try:
            idx = ts.index(target)
        except ValueError:
            diffs = [abs((t - target).total_seconds()) for t in ts]
            idx = diffs.index(min(diffs))
        out = {"temp_f": None, "humidity_pct": None, "wind_mph": None, "wind_dir_deg": None, "precip_prob": None}
        def getv(key):
            seq = hourly.get(key) or []
            return seq[idx] if idx < len(seq) else None
        temp_c = getv("temperature_2m")
        if temp_c is not None:
            out["temp_f"] = round(float(temp_c) * 9.0/5.0 + 32.0, 1)
        rh = getv("relative_humidity_2m")
        out["humidity_pct"] = None if rh is None else float(rh)
        wind_kmh = getv("wind_speed_10m")
        if wind_kmh is not None:
            out["wind_mph"] = round(float(wind_kmh) * 0.621371, 1)
        wd = getv("wind_direction_10m")
        out["wind_dir_deg"] = None if wd is None else float(wd)
        pp = getv("precipitation_probability")
        out["precip_prob"] = None if pp is None else int(pp)
        return out
    except Exception:
        return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--database-url", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--sleep", type=float, default=0.1)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    eng = create_engine(args.database_url)

    with eng.begin() as cx:
        cx.execute(text("""
            ALTER TABLE weather_game
              ADD COLUMN IF NOT EXISTS wind_out_mph NUMERIC,
              ADD COLUMN IF NOT EXISTS wind_cross_mph NUMERIC,
              ADD COLUMN IF NOT EXISTS is_forecast BOOL,
              ADD COLUMN IF NOT EXISTS valid_hour_utc TIMESTAMPTZ
        """))

    games = pd.read_sql(text("""
        SELECT game_id::bigint AS game_id, date, park_id
        FROM games
        WHERE date BETWEEN :s AND :e
        ORDER BY date, game_id
    """), eng, params={"s": args.start, "e": args.end})

    if games.empty:
        print("no games in window"); return

    if args.debug:
        print(f"[debug] games in window: {len(games)}")
        print("[debug] sample game_ids:", games["game_id"].astype(str).head(10).tolist())

    rows = []
    today = now_utc_date()

    cf_map: Dict[str, Optional[int]] = {}
    try:
        parks = pd.read_sql("SELECT park_id, cf_azimuth_deg FROM parks", eng)
        cf_map = {str(r.park_id): (None if pd.isna(r.cf_azimuth_deg) else int(r.cf_azimuth_deg))
                  for _, r in parks.iterrows()}
    except Exception:
        pass

    skipped = 0
    for gr in games.itertuples(index=False):
        gid = int(gr.game_id)
        lat, lon, roof, alt_ft, graw, how = coords_from_game(gid, debug=args.debug)
        if lat is None or lon is None:
            skipped += 1
            if args.debug:
                print(f"[debug] skip game_id={gid}: {how}")
            continue

        is_closed = (roof or "").lower() in {"roof_closed", "closed", "dome"}

        when = parse_game_datetime_utc(graw)
        if when is None:
            when = dt.datetime.combine(pd.to_datetime(gr.date).date(), dt.time(0, 0), tzinfo=dt.timezone.utc)

        wx = {"temp_f": None, "humidity_pct": None, "wind_mph": None, "wind_dir_deg": None, "precip_prob": None}
        is_forecast = (pd.to_datetime(gr.date).date() >= today)

        if not is_closed:
            fetched = fetch_open_meteo_hour(lat, lon, when)
            if fetched:
                wx = fetched

        cf_az = cf_map.get(str(gr.park_id)) if gr.park_id is not None else None
        wind_out, wind_cross = wind_components(wx.get("wind_mph"), wx.get("wind_dir_deg"), cf_az)

        rows.append({
            "game_id": gid,
            "temp_f": wx.get("temp_f"),
            "humidity_pct": wx.get("humidity_pct"),
            "wind_mph": 0.0 if is_closed else wx.get("wind_mph"),
            "wind_dir_deg": None if is_closed else wx.get("wind_dir_deg"),
            "precip_prob": None if is_closed else wx.get("precip_prob"),
            "altitude_ft": alt_ft,
            "air_density_idx": None,   # compute later if you want
            "is_forecast": is_forecast,
            "valid_hour_utc": when.replace(minute=0, second=0, microsecond=0),
            "wind_out_mph": 0.0 if is_closed and cf_az is not None else wind_out,
            "wind_cross_mph": 0.0 if is_closed and cf_az is not None else wind_cross,
        })

        if args.sleep:
            import time; time.sleep(args.sleep)

    if not rows:
        print("no weather rows")
        if MISSES:
            print("[missing coords] count:", len(MISSES))
        return

    df = pd.DataFrame(rows).drop_duplicates("game_id")

    with eng.begin() as cx:
        df.to_sql("tmp_weather_up", cx, index=False, if_exists="replace")
        # CAST on insert to avoid tmp TEXT dtype issues
        cx.execute(text("""
            INSERT INTO weather_game (game_id, temp_f, humidity_pct, wind_mph, wind_dir_deg, precip_prob,
                                      altitude_ft, air_density_idx, is_forecast, valid_hour_utc,
                                      wind_out_mph, wind_cross_mph)
            SELECT
              game_id::bigint,
              NULLIF(temp_f::text,'')::numeric,
              NULLIF(humidity_pct::text,'')::numeric,
              NULLIF(wind_mph::text,'')::numeric,
              NULLIF(wind_dir_deg::text,'')::numeric,
              NULLIF(precip_prob::text,'')::int,
              NULLIF(altitude_ft::text,'')::int,
              NULLIF(air_density_idx::text,'')::numeric,
              CASE
                WHEN is_forecast::text ILIKE 't%' OR is_forecast::text='1' OR is_forecast::text ILIKE 'true' THEN TRUE
                WHEN is_forecast::text ILIKE 'f%' OR is_forecast::text='0' OR is_forecast::text ILIKE 'false' THEN FALSE
                ELSE NULL
              END,
              (valid_hour_utc::timestamptz),
              NULLIF(wind_out_mph::text,'')::numeric,
              NULLIF(wind_cross_mph::text,'')::numeric
            FROM tmp_weather_up
            ON CONFLICT (game_id) DO UPDATE SET
              temp_f         = COALESCE(EXCLUDED.temp_f, weather_game.temp_f),
              humidity_pct   = COALESCE(EXCLUDED.humidity_pct, weather_game.humidity_pct),
              wind_mph       = COALESCE(EXCLUDED.wind_mph, weather_game.wind_mph),
              wind_dir_deg   = COALESCE(EXCLUDED.wind_dir_deg, weather_game.wind_dir_deg),
              precip_prob    = COALESCE(EXCLUDED.precip_prob, weather_game.precip_prob),
              altitude_ft    = COALESCE(EXCLUDED.altitude_ft, weather_game.altitude_ft),
              air_density_idx= COALESCE(EXCLUDED.air_density_idx, weather_game.air_density_idx),
              is_forecast    = COALESCE(EXCLUDED.is_forecast, weather_game.is_forecast),
              valid_hour_utc = COALESCE(EXCLUDED.valid_hour_utc, weather_game.valid_hour_utc),
              wind_out_mph   = COALESCE(EXCLUDED.wind_out_mph, weather_game.wind_out_mph),
              wind_cross_mph = COALESCE(EXCLUDED.wind_cross_mph, weather_game.wind_cross_mph)
        """))
        cx.execute(text("DROP TABLE IF EXISTS tmp_weather_up"))

    print(f"upserted weather rows: {len(df)} (skipped: {skipped})")

    # Summarize any coordinate misses so you can extend FALLBACK_VENUE_COORDS
    if MISSES:
        print("[missing coords] total misses:", len(MISSES))
        by_venue: Dict[Tuple[str,str,str], List[int]] = {}
        reasons: Dict[Tuple[str,str,str], str] = {}
        for nm, ct, st, gid, why in MISSES:
            k = (nm, ct, st)
            by_venue.setdefault(k, []).append(gid)
            # keep first reason seen for this key
            reasons.setdefault(k, why)
        for (nm, ct, st), gids in by_venue.items():
            sample = ", ".join(str(x) for x in gids[:3])
            print(f"  - {nm} ({ct}, {st}) | reason={reasons[(nm,ct,st)]} | sample_game_ids=[{sample}]")

if __name__ == "__main__":
    main()
