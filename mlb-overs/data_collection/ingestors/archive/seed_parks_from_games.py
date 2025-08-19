# ingestors/weather_game.py
from __future__ import annotations

import argparse
import datetime as dt
import math
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import requests
import statsapi
from sqlalchemy import create_engine, text


# ----------------- helpers -----------------
def now_utc_date() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()


def parse_game_datetime_utc(g: Dict[str, Any]) -> Optional[dt.datetime]:
    """Return scheduled first pitch in UTC (rounded to hour) if present."""
    try:
        iso = ((g.get("gameData") or {}).get("datetime") or {}).get("dateTime")
        if not iso:
            # fallback: across payloads sometimes 'gameDate' appears under gameData
            iso = (g.get("gameData") or {}).get("gameDate")
        if not iso:
            return None
        t = dt.datetime.fromisoformat(iso.replace("Z", "+00:00"))
        # round to nearest hour for Open-Meteo indexing
        minute = t.minute
        if minute >= 30:
            t = t + dt.timedelta(minutes=(60 - minute))
        else:
            t = t - dt.timedelta(minutes=minute)
        return t.astimezone(dt.timezone.utc)
    except Exception:
        return None


def geocode_latlon(q: str) -> Tuple[Optional[float], Optional[float]]:
    """Open-Meteo geocoder (no key)."""
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
            return float(it["latitude"]), float(it["longitude"])
    except Exception:
        pass
    return None, None


def coords_from_game(game_pk: int) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[int], Dict[str, Any]]:
    """
    Return (lat, lon, roof_type, altitude_ft, raw_game) using MLB game payload,
    falling back to geocoding if coords missing.
    """
    g = statsapi.get("game", {"gamePk": int(game_pk)}) or {}
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
    if lat and lon:
        return float(lat), float(lon), roof, altitude_ft, g

    # last-ditch geocode: "{venue} {city} {state} MLB stadium" → team city
    vname = v.get("name") or ""
    city = loc.get("city") or ""
    state = loc.get("state") or loc.get("stateAbbrev") or ""
    q = " ".join(x for x in [vname, city, state, "MLB stadium"] if x)
    glat, glon = geocode_latlon(q)
    if glat is None or glon is None:
        home_meta = ((g.get("gameData") or {}).get("teams") or {}).get("home", {}) or {}
        team_city = home_meta.get("locationName") or ""
        if team_city:
            glat, glon = geocode_latlon(f"{team_city}, {state or ''} MLB")
    return glat, glon, roof, altitude_ft, g


def wind_components(speed_mph: Optional[float], wind_dir_deg: Optional[float], cf_azimuth_deg: Optional[int]) -> Tuple[Optional[float], Optional[float]]:
    """
    Resolve wind into 'out-to-center' and 'cross' components given a center-field azimuth.
    If cf_azimuth_deg is missing, returns (None, None).
    """
    if speed_mph is None or wind_dir_deg is None or cf_azimuth_deg is None:
        return None, None
    # meteorological direction: wind FROM theta (deg clockwise from north)
    # stadium azimuth: direction TO center field (deg clockwise from north)
    # component toward CF is negative of dot product with wind FROM; convert to TO by adding 180 deg.
    theta_to = (wind_dir_deg + 180.0) % 360.0
    delta = math.radians(theta_to - float(cf_azimuth_deg))
    out_comp = speed_mph * math.cos(delta)
    cross_comp = speed_mph * math.sin(delta)
    return round(out_comp, 2), round(cross_comp, 2)


def fetch_open_meteo_hour(lat: float, lon: float, when_utc: dt.datetime) -> Dict[str, Optional[float]]:
    """Get hourly weather at a UTC time from Open-Meteo forecast/archives."""
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
        # Align index
        ts = [dt.datetime.fromisoformat(t.replace("Z", "+00:00")).replace(tzinfo=dt.timezone.utc) for t in times]
        # find the exact hour
        try:
            idx = ts.index(when_utc.replace(minute=0, second=0, microsecond=0))
        except ValueError:
            # fall back to nearest hour
            diffs = [abs((t - when_utc).total_seconds()) for t in ts]
            idx = diffs.index(min(diffs))
        out = {
            "temp_f": None,
            "humidity_pct": None,
            "wind_mph": None,
            "wind_dir_deg": None,
            "precip_prob": None,
        }
        def getv(key):
            seq = hourly.get(key) or []
            return seq[idx] if idx < len(seq) else None

        temp_c = getv("temperature_2m")
        if temp_c is not None:
            out["temp_f"] = round(float(temp_c) * 9.0/5.0 + 32.0, 1)
        out["humidity_pct"] = None if getv("relative_humidity_2m") is None else float(getv("relative_humidity_2m"))
        wind_kmh = getv("wind_speed_10m")
        if wind_kmh is not None:
            out["wind_mph"] = round(float(wind_kmh) * 0.621371, 1)
        out["wind_dir_deg"] = None if getv("wind_direction_10m") is None else float(getv("wind_direction_10m"))
        out["precip_prob"] = None if getv("precipitation_probability") is None else int(getv("precipitation_probability"))
        return out
    except Exception:
        return {}


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--database-url", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--sleep", type=float, default=0.1)
    args = ap.parse_args()

    eng = create_engine(args.database_url)

    # Ensure new columns exist
    with eng.begin() as cx:
        cx.execute(text("""
            ALTER TABLE weather_game
              ADD COLUMN IF NOT EXISTS wind_out_mph NUMERIC,
              ADD COLUMN IF NOT EXISTS wind_cross_mph NUMERIC,
              ADD COLUMN IF NOT EXISTS is_forecast BOOL,
              ADD COLUMN IF NOT EXISTS valid_hour_utc TIMESTAMPTZ
        """))

    # Get games (we'll resolve coords per-game)
    games = pd.read_sql(text("""
        SELECT game_id::bigint AS game_id, date, park_id
        FROM games
        WHERE date BETWEEN :s AND :e
        ORDER BY date, game_id
    """), eng, params={"s": args.start, "e": args.end})

    if games.empty:
        print("no games in window")
        return

    rows = []
    today = now_utc_date()

    # Optional: pull CF azimuth per park if you add it later
    cf_map = {}
    try:
        parks = pd.read_sql("SELECT park_id, cf_azimuth_deg FROM parks", eng)
        cf_map = {str(r.park_id): (None if pd.isna(r.cf_azimuth_deg) else int(r.cf_azimuth_deg)) for _, r in parks.iterrows()}
    except Exception:
        pass

    for gr in games.itertuples(index=False):
        gid = int(gr.game_id)
        lat, lon, roof, alt_ft, graw = coords_from_game(gid)
        if lat is None or lon is None:
            continue  # give up if we still don't have coords

        # roof handling: closed/dome → treat as no wind
        is_closed = (roof or "").lower() in {"roof_closed", "closed", "dome"}

        # hour to fetch
        when = parse_game_datetime_utc(graw) or dt.datetime.combine(pd.to_datetime(gr.date).to_pydatetime().date(), dt.time(0, 0), tzinfo=dt.timezone.utc)
        wx = {"temp_f": None, "humidity_pct": None, "wind_mph": None, "wind_dir_deg": None, "precip_prob": None}
        is_forecast = None

        if not is_closed:
            wx = fetch_open_meteo_hour(lat, lon, when)
            is_forecast = (gr.date >= today)
            if not wx:
                # leave nulls; still insert a row so features can join
                wx = {"temp_f": None, "humidity_pct": None, "wind_mph": None, "wind_dir_deg": None, "precip_prob": None}

        # wind components
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
            "air_density_idx": None,  # optional: compute later if desired
            "is_forecast": is_forecast,
            "valid_hour_utc": when.replace(minute=0, second=0, microsecond=0),
            "wind_out_mph": 0.0 if is_closed and cf_az is not None else wind_out,
            "wind_cross_mph": 0.0 if is_closed and cf_az is not None else wind_cross,
        })

    if not rows:
        print("no weather rows")
        return

    df = pd.DataFrame(rows).drop_duplicates("game_id")

    with eng.begin() as cx:
        df.to_sql("tmp_weather_up", cx, index=False, if_exists="replace")
        cx.execute(text("""
            INSERT INTO weather_game (game_id, temp_f, humidity_pct, wind_mph, wind_dir_deg, precip_prob,
                                      altitude_ft, air_density_idx, is_forecast, valid_hour_utc,
                                      wind_out_mph, wind_cross_mph)
            SELECT game_id, temp_f, humidity_pct, wind_mph, wind_dir_deg, precip_prob,
                   altitude_ft, air_density_idx, is_forecast, valid_hour_utc,
                   wind_out_mph, wind_cross_mph
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

    print(f"upserted weather rows: {len(df)}")


if __name__ == "__main__":
    main()
