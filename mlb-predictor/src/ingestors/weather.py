from __future__ import annotations

import argparse
from datetime import datetime, time, timezone

import requests

from src.utils.cli import add_date_range_args, resolve_date_range
from src.utils.db import query_df, upsert_rows
from src.utils.logging import get_logger


log = get_logger(__name__)


def _weather_endpoint(game_date) -> tuple[str, str]:
    today = datetime.now(timezone.utc).date()
    if game_date < today:
        return "https://archive-api.open-meteo.com/v1/archive", "archive"
    return "https://api.open-meteo.com/v1/forecast", "forecast"


def _target_local_dt(game_date, game_start_ts):
    if game_start_ts is not None:
        return game_start_ts.replace(tzinfo=None)
    return datetime.combine(game_date, time(hour=19, minute=0))


def _pick_hour(payload: dict[str, object], target_dt: datetime) -> dict[str, object] | None:
    hourly = payload.get("hourly") or {}
    timestamps = hourly.get("time") or []
    if not timestamps:
        return None
    parsed = [datetime.fromisoformat(value) for value in timestamps]
    index = min(range(len(parsed)), key=lambda idx: abs(parsed[idx] - target_dt))
    return {
        "time": parsed[index],
        "temperature_f": (hourly.get("temperature_2m") or [None])[index],
        "humidity_pct": (hourly.get("relative_humidity_2m") or [None])[index],
        "precipitation_pct": (hourly.get("precipitation_probability") or [None])[index],
        "pressure_hpa": (hourly.get("pressure_msl") or [None])[index],
        "wind_speed_mph": (hourly.get("wind_speed_10m") or [None])[index],
        "wind_direction_deg": (hourly.get("wind_direction_10m") or [None])[index],
    }


def _serialized_hour_payload(selected: dict[str, object]) -> dict[str, object]:
    payload = dict(selected)
    if isinstance(payload.get("time"), datetime):
        payload["time"] = payload["time"].isoformat()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch weather snapshots from Open-Meteo")
    add_date_range_args(parser)
    args = parser.parse_args()
    start_date, end_date = resolve_date_range(args)

    games = query_df(
        """
        SELECT g.game_id, g.game_date, g.game_start_ts, v.latitude, v.longitude, v.roof_type
        FROM games g
        JOIN dim_venues v ON v.venue_id = g.venue_id
        WHERE g.game_date BETWEEN :start_date AND :end_date
          AND v.latitude IS NOT NULL
          AND v.longitude IS NOT NULL
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    if games.empty:
        log.info("No games with venue coordinates available for weather snapshots")
        return 0

    snapshot_ts = datetime.now(timezone.utc)
    rows = []
    for game in games.itertuples(index=False):
        endpoint, weather_type = _weather_endpoint(game.game_date)
        params = {
            "latitude": float(game.latitude),
            "longitude": float(game.longitude),
            "start_date": game.game_date.isoformat(),
            "end_date": game.game_date.isoformat(),
            "hourly": "temperature_2m,relative_humidity_2m,precipitation_probability,pressure_msl,wind_speed_10m,wind_direction_10m",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "auto",
        }
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        selected = _pick_hour(payload, _target_local_dt(game.game_date, game.game_start_ts))
        if not selected:
            continue
        roof_type = (game.roof_type or "").lower() if game.roof_type else None
        rows.append(
            {
                "game_id": int(game.game_id),
                "game_date": game.game_date,
                "snapshot_ts": snapshot_ts,
                "source_name": "open-meteo",
                "weather_type": weather_type,
                "temperature_f": selected["temperature_f"],
                "wind_speed_mph": selected["wind_speed_mph"],
                "wind_direction_deg": selected["wind_direction_deg"],
                "humidity_pct": selected["humidity_pct"],
                "precipitation_pct": selected["precipitation_pct"],
                "pressure_hpa": selected["pressure_hpa"],
                "roof_open_flag": False if roof_type == "dome" else None,
                "raw_payload": _serialized_hour_payload(selected),
            }
        )

    inserted = upsert_rows("game_weather", rows, ["game_id", "snapshot_ts", "source_name"])
    log.info("Upserted %s weather rows", inserted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())