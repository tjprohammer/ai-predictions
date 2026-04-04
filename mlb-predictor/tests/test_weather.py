from __future__ import annotations

from datetime import date, datetime, timezone

from src.ingestors import weather


def test_weather_endpoint_coerces_sqlite_string_game_date():
    endpoint, weather_type = weather._weather_endpoint(weather._coerce_date("2026-04-03"))

    assert endpoint
    assert weather_type in {"archive", "forecast", "observed"}


def test_target_local_dt_coerces_sqlite_string_timestamp():
    game_date = weather._coerce_date("2026-04-04")
    game_start_ts = weather._coerce_datetime("2026-04-04 18:35:00+00:00")

    target_dt = weather._target_local_dt(game_date, game_start_ts)

    assert target_dt == datetime(2026, 4, 4, 18, 35)


def test_coerce_date_accepts_datetime_values():
    assert weather._coerce_date(datetime(2026, 4, 4, 12, 0, tzinfo=timezone.utc)) == date(2026, 4, 4)