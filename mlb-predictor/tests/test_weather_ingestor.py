from datetime import date, datetime
import importlib


weather_module = importlib.import_module("src.ingestors.weather")


def test_target_local_dt_converts_utc_first_pitch_to_venue_local_time():
    first_pitch_utc = datetime.fromisoformat("2026-04-07T22:45:00+00:00")

    local_time = weather_module._target_local_dt(
        date(2026, 4, 7),
        first_pitch_utc,
        timezone_name="America/New_York",
    )

    assert local_time == datetime(2026, 4, 7, 18, 45)