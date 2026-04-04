from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

import pandas as pd

from src.ingestors import pitcher_statcast


class _FakeConnection:
    def __init__(self, captured: dict[str, object]) -> None:
        self.captured = captured

    def execute(self, statement, rows) -> None:
        self.captured["statement"] = str(statement)
        self.captured["rows"] = rows


class _FakeBegin:
    def __init__(self, captured: dict[str, object]) -> None:
        self.captured = captured

    def __enter__(self):
        return _FakeConnection(self.captured)

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeEngine:
    def __init__(self, captured: dict[str, object]) -> None:
        self.captured = captured

    def begin(self):
        return _FakeBegin(self.captured)


def test_pitcher_statcast_main_coerces_sqlite_string_game_date(monkeypatch):
    monkeypatch.setattr(pitcher_statcast.argparse.ArgumentParser, "parse_args", lambda self: SimpleNamespace())
    monkeypatch.setattr(pitcher_statcast, "resolve_date_range", lambda _args: (date(2026, 4, 3), date(2026, 4, 3)))
    monkeypatch.setattr(
        pitcher_statcast,
        "query_df",
        lambda *_args, **_kwargs: pd.DataFrame([{"game_id": 1, "game_date": "2026-04-03", "pitcher_id": 77}]),
    )

    statcast_calls: list[tuple[str, str, int]] = []

    def fake_statcast_pitcher(*, start_dt, end_dt, player_id):
        statcast_calls.append((start_dt, end_dt, player_id))
        return pd.DataFrame(
            [
                {
                    "description": "called_strike",
                    "estimated_woba_using_speedangle": 0.29,
                    "estimated_slg_using_speedangle": 0.41,
                    "pitch_type": "FF",
                    "release_speed": 96.2,
                    "launch_speed": 98.0,
                    "launch_speed_angle": 6,
                }
            ]
        )

    monkeypatch.setattr(pitcher_statcast, "statcast_pitcher", fake_statcast_pitcher)
    captured: dict[str, object] = {}
    monkeypatch.setattr(pitcher_statcast, "get_engine", lambda: _FakeEngine(captured))

    assert pitcher_statcast.main() == 0
    assert statcast_calls == [("2026-04-03", "2026-04-03", 77)]
    assert "updated_at = :updated_at" in captured["statement"]
    assert "now()" not in captured["statement"]
    assert isinstance(captured["rows"][0]["updated_at"], datetime)