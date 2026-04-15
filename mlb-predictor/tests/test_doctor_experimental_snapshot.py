"""Doctor payload: experimental (NRFI/YRFI) market snapshot."""

from datetime import date

from src.api import app_logic


def test_experimental_snapshot_when_game_markets_missing(monkeypatch):
    monkeypatch.setattr(app_logic, "_table_exists", lambda _name: False)
    snap = app_logic._doctor_experimental_markets_snapshot(date(2026, 4, 12))
    assert snap["table_present"] is False
    assert snap["total_market_rows"] == 0
