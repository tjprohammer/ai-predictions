import importlib

import pytest


pytest.importorskip("httpx")
from fastapi.testclient import TestClient


app_module = importlib.import_module("src.api.app")
app_logic = importlib.import_module("src.api.app_logic")


def test_season_leaders_endpoint_returns_expected_payload(monkeypatch):
    stub = lambda target_date, limit: {
            "season": 2026,
            "season_start": "2026-01-01",
            "through_date": target_date.isoformat(),
            "pitcher_strikeouts": [{"player_name": "Tarik Skubal", "strikeouts": 41}],
            "hitter_hits": [{"player_name": "Bobby Witt Jr.", "hits": 22}],
            "team_runs": [{"team": "LAD", "runs": 68}],
            "team_strikeouts": [{"team": "COL", "strikeouts": 119}],
        }
    monkeypatch.setattr(app_module, "_fetch_season_leaderboards", stub)
    monkeypatch.setattr(app_logic, "_fetch_season_leaderboards", stub)

    client = TestClient(app_module.app)
    response = client.get("/api/leaders/season", params={"target_date": "2026-04-07", "limit": 8})

    assert response.status_code == 200
    payload = response.json()
    assert payload["target_date"] == "2026-04-07"
    assert payload["season"] == 2026
    assert payload["pitcher_strikeouts"][0]["player_name"] == "Tarik Skubal"
    assert payload["hitter_hits"][0]["player_name"] == "Bobby Witt Jr."
    assert payload["team_runs"][0]["team"] == "LAD"
    assert payload["team_strikeouts"][0]["team"] == "COL"