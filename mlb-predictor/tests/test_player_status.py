from datetime import date
import importlib

import src.ingestors.player_status as player_status_module


app_module = importlib.import_module("src.api.app")


def test_fetch_team_player_status_rows_classifies_active_and_injured(monkeypatch):
    monkeypatch.setattr(
        player_status_module,
        "statsapi_get",
        lambda *_args, **_kwargs: {
            "roster": [
                {
                    "person": {"id": 17, "fullName": "Healthy Hitter"},
                    "position": {"abbreviation": "OF"},
                    "status": {"code": "A", "description": "Active"},
                    "jerseyNumber": "7",
                },
                {
                    "person": {"id": 42, "fullName": "Injured Pitcher"},
                    "position": {"abbreviation": "P"},
                    "status": {"code": "D60", "description": "Injured 60-Day"},
                    "note": "Right shoulder inflammation.",
                    "jerseyNumber": "42",
                },
            ]
        },
    )

    status_rows, player_rows = player_status_module._fetch_team_player_status_rows(
        147,
        "NYY",
        date(2026, 4, 7),
        "2026-04-07T12:00:00+00:00",
    )

    assert len(status_rows) == 2
    assert len(player_rows) == 2
    assert status_rows[0]["is_available"] is True
    assert status_rows[0]["availability_bucket"] == "active"
    assert status_rows[1]["is_injured"] is True
    assert status_rows[1]["is_available"] is False
    assert status_rows[1]["status_note"] == "Right shoulder inflammation."
    assert player_rows[1]["active"] is False


def test_attach_player_status_context_merges_roster_fields():
    player = {"player_id": 42, "team": "NYY", "player_name": "Injured Pitcher"}

    result = app_module._attach_player_status_context(
        player,
        {
            (42, "NYY"): {
                "availability_bucket": "injured",
                "is_active_roster": False,
                "is_available": False,
                "is_injured": True,
                "roster_status_code": "D60",
                "roster_status_description": "Injured 60-Day",
                "roster_status_note": "Right shoulder inflammation.",
                "roster_snapshot_ts": "2026-04-07T12:00:00+00:00",
            }
        },
    )

    assert result["availability_bucket"] == "injured"
    assert result["is_available"] is False
    assert result["is_injured"] is True
    assert result["roster_status_description"] == "Injured 60-Day"