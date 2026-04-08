import importlib


boxscores_module = importlib.import_module("src.ingestors.boxscores")


def test_late_bullpen_stats_by_pitcher_tracks_innings_seven_plus_relief_damage():
    feed = {
        "liveData": {
            "plays": {
                "allPlays": [
                    {
                        "about": {"inning": 6, "halfInning": "top"},
                        "result": {"homeScore": 0, "awayScore": 0, "eventType": "strikeout", "isOut": True},
                        "matchup": {"pitcher": {"id": 100}},
                        "runners": [
                            {"movement": {"isOut": True}, "details": {"isScoringEvent": False, "earned": False}},
                        ],
                    },
                    {
                        "about": {"inning": 7, "halfInning": "top"},
                        "result": {"homeScore": 0, "awayScore": 0, "eventType": "strikeout", "isOut": True},
                        "matchup": {"pitcher": {"id": 900}},
                        "runners": [
                            {"movement": {"isOut": True}, "details": {"isScoringEvent": False, "earned": False}},
                        ],
                    },
                    {
                        "about": {"inning": 7, "halfInning": "top"},
                        "result": {"homeScore": 0, "awayScore": 1, "eventType": "home_run", "isOut": False},
                        "matchup": {"pitcher": {"id": 900}},
                        "runners": [
                            {"movement": {"isOut": False}, "details": {"isScoringEvent": True, "earned": True}},
                        ],
                    },
                ]
            }
        }
    }

    result = boxscores_module._late_bullpen_stats_by_pitcher(
        feed,
        home_team="DET",
        away_team="CLE",
        starter_ids_by_team={"DET": 100, "CLE": 200},
    )

    assert result == {
        900: {
            "late_innings_pitched": 0.1,
            "late_runs_allowed": 1,
            "late_earned_runs": 1,
            "late_hits_allowed": 1,
        }
    }