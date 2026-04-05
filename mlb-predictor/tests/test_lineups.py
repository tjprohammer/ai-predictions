import pandas as pd

import src.ingestors.lineups as lineups_module


def test_build_lineup_import_frame_uses_projected_lineups_when_csv_missing(monkeypatch, tmp_path):
    games = pd.DataFrame(
        [{"game_id": 1, "game_date": pd.Timestamp("2026-04-02").date(), "home_team": "NYY", "away_team": "BOS"}]
    )
    captured = {}

    def fake_load_games(start_date, end_date):
        captured["range"] = (start_date, end_date)
        return games.copy()

    def fake_build_lineup_input_frame(frame_games, existing, snapshot_ts):
        captured["games"] = frame_games.copy()
        captured["existing"] = existing.copy()
        captured["snapshot_ts"] = snapshot_ts
        return pd.DataFrame(
            [
                {
                    "game_id": 1,
                    "game_date": "2026-04-02",
                    "team": "NYY",
                    "player_id": 99,
                    "player_name": "Projected Hitter",
                    "lineup_slot": 1,
                    "position": None,
                    "confirmed": False,
                    "source_name": "projected_template",
                    "source_url": None,
                    "snapshot_ts": snapshot_ts,
                }
            ]
        )

    monkeypatch.setattr(lineups_module, "_load_games_for_range", fake_load_games)
    monkeypatch.setattr(lineups_module, "_fetch_statsapi_lineup_frame", lambda *_args, **_kwargs: pd.DataFrame(columns=lineups_module.LINEUP_COLUMNS))
    monkeypatch.setattr(lineups_module, "build_lineup_input_frame", fake_build_lineup_input_frame)

    frame = lineups_module._build_lineup_import_frame(tmp_path / "missing.csv", "2026-04-02", "2026-04-02")

    assert len(frame) == 1
    assert captured["range"] == ("2026-04-02", "2026-04-02")
    assert captured["existing"].empty
    assert list(captured["existing"].columns) == lineups_module.LINEUP_COLUMNS


def test_build_lineup_import_frame_keeps_csv_rows_and_fills_missing_teams(monkeypatch, tmp_path):
    csv_path = tmp_path / "lineups.csv"
    pd.DataFrame(
        [
            {
                "game_id": 1,
                "game_date": "2026-04-02",
                "team": "NYY",
                "player_id": 7,
                "player_name": "Manual Hitter",
                "lineup_slot": 1,
                "position": "CF",
                "confirmed": True,
                "source_name": "manual_edit",
                "source_url": None,
                "snapshot_ts": "2026-04-02T16:00:00+00:00",
            },
            {
                "game_id": 2,
                "game_date": "2026-04-01",
                "team": "LAD",
                "player_id": 8,
                "player_name": "Old Row",
                "lineup_slot": 1,
                "position": "SS",
                "confirmed": True,
                "source_name": "manual_edit",
                "source_url": None,
                "snapshot_ts": "2026-04-01T16:00:00+00:00",
            },
        ]
    ).to_csv(csv_path, index=False)

    games = pd.DataFrame(
        [{"game_id": 1, "game_date": pd.Timestamp("2026-04-02").date(), "home_team": "NYY", "away_team": "BOS"}]
    )
    captured = {}

    def fake_load_games(start_date, end_date):
        return games.copy()

    def fake_build_lineup_input_frame(frame_games, existing, snapshot_ts):
        captured["existing"] = existing.copy()
        return existing.copy()

    monkeypatch.setattr(lineups_module, "_load_games_for_range", fake_load_games)
    monkeypatch.setattr(lineups_module, "_fetch_statsapi_lineup_frame", lambda *_args, **_kwargs: pd.DataFrame(columns=lineups_module.LINEUP_COLUMNS))
    monkeypatch.setattr(lineups_module, "build_lineup_input_frame", fake_build_lineup_input_frame)

    frame = lineups_module._build_lineup_import_frame(csv_path, "2026-04-02", "2026-04-02")

    assert len(frame) == 1
    assert frame.iloc[0]["player_name"] == "Manual Hitter"
    assert len(captured["existing"]) == 1
    assert captured["existing"].iloc[0]["team"] == "NYY"


def test_fetch_statsapi_lineup_frame_prefers_earliest_batting_order_per_slot(monkeypatch):
    games = pd.DataFrame(
        [{"game_id": 77, "game_date": pd.Timestamp("2026-04-02").date(), "home_team": "NYY", "away_team": "BOS"}]
    )

    def fake_statsapi_get(_path):
        away_players = {
            f"ID{i}": {
                "person": {"id": i, "fullName": f"Away {i}"},
                "battingOrder": f"{slot}00",
                "position": {"abbreviation": "OF"},
            }
            for i, slot in enumerate(range(1, 10), start=1)
        }
        home_players = {
            f"ID{100 + i}": {
                "person": {"id": 100 + i, "fullName": f"Home {i}"},
                "battingOrder": f"{slot}00",
                "position": {"abbreviation": "IF"},
            }
            for i, slot in enumerate(range(1, 10), start=1)
        }
        home_players["ID199"] = {
            "person": {"id": 199, "fullName": "Home Replacement"},
            "battingOrder": "101",
            "position": {"abbreviation": "DH"},
        }
        return {
            "liveData": {
                "boxscore": {
                    "teams": {
                        "away": {"players": away_players},
                        "home": {"players": home_players},
                    }
                }
            }
        }

    monkeypatch.setattr(lineups_module, "statsapi_get", fake_statsapi_get)

    frame = lineups_module._fetch_statsapi_lineup_frame(games, "2026-04-02T12:00:00+00:00")

    assert len(frame) == 18
    assert frame[(frame["team"] == "NYY") & (frame["lineup_slot"] == 1)].iloc[0]["player_name"] == "Home 1"
    assert frame[(frame["team"] == "NYY") & (frame["lineup_slot"] == 1)].iloc[0]["source_name"] == "mlb_statsapi_lineups"
    assert bool(frame[(frame["team"] == "NYY") & (frame["lineup_slot"] == 1)].iloc[0]["confirmed"]) is True


def test_build_lineup_import_frame_prefers_statsapi_rows_over_manual_same_team(monkeypatch, tmp_path):
    csv_path = tmp_path / "lineups.csv"
    pd.DataFrame(
        [
            {
                "game_id": 1,
                "game_date": "2026-04-02",
                "team": "NYY",
                "player_id": 7,
                "player_name": "Manual Yankee",
                "lineup_slot": 1,
                "position": "CF",
                "confirmed": True,
                "source_name": "manual_edit",
                "source_url": None,
                "snapshot_ts": "2026-04-02T16:00:00+00:00",
            },
            {
                "game_id": 1,
                "game_date": "2026-04-02",
                "team": "BOS",
                "player_id": 8,
                "player_name": "Manual Red Sox",
                "lineup_slot": 1,
                "position": "SS",
                "confirmed": True,
                "source_name": "manual_edit",
                "source_url": None,
                "snapshot_ts": "2026-04-02T16:00:00+00:00",
            },
        ]
    ).to_csv(csv_path, index=False)

    games = pd.DataFrame(
        [{"game_id": 1, "game_date": pd.Timestamp("2026-04-02").date(), "home_team": "NYY", "away_team": "BOS"}]
    )
    statsapi_rows = pd.DataFrame(
        [
            {
                "game_id": 1,
                "game_date": pd.Timestamp("2026-04-02").date(),
                "team": "NYY",
                "player_id": 17,
                "player_name": "StatsAPI Yankee",
                "lineup_slot": 1,
                "position": "LF",
                "confirmed": True,
                "source_name": "mlb_statsapi_lineups",
                "source_url": "https://statsapi.mlb.com/api/v1.1/game/1/feed/live",
                "snapshot_ts": "2026-04-02T17:00:00+00:00",
            }
        ],
        columns=lineups_module.LINEUP_COLUMNS,
    )
    captured = {}

    monkeypatch.setattr(lineups_module, "_load_games_for_range", lambda *_args, **_kwargs: games.copy())
    monkeypatch.setattr(lineups_module, "_fetch_statsapi_lineup_frame", lambda *_args, **_kwargs: statsapi_rows.copy())

    def fake_build_lineup_input_frame(frame_games, existing, snapshot_ts):
        captured["existing"] = existing.copy()
        return existing.copy()

    monkeypatch.setattr(lineups_module, "build_lineup_input_frame", fake_build_lineup_input_frame)

    frame = lineups_module._build_lineup_import_frame(csv_path, "2026-04-02", "2026-04-02")

    assert len(frame) == 2
    assert set(frame["team"]) == {"NYY", "BOS"}
    assert frame[frame["team"] == "NYY"].iloc[0]["player_name"] == "StatsAPI Yankee"
    assert frame[frame["team"] == "BOS"].iloc[0]["player_name"] == "Manual Red Sox"
    assert frame[frame["team"] == "NYY"].iloc[0]["source_name"] == "mlb_statsapi_lineups"
    assert len(captured["existing"]) == 2