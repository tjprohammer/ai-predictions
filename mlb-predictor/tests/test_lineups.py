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
    monkeypatch.setattr(lineups_module, "build_lineup_input_frame", fake_build_lineup_input_frame)

    frame = lineups_module._build_lineup_import_frame(csv_path, "2026-04-02", "2026-04-02")

    assert len(frame) == 1
    assert frame.iloc[0]["player_name"] == "Manual Hitter"
    assert len(captured["existing"]) == 1
    assert captured["existing"].iloc[0]["team"] == "NYY"