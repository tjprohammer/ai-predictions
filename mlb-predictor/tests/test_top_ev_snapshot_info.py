"""Board payload exposes when Top EV run snapshots apply (automatic lock window)."""

import src.api.app_logic as app_logic


def test_top_ev_snapshot_info_frozen_run():
    pick = {"top_ev_frozen": True, "top_ev_snapshot_kind": "run"}
    game = {"game_id": 1, "game_start_ts": None, "top_ev_pick": pick}
    info = app_logic._build_top_ev_snapshot_info_for_board(game)
    assert info["state"] == "frozen_run"
    assert info["first_freeze_eligible_after_utc"] is None


def test_top_ev_snapshot_info_live_until_lock_window(monkeypatch):
    monkeypatch.setattr(app_logic, "_effective_top_ev_snapshot_lock_minutes", lambda: 10)
    monkeypatch.setattr(app_logic, "is_before_scheduled_first_pitch", lambda ts: True)
    monkeypatch.setattr(app_logic, "_is_game_top_ev_snapshot_lock_active", lambda g: False)
    monkeypatch.setattr(app_logic.get_settings, "__call__", lambda: None)
    from unittest.mock import MagicMock

    mock_settings = MagicMock()
    mock_settings.board_top_ev_run_snapshot_enabled = True
    monkeypatch.setattr(app_logic, "get_settings", lambda: mock_settings)

    game = {
        "game_id": 1,
        "game_start_ts": "2026-07-01T23:05:00Z",
        "top_ev_pick": {"top_ev_frozen": False, "top_ev_snapshot_kind": None},
    }
    info = app_logic._build_top_ev_snapshot_info_for_board(game)
    assert info["state"] == "live_until_lock_window"


def test_top_ev_snapshot_info_eligible_when_in_window_not_yet_frozen(monkeypatch):
    monkeypatch.setattr(app_logic, "_effective_top_ev_snapshot_lock_minutes", lambda: 10)
    monkeypatch.setattr(app_logic, "is_before_scheduled_first_pitch", lambda ts: True)
    monkeypatch.setattr(app_logic, "_is_game_top_ev_snapshot_lock_active", lambda g: True)
    from unittest.mock import MagicMock

    mock_settings = MagicMock()
    mock_settings.board_top_ev_run_snapshot_enabled = True
    monkeypatch.setattr(app_logic, "get_settings", lambda: mock_settings)

    game = {
        "game_id": 1,
        "game_start_ts": "2026-07-01T23:05:00Z",
        "top_ev_pick": {"top_ev_frozen": False, "top_ev_snapshot_kind": None},
    }
    info = app_logic._build_top_ev_snapshot_info_for_board(game)
    assert info["state"] == "eligible_now"
