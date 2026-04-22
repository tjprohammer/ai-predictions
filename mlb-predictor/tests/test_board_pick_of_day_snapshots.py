"""Pick-of-the-day run snapshots: stable short-list cards across board refresh."""

from datetime import date, datetime, timedelta, timezone

import pytest

import src.api.app_logic as app_logic


@pytest.fixture
def target_day() -> date:
    return date(2026, 7, 15)


def test_resolve_prefers_frozen_pick_over_live(target_day: date, monkeypatch) -> None:
    monkeypatch.setattr(app_logic, "table_exists", lambda name: name == "board_pick_of_day_run_snapshots")
    frozen_card = {
        "game_id": 7,
        "market_key": "run_line",
        "selection_label": "FROZEN AWAY +1.5",
        "weighted_ev": 0.05,
        "probability_edge": 0.02,
        "model_probability": 0.55,
        "positive": True,
        "pick_of_day_rank": 1,
    }
    monkeypatch.setattr(
        app_logic,
        "_fetch_board_pick_of_day_run_snapshots_map",
        lambda d: {7: frozen_card},
    )
    monkeypatch.setattr(app_logic, "_maybe_insert_board_pick_of_day_run_snapshots", lambda *a, **k: None)

    live_card = {
        "game_id": 7,
        "market_key": "run_line",
        "selection_label": "LIVE DIFFERENT",
        "weighted_ev": 0.99,
        "probability_edge": 0.5,
        "model_probability": 0.9,
        "positive": True,
        "pick_of_day_rank": 1,
    }
    board_row = {
        "game_id": 7,
        "game_start_ts": datetime.now(timezone.utc) + timedelta(hours=4),
        "away_team": "A",
        "home_team": "H",
        "best_bets": [],
        "market_cards": [],
    }

    monkeypatch.setattr(
        app_logic.best_bets_utils,
        "select_picks_of_the_day",
        lambda rows, for_live_slate_date=None, slate_game_date=None: [dict(live_card)],
    )

    out = app_logic._resolve_picks_of_the_day_for_board(
        target_day, [board_row], for_live_slate_date=None
    )
    assert len(out) == 1
    assert out[0]["selection_label"] == "FROZEN AWAY +1.5"
    assert out[0].get("pick_of_day_frozen") is True


def test_maybe_insert_skips_without_eager_outside_lock(target_day: date, monkeypatch) -> None:
    from src.utils.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("BOARD_PICK_OF_DAY_RUN_SNAPSHOT_ENABLED", "true")
    monkeypatch.setenv("BOARD_PICK_OF_DAY_RUN_SNAPSHOT_EAGER", "false")
    get_settings.cache_clear()
    monkeypatch.setattr(app_logic, "table_exists", lambda name: name == "board_pick_of_day_run_snapshots")
    monkeypatch.setattr(app_logic, "_fetch_board_pick_of_day_run_snapshots_map", lambda d: {})
    inserted: list[int] = []

    def capture(td: date, gid: int, payload: dict) -> None:
        inserted.append(int(gid))

    far = datetime.now(timezone.utc) + timedelta(hours=5)
    board_row = {"game_id": 44, "game_start_ts": far, "away_team": "X", "home_team": "Y"}
    card = {
        "game_id": 44,
        "market_key": "moneyline",
        "positive": True,
        "pick_of_day_rank": 1,
    }
    monkeypatch.setattr(app_logic, "_insert_board_pick_of_day_run_snapshot_row", capture)
    monkeypatch.setattr(app_logic, "_is_game_top_ev_snapshot_lock_active", lambda row: False)

    class _FakeDateModule:
        today = staticmethod(lambda: target_day)
        fromisoformat = staticmethod(date.fromisoformat)

    monkeypatch.setattr(app_logic, "date", _FakeDateModule)
    app_logic._maybe_insert_board_pick_of_day_run_snapshots(
        target_day,
        [board_row],
        [card],
        for_live_slate_date=None,
        force=False,
    )
    assert inserted == []

    app_logic._maybe_insert_board_pick_of_day_run_snapshots(
        target_day,
        [board_row],
        [card],
        for_live_slate_date=None,
        force=True,
    )
    assert inserted == [44]

    get_settings.cache_clear()
    monkeypatch.delenv("BOARD_PICK_OF_DAY_RUN_SNAPSHOT_ENABLED", raising=False)
    monkeypatch.delenv("BOARD_PICK_OF_DAY_RUN_SNAPSHOT_EAGER", raising=False)
    get_settings.cache_clear()


def test_merge_pod_retains_all_frozen_then_fills_live_slots(target_day: date, monkeypatch) -> None:
    """Frozen early-slate PoD rows are never dropped; remaining cap fills with live games not frozen."""
    from src.utils.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("BOARD_PICK_OF_DAY_RUN_SNAPSHOT_ENABLED", "true")
    monkeypatch.setenv("PICK_OF_THE_DAY_FULL_SLATE_PER_GAME", "false")
    monkeypatch.setenv("PICK_OF_THE_DAY_MAX", "3")
    get_settings.cache_clear()
    monkeypatch.setattr(app_logic, "table_exists", lambda name: name == "board_pick_of_day_run_snapshots")
    frozen = {
        1: {
            "game_id": 1,
            "pick_of_day_rank": 1,
            "selection_label": "FROZEN1",
            "market_key": "moneyline",
            "weighted_ev": 0.01,
            "probability_edge": 0.01,
            "model_probability": 0.9,
            "positive": True,
        },
        2: {
            "game_id": 2,
            "pick_of_day_rank": 2,
            "selection_label": "FROZEN2",
            "market_key": "moneyline",
            "weighted_ev": 0.02,
            "probability_edge": 0.02,
            "model_probability": 0.9,
            "positive": True,
        },
    }
    monkeypatch.setattr(app_logic, "_fetch_board_pick_of_day_run_snapshots_map", lambda d: frozen)
    rows = [
        {"game_id": 1, "away_team": "A1", "home_team": "H1"},
        {"game_id": 2, "away_team": "A2", "home_team": "H2"},
        {"game_id": 3, "away_team": "A3", "home_team": "H3"},
        {"game_id": 4, "away_team": "A4", "home_team": "H4"},
    ]
    live = [
        {
            "game_id": 3,
            "selection_label": "LIVE_HIGH",
            "market_key": "moneyline",
            "weighted_ev": 0.99,
            "probability_edge": 0.5,
            "model_probability": 0.95,
            "positive": True,
        },
        {
            "game_id": 4,
            "selection_label": "LIVE_NEXT",
            "market_key": "moneyline",
            "weighted_ev": 0.5,
            "probability_edge": 0.2,
            "model_probability": 0.9,
            "positive": True,
        },
    ]
    out = app_logic._merge_board_pick_of_day_run_snapshots_into_live(target_day, rows, live)
    assert len(out) == 3
    assert [int(x["game_id"]) for x in out] == [1, 2, 3]
    assert out[0].get("pick_of_day_frozen") is True
    assert out[2].get("pick_of_day_frozen") is not True

    frozen3 = {
        **frozen,
        5: {
            "game_id": 5,
            "pick_of_day_rank": 3,
            "selection_label": "FROZEN3",
            "market_key": "moneyline",
            "weighted_ev": 0.03,
            "probability_edge": 0.03,
            "model_probability": 0.9,
            "positive": True,
        },
    }
    monkeypatch.setattr(app_logic, "_fetch_board_pick_of_day_run_snapshots_map", lambda d: frozen3)
    rows5 = rows + [{"game_id": 5, "away_team": "A5", "home_team": "H5"}]
    monkeypatch.setenv("PICK_OF_THE_DAY_MAX", "2")
    get_settings.cache_clear()
    out2 = app_logic._merge_board_pick_of_day_run_snapshots_into_live(target_day, rows5, live)
    assert len(out2) == 3
    assert {int(x["game_id"]) for x in out2} == {1, 2, 5}
    assert all(x.get("pick_of_day_frozen") for x in out2)

    get_settings.cache_clear()
    monkeypatch.delenv("PICK_OF_THE_DAY_MAX", raising=False)
    monkeypatch.delenv("PICK_OF_THE_DAY_FULL_SLATE_PER_GAME", raising=False)
    monkeypatch.delenv("BOARD_PICK_OF_DAY_RUN_SNAPSHOT_ENABLED", raising=False)
    get_settings.cache_clear()


def test_merge_pod_full_slate_uncapped_with_run_snapshots(target_day: date, monkeypatch) -> None:
    """Full-slate PoD: every game stays visible; run snapshots must not truncate remaining live games."""
    from src.utils.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("BOARD_PICK_OF_DAY_RUN_SNAPSHOT_ENABLED", "true")
    monkeypatch.setenv("PICK_OF_THE_DAY_FULL_SLATE_PER_GAME", "true")
    monkeypatch.setenv("PICK_OF_THE_DAY_MAX", "3")
    get_settings.cache_clear()
    monkeypatch.setattr(app_logic, "table_exists", lambda name: name == "board_pick_of_day_run_snapshots")
    frozen = {
        1: {
            "game_id": 1,
            "pick_of_day_rank": 1,
            "selection_label": "FROZEN1",
            "market_key": "moneyline",
            "weighted_ev": 0.01,
            "probability_edge": 0.01,
            "model_probability": 0.9,
            "positive": True,
        },
        2: {
            "game_id": 2,
            "pick_of_day_rank": 2,
            "selection_label": "FROZEN2",
            "market_key": "moneyline",
            "weighted_ev": 0.02,
            "probability_edge": 0.02,
            "model_probability": 0.9,
            "positive": True,
        },
    }
    monkeypatch.setattr(app_logic, "_fetch_board_pick_of_day_run_snapshots_map", lambda d: frozen)
    rows = [
        {"game_id": 1, "away_team": "A1", "home_team": "H1"},
        {"game_id": 2, "away_team": "A2", "home_team": "H2"},
        {"game_id": 3, "away_team": "A3", "home_team": "H3"},
        {"game_id": 4, "away_team": "A4", "home_team": "H4"},
    ]
    live = [
        {
            "game_id": 3,
            "selection_label": "LIVE_HIGH",
            "market_key": "moneyline",
            "weighted_ev": 0.99,
            "probability_edge": 0.5,
            "model_probability": 0.95,
            "positive": True,
        },
        {
            "game_id": 4,
            "selection_label": "LIVE_NEXT",
            "market_key": "moneyline",
            "weighted_ev": 0.5,
            "probability_edge": 0.2,
            "model_probability": 0.9,
            "positive": True,
        },
    ]
    out = app_logic._merge_board_pick_of_day_run_snapshots_into_live(target_day, rows, live)
    assert len(out) == 4
    assert {int(x["game_id"]) for x in out} == {1, 2, 3, 4}
    frozen_rank1 = next(x for x in out if int(x["game_id"]) == 1)
    assert frozen_rank1.get("pick_of_day_frozen") is True
    assert int(frozen_rank1["pick_of_day_rank"]) != 1

    get_settings.cache_clear()
    monkeypatch.delenv("PICK_OF_THE_DAY_MAX", raising=False)
    monkeypatch.delenv("PICK_OF_THE_DAY_FULL_SLATE_PER_GAME", raising=False)
    monkeypatch.delenv("BOARD_PICK_OF_DAY_RUN_SNAPSHOT_ENABLED", raising=False)
    get_settings.cache_clear()


def test_merge_pod_skips_excluded_frozen_payload(target_day: date, monkeypatch) -> None:
    """F5 team-total run snapshots are ignored by default so stale rows do not block live PoD."""
    from src.utils.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("BOARD_PICK_OF_DAY_RUN_SNAPSHOT_ENABLED", "true")
    get_settings.cache_clear()
    monkeypatch.setattr(app_logic, "table_exists", lambda name: name == "board_pick_of_day_run_snapshots")
    bad_frozen = {
        "game_id": 7,
        "pick_of_day_rank": 1,
        "market_key": "first_five_team_total_away",
        "bet_side": "over",
        "selection_label": "BAD F5",
        "weighted_ev": 0.99,
        "probability_edge": 0.5,
        "model_probability": 0.9,
        "positive": True,
    }
    monkeypatch.setattr(app_logic, "_fetch_board_pick_of_day_run_snapshots_map", lambda d: {7: bad_frozen})
    rows = [{"game_id": 7, "away_team": "A", "home_team": "H"}]
    live = [
        {
            "game_id": 7,
            "market_key": "run_line",
            "selection_label": "LIVE RL",
            "weighted_ev": 0.5,
            "probability_edge": 0.2,
            "model_probability": 0.6,
            "positive": True,
        },
    ]
    out = app_logic._merge_board_pick_of_day_run_snapshots_into_live(target_day, rows, live)
    assert len(out) == 1
    assert out[0]["selection_label"] == "LIVE RL"
    assert out[0].get("pick_of_day_frozen") is not True

    monkeypatch.setenv("BOARD_SNAPSHOT_EXCLUDE_F5_TEAM_TOTALS", "false")
    out2 = app_logic._merge_board_pick_of_day_run_snapshots_into_live(target_day, rows, live)
    assert len(out2) == 1
    assert out2[0]["selection_label"] == "BAD F5"
    assert out2[0].get("pick_of_day_frozen") is True

    monkeypatch.delenv("BOARD_SNAPSHOT_EXCLUDE_F5_TEAM_TOTALS", raising=False)
    get_settings.cache_clear()


def test_maybe_insert_pod_skips_excluded_cards(target_day: date, monkeypatch) -> None:
    from src.utils.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("BOARD_PICK_OF_DAY_RUN_SNAPSHOT_ENABLED", "true")
    get_settings.cache_clear()
    monkeypatch.setattr(app_logic, "table_exists", lambda name: name == "board_pick_of_day_run_snapshots")
    monkeypatch.setattr(app_logic, "_fetch_board_pick_of_day_run_snapshots_map", lambda d: {})
    inserted: list[int] = []

    def capture(td: date, gid: int, payload: dict) -> None:
        inserted.append(int(gid))

    far = datetime.now(timezone.utc) + timedelta(hours=5)
    board_row = {"game_id": 55, "game_start_ts": far, "away_team": "X", "home_team": "Y"}
    f5_card = {
        "game_id": 55,
        "market_key": "first_five_team_total_away",
        "bet_side": "over",
        "positive": True,
        "pick_of_day_rank": 1,
    }
    monkeypatch.setattr(app_logic, "_insert_board_pick_of_day_run_snapshot_row", capture)
    monkeypatch.setattr(app_logic, "_is_game_top_ev_snapshot_lock_active", lambda row: False)

    class _FakeDateModule:
        today = staticmethod(lambda: target_day)
        fromisoformat = staticmethod(date.fromisoformat)

    monkeypatch.setattr(app_logic, "date", _FakeDateModule)
    app_logic._maybe_insert_board_pick_of_day_run_snapshots(
        target_day,
        [board_row],
        [f5_card],
        for_live_slate_date=None,
        force=True,
    )
    assert inserted == []

    monkeypatch.setenv("BOARD_SNAPSHOT_EXCLUDE_F5_TEAM_TOTALS", "false")
    app_logic._maybe_insert_board_pick_of_day_run_snapshots(
        target_day,
        [board_row],
        [f5_card],
        for_live_slate_date=None,
        force=True,
    )
    assert inserted == [55]

    inserted.clear()
    monkeypatch.setenv("BOARD_BATTER_TB_OVERS_ONLY", "true")
    tb_card = {
        "game_id": 55,
        "market_key": "batter_total_bases",
        "bet_side": "under",
        "positive": True,
        "pick_of_day_rank": 1,
    }
    app_logic._maybe_insert_board_pick_of_day_run_snapshots(
        target_day,
        [board_row],
        [tb_card],
        for_live_slate_date=None,
        force=True,
    )
    assert inserted == []

    monkeypatch.delenv("BOARD_PICK_OF_DAY_RUN_SNAPSHOT_ENABLED", raising=False)
    monkeypatch.delenv("BOARD_SNAPSHOT_EXCLUDE_F5_TEAM_TOTALS", raising=False)
    monkeypatch.delenv("BOARD_BATTER_TB_OVERS_ONLY", raising=False)
    get_settings.cache_clear()


def test_maybe_insert_pod_upgrades_snapshot_when_live_pick_strictly_better(
    target_day: date, monkeypatch
) -> None:
    """Lineup refresh can replace a stored PoD snapshot only when the new card ranks higher (sort key)."""
    from src.utils.settings import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("BOARD_PICK_OF_DAY_RUN_SNAPSHOT_ENABLED", "true")
    monkeypatch.setenv("BOARD_PICK_OF_DAY_RUN_SNAPSHOT_EAGER", "true")
    get_settings.cache_clear()
    monkeypatch.setattr(app_logic, "table_exists", lambda name: name == "board_pick_of_day_run_snapshots")

    low_snap = {
        "game_id": 44,
        "market_key": "moneyline",
        "selection_label": "LOW",
        "weighted_ev": 0.01,
        "probability_edge": 0.005,
        "model_probability": 0.56,
        "positive": True,
        "certainty_weight": 0.9,
        "game_certainty_pct": 90.0,
        "input_trust": {"grade": "A", "score": 0.82},
    }
    monkeypatch.setattr(
        app_logic,
        "_fetch_board_pick_of_day_run_snapshots_map",
        lambda d: {44: dict(low_snap)},
    )
    captured: list[dict] = []

    def capture(td: date, gid: int, payload: dict) -> None:
        captured.append(dict(payload))

    far = datetime.now(timezone.utc) + timedelta(hours=5)
    board_row = {"game_id": 44, "game_start_ts": far, "away_team": "X", "home_team": "Y"}
    high_card = {
        **low_snap,
        "weighted_ev": 0.5,
        "probability_edge": 0.25,
        "selection_label": "HIGH",
        "pick_of_day_rank": 1,
    }
    monkeypatch.setattr(app_logic, "_insert_board_pick_of_day_run_snapshot_row", capture)
    monkeypatch.setattr(app_logic, "_is_game_top_ev_snapshot_lock_active", lambda row: False)

    class _FakeDateModule:
        today = staticmethod(lambda: target_day)
        fromisoformat = staticmethod(date.fromisoformat)

    monkeypatch.setattr(app_logic, "date", _FakeDateModule)
    app_logic._maybe_insert_board_pick_of_day_run_snapshots(
        target_day,
        [board_row],
        [high_card],
        for_live_slate_date=None,
        force=False,
    )
    assert len(captured) == 1
    assert captured[0]["selection_label"] == "HIGH"

    captured.clear()
    worse_card = {
        **low_snap,
        "weighted_ev": 0.001,
        "probability_edge": 0.0005,
        "selection_label": "WORSE",
        "pick_of_day_rank": 1,
    }
    app_logic._maybe_insert_board_pick_of_day_run_snapshots(
        target_day,
        [board_row],
        [worse_card],
        for_live_slate_date=None,
        force=False,
    )
    assert captured == []

    get_settings.cache_clear()
    monkeypatch.delenv("BOARD_PICK_OF_DAY_RUN_SNAPSHOT_ENABLED", raising=False)
    monkeypatch.delenv("BOARD_PICK_OF_DAY_RUN_SNAPSHOT_EAGER", raising=False)
    get_settings.cache_clear()
