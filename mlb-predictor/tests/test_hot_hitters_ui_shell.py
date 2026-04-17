from pathlib import Path
import importlib

import pandas as pd


app_module = importlib.import_module("src.api.app")
app_logic = importlib.import_module("src.api.app_logic")


def _patch_pair(monkeypatch, name: str, value: object) -> None:
    monkeypatch.setattr(app_module, name, value)
    monkeypatch.setattr(app_logic, name, value)


def test_hot_hitters_html_has_streak_sort_and_recent_history_chart():
    html = Path("src/api/static/hot-hitters.html").read_text(encoding="utf-8")

    assert '<option value="streak" selected>Hit streak</option>' in html
    assert '<option value="game">Game / start time</option>' in html
    assert '<option value="handmatchup">Pitcher-hand matchup</option>' in html
    assert 'history-block' in html
    assert 'history-filter' in html
    assert 'function parseDisplayDate(value)' in html
    assert 'history-baseline-label">1</span>' in html
    assert 'hit games' in html
    assert 'data-stat-filter' in html
    assert 'const chartVal = Math.max(val, 0);' in html
    assert 'l15' not in html.lower()
    assert 'function filteredHistory(player, history)' in html
    assert 'function renderRecentHistory(row)' in html
    assert '<input id="hideStartedOnly" type="checkbox" />' in html
    assert 'id="playerSearchInput"' in html
    assert 'api/players/search' in html
    assert '<input id="strongHandOnly" type="checkbox" />' in html
    assert '<input id="availableOnly" type="checkbox" />' in html
    assert 'function hotHitterSearchText(row)' in html
    assert 'function availabilityMeta(row)' in html
    assert 'strongHandOnlyInput.addEventListener("change", renderRows);' in html
    assert 'availableOnlyInput.addEventListener("change", renderRows);' in html
    assert 'streak_only:' in html and "api/hot-hitters" in html
    assert 'id="boardSize"' in html
    assert "applyBoardSizeCap" in html


def test_row_hit_streak_value_matches_ui_rules():
    assert app_logic._row_hit_streak_value({"recent_hit_history": [{"hits": 1}, {"hits": 1}], "streak_len": 0, "streak_len_capped": 0}) == 2
    assert app_logic._row_hit_streak_value({"recent_hit_history": [], "streak_len": 4, "streak_len_capped": 4}) == 4


def test_fetch_recent_hit_history_map_groups_rows_by_player(monkeypatch):
    frame = pd.DataFrame(
        [
            {"player_id": 7, "game_date": "2026-04-03", "game_id": 101, "opponent": "NYY", "hits": 2, "at_bats": 4, "home_runs": 1, "total_bases": 5},
            {"player_id": 7, "game_date": "2026-04-01", "game_id": 99, "opponent": "BOS", "hits": 0, "at_bats": 3, "home_runs": 0, "total_bases": 0},
            {"player_id": 8, "game_date": "2026-04-02", "game_id": 100, "opponent": "SEA", "hits": 1, "at_bats": 5, "home_runs": 0, "total_bases": 1},
        ]
    )

    _patch_pair(monkeypatch, "_table_exists", lambda name: name == "player_game_batting")
    _patch_pair(monkeypatch, "_safe_frame", lambda *_args, **_kwargs: frame)

    result = app_module._fetch_recent_hit_history_map(app_module.date(2026, 4, 4), [7, 8], limit=10)

    assert len(result[7]) == 2
    assert result[7][0]["had_hit"] is True
    assert result[7][0]["home_runs"] == 1
    assert result[7][0]["total_bases"] == 5
    assert result[7][1]["had_hit"] is False
    assert result[8][0]["opponent"] == "SEA"