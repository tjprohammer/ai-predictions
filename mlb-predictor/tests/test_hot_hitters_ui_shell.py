from pathlib import Path
import importlib

import pandas as pd


app_module = importlib.import_module("src.api.app")


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
    assert '<label for="searchQuery">Search player, team, or game</label>' in html
    assert '<input id="strongHandOnly" type="checkbox" />' in html
    assert '<input id="availableOnly" type="checkbox" />' in html
    assert 'function hotHitterSearchText(row)' in html
    assert 'function availabilityMeta(row)' in html
    assert 'searchQueryInput.addEventListener("input", renderRows);' in html
    assert 'strongHandOnlyInput.addEventListener("change", renderRows);' in html
    assert 'availableOnlyInput.addEventListener("change", renderRows);' in html


def test_fetch_recent_hit_history_map_groups_rows_by_player(monkeypatch):
    frame = pd.DataFrame(
        [
            {"player_id": 7, "game_date": "2026-04-03", "game_id": 101, "opponent": "NYY", "hits": 2, "at_bats": 4, "home_runs": 1, "total_bases": 5},
            {"player_id": 7, "game_date": "2026-04-01", "game_id": 99, "opponent": "BOS", "hits": 0, "at_bats": 3, "home_runs": 0, "total_bases": 0},
            {"player_id": 8, "game_date": "2026-04-02", "game_id": 100, "opponent": "SEA", "hits": 1, "at_bats": 5, "home_runs": 0, "total_bases": 1},
        ]
    )

    monkeypatch.setattr(app_module, "_table_exists", lambda name: name == "player_game_batting")
    monkeypatch.setattr(app_module, "_safe_frame", lambda *_args, **_kwargs: frame)

    result = app_module._fetch_recent_hit_history_map(app_module.date(2026, 4, 4), [7, 8], limit=10)

    assert len(result[7]) == 2
    assert result[7][0]["had_hit"] is True
    assert result[7][0]["home_runs"] == 1
    assert result[7][0]["total_bases"] == 5
    assert result[7][1]["had_hit"] is False
    assert result[8][0]["opponent"] == "SEA"