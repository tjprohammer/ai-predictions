from pathlib import Path
import importlib

import pandas as pd


app_module = importlib.import_module("src.api.app")


def test_hot_hitters_html_has_team_sort_and_recent_history_strip():
    html = Path("src/api/static/hot-hitters.html").read_text(encoding="utf-8")

    assert '<option value="team" selected>Team</option>' in html
    assert 'Last 10 Games' in html
    assert 'Green = hit · red = no hit' in html
    assert 'history-tile' in html
    assert 'function renderRecentHistory(row)' in html
    assert '<input id="hideStartedOnly" type="checkbox" />' in html


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