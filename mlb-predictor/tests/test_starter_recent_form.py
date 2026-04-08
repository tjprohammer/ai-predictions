from datetime import date
import importlib

import pandas as pd
import pytest


app_module = importlib.import_module("src.api.app")


def test_fetch_starter_recent_form_reports_season_and_recent_era(monkeypatch):
    frame = pd.DataFrame(
        [
            {
                "game_date": "2026-04-05",
                "ip": 6.0,
                "earned_runs": 1,
                "strikeouts": 8,
                "walks": 2,
                "pitch_count": 95,
                "xwoba_against": 0.281,
                "csw_pct": 0.29,
                "whiff_pct": 0.24,
                "avg_fb_velo": 95.1,
            },
            {
                "game_date": "2026-03-30",
                "ip": 5.0,
                "earned_runs": 2,
                "strikeouts": 6,
                "walks": 1,
                "pitch_count": 87,
                "xwoba_against": 0.301,
                "csw_pct": 0.27,
                "whiff_pct": 0.22,
                "avg_fb_velo": 94.8,
            },
            {
                "game_date": "2026-03-24",
                "ip": 7.0,
                "earned_runs": 3,
                "strikeouts": 9,
                "walks": 2,
                "pitch_count": 101,
                "xwoba_against": 0.318,
                "csw_pct": 0.31,
                "whiff_pct": 0.26,
                "avg_fb_velo": 95.4,
            },
            {
                "game_date": "2026-03-18",
                "ip": 4.2,
                "earned_runs": 0,
                "strikeouts": 5,
                "walks": 3,
                "pitch_count": 84,
                "xwoba_against": 0.267,
                "csw_pct": 0.25,
                "whiff_pct": 0.2,
                "avg_fb_velo": 94.6,
            },
            {
                "game_date": "2026-03-12",
                "ip": 6.2,
                "earned_runs": 4,
                "strikeouts": 7,
                "walks": 2,
                "pitch_count": 98,
                "xwoba_against": 0.332,
                "csw_pct": 0.28,
                "whiff_pct": 0.23,
                "avg_fb_velo": 95.0,
            },
            {
                "game_date": "2025-09-20",
                "ip": 7.0,
                "earned_runs": 5,
                "strikeouts": 8,
                "walks": 2,
                "pitch_count": 103,
                "xwoba_against": 0.341,
                "csw_pct": 0.3,
                "whiff_pct": 0.25,
                "avg_fb_velo": 94.9,
            },
        ]
    )
    recent_starts = [{"game_date": "2026-04-05", "earned_runs": 1}]

    monkeypatch.setattr(app_module, "_table_exists", lambda name: name == "pitcher_starts")
    monkeypatch.setattr(app_module, "_safe_frame", lambda *_args, **_kwargs: frame)
    monkeypatch.setattr(
        app_module,
        "_fetch_pitcher_recent_starts",
        lambda pitcher_id, target_date, limit=5: recent_starts,
    )

    result = app_module._fetch_starter_recent_form(77, date(2026, 4, 7))

    assert result["sample_starts"] == 5
    assert result["season_starts"] == 5
    assert result["avg_earned_runs"] == pytest.approx(2.0)
    assert result["last_start_date"] == "2026-04-05"
    assert result["season_era"] == pytest.approx(3.06818, rel=1e-4)
    assert result["era_last3"] == pytest.approx(3.0, rel=1e-4)
    assert result["era_last5"] == pytest.approx(3.06818, rel=1e-4)
    assert result["recent_starts"] == recent_starts