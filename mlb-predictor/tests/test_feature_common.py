from datetime import date, datetime, timezone

import pandas as pd

from src.features.common import build_hitter_priors, build_pitcher_priors, hitter_snapshot, latest_market_snapshot, pitcher_snapshot


def test_build_hitter_priors_handles_empty_frame_with_object_dates():
    frame = pd.DataFrame(columns=["player_id", "game_date", "hits", "xba", "xwoba", "hard_hit_pct", "strikeouts", "plate_appearances"])

    result = build_hitter_priors(frame, 2025)

    assert result.empty


def test_build_pitcher_priors_coerces_string_game_dates():
    frame = pd.DataFrame(
        [
            {"pitcher_id": 1, "game_date": "2025-04-01", "xwoba_against": 0.31, "csw_pct": 0.28, "avg_fb_velo": 94.1},
            {"pitcher_id": 1, "game_date": "2026-04-01", "xwoba_against": 0.35, "csw_pct": 0.25, "avg_fb_velo": 93.4},
        ]
    )

    result = build_pitcher_priors(frame, 2025)

    assert float(result.loc[1, "prior_xwoba"]) == 0.31
    assert float(result.loc[1, "prior_csw"]) == 0.28


def test_hitter_snapshot_handles_empty_history_frame():
    frame = pd.DataFrame(columns=["player_id", "game_date", "hits", "xba", "xwoba", "hard_hit_pct", "strikeouts", "plate_appearances"])

    result = hitter_snapshot(1, date(2026, 4, 2), frame, pd.DataFrame(), 120)

    assert result["hit_rate_7"] is None
    assert result["streak_len_capped"] == 0


def test_pitcher_snapshot_handles_string_game_dates():
    frame = pd.DataFrame(
        [
            {"pitcher_id": 7, "game_date": "2026-03-28", "xwoba_against": 0.29, "csw_pct": 0.31, "avg_fb_velo": 95.0},
        ]
    )

    result = pitcher_snapshot(7, date(2026, 4, 2), frame, pd.DataFrame(), 10)

    assert result["xwoba"] == 0.29
    assert result["days_rest"] == 5


def test_latest_market_snapshot_returns_selected_sportsbook():
    markets = pd.DataFrame(
        [
            {
                "game_id": 7,
                "sportsbook": "DraftKings",
                "line_value": 8.5,
                "over_price": -110,
                "under_price": -110,
                "snapshot_ts": pd.Timestamp("2026-04-02T16:00:00Z"),
            },
            {
                "game_id": 7,
                "sportsbook": "FanDuel",
                "line_value": 9.0,
                "over_price": -108,
                "under_price": -112,
                "snapshot_ts": pd.Timestamp("2026-04-02T17:00:00Z"),
            },
        ]
    )

    result = latest_market_snapshot(7, datetime(2026, 4, 2, 18, 0, tzinfo=timezone.utc), markets)

    assert result["market_total"] == 9.0
    assert result["market_sportsbook"] == "FanDuel"