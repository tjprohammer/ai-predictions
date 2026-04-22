"""Top EV pick selection across mixed market cards."""

from src.utils.top_ev_pick import select_top_weighted_ev_pick


def test_select_top_weighted_ev_prefers_higher_weighted_ev():
    cands = [
        {"market_key": "moneyline", "weighted_ev": 0.02, "probability_edge": 0.03},
        {"market_key": "game_total", "weighted_ev": 0.06, "probability_edge": 0.01},
        {"market_key": "player_hits", "weighted_ev": 0.04, "probability_edge": 0.05},
    ]
    out = select_top_weighted_ev_pick(cands)
    assert out is not None
    assert out["market_key"] == "game_total"
    assert out["top_ev_candidate_count"] == 3


def test_select_top_weighted_ev_skips_hr_model_only():
    cands = [
        {"market_key": "player_home_run", "weighted_ev": None, "hr_model_only": True},
        {"market_key": "moneyline", "weighted_ev": 0.01, "probability_edge": 0.02},
    ]
    out = select_top_weighted_ev_pick(cands)
    assert out is not None
    assert out["market_key"] == "moneyline"
    assert out["top_ev_candidate_count"] == 1


def test_select_top_weighted_ev_excludes_hr_yes_even_with_high_ev():
    """HR is not in TOP_EV_ELIGIBLE_MARKET_KEYS — headline EV should not anchor on rare props."""
    cands = [
        {
            "market_key": "player_home_run",
            "weighted_ev": 0.5,
            "probability_edge": 0.2,
        },
        {"market_key": "moneyline", "weighted_ev": 0.01, "probability_edge": 0.02},
    ]
    out = select_top_weighted_ev_pick(cands)
    assert out is not None
    assert out["market_key"] == "moneyline"
    assert out["top_ev_candidate_count"] == 1


def test_select_top_weighted_ev_f5_team_total_fallback_when_only_priced_rows(monkeypatch):
    """F5 team totals are excluded from the primary pool; fallback fills when that is all we have."""
    cands = [
        {
            "market_key": "first_five_team_total_away",
            "weighted_ev": 0.08,
            "probability_edge": 0.04,
        },
    ]
    assert select_top_weighted_ev_pick(cands)["market_key"] == "first_five_team_total_away"
    assert select_top_weighted_ev_pick(cands, allow_f5_team_total_fallback=False) is None

    monkeypatch.setenv("TOP_EV_F5_TEAM_TOTAL_FALLBACK", "false")
    assert select_top_weighted_ev_pick(cands) is None
    monkeypatch.delenv("TOP_EV_F5_TEAM_TOTAL_FALLBACK", raising=False)
