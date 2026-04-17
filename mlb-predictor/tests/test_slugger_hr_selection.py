"""Slugger HR ranking uses power + market when model probability ties."""

from src.utils.slugger_hr_selection import _compute_slugger_context_score, slugger_hr_card_sort_key


def test_slugger_sort_prefers_higher_hr_per_pa_when_ev_tied():
    low_power = {
        "weighted_ev": None,
        "probability_edge": None,
        "model_probability": 1e-6,
        "_slugger_hr_per_pa": 0.02,
        "_slugger_implied": 0.05,
        "_slugger_proj_pa": 4.0,
    }
    high_power = {
        "weighted_ev": None,
        "probability_edge": None,
        "model_probability": 1e-6,
        "_slugger_hr_per_pa": 0.055,
        "_slugger_implied": 0.05,
        "_slugger_proj_pa": 4.0,
    }
    assert slugger_hr_card_sort_key(high_power) > slugger_hr_card_sort_key(low_power)


def test_slugger_sort_prefers_market_implied_when_power_tied():
    a = {
        "weighted_ev": None,
        "probability_edge": None,
        "model_probability": 1e-6,
        "_slugger_hr_per_pa": 0.04,
        "_slugger_implied": 0.08,
        "_slugger_proj_pa": 4.0,
    }
    b = {
        "weighted_ev": None,
        "probability_edge": None,
        "model_probability": 1e-6,
        "_slugger_hr_per_pa": 0.04,
        "_slugger_implied": 0.04,
        "_slugger_proj_pa": 4.0,
    }
    assert slugger_hr_card_sort_key(a) > slugger_hr_card_sort_key(b)


def test_slugger_context_score_prefers_todays_edge_over_raw_power():
    """Elite HR/PA must not beat a clearly better priced prop when WE is higher on the latter."""
    base_rec = {
        "hr_per_pa_blended": 0.028,
        "season_prior_hr_per_pa": 0.028,
        "xwoba_14": 0.32,
        "hard_hit_pct_14": 0.38,
        "hr_game_rate_30": 0.12,
        "hit_rate_7": 0.28,
        "hit_rate_14": 0.27,
        "streak_len_capped": 0.0,
        "park_hr_factor": 1.02,
        "opposing_starter_hr_per_9": 1.25,
        "opposing_starter_barrel_pct": 0.08,
        "bvp_ab": 0,
    }
    elite_power = {
        **base_rec,
        "hr_per_pa_blended": 0.058,
        "season_prior_hr_per_pa": 0.055,
    }
    value_pick = {**base_rec}

    card_elite = {
        "weighted_ev": 0.025,
        "probability_edge": 0.02,
        "certainty_weight": 0.7,
    }
    card_value = {
        "weighted_ev": 0.095,
        "probability_edge": 0.055,
        "certainty_weight": 0.72,
    }

    s_elite = _compute_slugger_context_score(elite_power, card_elite)
    s_value = _compute_slugger_context_score(value_pick, card_value)
    assert s_value > s_elite
