"""Board action score: feature alignment and optional inference."""

from src.models.board_action_score import (
    clear_loaded_model_cache,
    feature_vector_from_card,
    feature_vector_from_outcome_row,
    maybe_attach_action_score,
    training_feature_names,
)


def test_training_feature_names_length_matches_card_vector():
    card = {
        "market_key": "game_total",
        "model_probability": 0.55,
        "no_vig_probability": 0.52,
        "weighted_ev": 0.03,
        "probability_edge": 0.04,
        "input_trust": {"score": 0.85},
        "game_certainty_pct": 82.0,
        "positive": True,
    }
    xv = feature_vector_from_card(card)
    assert xv.shape == (1, len(training_feature_names()))


def test_card_and_outcome_rows_align():
    row = {
        "market": "game_total",
        "probability": 0.55,
        "market_line": 0.52,
        "opposite_probability": 0.42,
        "meta_payload": {
            "weighted_ev": 0.03,
            "probability_edge": 0.04,
            "input_trust_score": 0.85,
            "is_board_green_pick": True,
        },
    }
    a = feature_vector_from_outcome_row(row)
    card = {
        "market_key": "game_total",
        "model_probability": 0.55,
        "no_vig_probability": 0.52,
        "push_probability": 0.03,
        "weighted_ev": 0.03,
        "probability_edge": 0.04,
        "input_trust": {"score": 0.85},
        "game_certainty_pct": None,
        "positive": True,
    }
    b = feature_vector_from_card(card)
    assert a.shape == b.shape


def test_maybe_attach_without_artifact_leaves_card(monkeypatch, tmp_path):
    from src.models import board_action_score as bas

    monkeypatch.setattr(
        bas,
        "artifact_paths",
        lambda: (tmp_path / "missing_action_classifier.joblib", tmp_path / "missing_meta.json"),
    )
    clear_loaded_model_cache()
    card = {"market_key": "game_total", "model_probability": 0.5}
    out = maybe_attach_action_score(card)
    assert out is card
    assert "action_score" not in out
