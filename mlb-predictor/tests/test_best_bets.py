from src.utils import best_bets
from src.utils.settings import get_settings


def _card(game_id: int, weighted_ev: float, *, positive: bool) -> dict[str, object]:
    return {
        "game_id": game_id,
        "away_team": f"A{game_id}",
        "home_team": f"H{game_id}",
        "market_key": "moneyline",
        "selection_label": f"A{game_id} ML",
        "weighted_ev": weighted_ev,
        "probability_edge": weighted_ev / 2.0,
        "raw_ev": weighted_ev,
        "model_probability": 0.60,
        "certainty_weight": 0.90,
        "positive": positive,
        "input_trust": {"grade": "A", "score": 0.82},
        # Mirrors board ``data_quality.certainty_pct``; optional gate via BOARD_GREEN_MIN_GAME_CERTAINTY_PCT.
        "game_certainty_pct": 90.0,
    }


def test_watchlist_includes_positive_games_that_miss_green_board_cutoff():
    rows = []
    for game_id, weighted_ev in enumerate((0.90, 0.80, 0.70, 0.60, 0.50, 0.40), start=1):
        positive_card = _card(game_id, weighted_ev, positive=True)
        rows.append(
            {
                "game_id": game_id,
                "best_bets": [positive_card],
                "market_cards": [positive_card],
            }
        )

    green = best_bets.flatten_best_bets(rows)
    watchlist = best_bets.flatten_watchlist_markets(rows)

    assert [card["game_id"] for card in green] == [1, 2, 3, 4, 5, 6]
    assert [card["game_id"] for card in watchlist] == []


def test_annotate_market_card_sets_tier_and_gate_hints_for_raw_ev():
    game = {
        "game_id": 1,
        "away_team": "A",
        "home_team": "H",
        "certainty": {
            "starter_certainty": 0.7,
            "lineup_certainty": 0.7,
            "market_freshness": 0.7,
            "weather_freshness": 0.7,
            "bullpen_completeness": 0.7,
        },
        "totals": {"away_expected_runs": 4.0, "home_expected_runs": 4.0},
        "first5_totals": {},
    }
    cards, _ = best_bets.build_market_cards_for_game(
        game,
        {
            "moneyline": [
                {
                    "under_price": 150,
                    "over_price": -170,
                    "sportsbook": "Book",
                }
            ]
        },
    )
    ml = next(c for c in cards if c["market_key"] == "moneyline")
    assert ml.get("board_pick_tier") in ("strict", "soft_green", "watchlist", "raw_edge", "lean", "monitor")
    assert isinstance(ml.get("board_badges"), list)


def test_flatten_best_bets_prefers_one_pick_per_game_before_reusing_same_matchup():
    """High-EV game with many markets should not occupy all green-strip slots."""
    markets = (
        "moneyline",
        "run_line",
        "away_team_total",
        "home_team_total",
        "first_five_total",
    )
    cards_g1 = []
    for i, mk in enumerate(markets):
        c = _card(1, 0.90 - i * 0.01, positive=True)
        c["market_key"] = mk
        c["selection_label"] = f"pick-{mk}"
        cards_g1.append(c)
    rows = [
        {
            "game_id": 1,
            "best_bets": [cards_g1[0]],
            "market_cards": cards_g1[1:],
        }
    ]
    for gid in range(2, 10):
        rows.append(
            {
                "game_id": gid,
                "best_bets": [_card(gid, 0.50, positive=True)],
                "market_cards": [],
            }
        )

    green = best_bets.flatten_best_bets(rows, limit=5)
    game_ids = [int(c["game_id"]) for c in green]
    assert len(set(game_ids)) == 5
    assert 1 in game_ids


def test_flatten_best_bets_never_adds_second_pass_duplicates_same_game():
    """Green strip is capped at one pick per game; do not backfill slots with more markets from the same matchup."""
    markets = ("moneyline", "run_line", "away_team_total")
    cards_g1 = []
    for i, mk in enumerate(markets):
        c = _card(1, 0.95 - i * 0.01, positive=True)
        c["market_key"] = mk
        c["selection_label"] = f"g1-{mk}"
        cards_g1.append(c)
    rows = [
        {"game_id": 1, "best_bets": [cards_g1[0]], "market_cards": cards_g1[1:]},
        {"game_id": 2, "best_bets": [_card(2, 0.40, positive=True)], "market_cards": []},
    ]
    green = best_bets.flatten_best_bets(rows, limit=5)
    assert len(green) == 2
    assert [int(c["game_id"]) for c in green].count(1) == 1


def test_flatten_best_bets_tags_strict_vs_soft_green_strip_tier():
    strict = _card(1, 0.90, positive=True)
    soft = _card(2, 0.03, positive=False)
    soft["probability_edge"] = 0.05
    soft["model_probability"] = 0.56
    soft["certainty_weight"] = 0.90
    rows = [
        {"game_id": 1, "best_bets": [strict], "market_cards": []},
        {"game_id": 2, "best_bets": [], "market_cards": [soft]},
    ]
    green = best_bets.flatten_best_bets(rows)
    by_gid = {int(c["game_id"]): c for c in green}
    assert by_gid[1]["green_strip_tier"] == "strict"
    assert by_gid[2]["green_strip_tier"] == "soft"


def test_flatten_best_bets_includes_strict_positive_even_when_input_trust_not_actionable():
    """Full EV gates (`positive`) qualify for the strip; trust is not re-checked for strict rows."""
    good = _card(1, 0.90, positive=True)
    also_strict = _card(2, 0.90, positive=True)
    also_strict["input_trust"] = {"grade": "C", "score": 0.45}
    rows = [
        {"game_id": 1, "best_bets": [good], "market_cards": []},
        {"game_id": 2, "best_bets": [also_strict], "market_cards": []},
    ]
    green = best_bets.flatten_best_bets(rows)
    assert {int(c["game_id"]) for c in green} == {1, 2}


def test_qualifies_board_green_respects_min_game_certainty_pct(monkeypatch):
    monkeypatch.setenv("BOARD_GREEN_MIN_GAME_CERTAINTY_PCT", "84")
    get_settings.cache_clear()
    card = _card(1, 0.9, positive=True)
    card["game_certainty_pct"] = 70.0
    assert not best_bets.qualifies_board_green_strip(card)
    card["game_certainty_pct"] = 85.0
    assert best_bets.qualifies_board_green_strip(card)
    monkeypatch.delenv("BOARD_GREEN_MIN_GAME_CERTAINTY_PCT", raising=False)
    get_settings.cache_clear()


def test_flatten_best_bets_soft_green_requires_actionable_input_trust():
    soft_ok = _card(1, 0.03, positive=False)
    soft_ok["probability_edge"] = 0.05
    soft_ok["model_probability"] = 0.56
    soft_ok["certainty_weight"] = 0.90
    soft_bad = _card(2, 0.03, positive=False)
    soft_bad["probability_edge"] = 0.05
    soft_bad["model_probability"] = 0.56
    soft_bad["certainty_weight"] = 0.90
    soft_bad["input_trust"] = {"grade": "C", "score": 0.45}
    rows = [
        {"game_id": 1, "best_bets": [], "market_cards": [soft_ok]},
        {"game_id": 2, "best_bets": [], "market_cards": [soft_bad]},
    ]
    green = best_bets.flatten_best_bets(rows)
    assert [int(c["game_id"]) for c in green] == [1]


def test_watchlist_keeps_near_miss_games_when_no_positive_overflow_exists():
    rows = []
    for game_id, weighted_ev in enumerate((0.90, 0.80, 0.70, 0.60, 0.50), start=1):
        positive_card = _card(game_id, weighted_ev, positive=True)
        rows.append(
            {
                "game_id": game_id,
                "best_bets": [positive_card],
                "market_cards": [positive_card],
            }
        )

    near_miss = _card(6, 0.02, positive=False)
    near_miss["probability_edge"] = 0.05
    near_miss["model_probability"] = 0.56
    near_miss["certainty_weight"] = 0.90
    rows.append(
        {
            "game_id": 6,
            "best_bets": [],
            "market_cards": [near_miss],
        }
    )

    watchlist = best_bets.flatten_watchlist_markets(rows)

    assert [card["game_id"] for card in watchlist] == [6]


def test_flatten_watchlist_secondary_lines_keeps_non_green_markets_when_slate_has_greens():
    """Board mode skips whole games that have a green; secondary mode keeps other watchlist markets."""
    ml = _card(1, 0.90, positive=True)
    ml["market_key"] = "moneyline"
    ml["selection_label"] = "Away ML"
    total_soft = _card(1, 0.02, positive=False)
    total_soft["market_key"] = "game_total"
    total_soft["selection_label"] = "Over 8.5"
    total_soft["probability_edge"] = 0.05
    total_soft["model_probability"] = 0.56
    total_soft["certainty_weight"] = 0.90
    row = {"game_id": 1, "best_bets": [ml], "market_cards": [ml, total_soft]}
    board_wl = best_bets.flatten_watchlist_markets([row])
    assert board_wl == []
    sec_wl = best_bets.flatten_watchlist_markets([row], secondary_lines_only=True)
    assert len(sec_wl) == 1
    assert sec_wl[0]["market_key"] == "game_total"


def test_snapshot_recommendation_tiers_separates_green_board_and_watchlist():
    rows = []
    for game_id, weighted_ev in enumerate((0.90, 0.80, 0.70, 0.60, 0.50, 0.40), start=1):
        positive_card = _card(game_id, weighted_ev, positive=True)
        rows.append(
            {
                "game_id": game_id,
                "best_bets": [positive_card],
                "market_cards": [positive_card],
            }
        )

    snapshot = best_bets.snapshot_recommendation_tiers(rows)

    assert [card["game_id"] for card in snapshot["green_cards"]] == [1, 2, 3, 4, 5, 6]
    assert snapshot["watchlist_cards"] == []
    assert snapshot["green_lookup"][best_bets.recommendation_card_identity(snapshot["green_cards"][0])] == 1


def test_build_market_cards_supports_first_five_totals_best_bets():
    game = {
        "game_id": 77,
        "away_team": "ATL",
        "home_team": "NYM",
        "certainty": {
            "starter_certainty": 0.95,
            "lineup_certainty": 0.95,
            "market_freshness": 0.95,
            "weather_freshness": 0.95,
            "bullpen_completeness": 0.95,
        },
        "totals": {},
        "first5_totals": {
            "away_runs": 2.6,
            "home_runs": 2.4,
        },
    }

    cards, best_cards = best_bets.build_market_cards_for_game(
        game,
        {
            "first_five_total": [
                {
                    "line_value": 3.5,
                    "over_price": 100,
                    "under_price": -130,
                    "sportsbook": "TestBook",
                }
            ]
        },
    )

    first_five_total_card = next(card for card in cards if card["market_key"] == "first_five_total")

    assert first_five_total_card["selection_label"] == "F5 Over 3.5"
    assert first_five_total_card["positive"] is True
    assert [card["market_key"] for card in best_cards] == ["first_five_total"]


def test_grade_best_bet_pick_handles_first_five_total_results():
    grading = best_bets.grade_best_bet_pick(
        {
            "market_key": "first_five_total",
            "bet_side": "over",
            "line_value": 3.5,
        },
        first5_result={"total_runs": 4},
    )

    assert grading["graded"] is True
    assert grading["actual_side"] == "over"
    assert grading["success"] is True
    assert grading["actual_value"] == 1.0
    assert grading["actual_measure"] == 4


def test_build_market_cards_supports_first_five_spread_and_team_totals_best_bets():
    game = {
        "game_id": 88,
        "away_team": "ATL",
        "home_team": "NYM",
        "certainty": {
            "starter_certainty": 0.95,
            "lineup_certainty": 0.95,
            "market_freshness": 0.95,
            "weather_freshness": 0.95,
            "bullpen_completeness": 0.95,
        },
        "totals": {},
        "first5_totals": {
            "away_runs": 0.9,
            "home_runs": 3.1,
        },
    }

    cards, _ = best_bets.build_market_cards_for_game(
        game,
        {
            "first_five_spread": [
                {
                    "line_value": -0.5,
                    "over_price": 115,
                    "under_price": -135,
                    "sportsbook": "TestBook",
                }
            ],
            "first_five_team_total_home": [
                {
                    "line_value": 1.5,
                    "over_price": -105,
                    "under_price": -115,
                    "sportsbook": "TestBook",
                }
            ],
            "first_five_team_total_away": [
                {
                    "line_value": 1.5,
                    "over_price": 125,
                    "under_price": -145,
                    "sportsbook": "TestBook",
                }
            ],
        },
    )

    spread_card = next(card for card in cards if card["market_key"] == "first_five_spread")
    home_tt_card = next(card for card in cards if card["market_key"] == "first_five_team_total_home")
    away_tt_card = next(card for card in cards if card["market_key"] == "first_five_team_total_away")

    assert spread_card["selection_label"] == "F5 NYM -0.5"
    assert spread_card["positive"] is True
    assert home_tt_card["selection_label"] == "F5 NYM TT Over 1.5"
    assert away_tt_card["selection_label"] == "F5 ATL TT Under 1.5"


def test_grade_best_bet_pick_handles_first_five_spread_and_team_total_results():
    spread_grading = best_bets.grade_best_bet_pick(
        {
            "market_key": "first_five_spread",
            "bet_side": "home",
            "line_value": -0.5,
        },
        first5_result={"away_runs": 1, "home_runs": 2},
    )
    team_total_grading = best_bets.grade_best_bet_pick(
        {
            "market_key": "first_five_team_total_away",
            "bet_side": "under",
            "line_value": 1.5,
        },
        first5_result={"away_runs": 1},
    )

    assert spread_grading["graded"] is True
    assert spread_grading["actual_side"] == "home"
    assert spread_grading["success"] is True
    assert team_total_grading["graded"] is True
    assert team_total_grading["actual_side"] == "under"
    assert team_total_grading["success"] is True
    assert team_total_grading["actual_measure"] == 1


def test_dedupe_experimental_first_inning_keeps_nrfi_and_yrfi_per_game():
    rows = [
        {"game_id": 1, "market_type": "yrfi", "game_start_ts": "2026-04-01T17:00:00"},
        {"game_id": 1, "market_type": "nrfi", "game_start_ts": "2026-04-01T17:00:00"},
        {"game_id": 2, "market_type": "yrfi", "game_start_ts": "2026-04-01T20:00:00"},
    ]
    out = best_bets.dedupe_experimental_first_inning_by_game(rows, market_field="market_type")
    assert len(out) == 3
    g1_types = {r["market_type"] for r in out if r["game_id"] == 1}
    assert g1_types == {"nrfi", "yrfi"}
    assert next(r for r in out if r["game_id"] == 2)["market_type"] == "yrfi"


def test_build_market_cards_includes_game_total_from_total_market_rows():
    game = {
        "game_id": 101,
        "away_team": "ATL",
        "home_team": "NYM",
        "certainty": {
            "starter_certainty": 0.95,
            "lineup_certainty": 0.95,
            "market_freshness": 0.95,
            "weather_freshness": 0.95,
            "bullpen_completeness": 0.95,
        },
        "totals": {"predicted_total_runs": 8.4, "away_expected_runs": 4.1, "home_expected_runs": 4.3},
        "first5_totals": {},
    }
    cards, _ = best_bets.build_market_cards_for_game(
        game,
        {
            "total": [
                {
                    "line_value": 8.5,
                    "over_price": -105,
                    "under_price": -115,
                    "sportsbook": "TestBook",
                }
            ],
        },
    )
    gt = next(card for card in cards if card["market_key"] == "game_total")
    assert gt["market_label"] == "Game Total (Runs)"
    assert "Over" in gt["selection_label"] or "Under" in gt["selection_label"]


def test_build_market_cards_game_total_from_predictions_when_no_game_markets_row():
    """predictions_totals carries market_total + mean; board should still emit game_total if game_markets.total is empty."""
    game = {
        "game_id": 102,
        "away_team": "ATL",
        "home_team": "NYM",
        "certainty": {
            "starter_certainty": 0.95,
            "lineup_certainty": 0.95,
            "market_freshness": 0.95,
            "weather_freshness": 0.95,
            "bullpen_completeness": 0.95,
        },
        "totals": {
            "predicted_total_runs": 8.6,
            "market_total": 8.5,
            "away_expected_runs": 4.2,
            "home_expected_runs": 4.4,
        },
        "first5_totals": {},
    }
    cards, _ = best_bets.build_market_cards_for_game(game, {})
    gt = next(card for card in cards if card["market_key"] == "game_total")
    assert gt["market_label"] == "Game Total (Runs)"
    assert "placeholder" in (gt.get("market_summary") or "").lower()


def test_grade_best_bet_pick_game_total_uses_total_runs_or_sum():
    g = best_bets.grade_best_bet_pick(
        {
            "market_key": "game_total",
            "bet_side": "over",
            "line_value": 8.5,
        },
        actual_result={"total_runs": 9, "is_final": True},
    )
    assert g["graded"] is True
    assert g["actual_side"] == "over"
    assert g["success"] is True

    g2 = best_bets.grade_best_bet_pick(
        {
            "market_key": "game_total",
            "bet_side": "under",
            "line_value": 9.5,
        },
        actual_result={"away_runs": 4, "home_runs": 5, "is_final": True},
    )
    assert g2["graded"] is True
    assert g2["actual_side"] == "under"
    assert g2["actual_measure"] == 9


def test_grade_best_bet_pick_player_hits_yes():
    g = best_bets.grade_best_bet_pick(
        {"market_key": best_bets.PLAYER_HITS_MARKET_KEY, "bet_side": "yes"},
        actual_result={"is_final": True},
        player_prop_actuals={"hits": 2, "home_runs": 0},
    )
    assert g["graded"] is True
    assert g["actual_side"] == "yes"
    assert g["success"] is True


def test_grade_best_bet_pick_player_hr_yes():
    g = best_bets.grade_best_bet_pick(
        {"market_key": best_bets.PLAYER_HOME_RUN_MARKET_KEY, "bet_side": "yes"},
        actual_result={"is_final": True},
        player_prop_actuals={"hits": 1, "home_runs": 1},
    )
    assert g["graded"] is True
    assert g["actual_side"] == "yes"
    assert g["success"] is True


def test_grade_best_bet_pick_pitcher_strikeouts_under():
    g = best_bets.grade_best_bet_pick(
        {
            "market_key": best_bets.PITCHER_STRIKEOUTS_MARKET_KEY,
            "bet_side": "under",
            "line_value": 4.5,
        },
        actual_result={"is_final": True},
        pitcher_strikeouts_actual=4.0,
    )
    assert g["graded"] is True
    assert g["actual_side"] == "under"
    assert g["success"] is True


def test_team_best_pick_board_excludes_hitter_props_from_green_and_watchlist():
    """1+ Hits and HR props must never appear on green strip / team watchlist even if injected on a row."""
    hits_card = _card(99, 0.90, positive=True)
    hits_card["market_key"] = best_bets.PLAYER_HITS_MARKET_KEY
    hits_card["selection_label"] = "Player 1+ Hits"
    hr_card = _card(98, 0.88, positive=True)
    hr_card["market_key"] = best_bets.PLAYER_HOME_RUN_MARKET_KEY
    hr_card["selection_label"] = "To hit a HR"
    ml_card = _card(99, 0.85, positive=True)
    rows = [
        {"game_id": 99, "best_bets": [hits_card, hr_card, ml_card], "market_cards": []},
        {
            "game_id": 100,
            "best_bets": [],
            "market_cards": [
                {
                    **hits_card,
                    "game_id": 100,
                    "positive": False,
                    "weighted_ev": 0.05,
                    "probability_edge": 0.04,
                    "model_probability": 0.58,
                    "certainty_weight": 0.70,
                },
                {
                    **hr_card,
                    "game_id": 100,
                    "positive": False,
                    "weighted_ev": 0.04,
                    "probability_edge": 0.03,
                    "model_probability": 0.14,
                    "certainty_weight": 0.70,
                    "hr_model_only": True,
                },
            ],
        },
    ]
    green = best_bets.flatten_best_bets(rows)
    watch = best_bets.flatten_watchlist_markets(rows)
    assert not any(c.get("market_key") == best_bets.PLAYER_HITS_MARKET_KEY for c in green)
    assert not any(c.get("market_key") == best_bets.PLAYER_HOME_RUN_MARKET_KEY for c in green)
    assert not any(c.get("market_key") == best_bets.PLAYER_HITS_MARKET_KEY for c in watch)
    assert not any(c.get("market_key") == best_bets.PLAYER_HOME_RUN_MARKET_KEY for c in watch)
    assert best_bets.excluded_from_team_best_pick_board(best_bets.PLAYER_HITS_MARKET_KEY)
    assert best_bets.excluded_from_team_best_pick_board(best_bets.PLAYER_HOME_RUN_MARKET_KEY)
    assert not best_bets.excluded_from_team_best_pick_board("moneyline")


def test_build_player_hits_board_card_and_tiers():
    row = {
        "game_id": 1,
        "player_id": 99,
        "away_team": "BOS",
        "home_team": "NYY",
        "player_name": "J. Doe",
        "team": "BOS",
        "predicted_hit_probability": 0.58,
        "market_price": -120,
        "edge": 0.08,
        "fair_price": -140,
        "is_confirmed_lineup": True,
        "has_lineup_snapshot": True,
    }
    card = best_bets.build_player_hits_board_card(row)
    assert card is not None
    assert card["market_key"] == best_bets.PLAYER_HITS_MARKET_KEY
    ann = best_bets.annotate_market_card_for_display(card)
    assert ann.get("board_pick_tier")
    assert isinstance(ann.get("board_badges"), list)