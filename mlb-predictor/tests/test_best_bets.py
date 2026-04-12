from src.utils import best_bets


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

    assert [card["game_id"] for card in green] == [1, 2, 3, 4, 5]
    assert [card["game_id"] for card in watchlist] == [6]


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

    assert [card["game_id"] for card in snapshot["green_cards"]] == [1, 2, 3, 4, 5]
    assert [card["game_id"] for card in snapshot["watchlist_cards"]] == [6]
    assert snapshot["green_lookup"][best_bets.recommendation_card_identity(snapshot["green_cards"][0])] == 1
    assert snapshot["watchlist_lookup"][best_bets.recommendation_card_identity(snapshot["watchlist_cards"][0])] == 1


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