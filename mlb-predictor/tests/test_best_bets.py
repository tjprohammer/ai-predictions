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


def test_dedupe_experimental_first_inning_prefers_nrfi_when_both_exist():
    rows = [
        {"game_id": 1, "market_type": "yrfi", "game_start_ts": "2026-04-01T17:00:00"},
        {"game_id": 1, "market_type": "nrfi", "game_start_ts": "2026-04-01T17:00:00"},
        {"game_id": 2, "market_type": "yrfi", "game_start_ts": "2026-04-01T20:00:00"},
    ]
    out = best_bets.dedupe_experimental_first_inning_by_game(rows, market_field="market_type")
    assert len(out) == 2
    g1 = next(r for r in out if r["game_id"] == 1)
    assert g1["market_type"] == "nrfi"
    assert next(r for r in out if r["game_id"] == 2)["market_type"] == "yrfi"