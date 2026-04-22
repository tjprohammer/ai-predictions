from datetime import date

from src.utils import best_bets
from src.utils.settings import get_settings


def test_board_pick_composite_score_v1_respects_stronger_context(monkeypatch):
    base = {
        "market_key": "moneyline",
        "weighted_ev": 0.05,
        "probability_edge": 0.08,
        "certainty_weight": 0.8,
        "model_probability": 0.56,
        "game_certainty_pct": 80.0,
        "input_trust": {"grade": "B", "score": 0.75},
    }
    weak = dict(base)
    strong = dict(base)
    strong["game_certainty_pct"] = 95.0
    strong["input_trust"] = {"grade": "A", "score": 0.92}
    assert best_bets.board_pick_composite_score_v1(strong) > best_bets.board_pick_composite_score_v1(weak)


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
    """One green per game until slate slots fill; high-EV game should not take all strip slots."""
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
    """Green strip is capped per game; second pass still respects BOARD_GREEN_STRIP_MAX_PER_GAME_FIRST_PASS."""
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

    # Below BOARD_GREEN_SOFT_MIN_WEIGHTED_EV so this stays off the green strip (watchlist only).
    near_miss = _card(6, 0.010, positive=False)
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


def test_first_five_lane_research_only_gates_green_strip():
    assert best_bets.first_five_lane_research_only({"lane_status": "below_baseline"}) is True
    assert best_bets.first_five_lane_research_only({"lane_status": "above_baseline"}) is False
    card = {
        "market_key": "first_five_total",
        "positive": True,
        "weighted_ev": 0.2,
        "probability_edge": 0.1,
        "certainty_weight": 0.9,
        "model_probability": 0.58,
        "game_certainty_pct": 90.0,
        "input_trust": {"grade": "A", "score": 0.9},
        "lane_research_only": True,
    }
    assert best_bets.qualifies_board_green_strip(card) is False


def test_select_picks_of_the_day_filters_soft_and_trust_and_prob_floor(monkeypatch):
    monkeypatch.delenv("PICK_OF_THE_DAY_MIN_MODEL_PROB", raising=False)
    monkeypatch.delenv("PICK_OF_THE_DAY_MAX", raising=False)
    soft = _card(1, 0.08, positive=False)
    strict_low_mp = {**_card(2, 0.08, positive=True), "model_probability": 0.54}
    strict_ok = _card(3, 0.10, positive=True)
    strict_c_trust = {**_card(4, 0.10, positive=True), "input_trust": {"grade": "C", "score": 0.5}}
    rows = [
        {"game_id": 1, "best_bets": [soft], "market_cards": []},
        {"game_id": 2, "best_bets": [strict_low_mp], "market_cards": []},
        {"game_id": 3, "best_bets": [strict_ok], "market_cards": []},
        {"game_id": 4, "best_bets": [strict_c_trust], "market_cards": []},
    ]
    out = best_bets.select_picks_of_the_day(rows)
    assert len(out) == 2
    assert [int(x["game_id"]) for x in out] == [3, 4]
    assert out[0]["pick_of_day_rank"] == 1
    assert out[0].get("pick_of_day_trust_tier") == "primary"
    assert out[1].get("pick_of_day_trust_tier") == "relaxed"


def test_select_picks_of_the_day_relaxes_trust_when_only_c_strict(monkeypatch):
    monkeypatch.delenv("PICK_OF_THE_DAY_MIN_MODEL_PROB", raising=False)
    monkeypatch.delenv("PICK_OF_THE_DAY_MAX", raising=False)
    strict_c = {**_card(9, 0.12, positive=True), "input_trust": {"grade": "C", "score": 0.48}}
    rows = [{"game_id": 9, "best_bets": [strict_c], "market_cards": []}]
    out = best_bets.select_picks_of_the_day(rows)
    assert len(out) == 1
    assert int(out[0]["game_id"]) == 9
    assert out[0]["pick_of_day_trust_tier"] == "relaxed"
    assert "relaxed" in str(out[0].get("pick_of_day_note") or "")


def test_select_picks_of_the_day_still_empty_when_only_d_trust(monkeypatch):
    monkeypatch.delenv("PICK_OF_THE_DAY_MIN_MODEL_PROB", raising=False)
    monkeypatch.delenv("PICK_OF_THE_DAY_MAX", raising=False)
    strict_d = {
        **_card(8, 0.12, positive=True),
        "input_trust": {"grade": "D", "score": 0.25},
        "game_certainty_pct": 70.0,
    }
    rows = [{"game_id": 8, "best_bets": [strict_d], "market_cards": []}]
    assert best_bets.select_picks_of_the_day(rows) == []


def test_select_picks_of_the_day_includes_d_when_game_certainty_high(monkeypatch):
    monkeypatch.delenv("PICK_OF_THE_DAY_MIN_MODEL_PROB", raising=False)
    monkeypatch.delenv("PICK_OF_THE_DAY_MAX", raising=False)
    strict_d = {
        **_card(8, 0.12, positive=True),
        "input_trust": {"grade": "D", "score": 0.25},
        "game_certainty_pct": 94.0,
    }
    rows = [{"game_id": 8, "best_bets": [strict_d], "market_cards": []}]
    out = best_bets.select_picks_of_the_day(rows)
    assert len(out) == 1
    assert int(out[0]["game_id"]) == 8
    assert out[0].get("pick_of_day_trust_tier") == "relaxed"


def test_select_picks_of_the_day_respects_disable_high_certainty_d(monkeypatch):
    monkeypatch.setenv("PICK_OF_THE_DAY_ALLOW_HIGH_CERTAINTY_D", "false")
    monkeypatch.delenv("PICK_OF_THE_DAY_MIN_MODEL_PROB", raising=False)
    monkeypatch.delenv("PICK_OF_THE_DAY_MAX", raising=False)
    strict_d = {
        **_card(8, 0.12, positive=True),
        "input_trust": {"grade": "D", "score": 0.25},
        "game_certainty_pct": 94.0,
    }
    rows = [{"game_id": 8, "best_bets": [strict_d], "market_cards": []}]
    assert best_bets.select_picks_of_the_day(rows) == []


def test_select_picks_of_the_day_backfills_second_game_after_ab_then_d(monkeypatch):
    monkeypatch.delenv("PICK_OF_THE_DAY_MIN_MODEL_PROB", raising=False)
    monkeypatch.delenv("PICK_OF_THE_DAY_MAX", raising=False)
    a_pick = _card(1, 0.12, positive=True)
    d_pick = {
        **_card(2, 0.11, positive=True),
        "input_trust": {"grade": "D", "score": 0.2},
        "game_certainty_pct": 94.0,
    }
    rows = [
        {"game_id": 1, "best_bets": [a_pick], "market_cards": []},
        {"game_id": 2, "best_bets": [d_pick], "market_cards": []},
    ]
    out = best_bets.select_picks_of_the_day(rows)
    assert len(out) == 2
    assert [int(x["game_id"]) for x in out] == [1, 2]


def test_select_picks_of_the_day_relaxation_env_off_requires_ab(monkeypatch):
    monkeypatch.setenv("PICK_OF_THE_DAY_TRUST_RELAXATION", "false")
    monkeypatch.delenv("PICK_OF_THE_DAY_MIN_MODEL_PROB", raising=False)
    monkeypatch.delenv("PICK_OF_THE_DAY_MAX", raising=False)
    strict_c = {**_card(11, 0.12, positive=True), "input_trust": {"grade": "C", "score": 0.48}}
    rows = [{"game_id": 11, "best_bets": [strict_c], "market_cards": []}]
    assert best_bets.select_picks_of_the_day(rows) == []


def test_flatten_best_bets_and_picks_skip_started_games_on_live_slate(monkeypatch):
    """Today's slate: no green strip / pick-of-day for games past scheduled first pitch."""
    slate_day = date(2026, 4, 16)

    class _FixedToday:
        @staticmethod
        def today():
            return slate_day

    monkeypatch.setattr(best_bets, "date", _FixedToday)
    monkeypatch.setattr(best_bets, "is_before_scheduled_first_pitch", lambda _ts: False)
    row = {
        "game_id": 1,
        "game_start_ts": "2026-04-16T17:00:00Z",
        "best_bets": [_card(1, 0.9, positive=True)],
        "market_cards": [],
    }
    assert best_bets.flatten_best_bets([row], for_live_slate_date=slate_day) == []
    assert best_bets.select_picks_of_the_day([row], for_live_slate_date=slate_day) == []
    monkeypatch.setattr(best_bets, "is_before_scheduled_first_pitch", lambda _ts: True)
    assert len(best_bets.flatten_best_bets([row], for_live_slate_date=slate_day)) == 1
    assert len(best_bets.select_picks_of_the_day([row], for_live_slate_date=slate_day)) == 1
    assert len(best_bets.flatten_best_bets([row], for_live_slate_date=None)) == 1
    assert len(best_bets.flatten_best_bets([row], for_live_slate_date=date(2026, 4, 15))) == 1


def test_headline_picks_trust_preview_status_over_clock_near_first_pitch(monkeypatch):
    """StatsAPI still says preview — keep strip even if wall clock thinks first pitch passed (skew)."""
    slate_day = date(2026, 4, 16)

    class _FixedToday:
        @staticmethod
        def today():
            return slate_day

    monkeypatch.setattr(best_bets, "date", _FixedToday)
    monkeypatch.setattr(best_bets, "is_before_scheduled_first_pitch", lambda _ts: False)
    row = {
        "game_id": 1,
        "status": "preview",
        "game_start_ts": "2026-04-16T17:00:00Z",
        "best_bets": [_card(1, 0.9, positive=True)],
        "market_cards": [],
    }
    assert len(best_bets.flatten_best_bets([row], for_live_slate_date=slate_day)) == 1
    assert len(best_bets.select_picks_of_the_day([row], for_live_slate_date=slate_day)) == 1


def test_headline_picks_hide_when_status_live_even_if_clock_pregame(monkeypatch):
    slate_day = date(2026, 4, 16)

    class _FixedToday:
        @staticmethod
        def today():
            return slate_day

    monkeypatch.setattr(best_bets, "date", _FixedToday)
    monkeypatch.setattr(best_bets, "is_before_scheduled_first_pitch", lambda _ts: True)
    row = {
        "game_id": 1,
        "status": "live",
        "game_start_ts": "2026-04-16T23:00:00Z",
        "best_bets": [_card(1, 0.9, positive=True)],
        "market_cards": [],
    }
    assert best_bets.flatten_best_bets([row], for_live_slate_date=slate_day) == []
    assert best_bets.select_picks_of_the_day([row], for_live_slate_date=slate_day) == []


def test_headline_picks_unparseable_start_time_not_suppressed(monkeypatch):
    slate_day = date(2026, 4, 16)

    class _FixedToday:
        @staticmethod
        def today():
            return slate_day

    monkeypatch.setattr(best_bets, "date", _FixedToday)
    monkeypatch.setattr(best_bets, "is_before_scheduled_first_pitch", lambda _ts: False)
    row = {
        "game_id": 1,
        "game_start_ts": "not-a-timestamp",
        "best_bets": [_card(1, 0.9, positive=True)],
        "market_cards": [],
    }
    assert len(best_bets.flatten_best_bets([row], for_live_slate_date=slate_day)) == 1


def test_select_picks_of_the_day_soft_slate_allows_soft_numeric_ev(monkeypatch):
    """Board date today or later: soft numeric band can enter PoD (not only strict ``positive``)."""
    monkeypatch.delenv("PICK_OF_THE_DAY_MIN_MODEL_PROB", raising=False)
    monkeypatch.delenv("PICK_OF_THE_DAY_MAX", raising=False)
    fixed_today = date(2026, 4, 16)
    slate = date(2026, 4, 22)

    class _FixedDate:
        today = staticmethod(lambda: fixed_today)
        fromisoformat = staticmethod(date.fromisoformat)

    monkeypatch.setattr(best_bets, "date", _FixedDate)
    soft = {
        "game_id": 1,
        "away_team": "A1",
        "home_team": "H1",
        "market_key": "moneyline",
        "selection_label": "A1 ML",
        "weighted_ev": 0.02,
        "probability_edge": 0.03,
        "certainty_weight": 0.71,
        "model_probability": 0.52,
        "positive": False,
        "input_trust": {"grade": "A", "score": 0.82},
        "game_certainty_pct": 90.0,
    }
    rows = [{"game_id": 1, "best_bets": [soft], "market_cards": []}]
    out = best_bets.select_picks_of_the_day(rows, slate_game_date=slate)
    assert len(out) == 1
    assert "Game-day/future slate" in str(out[0].get("pick_of_day_note") or "")

    out_today = best_bets.select_picks_of_the_day(rows, slate_game_date=fixed_today)
    assert len(out_today) == 1

    out_past = best_bets.select_picks_of_the_day(rows, slate_game_date=fixed_today.replace(day=15))
    assert out_past == []

    monkeypatch.setenv("PICK_OF_THE_DAY_ALLOW_SOFT_GREEN_ON_FUTURE_SLATE", "false")
    assert best_bets.select_picks_of_the_day(rows, slate_game_date=slate) == []
    monkeypatch.delenv("PICK_OF_THE_DAY_ALLOW_SOFT_GREEN_ON_FUTURE_SLATE", raising=False)


def test_select_picks_soft_slate_c_trust_soft_numeric_not_on_green_strip(monkeypatch):
    """Thin trust (C): PoD soft-slate path can take the card; green strip still rejects."""
    monkeypatch.delenv("PICK_OF_THE_DAY_MIN_MODEL_PROB", raising=False)
    monkeypatch.delenv("PICK_OF_THE_DAY_MAX", raising=False)
    fixed_today = date(2026, 4, 22)

    class _FixedDate:
        today = staticmethod(lambda: fixed_today)
        fromisoformat = staticmethod(date.fromisoformat)

    monkeypatch.setattr(best_bets, "date", _FixedDate)
    soft_c = {
        "game_id": 1,
        "away_team": "A1",
        "home_team": "H1",
        "market_key": "moneyline",
        "selection_label": "A1 ML",
        "weighted_ev": 0.02,
        "probability_edge": 0.03,
        "certainty_weight": 0.71,
        "model_probability": 0.55,
        "positive": False,
        "input_trust": {"grade": "C", "score": 0.5},
        "game_certainty_pct": 90.0,
    }
    rows = [{"game_id": 1, "best_bets": [soft_c], "market_cards": []}]
    assert best_bets.qualifies_board_green_strip(soft_c) is False
    out = best_bets.select_picks_of_the_day(rows, slate_game_date=fixed_today)
    assert len(out) == 1


def test_select_picks_of_the_day_excludes_batter_tb_under_when_overs_only(monkeypatch):
    monkeypatch.delenv("PICK_OF_THE_DAY_MIN_MODEL_PROB", raising=False)
    monkeypatch.delenv("PICK_OF_THE_DAY_MAX", raising=False)
    monkeypatch.setenv("PICK_OF_THE_DAY_BATTER_TB_OVERS_ONLY", "true")
    tb_under = {
        **_card(1, 0.15, positive=True),
        "market_key": "batter_total_bases",
        "bet_side": "under",
        "selection_label": "Under 1.5 TB",
    }
    tb_over = {
        **_card(2, 0.10, positive=True),
        "market_key": "batter_total_bases",
        "bet_side": "over",
        "selection_label": "Over 1.5 TB",
    }
    rows = [
        {"game_id": 1, "best_bets": [tb_under], "market_cards": []},
        {"game_id": 2, "best_bets": [tb_over], "market_cards": []},
    ]
    out = best_bets.select_picks_of_the_day(rows)
    assert len(out) == 1
    assert int(out[0]["game_id"]) == 2
    assert str(out[0].get("bet_side") or "").lower() == "over"

    monkeypatch.setenv("PICK_OF_THE_DAY_BATTER_TB_OVERS_ONLY", "false")
    out2 = best_bets.select_picks_of_the_day(rows)
    assert len(out2) == 2


def test_select_picks_of_the_day_one_per_game_and_respects_default_max(monkeypatch):
    monkeypatch.delenv("PICK_OF_THE_DAY_MIN_MODEL_PROB", raising=False)
    monkeypatch.delenv("PICK_OF_THE_DAY_MAX", raising=False)
    rows = []
    for gid in range(1, 9):
        c = _card(gid, 0.05 + gid * 0.01, positive=True)
        alt = {**c, "selection_label": f"alt {gid}", "weighted_ev": 0.99}
        rows.append({"game_id": gid, "best_bets": [c], "market_cards": [alt]})
    out = best_bets.select_picks_of_the_day(rows)
    assert len(out) == 8
    assert {int(x["game_id"]) for x in out} == set(range(1, 9))
    assert sorted(int(x["pick_of_day_rank"]) for x in out) == list(range(1, 9))
    assert all(str(x.get("selection_label") or "").startswith("alt ") for x in out)


def test_pick_of_the_day_max_env_clamped_at_ten(monkeypatch):
    monkeypatch.setenv("PICK_OF_THE_DAY_MAX", "99")
    _, max_n = best_bets._pick_of_the_day_config()
    assert max_n == 10
    monkeypatch.delenv("PICK_OF_THE_DAY_MAX", raising=False)


def test_select_picks_time_quartile_spread_keeps_late_games_in_short_list(monkeypatch):
    """When spread is on, reserve slots across start-time quartiles before second pick from same window."""
    monkeypatch.delenv("PICK_OF_THE_DAY_MIN_MODEL_PROB", raising=False)
    monkeypatch.setenv("PICK_OF_THE_DAY_FULL_SLATE_PER_GAME", "false")
    monkeypatch.setenv("PICK_OF_THE_DAY_MAX", "3")
    monkeypatch.setenv("PICK_OF_THE_DAY_TIME_BUCKET_SPREAD", "true")

    def row(gid: int, wev: float, hour: int) -> dict:
        c = _card(gid, wev, positive=True)
        return {
            "game_id": gid,
            "game_start_ts": f"2026-07-15T{hour:02d}:05:00Z",
            "best_bets": [c],
            "market_cards": [],
        }

    rows = [
        row(1, 0.20, 10),
        row(2, 0.19, 11),
        row(3, 0.12, 15),
        row(4, 0.08, 18),
        row(5, 0.05, 22),
    ]
    out = best_bets.select_picks_of_the_day(rows)
    assert {int(x["game_id"]) for x in out} == {1, 3, 4}
    assert "quartile spread" in str(out[0].get("pick_of_day_note") or "").lower()

    monkeypatch.setenv("PICK_OF_THE_DAY_TIME_BUCKET_SPREAD", "false")
    out_flat = best_bets.select_picks_of_the_day(rows)
    assert [int(x["game_id"]) for x in out_flat] == [1, 2, 3]

    monkeypatch.delenv("PICK_OF_THE_DAY_MAX", raising=False)
    monkeypatch.delenv("PICK_OF_THE_DAY_TIME_BUCKET_SPREAD", raising=False)
    monkeypatch.delenv("PICK_OF_THE_DAY_FULL_SLATE_PER_GAME", raising=False)