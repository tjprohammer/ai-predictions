from datetime import date

import pandas as pd

from src.ingestors.market_totals import (
    _extract_covers_player_prop_matchup_ids,
    _extract_covers_player_prop_rows,
    _extract_odds_api_event_prop_rows,
    _extract_rotowire_strikeout_rows,
    _format_required_player_prop_gap_summary,
    _parse_covers_date,
    _required_player_prop_coverage_gaps,
)


def test_required_player_prop_coverage_gaps_ignores_populated_rows():
    frame = pd.DataFrame(
        [
            {
                "game_date": "2026-04-01",
                "market_type": "pitcher_strikeouts",
                "player_name": "Pitcher A",
                "line_value": 5.5,
                "over_price": -110,
                "under_price": -110,
            }
        ]
    )

    gaps = _required_player_prop_coverage_gaps(frame, {"pitcher_strikeouts"})

    assert gaps == []


def test_required_player_prop_coverage_gaps_reports_blank_templates():
    frame = pd.DataFrame(
        [
            {
                "game_date": "2026-04-01",
                "market_type": "pitcher_strikeouts",
                "player_name": "Pitcher A",
                "line_value": None,
                "over_price": None,
                "under_price": None,
            },
            {
                "game_date": "2026-04-01",
                "market_type": "pitcher_strikeouts",
                "player_name": "Pitcher B",
                "line_value": None,
                "over_price": None,
                "under_price": None,
            },
        ]
    )

    gaps = _required_player_prop_coverage_gaps(frame, {"pitcher_strikeouts"})

    assert gaps == [
        {
            "game_date": "2026-04-01",
            "market_type": "pitcher_strikeouts",
            "blank_rows": 2,
            "players_preview": "Pitcher A, Pitcher B",
        }
    ]
    assert "2026-04-01 pitcher_strikeouts (2 blank rows): Pitcher A, Pitcher B" == _format_required_player_prop_gap_summary(gaps)


def test_extract_odds_api_event_prop_rows_maps_batter_hits_and_pitcher_strikeouts():
    event_payload = {
        "bookmakers": [
            {
                "key": "fanduel",
                "markets": [
                    {
                        "key": "batter_hits",
                        "last_update": "2026-04-01T18:30:00Z",
                        "outcomes": [
                            {"name": "Over", "description": "Nick Kurtz", "price": -145, "point": 0.5},
                            {"name": "Under", "description": "Nick Kurtz", "price": 114, "point": 0.5},
                        ],
                    },
                    {
                        "key": "pitcher_strikeouts",
                        "last_update": "2026-04-01T18:30:00Z",
                        "outcomes": [
                            {"name": "Over", "description": "Chris Sale", "price": -108, "point": 7.5},
                            {"name": "Under", "description": "Chris Sale", "price": -112, "point": 7.5},
                        ],
                    },
                ],
            }
        ]
    }
    matched_game = {
        "game_id": 824940,
        "game_date": date(2026, 4, 1),
        "home_team": "ATL",
        "away_team": "ATH",
    }
    slate_player_lookup = {
        (824940, "nick kurtz"): [{"player_id": 701762, "team": "ATH", "player_name": "Nick Kurtz"}],
        (824940, "chris sale"): [{"player_id": 519242, "team": "ATL", "player_name": "Chris Sale"}],
    }

    rows = _extract_odds_api_event_prop_rows(event_payload, matched_game, slate_player_lookup)

    assert rows == [
        {
            "game_id": 824940,
            "game_date": date(2026, 4, 1),
            "player_id": 701762,
            "player_name": "Nick Kurtz",
            "team": "ATH",
            "sportsbook": "fanduel",
            "market_type": "player_hits",
            "line_value": 0.5,
            "over_price": -145,
            "under_price": 114,
            "snapshot_ts": pd.Timestamp("2026-04-01T18:30:00Z").to_pydatetime(),
            "is_opening": False,
            "is_closing": False,
            "source_name": "the_odds_api",
        },
        {
            "game_id": 824940,
            "game_date": date(2026, 4, 1),
            "player_id": 519242,
            "player_name": "Chris Sale",
            "team": "ATL",
            "sportsbook": "fanduel",
            "market_type": "pitcher_strikeouts",
            "line_value": 7.5,
            "over_price": -108,
            "under_price": -112,
            "snapshot_ts": pd.Timestamp("2026-04-01T18:30:00Z").to_pydatetime(),
            "is_opening": False,
            "is_closing": False,
            "source_name": "the_odds_api",
        },
    ]


def test_extract_odds_api_event_prop_rows_skips_unresolved_players():
    event_payload = {
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "batter_hits",
                        "last_update": "2026-04-01T18:30:00Z",
                        "outcomes": [
                            {"name": "Over", "description": "Unknown Batter", "price": -120, "point": 0.5},
                            {"name": "Under", "description": "Unknown Batter", "price": -110, "point": 0.5},
                        ],
                    }
                ],
            }
        ]
    }
    matched_game = {
        "game_id": 824940,
        "game_date": date(2026, 4, 1),
        "home_team": "ATL",
        "away_team": "ATH",
    }

    rows = _extract_odds_api_event_prop_rows(event_payload, matched_game, {})

    assert rows == []


def test_parse_covers_date_supports_full_month_names():
    parsed = _parse_covers_date("April 1", 2026)

    assert parsed == date(2026, 4, 1)


def test_extract_covers_player_prop_matchup_ids_filters_requested_dates():
    page_html = """
    <section class="row covers-Covers-Props-Matchup-CTA-Container">
        <p class="message-label">April 1</p>
        <a href='/sport/baseball/mlb/matchup/368637/picks#props'>First</a>
        <a href='/sport/baseball/mlb/matchup/369363/picks#props'>Second</a>
        <p class="message-label">April 2</p>
        <a href='/sport/baseball/mlb/matchup/400001/picks#props'>Third</a>
    </section>
    """

    groups = _extract_covers_player_prop_matchup_ids(page_html, date(2026, 4, 1), date(2026, 4, 1))

    assert groups == {date(2026, 4, 1): ["368637", "369363"]}


def test_extract_covers_player_prop_rows_maps_hits_rows():
    partial_html = """
    <article aria-labelledby="article-h2-player-13099-MLB_GAME_PLAYER_HITS" class="player-prop-article MLB ">
        <div class="player-and-team-wrapper">
            <div class="team-logos">
                <span class="away-shortname shortname">NYY</span>
                <span>@</span>
                <span class="home-shortname shortname">SEA</span>
            </div>
            <div class="player-headshot-name">
                <div class="picture-div">
                    <img alt="Giancarlo Stanton" />
                </div>
            </div>
        </div>
        <div class="other-over-odds" data-num-col="2">
            0.5
            <div class="player-event">Total Hits</div>
        </div>
        <div class="player-compareOdds-div">
            <div class="other-odds-row" data-cols="2">
                <div class="other-odds-label">
                    <img src="https://img.covers.com/covers/data/sportsbooks/bet365.svg" />
                </div>
                <div class="other-over-odds" data-num-col="1">
                    <a>o0.5 <span class="oddtype">-145</span></a>
                </div>
                <div class="other-under-odds" data-num-col="1">
                    <a>u0.5 <span class="oddtype">+115</span></a>
                </div>
            </div>
        </div>
    </article>
    """
    game_lookup = {
        (date(2026, 4, 1), "SEA", "NYY"): {
            "game_id": 824941,
            "game_date": date(2026, 4, 1),
            "home_team": "SEA",
            "away_team": "NYY",
        }
    }
    slate_player_lookup = {
        (824941, "giancarlo stanton"): [
            {"player_id": 13099, "team": "NYY", "player_name": "Giancarlo Stanton"}
        ]
    }
    snapshot_ts = pd.Timestamp("2026-04-01T18:30:00Z").to_pydatetime()

    rows = _extract_covers_player_prop_rows(
        partial_html,
        date(2026, 4, 1),
        "player_hits",
        game_lookup,
        slate_player_lookup,
        snapshot_ts=snapshot_ts,
    )

    assert rows == [
        {
            "game_id": 824941,
            "game_date": date(2026, 4, 1),
            "player_id": 13099,
            "player_name": "Giancarlo Stanton",
            "team": "NYY",
            "sportsbook": "bet365",
            "market_type": "player_hits",
            "line_value": 0.5,
            "over_price": -145,
            "under_price": 115,
            "snapshot_ts": snapshot_ts,
            "is_opening": False,
            "is_closing": False,
            "source_name": "covers_player_props",
        }
    ]


def test_extract_rotowire_strikeout_rows_maps_books_and_opponent_notation():
    page_html = """
    <script>
    const settings = {
        data: [{"name":"Chris Sale","team":"ATL","opp":"@ATH","draftkings_strikeouts":"7.5","draftkings_strikeoutsUnder":"-110","draftkings_strikeoutsOver":"-120","mgm_strikeouts":"6.5","mgm_strikeoutsUnder":"","mgm_strikeoutsOver":""}],
        theme: 'table-theme-odds'
    };
    </script>
    """
    game_lookup = {
        (date(2026, 4, 1), "ATL", "ATH"): {
            "game_id": 824940,
            "game_date": date(2026, 4, 1),
            "home_team": "ATH",
            "away_team": "ATL",
        }
    }
    slate_player_lookup = {
        (824940, "chris sale"): [
            {"player_id": 519242, "team": "ATL", "player_name": "Chris Sale"}
        ]
    }
    snapshot_ts = pd.Timestamp("2026-04-01T18:30:00Z").to_pydatetime()

    rows = _extract_rotowire_strikeout_rows(
        page_html,
        date(2026, 4, 1),
        game_lookup,
        slate_player_lookup,
        snapshot_ts=snapshot_ts,
    )

    assert rows == [
        {
            "game_id": 824940,
            "game_date": date(2026, 4, 1),
            "player_id": 519242,
            "player_name": "Chris Sale",
            "team": "ATL",
            "sportsbook": "draftkings",
            "market_type": "pitcher_strikeouts",
            "line_value": 7.5,
            "over_price": -120,
            "under_price": -110,
            "snapshot_ts": snapshot_ts,
            "is_opening": False,
            "is_closing": False,
            "source_name": "rotowire_player_props",
        },
        {
            "game_id": 824940,
            "game_date": date(2026, 4, 1),
            "player_id": 519242,
            "player_name": "Chris Sale",
            "team": "ATL",
            "sportsbook": "mgm",
            "market_type": "pitcher_strikeouts",
            "line_value": 6.5,
            "over_price": None,
            "under_price": None,
            "snapshot_ts": snapshot_ts,
            "is_opening": False,
            "is_closing": False,
            "source_name": "rotowire_player_props",
        },
    ]