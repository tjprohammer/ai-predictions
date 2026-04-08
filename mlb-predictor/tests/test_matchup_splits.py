"""Tests for the matchup-splits ingestor (StatMuse scraping & parsing)."""

from src.ingestors.matchup_splits import (
    TEAM_NICKNAMES,
    _bvp_url,
    _enrich_pvt_from_totals_row,
    _parse_bvp,
    _parse_pitcher_vs_team,
    _parse_platoon,
    _pitcher_vs_team_url,
    _platoon_url,
    _slugify,
    _team_abbr_to_opponent_id,
)


# ── _slugify ─────────────────────────────────────────────────────────────────

def test_slugify_basic():
    assert _slugify("Aaron Judge") == "aaron-judge"


def test_slugify_accented():
    assert _slugify("José Ramírez") == "jose-ramirez"


def test_slugify_suffix():
    assert _slugify("Vladimir Guerrero Jr.") == "vladimir-guerrero-jr"


def test_slugify_apostrophe():
    assert _slugify("Ke'Bryan Hayes") == "kebryan-hayes"


def test_slugify_extra_whitespace():
    assert _slugify("  Mike Trout  ") == "mike-trout"


# ── URL construction ─────────────────────────────────────────────────────────

def test_bvp_url():
    url = _bvp_url("Aaron Judge", "Gerrit Cole")
    assert url == "https://www.statmuse.com/mlb/ask/aaron-judge-vs-gerrit-cole-career"


def test_pitcher_vs_team_url():
    url = _pitcher_vs_team_url("Gerrit Cole", "HOU")
    assert url == "https://www.statmuse.com/mlb/ask/gerrit-cole-vs-astros"


def test_pitcher_vs_team_url_multi_word_nickname():
    url = _pitcher_vs_team_url("Chris Sale", "CWS")
    assert "white-sox" in url


# ── team nickname coverage ───────────────────────────────────────────────────

def test_all_30_teams_have_nicknames():
    expected_abbrs = {
        "ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE", "COL",
        "DET", "HOU", "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM",
        "NYY", "PHI", "PIT", "SDP", "SFG", "SEA", "STL", "TBR", "TEX",
        "TOR", "WSH",
    }
    # ATH and OAK are aliases for Athletics
    assert expected_abbrs.issubset(set(TEAM_NICKNAMES.keys()))


# ── _team_abbr_to_opponent_id ────────────────────────────────────────────────

def test_opponent_id_deterministic():
    id1 = _team_abbr_to_opponent_id("NYY")
    id2 = _team_abbr_to_opponent_id("NYY")
    assert id1 == id2


def test_opponent_id_different_teams():
    assert _team_abbr_to_opponent_id("NYY") != _team_abbr_to_opponent_id("NYM")


# ── BvP parsing ──────────────────────────────────────────────────────────────

def test_parse_bvp_short_pattern():
    html = (
        '<h1 class="something">Aaron Judge is 0-2 in 3 plate appearances '
        "against Gerrit Cole in his career.</h1>"
    )
    result = _parse_bvp(html)
    assert result is not None
    assert result["hits"] == 0
    assert result["at_bats"] == 2
    assert result["plate_appearances"] == 3
    assert result["batting_avg"] == 0.0


def test_parse_bvp_full_pattern():
    html = (
        "Mike Trout is 5-45 with 2 homers and 2 RBIs "
        "in 56 plate appearances against Justin Verlander in his career."
    )
    result = _parse_bvp(html)
    assert result is not None
    assert result["hits"] == 5
    assert result["at_bats"] == 45
    assert result["home_runs"] == 2
    assert result["rbi"] == 2
    assert result["plate_appearances"] == 56
    assert result["batting_avg"] == 0.1111


def test_parse_bvp_returns_none_for_no_data():
    html = "<html><body>We could not find any results for that query.</body></html>"
    result = _parse_bvp(html)
    assert result is None


def test_parse_bvp_singular_hit():
    html = "Mike Trout is 1-3 in 4 plate appearances against Justin Verlander."
    result = _parse_bvp(html)
    assert result is not None
    assert result["hits"] == 1
    assert result["at_bats"] == 3


def test_parse_bvp_full_pattern_singular():
    html = (
        "Player X is 1-2 with 1 homer and 1 RBI "
        "in 3 plate appearances versus the Dodgers."
    )
    result = _parse_bvp(html)
    assert result is not None
    assert result["batting_avg"] == 0.5
    assert result["hits"] == 1
    assert result["home_runs"] == 1
    assert result["plate_appearances"] == 3


# ── pitcher-vs-team parsing ──────────────────────────────────────────────────

def test_parse_pitcher_vs_team_record_pattern():
    html = (
        "Gerrit Cole is 1-2 with an ERA of 2.57 and "
        "28 strikeouts in 4 appearances versus the Astros in his career."
    )
    result = _parse_pitcher_vs_team(html)
    assert result is not None
    assert result["era"] == 2.57
    assert result["strikeouts"] == 28
    assert result["games"] == 4


def test_parse_pitcher_vs_team_alt_pattern():
    html = "Sandy Alcantara with an ERA of 2.89 and 45 strikeouts in 8 starts against the Braves."
    result = _parse_pitcher_vs_team(html)
    assert result is not None
    assert result["era"] == 2.89
    assert result["strikeouts"] == 45
    assert result["games"] == 8


def test_parse_pitcher_vs_team_returns_none():
    html = "<html><body>Sorry, no results found.</body></html>"
    result = _parse_pitcher_vs_team(html)
    assert result is None


# ── platoon URL construction ─────────────────────────────────────────────────

def test_platoon_url_left():
    url = _platoon_url("Aaron Judge", "left", 2025)
    assert url == "https://www.statmuse.com/mlb/ask/aaron-judge-vs-left-handed-pitchers-2025"


def test_platoon_url_right():
    url = _platoon_url("José Ramírez", "right", 2026)
    assert url == "https://www.statmuse.com/mlb/ask/jose-ramirez-vs-right-handed-pitchers-2026"


# ── platoon parsing (sentence) ───────────────────────────────────────────────

def test_parse_platoon_sentence():
    html = (
        "Aaron Judge had a batting average of .339 with 16 home runs "
        "and 30 RBIs in 161 plate appearances against left-handed "
        "pitchers in 2025."
    )
    result = _parse_platoon(html)
    assert result is not None
    assert result["batting_avg"] == 0.339
    assert result["home_runs"] == 16
    assert result["rbi"] == 30
    assert result["plate_appearances"] == 161


def test_parse_platoon_sentence_right():
    html = (
        "Mookie Betts had a batting average of .275 with 8 home runs "
        "and 22 RBIs in 200 plate appearances against right-handed "
        "pitchers in 2025."
    )
    result = _parse_platoon(html)
    assert result is not None
    assert result["batting_avg"] == 0.275
    assert result["home_runs"] == 8
    assert result["plate_appearances"] == 200


def test_parse_platoon_sentence_singular():
    html = (
        "Player X had a batting average of .200 with 1 home run "
        "and 1 RBI in 10 plate appearances against left-handed "
        "pitcher in 2026."
    )
    result = _parse_platoon(html)
    assert result is not None
    assert result["batting_avg"] == 0.2
    assert result["home_runs"] == 1
    assert result["plate_appearances"] == 10


def test_parse_platoon_short_sentence():
    html = "Rookie X hit .250 against left-handed pitchers in 2026."
    result = _parse_platoon(html)
    assert result is not None
    assert result["batting_avg"] == 0.25


def test_parse_platoon_hab_current_season():
    """Current-season platoon uses 'is H-AB with HR HR and RBI RBIs in PA PA' format."""
    html = (
        "Aaron Judge is 3-6 with 2 home runs and 3 RBIs "
        "in 6 plate appearances against left-handed pitchers this season."
    )
    result = _parse_platoon(html)
    assert result is not None
    assert result["hits"] == 3
    assert result["at_bats"] == 6
    assert result["home_runs"] == 2
    assert result["rbi"] == 3
    assert result["plate_appearances"] == 6
    assert result["batting_avg"] == 0.5


# ── platoon parsing (stat table) ─────────────────────────────────────────────

def test_parse_platoon_table_row():
    # Simulates the pipe-delimited stat-table row from StatMuse
    html = (
        "| 1 | Aaron Judge | A. Judge | 2025 | NYY NYY "
        "| 77 | 124 | 42 | 7 | 0 | 16 | 30 | 36 | 0 | 37 "
        "| 161 | 97 | 23 | 0 | 1 | 22 "
        "| .339 | .484 | .782 | 1.267 |"
    )
    result = _parse_platoon(html)
    assert result is not None
    assert result["games"] == 77
    assert result["at_bats"] == 124
    assert result["hits"] == 42
    assert result["doubles"] == 7
    assert result["triples"] == 0
    assert result["home_runs"] == 16
    assert result["rbi"] == 30
    assert result["walks"] == 36
    assert result["strikeouts"] == 37
    assert result["plate_appearances"] == 161
    assert result["batting_avg"] == 0.339
    assert result["obp"] == 0.484
    assert result["slg"] == 0.782
    assert result["ops"] == 1.267


def test_parse_platoon_table_preferred_over_sentence():
    # When both table and sentence are present, table has richer data
    html = (
        "Aaron Judge had a batting average of .339 with 16 home runs "
        "and 30 RBIs in 161 plate appearances against left-handed "
        "pitchers in 2025.\n"
        "| 1 | Aaron Judge | A. Judge | 2025 | NYY NYY "
        "| 77 | 124 | 42 | 7 | 0 | 16 | 30 | 36 | 0 | 37 "
        "| 161 | 97 | 23 | 0 | 1 | 22 "
        "| .339 | .484 | .782 | 1.267 |"
    )
    result = _parse_platoon(html)
    assert result is not None
    # Table row gives us OBP/SLG/OPS that the sentence doesn't
    assert result["obp"] == 0.484
    assert result["slg"] == 0.782
    assert result["ops"] == 1.267
    assert result["games"] == 77


def test_parse_platoon_returns_none():
    html = "<html><body>We could not find any results for that query.</body></html>"
    result = _parse_platoon(html)
    assert result is None


def test_parse_platoon_table_minimal_variable_cols():
    # Table with fewer variable columns between SO and AVG
    html = (
        "| 1 | Player X | P. X | 2026 | BOS BOS "
        "| 10 | 30 | 8 | 2 | 0 | 1 | 5 | 3 | 1 | 7 "
        "| 35 | 4 | 1 "
        "| .267 | .343 | .400 | .743 |"
    )
    result = _parse_platoon(html)
    assert result is not None
    assert result["games"] == 10
    assert result["hits"] == 8
    assert result["plate_appearances"] == 35
    assert result["batting_avg"] == 0.267


# ── PvT enrichment (Phase 3) ─────────────────────────────────────────


_COLE_VS_ASTROS_TOTAL_ROW = (
    "| | Total | | | | | | 4 | 2.57 | 28 | 1 | 1 | 0 "
    "| 28.0 | 19 | 8 | 8 | 4 | 5 | 0 | 0 | 1 | 104 | 0 |"
)


def test_enrich_pvt_totals_row_basic():
    """Enrichment extracts IP, H, ER, BB and computes WHIP/K9."""
    row = {"games": 4, "era": 2.57, "strikeouts": 28}
    result = _enrich_pvt_from_totals_row(_COLE_VS_ASTROS_TOTAL_ROW, row)
    assert result["innings_pitched"] == 28.0
    assert result["earned_runs"] == 8
    assert result["hits"] == 19
    assert result["walks"] == 5
    # WHIP = (19+5)/28 = 0.857
    assert result["whip"] == 0.857
    # K/9 = 28*9/28 = 9.0
    assert result["k_per_9"] == 9.0


def test_enrich_pvt_totals_row_no_match():
    """Enrichment returns input dict unchanged when Total row is missing."""
    row = {"games": 2, "era": 3.50, "strikeouts": 10}
    result = _enrich_pvt_from_totals_row("<html>nothing here</html>", row)
    assert result is row
    assert "innings_pitched" not in result
    assert "whip" not in result


def test_enrich_pvt_totals_row_zero_ip():
    """Enrichment doesn't crash when IP is zero (no division by zero)."""
    html = (
        "| | Total | | | | | | 1 | 99.00 | 0 | 0 | 0 | 0 "
        "| 0.0 | 3 | 2 | 2 | 1 | 1 | 0 | 0 | 0 | 10 | 0 |"
    )
    row = {"games": 1, "era": 99.0, "strikeouts": 0}
    result = _enrich_pvt_from_totals_row(html, row)
    assert result["innings_pitched"] == 0.0
    assert "whip" not in result
    assert "k_per_9" not in result


def test_enrich_pvt_preserves_existing_games():
    """setdefault keeps the caller-provided games value."""
    row = {"games": 99, "era": 2.57, "strikeouts": 28}
    result = _enrich_pvt_from_totals_row(_COLE_VS_ASTROS_TOTAL_ROW, row)
    assert result["games"] == 99  # not overwritten to 4


# ── _fetch_page 422 handling ─────────────────────────────────────────────────

def test_fetch_page_returns_empty_on_422(monkeypatch):
    """422 from StatMuse means 'no data', not a server error."""
    import requests
    from src.ingestors.matchup_splits import _fetch_page

    class FakeResp:
        status_code = 422
        text = ""
    monkeypatch.setattr(requests, "get", lambda *a, **kw: FakeResp())
    result = _fetch_page("https://www.statmuse.com/mlb/ask/foo")
    assert result == ""


def test_fetch_page_returns_none_on_5xx(monkeypatch):
    """5xx errors should still be treated as real failures (return None)."""
    import requests
    from src.ingestors.matchup_splits import _fetch_page

    class FakeResp:
        status_code = 500
        def raise_for_status(self):
            raise requests.HTTPError("500 Server Error")
    monkeypatch.setattr(requests, "get", lambda *a, **kw: FakeResp())
    result = _fetch_page("https://www.statmuse.com/mlb/ask/foo")
    assert result is None
