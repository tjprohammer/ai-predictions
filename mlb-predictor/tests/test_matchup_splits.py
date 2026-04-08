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
        '<h1 class="something">Aaron Judge is 0-for-2 in 3 plate appearances '
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
        "Shohei Ohtani has a batting average of .232 with 73 hits, "
        "15 home runs, 37 RBIs and 43 runs scored in 86 games "
        "versus the Astros in his career."
    )
    result = _parse_bvp(html)
    assert result is not None
    assert result["batting_avg"] == 0.232
    assert result["hits"] == 73
    assert result["home_runs"] == 15
    assert result["rbi"] == 37
    assert result["runs"] == 43
    assert result["games"] == 86


def test_parse_bvp_returns_none_for_no_data():
    html = "<html><body>We could not find any results for that query.</body></html>"
    result = _parse_bvp(html)
    assert result is None


def test_parse_bvp_singular_hit():
    html = "Mike Trout is 1-for-3 in 4 plate appearances against Justin Verlander."
    result = _parse_bvp(html)
    assert result is not None
    assert result["hits"] == 1
    assert result["at_bats"] == 3


def test_parse_bvp_full_pattern_singular():
    html = (
        "Player X has a batting average of .500 with 1 hit, "
        "1 home run, 1 RBI and 1 run scored in 1 game versus the Dodgers."
    )
    result = _parse_bvp(html)
    assert result is not None
    assert result["batting_avg"] == 0.5
    assert result["hits"] == 1
    assert result["home_runs"] == 1
    assert result["games"] == 1


# ── pitcher-vs-team parsing ──────────────────────────────────────────────────

def test_parse_pitcher_vs_team_record_pattern():
    html = (
        "Gerrit Cole has a 12-5 record with a 3.45 ERA and "
        "180 strikeouts in 22 games against the Astros."
    )
    result = _parse_pitcher_vs_team(html)
    assert result is not None
    assert result["era"] == 3.45
    assert result["strikeouts"] == 180
    assert result["games"] == 22


def test_parse_pitcher_vs_team_alt_pattern():
    html = "Sandy Alcantara has a 2.89 ERA with 45 strikeouts in 8 starts against the Braves."
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
        "Player X had a batting average of .500 with 1 home run "
        "and 1 RBI in 4 plate appearances against left-handed "
        "pitcher in 2026."
    )
    result = _parse_platoon(html)
    assert result is not None
    assert result["batting_avg"] == 0.5
    assert result["home_runs"] == 1
    assert result["plate_appearances"] == 4


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
