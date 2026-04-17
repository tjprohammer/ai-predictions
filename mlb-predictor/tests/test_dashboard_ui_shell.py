from pathlib import Path
import re


def test_dashboard_html_contains_visible_loading_feedback():
    html = Path("src/api/static/index.html").read_text(encoding="utf-8")

    assert 'id="activityBanner"' in html
    assert 'id="activitySpinner"' in html
    assert 'id="activityProgress"' in html
    assert 'id="activityProgressBar"' in html
    assert 'role="progressbar"' in html
    assert 'Loading board...' in html
    assert 'renderLoadingPanel(' in html
    assert 'summarizeUpdateJobProgress(job)' in html
    assert 'progressVisible: true,' in html
    assert 'progressLabel: "Queued · starting"' in html


def test_dashboard_html_keeps_primary_update_and_button_types():
    html = Path("src/api/static/index.html").read_text(encoding="utf-8")

    assert 'id="refreshButton" type="button"' in html
    for action in (
        "refresh_everything",
        "prepare_slate",
        "import_manual_inputs",
        "update_lineups_only",
        "update_markets_only",
        "refresh_results",
        "rebuild_predictions",
        "grade_predictions",
    ):
        assert re.search(
            rf'data-update-action="{action}"[^>]+type="button"',
            html,
        )


def test_dashboard_html_links_to_doctor_surface():
    html = Path("src/api/static/index.html").read_text(encoding="utf-8")

    assert 'data-app-link="doctor"' in html
    assert '>Doctor</span></a' in html


def test_dashboard_html_exposes_board_and_review_workspaces():
    html = Path("src/api/static/index.html").read_text(encoding="utf-8")

    assert 'data-dashboard-view="board"' in html
    assert 'data-dashboard-view="review"' in html
    assert 'Review Workspace' in html
    assert 'Live Board' in html


def test_dashboard_html_links_to_matchup_page():
    html = Path("src/api/static/index.html").read_text(encoding="utf-8")

    assert '<a class="game-toggle" href="${matchupHref}">' in html
    assert 'game?game_id=' in html


def test_dashboard_html_exposes_player_availability_and_platoon_helpers():
    html = Path("src/api/static/index.html").read_text(encoding="utf-8")

    assert 'function playerAvailabilityTag(player)' in html
    assert 'compact-hit-player-chips' in html
    assert 'Roster Status' in html
    assert 'function renderLateBullpenBadge(game)' in html
    assert 'Late Pen ' in html


def test_matchup_detail_page_exists():
    html = Path("src/api/static/game.html").read_text(encoding="utf-8")

    assert 'MLB Predictor Matchup Detail' in html
    assert 'api/games/' in html
    assert 'Bullpen snapshot' in html
    assert 'last 3 bullpen days' in html
    assert 'Lineup source' in html
    assert 'Projected template lineup' in html
    assert 'Snapshot order' in html
    assert 'Actual Batting Order' in html
    assert "Full 1-9 order with form tags (hot, heating up, cold, steady)." in html
    assert 'ERA Season / Last 3 / Last 5' in html
    assert 'function batterHandMatchup(player)' in html
    assert "Strong vs today's" in html
    assert "From the 6th inning on (last up to five bullpen days each):" in html
    assert 'Late Bullpen' in html
    assert 'Green = hit · red = no hit' in html
    assert 'Last ${history.length} Games' in html


def test_matchup_detail_page_has_matchup_splits_section():
    html = Path("src/api/static/game.html").read_text(encoding="utf-8")

    # Section container and heading
    assert 'id="matchupSection"' in html
    assert 'id="matchupContent"' in html
    assert "Matchup Splits" in html
    assert "Batter vs. Pitcher &amp; Team History" in html

    # JS rendering functions (signatures include optional tier legend object)
    assert "function loadMatchupSplits(gameId)" in html
    assert "function renderBvpTable(" in html
    assert "function renderPvtTable(" in html
    assert "function renderPlatoonTable(" in html
    assert "function renderH2hCards(" in html

    # BvP table columns
    assert "vs. Pitcher" in html
    assert "formatBattingAverage(r.ops)" in html

    # PvT table columns
    assert "Pitcher vs. Opposing Team (Career)" in html
    assert "formatNumber(r.whip, 3)" in html
    assert "formatNumber(r.k_per_9, 2)" in html
    assert "formatNumber(r.innings_pitched, 1)" in html

    # Platoon table rendering
    assert "vs LHP" in html
    assert "vs RHP" in html

    # H2H cards
    assert "Avg Total Runs" in html
    assert "Over %" in html
    assert "Games Played" in html

    # Fetch call wired into page load
    assert "loadMatchupSplits(gameId)" in html
    assert "api/games/" in html