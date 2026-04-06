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


def test_matchup_detail_page_exists():
    html = Path("src/api/static/game.html").read_text(encoding="utf-8")

    assert 'MLB Predictor Matchup Detail' in html
    assert 'api/games/' in html
    assert 'runs allowed' in html
    assert 'Lineup source' in html
    assert 'Projected template lineup' in html
    assert 'Snapshot order' in html
    assert 'Green = hit · red = no hit' in html
    assert 'Last ${history.length} Games' in html