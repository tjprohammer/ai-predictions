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