from pathlib import Path
import re


def test_dashboard_html_contains_visible_loading_feedback():
    html = Path("src/api/static/index.html").read_text(encoding="utf-8")

    assert 'id="activityBanner"' in html
    assert 'id="activitySpinner"' in html
    assert 'Loading board...' in html
    assert 'renderLoadingPanel(' in html


def test_dashboard_html_keeps_primary_update_and_button_types():
    html = Path("src/api/static/index.html").read_text(encoding="utf-8")

    assert 'data-update-action="refresh_everything"' in html
    assert 'id="refreshButton" type="button"' in html
    assert re.search(
        r'data-update-action="rebuild_predictions"[^>]+type="button"',
        html,
    )