from pathlib import Path


def test_pitchers_html_has_recent_strikeout_chart():
    html = Path("src/api/static/pitchers.html").read_text(encoding="utf-8")

    assert '<input id="hideStartedOnly" type="checkbox" />' in html
    assert 'id="leadersGrid"' in html
    assert 'api/leaders/season?' in html
    assert 'Season Leaders' in html
    assert 'Season Ks' in html
    assert "function recentStrikeoutChart(row)" in html
    assert "function pitcherHistoryFilterOptions()" in html
    assert "async function hydratePitcherHistory(rows)" in html
    assert "api/pitchers/${row.pitcher_id}/recent-starts" in html
    assert "api/trends/pitchers/${row.pitcher_id}" in html
    assert "history-baseline-label" in html
    assert "Green = cleared line · red = under line" in html
    assert 'data-pitcher-key="${pitcherKey}"' in html
    assert "recent_form.recent_starts" in html