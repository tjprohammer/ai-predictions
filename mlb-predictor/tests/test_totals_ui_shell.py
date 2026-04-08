from pathlib import Path


def test_totals_html_uses_clickable_game_cards_and_matchup_link():
    html = Path("src/api/static/totals.html").read_text(encoding="utf-8")

    assert '<input id="hideStartedOnly" type="checkbox" />' in html
    assert '<option value="latebullpen">Late bullpen risk</option>' in html
    assert 'class="game-card-link"' in html
    assert 'class="game-card-glance"' in html
    assert 'Open Matchup Page' in html
    assert 'game?game_id=' in html
    assert 'Open the matchup page for lineups, hitter history, starter trends, bullpen detail, and the full breakdown.' in html
    assert 'function bullpenLateQuality(snapshot)' in html
    assert 'function gameLateBullpenRisk(game)' in html
    assert 'Late Bullpen Risk' in html
    assert 'Late innings (7+)' in html
    assert 'actual relief work in innings 7+ from MLB play-by-play' in html
