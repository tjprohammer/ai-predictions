from pathlib import Path


def test_totals_html_uses_clickable_game_cards_and_matchup_link():
    html = Path("src/api/static/totals.html").read_text(encoding="utf-8")

    assert 'class="game-card-link"' in html
    assert 'class="game-card-glance"' in html
    assert 'Open Matchup Page' in html
    assert 'game?game_id=' in html
    assert 'Open the matchup page for lineups, hitter history, starter trends, bullpen detail, and the full breakdown.' in html
