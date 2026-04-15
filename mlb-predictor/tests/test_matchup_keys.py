from src.utils.matchup_keys import team_abbr_to_opponent_id


def test_team_abbr_to_opponent_id_stable():
    assert team_abbr_to_opponent_id("ATL") == team_abbr_to_opponent_id("ATL")
    assert team_abbr_to_opponent_id("ATL") != team_abbr_to_opponent_id("NYY")
