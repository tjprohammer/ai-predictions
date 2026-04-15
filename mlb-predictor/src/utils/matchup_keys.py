"""Stable keys for matchup_splits.opponent_id (team-level rows) — keep in sync with ingest."""

from __future__ import annotations


def team_abbr_to_opponent_id(team_abbr: str) -> int:
    """Map a team abbreviation to the numeric opponent_id used in matchup_splits.

    Must match ``_team_abbr_to_opponent_id`` in ``src/ingestors/matchup_splits.py``:
    sum of ASCII code points of the uppercased 3-char abbreviation (padded).
    """
    return sum(ord(c) for c in str(team_abbr).upper().ljust(3))
