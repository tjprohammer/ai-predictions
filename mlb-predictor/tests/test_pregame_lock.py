"""Tests for pregame ingest lock (markets / lineups / probable starters)."""

from datetime import datetime, timedelta, timezone

from src.utils.pregame_lock import is_before_scheduled_first_pitch, is_pregame_ingest_locked


def test_lock_disabled_when_minutes_zero() -> None:
    start = datetime(2026, 4, 15, 23, 0, tzinfo=timezone.utc)
    assert is_pregame_ingest_locked(start, lock_minutes=0) is False


def test_lock_outside_window() -> None:
    first_pitch = datetime(2026, 4, 15, 23, 0, tzinfo=timezone.utc)
    now = first_pitch - timedelta(minutes=45)
    assert is_pregame_ingest_locked(first_pitch, lock_minutes=30, now=now) is False


def test_lock_inside_window() -> None:
    first_pitch = datetime(2026, 4, 15, 23, 0, tzinfo=timezone.utc)
    now = first_pitch - timedelta(minutes=29)
    assert is_pregame_ingest_locked(first_pitch, lock_minutes=30, now=now) is True


def test_lock_exactly_at_boundary() -> None:
    first_pitch = datetime(2026, 4, 15, 23, 0, tzinfo=timezone.utc)
    now = first_pitch - timedelta(minutes=30)
    assert is_pregame_ingest_locked(first_pitch, lock_minutes=30, now=now) is True


def test_missing_start_never_locked() -> None:
    assert is_pregame_ingest_locked(None, lock_minutes=30) is False


def test_before_first_pitch_true_only_pregame() -> None:
    first_pitch = datetime(2026, 4, 15, 23, 0, tzinfo=timezone.utc)
    assert is_before_scheduled_first_pitch(first_pitch, now=first_pitch - timedelta(seconds=1)) is True
    assert is_before_scheduled_first_pitch(first_pitch, now=first_pitch) is False
    assert is_before_scheduled_first_pitch(first_pitch, now=first_pitch + timedelta(minutes=1)) is False


def test_ingest_locked_after_start_but_not_before_pitch_semantics() -> None:
    """After first pitch, ingest skip stays on; snapshot insert must use ``is_before_scheduled_first_pitch``."""
    first_pitch = datetime(2026, 4, 15, 23, 0, tzinfo=timezone.utc)
    after = first_pitch + timedelta(hours=2)
    assert is_pregame_ingest_locked(first_pitch, lock_minutes=30, now=after) is True
    assert is_before_scheduled_first_pitch(first_pitch, now=after) is False
