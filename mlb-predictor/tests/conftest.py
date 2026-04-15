"""
Keep tests on SQLite so a developer machine with Postgres in config/.env does not
break unrelated unit tests or create cross-test engine state.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _tests_use_isolated_sqlite(monkeypatch, tmp_path):
    db_file = tmp_path / "pytest_mlb.sqlite3"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file.resolve().as_posix()}")

    from src.utils import db as db_module
    from src.utils.settings import get_settings

    get_settings.cache_clear()
    db_module.get_engine.cache_clear()
