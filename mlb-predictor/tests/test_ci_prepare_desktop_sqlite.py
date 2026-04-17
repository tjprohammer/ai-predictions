"""Tests for scripts/ci_prepare_desktop_sqlite.py."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ci_prepare_desktop_sqlite.py"


def test_ci_prepare_exits_nonzero_without_ci_env():
    env = {k: v for k, v in os.environ.items() if k != "CI"}
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=SCRIPT.parent.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "refused" in (completed.stderr or "").lower()
