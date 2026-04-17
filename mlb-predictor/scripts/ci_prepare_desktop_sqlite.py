"""Create ``db/mlb_predictor.sqlite3`` with migrations only (no data export).

Used by GitHub Actions Windows release builds: avoids PostgreSQL and
``build_desktop_sqlite_seed.py`` while still bundling a valid schema. Pair with
``build_windows_release.py --allow-incomplete-sqlite-seed`` so PyInstaller accepts
empty history tables.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    if (os.environ.get("CI") or "").strip().lower() not in {"1", "true", "yes"}:
        print(
            "ci_prepare_desktop_sqlite: refused (this script replaces db/mlb_predictor.sqlite3; "
            "it is intended for GitHub Actions where CI=true).",
            file=sys.stderr,
        )
        return 1

    seed = ROOT / "db" / "mlb_predictor.sqlite3"
    seed.parent.mkdir(parents=True, exist_ok=True)
    if seed.exists():
        seed.unlink()

    url = f"sqlite:///{seed.resolve().as_posix()}"
    os.environ["DATABASE_URL"] = url

    from sqlalchemy import create_engine

    from src.utils.db_migrate import run_migrations

    engine = create_engine(url)
    applied = run_migrations(engine=engine)
    print(f"ci_prepare_desktop_sqlite: applied {applied} migration(s) -> {seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
