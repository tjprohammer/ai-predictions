from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from src.utils.db import get_engine


def main() -> int:
    migration_dir = Path(__file__).resolve().parents[2] / "db" / "migrations"
    migration_files = sorted(migration_dir.glob("*.sql"))
    if not migration_files:
        print("No migration files found")
        return 0

    engine = get_engine()
    for migration_path in migration_files:
        print(f"Applying {migration_path.name} ...")
        with engine.begin() as connection:
            connection.execute(text(migration_path.read_text()))
        print("  done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())