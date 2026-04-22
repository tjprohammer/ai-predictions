"""Remove all frozen board pick rows (green, Top EV, pick-of-the-day run snapshots).

Uses ``DATABASE_URL`` / project settings. Does not touch ``market_snapshots`` or other tables.

Examples:
  python scripts/clear_board_pick_snapshots.py --dry-run
  python scripts/clear_board_pick_snapshots.py --yes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sqlalchemy import text

from src.utils.db import get_engine, table_exists
from src.utils.settings import get_settings

BOARD_PICK_SNAPSHOT_TABLES: tuple[str, ...] = (
    "board_green_snapshots",
    "board_green_run_snapshots",
    "board_top_ev_snapshots",
    "board_top_ev_run_snapshots",
    "board_pick_of_day_run_snapshots",
)


def _count_rows(table: str) -> int:
    engine = get_engine()
    with engine.connect() as conn:
        return int(conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar() or 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print per-table row counts and exit without deleting",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm deletion (required unless --dry-run)",
    )
    args = parser.parse_args()

    get_settings()
    missing = [t for t in BOARD_PICK_SNAPSHOT_TABLES if not table_exists(t)]
    present = [t for t in BOARD_PICK_SNAPSHOT_TABLES if t not in missing]

    if not present:
        print("No board pick snapshot tables exist yet; nothing to do.")
        return 0

    counts = {t: _count_rows(t) for t in present}
    total = sum(counts.values())
    print(f"Database: {get_settings().database_url.split('@')[-1] if '@' in get_settings().database_url else get_settings().database_url}")
    for t in present:
        print(f"  {t}: {counts[t]} rows")
    print(f"  total: {total} rows")

    if args.dry_run:
        return 0

    if not args.yes:
        print("Refusing to delete without --yes (or use --dry-run).")
        return 2

    engine = get_engine()
    with engine.begin() as conn:
        for t in present:
            conn.execute(text(f"DELETE FROM {t}"))

    print("Deleted all rows from the tables listed above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
