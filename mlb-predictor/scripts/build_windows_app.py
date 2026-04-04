from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADD_DATA_SEPARATOR = ";" if os.name == "nt" else ":"
SQLITE_SEED_PATH = ROOT / "db" / "mlb_predictor.sqlite3"
DESKTOP_HISTORY_REQUIRED_TABLES = (
    ("team_offense_daily", "team offense history"),
    ("bullpens_daily", "bullpen history"),
    ("player_game_batting", "hitter game logs"),
    ("player_trend_daily", "hot hitter trend history"),
    ("pitcher_starts", "pitcher start history"),
)
DYNAMIC_PACKAGES = [
    "src.ingestors",
    "src.features",
    "src.models",
    "src.transforms",
]
DYNAMIC_MODULES = [
    "src.ingestors.games",
    "src.ingestors.starters",
    "src.ingestors.prepare_slate_inputs",
    "src.ingestors.lineups",
    "src.ingestors.market_totals",
    "src.ingestors.boxscores",
    "src.ingestors.player_batting",
    "src.transforms.offense_daily",
    "src.transforms.bullpens_daily",
    "src.transforms.product_surfaces",
    "src.features.totals_builder",
    "src.features.first5_totals_builder",
    "src.features.hits_builder",
    "src.features.strikeouts_builder",
    "src.models.predict_totals",
    "src.models.predict_first5_totals",
    "src.models.predict_hits",
    "src.models.predict_strikeouts",
]


def add_data_argument(source: Path, destination: str) -> str:
    return f"{source}{ADD_DATA_SEPARATOR}{destination}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Windows desktop app shell with PyInstaller")
    parser.add_argument("--name", default="MLBPredictor", help="Executable/app directory name")
    parser.add_argument("--onefile", action="store_true", help="Build a one-file executable instead of the default one-folder app")
    parser.add_argument(
        "--allow-incomplete-sqlite-seed",
        action="store_true",
        help="Skip the bundled desktop SQLite seed completeness check before packaging",
    )
    return parser


def _sqlite_seed_row_counts(seed_path: Path) -> dict[str, int]:
    row_counts: dict[str, int] = {}
    if not seed_path.exists():
        return {table_name: 0 for table_name, _label in DESKTOP_HISTORY_REQUIRED_TABLES}

    connection = sqlite3.connect(seed_path)
    try:
        existing_tables = {
            str(name)
            for (name,) in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        for table_name, _label in DESKTOP_HISTORY_REQUIRED_TABLES:
            if table_name not in existing_tables:
                row_counts[table_name] = 0
                continue
            row_counts[table_name] = int(connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
    finally:
        connection.close()
    return row_counts


def _validate_sqlite_seed(seed_path: Path) -> str | None:
    row_counts = _sqlite_seed_row_counts(seed_path)
    missing = [
        f"{label} ({table_name})"
        for table_name, label in DESKTOP_HISTORY_REQUIRED_TABLES
        if row_counts.get(table_name, 0) <= 0
    ]
    if not missing:
        return None
    return (
        "Bundled desktop SQLite seed is incomplete. The desktop package is missing historical rows for "
        f"{', '.join(missing)}. Refresh db\\mlb_predictor.sqlite3 with scripts\\build_desktop_sqlite_seed.py "
        "before packaging, or rerun with --allow-incomplete-sqlite-seed if this build is intentionally for UI-only debugging."
    )


def main() -> int:
    args = build_parser().parse_args()
    if not args.allow_incomplete_sqlite_seed:
        validation_error = _validate_sqlite_seed(SQLITE_SEED_PATH)
        if validation_error is not None:
            print(validation_error)
            return 1
    launcher_path = ROOT / "src" / "desktop" / "launcher.py"
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",
        "--name",
        args.name,
        "--paths",
        str(ROOT),
        "--collect-submodules",
        "uvicorn",
        "--collect-submodules",
        "webview",
        "--hidden-import",
        "src.api.app",
        "--add-data",
        add_data_argument(ROOT / "data", "data"),
        "--add-data",
        add_data_argument(ROOT / "config", "config"),
        "--add-data",
        add_data_argument(ROOT / "db", "db"),
        "--add-data",
        add_data_argument(ROOT / "src", "src"),
    ]
    for package_name in DYNAMIC_PACKAGES:
        command.extend(["--collect-submodules", package_name])
    for module_name in DYNAMIC_MODULES:
        command.extend(["--hidden-import", module_name])
    if args.onefile:
        command.append("--onefile")
    command.append(str(launcher_path))

    completed = subprocess.run(command, cwd=ROOT)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())