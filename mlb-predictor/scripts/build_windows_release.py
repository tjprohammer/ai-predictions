from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Windows desktop app and installer bundle")
    parser.add_argument("--name", default="MLBPredictor", help="Desktop executable/app directory name")
    parser.add_argument("--onefile", action="store_true", help="Build the desktop app as a single executable")
    parser.add_argument("--app-version", default="0.1.0", help="Installer app version label")
    parser.add_argument("--require-inno", action="store_true", help="Fail the release build if Inno Setup is unavailable instead of falling back to the portable installer bundle")
    return parser


def _run_step(command: list[str]) -> int:
    completed = subprocess.run(command, cwd=ROOT)
    return int(completed.returncode)


def main() -> int:
    args = build_parser().parse_args()

    seed_result = _run_step([sys.executable, "scripts/build_desktop_sqlite_seed.py"])
    if seed_result != 0:
        return seed_result

    desktop_command = [sys.executable, "scripts/build_windows_app.py", "--name", args.name]
    if args.onefile:
        desktop_command.append("--onefile")
    desktop_result = _run_step(desktop_command)
    if desktop_result != 0:
        return desktop_result

    installer_command = [
        sys.executable,
        "scripts/build_windows_installer.py",
        "--app-name",
        args.name,
        "--dist-dir",
        str(ROOT / "dist" / args.name),
        "--app-version",
        args.app_version,
    ]
    if args.require_inno:
        installer_command.append("--require-inno")
    return _run_step(installer_command)


if __name__ == "__main__":
    raise SystemExit(main())