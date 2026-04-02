from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADD_DATA_SEPARATOR = ";" if os.name == "nt" else ":"


def add_data_argument(source: Path, destination: str) -> str:
    return f"{source}{ADD_DATA_SEPARATOR}{destination}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Windows desktop app shell with PyInstaller")
    parser.add_argument("--name", default="MLBPredictor", help="Executable/app directory name")
    parser.add_argument("--onefile", action="store_true", help="Build a one-file executable instead of the default one-folder app")
    return parser


def main() -> int:
    args = build_parser().parse_args()
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
        add_data_argument(ROOT / "src" / "api" / "static", "src/api/static"),
    ]
    if args.onefile:
        command.append("--onefile")
    command.append(str(launcher_path))

    completed = subprocess.run(command, cwd=ROOT)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())