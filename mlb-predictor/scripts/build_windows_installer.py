from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
INSTALLER_DIR = ROOT / "installer" / "windows"
DEFAULT_APP_NAME = "MLBPredictor"
DEFAULT_DIST_DIR = ROOT / "dist" / DEFAULT_APP_NAME
DEFAULT_RELEASE_DIR = ROOT / "release"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Windows installer for MLB Predictor")
    parser.add_argument("--app-name", default=DEFAULT_APP_NAME, help="Desktop app directory name under dist/")
    parser.add_argument("--dist-dir", default=str(DEFAULT_DIST_DIR), help="Built desktop app directory")
    parser.add_argument("--release-dir", default=str(DEFAULT_RELEASE_DIR), help="Installer output directory")
    parser.add_argument("--app-version", default="0.1.0", help="Installer app version label")
    return parser


def find_inno_setup() -> Path | None:
    candidates = [
        Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"),
        Path(r"C:\Program Files\Inno Setup 6\ISCC.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_inno_installer(iscc_path: Path, dist_dir: Path, release_dir: Path, app_version: str) -> Path:
    release_dir.mkdir(parents=True, exist_ok=True)
    script_path = INSTALLER_DIR / "MLBPredictor.iss"
    command = [
        str(iscc_path),
        f"/DSourceDir={dist_dir}",
        f"/DOutputDir={release_dir}",
        f"/DAppVersion={app_version}",
        str(script_path),
    ]
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return release_dir / "MLBPredictorSetup.exe"


def build_portable_installer_bundle(dist_dir: Path, release_dir: Path) -> Path:
    release_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = release_dir / "MLBPredictor-Windows"
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    (bundle_dir / "MLBPredictor").parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(dist_dir, bundle_dir / "MLBPredictor")
    shutil.copy2(INSTALLER_DIR / "install_mlb_predictor.ps1", bundle_dir / "install_mlb_predictor.ps1")
    shutil.copy2(INSTALLER_DIR / "uninstall_mlb_predictor.ps1", bundle_dir / "uninstall_mlb_predictor.ps1")

    zip_path = release_dir / "MLBPredictor-Windows-PortableInstaller.zip"
    if zip_path.exists():
        zip_path.unlink()
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zip_file:
        for path in bundle_dir.rglob("*"):
            if path.is_file():
                zip_file.write(path, path.relative_to(bundle_dir.parent))
    return zip_path


def main() -> int:
    args = build_parser().parse_args()
    dist_dir = Path(args.dist_dir).resolve()
    if args.dist_dir == str(DEFAULT_DIST_DIR):
        dist_dir = (ROOT / "dist" / args.app_name).resolve()
    release_dir = Path(args.release_dir).resolve()
    if not dist_dir.exists():
        print(f"Desktop bundle not found: {dist_dir}")
        return 1

    iscc_path = find_inno_setup()
    if iscc_path is not None:
        artifact = build_inno_installer(iscc_path, dist_dir, release_dir, args.app_version)
        print(f"Built Inno Setup installer: {artifact}")
        return 0

    artifact = build_portable_installer_bundle(dist_dir, release_dir)
    print(f"Inno Setup not found. Built portable installer bundle instead: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())