from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
INSTALLER_DIR = ROOT / "installer" / "windows"
DEFAULT_APP_NAME = "MLBPredictor"
DEFAULT_DISPLAY_NAME = "MLB Predictor"
DEFAULT_DIST_DIR = ROOT / "dist" / DEFAULT_APP_NAME
DEFAULT_RELEASE_DIR = ROOT / "release"
PORTABLE_INSTALLER_FILES = [
    "install_mlb_predictor.cmd",
    "install_mlb_predictor.ps1",
    "uninstall_mlb_predictor.cmd",
    "uninstall_mlb_predictor.ps1",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Windows installer for MLB Predictor")
    parser.add_argument("--app-name", default=DEFAULT_APP_NAME, help="Desktop app directory name under dist/")
    parser.add_argument("--display-name", default=DEFAULT_DISPLAY_NAME, help="Installer display name")
    parser.add_argument("--dist-dir", default=str(DEFAULT_DIST_DIR), help="Built desktop app directory")
    parser.add_argument("--release-dir", default=str(DEFAULT_RELEASE_DIR), help="Installer output directory")
    parser.add_argument("--app-version", default="0.1.0", help="Installer app version label")
    parser.add_argument("--exe-name", help="Executable name inside the built app directory (defaults to <app-name>.exe)")
    parser.add_argument("--portable-only", action="store_true", help="Skip Inno Setup and build only the portable installer bundle")
    parser.add_argument("--require-inno", action="store_true", help="Fail instead of falling back to the portable bundle when Inno Setup is unavailable")
    return parser


def _normalized_version_label(app_version: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", app_version.strip())
    sanitized = re.sub(r"-{2,}", "-", sanitized).strip("-.") or "0.1.0"
    return sanitized if sanitized.lower().startswith("v") else f"v{sanitized}"


def _release_artifact_base_name(app_name: str, app_version: str, suffix: str) -> str:
    return f"{app_name}-Windows-{_normalized_version_label(app_version)}-{suffix}"


def _windows_version_value(app_version: str) -> str:
    numeric_parts = re.findall(r"[0-9]+", app_version)
    parts = (numeric_parts + ["0", "0", "0", "0"])[:4]
    return ".".join(parts)


def find_inno_setup() -> Path | None:
    candidates: list[Path] = []
    for env_name in ("ISCC_PATH", "INNO_SETUP_ISCC"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(Path(env_value))
    inno_home = os.environ.get("INNO_SETUP_HOME")
    if inno_home:
        candidates.append(Path(inno_home) / "ISCC.exe")
    for executable_name in ("ISCC.exe", "iscc"):
        which_match = shutil.which(executable_name)
        if which_match:
            candidates.append(Path(which_match))
    candidates.extend([
        Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"),
        Path(r"C:\Program Files\Inno Setup 6\ISCC.exe"),
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_inno_installer(
    iscc_path: Path,
    app_name: str,
    dist_dir: Path,
    release_dir: Path,
    app_version: str,
    display_name: str,
    exe_name: str,
    install_dir_name: str,
) -> Path:
    release_dir.mkdir(parents=True, exist_ok=True)
    script_path = INSTALLER_DIR / "MLBPredictor.iss"
    output_base_filename = _release_artifact_base_name(app_name, app_version, "Setup")
    command = [
        str(iscc_path),
        f"/DAppName={display_name}",
        f"/DAppExeName={exe_name}",
        f"/DAppInstallDirName={install_dir_name}",
        f"/DVersionInfoProductVersion={_windows_version_value(app_version)}",
        f"/DOutputBaseFilename={output_base_filename}",
        f"/DSourceDir={dist_dir}",
        f"/DOutputDir={release_dir}",
        f"/DAppVersion={app_version}",
        str(script_path),
    ]
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return release_dir / f"{output_base_filename}.exe"


def build_portable_installer_bundle(app_name: str, dist_dir: Path, release_dir: Path, app_version: str) -> Path:
    release_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = release_dir / _release_artifact_base_name(app_name, app_version, "Portable")
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    (bundle_dir / app_name).parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(dist_dir, bundle_dir / app_name)
    for installer_name in PORTABLE_INSTALLER_FILES:
        shutil.copy2(INSTALLER_DIR / installer_name, bundle_dir / installer_name)

    zip_path = release_dir / f"{_release_artifact_base_name(app_name, app_version, 'PortableInstaller')}.zip"
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
    exe_name = args.exe_name or f"{args.app_name}.exe"
    if not dist_dir.exists():
        print(f"Desktop bundle not found: {dist_dir}")
        return 1
    if not (dist_dir / exe_name).exists():
        print(f"Desktop executable not found: {dist_dir / exe_name}")
        return 1

    if args.portable_only:
        artifact = build_portable_installer_bundle(args.app_name, dist_dir, release_dir, args.app_version)
        print(f"Built portable installer bundle: {artifact}")
        return 0

    iscc_path = find_inno_setup()
    if iscc_path is not None:
        artifact = build_inno_installer(
            iscc_path,
            args.app_name,
            dist_dir,
            release_dir,
            args.app_version,
            args.display_name,
            exe_name,
            args.app_name,
        )
        print(f"Built Inno Setup installer: {artifact}")
        return 0

    if args.require_inno:
        print("Inno Setup was not found. Install Inno Setup 6 or set ISCC_PATH/INNO_SETUP_HOME, then rerun the build.")
        return 1

    artifact = build_portable_installer_bundle(args.app_name, dist_dir, release_dir, args.app_version)
    print(f"Inno Setup not found. Built portable installer bundle instead: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())