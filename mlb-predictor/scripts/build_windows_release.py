from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.build_windows_installer as build_windows_installer

PRIMARY_RELEASE_SUFFIXES = {
    ".exe": "setup",
    ".zip": "portable_installer",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Windows desktop app and installer bundle")
    parser.add_argument("--name", default="MLBPredictor", help="Desktop executable/app directory name")
    parser.add_argument("--onefile", action="store_true", help="Build the desktop app as a single executable")
    parser.add_argument(
        "--sign",
        action="store_true",
        help="Sign the built desktop app and installer using signtool.exe and WINDOWS_SIGN_* environment variables",
    )
    parser.add_argument("--app-version", default="1.1.0", help="Installer app version label")
    parser.add_argument("--release-dir", default=str(ROOT / "release"), help="Directory where release artifacts are written")
    parser.add_argument("--require-inno", action="store_true", help="Fail the release build if Inno Setup is unavailable instead of falling back to the portable installer bundle")
    return parser


def _run_step(command: list[str]) -> int:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    completed = subprocess.run(command, cwd=ROOT, env=env)
    return int(completed.returncode)


def _release_artifact_prefix(app_name: str, app_version: str) -> str:
    version_label = build_windows_installer._normalized_version_label(app_version)
    return f"{app_name}-Windows-{version_label}"


def _artifact_signature(path: Path) -> tuple[int, int]:
    stat_result = path.stat()
    return (stat_result.st_size, stat_result.st_mtime_ns)


def _find_release_artifacts(release_dir: Path, artifact_prefix: str) -> list[Path]:
    if not release_dir.exists():
        return []
    return sorted(
        path
        for path in release_dir.iterdir()
        if path.is_file() and path.name.startswith(artifact_prefix) and path.suffix.lower() in PRIMARY_RELEASE_SUFFIXES
    )


def _snapshot_release_artifacts(release_dir: Path, artifact_prefix: str) -> dict[Path, tuple[int, int]]:
    return {path: _artifact_signature(path) for path in _find_release_artifacts(release_dir, artifact_prefix)}


def _detect_built_release_artifacts(
    release_dir: Path,
    artifact_prefix: str,
    before_snapshot: dict[Path, tuple[int, int]],
) -> list[Path]:
    current_paths = _find_release_artifacts(release_dir, artifact_prefix)
    built_paths = [path for path in current_paths if before_snapshot.get(path) != _artifact_signature(path)]
    return built_paths or current_paths


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_release_records(artifacts: list[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for artifact in sorted(artifacts):
        records.append(
            {
                "file_name": artifact.name,
                "artifact_type": PRIMARY_RELEASE_SUFFIXES.get(artifact.suffix.lower(), "unknown"),
                "size_bytes": artifact.stat().st_size,
                "sha256": _sha256_file(artifact),
            }
        )
    return records


def _render_checksums(records: list[dict[str, object]]) -> str:
    return "\n".join(f"{record['sha256']} *{record['file_name']}" for record in records) + "\n"


def _render_release_notes_draft(
    app_name: str,
    app_version: str,
    artifact_records: list[dict[str, object]],
    manifest_name: str,
    checksums_name: str,
) -> str:
    artifact_lines = "\n".join(f"- {record['file_name']}" for record in artifact_records)
    return f"""# {app_name} {app_version}\n\n## Assets\n{artifact_lines}\n\n## Release Metadata\n- Manifest: {manifest_name}\n- Checksums: {checksums_name}\n\n## Highlights\n- TODO: summarize desktop and model changes for this beta.\n\n## Tester Instructions\n- Download a release asset from GitHub Releases, not the repository source ZIP.\n- Use the Setup executable when you want a normal Windows install flow.\n- Use the portable ZIP when Inno Setup was unavailable or you want a portable install path.\n- Verify the board, results, and Update Center load on a clean runtime.\n\n## Known Limitations\n- The full embedded-database cutover is still not complete for broader ingestion and training paths.\n- Desktop binary updates are still manual downloads; there is no auto-updater yet.\n"""


def _write_release_sidecars(app_name: str, app_version: str, release_dir: Path, artifacts: list[Path]) -> dict[str, Path]:
    release_dir.mkdir(parents=True, exist_ok=True)
    artifact_prefix = _release_artifact_prefix(app_name, app_version)
    manifest_path = release_dir / f"{artifact_prefix}-manifest.json"
    checksums_path = release_dir / f"{artifact_prefix}-checksums.txt"
    release_notes_path = release_dir / f"{artifact_prefix}-release-notes.md"

    artifact_records = _build_release_records(artifacts)
    manifest_payload = {
        "app_name": app_name,
        "app_version": app_version,
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "release_dir": str(release_dir),
        "artifacts": artifact_records,
        "sidecars": {
            "manifest": manifest_path.name,
            "checksums": checksums_path.name,
            "release_notes_draft": release_notes_path.name,
        },
    }

    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
    checksums_path.write_text(_render_checksums(artifact_records), encoding="utf-8")
    release_notes_path.write_text(
        _render_release_notes_draft(
            app_name,
            app_version,
            artifact_records,
            manifest_path.name,
            checksums_path.name,
        ),
        encoding="utf-8",
    )
    return {
        "manifest": manifest_path,
        "checksums": checksums_path,
        "release_notes": release_notes_path,
    }


def main() -> int:
    args = build_parser().parse_args()
    release_dir = Path(args.release_dir).resolve()
    artifact_prefix = _release_artifact_prefix(args.name, args.app_version)
    sign_requested = getattr(args, "sign", False)

    # Skip the seed rebuild when the configured source IS the SQLite seed file
    # itself (common in local-development-as-source workflows).  Deleting and
    # re-creating from the same file would zero it out.
    from src.utils.settings import get_settings as _get_settings

    _source_url = _get_settings().database_url.strip()
    _seed_path = ROOT / "db" / "mlb_predictor.sqlite3"
    _source_is_seed = _source_url.startswith("sqlite") and _seed_path.resolve().as_posix().lower() in _source_url.lower()

    if _source_is_seed:
        print(f"Source database IS the SQLite seed ({_seed_path.name}) — skipping seed rebuild.")
    else:
        seed_result = _run_step([sys.executable, "scripts/build_desktop_sqlite_seed.py"])
        if seed_result != 0:
            return seed_result

    desktop_command = [sys.executable, "scripts/build_windows_app.py", "--name", args.name]
    if args.onefile:
        desktop_command.append("--onefile")
    if sign_requested:
        desktop_command.append("--sign")
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
        "--release-dir",
        str(release_dir),
        "--app-version",
        args.app_version,
    ]
    if sign_requested:
        installer_command.append("--sign")
    if args.require_inno:
        installer_command.append("--require-inno")
    before_snapshot = _snapshot_release_artifacts(release_dir, artifact_prefix)
    installer_result = _run_step(installer_command)
    if installer_result != 0:
        return installer_result

    built_artifacts = _detect_built_release_artifacts(release_dir, artifact_prefix, before_snapshot)
    if not built_artifacts:
        print(f"No release artifacts found under {release_dir} for {artifact_prefix}")
        return 1

    sidecars = _write_release_sidecars(args.name, args.app_version, release_dir, built_artifacts)
    print(f"Wrote release manifest: {sidecars['manifest']}")
    print(f"Wrote checksums file: {sidecars['checksums']}")
    print(f"Wrote release notes draft: {sidecars['release_notes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())