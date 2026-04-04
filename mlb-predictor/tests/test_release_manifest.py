import json

import scripts.build_windows_release as build_windows_release


def test_detect_built_release_artifacts_prefers_changed_files(tmp_path):
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    prefix = build_windows_release._release_artifact_prefix("MLBPredictor", "0.2.0-beta1")

    setup_path = release_dir / f"{prefix}-Setup.exe"
    portable_path = release_dir / f"{prefix}-PortableInstaller.zip"
    setup_path.write_text("old-setup", encoding="utf-8")
    portable_path.write_text("old-portable", encoding="utf-8")

    before_snapshot = build_windows_release._snapshot_release_artifacts(release_dir, prefix)
    setup_path.write_text("new-setup", encoding="utf-8")

    built_paths = build_windows_release._detect_built_release_artifacts(release_dir, prefix, before_snapshot)

    assert built_paths == [setup_path]


def test_write_release_sidecars_creates_manifest_checksums_and_notes(tmp_path):
    release_dir = tmp_path / "release"
    release_dir.mkdir()
    setup_path = release_dir / "MLBPredictor-Windows-v0.2.0-beta1-Setup.exe"
    portable_path = release_dir / "MLBPredictor-Windows-v0.2.0-beta1-PortableInstaller.zip"
    setup_path.write_text("setup-binary", encoding="utf-8")
    portable_path.write_text("portable-binary", encoding="utf-8")

    sidecars = build_windows_release._write_release_sidecars(
        "MLBPredictor",
        "0.2.0-beta1",
        release_dir,
        [setup_path, portable_path],
    )

    manifest = json.loads(sidecars["manifest"].read_text(encoding="utf-8"))
    checksums_text = sidecars["checksums"].read_text(encoding="utf-8")
    notes_text = sidecars["release_notes"].read_text(encoding="utf-8")

    assert manifest["app_version"] == "0.2.0-beta1"
    assert [artifact["file_name"] for artifact in manifest["artifacts"]] == sorted([setup_path.name, portable_path.name])
    assert sidecars["checksums"].name == "MLBPredictor-Windows-v0.2.0-beta1-checksums.txt"
    assert sidecars["release_notes"].name == "MLBPredictor-Windows-v0.2.0-beta1-release-notes.md"
    assert f"*{setup_path.name}" in checksums_text
    assert f"*{portable_path.name}" in checksums_text
    assert "## Assets" in notes_text
    assert setup_path.name in notes_text
    assert portable_path.name in notes_text