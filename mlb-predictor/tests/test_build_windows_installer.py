from zipfile import ZipFile

import scripts.build_windows_installer as build_windows_installer


def test_portable_installer_bundle_includes_cmd_wrappers(tmp_path, monkeypatch):
    dist_dir = tmp_path / "dist" / "MLBPredictor"
    dist_dir.mkdir(parents=True)
    (dist_dir / "MLBPredictor.exe").write_text("exe", encoding="utf-8")

    installer_dir = tmp_path / "installer"
    installer_dir.mkdir()
    for name in build_windows_installer.PORTABLE_INSTALLER_FILES:
        (installer_dir / name).write_text(name, encoding="utf-8")

    monkeypatch.setattr(build_windows_installer, "INSTALLER_DIR", installer_dir)

    release_dir = tmp_path / "release"
    zip_path = build_windows_installer.build_portable_installer_bundle(dist_dir, release_dir)

    assert zip_path.exists()
    with ZipFile(zip_path) as zip_file:
        names = set(zip_file.namelist())

    assert "MLBPredictor-Windows/install_mlb_predictor.cmd" in names
    assert "MLBPredictor-Windows/install_mlb_predictor.ps1" in names
    assert "MLBPredictor-Windows/uninstall_mlb_predictor.cmd" in names
    assert "MLBPredictor-Windows/uninstall_mlb_predictor.ps1" in names
    assert "MLBPredictor-Windows/MLBPredictor/MLBPredictor.exe" in names