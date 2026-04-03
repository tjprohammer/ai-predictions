import subprocess
from pathlib import Path
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


def test_install_script_succeeds_when_launch_is_canceled(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "MLBPredictor.exe").write_text("exe", encoding="utf-8")

    install_dir = tmp_path / "installed"
    install_script = Path(build_windows_installer.INSTALLER_DIR / "install_mlb_predictor.ps1")
    install_script_path = str(install_script).replace("'", "''")
    source_dir_path = str(source_dir).replace("'", "''")
    install_dir_path = str(install_dir).replace("'", "''")

    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        (
            "function Start-Process { throw [System.InvalidOperationException]::new('The operation was canceled by the user.') }; "
            f"& '{install_script_path}' -SourceDir '{source_dir_path}' -InstallDir '{install_dir_path}' -NoDesktopShortcut -NoStartMenuShortcut"
        ),
    ]

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    combined_output = f"{completed.stdout}\n{completed.stderr}"

    assert completed.returncode == 0
    assert install_dir.exists()
    assert (install_dir / "MLBPredictor.exe").exists()
    assert "Installed MLB Predictor to" in completed.stdout
    assert "Windows blocked or canceled the first" in combined_output
    assert "You can start it manually" in combined_output