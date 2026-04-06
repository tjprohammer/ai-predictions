import subprocess
from pathlib import Path
from zipfile import ZipFile

import scripts.build_windows_installer as build_windows_installer
from scripts.windows_signing import SigningConfig


def test_release_artifact_base_name_normalizes_version_label():
    assert (
        build_windows_installer._release_artifact_base_name("MLBPredictor", "0.2.0-beta 1", "Setup")
        == "MLBPredictor-Windows-v0.2.0-beta-1-Setup"
    )


def test_windows_version_value_strips_beta_label():
    assert build_windows_installer._windows_version_value("0.2.0-beta") == "0.2.0.0"
    assert build_windows_installer._windows_version_value("1.4") == "1.4.0.0"


def test_find_inno_setup_uses_env_and_path(monkeypatch, tmp_path):
    iscc_path = tmp_path / "Inno Setup 6" / "ISCC.exe"
    iscc_path.parent.mkdir(parents=True)
    iscc_path.write_text("exe", encoding="utf-8")

    monkeypatch.setenv("ISCC_PATH", str(iscc_path))
    monkeypatch.setattr(build_windows_installer.shutil, "which", lambda executable: None)

    assert build_windows_installer.find_inno_setup() == iscc_path


def test_build_inno_installer_passes_dynamic_defines(monkeypatch, tmp_path):
    commands: list[list[str]] = []

    class FakeCompletedProcess:
        def __init__(self, returncode=0):
            self.returncode = returncode

    def fake_run(command, cwd):
        commands.append(command)
        return FakeCompletedProcess(0)

    monkeypatch.setattr(build_windows_installer.subprocess, "run", fake_run)

    artifact = build_windows_installer.build_inno_installer(
        tmp_path / "ISCC.exe",
        "MLBPredictor",
        tmp_path / "dist" / "MLBPredictor",
        tmp_path / "release",
        "0.2.0",
        "MLB Predictor",
        "MLBPredictor.exe",
        "MLBPredictor",
    )

    assert artifact == tmp_path / "release" / "MLBPredictor-Windows-v0.2.0-Setup.exe"
    assert any(arg == "/DAppName=MLB Predictor" for arg in commands[0])
    assert any(arg == "/DAppExeName=MLBPredictor.exe" for arg in commands[0])
    assert any(arg == "/DAppInstallDirName=MLBPredictor" for arg in commands[0])
    assert any(arg == "/DVersionInfoProductVersion=0.2.0.0" for arg in commands[0])
    assert any(arg == "/DOutputBaseFilename=MLBPredictor-Windows-v0.2.0-Setup" for arg in commands[0])


def test_build_inno_installer_signs_setup_when_requested(monkeypatch, tmp_path):
    signed = {}

    class FakeCompletedProcess:
        def __init__(self, returncode=0):
            self.returncode = returncode

    monkeypatch.setattr(build_windows_installer.subprocess, "run", lambda command, cwd: FakeCompletedProcess(0))
    monkeypatch.setattr(
        build_windows_installer,
        "sign_file",
        lambda path, _config: signed.update({"path": path}),
    )

    artifact = build_windows_installer.build_inno_installer(
        tmp_path / "ISCC.exe",
        "MLBPredictor",
        tmp_path / "dist" / "MLBPredictor",
        tmp_path / "release",
        "0.2.0",
        "MLB Predictor",
        "MLBPredictor.exe",
        "MLBPredictor",
        signing_config=SigningConfig(signtool_path=tmp_path / "signtool.exe"),
    )

    assert signed["path"] == artifact


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
    zip_path = build_windows_installer.build_portable_installer_bundle("MLBPredictor", dist_dir, release_dir, "0.2.0-beta1")

    assert zip_path.exists()
    assert zip_path.name == "MLBPredictor-Windows-v0.2.0-beta1-PortableInstaller.zip"
    with ZipFile(zip_path) as zip_file:
        names = set(zip_file.namelist())

    assert "MLBPredictor-Windows-v0.2.0-beta1-Portable/install_mlb_predictor.cmd" in names
    assert "MLBPredictor-Windows-v0.2.0-beta1-Portable/install_mlb_predictor.ps1" in names
    assert "MLBPredictor-Windows-v0.2.0-beta1-Portable/uninstall_mlb_predictor.cmd" in names
    assert "MLBPredictor-Windows-v0.2.0-beta1-Portable/uninstall_mlb_predictor.ps1" in names
    assert "MLBPredictor-Windows-v0.2.0-beta1-Portable/MLBPredictor/MLBPredictor.exe" in names


def test_portable_installer_bundle_signs_nested_app_files(monkeypatch, tmp_path):
    dist_dir = tmp_path / "dist" / "MLBPredictor"
    dist_dir.mkdir(parents=True)
    (dist_dir / "MLBPredictor.exe").write_text("exe", encoding="utf-8")

    installer_dir = tmp_path / "installer"
    installer_dir.mkdir()
    for name in build_windows_installer.PORTABLE_INSTALLER_FILES:
        (installer_dir / name).write_text(name, encoding="utf-8")

    signed = {}
    monkeypatch.setattr(build_windows_installer, "INSTALLER_DIR", installer_dir)
    monkeypatch.setattr(
        build_windows_installer,
        "discover_signable_files",
        lambda path: [path / "MLBPredictor.exe"],
    )
    monkeypatch.setattr(
        build_windows_installer,
        "sign_files",
        lambda paths, _config: signed.update({"paths": paths}),
    )

    build_windows_installer.build_portable_installer_bundle(
        "MLBPredictor",
        dist_dir,
        tmp_path / "release",
        "0.2.0-beta1",
        signing_config=SigningConfig(signtool_path=tmp_path / "signtool.exe"),
    )

    assert signed["paths"] == [tmp_path / "release" / "MLBPredictor-Windows-v0.2.0-beta1-Portable" / "MLBPredictor" / "MLBPredictor.exe"]


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


def test_main_requires_inno_when_requested(tmp_path, monkeypatch, capsys):
    dist_dir = tmp_path / "dist" / "MLBPredictor"
    dist_dir.mkdir(parents=True)
    (dist_dir / "MLBPredictor.exe").write_text("exe", encoding="utf-8")

    monkeypatch.setattr(build_windows_installer, "find_inno_setup", lambda: None)
    monkeypatch.setattr(
        build_windows_installer,
        "build_parser",
        lambda: type(
            "Parser",
            (),
            {
                "parse_args": lambda self: type(
                    "Args",
                    (),
                    {
                        "app_name": "MLBPredictor",
                        "display_name": "MLB Predictor",
                        "dist_dir": str(dist_dir),
                        "release_dir": str(tmp_path / "release"),
                        "app_version": "0.1.0",
                        "exe_name": None,
                        "portable_only": False,
                        "require_inno": True,
                    },
                )()
            },
        )(),
    )

    result = build_windows_installer.main()
    output = capsys.readouterr().out

    assert result == 1
    assert "Inno Setup was not found" in output