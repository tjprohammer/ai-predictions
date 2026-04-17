import subprocess
import sys
from pathlib import Path

import scripts.build_windows_release as build_windows_release


def test_build_windows_release_builds_seed_before_packaging(monkeypatch):
    commands: list[list[str]] = []
    sidecar_calls: list[tuple[str, str]] = []

    class FakeCompletedProcess:
        def __init__(self, returncode=0):
            self.returncode = returncode

    def fake_run(command, cwd, **kwargs):
        commands.append(command)
        return FakeCompletedProcess(0)

    monkeypatch.setattr(build_windows_release.subprocess, "run", fake_run)
    monkeypatch.setattr(build_windows_release.sys, "argv", ["build_windows_release.py"])
    monkeypatch.setattr(build_windows_release, "_snapshot_release_artifacts", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        build_windows_release,
        "_detect_built_release_artifacts",
        lambda *_args, **_kwargs: [build_windows_release.ROOT / "release" / "MLBPredictor-Windows-v1.1.0-Setup.exe"],
    )
    monkeypatch.setattr(
        build_windows_release,
        "_write_release_sidecars",
        lambda app_name, app_version, *_args, **_kwargs: sidecar_calls.append((app_name, app_version))
        or {
            "manifest": build_windows_release.ROOT / "release" / "manifest.json",
            "checksums": build_windows_release.ROOT / "release" / "checksums.txt",
            "release_notes": build_windows_release.ROOT / "release" / "release-notes.md",
        },
    )

    # Force a non-matching DATABASE_URL so the seed step is NOT skipped.
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://user:pass@host/db")
    from src.utils.settings import get_settings
    get_settings.cache_clear()

    result = build_windows_release.main()

    assert result == 0
    assert commands[0][1] == "scripts/build_desktop_sqlite_seed.py"
    assert commands[1][1] == "scripts/build_windows_app.py"
    assert commands[2][1] == "scripts/build_windows_installer.py"
    assert sidecar_calls == [("MLBPredictor", "1.1.0")]


def test_build_windows_release_stops_when_seed_build_fails(monkeypatch):
    commands: list[list[str]] = []

    class FakeCompletedProcess:
        def __init__(self, returncode=0):
            self.returncode = returncode

    def fake_run(command, cwd, **kwargs):
        commands.append(command)
        if command[1] == "scripts/build_desktop_sqlite_seed.py":
            return FakeCompletedProcess(2)
        return FakeCompletedProcess(0)

    monkeypatch.setattr(build_windows_release.subprocess, "run", fake_run)
    monkeypatch.setattr(build_windows_release.sys, "argv", ["build_windows_release.py"])

    # Force a non-matching DATABASE_URL so the seed step is NOT skipped.
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://user:pass@host/db")
    from src.utils.settings import get_settings
    get_settings.cache_clear()

    result = build_windows_release.main()

    assert result == 2
    assert len(commands) == 1


def test_build_windows_release_passes_require_inno_to_installer(monkeypatch):
    commands: list[list[str]] = []

    class FakeCompletedProcess:
        def __init__(self, returncode=0):
            self.returncode = returncode

    def fake_run(command, cwd, **kwargs):
        commands.append(command)
        return FakeCompletedProcess(0)

    monkeypatch.setattr(build_windows_release.subprocess, "run", fake_run)
    monkeypatch.setattr(
        build_windows_release.sys,
        "argv",
        ["build_windows_release.py", "--require-inno"],
    )
    monkeypatch.setattr(build_windows_release, "_snapshot_release_artifacts", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        build_windows_release,
        "_detect_built_release_artifacts",
        lambda *_args, **_kwargs: [build_windows_release.ROOT / "release" / "MLBPredictor-Windows-v1.1.0-Setup.exe"],
    )
    monkeypatch.setattr(
        build_windows_release,
        "_write_release_sidecars",
        lambda *_args, **_kwargs: {
            "manifest": build_windows_release.ROOT / "release" / "manifest.json",
            "checksums": build_windows_release.ROOT / "release" / "checksums.txt",
            "release_notes": build_windows_release.ROOT / "release" / "release-notes.md",
        },
    )

    result = build_windows_release.main()

    assert result == 0
    assert commands[-1][-1] == "--require-inno"


def test_build_windows_release_passes_app_version_to_installer(monkeypatch):
    commands: list[list[str]] = []

    class FakeCompletedProcess:
        def __init__(self, returncode=0):
            self.returncode = returncode

    def fake_run(command, cwd, **kwargs):
        commands.append(command)
        return FakeCompletedProcess(0)

    monkeypatch.setattr(build_windows_release.subprocess, "run", fake_run)
    monkeypatch.setattr(
        build_windows_release.sys,
        "argv",
        ["build_windows_release.py", "--app-version", "0.2.0-beta1"],
    )
    monkeypatch.setattr(build_windows_release, "_snapshot_release_artifacts", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        build_windows_release,
        "_detect_built_release_artifacts",
        lambda *_args, **_kwargs: [build_windows_release.ROOT / "release" / "MLBPredictor-Windows-v0.2.0-beta1-Setup.exe"],
    )
    monkeypatch.setattr(
        build_windows_release,
        "_write_release_sidecars",
        lambda *_args, **_kwargs: {
            "manifest": build_windows_release.ROOT / "release" / "manifest.json",
            "checksums": build_windows_release.ROOT / "release" / "checksums.txt",
            "release_notes": build_windows_release.ROOT / "release" / "release-notes.md",
        },
    )

    result = build_windows_release.main()

    assert result == 0
    assert "--app-version" in commands[-1]
    assert "0.2.0-beta1" in commands[-1]


def test_build_windows_release_passes_release_dir_to_installer(monkeypatch, tmp_path):
    commands: list[list[str]] = []

    class FakeCompletedProcess:
        def __init__(self, returncode=0):
            self.returncode = returncode

    def fake_run(command, cwd, **kwargs):
        commands.append(command)
        return FakeCompletedProcess(0)

    monkeypatch.setattr(build_windows_release.subprocess, "run", fake_run)
    monkeypatch.setattr(
        build_windows_release.sys,
        "argv",
        ["build_windows_release.py", "--release-dir", str(tmp_path / "release-output")],
    )
    monkeypatch.setattr(build_windows_release, "_snapshot_release_artifacts", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        build_windows_release,
        "_detect_built_release_artifacts",
        lambda *_args, **_kwargs: [tmp_path / "release-output" / "MLBPredictor-Windows-v1.1.0-Setup.exe"],
    )
    monkeypatch.setattr(
        build_windows_release,
        "_write_release_sidecars",
        lambda *_args, **_kwargs: {
            "manifest": tmp_path / "release-output" / "manifest.json",
            "checksums": tmp_path / "release-output" / "checksums.txt",
            "release_notes": tmp_path / "release-output" / "release-notes.md",
        },
    )

    result = build_windows_release.main()

    assert result == 0
    assert "--release-dir" in commands[-1]
    assert str((tmp_path / "release-output").resolve()) in commands[-1]


def test_build_windows_release_script_runs_directly_with_help():
    script_path = Path(build_windows_release.ROOT / "scripts" / "build_windows_release.py")

    completed = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=build_windows_release.ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "Build the Windows desktop app and installer bundle" in completed.stdout


def test_build_windows_release_passes_sign_to_app_and_installer(monkeypatch):
    commands: list[list[str]] = []

    class FakeCompletedProcess:
        def __init__(self, returncode=0):
            self.returncode = returncode

    def fake_run(command, cwd, **kwargs):
        commands.append(command)
        return FakeCompletedProcess(0)

    monkeypatch.setattr(build_windows_release.subprocess, "run", fake_run)
    monkeypatch.setattr(
        build_windows_release.sys,
        "argv",
        ["build_windows_release.py", "--sign"],
    )
    monkeypatch.setattr(build_windows_release, "_snapshot_release_artifacts", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        build_windows_release,
        "_detect_built_release_artifacts",
        lambda *_args, **_kwargs: [build_windows_release.ROOT / "release" / "MLBPredictor-Windows-v1.1.0-Setup.exe"],
    )
    monkeypatch.setattr(
        build_windows_release,
        "_write_release_sidecars",
        lambda *_args, **_kwargs: {
            "manifest": build_windows_release.ROOT / "release" / "manifest.json",
            "checksums": build_windows_release.ROOT / "release" / "checksums.txt",
            "release_notes": build_windows_release.ROOT / "release" / "release-notes.md",
        },
    )

    result = build_windows_release.main()

    assert result == 0
    assert "--sign" in commands[1]
    assert "--sign" in commands[2]