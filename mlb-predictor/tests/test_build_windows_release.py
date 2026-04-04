import scripts.build_windows_release as build_windows_release


def test_build_windows_release_builds_seed_before_packaging(monkeypatch):
    commands: list[list[str]] = []

    class FakeCompletedProcess:
        def __init__(self, returncode=0):
            self.returncode = returncode

    def fake_run(command, cwd):
        commands.append(command)
        return FakeCompletedProcess(0)

    monkeypatch.setattr(build_windows_release.subprocess, "run", fake_run)
    monkeypatch.setattr(build_windows_release.sys, "argv", ["build_windows_release.py"])

    result = build_windows_release.main()

    assert result == 0
    assert commands[0][1] == "scripts/build_desktop_sqlite_seed.py"
    assert commands[1][1] == "scripts/build_windows_app.py"
    assert commands[2][1] == "scripts/build_windows_installer.py"


def test_build_windows_release_stops_when_seed_build_fails(monkeypatch):
    commands: list[list[str]] = []

    class FakeCompletedProcess:
        def __init__(self, returncode=0):
            self.returncode = returncode

    def fake_run(command, cwd):
        commands.append(command)
        if command[1] == "scripts/build_desktop_sqlite_seed.py":
            return FakeCompletedProcess(2)
        return FakeCompletedProcess(0)

    monkeypatch.setattr(build_windows_release.subprocess, "run", fake_run)
    monkeypatch.setattr(build_windows_release.sys, "argv", ["build_windows_release.py"])

    result = build_windows_release.main()

    assert result == 2
    assert len(commands) == 1


def test_build_windows_release_passes_require_inno_to_installer(monkeypatch):
    commands: list[list[str]] = []

    class FakeCompletedProcess:
        def __init__(self, returncode=0):
            self.returncode = returncode

    def fake_run(command, cwd):
        commands.append(command)
        return FakeCompletedProcess(0)

    monkeypatch.setattr(build_windows_release.subprocess, "run", fake_run)
    monkeypatch.setattr(
        build_windows_release.sys,
        "argv",
        ["build_windows_release.py", "--require-inno"],
    )

    result = build_windows_release.main()

    assert result == 0
    assert commands[-1][-1] == "--require-inno"


def test_build_windows_release_passes_app_version_to_installer(monkeypatch):
    commands: list[list[str]] = []

    class FakeCompletedProcess:
        def __init__(self, returncode=0):
            self.returncode = returncode

    def fake_run(command, cwd):
        commands.append(command)
        return FakeCompletedProcess(0)

    monkeypatch.setattr(build_windows_release.subprocess, "run", fake_run)
    monkeypatch.setattr(
        build_windows_release.sys,
        "argv",
        ["build_windows_release.py", "--app-version", "0.2.0-beta1"],
    )

    result = build_windows_release.main()

    assert result == 0
    assert "--app-version" in commands[-1]
    assert "0.2.0-beta1" in commands[-1]