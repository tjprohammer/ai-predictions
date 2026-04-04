import scripts.build_windows_app as build_windows_app


def test_build_command_collects_dynamic_pipeline_packages(monkeypatch):
    captured = {}

    class FakeCompletedProcess:
        returncode = 0

    def fake_run(command, cwd):
        captured["command"] = command
        captured["cwd"] = cwd
        return FakeCompletedProcess()

    monkeypatch.setattr(build_windows_app, "_validate_sqlite_seed", lambda _path: None)
    monkeypatch.setattr(build_windows_app.subprocess, "run", fake_run)
    monkeypatch.setattr(build_windows_app.sys, "argv", ["build_windows_app.py"])

    result = build_windows_app.main()

    assert result == 0
    command = captured["command"]
    collected_packages = [
        command[index + 1]
        for index in range(len(command) - 1)
        if command[index] == "--collect-submodules"
    ]
    hidden_imports = [
        command[index + 1]
        for index in range(len(command) - 1)
        if command[index] == "--hidden-import"
    ]
    add_data_args = [
        command[index + 1]
        for index in range(len(command) - 1)
        if command[index] == "--add-data"
    ]
    for package_name in build_windows_app.DYNAMIC_PACKAGES:
        assert package_name in collected_packages
    for module_name in build_windows_app.DYNAMIC_MODULES:
        assert module_name in hidden_imports
    assert build_windows_app.add_data_argument(build_windows_app.ROOT / "src", "src") in add_data_args


def test_build_command_fails_when_sqlite_seed_is_incomplete(monkeypatch):
    called = {"run": False}

    monkeypatch.setattr(
        build_windows_app,
        "_validate_sqlite_seed",
        lambda _path: "Bundled desktop SQLite seed is incomplete.",
    )
    monkeypatch.setattr(
        build_windows_app.subprocess,
        "run",
        lambda *args, **kwargs: called.__setitem__("run", True),
    )
    monkeypatch.setattr(build_windows_app.sys, "argv", ["build_windows_app.py"])

    result = build_windows_app.main()

    assert result == 1
    assert called["run"] is False


def test_build_command_allows_override_for_incomplete_sqlite_seed(monkeypatch):
    captured = {}

    class FakeCompletedProcess:
        returncode = 0

    monkeypatch.setattr(
        build_windows_app,
        "_validate_sqlite_seed",
        lambda _path: "Bundled desktop SQLite seed is incomplete.",
    )
    monkeypatch.setattr(
        build_windows_app.subprocess,
        "run",
        lambda command, cwd: captured.update({"command": command, "cwd": cwd}) or FakeCompletedProcess(),
    )
    monkeypatch.setattr(
        build_windows_app.sys,
        "argv",
        ["build_windows_app.py", "--allow-incomplete-sqlite-seed"],
    )

    result = build_windows_app.main()

    assert result == 0
    assert captured["cwd"] == build_windows_app.ROOT