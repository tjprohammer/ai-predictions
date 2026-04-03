import scripts.build_windows_app as build_windows_app


def test_build_command_collects_dynamic_pipeline_packages(monkeypatch):
    captured = {}

    class FakeCompletedProcess:
        returncode = 0

    def fake_run(command, cwd):
        captured["command"] = command
        captured["cwd"] = cwd
        return FakeCompletedProcess()

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