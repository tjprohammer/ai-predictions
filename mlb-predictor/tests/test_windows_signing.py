from pathlib import Path

import pytest

import scripts.windows_signing as windows_signing


def test_resolve_signing_config_requires_certificate_selector(monkeypatch, tmp_path):
    signtool = tmp_path / "signtool.exe"
    signtool.write_text("exe", encoding="utf-8")

    monkeypatch.setattr(windows_signing, "find_signtool", lambda: signtool)
    monkeypatch.delenv("WINDOWS_SIGN_CERT_FILE", raising=False)
    monkeypatch.delenv("WINDOWS_SIGN_CERT_SUBJECT", raising=False)
    monkeypatch.delenv("WINDOWS_SIGN_CERT_SHA1", raising=False)

    with pytest.raises(windows_signing.SigningConfigurationError):
        windows_signing.resolve_signing_config(True)


def test_sign_file_builds_signtool_command(monkeypatch, tmp_path):
    target = tmp_path / "MLBPredictor.exe"
    target.write_text("exe", encoding="utf-8")
    cert = tmp_path / "certificate.pfx"
    cert.write_text("cert", encoding="utf-8")
    captured = {}

    class FakeCompletedProcess:
        def __init__(self, returncode=0):
            self.returncode = returncode

    monkeypatch.setattr(
        windows_signing.subprocess,
        "run",
        lambda command: captured.update({"command": command}) or FakeCompletedProcess(0),
    )

    config = windows_signing.SigningConfig(
        signtool_path=tmp_path / "signtool.exe",
        certificate_file=cert,
        certificate_password="secret",
        timestamp_url="http://timestamp.example.test",
    )
    windows_signing.sign_file(target, config)

    assert captured["command"][-1] == str(target)
    assert "/f" in captured["command"]
    assert str(cert) in captured["command"]
    assert "/tr" in captured["command"]
    assert "http://timestamp.example.test" in captured["command"]