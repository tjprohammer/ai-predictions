from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


SIGNABLE_EXTENSIONS = {".exe", ".dll"}
DEFAULT_TIMESTAMP_URL = "http://timestamp.digicert.com"
DEFAULT_DIGEST_ALGORITHM = "SHA256"


class SigningConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class SigningConfig:
    signtool_path: Path
    timestamp_url: str = DEFAULT_TIMESTAMP_URL
    digest_algorithm: str = DEFAULT_DIGEST_ALGORITHM
    certificate_file: Path | None = None
    certificate_password: str | None = None
    certificate_subject: str | None = None
    certificate_thumbprint: str | None = None


def find_signtool() -> Path | None:
    candidates: list[Path] = []
    for env_name in ("WINDOWS_SIGNTOOL_PATH", "SIGNTOOL_PATH"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(Path(env_value))

    which_match = shutil.which("signtool.exe")
    if which_match:
        candidates.append(Path(which_match))

    sdk_roots = [
        Path(r"C:\Program Files (x86)\Windows Kits\10\bin"),
        Path(r"C:\Program Files\Windows Kits\10\bin"),
    ]
    for sdk_root in sdk_roots:
        if not sdk_root.exists():
            continue
        candidates.extend(sorted(sdk_root.glob("**/x64/signtool.exe"), reverse=True))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_signing_config(enable_signing: bool) -> SigningConfig | None:
    if not enable_signing:
        return None

    signtool_path = find_signtool()
    if signtool_path is None:
        raise SigningConfigurationError(
            "Windows signing was requested, but signtool.exe was not found. "
            "Set WINDOWS_SIGNTOOL_PATH or install the Windows SDK."
        )

    certificate_file_value = os.environ.get("WINDOWS_SIGN_CERT_FILE")
    certificate_file = Path(certificate_file_value).expanduser() if certificate_file_value else None
    certificate_subject = (os.environ.get("WINDOWS_SIGN_CERT_SUBJECT") or "").strip() or None
    certificate_thumbprint = (os.environ.get("WINDOWS_SIGN_CERT_SHA1") or "").strip() or None
    certificate_password = os.environ.get("WINDOWS_SIGN_CERT_PASSWORD") or None

    if certificate_file is not None and not certificate_file.exists():
        raise SigningConfigurationError(f"Windows signing certificate file was not found: {certificate_file}")

    if certificate_file is None and certificate_subject is None and certificate_thumbprint is None:
        raise SigningConfigurationError(
            "Windows signing was requested, but no certificate selector was configured. "
            "Set WINDOWS_SIGN_CERT_FILE, WINDOWS_SIGN_CERT_SUBJECT, or WINDOWS_SIGN_CERT_SHA1."
        )

    timestamp_url = (os.environ.get("WINDOWS_SIGN_TIMESTAMP_URL") or DEFAULT_TIMESTAMP_URL).strip()
    digest_algorithm = (os.environ.get("WINDOWS_SIGN_DIGEST_ALGORITHM") or DEFAULT_DIGEST_ALGORITHM).strip().upper()
    return SigningConfig(
        signtool_path=signtool_path,
        timestamp_url=timestamp_url,
        digest_algorithm=digest_algorithm,
        certificate_file=certificate_file,
        certificate_password=certificate_password,
        certificate_subject=certificate_subject,
        certificate_thumbprint=certificate_thumbprint,
    )


def discover_signable_files(path: Path) -> list[Path]:
    target = Path(path)
    if target.is_file():
        return [target] if target.suffix.lower() in SIGNABLE_EXTENSIONS else []
    if not target.exists():
        return []
    return sorted(
        candidate
        for candidate in target.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in SIGNABLE_EXTENSIONS
    )


def sign_file(path: Path, signing_config: SigningConfig) -> None:
    target = Path(path)
    if not target.exists():
        raise SigningConfigurationError(f"Cannot sign missing file: {target}")

    command = [
        str(signing_config.signtool_path),
        "sign",
        "/fd",
        signing_config.digest_algorithm,
        "/td",
        signing_config.digest_algorithm,
        "/tr",
        signing_config.timestamp_url,
    ]

    if signing_config.certificate_file is not None:
        command.extend(["/f", str(signing_config.certificate_file)])
    if signing_config.certificate_password is not None:
        command.extend(["/p", signing_config.certificate_password])
    if signing_config.certificate_subject is not None:
        command.extend(["/n", signing_config.certificate_subject])
    if signing_config.certificate_thumbprint is not None:
        command.extend(["/sha1", signing_config.certificate_thumbprint])
    if signing_config.certificate_subject is None and signing_config.certificate_thumbprint is None:
        command.append("/a")

    command.append(str(target))
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise SigningConfigurationError(f"signtool failed for {target} with exit code {completed.returncode}")


def sign_files(paths: list[Path], signing_config: SigningConfig) -> list[Path]:
    signed_paths: list[Path] = []
    for candidate in sorted({Path(path) for path in paths}):
        sign_file(candidate, signing_config)
        signed_paths.append(candidate)
    return signed_paths