## MLB Predictor Beta Release Checklist

### 1. Pick the beta version

- [ ] Choose the next version label, e.g. `0.1.0-beta1`
- [ ] Confirm the target branch/commit is what you want to ship
- [ ] Write a short beta summary: what's new, known caveats, and who should test it

### 2. Verify the app before packaging

- [ ] Run the full test suite
- [ ] Run a local API smoke test for board, scorecards, review, and CLV
- [ ] Run `python scripts\run_desktop_smoke.py --json` and confirm the doctor status is not `error`
- [ ] Run `python scripts\run_desktop_smoke.py --json --exercise-update-actions` and confirm all six Update Center actions return `ok: true`
- [ ] Run `Refresh Everything` once against a realistic target date
- [ ] Confirm the desktop bundle still launches on a clean runtime

Suggested commands:

```powershell
cd S:\Projects\AI_Predictions\mlb-predictor
python -m pytest tests/ -q
python -m src.utils.doctor --json
python scripts\run_desktop_smoke.py --json
python scripts\run_desktop_smoke.py --json --exercise-update-actions
python scripts\build_windows_release.py --app-version 0.1.0-beta1
```

### 3. Build the release artifacts

If Inno Setup 6 is installed and you want a real installer:

```powershell
cd S:\Projects\AI_Predictions\mlb-predictor
python scripts\build_windows_release.py --app-version 0.1.0-beta1 --require-inno
```

If you have a code-signing certificate configured and want a signed installer/app build:

```powershell
$env:WINDOWS_SIGN_CERT_FILE = 'C:\path\to\certificate.pfx'
$env:WINDOWS_SIGN_CERT_PASSWORD = '***'
$env:WINDOWS_SIGN_TIMESTAMP_URL = 'http://timestamp.digicert.com'
python scripts\build_windows_release.py --app-version 0.1.0-beta1 --require-inno --sign
```

Notes:

- `--sign` uses `signtool.exe` plus `WINDOWS_SIGN_*` environment variables
- you can use `WINDOWS_SIGN_CERT_SUBJECT` or `WINDOWS_SIGN_CERT_SHA1` instead of `WINDOWS_SIGN_CERT_FILE` when the certificate already exists in the Windows certificate store
- signing the installer and app executable is the primary fix for the Windows SmartScreen `unrecognized app` warning shown to testers

If Inno Setup is not installed yet, build the portable beta bundle:

```powershell
cd S:\Projects\AI_Predictions\mlb-predictor
python scripts\build_windows_installer.py --app-version 0.1.0-beta1 --portable-only
```

Expected artifact patterns:

- `release\MLBPredictor-Windows-v0.1.0-beta1-Setup.exe`
- `release\MLBPredictor-Windows-v0.1.0-beta1-PortableInstaller.zip`
- `release\MLBPredictor-Windows-v0.1.0-beta1-manifest.json`
- `release\MLBPredictor-Windows-v0.1.0-beta1-checksums.txt`
- `release\MLBPredictor-Windows-v0.1.0-beta1-release-notes.md`

### 4. Smoke-test the packaged output

- [ ] Install or extract the built artifact
- [ ] Launch the packaged app on a clean `%LOCALAPPDATA%` runtime
- [ ] Open `/doctor` and confirm runtime checks, recent jobs, and readiness render
- [ ] Verify the board loads real games
- [ ] Verify scorecards, top misses, and CLV review load
- [ ] Verify the in-app loading bar advances while an Update Center action runs
- [ ] Verify `Refresh Everything` completes successfully in-app

### 5. Publish the beta

- [ ] Create or update the GitHub Release draft
- [ ] Upload the versioned Windows asset(s)
- [ ] Upload or paste the generated checksums and manifest details
- [ ] Add release notes with:
  - [ ] highlights
  - [ ] tester instructions
  - [ ] known limitations
  - [ ] whether the asset is portable ZIP or full `Setup.exe`
  - [ ] update the generated release-notes draft instead of writing from scratch

### 6. Post-publish follow-up

- [ ] Share the GitHub Release link with testers
- [ ] Ask testers to report install issues, update issues, and layout issues separately
- [ ] Track any first-run failures, empty panels, or refresh-job failures
- [ ] Capture whether testers prefer portable ZIP or `Setup.exe`
