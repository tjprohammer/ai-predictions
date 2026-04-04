## MLB Predictor Beta Release Checklist

### 1. Pick the beta version

- [ ] Choose the next version label, e.g. `0.1.0-beta1`
- [ ] Confirm the target branch/commit is what you want to ship
- [ ] Write a short beta summary: what's new, known caveats, and who should test it

### 2. Verify the app before packaging

- [ ] Run the full test suite
- [ ] Run a local API smoke test for board, scorecards, review, and CLV
- [ ] Run `Refresh Everything` once against a realistic target date
- [ ] Confirm the desktop bundle still launches on a clean runtime

Suggested commands:

```powershell
cd S:\Projects\AI_Predictions\mlb-predictor
python -m pytest tests/ -q
python scripts\build_windows_release.py --app-version 0.1.0-beta1
```

### 3. Build the release artifacts

If Inno Setup 6 is installed and you want a real installer:

```powershell
cd S:\Projects\AI_Predictions\mlb-predictor
python scripts\build_windows_release.py --app-version 0.1.0-beta1 --require-inno
```

If Inno Setup is not installed yet, build the portable beta bundle:

```powershell
cd S:\Projects\AI_Predictions\mlb-predictor
python scripts\build_windows_installer.py --app-version 0.1.0-beta1 --portable-only
```

Expected artifact patterns:

- `release\MLBPredictor-Windows-v0.1.0-beta1-Setup.exe`
- `release\MLBPredictor-Windows-v0.1.0-beta1-PortableInstaller.zip`

### 4. Smoke-test the packaged output

- [ ] Install or extract the built artifact
- [ ] Launch the packaged app on a clean `%LOCALAPPDATA%` runtime
- [ ] Verify the board loads real games
- [ ] Verify scorecards, top misses, and CLV review load
- [ ] Verify `Refresh Everything` completes successfully in-app

### 5. Publish the beta

- [ ] Create or update the GitHub Release draft
- [ ] Upload the versioned Windows asset(s)
- [ ] Add release notes with:
  - [ ] highlights
  - [ ] tester instructions
  - [ ] known limitations
  - [ ] whether the asset is portable ZIP or full `Setup.exe`

### 6. Post-publish follow-up

- [ ] Share the GitHub Release link with testers
- [ ] Ask testers to report install issues, update issues, and layout issues separately
- [ ] Track any first-run failures, empty panels, or refresh-job failures
- [ ] Capture whether testers prefer portable ZIP or `Setup.exe`