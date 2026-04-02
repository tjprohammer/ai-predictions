# MLB Predictor Release

## What This Release Is

This release ships the Windows desktop build of MLB Predictor.

## Install

1. Download `MLBPredictor-Windows-PortableInstaller.zip` or `MLBPredictorSetup.exe` if that installer asset is attached.
2. If you downloaded the portable ZIP, extract it and run `install_mlb_predictor.ps1`.
3. Launch `MLBPredictor` from the installed shortcut or install folder.

## First Launch Behavior

On first launch the desktop app will:

- create a per-user runtime folder under `%LOCALAPPDATA%\MLBPredictor`
- create the local SQLite database under `%LOCALAPPDATA%\MLBPredictor\db\mlb_predictor.sqlite3`
- apply bundled schema migrations automatically
- seed bundled park-factor reference data automatically

No Postgres install is required for normal desktop startup.

## Daily Data Refreshes

Use the in-app Update Center for day-to-day workflow:

- `Prepare slate` refreshes schedule, starters, and editable templates
- `Import manual inputs` reads edited lineup and market files
- `Refresh results` pulls completed-game data and refreshes form tables
- `Rebuild predictions` rescoring totals, first-five totals, hitter cards, strikeout cards, and published surfaces

You do not need to download a new GitHub release every day to refresh slate data.

## App Upgrades

This release does not include a desktop auto-updater yet.

To upgrade the app itself, download the next GitHub Release and install it over the existing app. Local user data should remain intact unless `%LOCALAPPDATA%\MLBPredictor` is removed manually.

## Highlights

- Replace this section with the release-specific changes.
- Call out any migration, packaging, UI, or prediction-surface changes.
- Mention any known limitations testers should expect.