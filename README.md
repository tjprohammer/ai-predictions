# AI Predictions

This repository currently ships the MLB desktop app from the [mlb-predictor](mlb-predictor/README.md) project folder.

## Quick Start For Testers

The best way to test the packaged app is from GitHub Releases, not from the source-code download on the repository home page.

1. Open the latest release for this repository.
2. Download `MLBPredictor-Windows-PortableInstaller.zip` or `MLBPredictorSetup.exe` when that installer asset is published.
3. If you downloaded the portable ZIP, extract it and run `install_mlb_predictor.ps1`.
4. Launch `MLBPredictor` from the shortcut or install folder.

What happens on first launch:

- the app creates its runtime folder under `%LOCALAPPDATA%\MLBPredictor`
- the app creates a local SQLite database under `%LOCALAPPDATA%\MLBPredictor\db\mlb_predictor.sqlite3`
- bundled schema migrations run automatically
- bundled park-factor reference data is seeded automatically

That means testers do not need Postgres or manual database setup just to open the Windows app.

## Daily Updates vs App Updates

These are separate workflows.

- Daily data updates happen inside the app through the Update Center. Use that to prepare the slate, import edited inputs, refresh results, and rebuild predictions.
- App updates happen by downloading a newer GitHub Release and installing it over the existing app.

There is no desktop auto-updater yet. New binaries are still delivered as manual release downloads.

Because the installed binaries and the per-user SQLite data live in different locations, installing a newer release should preserve local app data unless `%LOCALAPPDATA%\MLBPredictor` is deleted manually.

## Repository Layout

- [mlb-predictor](mlb-predictor) contains the current MLB product, API, desktop packaging, data pipeline, tests, and release scripts.
- [.github/RELEASE_TEMPLATE.md](.github/RELEASE_TEMPLATE.md) contains the standard GitHub release notes template for desktop builds.

## Developer Docs

For local setup, pipeline commands, packaging, and desktop runtime details, start with [mlb-predictor/README.md](mlb-predictor/README.md).