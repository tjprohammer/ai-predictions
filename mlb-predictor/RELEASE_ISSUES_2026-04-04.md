# Release Issues - 2026-04-04

Purpose: track the issues found while testing the `v0.2.0-beta1` desktop release.

## Status
- Overall status: open
- Source: live release testing after publishing `v0.2.0-beta1`
- Priority focus: desktop workflow reliability, grading correctness, board usability

## Status Workflow
- `open`: reproduced or reported, no implementation yet
- `in progress`: actively being worked
- `fixed in code`: code and focused regression tests are in place, but app-level verification is still pending
- `verified in app`: confirmed in the running desktop app or packaged flow
- `closed`: fully done and no further follow-up needed

## Open Issues

### 1. Loading bar is not visible
- Status: fixed in code
- Priority: high
- GitHub: #2
- Reported behavior: user does not see the loading/progress bar during long-running actions.
- Expected behavior: desktop actions such as rebuild predictions should show a clearly visible progress/loading state.
- Likely areas:
  - `src/api/static/index.html`
  - update job polling and progress state wiring
- Progress update:
  - progress rail is now shown immediately when an update action starts
  - dashboard shell regression coverage was updated and is passing
- Remaining verification:
  - confirm the progress rail is visually obvious in the live desktop app during a long-running job
- Acceptance criteria:
  - progress/loading state is visible on desktop during long actions
  - progress remains visible until the job completes or fails
  - failed jobs clearly show failure state

### 2. Grade Predictions is not working on mixed-status slates
- Status: closed
- Priority: high
- GitHub: #3
- Reported behavior: `Grade Predictions` does not work correctly when some games are final, some are live, and some have not started.
- Expected behavior: grading should process only games that are eligible to grade and safely ignore live/not-started games.
- Progress update:
  - grading now runs on the selected date instead of a hard-coded yesterday offset
  - product-surface grading now ignores live and pregame rows and only grades final games
  - focused regression coverage is passing
- Verified in app:
  - live desktop-runtime smoke passed `Grade Predictions` against the app on `2026-04-04`
  - the same smoke output showed completed outcome rows and pending rows on the same selected date, which is the mixed-status case this issue was targeting
- Key scenario:
  - same slate contains final + live + pregame games
- Questions to resolve:
  - should grading be incremental for only final games?
  - should already-graded final games be skipped or refreshed?

### 3. Rebuild Predictions does not seem accurate
- Status: verified in app
- Priority: high
- GitHub: #4
- Reported behavior: `Rebuild Predictions` output does not look trustworthy/accurate.
- Expected behavior: rebuild should produce stable, explainable predictions that align with current slate data and market context.
- Investigation areas:
  - model inputs used at rebuild time
  - stale/manual input usage
  - duplicate prediction rows / latest snapshot selection
  - totals/hits/strikeout pipeline consistency
- Progress update:
  - identified and fixed a concrete rebuild bug where selected-date prediction tables kept older model versions alongside newly rebuilt versions
  - live local runtime verification showed mixed-version stacking before the fix:
    - totals `31 rows / 16 distinct games`
    - hits `558 rows / 288 distinct game-player pairs`
    - strikeouts `66 rows / 34 distinct game-pitcher pairs`
  - after the fix, the same runtime now keeps only one selected-date model version per lane:
    - totals `15 rows / 15 distinct games`
    - hits `270 rows / 270 distinct game-player pairs`
    - strikeouts `32 rows / 32 distinct game-pitcher pairs`
  - identified and fixed a desktop-runtime path isolation bug where launcher-loaded repo `.env` values could keep the app reading repo `data/` paths instead of `%LOCALAPPDATA%\MLBPredictor\data`
  - verified live local runtime path resolution now points `DATA_DIR`, `FEATURE_DIR`, `MODEL_DIR`, and `REPORT_DIR` into `%LOCALAPPDATA%\MLBPredictor`
  - identified and fixed a selected-date feature persistence bug where SQLite feature tables kept stale rows for games that had moved off the slate date even after parquet scoring inputs had dropped them
  - after rerunning live `Rebuild Predictions`, runtime feature tables now match runtime prediction tables exactly for `2026-04-04`:
    - totals `15 games`
    - hits `270 game-player pairs`
    - strikeouts `32 game-pitcher pairs`
- Verified in app:
  - live desktop `Rebuild Predictions` completed successfully for `2026-04-04` with no failed steps and no rebuild blocker
  - desktop runtime prediction tables now hold exactly one selected-date model version per lane:
    - totals `15 rows / 15 distinct games / 1 model version`
    - hits `270 rows / 270 distinct game-player pairs / 1 model version`
    - strikeouts `32 rows / 32 distinct game-pitcher pairs / 1 model version`
  - desktop runtime feature tables still match the rebuilt prediction tables exactly for the selected date
  - live board payload after rebuild serves current model versions, populated review summaries, and confirmed lineup-backed hitter targets
  - desktop smoke verification completed successfully against `http://127.0.0.1:8192/` after the rebuild checks
- Conclusion:
  - the concrete rebuild trust issues were caused by data consistency and runtime-path bugs, and those are now verified fixed in the app
  - any remaining user trust concerns are more likely to be model-quality questions than a broken rebuild workflow

### 4. Hitter sorting and recent hit history need improvement
- Status: verified in app
- Priority: medium-high
- GitHub: #5
- Requested improvement:
  - sort hitters by team
  - show last 10 days of hit results
  - green square = hit
  - red square = no hit
  - show dates with those squares
- Expected behavior:
  - hitter board groups or sorts cleanly by team
  - recent 10-game or 10-day hit streak view is easy to scan
- Likely UI/data areas:
  - hitter board payload in `src/api/app.py`
  - dashboard rendering in `src/api/static/index.html`
- Progress update:
  - the hot-hitters API now attaches recent per-player hit history from the latest 10 prior game logs
  - the hot-hitters page now defaults to team sorting and groups cards by team
  - each hitter card now shows a dated hit/miss history strip with green hit tiles and red no-hit tiles
  - focused UI-shell and API helper regressions are passing
- Verified in app:
  - live local verification confirmed the API now returns `recent_hit_history` rows for hitters
  - the running hot-hitters page serves the team-sort default and history-tile markup successfully

### 5. Game board detail should open as a separate page
- Status: verified in app
- Priority: medium-high
- GitHub: #6
- Reported behavior: the main game board feels too dense and overloaded.
- Requested improvement:
  - clicking a game should open a dedicated game detail page
  - lineups and stats should be shown more neatly
  - simplify / declutter the main board slightly
- Expected behavior:
  - main board stays compact and decision-focused
  - game detail has enough room for stats, lineups, and context
- Progress update:
  - the main game board now routes card clicks to a dedicated matchup page
  - the dedicated page is live at `/game` and renders lineups, starters, totals context, and matchup notes
  - focused dashboard shell regressions are passing
- Verified in app:
  - the local desktop runtime serves the matchup page successfully
  - live verification confirmed the board route and detail page after a fresh restart and `Prepare Slate`

### 6. Prepare Slate and Import Inputs are confusing and not working
- Status: closed
- Priority: high
- GitHub: #7
- Reported behavior:
  - `Prepare Slate` and `Import Inputs` buttons do not work
  - naming is confusing
- User feedback:
  - `Import Inputs` may need to be renamed to something like `Update Lineups`
- Current product question:
  - what does `Prepare Slate` do from an operator perspective?
  - what is the difference between preparing slate artifacts and importing manual inputs?
- Follow-up needed:
  - clarify button purpose in UX copy
  - likely rename actions to operator-friendly terms
  - fix broken behavior
- Progress update:
  - `Import Inputs` was renamed in the dashboard to `Update Lineups & Markets`
  - `Prepare Slate` and `Update Lineups & Markets` now rebuild and republish the selected-date board after input refreshes
  - focused update-job and dashboard regressions are passing
- Remaining verification:
  - operator copy can still be improved later, but the workflow now runs successfully in the live desktop runtime
- Verified in app:
  - live desktop-runtime smoke passed both `Prepare Slate` and `Update Lineups & Markets` with zero update-action failures
  - both actions rebuilt and republished selected-date predictions successfully in the app session

### 7. Market sources may need to be more reliable
- Status: open
- Priority: medium
- GitHub: #8
- Reported concern: current market sourcing may not be reliable enough.
- Expected behavior: market lines should come from dependable sources with clear fallback rules.
- Investigation areas:
  - source quality / coverage by market type
  - stale data handling
  - fallback precedence
  - manual overrides vs automated ingest

### 9. Add inning-aware late-game bullpen quality metrics
- Status: open
- Priority: medium
- GitHub: #10
- Requested improvement:
  - track bullpen quality with true inning-aware late-game context instead of only last-3-game aggregates
  - surface late-game runs allowed and related bullpen quality signals more directly
- Current state:
  - totals and matchup pages now show recent bullpen workload plus recent run prevention
  - the current implementation is still based on last-3-game bullpen logs, not inning-by-inning late-game splits
- Likely areas:
  - bullpen ingest/schema expansion
  - totals feature payloads and matchup payloads
  - totals and matchup detail UI

### 10. Matchup page hitter stats and lineup accuracy are not trustworthy
- Status: verified in app
- Priority: high
- GitHub: #11
- Reported problems:
  - matchup detail is missing believable batting averages for players
  - we are missing recent 10 to 15 game hit history on the matchup page
  - recent hitter results should use green squares for hits and red squares for no-hit games
  - total bases and home runs should be visible in that recent-game view
  - season batting average is showing `.000` where that is clearly wrong
  - displayed lineups do not match the MLB app or ESPN closely enough to trust
- Expected behavior:
  - matchup detail should show believable season batting average and recent hitter form
  - recent hitter results should be easy to scan visually
  - lineups should either match trusted sources materially better or be clearly labeled as projected/inferred when they are not confirmed
- Verified progress:
  - season batting averages on matchup detail now load correctly from `player_game_batting`
  - matchup detail now exposes 15-game hit history with hit/no-hit tiles, total bases, and home runs
  - `src.ingestors.lineups` now pulls starter batting orders from MLB StatsAPI `feed/live` and prefers those rows over manual or projected templates for covered teams
  - matchup detail now uses the latest lineup snapshot as the primary player list, with feature and prediction context layered on top when available
- Verified in app:
  - running `Update Lineups & Markets` on the desktop runtime for `2026-04-04` wrote confirmed `mlb_statsapi_lineups` rows into the runtime SQLite database for `game_id = 824295`
  - live `/api/games/824295/detail` now returns confirmed `mlb_statsapi_lineups` players in batting-order sequence for both teams while preserving season BA and 15-game hit history
  - when StatsAPI does not expose a full nine-man order yet, the page still labels the lineup clearly as projected rather than presenting it as confirmed
- Likely areas:
  - matchup detail payload shaping in `src/api/app.py`
  - matchup page rendering in `src/api/static/game.html`
  - hitter season and recent batting joins from `player_game_batting`
  - lineup ingestion and source precedence in `src/ingestors/lineups.py` and slate inputs

### 11. Current-slate weather data is missing or stale on the board
- Status: fixed in code
- Priority: high
- GitHub: #12
- Reported behavior:
  - current-day game readiness can stay yellow with `no_weather` and `weather_source_stale`
  - doctor output can show `has_weather = 0` across the active slate even after running selected-date prep flows
- Expected behavior:
  - selected-date prep and full refresh actions should ingest the current slate's weather context
  - the board and doctor surfaces should show fresh weather coverage for games where weather is expected to be available
- Investigation summary:
  - the live readiness issue was not just stale status text; `game_readiness` rows for the active slate were genuinely missing weather
  - root cause was that `Refresh Everything` and `Prepare Slate` were not invoking `src.ingestors.weather` for the selected date
  - `Refresh Daily Results` and the old grading path already included weather, which is why this only surfaced on current-slate prep flows
- Progress update:
  - `src/api/app.py` update-job sequences now include `src.ingestors.weather --target-date YYYY-MM-DD` in both `Refresh Everything` and `Prepare Slate`
  - targeted regression coverage for update-job sequencing was updated and is passing
  - `doctor --json` was also patched to run cleanly against the desktop SQLite runtime so weather/readiness verification is easier to inspect
- Remaining verification:
  - rerun a live selected-date refresh against the running desktop app and confirm doctor/readiness output no longer reports broad `no_weather` coverage for the slate
  - verify whether any remaining warnings are true upstream weather-source gaps versus local pipeline issues
- Likely areas:
  - `src/api/app.py`
  - `src/ingestors/weather.py`
  - doctor/readiness payloads used by the desktop release verification flow

### 8. Refresh Daily Results is not working on mixed-status slates
- Status: closed
- Priority: high
- GitHub: #9
- Reported behavior: `Refresh Daily Results` is not handling a slate where some games are final, some are live, and some have not started.
- Expected behavior: results refresh should be partial and state-aware.
- Progress update:
  - results refresh now ingests observed weather for the selected date and republishes review surfaces from selected-date data
  - product surfaces now avoid grading live and pregame rows during refresh
  - focused regression coverage is passing
- Verified in app:
  - live desktop-runtime smoke passed `Refresh Daily Results` with zero update-action failures
  - the same selected-date smoke showed a mixed slate with final review rows present alongside pending outcomes, confirming the action now behaves safely on partial slates
- Desired behavior to define:
  - final games: ingest final outcomes
  - live games: update in-progress state without final grading assumptions
  - not-started games: leave untouched
- Likely overlap with issue 2:
  - grading and results refresh need a clear shared game-status model

## Suggested Next Order
1. Visually confirm the loading/progress state in the live desktop UI, then close issue #2 if it is clearly visible.
2. Make the loading/progress state visually stronger if the current live desktop confirmation is still weak.
3. Do a final release-readiness pass across the remaining open issues and desktop workflow before cutting the next build.
4. Fix matchup page hitter stats and lineup accuracy.
5. Visually confirm or further strengthen the desktop loading/progress state.
6. Review market-source reliability and fallback strategy.
7. Add inning-aware late-game bullpen quality metrics.

## Notes
- This file is intentionally user-facing and release-focused, not a low-level engineering dump.
- Keep this updated as issues are reproduced, fixed, or split into smaller tasks.