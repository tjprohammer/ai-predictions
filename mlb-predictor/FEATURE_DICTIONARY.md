# MLB Feature Dictionary

## Purpose

This document is the working feature contract for the `mlb-predictor` rework.

It has two jobs:

- inventory the current live feature sets that already exist in code
- define the contract every existing or new feature must satisfy before it can be trusted in totals, first-five totals, hits, strikeouts, or certainty scoring

This file is intentionally seeded from the current codebase so Phase 1 can start from real builders and contract lists instead of a blank page.

## Status

This is the initial execution artifact, not the finished audit.

- exact current feature names are pulled from `src/features/contracts.py`
- current builder modules and source tables are pulled from the existing feature builders
- exact formulas, missing-data fallbacks, and leakage review still need to be completed during Phase 1

Use the status values below while filling this in:

- `seeded` means the feature exists in current code and the dictionary has been initialized
- `verified` means formula, source, and timing have been audited against code and data
- `needs-audit` means the feature exists but the contract is still incomplete
- `planned` means the feature is part of the rework but does not exist yet

## Audit Outcomes

Use the following audit outcomes during Phase 1:

- `keep`: trusted and retained in the core model
- `downweight`: still useful, but should matter less than stronger process features
- `calibration-only`: useful for disagreement handling, confidence, or post-model calibration, not the raw model core
- `product-only`: useful for UI, explanations, or sorting, not model training
- `retire`: noisy, redundant, stale, or leakage-prone
- `planned`: rework target that does not exist yet
- `replace`: temporary current feature that should be swapped for a stronger certainty-aware version

These decisions are initial directional calls, not final verdicts. Phase 1 should confirm or change them as formulas, fallback behavior, and leakage risk are verified.

## Field Role Tags

In addition to the audit outcome above, each field should eventually be classified into one implementation role:

- `core predictor`: belongs in the raw model feature set
- `calibration input`: belongs after the raw model for calibration, disagreement handling, or edge interpretation
- `certainty signal`: belongs in trust, publish, downgrade, or suppress logic
- `diagnostic flag`: belongs in audit, debugging, and explanation support, not the core model
- `product-only field`: belongs in UI, labels, explanations, or sorting, not model training

## Required Contract Fields

Every feature should define the following fields before it is considered production-trusted:

- feature name
- market lane
- bucket: `skill`, `availability`, or `volatility`
- builder module
- destination feature table
- primary source table or API
- exact formula
- refresh cadence
- earliest safe board state
- pregame-safe flag
- missing-data fallback
- leakage risk
- field role
- downstream consumers
- action
- priority
- notes
- status

## Board-State Tags

Use these minimum-safe board states when auditing a feature:

- `early` for night-before or very early same-day usage
- `morning` for same-day usage before lineups confirm
- `lineup_confirmed` for official batting-order usage
- `pre_lock` for near-first-pitch usage

## Canonical Feature-Set Map

| Lane | Builder module | Feature table | Prediction table | Current contract symbol | Current target |
| --- | --- | --- | --- | --- | --- |
| Full-game totals | `src.features.totals_builder` | `game_features_totals` | `predictions_totals` | `TOTALS_FEATURE_COLUMNS` | `actual_total_runs` |
| First-five totals | `src.features.first5_totals_builder` | `game_features_first5_totals` | `predictions_first5_totals` | `FIRST5_TOTALS_FEATURE_COLUMNS` | `actual_total_runs_first5` |
| Player hits | `src.features.hits_builder` | `player_features_hits` | `predictions_player_hits` | `HITS_FEATURE_COLUMNS` | `got_hit` |
| Pitcher strikeouts | `src.features.strikeouts_builder` | `game_features_pitcher_strikeouts` | `predictions_pitcher_strikeouts` | `STRIKEOUTS_FEATURE_COLUMNS` | `actual_strikeouts` |

## Source Systems In Current Builders

These are the main inputs already used by the existing feature builders:

- `games`
- `team_offense_daily`
- `bullpens_daily`
- `pitcher_starts`
- `lineups`
- `player_game_batting`
- `game_weather`
- `game_markets`
- `player_prop_markets`
- `market_selection_freezes`
- `park_factors`
- `dim_players`

## Current Live Feature Inventory

The exact feature names below are the current contract columns from `src/features/contracts.py`. Phase 1 should add formulas, timing rules, and fallback rules row by row.

### Full-Game Totals

#### Metadata Columns

- `game_id`
- `game_date`
- `home_team`
- `away_team`
- `market_sportsbook`
- `prediction_ts`
- `game_start_ts`
- `line_snapshot_ts`
- `feature_cutoff_ts`
- `feature_version`

#### Feature Columns

| Feature | Bucket | Primary source(s) | Earliest safe state | Action | Priority | Downstream consumers | Notes | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `home_runs_rate_blended` | `skill` | `team_offense_daily`, priors | `early` | `keep` | `high` | totals model, totals API surfaces | Core team scoring baseline and likely central to expected-runs modeling. | `seeded` |
| `away_runs_rate_blended` | `skill` | `team_offense_daily`, priors | `early` | `keep` | `high` | totals model, totals API surfaces | Core team scoring baseline and likely central to expected-runs modeling. | `seeded` |
| `home_hits_rate_blended` | `skill` | `team_offense_daily`, priors | `early` | `downweight` | `low` | totals model | Useful support signal, but clearly secondary to run creation, lineup quality, and pitching context. | `seeded` |
| `away_hits_rate_blended` | `skill` | `team_offense_daily`, priors | `early` | `downweight` | `low` | totals model | Useful support signal, but clearly secondary to run creation, lineup quality, and pitching context. | `seeded` |
| `home_xwoba_blended` | `skill` | `team_offense_daily`, priors | `early` | `keep` | `high` | totals model | Strong process metric for team offense quality and worth keeping central. | `seeded` |
| `away_xwoba_blended` | `skill` | `team_offense_daily`, priors | `early` | `keep` | `high` | totals model | Strong process metric for team offense quality and worth keeping central. | `seeded` |
| `home_iso_blended` | `skill` | `team_offense_daily`, priors | `early` | `keep` | `medium` | totals model | Power context matters, but should sit below broader run-creation indicators and lineup strength. | `seeded` |
| `away_iso_blended` | `skill` | `team_offense_daily`, priors | `early` | `keep` | `medium` | totals model | Power context matters, but should sit below broader run-creation indicators and lineup strength. | `seeded` |
| `home_bb_pct_blended` | `skill` | `team_offense_daily`, priors | `early` | `downweight` | `low` | totals model | Plate-discipline context helps, but in the current totals lane it looks more supportive than central. | `seeded` |
| `away_bb_pct_blended` | `skill` | `team_offense_daily`, priors | `early` | `downweight` | `low` | totals model | Plate-discipline context helps, but in the current totals lane it looks more supportive than central. | `seeded` |
| `home_k_pct_blended` | `skill` | `team_offense_daily`, priors | `early` | `keep` | `medium` | totals model, product surfaces | Useful mostly through starter-versus-lineup interaction rather than as a standalone totals driver. | `seeded` |
| `away_k_pct_blended` | `skill` | `team_offense_daily`, priors | `early` | `keep` | `medium` | totals model, product surfaces | Useful mostly through starter-versus-lineup interaction rather than as a standalone totals driver. | `seeded` |
| `home_starter_xwoba_blended` | `skill` | `pitcher_starts`, priors | `early` | `keep` | `high` | totals model | Core starter skill input and should remain central. | `seeded` |
| `away_starter_xwoba_blended` | `skill` | `pitcher_starts`, priors | `early` | `keep` | `high` | totals model | Core starter skill input and should remain central. | `seeded` |
| `home_starter_csw_blended` | `skill` | `pitcher_starts`, priors | `early` | `keep` | `high` | totals model | Strong process metric for starter bat-missing and command quality. | `seeded` |
| `away_starter_csw_blended` | `skill` | `pitcher_starts`, priors | `early` | `keep` | `high` | totals model | Strong process metric for starter bat-missing and command quality. | `seeded` |
| `home_starter_rest_days` | `availability` | `pitcher_starts` | `early` | `keep` | `high` | totals model | Useful workload and readiness proxy that should survive into certainty-aware modeling. | `seeded` |
| `away_starter_rest_days` | `availability` | `pitcher_starts` | `early` | `keep` | `high` | totals model | Useful workload and readiness proxy that should survive into certainty-aware modeling. | `seeded` |
| `home_bullpen_pitches_last3` | `availability` | `bullpens_daily` | `early` | `keep` | `high` | totals model | One of the strongest bullpen-side features because it measures actual short-term workload rather than noisy outcomes. | `seeded` |
| `away_bullpen_pitches_last3` | `availability` | `bullpens_daily` | `early` | `keep` | `high` | totals model | One of the strongest bullpen-side features because it measures actual short-term workload rather than noisy outcomes. | `seeded` |
| `home_bullpen_innings_last3` | `availability` | `bullpens_daily` | `early` | `keep` | `high` | totals model | One of the strongest bullpen-side features because it measures actual short-term workload rather than noisy outcomes. | `seeded` |
| `away_bullpen_innings_last3` | `availability` | `bullpens_daily` | `early` | `keep` | `high` | totals model | One of the strongest bullpen-side features because it measures actual short-term workload rather than noisy outcomes. | `seeded` |
| `home_bullpen_b2b` | `availability` | `bullpens_daily` | `early` | `keep` | `high` | totals model | Back-to-back usage is a clean availability flag and should be treated as part of the core bullpen fatigue context. | `seeded` |
| `away_bullpen_b2b` | `availability` | `bullpens_daily` | `early` | `keep` | `high` | totals model | Back-to-back usage is a clean availability flag and should be treated as part of the core bullpen fatigue context. | `seeded` |
| `home_bullpen_runs_allowed_last3` | `volatility` | `bullpens_daily` | `early` | `downweight` | `low` | totals model | Short-run bullpen outcomes are noisy and likely weaker than workload or leverage availability. | `seeded` |
| `away_bullpen_runs_allowed_last3` | `volatility` | `bullpens_daily` | `early` | `downweight` | `low` | totals model | Short-run bullpen outcomes are noisy and likely weaker than workload or leverage availability. | `seeded` |
| `home_bullpen_earned_runs_last3` | `volatility` | `bullpens_daily` | `early` | `downweight` | `low` | totals model | Short-run bullpen outcomes are noisy and likely weaker than workload or leverage availability. | `seeded` |
| `away_bullpen_earned_runs_last3` | `volatility` | `bullpens_daily` | `early` | `downweight` | `low` | totals model | Short-run bullpen outcomes are noisy and likely weaker than workload or leverage availability. | `seeded` |
| `home_bullpen_hits_allowed_last3` | `volatility` | `bullpens_daily` | `early` | `downweight` | `low` | totals model | Outcome-based bullpen signals should not outrank workload and baseline talent. | `seeded` |
| `away_bullpen_hits_allowed_last3` | `volatility` | `bullpens_daily` | `early` | `downweight` | `low` | totals model | Outcome-based bullpen signals should not outrank workload and baseline talent. | `seeded` |
| `home_bullpen_era_last3` | `volatility` | `bullpens_daily` | `early` | `retire` | `low` | totals model | This is a high-noise summary and a leading retirement candidate. | `seeded` |
| `away_bullpen_era_last3` | `volatility` | `bullpens_daily` | `early` | `retire` | `low` | totals model | This is a high-noise summary and a leading retirement candidate. | `seeded` |
| `home_lineup_top5_xwoba` | `availability` | `lineups`, `player_game_batting`, priors | `morning` | `keep` | `high` | totals model, board surfaces | This now looks like one of the clearest bridges between team baseline offense and actual same-slate run environment. | `seeded` |
| `away_lineup_top5_xwoba` | `availability` | `lineups`, `player_game_batting`, priors | `morning` | `keep` | `high` | totals model, board surfaces | This now looks like one of the clearest bridges between team baseline offense and actual same-slate run environment. | `seeded` |
| `home_lineup_k_pct` | `availability` | `lineups`, `player_game_batting` | `morning` | `keep` | `high` | totals model, board surfaces | Good lineup-quality input once timing rules are audited. | `seeded` |
| `away_lineup_k_pct` | `availability` | `lineups`, `player_game_batting` | `morning` | `keep` | `high` | totals model, board surfaces | Good lineup-quality input once timing rules are audited. | `seeded` |
| `venue_run_factor` | `skill` | `park_factors` | `early` | `keep` | `medium` | totals model, board surfaces | Stable environment baseline still matters, but the audited builder suggests starter, lineup, and bullpen workload should sit ahead of it. | `seeded` |
| `venue_hr_factor` | `skill` | `park_factors` | `early` | `keep` | `medium` | totals model, board surfaces | Useful park context, especially for power-driven totals environments. | `seeded` |
| `temperature_f` | `volatility` | `game_weather` | `early` | `keep` | `medium` | totals model, board surfaces | Weather belongs in the numeric model, but the current audited lane makes it more of a support input than a lead driver. | `seeded` |
| `wind_speed_mph` | `volatility` | `game_weather` | `early` | `keep` | `medium` | totals model, board surfaces | Weather belongs in the numeric model, but the current audited lane makes it more of a support input than a lead driver. | `seeded` |
| `wind_direction_deg` | `volatility` | `game_weather` | `early` | `downweight` | `low` | totals model, board surfaces | Keep only as secondary directionality unless the encoding proves clearly additive in testing. | `seeded` |
| `humidity_pct` | `volatility` | `game_weather` | `early` | `downweight` | `low` | totals model | Secondary weather context that should sit below temperature and wind. | `seeded` |
| `market_total` | `volatility` | `game_markets` | `early` | `calibration-only` | `high` | totals model, calibration, product surfaces | Important calibration input, but the rework should keep it out of the core raw model logic. | `seeded` |
| `market_over_price` | `volatility` | `game_markets` | `early` | `calibration-only` | `low` | totals model, calibration | Useful pricing context, but less important than total level and movement for this lane. | `seeded` |
| `market_under_price` | `volatility` | `game_markets` | `early` | `calibration-only` | `low` | totals model, calibration | Useful pricing context, but less important than total level and movement for this lane. | `seeded` |
| `line_movement` | `volatility` | `game_markets`, `market_selection_freezes` | `morning` | `calibration-only` | `high` | totals model, calibration, explanations | Better used for disagreement and freshness handling than as a primary model driver. | `seeded` |

#### Code-Audited Formula And Fallback Notes

This pass audits the current full-game totals formulas against `src/features/totals_builder.py`, `src/features/common.py`, and `src/features/priors.py`.

The rows above still retain their current status labels for now, but the feature families below now have explicit code-level formulas, cadence notes, fallback behavior, and leakage notes recorded.

After the code audit, the full-game totals lane should be treated as a run-environment model built from team scoring baselines, starter quality, lineup quality, and bullpen availability or workload first, with park, weather, and market-price fields kept in supporting or calibration roles.

| Feature family | Applies to | Exact current formula | Refresh cadence | Pregame-safe | Missing-data fallback | Leakage risk |
| --- | --- | --- | --- | --- | --- | --- |
| Team offense blended rates | `home_runs_rate_blended`, `away_runs_rate_blended`, `home_hits_rate_blended`, `away_hits_rate_blended`, `home_xwoba_blended`, `away_xwoba_blended`, `home_iso_blended`, `away_iso_blended`, `home_bb_pct_blended`, `away_bb_pct_blended`, `home_k_pct_blended`, `away_k_pct_blended` | `offense_snapshot(team, game_date, team_offense, team_priors, full_weight_games, prior_blend_mode, prior_weight_multiplier)` filters `team_offense_daily` to rows where `team == target_team` and `game_date < target_game_date`, keeps the most recent `full_weight_games`, takes `_safe_mean(metric)`, and blends that value with the prior-season team aggregate from `build_team_priors(...)` via `blend_with_prior(current_value, prior_value, sample_size=len(history), full_weight=full_weight_games, mode, prior_weight_multiplier)`. | Recomputed whenever `src.features.totals_builder` runs, typically after daily aggregate refresh. | Yes. The helper hard-filters to `game_date < target_game_date`. | If current history is missing, use the prior-season team aggregate. If the prior is missing, use the current value. If both are missing, return `None`. | Low for normal slates because same-day rows are excluded by date. Audit postponed or date-correction edge cases separately, but the base date filter is time-safe. |
| Starter blended skill and rest | `home_starter_xwoba_blended`, `away_starter_xwoba_blended`, `home_starter_csw_blended`, `away_starter_csw_blended`, `home_starter_rest_days`, `away_starter_rest_days` | `totals_builder` picks the starter rows for the game from `pitcher_starts`, sorts by `is_probable` and `pitcher_id`, then passes the selected `pitcher_id` into `pitcher_snapshot(...)`. `pitcher_snapshot(...)` filters `pitcher_starts` to `pitcher_id == selected_pitcher` and `game_date < target_game_date`, keeps the most recent `full_weight_starts`, blends `_safe_mean(xwoba_against)` and `_safe_mean(csw_pct)` with prior-season pitcher aggregates from `build_pitcher_priors(...)`, and sets `days_rest = (target_game_date - last_game_date).days`. | Recomputed whenever `src.features.totals_builder` runs, using the current starter mapping already present in `pitcher_starts`. | Yes, with one operational caveat: same-day starter identity can still be wrong if the mapped starter row is wrong or only probable. | If `pitcher_id` is missing, return `None` for all starter values. If current history is missing, use the prior-season pitcher aggregate. If prior is missing, use the current value. If both are missing, return `None`. `days_rest` becomes `None` if no prior start exists. | Low for historical performance values because same-day starts are excluded. Medium for starter identity trust, because this lane currently depends on whatever starter row was mapped for the slate. |
| Bullpen last-three workload and outcomes | `home_bullpen_pitches_last3`, `away_bullpen_pitches_last3`, `home_bullpen_innings_last3`, `away_bullpen_innings_last3`, `home_bullpen_b2b`, `away_bullpen_b2b`, `home_bullpen_runs_allowed_last3`, `away_bullpen_runs_allowed_last3`, `home_bullpen_earned_runs_last3`, `away_bullpen_earned_runs_last3`, `home_bullpen_hits_allowed_last3`, `away_bullpen_hits_allowed_last3`, `home_bullpen_era_last3`, `away_bullpen_era_last3` | `bullpen_snapshot(team, game_date, bullpens)` filters `bullpens_daily` to `team == target_team` and `game_date < target_game_date`, sorts by `game_date`, keeps the last three rows, and computes: pitches as `sum(pitches_thrown)`, innings as baseball-IP converted from summed outs, `b2b` as any bullpen row on `game_date - 1`, runs allowed as `sum(runs_allowed)` when that column exists else `sum(earned_runs)`, earned runs as `sum(earned_runs)`, hits allowed as `sum(hits_allowed)`, and ERA as `(earned_runs * 9) / innings_decimal` when innings are positive. | Recomputed whenever `src.features.totals_builder` runs, based on refreshed bullpen aggregates. | Yes. The helper hard-filters to `game_date < target_game_date`. | Missing bullpen history returns zeros for pitches, innings, back-to-back flag, runs, earned runs, and hits, while `era_last3` returns `None` when innings are zero. If `runs_allowed` is absent in the table, the helper falls back to `earned_runs`. | Low for temporal leakage because same-day rows are excluded. Medium for modeling noise, especially in the short-run outcome features; these are strong downweight or retirement candidates even though they are time-safe. |
| Lineup-derived team context | `home_lineup_top5_xwoba`, `away_lineup_top5_xwoba`, `home_lineup_k_pct`, `away_lineup_k_pct` | `lineup_snapshot(game_id, team, cutoff_ts, lineups, player_batting, hitter_priors, game_date, full_weight_pa, prior_blend_mode, prior_weight_multiplier)` selects the latest lineup snapshot for the team with `snapshot_ts <= feature_cutoff_ts`. If no lineup exists, it falls back to `infer_lineup_from_history(team, game_date, player_batting)`. For each selected hitter it calls `hitter_snapshot(...)`. `top5_xwoba` is the mean of the first five hitters' `xwoba_14` values, falling back player-by-player to `season_prior_xwoba` when `xwoba_14` is missing. `lineup_k_pct` is the mean of each selected hitter's `k_pct_14`. The helper also records whether any selected lineup row is confirmed. | Snapshot-driven at feature-build time. Values update whenever lineups are refreshed and `src.features.totals_builder` reruns. | Yes, but only at the appropriate board state. Projected or inferred lineups are still pregame-safe; they are just lower-certainty inputs. | If no eligible lineup snapshot exists, infer a lineup from recent batting history over the lookback window. If no inferred lineup exists, return `None` for `top5_xwoba` and `lineup_k_pct`, with `confirmed = False`. Within `top5_xwoba`, use `season_prior_xwoba` when `xwoba_14` is missing for a hitter. | Low temporal leakage because lineup snapshots are cut off at `feature_cutoff_ts`. Medium certainty risk because inferred and projected lineups are mixed into the current feature definition. |
| Park factors | `venue_run_factor`, `venue_hr_factor` | `park_snapshot(home_team, season, park_factors, fallback_season)` selects the home-team park row for the current season. If no current-season row exists, it falls back to the configured fallback season and returns `run_factor` and `hr_factor`. | Static or seasonally updated reference data; consumed whenever `src.features.totals_builder` runs. | Yes. Park factors are pregame reference data. | If the current season row is missing, fall back to the configured fallback season. If both are missing, return `None`. | Low. This is reference data with no same-game leakage path. |
| Weather snapshot | `temperature_f`, `wind_speed_mph`, `wind_direction_deg`, `humidity_pct` | `latest_weather_snapshot(game_id, cutoff_ts, weather)` filters `game_weather` to rows with `game_id == target_game_id` and `snapshot_ts <= feature_cutoff_ts`, sorts by `snapshot_ts`, and returns the latest values for temperature, wind speed, wind direction, and humidity. The cutoff itself is produced by `default_cutoff(game_date, game_start_ts)`, which uses `game_start_ts` when it is still in the future, or `now` when the stored start time is already in the past. | Snapshot-driven at feature-build time. Values update whenever weather snapshots refresh and `src.features.totals_builder` reruns. | Yes for standard slates, because the helper caps weather rows at `feature_cutoff_ts`. | If no eligible weather snapshot exists, return `None` for all weather fields. | Low for standard pregame slates. Medium for postponed or stale-start-time cases because `default_cutoff(...)` intentionally moves the cutoff to `now` when the stored start time is already past, allowing later pregame snapshots to be included. |
| Market snapshot and line movement | `market_total`, `market_over_price`, `market_under_price`, `line_movement`, `market_sportsbook`, `line_snapshot_ts` | `latest_market_snapshot(game_id, cutoff_ts, markets, freezes, market_type='total')` filters `game_markets` to rows with `game_id == target_game_id` and `snapshot_ts <= feature_cutoff_ts`. If `market_selection_freezes` has a frozen sportsbook and timestamp for that game and market type, it prefers the frozen sportsbook snapshot when a matching candidate exists. It sorts the remaining rows by `snapshot_ts`, uses the last row as the latest market, the first row as opening market, and sets `line_movement = latest.line_value - opening.line_value` when both values exist. | Snapshot-driven at feature-build time. Values update whenever market snapshots refresh and `src.features.totals_builder` reruns. | Yes, because rows are capped at `feature_cutoff_ts`. | If no eligible market snapshot exists, return `None` for total, sportsbook, prices, snapshot time, and line movement. If opening or latest line values are missing, `line_movement` is `None`. | Low for standard pregame use because snapshots are cutoff-bounded. Medium operational risk if frozen-market logic points to stale books; that is a calibration and freshness concern, not a classic leakage issue. |

### First-Five Totals

First-five totals currently mirror the full-game totals contract with a separate builder, feature table, prediction table, target, and market lane.

#### Metadata Columns

- `game_id`
- `game_date`
- `home_team`
- `away_team`
- `market_sportsbook`
- `prediction_ts`
- `game_start_ts`
- `line_snapshot_ts`
- `feature_cutoff_ts`
- `feature_version`

#### Feature Columns

| Feature | Action | Priority | Notes |
| --- | --- | --- | --- |
| `home_runs_rate_blended` | `keep` | `high` | Still a core baseline, but it should remain subordinate to starter quality and early-order lineup context. |
| `away_runs_rate_blended` | `keep` | `high` | Still a core baseline, but it should remain subordinate to starter quality and early-order lineup context. |
| `home_hits_rate_blended` | `downweight` | `low` | Raw hit volume is less direct than starter suppression and top-of-order quality for first-five scoring. |
| `away_hits_rate_blended` | `downweight` | `low` | Raw hit volume is less direct than starter suppression and top-of-order quality for first-five scoring. |
| `home_xwoba_blended` | `keep` | `high` | Stable process metric that should remain central in first-five too. |
| `away_xwoba_blended` | `keep` | `high` | Stable process metric that should remain central in first-five too. |
| `home_iso_blended` | `keep` | `medium` | Useful power context, but still secondary to broader run creation. |
| `away_iso_blended` | `keep` | `medium` | Useful power context, but still secondary to broader run creation. |
| `home_bb_pct_blended` | `keep` | `medium` | Plate discipline helps, but should not dominate. |
| `away_bb_pct_blended` | `keep` | `medium` | Plate discipline helps, but should not dominate. |
| `home_k_pct_blended` | `keep` | `high` | More important here than in full-game totals because the lane is dominated by the starter-versus-lineup phase of the game. |
| `away_k_pct_blended` | `keep` | `high` | More important here than in full-game totals because the lane is dominated by the starter-versus-lineup phase of the game. |
| `home_starter_xwoba_blended` | `keep` | `high` | Starter talent is even more central in first-five than full game. |
| `away_starter_xwoba_blended` | `keep` | `high` | Starter talent is even more central in first-five than full game. |
| `home_starter_csw_blended` | `keep` | `high` | Starter miss-bat quality should remain core. |
| `away_starter_csw_blended` | `keep` | `high` | Starter miss-bat quality should remain core. |
| `home_starter_rest_days` | `keep` | `high` | Useful readiness and workload signal. |
| `away_starter_rest_days` | `keep` | `high` | Useful readiness and workload signal. |
| `home_lineup_top5_xwoba` | `keep` | `high` | Stronger than in full-game totals because early innings concentrate plate appearances among the best hitters. |
| `away_lineup_top5_xwoba` | `keep` | `high` | Stronger than in full-game totals because early innings concentrate plate appearances among the best hitters. |
| `home_lineup_k_pct` | `keep` | `high` | Strong lineup-quality context for first-five starter interaction. |
| `away_lineup_k_pct` | `keep` | `high` | Strong lineup-quality context for first-five starter interaction. |
| `venue_run_factor` | `keep` | `medium` | Stable environment context still matters, but the shorter horizon makes it less central than starter and lineup features. |
| `venue_hr_factor` | `keep` | `medium` | Useful power context, though still secondary to starter quality and early-order skill. |
| `temperature_f` | `keep` | `medium` | Weather should remain in the model, but over five innings it should sit below starter and lineup drivers. |
| `wind_speed_mph` | `keep` | `medium` | Weather should remain in the model, but over five innings it should sit below starter and lineup drivers. |
| `wind_direction_deg` | `downweight` | `low` | Keep only as secondary directional context unless it proves meaningfully additive in first-five testing. |
| `humidity_pct` | `downweight` | `low` | Secondary weather input. |
| `market_total` | `calibration-only` | `high` | First-five market should calibrate, not drive the raw model. |
| `market_over_price` | `calibration-only` | `low` | Useful pricing context, but the lane often lacks deep first-five book coverage. |
| `market_under_price` | `calibration-only` | `low` | Useful pricing context, but the lane often lacks deep first-five book coverage. |
| `line_movement` | `calibration-only` | `medium` | Better for freshness and disagreement handling than core modeling, but coverage gaps make it less reliable than on full-game totals. |

After the code audit, the first-five lane should be treated as a starter-plus-top-of-lineup model first, with park, weather, and market features kept in supporting or calibration roles rather than inheriting full-game totals weight by default.

#### Code-Audited Formula And Fallback Notes

This pass audits the current first-five totals formulas against `src/features/first5_totals_builder.py`, `src/features/common.py`, and `src/features/contracts.py`.

| Feature family | Applies to | Exact current formula | Refresh cadence | Pregame-safe | Missing-data fallback | Leakage risk |
| --- | --- | --- | --- | --- | --- | --- |
| Team offense blended rates | `home_runs_rate_blended`, `away_runs_rate_blended`, `home_hits_rate_blended`, `away_hits_rate_blended`, `home_xwoba_blended`, `away_xwoba_blended`, `home_iso_blended`, `away_iso_blended`, `home_bb_pct_blended`, `away_bb_pct_blended`, `home_k_pct_blended`, `away_k_pct_blended` | The first-five builder reuses `offense_snapshot(team, game_date, team_offense, team_priors, full_weight_games, prior_blend_mode, prior_weight_multiplier)` exactly as the full-game totals lane does. The helper filters `team_offense_daily` to `team == target_team` and `game_date < target_game_date`, keeps the most recent `full_weight_games`, computes the current mean for each metric, and blends that value with the prior-season aggregate from `build_team_priors(...)` via `blend_with_prior(...)`. | Recomputed whenever `src.features.first5_totals_builder` runs after daily offense aggregates refresh. | Yes. The helper only uses rows with `game_date < target_game_date`. | If current history is missing, use the prior-season team aggregate. If the prior is missing, use the current value. If both are missing, return `None`. | Low. This is the same pregame-safe blended offense baseline used in the full-game totals lane. |
| Starter blended skill and rest | `home_starter_xwoba_blended`, `away_starter_xwoba_blended`, `home_starter_csw_blended`, `away_starter_csw_blended`, `home_starter_rest_days`, `away_starter_rest_days` | `first5_totals_builder` selects the current-game starter IDs from `pitcher_starts` by sorting the game's starter rows on `is_probable` and `pitcher_id`, then choosing the first `home` and `away` row. It passes those IDs into `pitcher_snapshot(...)`, which filters prior starts to `pitcher_id == selected_pitcher` and `game_date < target_game_date`, keeps the most recent `full_weight_starts`, blends `_safe_mean(xwoba_against)` and `_safe_mean(csw_pct)` with prior-season pitcher aggregates from `build_pitcher_priors(...)`, and computes `days_rest = (target_game_date - last_game_date).days`. | Recomputed whenever `src.features.first5_totals_builder` runs and the starter mapping changes. | Yes, with a starter-identity caveat. | If a starter ID is missing, the starter fields are `None`. If current history is missing, use the prior-season pitcher aggregate. If the prior is missing, use the current value. If both are missing, return `None`. `days_rest` is `None` if no prior start exists. | Low for historical leakage because same-day starts are excluded. Medium for operational trust because the lane still depends on the currently mapped starter row, which can be probable or wrong. |
| Lineup-derived early scoring context | `home_lineup_top5_xwoba`, `away_lineup_top5_xwoba`, `home_lineup_k_pct`, `away_lineup_k_pct` | The builder reuses `lineup_snapshot(game_id, team, feature_cutoff_ts, lineups, player_batting, hitter_priors, game_date, min_pa_full_weight, prior_blend_mode, prior_weight_multiplier)`. That helper takes the latest lineup snapshot for the team with `snapshot_ts <= feature_cutoff_ts`, or falls back to `infer_lineup_from_history(team, game_date, player_batting)` when no snapshot exists. For each selected hitter it calls `hitter_snapshot(...)`. `top5_xwoba` is the mean of the first five hitters' `xwoba_14`, falling back hitter-by-hitter to `season_prior_xwoba` when needed. `lineup_k_pct` is the mean of the selected hitters' `k_pct_14`. | Snapshot-driven at feature-build time. Values update whenever lineups refresh and `src.features.first5_totals_builder` reruns. | Yes, but only at the correct board state. Projected and inferred lineups are still pregame-safe; they are just lower-certainty than confirmed lineups. | If no eligible lineup snapshot exists, infer from recent batting history. If inference also fails, the lineup outputs return `None`. Within `top5_xwoba`, missing `xwoba_14` falls back to `season_prior_xwoba` for that hitter. | Low temporal leakage because lineup snapshots are cutoff-bounded and inference uses only prior games. Medium certainty risk because inferred and projected lineups remain mixed into the current feature definition. |
| Park and weather context | `venue_run_factor`, `venue_hr_factor`, `temperature_f`, `wind_speed_mph`, `wind_direction_deg`, `humidity_pct` | `park_snapshot(home_team, season, park_factors, fallback_season)` selects the home-team park row for the current season and falls back to the configured fallback season when needed. `latest_weather_snapshot(game_id, feature_cutoff_ts, weather)` returns the latest weather row with `snapshot_ts <= feature_cutoff_ts` and exposes temperature, wind speed, wind direction, and humidity. `feature_cutoff_ts` itself comes from `default_cutoff(game_date, game_start_ts)`, which uses the scheduled start time when it is in the future or `now` when the stored start time is already in the past. | Snapshot-driven for weather and static or seasonally updated for park factors. Values refresh whenever the builder reruns and new weather or park data exists. | Yes for standard slates, because park is reference data and weather is cutoff-bounded. | Park falls back to the configured fallback season, otherwise `None`. Weather returns `None` when no eligible snapshot exists. | Low overall. As with full-game totals, weather has medium operational risk for postponed or stale-start-time cases because `default_cutoff(...)` intentionally moves the cutoff to `now` once the stored start is in the past. |
| First-five market snapshot | `market_total`, `market_over_price`, `market_under_price`, `line_movement`, `market_sportsbook`, `line_snapshot_ts` | The builder calls `latest_market_snapshot(game_id, feature_cutoff_ts, markets, freezes, market_type='first_five_total')`. That helper filters `game_markets` to `game_id == target_game_id`, `market_type == 'first_five_total'`, and `snapshot_ts <= feature_cutoff_ts`. If `market_selection_freezes` contains a frozen sportsbook for the first-five market, it prefers that sportsbook snapshot when available. It sorts the remaining rows by `snapshot_ts`, uses the last row as the latest market, the first row as opening market, and computes `line_movement = latest.line_value - opening.line_value` when both values exist. | Snapshot-driven at feature-build time. Values update whenever first-five market snapshots refresh and the builder reruns. | Yes. The helper only uses rows up to the feature cutoff timestamp. | If no eligible first-five market exists, return `None` for sportsbook, prices, market total, snapshot time, and line movement. If opening or latest line values are missing, `line_movement` is `None`. | Low for classical leakage because the snapshot is cutoff-bounded. Medium product risk because first-five market coverage is still incomplete on many slates, leaving calibration fields null. |
| Outcome target | `actual_total_runs_first5` | The builder copies `game.total_runs_first5` from the `games` table into `actual_total_runs_first5` and persists it as `FIRST5_TOTALS_TARGET_COLUMN`. This is the training and evaluation label for the lane, not a pregame feature. | Recomputed whenever `src.features.first5_totals_builder` runs for historical or already-played slates. | Not applicable as a model input. It is a postgame target. | If the game record does not yet have `total_runs_first5`, the target is `None`. | High if treated as a feature, but safe when kept strictly as the label. |

### Player Hits

#### Metadata Columns

- `game_id`
- `game_date`
- `player_id`
- `team`
- `opponent`
- `prediction_ts`
- `game_start_ts`
- `line_snapshot_ts`
- `feature_cutoff_ts`
- `feature_version`

#### Feature Columns

| Feature | Bucket | Primary source(s) | Earliest safe state | Action | Priority | Downstream consumers | Notes | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `player_name` | `availability` | `dim_players` | `early` | `product-only` | `low` | hits surfaces | Identity field for UI and explanations, not a real modeling feature. | `seeded` |
| `home_away` | `availability` | `games` | `early` | `downweight` | `low` | hits model | Mild context field that should sit below lineup slot and matchup quality. | `seeded` |
| `lineup_slot` | `availability` | `lineups`, inferred lineup history | `morning` | `keep` | `high` | hits model, hits surfaces | Opportunity proxy and one of the most important hits-lane inputs. | `seeded` |
| `is_confirmed_lineup` | `availability` | `lineups` | `lineup_confirmed` | `keep` | `high` | hits model, certainty layer, hits surfaces | Core certainty signal for any lineup-dependent player prop. | `seeded` |
| `projected_plate_appearances` | `availability` | `lineups`, lineup history | `morning` | `keep` | `high` | hits model | Opportunity proxy that should remain central. | `seeded` |
| `hit_rate_7` | `skill` | `player_game_batting` | `early` | `downweight` | `medium` | hits model | Very short-run form should stay supportive rather than central. | `seeded` |
| `hit_rate_14` | `skill` | `player_game_batting` | `early` | `downweight` | `medium` | hits model | Recent form is useful, but should not outrank process metrics. | `seeded` |
| `hit_rate_30` | `skill` | `player_game_batting` | `early` | `keep` | `medium` | hits model | Longer-window hit rate is steadier than very short-run streak form. | `seeded` |
| `hit_rate_blended` | `skill` | `player_game_batting`, priors | `early` | `keep` | `high` | hits model | Blended baseline is more defensible than raw streak metrics alone. | `seeded` |
| `xba_14` | `skill` | `player_game_batting` | `early` | `keep` | `high` | hits model | Strong process input for contact quality and should remain central. | `seeded` |
| `xwoba_14` | `skill` | `player_game_batting` | `early` | `keep` | `high` | hits model | Strong process input for overall quality of contact and approach. | `seeded` |
| `hard_hit_pct_14` | `skill` | `player_game_batting` | `early` | `keep` | `high` | hits model | Good support feature when paired with xwOBA and opportunity. | `seeded` |
| `k_pct_14` | `skill` | `player_game_batting` | `early` | `keep` | `medium` | hits model | Useful risk indicator, but not a primary hit driver by itself. | `seeded` |
| `season_prior_hit_rate` | `skill` | priors | `early` | `keep` | `high` | hits model | Helpful stabilizer against noisy short-run form. | `seeded` |
| `season_prior_xba` | `skill` | priors | `early` | `keep` | `medium` | hits model | Prior skill estimate remains useful as a stabilizer. | `seeded` |
| `season_prior_xwoba` | `skill` | priors | `early` | `keep` | `medium` | hits model | Prior skill estimate remains useful as a stabilizer. | `seeded` |
| `opposing_starter_xwoba` | `skill` | `pitcher_starts` | `early` | `keep` | `high` | hits model | Opposing starter quality should remain central to hitter context. | `seeded` |
| `opposing_starter_csw` | `skill` | `pitcher_starts` | `early` | `keep` | `high` | hits model | Opposing starter bat-missing ability should remain central to hitter context. | `seeded` |
| `opposing_bullpen_pitches_last3` | `availability` | `bullpens_daily` | `early` | `downweight` | `medium` | hits model | Bullpen workload matters, but less directly than starter and lineup opportunity. | `seeded` |
| `opposing_bullpen_innings_last3` | `availability` | `bullpens_daily` | `early` | `downweight` | `medium` | hits model | Bullpen workload matters, but less directly than starter and lineup opportunity. | `seeded` |
| `venue_run_factor` | `skill` | `park_factors` | `early` | `keep` | `medium` | hits model, hits surfaces | Stable park baseline is useful, but secondary to batter and matchup context. | `seeded` |
| `park_hr_factor` | `skill` | `park_factors` | `early` | `downweight` | `low` | hits model | More power-specific than hit-specific and likely secondary for 1+ hit props. | `seeded` |
| `temperature_f` | `volatility` | `game_weather` | `early` | `downweight` | `low` | hits model, hits surfaces | Weather matters, but usually less than lineup slot and contact quality for hits. | `seeded` |
| `wind_speed_mph` | `volatility` | `game_weather` | `early` | `downweight` | `low` | hits model, hits surfaces | Weather matters, but usually less than lineup slot and contact quality for hits. | `seeded` |
| `team_run_environment` | `volatility` | totals-side team context | `morning` | `replace` | `medium` | hits model | Current context proxy may survive, but a cleaner opportunity environment feature is preferred. | `seeded` |
| `streak_len_capped` | `volatility` | `player_game_batting` | `early` | `product-only` | `medium` | hits model, hot-hitters surfaces | Best kept as UI and explainer context unless supported by stronger process metrics. | `seeded` |

#### Code-Audited Formula And Fallback Notes

This pass audits the current hits formulas against `src/features/hits_builder.py`, `src/features/common.py`, and `src/features/priors.py`.

| Feature family | Applies to | Exact current formula | Refresh cadence | Pregame-safe | Missing-data fallback | Leakage risk |
| --- | --- | --- | --- | --- | --- | --- |
| Player identity and lineup source | `player_name`, `home_away`, `lineup_slot`, `is_confirmed_lineup` | `hits_builder` first gathers the latest eligible lineup snapshot for each team with `snapshot_ts <= feature_cutoff_ts`. If a team has no snapshot, it falls back to `infer_lineup_from_history(team, game_date, player_batting, players)`. That helper filters `player_game_batting` to `team == target_team` and `game_date < target_game_date`, prefers the last `lookback_days = 21` days, weights recent lineup slots by `selection_weight = recency_weight * max(plate_appearances, 1.0)` where `recency_weight = 1 / (days_ago + 1)`, estimates a lineup slot per hitter, then assigns unique slots `1..9`. `player_name` comes from lineup snapshots when present or from `dim_players.full_name` during inferred-lineup construction. `home_away` is `H` when the hitter team matches the game home team, else `A`. `is_confirmed_lineup` is `bool(lineup_row.is_confirmed)`. | Snapshot-driven at feature-build time. Values update whenever lineups refresh and `src.features.hits_builder` reruns. | Yes, with certainty caveats. Projected and inferred lineups are pregame-safe but lower-trust than confirmed lineups. | If no eligible lineup snapshot exists for a team, infer a lineup from recent history. If inference also fails, the team contributes no rows. During inference, missing player names fall back to the `player_id` string. | Low temporal leakage because lineup snapshots are bounded by `feature_cutoff_ts` and history uses `game_date < target_game_date`. Medium certainty risk because inferred lineups can still be wrong even when they are time-safe. |
| Projected opportunity | `projected_plate_appearances` | `projected_plate_appearances(lineup_slot)` uses a fixed lookup: `{1: 4.8, 2: 4.7, 3: 4.6, 4: 4.5, 5: 4.3, 6: 4.1, 7: 4.0, 8: 3.9, 9: 3.8}`, defaulting to `4.0` for unexpected slots. | Recomputed whenever `src.features.hits_builder` runs. | Yes. It depends only on the lineup slot chosen at the current cutoff. | If `lineup_slot` is missing, return `None`. If the slot is outside the explicit map, return `4.0`. | Low for temporal leakage. Medium model-risk because the PA curve is a fixed heuristic rather than a learned opportunity model. |
| Hitter recent form and prior blend | `hit_rate_7`, `hit_rate_14`, `hit_rate_30`, `hit_rate_blended`, `xba_14`, `xwoba_14`, `hard_hit_pct_14`, `k_pct_14`, `season_prior_hit_rate`, `season_prior_xba`, `season_prior_xwoba`, `streak_len_capped` | `hitter_snapshot(player_id, game_date, player_batting, hitter_priors, full_weight_pa, prior_blend_mode, prior_weight_multiplier)` filters `player_game_batting` to `player_id == target_player` and `game_date < target_game_date`, sorts by `game_date`, and creates rolling windows at 7, 14, and 30 days. `hit_rate_*` are `_safe_mean((hits > 0).astype(float))` over each window. `hit_rate_blended` blends the 30-day hit rate with the prior-season hit rate from `build_hitter_priors(...)` using `blend_with_prior(current_value=hit_rate_30, prior_value=prior_hit_rate, sample_size=pa_30, full_weight=min_pa_full_weight, mode, prior_weight_multiplier)`. `xba_14`, `xwoba_14`, and `hard_hit_pct_14` are 14-day means. `k_pct_14` is `sum(strikeouts) / sum(plate_appearances)` over the 14-day window via `safe_rate(...)`. `streak_len_capped` walks backward through prior games and counts consecutive hit games up to 5. Prior-season fields are pulled directly from `build_hitter_priors(...)`, which aggregates prior-season `had_hit`, `xba`, and `xwoba`. | Recomputed whenever `src.features.hits_builder` runs, using the current historical batting table. | Yes. The helper hard-filters to `game_date < target_game_date`. | If a current window is empty, return `None` for the window metric. `hit_rate_blended` falls back to the prior-season hit rate when current data is missing, or to the current value when the prior is missing, or `None` when both are missing. `streak_len_capped` returns `0` when there is no history. | Low temporal leakage because same-day batting rows are excluded. Medium modeling-noise risk for short-run recency metrics and streak length, which are intentionally documented as lower-trust features. |
| Opposing starter context | `opposing_starter_xwoba`, `opposing_starter_csw` | `hits_builder` identifies the opponent starter from the game's `pitcher_starts` rows, then reuses `pitcher_snapshot(...)` to populate the opposing starter values. The formulas are the same as the totals lane: recent means of `xwoba_against` and `csw_pct` blended with prior-season pitcher aggregates from `build_pitcher_priors(...)`, using only starts where `game_date < target_game_date`. | Recomputed whenever `src.features.hits_builder` runs. | Yes, with a starter-identity caveat. | If the opponent starter is missing, both fields are `None`. If current history is missing, use the prior-season pitcher aggregate. If the prior is missing, use the current value. If both are missing, return `None`. | Low for historical leakage, medium for starter identity trust when the slate starter mapping is still probable or wrong. |
| Opposing bullpen workload | `opposing_bullpen_pitches_last3`, `opposing_bullpen_innings_last3` | `hits_builder` uses `bullpen_snapshot(opponent_team, game_date, bullpens)` and takes the `pitches_last3` and `innings_last3` outputs. `bullpen_snapshot(...)` uses only bullpen rows where `game_date < target_game_date`, keeps the last three rows, sums `pitches_thrown`, and converts summed outs back into baseball innings notation. | Recomputed whenever `src.features.hits_builder` runs. | Yes. Same-day bullpen rows are excluded by date. | Missing bullpen history returns `0` for both values. | Low temporal leakage. Medium modeling-value risk because bullpen workload is only an indirect context feature for 1+ hit props. |
| Park, weather, and team scoring context | `venue_run_factor`, `park_hr_factor`, `temperature_f`, `wind_speed_mph`, `team_run_environment` | `venue_run_factor` and `park_hr_factor` come from `park_snapshot(home_team, season, park_factors, fallback_season)`, which uses the current-season home-team park row and falls back to the configured fallback season when needed. `temperature_f` and `wind_speed_mph` come from `latest_weather_snapshot(game_id, feature_cutoff_ts, weather)`, which uses the most recent weather snapshot with `snapshot_ts <= feature_cutoff_ts`. `team_run_environment` is set directly from the batting team's `offense_snapshot(...)["runs_rate"]`, meaning it currently reuses the totals-side team scoring baseline rather than a hitter-specific opportunity or lineup-strength transform. | Recomputed whenever `src.features.hits_builder` runs. Weather changes with snapshot refreshes; park and team scoring context update with aggregate refreshes. | Yes. Park is static reference data, weather is cutoff-bounded, and `team_run_environment` uses only offense rows where `game_date < target_game_date`. | Park falls back to the configured fallback season, otherwise `None`. Weather returns `None` when no eligible snapshot exists. `team_run_environment` falls back through `offense_snapshot(...)`, which uses prior-season team offense when current history is missing, current value when prior is missing, and `None` when both are missing. | Low temporal leakage overall. Medium design risk for `team_run_environment` because it is a totals-side proxy rather than a dedicated hitter-opportunity feature, which is why it is marked for replacement. |
| Outcome target | `got_hit` | `hits_builder` builds `actual_map` from `player_game_batting` rows already recorded for the same `game_id`, then sets `got_hit = None if actual_player_hits is None else (actual_player_hits > 0)`. This is the postgame label used for training and evaluation, not a pregame feature. | Recomputed whenever `src.features.hits_builder` runs for historical or already-played slates. | Not applicable as a model input. It is a postgame target. | If the player's box-score hit count is unavailable, the target is `None`. | High if treated as a feature, but it is only the training target. Safe when kept strictly as a label. |

### Pitcher Strikeouts

#### Metadata Columns

- `game_id`
- `game_date`
- `pitcher_id`
- `team`
- `opponent`
- `prediction_ts`
- `game_start_ts`
- `line_snapshot_ts`
- `feature_cutoff_ts`
- `feature_version`

#### Feature Columns

| Feature | Bucket | Primary source(s) | Earliest safe state | Action | Priority | Downstream consumers | Notes | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `throws` | `availability` | `dim_players` | `early` | `keep` | `low` | strikeout model, surfaces | Necessary to interpret lineup-handedness shares, but by itself it is just routing context rather than a major driver. | `seeded` |
| `days_rest` | `availability` | `pitcher_starts` | `early` | `keep` | `high` | strikeout model | Workload and readiness are core strikeout-lane inputs. | `seeded` |
| `projected_innings` | `availability` | pitcher history, starter context | `morning` | `keep` | `high` | strikeout model | This looks like the single clearest leash proxy in the current builder and should stay top-tier. | `seeded` |
| `recent_avg_ip_3` | `skill` | `pitcher_starts` | `early` | `keep` | `low` | strikeout model | Useful only as support around `projected_innings`, not as a separate core feature. | `seeded` |
| `recent_avg_ip_5` | `skill` | `pitcher_starts` | `early` | `keep` | `medium` | strikeout model | A steadier leash support feature than the three-start version, but still secondary to explicit projected innings. | `seeded` |
| `recent_avg_strikeouts_3` | `skill` | `pitcher_starts` | `early` | `downweight` | `medium` | strikeout model | Recent outcome form is useful, but weaker than rate and leash signals. | `seeded` |
| `recent_avg_strikeouts_5` | `skill` | `pitcher_starts` | `early` | `downweight` | `low` | strikeout model | Raw recent outcomes should stay behind whiff skill, rate skill, and leash proxies. | `seeded` |
| `recent_k_per_batter_3` | `skill` | `pitcher_starts` | `early` | `keep` | `high` | strikeout model | Rate-based strikeout skill is stronger than raw recent K totals. | `seeded` |
| `recent_k_per_batter_5` | `skill` | `pitcher_starts` | `early` | `keep` | `high` | strikeout model | Rate-based strikeout skill is stronger than raw recent K totals. | `seeded` |
| `recent_avg_pitch_count_3` | `availability` | `pitcher_starts` | `early` | `keep` | `high` | strikeout model | Recent pitch count is a top-tier leash and trust signal in the current builder. | `seeded` |
| `recent_whiff_pct_5` | `skill` | `pitcher_starts` | `early` | `keep` | `high` | strikeout model | Miss-bat process metric should remain central. | `seeded` |
| `recent_csw_pct_5` | `skill` | `pitcher_starts` | `early` | `keep` | `high` | strikeout model | Miss-bat and command process metric should remain central. | `seeded` |
| `recent_xwoba_5` | `skill` | `pitcher_starts` | `early` | `downweight` | `low` | strikeout model | Useful context, but it should clearly sit behind whiff, rate, and leash features in a strikeout model. | `seeded` |
| `baseline_strikeouts` | `skill` | priors, rolling history | `early` | `keep` | `high` | strikeout model | Strong baseline anchor that should survive the rework. | `seeded` |
| `opponent_k_pct_blended` | `skill` | `team_offense_daily`, lineup summary | `morning` | `keep` | `high` | strikeout model, board surfaces | Important opponent tendency feature once lineup timing rules are explicit. | `seeded` |
| `same_hand_share` | `availability` | lineup handedness summary | `morning` | `keep` | `high` | strikeout model | Handedness mix is core for matchup-specific strikeout context. | `seeded` |
| `opposite_hand_share` | `availability` | lineup handedness summary | `morning` | `keep` | `high` | strikeout model | Handedness mix is core for matchup-specific strikeout context. | `seeded` |
| `switch_share` | `availability` | lineup handedness summary | `morning` | `keep` | `low` | strikeout model | Useful supplement to handedness mix, but clearly below same-hand and opposite-hand shares. | `seeded` |
| `lineup_right_count` | `availability` | lineups, inferred lineup history | `morning` | `keep` | `medium` | strikeout model | Useful lineup composition detail, but the normalized shares are cleaner primary inputs. | `seeded` |
| `lineup_left_count` | `availability` | lineups, inferred lineup history | `morning` | `keep` | `medium` | strikeout model | Useful lineup composition detail, but the normalized shares are cleaner primary inputs. | `seeded` |
| `lineup_switch_count` | `availability` | lineups, inferred lineup history | `morning` | `keep` | `medium` | strikeout model | Useful supplement to lineup composition, but usually secondary. | `seeded` |
| `known_hitters` | `availability` | lineups | `morning` | `keep` | `medium` | strikeout model, certainty layer | More valuable as a certainty and completeness indicator than as a direct strikeout driver. | `seeded` |
| `confirmed_hitters` | `availability` | lineups | `lineup_confirmed` | `keep` | `high` | strikeout model, certainty layer | Key certainty signal and likely part of the future lineup certainty score. | `seeded` |
| `total_hitters` | `availability` | lineups, inferred lineup history | `morning` | `keep` | `medium` | strikeout model, certainty layer | Useful denominator for certainty and lineup completeness more than as a direct model signal. | `seeded` |
| `handedness_adjustment_applied` | `availability` | strikeout feature builder logic | `morning` | `product-only` | `low` | strikeout model | Good audit and interpretability flag, but it should not matter as much as the underlying matchup fields themselves. | `seeded` |
| `handedness_data_missing` | `volatility` | strikeout feature builder logic | `morning` | `keep` | `high` | strikeout model, certainty layer | Missing handedness should directly reduce trust in matchup-specific adjustments. | `seeded` |

#### Code-Audited Formula And Fallback Notes

This pass audits the current strikeout formulas against `src/features/strikeouts_builder.py`, `src/features/common.py`, and `src/features/contracts.py`.

After the code audit, the strikeout lane should be treated as a leash-plus-bat-missing-skill model first, then adjusted by opponent strikeout tendency and lineup handedness or confirmation, with completeness and interpretability flags kept secondary.

| Feature family | Applies to | Exact current formula | Refresh cadence | Pregame-safe | Missing-data fallback | Leakage risk |
| --- | --- | --- | --- | --- | --- | --- |
| Starter identity and rest context | `throws`, `days_rest` | `throws` is taken directly from the selected `pitcher_starts` row as `str(starter.throws).strip().upper()[:1]`. `days_rest` is also taken directly from `starter.days_rest` on that same current-game row. The builder does call `pitcher_snapshot(...)`, but it does not use that helper's computed `days_rest`; the persisted field comes from the current starter record itself. | Recomputed whenever `src.features.strikeouts_builder` runs and the current starter mapping changes. | Yes, with a starter-identity caveat. These are current-slate context fields, not postgame stats. | If `throws` is blank it becomes `None`. If the current starter row has no `days_rest`, the field stays `None`. If the wrong or only-probable starter row is selected, the context is still time-safe but may be wrong. | Low temporal leakage, medium operational risk because the feature depends on the accuracy of the starter row already present in `pitcher_starts`. |
| Recent workload, whiff skill, and strikeout baseline | `projected_innings`, `recent_avg_ip_3`, `recent_avg_ip_5`, `recent_avg_strikeouts_3`, `recent_avg_strikeouts_5`, `recent_k_per_batter_3`, `recent_k_per_batter_5`, `recent_avg_pitch_count_3`, `recent_whiff_pct_5`, `recent_csw_pct_5`, `recent_xwoba_5`, `baseline_strikeouts` | `history` is all prior `pitcher_starts` rows for the pitcher with `game_date < target_game_date`, sorted by date. `_recent_pitcher_form(history)` uses `tail(3)` and `tail(5)` windows. `recent_avg_ip_*` are means of prior `ip`. `recent_avg_strikeouts_*` are means of prior `strikeouts`. `recent_k_per_batter_*` are `_safe_rate_sum(strikeouts, batters_faced)` over the window, but when `batters_faced` is missing `_usage_opportunity_series(...)` falls back to `pitch_count` as the denominator. `recent_avg_pitch_count_3` is the mean of `batters_faced` over the last three starts, again falling back to `pitch_count` when `batters_faced` is unavailable. `recent_whiff_pct_5`, `recent_csw_pct_5`, and `recent_xwoba_5` are simple five-start means. `baseline_strikeouts = recent_avg_strikeouts_5 or recent_avg_strikeouts_3`. `projected_innings = recent_avg_ip_5 or recent_avg_ip_3`, with falsey or NaN values cleared back to `None`. | Recomputed whenever `src.features.strikeouts_builder` runs, using the current historical start table. | Yes. All history is filtered to `game_date < target_game_date`. | Empty windows return `None` for the affected metrics. `baseline_strikeouts` falls back from five-start average to three-start average, otherwise `None`. `projected_innings` falls back from five-start average to three-start average, otherwise `None`. When `batters_faced` is missing, rate and usage calculations fall back to `pitch_count`, which is operationally useful but changes the denominator semantics. | Low temporal leakage because same-day starts are excluded. Medium modeling-risk because the main leash proxy is heuristic and the batter-faced to pitch-count fallback mixes two different opportunity scales. |
| Opponent strikeout baseline | `opponent_k_pct_blended` | The opponent team is selected from the current game row, then `offense_snapshot(opponent, game_date, team_offense, team_priors, full_weight_games, prior_blend_mode, prior_weight_multiplier)["k_pct"]` is used. That helper filters `team_offense_daily` to `team == opponent` and `game_date < target_game_date`, keeps the most recent `full_weight_games`, computes the current mean strikeout rate, and blends it with the prior-season team strikeout rate from `build_team_priors(...)`. | Recomputed whenever `src.features.strikeouts_builder` runs after team-offense refreshes. | Yes. The helper hard-filters to prior dates only. | If current history is missing, use the prior-season team strikeout rate. If the prior is missing, use the current value. If both are missing, return `None`. | Low. This is a standard pregame blended team tendency feature. |
| Opponent lineup handedness and completeness | `same_hand_share`, `opposite_hand_share`, `switch_share`, `lineup_right_count`, `lineup_left_count`, `lineup_switch_count`, `known_hitters`, `confirmed_hitters`, `total_hitters`, `handedness_adjustment_applied`, `handedness_data_missing` | `_latest_lineup_for_team(...)` first looks for the latest lineup snapshot for the opponent with `snapshot_ts <= feature_cutoff_ts`. If none exists, it falls back to `infer_lineup_from_history(team, game_date, player_batting, players)`, then merges `dim_players.bats`. `_summarize_lineup_handedness(...)` counts `R`, `L`, and `S` batters, sets `known_hitters = R + L + S`, and counts `confirmed_hitters` from the lineup rows. `_same_hand_shares(throw_hand, handedness)` converts those counts into shares only when the pitcher's throwing hand is `R` or `L` and `known_hitters > 0`; otherwise all three share fields are `None`. `total_hitters` is `len(opponent_lineup)`. `handedness_adjustment_applied` is true only when both same-hand and opposite-hand shares are populated. `handedness_data_missing` is true when `known_hitters == 0`. | Snapshot-driven at feature-build time. Values update whenever lineups refresh and `src.features.strikeouts_builder` reruns. | Yes, with certainty caveats. Projected and inferred lineups are pregame-safe but lower-trust than confirmed snapshots. | If no eligible lineup snapshot exists, infer from recent batting history. If inference also fails, the lineup is empty, counts are zero, share fields are `None`, `confirmed_hitters` is `0`, `total_hitters` is `0`, and `handedness_data_missing` becomes true. Missing `bats` values reduce `known_hitters` because only `R`, `L`, and `S` count toward the denominator. | Low temporal leakage because lineup snapshots are cutoff-bounded and inference uses only prior games. Medium certainty risk because inferred lineups and missing handedness can materially change matchup-specific adjustments. |
| Postgame label and realized lineup strikeout rate | `actual_strikeouts`, `opponent_lineup_k_pct` | `actual_strikeouts` is copied from `starter.strikeouts` on the current game's `pitcher_starts` row and is the training target. After the initial feature rows are built, the builder aggregates same-game `player_game_batting` by opponent team to compute `opponent_lineup_k_pct = strikeouts / plate_appearances` for that actual lineup and merges it back into `feature_frame`. This realized lineup strikeout rate is then retained in `feature_payload`, and the API uses it as the preferred display value when present. It is not part of `STRIKEOUTS_FEATURE_COLUMNS`, but it is persisted alongside the row payload. | Recomputed whenever `src.features.strikeouts_builder` runs for historical or already-played slates. | `actual_strikeouts`: not applicable as a feature, because it is the postgame target. `opponent_lineup_k_pct`: no, because it depends on realized same-game batting results. | If the starter strikeout total is unavailable, the target is missing. If opponent batting rows or plate appearances are missing, `opponent_lineup_k_pct` remains `None`. | High if either field is treated as a pregame model input. Safe only when `actual_strikeouts` remains the label and `opponent_lineup_k_pct` remains a diagnostic or postgame explanation field. |

## Planned Certainty-Layer Features

These are proposed canonical names for the first certainty-pass features in Phase 2.

| Proposed feature | Bucket | Intended lane(s) | Earliest safe state | Action | Priority | Notes | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `starter_certainty_score` | `availability` | totals, first-five, strikeouts | `early` | `planned` | `high` | Should distinguish probable from confirmed starter trust and directly feed confidence. | `planned` |
| `lineup_certainty_score` | `availability` | totals, first-five, hits, strikeouts | `morning` | `planned` | `high` | Should distinguish inferred, projected, snapshot, and confirmed lineups. | `planned` |
| `weather_freshness_score` | `volatility` | totals, first-five, hits | `early` | `planned` | `high` | Should incorporate observation age and missing fields. | `planned` |
| `market_freshness_score` | `volatility` | totals, first-five, hits, strikeouts | `early` | `planned` | `high` | Should reflect line age, stale-market conditions, and capture timing trust. | `planned` |
| `bullpen_completeness_score` | `availability` | totals, first-five | `early` | `planned` | `medium` | Should reflect whether recent bullpen context is complete enough to trust. | `planned` |
| `missing_fallback_count` | `volatility` | all | `early` | `planned` | `high` | Should count key fields that were inferred or defaulted instead of hiding missingness. | `planned` |
| `board_state` | `volatility` | all | `early` | `planned` | `high` | Should be explicit and testable rather than implied by UI timing. | `planned` |

## Phase 1 Audit Order

Audit these groups in this order:

1. full-game totals and first-five totals shared features
2. lineup-driven availability fields
3. bullpen workload versus bullpen outcome fields
4. player hits recency and lineup-slot fields
5. strikeout leash and handedness fields
6. certainty-layer additions

## Open Questions To Resolve In Phase 1

- Which current bullpen outcome fields should be retired or downweighted in favor of workload and leverage availability?
- Which lineup-derived fields are safe on projected lineups versus only safe on confirmed lineups?
- Which market-derived fields are calibration-only and should stay out of the primary model feature set?
- Which streak-style hitter fields should remain product-facing but receive lower model weight?
- Which new certainty fields belong in feature tables versus transform or API layers?

## Retrain Comparison: Core-Predictor-Only Selection (April 2026)

After classifying every field and wiring all four trainers to `feature_columns_for_roles(lane, [FIELD_ROLE_CORE_PREDICTOR])`, each lane was retrained on the same feature snapshots. Results below compare the old (all-fields) artifacts against the new (core-predictor-only) artifacts.

### Full-Game Totals

Old: 38 features (included market fields). New: 42 core predictors (gained bullpen detail, dropped market fields).

| Candidate | Metric | Before | After | Delta |
|-----------|--------|--------|-------|-------|
| GBR (winner) | MAE | 4.927 | 5.027 | +2.0% |
| GBR | RMSE | 5.971 | 5.987 | +0.3% |
| Ridge | MAE | 16.452 | 5.871 | **-64%** |
| Ridge | RMSE | 25.155 | 7.356 | **-71%** |

GBR essentially flat. Ridge massively improved — market fields were causing scale/collinearity harm in the linear model.

### First-Five Totals

Old: 32 features (included market fields). New: 28 core predictors.

| Candidate | Metric | Before | After | Delta |
|-----------|--------|--------|-------|-------|
| GBR (winner) | MAE | 2.410 | 2.620 | +8.7% |
| GBR | RMSE | 3.066 | 3.278 | +6.9% |
| Ridge | MAE | 2.468 | 3.224 | +30.6% |
| Ridge | RMSE | 3.097 | 6.371 | +105.7% |

Largest degradation across lanes. `market_total` carries non-redundant information for first-five prediction. Will recover when the Phase 2 calibration layer reintroduces market fields post-model.

### Player Hits (1+ Hit Classification)

Old: 26 features (included `is_confirmed_lineup`, `streak_len_capped`). New: 24 core predictors.

| Candidate | Metric | Before | After | Delta |
|-----------|--------|--------|-------|-------|
| HGB calibrated (winner) | Log loss | 0.6869 | 0.6794 | **-1.1%** |
| HGB calibrated | Brier | 0.2462 | 0.2424 | **-1.5%** |
| Logistic calibrated | Log loss | 0.6891 | 0.6818 | -1.1% |
| Logistic calibrated | Brier | 0.2468 | 0.2431 | -1.5% |

Clean improvement. Removing certainty and product-only fields helped calibration.

### Pitcher Strikeouts

Old: 25 features (included 5 certainty/diagnostic). New: 20 core predictors.

| Candidate | Metric | Before | After | Delta |
|-----------|--------|--------|-------|-------|
| Ridge (winner) | MAE | 1.901 | 1.964 | +3.3% |
| Ridge | RMSE | 2.313 | 2.393 | +3.5% |
| GBR | MAE | 2.176 | 2.026 | **-6.9%** |
| GBR | RMSE | 2.610 | 2.514 | **-3.7%** |

Ridge slightly degraded; GBR improved. The removed certainty signals (`known_hitters`, `confirmed_hitters`, etc.) had some leakage benefit for Ridge but masked real skill prediction.

### Summary

- **Totals and hits**: core-predictor-only selection is a clear win or neutral.
- **Strikeouts**: small Ridge regression acceptable; certainty fields will return as trust/suppress inputs in Phase 2.
- **First-five totals**: most sensitive to market_total removal; calibration layer is the priority follow-up for this lane.