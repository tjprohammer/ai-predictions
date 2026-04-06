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
| `home_hits_rate_blended` | `skill` | `team_offense_daily`, priors | `early` | `downweight` | `medium` | totals model | Useful support signal, but weaker than run creation and contact-quality context. | `seeded` |
| `away_hits_rate_blended` | `skill` | `team_offense_daily`, priors | `early` | `downweight` | `medium` | totals model | Useful support signal, but weaker than run creation and contact-quality context. | `seeded` |
| `home_xwoba_blended` | `skill` | `team_offense_daily`, priors | `early` | `keep` | `high` | totals model | Strong process metric for team offense quality and worth keeping central. | `seeded` |
| `away_xwoba_blended` | `skill` | `team_offense_daily`, priors | `early` | `keep` | `high` | totals model | Strong process metric for team offense quality and worth keeping central. | `seeded` |
| `home_iso_blended` | `skill` | `team_offense_daily`, priors | `early` | `keep` | `medium` | totals model | Power context matters, but should sit below broader run-creation indicators. | `seeded` |
| `away_iso_blended` | `skill` | `team_offense_daily`, priors | `early` | `keep` | `medium` | totals model | Power context matters, but should sit below broader run-creation indicators. | `seeded` |
| `home_bb_pct_blended` | `skill` | `team_offense_daily`, priors | `early` | `keep` | `medium` | totals model | Plate-discipline context is useful, but not a top-tier driver by itself. | `seeded` |
| `away_bb_pct_blended` | `skill` | `team_offense_daily`, priors | `early` | `keep` | `medium` | totals model | Plate-discipline context is useful, but not a top-tier driver by itself. | `seeded` |
| `home_k_pct_blended` | `skill` | `team_offense_daily`, priors | `early` | `keep` | `medium` | totals model, product surfaces | Strikeout tendency remains useful, especially when paired with starter miss bats. | `seeded` |
| `away_k_pct_blended` | `skill` | `team_offense_daily`, priors | `early` | `keep` | `medium` | totals model, product surfaces | Strikeout tendency remains useful, especially when paired with starter miss bats. | `seeded` |
| `home_starter_xwoba_blended` | `skill` | `pitcher_starts`, priors | `early` | `keep` | `high` | totals model | Core starter skill input and should remain central. | `seeded` |
| `away_starter_xwoba_blended` | `skill` | `pitcher_starts`, priors | `early` | `keep` | `high` | totals model | Core starter skill input and should remain central. | `seeded` |
| `home_starter_csw_blended` | `skill` | `pitcher_starts`, priors | `early` | `keep` | `high` | totals model | Strong process metric for starter bat-missing and command quality. | `seeded` |
| `away_starter_csw_blended` | `skill` | `pitcher_starts`, priors | `early` | `keep` | `high` | totals model | Strong process metric for starter bat-missing and command quality. | `seeded` |
| `home_starter_rest_days` | `availability` | `pitcher_starts` | `early` | `keep` | `high` | totals model | Useful workload and readiness proxy that should survive into certainty-aware modeling. | `seeded` |
| `away_starter_rest_days` | `availability` | `pitcher_starts` | `early` | `keep` | `high` | totals model | Useful workload and readiness proxy that should survive into certainty-aware modeling. | `seeded` |
| `home_bullpen_pitches_last3` | `availability` | `bullpens_daily` | `early` | `keep` | `high` | totals model | Workload signal is more actionable than short-run bullpen outcomes. | `seeded` |
| `away_bullpen_pitches_last3` | `availability` | `bullpens_daily` | `early` | `keep` | `high` | totals model | Workload signal is more actionable than short-run bullpen outcomes. | `seeded` |
| `home_bullpen_innings_last3` | `availability` | `bullpens_daily` | `early` | `keep` | `high` | totals model | Workload signal is more actionable than short-run bullpen outcomes. | `seeded` |
| `away_bullpen_innings_last3` | `availability` | `bullpens_daily` | `early` | `keep` | `high` | totals model | Workload signal is more actionable than short-run bullpen outcomes. | `seeded` |
| `home_bullpen_b2b` | `availability` | `bullpens_daily` | `early` | `keep` | `high` | totals model | Back-to-back usage is a clean fatigue and availability flag. | `seeded` |
| `away_bullpen_b2b` | `availability` | `bullpens_daily` | `early` | `keep` | `high` | totals model | Back-to-back usage is a clean fatigue and availability flag. | `seeded` |
| `home_bullpen_runs_allowed_last3` | `volatility` | `bullpens_daily` | `early` | `downweight` | `low` | totals model | Short-run bullpen outcomes are noisy and likely weaker than workload or leverage availability. | `seeded` |
| `away_bullpen_runs_allowed_last3` | `volatility` | `bullpens_daily` | `early` | `downweight` | `low` | totals model | Short-run bullpen outcomes are noisy and likely weaker than workload or leverage availability. | `seeded` |
| `home_bullpen_earned_runs_last3` | `volatility` | `bullpens_daily` | `early` | `downweight` | `low` | totals model | Short-run bullpen outcomes are noisy and likely weaker than workload or leverage availability. | `seeded` |
| `away_bullpen_earned_runs_last3` | `volatility` | `bullpens_daily` | `early` | `downweight` | `low` | totals model | Short-run bullpen outcomes are noisy and likely weaker than workload or leverage availability. | `seeded` |
| `home_bullpen_hits_allowed_last3` | `volatility` | `bullpens_daily` | `early` | `downweight` | `low` | totals model | Outcome-based bullpen signals should not outrank workload and baseline talent. | `seeded` |
| `away_bullpen_hits_allowed_last3` | `volatility` | `bullpens_daily` | `early` | `downweight` | `low` | totals model | Outcome-based bullpen signals should not outrank workload and baseline talent. | `seeded` |
| `home_bullpen_era_last3` | `volatility` | `bullpens_daily` | `early` | `retire` | `low` | totals model | This is a high-noise summary and a leading retirement candidate. | `seeded` |
| `away_bullpen_era_last3` | `volatility` | `bullpens_daily` | `early` | `retire` | `low` | totals model | This is a high-noise summary and a leading retirement candidate. | `seeded` |
| `home_lineup_top5_xwoba` | `availability` | `lineups`, `player_game_batting`, priors | `morning` | `keep` | `high` | totals model, board surfaces | Mixed skill and availability signal; likely keep, but split the components in the rework. | `seeded` |
| `away_lineup_top5_xwoba` | `availability` | `lineups`, `player_game_batting`, priors | `morning` | `keep` | `high` | totals model, board surfaces | Mixed skill and availability signal; likely keep, but split the components in the rework. | `seeded` |
| `home_lineup_k_pct` | `availability` | `lineups`, `player_game_batting` | `morning` | `keep` | `high` | totals model, board surfaces | Good lineup-quality input once timing rules are audited. | `seeded` |
| `away_lineup_k_pct` | `availability` | `lineups`, `player_game_batting` | `morning` | `keep` | `high` | totals model, board surfaces | Good lineup-quality input once timing rules are audited. | `seeded` |
| `venue_run_factor` | `skill` | `park_factors` | `early` | `keep` | `high` | totals model, board surfaces | Stable environment baseline that should remain core. | `seeded` |
| `venue_hr_factor` | `skill` | `park_factors` | `early` | `keep` | `medium` | totals model, board surfaces | Useful park context, especially for power-driven totals environments. | `seeded` |
| `temperature_f` | `volatility` | `game_weather` | `early` | `keep` | `high` | totals model, board surfaces | Weather belongs in the numeric model, not only in explanations. | `seeded` |
| `wind_speed_mph` | `volatility` | `game_weather` | `early` | `keep` | `high` | totals model, board surfaces | Weather belongs in the numeric model, not only in explanations. | `seeded` |
| `wind_direction_deg` | `volatility` | `game_weather` | `early` | `keep` | `medium` | totals model, board surfaces | Worth keeping if direction is encoded in a baseball-useful way. | `seeded` |
| `humidity_pct` | `volatility` | `game_weather` | `early` | `downweight` | `low` | totals model | Secondary weather context that should sit below temperature and wind. | `seeded` |
| `market_total` | `volatility` | `game_markets` | `early` | `calibration-only` | `high` | totals model, calibration, product surfaces | Important calibration input, but the rework should keep it out of the core raw model logic. | `seeded` |
| `market_over_price` | `volatility` | `game_markets` | `early` | `calibration-only` | `medium` | totals model, calibration | Useful for pricing context and calibration rather than raw projection. | `seeded` |
| `market_under_price` | `volatility` | `game_markets` | `early` | `calibration-only` | `medium` | totals model, calibration | Useful for pricing context and calibration rather than raw projection. | `seeded` |
| `line_movement` | `volatility` | `game_markets`, `market_selection_freezes` | `morning` | `calibration-only` | `high` | totals model, calibration, explanations | Better used for disagreement and freshness handling than as a primary model driver. | `seeded` |

#### Code-Audited Formula And Fallback Notes

This pass audits the current full-game totals formulas against `src/features/totals_builder.py`, `src/features/common.py`, and `src/features/priors.py`.

The rows above still retain their current status labels for now, but the feature families below now have explicit code-level formulas, cadence notes, fallback behavior, and leakage notes recorded.

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
| `home_runs_rate_blended` | `keep` | `high` | Shared baseline with full-game totals and still central for first-five expected runs. |
| `away_runs_rate_blended` | `keep` | `high` | Shared baseline with full-game totals and still central for first-five expected runs. |
| `home_hits_rate_blended` | `downweight` | `medium` | Shared support feature that should stay secondary to run-quality context. |
| `away_hits_rate_blended` | `downweight` | `medium` | Shared support feature that should stay secondary to run-quality context. |
| `home_xwoba_blended` | `keep` | `high` | Stable process metric that should remain central in first-five too. |
| `away_xwoba_blended` | `keep` | `high` | Stable process metric that should remain central in first-five too. |
| `home_iso_blended` | `keep` | `medium` | Useful power context, but still secondary to broader run creation. |
| `away_iso_blended` | `keep` | `medium` | Useful power context, but still secondary to broader run creation. |
| `home_bb_pct_blended` | `keep` | `medium` | Plate discipline helps, but should not dominate. |
| `away_bb_pct_blended` | `keep` | `medium` | Plate discipline helps, but should not dominate. |
| `home_k_pct_blended` | `keep` | `medium` | Helps starter-versus-lineup strikeout interaction. |
| `away_k_pct_blended` | `keep` | `medium` | Helps starter-versus-lineup strikeout interaction. |
| `home_starter_xwoba_blended` | `keep` | `high` | Starter talent is even more central in first-five than full game. |
| `away_starter_xwoba_blended` | `keep` | `high` | Starter talent is even more central in first-five than full game. |
| `home_starter_csw_blended` | `keep` | `high` | Starter miss-bat quality should remain core. |
| `away_starter_csw_blended` | `keep` | `high` | Starter miss-bat quality should remain core. |
| `home_starter_rest_days` | `keep` | `high` | Useful readiness and workload signal. |
| `away_starter_rest_days` | `keep` | `high` | Useful readiness and workload signal. |
| `home_lineup_top5_xwoba` | `keep` | `high` | Mixed skill and availability signal that needs cleaner separation later. |
| `away_lineup_top5_xwoba` | `keep` | `high` | Mixed skill and availability signal that needs cleaner separation later. |
| `home_lineup_k_pct` | `keep` | `high` | Strong lineup-quality context for first-five starter interaction. |
| `away_lineup_k_pct` | `keep` | `high` | Strong lineup-quality context for first-five starter interaction. |
| `venue_run_factor` | `keep` | `high` | Stable environment baseline still matters in first five. |
| `venue_hr_factor` | `keep` | `medium` | Useful park context, but secondary to run baseline and starter quality. |
| `temperature_f` | `keep` | `high` | Weather should affect first-five numbers too. |
| `wind_speed_mph` | `keep` | `high` | Weather should affect first-five numbers too. |
| `wind_direction_deg` | `keep` | `medium` | Keep if encoded into baseball-useful directionality. |
| `humidity_pct` | `downweight` | `low` | Secondary weather input. |
| `market_total` | `calibration-only` | `high` | First-five market should calibrate, not drive the raw model. |
| `market_over_price` | `calibration-only` | `medium` | Use for pricing and calibration context only. |
| `market_under_price` | `calibration-only` | `medium` | Use for pricing and calibration context only. |
| `line_movement` | `calibration-only` | `high` | Better for freshness and disagreement handling than core modeling. |

Phase 1 should still decide which full-game totals decisions remain shared and which first-five-specific replacements are needed.

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
| `throws` | `availability` | `dim_players` | `early` | `keep` | `medium` | strikeout model, surfaces | Handedness context matters, but mainly as part of matchup construction. | `seeded` |
| `days_rest` | `availability` | `pitcher_starts` | `early` | `keep` | `high` | strikeout model | Workload and readiness are core strikeout-lane inputs. | `seeded` |
| `projected_innings` | `availability` | pitcher history, starter context | `morning` | `keep` | `high` | strikeout model | Leash expectation is one of the most important strikeout features. | `seeded` |
| `recent_avg_ip_3` | `skill` | `pitcher_starts` | `early` | `keep` | `medium` | strikeout model | Useful recent leash context, but should not outrank explicit projected innings. | `seeded` |
| `recent_avg_ip_5` | `skill` | `pitcher_starts` | `early` | `keep` | `medium` | strikeout model | Useful recent leash context, but should not outrank explicit projected innings. | `seeded` |
| `recent_avg_strikeouts_3` | `skill` | `pitcher_starts` | `early` | `downweight` | `medium` | strikeout model | Recent outcome form is useful, but weaker than rate and leash signals. | `seeded` |
| `recent_avg_strikeouts_5` | `skill` | `pitcher_starts` | `early` | `downweight` | `medium` | strikeout model | Recent outcome form is useful, but weaker than rate and leash signals. | `seeded` |
| `recent_k_per_batter_3` | `skill` | `pitcher_starts` | `early` | `keep` | `high` | strikeout model | Rate-based strikeout skill is stronger than raw recent K totals. | `seeded` |
| `recent_k_per_batter_5` | `skill` | `pitcher_starts` | `early` | `keep` | `high` | strikeout model | Rate-based strikeout skill is stronger than raw recent K totals. | `seeded` |
| `recent_avg_pitch_count_3` | `availability` | `pitcher_starts` | `early` | `keep` | `high` | strikeout model | Recent pitch count is a direct leash and trust signal. | `seeded` |
| `recent_whiff_pct_5` | `skill` | `pitcher_starts` | `early` | `keep` | `high` | strikeout model | Miss-bat process metric should remain central. | `seeded` |
| `recent_csw_pct_5` | `skill` | `pitcher_starts` | `early` | `keep` | `high` | strikeout model | Miss-bat and command process metric should remain central. | `seeded` |
| `recent_xwoba_5` | `skill` | `pitcher_starts` | `early` | `keep` | `medium` | strikeout model | Useful opposing-contact-quality context, but secondary to whiff and leash features. | `seeded` |
| `baseline_strikeouts` | `skill` | priors, rolling history | `early` | `keep` | `high` | strikeout model | Strong baseline anchor that should survive the rework. | `seeded` |
| `opponent_k_pct_blended` | `skill` | `team_offense_daily`, lineup summary | `morning` | `keep` | `high` | strikeout model, board surfaces | Important opponent tendency feature once lineup timing rules are explicit. | `seeded` |
| `same_hand_share` | `availability` | lineup handedness summary | `morning` | `keep` | `high` | strikeout model | Handedness mix is core for matchup-specific strikeout context. | `seeded` |
| `opposite_hand_share` | `availability` | lineup handedness summary | `morning` | `keep` | `high` | strikeout model | Handedness mix is core for matchup-specific strikeout context. | `seeded` |
| `switch_share` | `availability` | lineup handedness summary | `morning` | `keep` | `medium` | strikeout model | Useful supplement to handedness mix, but usually lower-impact than left-right counts. | `seeded` |
| `lineup_right_count` | `availability` | lineups, inferred lineup history | `morning` | `keep` | `high` | strikeout model | Concrete lineup-composition signal and should remain central. | `seeded` |
| `lineup_left_count` | `availability` | lineups, inferred lineup history | `morning` | `keep` | `high` | strikeout model | Concrete lineup-composition signal and should remain central. | `seeded` |
| `lineup_switch_count` | `availability` | lineups, inferred lineup history | `morning` | `keep` | `medium` | strikeout model | Useful supplement to lineup composition, but usually secondary. | `seeded` |
| `known_hitters` | `availability` | lineups | `morning` | `keep` | `high` | strikeout model, certainty layer | Useful certainty-adjacent count of how complete the lineup input is. | `seeded` |
| `confirmed_hitters` | `availability` | lineups | `lineup_confirmed` | `keep` | `high` | strikeout model, certainty layer | Key certainty signal and likely part of the future lineup certainty score. | `seeded` |
| `total_hitters` | `availability` | lineups, inferred lineup history | `morning` | `keep` | `high` | strikeout model, certainty layer | Useful denominator for certainty and lineup completeness. | `seeded` |
| `handedness_adjustment_applied` | `availability` | strikeout feature builder logic | `morning` | `keep` | `medium` | strikeout model | Good audit and interpretability flag for lineup-mix adjustments. | `seeded` |
| `handedness_data_missing` | `volatility` | strikeout feature builder logic | `morning` | `keep` | `high` | strikeout model, certainty layer | Missing handedness should directly reduce trust in matchup-specific adjustments. | `seeded` |

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