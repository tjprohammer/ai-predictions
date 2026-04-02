# MLB Predictor Product Excellence Plan

## Objective

Turn the current dashboard into a game-centric prediction product that is useful in three modes:

1. Pre-game decision support for the current slate.
2. Historical review of prior slates and prediction outcomes.
3. Ongoing tracking of hitters, pitchers, streaks, model quality, and market performance.

The core requirement is that the app must be excellent at both displaying and predicting game outcomes. That means the UI and the modeling roadmap have to evolve together instead of treating the frontend like a thin wrapper around raw rows.

## Current Problems

### Product problems

- The current UI still requires too much scanning and interpretation.
- Important context is split across totals, hitters, starters, weather, and game metadata.
- Prior-day review is weak, so it is hard to build trust in the model.
- The app does not yet clearly show which hitters are hot or cold.
- There is no pitcher strikeout prediction lane yet.
- There is no strong tracking layer for previous picks, outcomes, streaks, or model accuracy by market.

### UX problems

- The first screen should emphasize the slate itself, not infrastructure or logs.
- Users should see compact game cards first, then expand for deeper detail.
- Player lists need to be smaller and denser at the slate level.
- Details like full lineups, player stats, and team stats should open on demand.

### Data/model gaps

- Hitter heat and streak context is not yet surfaced in a product-quality way.
- Starter stat display is shallow because the UI only exposes a small subset of current fields.
- There is no pitcher strikeout model or feature pipeline.
- Historical outcomes are not being used as a first-class product surface.
- Some source data quality issues remain, including occasional encoding issues and sparse starter metadata.

## Product Direction

## 1. Slate-first compact board

The top-level page should be a compact game board for a selected date.

Each game row or card should show:

- Away team at home team.
- First pitch time.
- Venue and weather summary.
- Probable away starter and home starter.
- Predicted total runs.
- Market total.
- Over or under lean.
- Predicted team runs for both sides.
- A compact “top bats” summary for each team.
- A compact “starter edge / K context” summary.
- A clear status marker for `live`, `final`, or `pregame`.

This view should be readable in one screen without forcing the user through large player grids.

## 2. Click-to-expand game detail

Selecting a game should open a detailed panel or route.

The detailed game view should show:

- Full confirmed or projected lineups for both teams.
- Top hitter probabilities for each team.
- Hitter heat and streak indicators.
- Starter stats and matchup notes.
- Team offense vs opposing starter context.
- Bullpen context.
- Weather and park context.
- Totals prediction rationale.
- Historical results for the same teams and recent form.
- Final outcome once the game is complete.

This should be the place where the app becomes analysis-grade instead of just summary-grade.

## 3. Historical review and tracking

The app should make previous slates easy to review.

For any prior day, show:

- What the model predicted.
- What the market line was.
- What actually happened.
- Which hitters were recommended.
- Which hitters got a hit.
- Whether streak and hot/cold tags were directionally useful.
- Which starters beat or missed expectation.
- How totals and props performed by game.

This is essential for trust, debugging, and model iteration.

## UX Roadmap

## Phase 1. Compact slate board

Replace the large card stack with a denser board where each game is initially collapsed.

Top-level card content:

- Matchup header.
- Totals call.
- Starter names.
- Expected runs by team.
- Two to three top hitters per team.
- Quick tags such as `hot`, `cold`, `confirmed`, `market-backed`, `live`, `final`.

Interaction:

- Click the card header to expand.
- Keep only one or a few games expanded at a time.
- Preserve filters while navigating dates.

## Phase 2. Detailed game drawer or route

Add either:

- an expandable in-place detail section, or
- a dedicated route such as `/games/{game_id}`.

Recommended default:

- Use an expandable detail section first for speed.
- Move to a dedicated route once the detail surface becomes large enough.

## Phase 3. Historical and analytics views

Add dedicated views for:

- prior slates,
- model performance,
- player trend tracking,
- pitcher trend tracking,
- recommendation history.

## Data Surfaces To Add

## A. Hitter heat and streak surface

For each hitter, expose:

- current hit streak,
- hits in last 3,
- hits in last 5,
- hits in last 7,
- rolling hit rate,
- rolling xwOBA,
- rolling hard-hit rate,
- lineup slot stability,
- confirmed vs projected status.

Suggested simple presentation:

- `Hot`: strong recent hit rate and quality-of-contact over a rolling window.
- `Cold`: below-baseline recent form.
- `Neutral`: neither extreme.

This should be transparent and rule-based at first, then refined if needed.

## B. Starter detail surface

For each probable starter, expose:

- handedness,
- rest days,
- recent innings,
- recent strikeouts,
- CSW%,
- whiff%,
- xwOBA allowed,
- pitch count trend,
- recent form over last 3 and last 5 starts,
- opponent team strikeout tendency.

## C. Team matchup surface

For each team in a game, expose:

- expected runs,
- recent runs scored,
- recent hits,
- recent xwOBA,
- top-of-lineup quality,
- strikeout tendency,
- bullpen fatigue context,
- park factor context.

## D. Pitcher strikeout prediction surface

Add a new prediction lane for pitcher strikeouts.

### Input features

- pitcher recent strikeout rate,
- rolling batters faced,
- projected innings,
- opponent team K%,
- projected lineup handedness mix,
- umpire data later if available,
- park/weather context,
- pitch count and workload trend,
- rest days.

### Outputs

- predicted strikeouts,
- probability of clearing key thresholds,
- fair odds for common K lines,
- edge versus market if market data becomes available.

### New likely tables

- `game_features_pitcher_strikeouts`
- `predictions_pitcher_strikeouts`
- optional `pitcher_form_daily`

## Outcome Tracking Plan

## 1. Per-game outcome summary

For completed games, store and display:

- final score,
- actual total,
- actual winner,
- actual team runs,
- starter final line,
- bullpen final contribution.

## 2. Hitter recommendation outcome tracking

For each displayed hitter call, store and display:

- predicted hit probability,
- whether the hitter got a hit,
- current streak at pick time,
- hot/cold label at pick time,
- lineup slot at pick time,
- eventual result.

This should support reviewing whether the app is good at both picking hits and identifying streak continuation.

## 3. Model performance dashboard

Track over time:

- totals win rate vs market thresholds,
- hit prediction calibration,
- hot/cold label usefulness,
- starter K model calibration once added,
- performance by team,
- performance by park,
- performance by confirmed vs projected lineup status.

## Technical Roadmap

## Phase 0. Stabilize current board

- Keep the page compact and readable.
- Ensure the slate renders before the pipeline log.
- Add graceful empty states for dates with no games.
- Remove any layout that hides actual game content below the fold.

## Phase 1. Add game summary API

Consolidate the slate into one clear backend shape.

Recommended endpoint set:

- `GET /api/games/board?target_date=YYYY-MM-DD`
- `GET /api/games/{game_id}` for detail view
- `GET /api/history/slate?target_date=YYYY-MM-DD`
- `GET /api/players/{player_id}/trend`
- `GET /api/pitchers/{player_id}/trend`

## Phase 2. Add game detail API

The game detail response should include:

- full lineups,
- team trend snapshots,
- starter trend snapshots,
- top hit calls,
- totals rationale,
- final outcomes if complete.

## Phase 3. Add tracking tables and materialized summaries

Recommended additions:

- `prediction_outcomes_daily`
- `player_trend_daily`
- `pitcher_trend_daily`
- `model_scorecards_daily`

These can be assembled incrementally from existing tables plus final game results.

## Phase 4. Add pitcher strikeout model

- Build feature contract.
- Backfill historical strikeout outcomes.
- Train baseline model.
- Score daily slate.
- Add UI section next to totals and hitter targets.

## Compact UI Specification

## Top-level game card

Collapsed card should show:

- matchup,
- start time,
- probable starters,
- model total and market total,
- expected runs by team,
- quick hitter chips,
- quick heat indicators,
- game status.

Expanded card should show:

- full lineups,
- player stats,
- team stats,
- starter stats,
- hit streaks,
- hot/cold indicators,
- totals rationale,
- final results if available.

## Recommended visual structure

- one compact row or medium card per game,
- expandable detail region,
- badges for `hot`, `cold`, `confirmed`, `live`, `final`,
- historical day selector,
- sticky date controls,
- less hero space and more data density.

## Definition Of “Hot” And “Cold”

Start simple and visible.

### Hitter hot label draft

Mark `hot` if a hitter meets at least two of:

- hit in 4 of last 5,
- hit in 5 of last 7,
- rolling hit rate above player baseline,
- rolling xwOBA materially above player prior.

### Hitter cold label draft

Mark `cold` if a hitter meets at least two of:

- hit in 1 or fewer of last 5,
- rolling hit rate materially below baseline,
- rolling xwOBA materially below baseline.

### Pitcher hot/cold draft

Mark `hot` if recent strikeout and whiff indicators are strong with stable workload.

Mark `cold` if recent command, whiff, or contact suppression materially deteriorates.

The labels should always be accompanied by the exact underlying numbers in the expanded view.

## Recommended Build Order

## Immediate

1. Make the slate board compact and collapsible.
2. Put only the most important per-game summary on the default view.
3. Add a detail view that expands lineups, player stats, team stats, and starter stats.
4. Add prior-day result display for totals and hitter calls.

## Short-term

1. Add hitter hot/cold labels and trend summaries.
2. Add starter trend summaries.
3. Add model outcome tracking for previous days.
4. Add calibration and scorecard views.

## Mid-term

1. Add pitcher strikeout prediction model and UI lane.
2. Add richer team-vs-starter and lineup-vs-pitcher context.
3. Add saved recommendation history and performance breakdowns.

## Success Criteria

The app is successful when:

- a user can understand the current slate in under one minute,
- every game has a readable compact summary,
- clicking a game reveals complete context without cluttering the main board,
- prior-day predictions and outcomes are easy to review,
- hitters clearly show hot/cold/streak context,
- starters clearly show recent form and strikeout potential,
- the app tracks how good the predictions actually were,
- the app feels like a real baseball prediction product rather than a raw data viewer.

## Next Implementation Slice

The highest-value next slice is:

1. Collapse the current game cards into a denser slate board.
2. Add expandable game detail sections.
3. Add prior-day actual outcomes directly on each game.
4. Add hitter hot/cold and streak tags.
5. Begin the feature contract for pitcher strikeout predictions.
