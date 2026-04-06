# MLB Model Rework Plan

## Purpose

This document defines the next modeling rework for `mlb-predictor`.

The goal is to make the app behave like a pregame decision engine instead of a flat stat aggregator. The system should use only pregame-safe information, react correctly to late-cycle input changes, and separate raw projection strength from trust in the current input state.

This rework applies to:

- full-game totals
- first-five totals where market coverage exists
- player 1+ hit props
- starter strikeout props

## Primary Product Change

The app should no longer present every prediction as if it is equally trustworthy.

Every prediction should produce two outputs:

1. **Projection**
	 - expected total runs
	 - expected hitter outcome
	 - expected strikeout outcome
2. **Certainty**
	 - how reliable the current inputs are
	 - whether the play is actionable now or should wait for better confirmation

This is the core system change: edge and certainty must be separate. A strong edge with weak inputs should not be displayed the same way as a strong edge with confirmed inputs.

## Design Principles

### Skill Is Not Certainty

A team or player can project well while the input quality is still weak. Probable starter plus projected lineup should not carry the same trust as confirmed starter plus confirmed lineup.

### Recent Form Is Supporting Evidence

Hot streaks and short-run outcomes should only receive extra weight when they agree with stronger indicators such as xwOBA, hard-hit rate, lineup slot, and matchup quality.

### Bullpen State Must Be Explicit

Bullpen quality and bullpen freshness are separate. A strong bullpen that is taxed is not equivalent to a strong rested bullpen.

### Market Data Calibrates Rather Than Dictates

The market should inform disagreement handling, confidence, and closing-line analysis. It should not become the core prediction engine.

### The Same Game Can Evolve Across The Day

Night-before, morning, lineup-confirmed, and pre-lock views should be treated as distinct decision states with different feature allowances and confidence ceilings.

## Board-State Spec

The prediction system should operate in explicit board states.

### State 1: Early Board

Typical use:

- night before
- very early same day

Allowed inputs:

- probable starters
- early market
- baseline team offense and pitching
- park factors
- forecast weather
- recent bullpen usage

Rules:

- predictions allowed
- certainty capped at `C`
- matchup shell emphasized over hard plays

### State 2: Morning Board

Typical use:

- same-day morning before lineups confirm

Allowed inputs:

- updated probable or confirmed starters
- refreshed market
- improved weather
- bullpen state
- projected lineups

Rules:

- predictions allowed
- certainty capped at `B` unless lineups confirm unusually early

### State 3: Lineup-Confirmed Board

Typical use:

- official batting orders posted

Allowed inputs:

- confirmed nine-man lineups
- batting-order slots
- handedness mix
- replacement-player penalties
- confirmed starters
- refreshed market
- updated weather

Rules:

- full ranking logic enabled
- certainty can reach `A`
- hitter and strikeout reranking should be materially more aggressive here

### State 4: Pre-Lock Final

Typical use:

- near first pitch

Allowed inputs:

- latest market
- latest weather
- confirmed lineups
- confirmed starters
- finalized bullpen usage context through prior games

Rules:

- final publish, downgrade, or suppress decision
- final confidence label

## Model Architecture

### Totals Stack

Recommended structure:

1. predict away-team expected runs
2. predict home-team expected runs
3. combine them into a projected total
4. compare versus market total
5. apply calibration and confidence gating

Why this structure:

- it is easier to debug than a single flat total model
- it lets starter, lineup, bullpen, weather, and park effects remain interpretable by side
- it prevents totals logic from becoming a market-copying shortcut

### Hitter Stack

Recommended structure:

- projected plate appearances
- lineup slot value
- confirmed lineup bonus
- platoon matchup quality
- opposing starter profile
- recent contact-quality support

Rule:

If lineup certainty is weak, the hitter can still appear internally, but ranking and confidence should drop materially.

### Strikeout Stack

Recommended structure:

- expected leash
- recent pitch count
- opponent lineup strikeout tendencies
- opponent handedness mix
- pitcher whiff and CSW indicators
- starter certainty

Rule:

If leash risk is elevated or starter identity is not firmly confirmed, suppress aggressive strikeout calls.

## Feature Framework

Every feature should belong to one of three buckets.

### Skill Features

These describe actual baseball ability.

Examples:

- pitcher xwOBA allowed
- whiff rate
- CSW
- velocity trend
- team xwOBA
- lineup top-five strength
- hitter hard-hit rate
- handedness splits

### Availability Features

These describe whether the expected talent is likely to show up in the current game context.

Examples:

- confirmed starter flag
- confirmed lineup flag
- batting-order slot
- bullpen availability
- recent pitch count
- days rest
- roof status

### Volatility Features

These describe uncertainty or instability.

Examples:

- weather freshness
- market freshness
- probable versus confirmed starter
- projected versus confirmed lineup
- stale feed flags
- missing-data fallback count

This separation should exist in both the feature inventory and the product logic so the app does not blend baseball skill with betting-context quality.

## Core Rework Areas

### Starter Features

Base inputs:

- confirmed versus probable starter
- days rest
- recent pitch count
- rolling innings, strikeouts, and walks
- xwOBA allowed
- whiff percentage
- CSW percentage
- velocity trend
- handedness-adjusted opponent profile

Derived targets:

- starter certainty score
- pitch leash score
- velocity warning flag
- command risk flag
- handedness-adjusted opponent threat score

### Lineup Features

Base inputs:

- confirmed lineup flag
- projected lineup flag
- batting-order slot
- top-five lineup xwOBA
- lineup K percentage
- lineup handedness mix
- missing-regular penalty
- bench or replacement-player penalty

Derived targets:

- lineup certainty score
- top-order strength
- bottom-third weakness
- platoon pressure score

### Bullpen Features

Base inputs:

- last three-day pitches
- last three-day innings
- back-to-back usage
- leverage-arm usage
- available rest days
- season baseline bullpen quality

Derived targets:

- bullpen fatigue index
- high-leverage availability score
- late-inning risk score
- bullpen asymmetry flag

Important rule:

Do not lean on short-run bullpen outcome stats like recent runs allowed or recent bullpen ERA proxies when better workload and talent signals are available.

### Park And Weather Features

Base inputs:

- park run factor
- park home-run factor
- temperature
- humidity
- wind speed
- wind direction
- roof status
- weather observation timestamp

Derived targets:

- run environment score
- home-run carry score
- weather freshness score
- weather uncertainty flag

### Hitter And Team-Offense Form Features

Base inputs:

- 7, 14, and 30-day hit rate
- 14-day xBA
- 14-day xwOBA
- 14-day hard-hit percentage
- recent team runs
- recent team xwOBA
- recent team strikeout rate

Derived targets:

- hitter heat score
- hitter regression-risk score
- team offense trend score

Important rule:

Recent form should not materially upgrade a play unless it agrees with stronger process metrics and confirmed opportunity.

## Data Contracts

Before implementation, every feature must define:

- exact feature name
- exact formula
- source table or source API
- refresh cadence
- earliest safe usage time
- whether it is pregame-safe
- missing-data fallback
- leakage risk
- downstream model consumers

No new feature should be added to totals, hits, or strikeouts without this contract.

## Confidence And Decision Policy

Confidence should be modeled separately from raw projection.

Primary inputs:

- starter certainty
- lineup certainty
- weather freshness
- market freshness
- bullpen completeness
- missing-data count
- board state
- inferred-versus-confirmed input count

### Confidence Labels

`A`

- confirmed starters
- confirmed lineups
- fresh market
- fresh weather
- complete bullpen context

`B`

- one key area still inferred

`C`

- multiple important fields still projected or stale
- suitable for watchlist use, not strong action

`D`

- major missing input
- may exist internally, but should be de-emphasized or suppressed on main surfaces

### Publish

Publish when:

- edge exceeds the threshold for that market
- confidence is high enough
- no major uncertainty flags are active

### Downgrade

Downgrade when:

- starter is not confirmed
- lineup is still projected
- weather is stale
- market is stale
- bullpen context is incomplete

### Suppress

Suppress when:

- multiple critical inputs are inferred
- key feature fallbacks were used too often
- lineup-slot-dependent player props are unconfirmed
- disagreement with market is large but certainty is weak

## Backtesting Spec

Backtest by board state, not only by game outcome.

Required slices:

- probable starter versus confirmed starter
- projected lineup versus confirmed lineup
- indoor versus outdoor
- cold versus warm
- wind in versus wind out
- stale market versus fresh market
- high-certainty versus low-certainty
- month and season phase

Primary metrics:

- totals MAE
- win rate versus market side
- probability calibration by bucket
- CLV direction
- CLV magnitude

Secondary metrics:

- high-certainty-only performance
- degradation when inputs are incomplete
- stability by month
- performance when the model most disagrees with market

Success should be judged on decision quality in high-trust spots, not only on average MAE.

## Project Mapping

This rework should map into the existing codebase as follows:

- `db/migrations/`
	- schema additions for certainty, freshness, and bullpen context
- `src/features/`
	- certainty features, lineup transforms, bullpen transforms, and park or weather transforms
- `src/models/`
	- reworked totals modeling, hitter ranking logic, strikeout modeling, and confidence modeling
- `src/backtest/`
	- board-state backtests, certainty-sliced evaluation, and old-versus-new comparisons
- `src/transforms/`
	- confidence labels, publish or suppress logic, and explanation drivers
- `src/api/`
	- prediction payload updates, certainty explanations, and product-surface output contracts

## Implementation Phases

### Phase 1: Audit Current Features

Deliverables:

- totals feature inventory
- hits feature inventory
- strikeout feature inventory
- stale, redundant, and leakage-risk review

Current decisions from the audit so far:

- full-game totals should be treated as a run-environment model built from team scoring baselines, starter quality, lineup quality, and bullpen availability or workload first
- first-five totals should be treated as a starter-plus-top-of-lineup model first, with park, weather, and market inputs in supporting roles
- player hits should be treated as an opportunity-plus-contact-quality model first, with lineup slot, projected plate appearances, xBA, xwOBA, hard-hit rate, and opposing starter context ahead of streak metrics
- pitcher strikeouts should be treated as a leash-plus-bat-missing-skill model first, then adjusted by opponent strikeout tendency and lineup handedness or confirmation
- bullpen workload features survive for full-game totals, but short-run bullpen outcome summaries should be downweighted or retired
- market level and line movement remain calibration inputs; side prices stay secondary and should not drive the raw projection layer
- projected and inferred lineups are pregame-safe but should feed certainty scoring and actionability, not be treated as equivalent to confirmed lineups
- realized same-game fields such as strikeouts-side `opponent_lineup_k_pct` are postgame diagnostics only and must stay out of the pregame model core

Immediate implementation sequence implied by the audit:

1. preserve and formalize the core surviving features for each lane
2. classify every field as core predictor, calibration input, certainty signal, diagnostic flag, or product-only field
3. retire or downweight noisy bullpen outcomes, raw streak fields, and weak weather or price-detail features
4. replace proxy features such as hits-side `team_run_environment` with clearer opportunity-aware versions
5. build certainty scores around starter identity, lineup completeness, market freshness, weather freshness, and fallback counts

### Phase 2: Build The Certainty Layer

Deliverables:

- starter certainty score
- lineup certainty score
- weather freshness score
- market freshness score
- bullpen completeness score

### Phase 3: Rework Totals Features

Deliverables:

- team-level expected-runs structure
- bullpen fatigue inputs
- lineup-quality transforms
- park and weather transforms

### Phase 4: Rework Hitter And Strikeout Stacks

Deliverables:

- lineup-slot-aware hitter logic
- platoon-aware hitter weighting
- leash-aware strikeout logic
- certainty-aware reranking

### Phase 5: Calibration And Product Logic

Deliverables:

- confidence buckets
- publish and suppress rules
- explanation drivers
- actionability rules by board state

### Phase 6: Backtest Old Versus New

Deliverables:

- MAE comparison
- win-rate comparison
- certainty-bucket comparison
- CLV comparison
- board-state comparison

## Immediate Deliverables

The rework should start with three concrete artifacts plus one product contract:

1. **Feature dictionary**
	 - exact formula
	 - source table or API
	 - refresh timing
	 - pregame-safe status
	 - missing fallback
	 - leakage risk
2. **Board-state spec**
	 - early board
	 - morning board
	 - lineup-confirmed board
	 - pre-lock final
3. **Decision policy**
	 - publish rules
	 - downgrade rules
	 - suppress rules
	 - rerank triggers after confirmation
4. **Backtest harness**
	 - old versus new comparisons split by board state and confidence bucket

## Definition Of Done

This rework is complete when:

- the system outputs both projection and certainty
- confidence is visible in product surfaces
- late-cycle updates can change both confidence and ranking
- high-certainty plays are measurably stronger than low-certainty plays
- the backtest shows better reliability on actionable spots than the current system