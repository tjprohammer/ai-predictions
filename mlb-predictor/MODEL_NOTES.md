# Model Notes — What We Track, What's Missing, What to Do Next

## What the Bundled App Remembers

When we package the desktop app, the seed database ships with **everything**:

- Full prior-season + current-season pitcher starts (IP, K, ERA, Statcast metrics)
- Full player batting history (hits, xBA, xwOBA, hard-hit%, barrel%)
- Rolling trend tables (pitcher_trend_daily, player_trend_daily)
- Team offense aggregates (BA, OBP, SLG, ISO, K%, xwOBA)
- Bullpen workload and performance (innings, ERA, late-inning splits)
- Park factors, weather, market lines
- All predictions, features, and model scorecards

So yes — the app "remembers" who the good pitchers are, which hitters are hot, team-level leaders, etc. That data persists across sessions and gets updated with each daily refresh.

---

## Prior-Season Data — How It Works Today

The system already loads the **single prior season** (2025) and blends it with current-season data:

| Category     | Prior Fields                           | Full-Weight Threshold |
| ------------ | -------------------------------------- | --------------------- |
| **Pitchers** | xwOBA against, CSW%, avg fastball velo | 10 starts             |
| **Hitters**  | Hit rate, xBA, xwOBA, hard-hit%, K%    | 120 PA                |
| **Teams**    | Runs/game, K%, walk%, OBP, SLG         | 30 games              |

Blending formula: `weight × current + (1 - weight) × prior`, where weight ramps from 0→1 linearly up to the threshold. Early season = mostly prior; mid-season = mostly current.

**There is NO multi-year prior mechanism.** Only one prior season is used. Career stats are not tracked.

---

## Pitcher Features — What's Actually Used by Each Lane

### Totals & First-5 Totals (the over/under models)

These models compress an **entire starting pitcher into just 2 Statcast features + rest days**:

| Feature                 | Description                              |
| ----------------------- | ---------------------------------------- |
| `starter_xwoba_blended` | Prior-blended expected wOBA allowed      |
| `starter_csw_blended`   | Prior-blended called-strike + whiff rate |
| `starter_rest_days`     | Days since last start                    |

First-5 adds derived gap features:

- `starter_xwoba_diff` — xwOBA gap between opposing starters
- `starter_csw_diff` — CSW gap between opposing starters
- `starter_quality_gap` — composite quality gap
- `starter_asymmetry_score` — 0-1 mismatch score (60% xwOBA gap, 40% CSW gap)

### Strikeouts Lane (already tracks ace-level detail)

The strikeouts model has **much richer** pitcher features:

- Season totals: starts, innings, strikeouts, K/start, K/batter
- Recent rolling: avg IP (3/5), avg K (3/5), K per batter (3/5), pitch count (3)
- Statcast rolling: whiff% (5), CSW% (5), xwOBA (5)
- Fastball velocity, throw hand, projected innings
- Opposing lineup K%, handedness splits

### Hits Lane

Only sees the opposing starter through `opposing_starter_xwoba` and `opposing_starter_csw`.

---

## The Ace Gap — What's Missing

**The core problem:** The totals and first-5 models can't tell the difference between Gerrit Cole and a #5 starter beyond a small xwOBA/CSW gap. They can't see:

| Missing Feature                     | Why It Matters                           | Data Available?                |
| ----------------------------------- | ---------------------------------------- | ------------------------------ |
| **K/9 or K%**                       | Aces strike out 10+ per 9 innings        | Yes — in `pitcher_starts`      |
| **Average IP per start**            | Aces go 6+ IP; #5 guys go 4-5            | Yes — in `pitcher_starts`      |
| **Whiff%**                          | Swing-and-miss rate = stuff quality      | Yes — in `pitcher_starts`      |
| **Hard-hit% allowed**               | Contact management                       | Yes — in `pitcher_starts`      |
| **Barrel% allowed**                 | Damage prevention                        | Yes — in `pitcher_starts`      |
| **Fastball velocity**               | Already computed in `pitcher_snapshot()` | Yes — in `pitcher_starts`      |
| **ERA / FIP**                       | Classic ace identifier                   | ERA calculable, FIP not stored |
| **Ground ball / fly ball rate**     | Batted-ball profile                      | **Not ingested**               |
| **Pitch repertoire quality**        | Pitch mix dominance                      | **Not ingested**               |
| **First-time-through-order splits** | Aces dominate 1st pass; marginals don't  | **Not tracked**                |

**Key insight:** 5 of the top 6 missing features **already exist in the database** — they're just not wired into the totals/first-5 feature contracts. This is low-hanging fruit.

---

## Recommendations

### 1. Wire Existing DB Fields into Totals/First-5 Features (HIGH PRIORITY)

These fields already exist in `pitcher_starts` and are computed by `pitcher_snapshot()` but never make it into the feature builders:

- `starter_avg_fb_velo_blended` — fastball velocity (stuff quality)
- `starter_whiff_pct_blended` — swing-and-miss rate
- `starter_hard_hit_pct_blended` — contact quality allowed
- `starter_barrel_pct_blended` — damage prevention
- `starter_avg_ip` — average innings pitched (depth/durability)
- `starter_k_per_9` — strikeout rate (derivable from K/IP in pitcher_starts)

**Effort: Low.** The data is there. We just need to add these to `contracts.py` and the feature builders.

### 2. Prior-Year Stars — Is It Worth It? (YES, but we already do most of it)

We already blend prior-season xwOBA, CSW, and velo for pitchers. What we DON'T carry forward:

- Prior-season K-rate, IP depth, whiff%, barrel%, hard-hit%
- Career numbers (only 1 prior year)

**Recommendation:** Extend `build_pitcher_priors()` to also aggregate K/9, whiff%, hard-hit%, barrel%, and avg IP from the prior season. This gives us ace identification from day 1 of a new season when current-season data is thin.

**Multi-year career priors are NOT worth it yet.** The single prior season covers 95% of the value. Career stats would require backfilling 3-5 years of data and add complexity with minimal gain at this stage.

### 3. First-5 Innings — Ace Impact is Outsized Here (CRITICAL)

First-5 totals are ~95% starter-driven (no bullpen). This is where ace identification matters most. The current model only sees xwOBA + CSW, which compresses the true ace advantage. A Cole vs. #5 starter first-5 should show dramatically different run expectations.

**Priority order for first-5:**

1. Add `starter_avg_ip_blended` — if a guy averages 6.5 IP, he's going 5 innings almost always
2. Add `starter_k_per_9_blended` — K-rate directly suppresses early-inning scoring
3. Add `starter_whiff_pct_blended` — refined swing-and-miss dominance
4. Add `starter_avg_fb_velo_blended` — stuff indicator

### 4. Derived "Starter Quality Tier" Feature (MEDIUM PRIORITY)

Create a composite categorical feature:

- **Ace** (top 15%): sub-0.280 xwOBA, 28%+ CSW, 96+ mph, 9+ K/9
- **Quality** (top 40%): sub-0.310 xwOBA, 26%+ CSW
- **Mid-rotation** (middle)
- **Back-end** (bottom 25%)

This gives the model a discrete signal to key off of for matchup asymmetry.

---

## Current Model Performance Reality Check

As of 2026-04-06 (from backtest benchmarks):

| Lane               | vs. Baselines                                          | Status                                  |
| ------------------ | ------------------------------------------------------ | --------------------------------------- |
| **Strikeouts**     | ~4.5% MAE improvement over baselines                   | **Only lane clearly beating baselines** |
| **Totals**         | Loses to median/team_average baselines                 | Needs work                              |
| **First-5 Totals** | Loses to team_average (wins only high-asymmetry slice) | Needs ace features                      |
| **Hits**           | Loses to base_rate                                     | Needs work                              |

The ace-feature additions are most likely to help **first-5 totals** since that lane already wins when starter asymmetry is high — giving it more features to detect asymmetry should widen that edge.

---

## Summary: What To Build Next

1. **Quick win (1-2 hours):** Surface existing `whiff_pct`, `hard_hit_pct`, `barrel_pct`, `avg_fb_velo` from `pitcher_starts` into totals/first5 feature contracts
2. **Medium effort (half day):** Derive and add `starter_avg_ip`, `starter_k_per_9` features; extend prior-season blending to cover these new fields
3. **Larger effort (1-2 days):** Starter quality tier feature, retrain models, run sliced evaluation to measure improvement
4. **Not now:** Multi-year career priors, pitch repertoire data, first-time-through-order splits
