# StatMuse Matchup Data Integration Plan

**Created:** 2026-04-08  
**Source:** https://www.statmuse.com/mlb  
**Status:** Planning

---

## Overview

StatMuse is a free natural-language sports stat engine backed by Retrosheet + live MLB data. It exposes deterministic URL patterns that return structured stat tables — no API key required. We can query career and season-level matchup data that we currently have **zero** coverage on for most of our prediction models.

**Goal:** Ingest batter-vs-pitcher, batter-vs-team, platoon (L/R), and pitcher-vs-team matchup history from StatMuse. Use it to (1) improve model features for hits, strikeouts, and totals, and (2) surface matchup context on the game detail page and a new dedicated matchup page in the UI.

---

## What StatMuse Gives Us

### URL Patterns

| Query Type | URL Pattern | Returns |
|---|---|---|
| Batter vs. Team (career) | `/mlb/ask/{player}-stats-vs-{team}` | Career game log: G, AB, H, HR, RBI, BB, SO, TB per game |
| Batter vs. Pitcher (career) | `/mlb/ask/{batter}-vs-{pitcher}-career` | Career head-to-head: AB, H, HR, K, BB |
| Batter platoon splits | `/mlb/ask/{player}-vs-left-handed-pitchers-{year}` | Season split: AVG, HR, RBI, PA, OBP, SLG, OPS |
| Pitcher vs. Team (career) | `/mlb/ask/{pitcher}-vs-{team}` | Career log: G, IP, K, BB, ER, ERA, WHIP |
| Pitcher vs. Hitter (career) | `/mlb/ask/{pitcher}-vs-{batter}-career` | Same as BvP but from pitcher perspective |

### Data Quality
- Sourced from Retrosheet (historical) + current MLB feeds
- Full career depth, game-by-game logs
- Structured HTML tables — parseable with standard scraping
- Free tier, no API key (rate-limit respectfully)

---

## Current Feature Gaps This Fills

| Gap | Current State | Models Affected |
|---|---|---|
| **Batter vs. Pitcher (BvP)** | Nothing | Hits, Strikeouts |
| **Batter vs. Team** | Nothing | Hits, Totals |
| **L/R Platoon Splits** | Only hand-count proxies in K model | Hits, Strikeouts |
| **Pitcher vs. Team (full)** | Only `pitcher_vs_team_k_rate` in K model | Strikeouts, Totals |
| **Team Head-to-Head** | Nothing | Totals |

---

## Phase 1 — Batter vs. Opposing Starter (Highest Value)

**Why first:** This is the single biggest feature gap. A hitter's career numbers against today's starter are the most direct matchup signal we can get, and we have literally zero BvP data in any model today.

### New Features (per hitter per game)
- `bvp_career_avg` — career batting average vs. this pitcher
- `bvp_career_k_rate` — career strikeout rate vs. this pitcher
- `bvp_career_hr_rate` — career HR rate vs. this pitcher
- `bvp_career_pa` — plate appearances (sample-size weight)
- `bvp_career_obp` — on-base percentage vs. this pitcher
- `bvp_blended_avg` — weighted blend of BvP avg and season avg (decays toward season when PA < threshold, e.g. 15 PA)

### Model Integration
- **Hits model:** `bvp_career_avg`, `bvp_career_pa`, `bvp_blended_avg` as core predictors
- **Strikeouts model:** `bvp_career_k_rate`, `bvp_career_pa` as core predictors (hitter-side K tendency vs. this specific pitcher)
- **Totals model:** Aggregate `lineup_bvp_avg` (mean BvP avg across confirmed lineup) as a team-level totals feature

### Queries Per Slate
- ~9 hitters × 2 teams × 15 games = ~270 queries
- Cached by `(batter_id, pitcher_id)` — only need to re-query once per season or when career row ages out
- At 1 req/sec = ~4.5 min

---

## Phase 2 — Platoon Splits (L/R)

**Why second:** We already know the opposing starter's throwing hand. Hitters perform very differently against same-side vs. opposite-side pitchers. Currently only the K model has rough hand-count proxies.

### New Features (per hitter per game)
- `hitter_avg_vs_hand` — this year's AVG vs. LHP or RHP (matching today's starter)
- `hitter_ops_vs_hand` — OPS vs. that hand
- `hitter_k_pct_vs_hand` — K% vs. that hand
- `hitter_pa_vs_hand` — sample size
- `platoon_advantage` — boolean/float: does the hitter have the platoon advantage? (opposite hand = advantage)

### Model Integration
- **Hits model:** `hitter_avg_vs_hand`, `platoon_advantage` as core predictors
- **Strikeouts model:** `hitter_k_pct_vs_hand` replaces the raw `same_hand_share` proxy
- **Totals model:** `lineup_platoon_advantage_pct` (% of lineup with platoon advantage)

### Queries Per Slate
- ~270 hitters × 1 query each (current year vs. LHP or RHP)
- Heavily cacheable by `(player_id, hand, year)` — 1 query per player per season per hand
- Most hitters reappear across games, so cache hit rate should be high after first week

---

## Phase 3 — Pitcher Career vs. Opposing Team

**Why third:** Upgrades our single `pitcher_vs_team_k_rate` field to a richer matchup profile.

### New Features (per starter per game)
- `starter_vs_team_era` — career ERA vs. this opponent
- `starter_vs_team_k_per_9` — career K/9 vs. this opponent
- `starter_vs_team_whip` — career WHIP vs. this opponent
- `starter_vs_team_games` — sample size
- `starter_vs_team_avg_ip` — average innings pitched vs. this team

### Model Integration
- **Strikeouts model:** `starter_vs_team_k_per_9` and `starter_vs_team_games` (replaces derived `pitcher_vs_team_k_rate`)
- **Totals model:** `starter_vs_team_era`, `starter_vs_team_whip` (direct run-scoring signal)
- **First5 model:** Same pitcher-vs-team features (most starters face same team in early innings)

### Queries Per Slate
- 2 starters × 15 games = 30 queries
- Cached by `(pitcher_id, opponent_team, season)`
- Negligible load

---

## Phase 4 — Team Head-to-Head (Lowest Priority)

### New Features (per game)
- `h2h_season_record` — current season series record
- `h2h_recent_runs_avg` — average total runs in last N meetings
- `h2h_season_over_pct` — % of season meetings that went over market total

### Model Integration
- **Totals model only** — noisier than player-level data but captures team familiarity effects

---

## Technical Architecture

### New Ingestor: `src/ingestors/matchup_splits.py`

```
Pipeline position: After lineups + starters, before feature builders

Input:  Today's slate (games table) + lineups + pitcher_starts
Output: matchup_splits table rows

Flow:
1. For each game, get confirmed/projected lineup and opposing starter
2. Check cache — skip if (batter, pitcher) already queried this season
3. Construct StatMuse URL for each missing matchup
4. Fetch + parse HTML stat table
5. Upsert into matchup_splits table
6. Rate limit: 1 req/sec with exponential backoff on 429/5xx
```

### New DB Table: `matchup_splits`

```sql
CREATE TABLE IF NOT EXISTS matchup_splits (
    player_id     BIGINT    NOT NULL,
    opponent_id   BIGINT    NOT NULL,   -- pitcher_id for BvP, 0 for team/platoon
    split_type    VARCHAR(20) NOT NULL, -- 'bvp', 'vs_team', 'platoon_lhp', 'platoon_rhp', 'pitcher_vs_team'
    season        SMALLINT,             -- NULL for career, year for season splits
    games         SMALLINT,
    plate_appearances SMALLINT,
    at_bats       SMALLINT,
    hits          SMALLINT,
    home_runs     SMALLINT,
    walks         SMALLINT,
    strikeouts    SMALLINT,
    rbi           SMALLINT,
    batting_avg   NUMERIC(5,4),
    obp           NUMERIC(5,4),
    slg           NUMERIC(5,4),
    ops           NUMERIC(5,4),
    -- Pitcher-specific (for pitcher_vs_team)
    innings_pitched NUMERIC(5,1),
    earned_runs   SMALLINT,
    era           NUMERIC(5,2),
    whip          NUMERIC(5,3),
    k_per_9       NUMERIC(5,2),
    -- Metadata
    fetched_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source_url    TEXT,
    PRIMARY KEY (player_id, opponent_id, split_type, season)
);
```

### New Feature Builder Integration

Update existing builders to pull from `matchup_splits`:

- `src/features/hits_builder.py` — join BvP + platoon features per hitter
- `src/features/strikeouts_builder.py` — join BvP K-rate + pitcher-vs-team K/9
- `src/features/totals_builder.py` — aggregate lineup BvP avg + pitcher-vs-team ERA
- `src/features/first5_totals_builder.py` — same pitcher-vs-team features

### Caching Strategy

| Split Type | Cache Key | Refresh Frequency |
|---|---|---|
| BvP (career) | `(batter_id, pitcher_id, 'bvp', NULL)` | Once per season (career stats change slowly) |
| Platoon | `(player_id, 0, 'platoon_lhp', 2026)` | Weekly (current season stats evolve) |
| Pitcher vs. Team | `(pitcher_id, team_code, 'pitcher_vs_team', NULL)` | Once per series |

### Rate Limiting
- 1 request per second baseline
- Exponential backoff on HTTP 429 / 5xx
- Circuit breaker: if 5 consecutive failures, skip remaining and log warning
- Total budget: ~300 queries per daily slate = ~5 minutes

---

## UI Integration

### 1. Game Detail Page Enhancement (`/game?id=...`)

Add a **Matchup History** section to each hitter's row in the existing game detail lineup view:

```
[Hitter Name] vs. [Opposing Starter]
Career: 8-for-24 (.333), 2 HR, 6 K | vs LHP this year: .290/.380/.520
```

**Fields to show per hitter:**
- BvP career line: PA, AVG, HR, K, OBP, SLG
- Platoon split for today's matchup (vs LHP or vs RHP)
- Sample size indicator (e.g. "small sample" badge if PA < 10)
- Visual: green/red shading based on BvP avg relative to season avg

**Fields to show for starters:**
- Pitcher vs. opposing team career: G, IP, ERA, K/9, WHIP
- Recent trend vs. this team (last 3 starts if available)

### 2. Dedicated Matchup Page (`/matchups`)

New standalone page at `/matchups` that shows all matchup data for today's slate:

**Layout:**
- Game selector (dropdown or cards for each game)
- Per-game view with two panels (home team / away team)
- Each panel shows:
  - **Starter profile vs. opposing lineup** — career stats against each confirmed hitter
  - **Lineup matchup grid** — each hitter's BvP line + platoon split
  - **Aggregate matchup score** — "lineup familiarity" rating based on total PA against this pitcher
  - **Platoon breakdown** — % of lineup with platoon advantage
  - **Historical context** — last 3 meetings between these teams (score, total runs)

**API Endpoint:** `/api/matchups/game/{game_id}`

Returns:
```json
{
  "game_id": 12345,
  "home_team": "NYY",
  "away_team": "BOS",
  "home_starter": {
    "name": "Gerrit Cole",
    "vs_team_career": { "games": 18, "era": 3.21, "k_per_9": 10.4, "whip": 1.05 },
    "vs_lineup": [
      {
        "hitter": "Rafael Devers",
        "bvp": { "pa": 32, "avg": ".281", "hr": 3, "k": 8 },
        "platoon": { "avg_vs_hand": ".305", "ops_vs_hand": ".890", "pa": 142 }
      }
    ]
  },
  "away_starter": { ... },
  "team_h2h_recent": [ ... ],
  "matchup_scores": {
    "home_lineup_familiarity": 0.72,
    "away_lineup_familiarity": 0.45,
    "home_platoon_advantage_pct": 0.56,
    "away_platoon_advantage_pct": 0.67
  }
}
```

### 3. Matchup Slate Overview (on `/matchups` landing)

Before drilling into a game, show a slate-level overview:

| Game | Away Starter vs. Home Lineup | Home Starter vs. Away Lineup | Platoon Edge |
|---|---|---|---|
| NYY @ BOS | Cole: 3.21 ERA vs BOS (18 G) | Sale: 2.85 ERA vs NYY (22 G) | BOS +12% |
| LAD @ CHC | Yamamoto: 4.10 ERA vs CHC (3 G) | Hendricks: 3.90 ERA vs LAD (5 G) | Even |

---

## Pipeline Integration

### Refresh Everything Step Placement

```
Current steps (20):
  games → starters → slate → lineups → player_status →
  market_totals → weather → freeze_markets → validator →
  offense_daily → bullpens_daily →
  totals_builder → first5_builder → hits_builder → strikeouts_builder →
  predict_totals → predict_first5 → predict_hits → predict_strikeouts →
  product_surfaces

With matchup_splits (21 steps):
  games → starters → slate → lineups → player_status →
  market_totals → weather → MATCHUP_SPLITS → freeze_markets → validator →
  offense_daily → bullpens_daily →
  totals_builder → first5_builder → hits_builder → strikeouts_builder →
  predict_totals → predict_first5 → predict_hits → predict_strikeouts →
  product_surfaces
```

Matchup splits run after lineups (need to know who's playing) and before feature builders (need to feed features).

### Backfill Strategy

For historical data (2025-2026 seasons already in DB):
- Career BvP and pitcher-vs-team stats are time-invariant at season level
- Can backfill in bulk: extract unique `(batter, pitcher)` pairs from `lineups` + `pitcher_starts`, then batch-query StatMuse
- ~2000 unique batter-pitcher pairs per season × 1 req/sec = ~33 minutes one-time

---

## Success Metrics

After integration, evaluate on the existing backtest harness:

| Model | Current Best Baseline | Target Improvement |
|---|---|---|
| **Hits** | Base rate wins | BvP + platoon should give first real edge over base rate |
| **Strikeouts** | ~4.5% MAE improvement | Pitcher-vs-team + hitter BvP K-rate should widen the gap |
| **Totals** | Median/team_avg wins | Lineup BvP aggregate + pitcher-vs-team ERA may close the gap |
| **First5** | Wins high-starter-asymmetry slice | Pitcher-vs-team features should sharpen this further |

---

## Implementation Order

1. **DB migration** — `matchup_splits` table
2. **Ingestor** — `src/ingestors/matchup_splits.py` with HTML parsing + caching
3. **Feature integration** — wire into hits, strikeouts, totals, first5 builders
4. **Pipeline step** — add to refresh_everything between lineups and freeze_markets
5. **Backfill** — run one-time historical BvP extraction
6. **Retrain models** — evaluate feature importance and backtest improvement
7. **API endpoint** — `/api/matchups/game/{game_id}`
8. **Game detail UI** — add matchup section per hitter
9. **Matchups page** — new `/matchups` route with slate overview + per-game drill-down
10. **Polish** — sample-size badges, color coding, platoon advantage indicators
