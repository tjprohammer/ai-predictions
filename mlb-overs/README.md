MLB Overs backend scaffold

Structure

- ingestors/
- features/
- models/
- sql/
- api/
- configs/
- scripts/

Migration plan

- Port collectors into ingestors/\* with idempotent upserts.
- Centralize feature building in features/build_features.py.
- Training + infer go in models/.
- Expose FastAPI in api/app.py.
- Use scripts/\*.ps1 on Windows to orchestrate.

games.py – Schedule, parks, umpires, SPs, total runs → upsert games.

odds_totals.py – Opening/closing totals snapshots from The Odds API → upsert market_moves.

offense_daily.py – Exact daily team offense from Statcast (rolling last‑100 possible) → upsert teams_offense_daily.

pitchers_last10.py – Probables + last 10 starts per SP → upsert pitchers_starts.

bullpens_daily.py – Daily bullpen usage/availability → upsert bullpens_daily.

lineups_weather.py – Pregame weather + confirmed lineups → upsert weather_game & lineups.

(features/formulas.py)

SP last‑10 EWMA, days rest, velo_delta_3g

Bullpen form/availability metrics

Offense 14/30/100 EWMA, splits vs hand, home/away

Weather run factor × park

Market steam (close − open)

Interaction terms per spec

(features/build_features.py)

Produce one row per game_id with all features + target y = total_runs. Save as features/train.parquet.

Modeling

Time‑series split CV

XGBoost/LGBM regressor → isotonic calibration for P(Over)

Optional residual‑to‑market model for further calibration

# run a single date

make gameday DATE=2025-08-10




-------------------------------------------------------
Database + schema

Added markets_totals table (with snapshot and close rows) and the unique arbiter for snapshots.

Confirmed indexes and FK to games.

Extended bullpens_daily table to include form & availability fields (ERA/FIP/K-BB%, HR/9, closer/setup pitches D-1, back-to-back flag).

Ingestors

Games (ingestors.games) — daily schedule with home_sp_id/away_sp_id.

Pitcher starts (ingestors.pitchers_last10) — last-10 per pitcher with CSW%, velo, xwOBA allowed, etc.

Bullpen daily (ingestors.bullpens_daily)

Scrapes MLB StatsAPI box scores.

Aggregates bullpen performance by team/day.

Tracks availability (closer/setup pitches D-1; back-to-back).

ESPN totals (ingestors.espn_totals)

Scoreboard fetch with robust team name normalization.

Inserts snapshot rows; supports close rows.

Fallback to event summary endpoint so we capture totals when the scoreboard is missing them.

Features

features/build_features.py now builds a single row per game with:

Market line: latest snapshot (else close) as k_close.

Team offense (today): xwOBA/ISO/BB%/K% for home and away, keyed via nickname normalization.

Bullpen form (today): ERA/FIP/K-BB%, HR/9.

Bullpen availability: closer/setup D-1 pitches; back-to-back flag.

Recent bullpen usage: 3-day window flag via helper (home_bp_use_3d, home_bp_fatigued, etc.).

Starter rolling form: last 3/5/10 CSW%, velo, xwOBA allowed, xSLG allowed for both starters.

Inference

models/infer.py produces a baseline prediction:

Uses k_close as anchor.

Adjusts with team offense form, starter form, bullpen FIP and availability.

Outputs y_pred, edge = y_pred - k_close, and calibrated over/under probabilities.

Fixed PowerShell merge issues; created a readable predictions table (CSV + console view).

Debugging / Dev hygiene

Resolved module path issues for ingestors in PowerShell.

Removed brittle SQL and column name assumptions.

Added sanity checks / debug prints for columns.

Confirmed we’re no longer defaulting k_close to 8.5 (that was flattening all results).

What’s next (priority-ordered)
Strengthen market data (lines)

Add one more free source (e.g., Covers, Rotowire page scrape) to cross-check totals.

Store multi-book snapshots (book column already in place).

Compute open → close deltas per game and include as features.

Pitcher vs Team module

New ingestor: roll the last 2 seasons of pitcher vs opponent from pitchers_starts.

Features: vs-opponent xwOBA allowed, K%, BB%, HR/9, and PA sample size.

Merge on home_sp_id/away_sp_id + opponent team.

Offense splits vs L/R

If not in teams_offense_daily, add an ingestor for vs RHP/LHP xwOBA/ISO.

Use probable starter hand to pick the right split for each lineup side.

Optional: incorporate projected lineup handedness mix if you scrape lineup cards later.

Park & Weather

Create/curate parks table (run factor, HR factor, altitude, roof type).

Add weather_game (temp, wind mph/dir, humidity) via a free weather page scrape by game time.

Add features: park run factor, expected temp & wind (direction-aware for stadium, if feasible).

Backfill & train a real model

Backfill a few months of games with totals + final scores.

Build a training set with all the features above and target = total_runs (or classification >/< line).

Train quick regressors (LightGBM/XGBoost/ElasticNet) and compare to the heuristic.

Add cross-validation, feature importance, and calibration.

Quality checks & reporting

Daily validation script: % of games missing totals, starter IDs, offense splits.

Simple monitoring: log prediction drift vs market; keep hit rate ledger by threshold (e.g., edges > 0.5 runs).

Export a compact “slate report” (CSV/HTML) with matchup, line, prediction, edge, confidence, key drivers.

Typical daily run order (now)
powershell
Copy
Edit
# lines + core data
python -m ingestors.games --start YYYY-MM-DD --end YYYY-MM-DD
python -m ingestors.pitchers_last10 --start YYYY-MM-DD --end YYYY-MM-DD
python -m ingestors.bullpens_daily --start YYYY-MM-DD --end YYYY-MM-DD
python -m ingestors.espn_totals --date YYYY-MM-DD

# features + predictions
python features\build_features.py --database-url $env:DATABASE_URL --out features\train.parquet
python models\infer.py --database-url $env:DATABASE_URL --out predictions_today.parquet

python predict_from_range.py --start 2025-08-15 --end 2025-08-16 --thr 1.5

$env:PYTHONIOENCODING="utf-8"; $env:PYTHONUTF8="1"; python daily_api_workflow.py --date 2025-08-16 --stages features,predict,probabilities

# readable slate
@'
import os, pandas as pd
from sqlalchemy import create_engine
eng = create_engine(os.getenv("DATABASE_URL","postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb"))
pred = pd.read_parquet("predictions_today.parquet")
games = pd.read_sql("""
  SELECT game_id, date, away_team||' @ '||home_team AS matchup
  FROM games WHERE date=(SELECT MAX(date) FROM games)
""", eng)
df = pred.merge(games, on="game_id", how="left")
print(df[["game_id","date","matchup","k_close","y_pred","edge","p_over_cal","p_under_cal"]]
      .sort_values("edge", ascending=False).to_string(index=False))
df.to_csv("predictions_readable.csv", index=False)
'@ | python -
Where you’ll see immediate improvement
After the ESPN fallback and removing the 8.5 default, you should see varied k_close and non-flat predictions.

Adding starter vs team and L/R offense splits typically moves edges by ~0.2–0.5 runs in some matchups.

Park/weather can shift totals materially (Wrigley wind days, Coors, domes).

If you want, I can drop in:

the pitcher_vs_team ingestor + feature merge,

the L/R splits ingestor (if you don’t already store them),

and a tiny LightGBM trainer to replace the heuristic once we’ve backfilled a few weeks.

What do you want to tackle first: pitcher-vs-team or offense L/R splits?