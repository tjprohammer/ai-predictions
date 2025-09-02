# Hitting Module (Props: HITS 0.5 / 1.5)

This adds BvP, hot streaks (L5/L10/L15), and expected PA/AB features to project hitter probabilities for ≥1 and ≥2 hits.
Outputs with EV/Kelly are stored in `hitter_prop_predictions`.

## Setup

```bash
psql mlb -f mlb/hitting/sql/schema.sql
```

Backfill the materialized views after you load historical player logs:

```bash
python mlb/hitting/backfill/backfill_hitting.py --start 2024-03-01 --end 2025-08-28
```

Daily run (after you ingest `player_props_odds`):

```bash
python -m mlb.hitting.predict.run_hitprops --date YYYY-MM-DD
```

Optional: train isotonic calibration curves:

```bash
python mlb/hitting/models/train_hits_calibration.py --start 2025-07-01 --end 2025-08-28 --out mlb/hitting/models/calibration_curves.joblib
```

## Notes

- Per-PA hit rate is an empirical-Bayes blend of L10 form (fallback L5/L15) and a small BvP weight when AB≥15, anchored
  to a league prior, then converted to per-PA and per-AB.
- ≥1 / ≥2 hit probabilities use binomial closed forms with expected AB from lineup & hand vs. team distribution.
- Extend to TB/R/RBI easily using the same scaffolding.
