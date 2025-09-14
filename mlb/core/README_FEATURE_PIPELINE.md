# Feature Engineering & Backfill Pipeline

## Purpose

Centralizes scripts that populate real whitelist features in `enhanced_games` and derives composite / penalty metrics.

## Scripts

- `working_pitcher_ingestor.py` – Populates starter ERA, WHIP, K/9, BB/9, HR/9.
- `working_team_ingestor.py` – Populates offensive season/L30/L7 metrics (RPG, ISO, power, wRC+, plus combined aggregates).
- `working_bullpen_ingestor.py` – Populates bullpen proxy metrics (ERA, FIP, combined ERA, innings, impact factor).
- `compute_composites.py` – Derives: `pitcher_strength_composite`, `offensive_power_composite`, `environmental_impact_composite`, `expected_weather_run_impact`, `park_effect_recent`.
- `lineup_penalties.py` – Derives: `home_lineup_penalty`, `away_lineup_penalty` (heuristic placeholder; to be replaced with player-level delta logic).
- `historical_backfill.py` – Drives historical backfill over a date range, invoking the ingestors + composites sequentially.

## Recommended Daily Order

1. working_pitcher_ingestor.py
2. working_team_ingestor.py
3. working_bullpen_ingestor.py
4. lineup_penalties.py
5. compute_composites.py (after base + lineup metrics)
6. whitelist_feature_audit.py (coverage report)
7. evaluate_whitelist_performance.py (model evaluation)
8. prediction generation scripts

## Historical Backfill

Example:

```
python mlb/core/historical_backfill.py --start 2025-03-20 --end 2025-09-04 --sleep 0.5
```

Use `--force` to re-run dates already populated.

## Future Enhancements (Planned)

- Replace heuristic lineup penalties with player projection deltas.
- Refine environmental impact with humidity, barometric pressure, and park-specific wind vectors.
- Introduce quality-of-opposition adjustments for pitcher & offense composites.
- Persist coverage % snapshots for longitudinal monitoring.

## Notes

All scripts idempotently `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` so they are safe to run as schema evolves.
