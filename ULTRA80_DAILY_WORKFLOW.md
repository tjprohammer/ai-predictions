# Ultra 80 Daily Workflow Setup

## Quick Start (2-Job Workflow)

The Ultra 80 system is now set up as two simple commands for daily use:

### 🚀 One-Time Bootstrap

Run this ONCE to initialize the system with historical data:

```bash
bootstrap_ultra80.bat
```

This trains on ~60 days of historical data and saves the initial state.

### 🌙 Nightly Update (Automated)

Run this after each day's games are complete (schedule for 3AM):

```bash
nightly_update.bat
```

This updates models with yesterday's results and saves the updated state.

### 🎯 Pregame Slate (Manual)

Run this anytime to see today's predictions (fast, no training):

```bash
pregame_slate.bat
```

This loads the saved state and generates predictions for today's games.

## Output Files

### CSV Predictions

- `outputs/slate_YYYY-MM-DD_predictions_*.csv` - All games with predictions, intervals, EV
- `outputs/backtest_predictions_*.csv` - Historical backtest results

### One-Pager Summary

- `outputs/onepager_YYYY-MM-DD.md` - Top 3 recommended bets with stake guide

### Console Output

- Pretty-printed table of all today's games sorted by EV and trust
- Recommended bets with trust scores, EV percentages, and book info

## Environment Variables (Advanced)

You can also run manually with environment variables:

### Bootstrap (60-90 days)

```bash
set START_DATE=2025-06-01
set END_DATE=2025-08-01
set RUN_MODE=TRAIN_ONLY
python mlb-overs\pipelines\incremental_ultra_80_system.py
```

### Nightly Update (yesterday only)

```bash
set START_DATE=2025-08-26
set END_DATE=2025-08-27
set RUN_MODE=TRAIN_ONLY
python mlb-overs\pipelines\incremental_ultra_80_system.py
```

### Pregame Slate (today)

```bash
set SLATE_DATE=2025-08-27
set RUN_MODE=SLATE_ONLY
python mlb-overs\pipelines\incremental_ultra_80_system.py
```

## Data Requirements

Before running pregame_slate.bat, ensure these tables are updated:

- `daily_games` - Today's matchups
- `totals_odds` - Today's market totals and odds
- Optional: `pitcher_rolling_stats`, `team_rolling_stats` for latest data

## Key Features

✅ **Fixed Calibration Leak** - No look-ahead bias in probability calibration  
✅ **Odds Line Matching** - Odds matched to exact total being evaluated  
✅ **Missing Data Detection** - Proper uncertainty widening when data is missing  
✅ **WHIP Defaults** - Correct 1.30 default instead of 4.5  
✅ **Conformal Prediction** - 80% coverage target with finite-sample correction  
✅ **Chronological Backtesting** - Fair historical evaluation without future leakage

## Troubleshooting

### "Model not ready" error

- Run `bootstrap_ultra80.bat` first to initialize the system
- Check that `incremental_ultra80_state.joblib` exists in the root directory

### High/Low predictions

- System may need more training data if predictions seem off
- Check the console output for coverage, MAE, and ROI metrics

### No games found

- Verify `daily_games` table has today's matchups
- Check `SLATE_DATE` format is YYYY-MM-DD

## Scheduling (Windows Task Scheduler)

1. **Nightly Update**: Schedule `nightly_update.bat` for 3:00 AM daily
2. **Pregame Slate**: Run `pregame_slate.bat` manually or schedule for desired time

The system is designed to be simple, fast, and reliable for daily baseball prediction workflows.
