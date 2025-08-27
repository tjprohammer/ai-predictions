# MLB Betting System Daily Runbook

## Complete Daily Workflow

This runbook demonstrates the enhanced daily workflow that integrates all surgical hardening improvements.

### Option 1: Full Daily Pipeline (Recommended)

```bash
# Run the complete enhanced pipeline for today
python daily_api_workflow.py --stages markets,features,predict,odds,health,prob,export,audit

# Or for a specific date
python daily_api_workflow.py --date 2025-08-23 --stages markets,features,predict,odds,health,prob,export,audit
```

### Option 2: Minimal Enhanced Pipeline (Testing/Development)

```bash
# Just run the enhanced probability system with health checks
python daily_api_workflow.py --stages odds,health,prob

# Or test without health gate
python daily_api_workflow.py --stages odds,prob
```

### Option 3: Individual Components (Debugging)

```bash
# 1. Load odds data for missing games
python load_totals_odds.py sample_odds.csv

# 2. Run health gate validation
python health_gate.py --date 2025-08-18

# 3. Generate enhanced probability predictions
python probabilities_and_ev.py --date 2025-08-18

# 4. Test API endpoints
curl http://localhost:8000/comprehensive-games
curl http://localhost:8000/latest-predictions
```

## Workflow Stages Explained

### 🏪 **markets**

- Pulls market data and base odds from APIs
- Updates `enhanced_games` table with current market information
- Sets up base totals and odds for predictions

### 🔧 **features**

- Builds enhanced features for ML prediction
- Processes pitcher stats, team performance, weather, etc.
- Prepares data for model inference

### 🤖 **predict**

- Generates base ML predictions using trained models
- Creates `predicted_total` for each game
- Stores base predictions in `enhanced_games`

### 📊 **odds**

- **NEW STAGE** - Loads comprehensive odds data
- Auto-generates market-based odds for games missing book lines
- Ensures all games have odds data in `totals_odds` table
- **Critical for enhanced predictions**

### 🛡️ **health**

- **NEW STAGE** - Validates system calibration health
- Checks Brier score (< 0.25) and ECE (< 0.05) thresholds
- **HALTS TRADING** if calibration is poor
- Uses `health_gate.py` for validation

### 🎯 **prob**

- **ENHANCED** - Generates probability predictions with EV/Kelly sizing
- Uses isotonic calibration on rolling window
- Applies sign agreement filter and betting guardrails
- **Only processes games with odds data**
- Stores results in `probability_predictions` table

### 📁 **export**

- Exports results to CSV/JSON files
- Creates daily betting recommendations
- Prepares data for external systems

### 🔍 **audit**

- Validates data quality and predictions
- Checks for anomalies and data consistency
- Logs summary statistics

## Key Files Updated

1. **`daily_api_workflow.py`** - Enhanced with `odds` and `health` stages
2. **`probabilities_and_ev.py`** - Already handles surgical hardening (exact lines, bankroll caps, etc.)
3. **`health_gate.py`** - New calibration validation
4. **`load_totals_odds.py`** - Odds data loading
5. **`app.py`** - API endpoints for UI integration

## Database Tables

- **`enhanced_games`** - Base game data and market totals
- **`totals_odds`** - Book odds data (populated by `odds` stage)
- **`probability_predictions`** - Enhanced predictions with EV/Kelly
- **`latest_probability_predictions`** - View of latest predictions

## Expected Outputs

### Successful Run:

```
🏪 Markets: Loaded 15 games with market data
🔧 Features: Built features for 15 games
🤖 Predict: Generated predictions for 15 games
📊 Odds: Loaded odds for 15 games (5 book + 10 market-based)
🛡️ Health: PASS - Brier: 0.23, ECE: 0.04 ✅
🎯 Prob: 10 games passed sign agreement, 5 betting recommendations
📁 Export: Exported betting recommendations ($515 total stake)
🔍 Audit: All validations passed
```

### Health Gate Failure:

```
🛡️ Health: FAIL - Brier: 0.31, ECE: 0.08 ❌
ERROR: Health gate validation failed - trading halted for safety
```

## API Integration

The enhanced workflow ensures these endpoints work:

- **`GET /comprehensive-games`** - All 15 games for UI
- **`GET /latest-predictions`** - Enhanced predictions with probabilities
- **`GET /comprehensive-games/today`** - Today's games with full data

## Safety Features

1. **🛡️ Health Gate** - Prevents trading with poor calibration
2. **📏 Exact Line Recording** - Records precise book totals and sources
3. **💰 Automatic Bankroll Caps** - Scales stakes to respect daily limits
4. **🔍 Sign Agreement Filter** - Only bets when model agrees with market direction
5. **📊 Latest View** - Always shows most recent predictions per game

## Testing the Flow

```bash
# Test with tomorrow's date (won't actually run tomorrow's games)
python daily_api_workflow.py --date 2025-08-18 --stages features,odds,health,prob

# This will:
# 1. ✅ Build features for games in enhanced_games
# 2. ✅ Generate market-based odds for missing games
# 3. ✅ Validate health gate (should pass with good calibration)
# 4. ✅ Generate probability predictions
# 5. ✅ Show betting recommendations
```

This provides a complete test of the surgical hardening improvements without affecting today's actual trading!
