# Incremental Learning Configuration

## Environment Variables for Learning Frequency

You can control how often the incremental Ultra-80 system learns from recent games using environment variables:

### `INCREMENTAL_LEARNING_DAYS`
Controls how many days back to look for games when updating the model.

**Examples:**

```bash
# Learn from last 3 days (default)
set INCREMENTAL_LEARNING_DAYS=3
run_daily_workflow.bat

# Learn from last 14 days (optimal - based on A/B testing)
set INCREMENTAL_LEARNING_DAYS=14
run_daily_workflow.bat

# Learn from last 7 days (weekly learning)
set INCREMENTAL_LEARNING_DAYS=7
run_daily_workflow.bat

# Learn from last 14 days (bi-weekly)
set INCREMENTAL_LEARNING_DAYS=14
run_daily_workflow.bat
```

## Setting Permanent Environment Variable

### Windows (PowerShell):
```powershell
# Set for current session
$env:INCREMENTAL_LEARNING_DAYS = "7"

# Set permanently for user
[Environment]::SetEnvironmentVariable("INCREMENTAL_LEARNING_DAYS", "7", "User")
```

### Windows (Command Prompt):
```cmd
# Set for current session
set INCREMENTAL_LEARNING_DAYS=7

# Set permanently
setx INCREMENTAL_LEARNING_DAYS 7
```

## Learning Frequency Recommendations

- **Daily (3 days)**: Default - good for active learning, responsive to recent changes
- **Weekly (7 days)**: More stable, less noise from day-to-day variance
- **Bi-weekly (14 days)**: Most stable, good for established models
- **Custom**: Adjust based on your needs and model performance

## How It Works

1. **Daily Workflow Runs**: Every time you run the daily workflow
2. **Loads Existing State**: Loads saved model state from `incremental_ultra80_state.joblib`
3. **Learns from Recent Games**: Fetches completed games from the last N days
4. **Updates Model**: Incrementally updates model weights, team stats, and calibration
5. **Saves Updated State**: Saves the improved model back to state file

## Monitoring Learning

The workflow logs will show:
```
📚 Learning from recent games (7 day window): 2025-08-21 to 2025-08-27
✅ Updated models from 42 recent games
💾 Saved state to incremental_ultra80_state.joblib
```

## Performance Considerations

- **More days = More computation**: Larger windows take longer to process
- **More days = More stability**: Larger windows reduce day-to-day noise
- **Fewer days = More responsive**: Smaller windows adapt faster to recent changes
- **Optimal range**: 3-14 days depending on your strategy
