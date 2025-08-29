# Ultra-80 Performance Tuning Guide

## Environment Variables for Better Predictions

### Learning Rate & Sensitivity
```bash
# Make model more responsive to recent changes
set INCREMENTAL_LEARNING_DAYS=14       # Use 14-day windows (A/B tested optimal)
set PI_SAFETY=1.05                     # Tighter prediction intervals
```

### Variance & Confidence Tuning
```bash
# Adjust prediction confidence based on park factors
# These are in the system config - you can modify them in the code:

var_boost = {
    'pf': 0.12,     # Park factor variance boost (increase for more uncertainty in hitter-friendly parks)
    'wind': 0.10,   # Wind variance boost (increase for more uncertainty in windy conditions)  
    'heat': 0.06    # Temperature variance boost (increase for more uncertainty in extreme heat)
}
```

### Coverage Target
```bash
# The system targets 80% coverage by default
# You can adjust this for tighter/wider intervals:

coverage_target = 0.80  # 80% (default)
coverage_target = 0.85  # 85% (wider intervals, more conservative)
coverage_target = 0.75  # 75% (tighter intervals, more aggressive)
```

## Performance Optimization Strategies

### 1. Feature Engineering
- **Weather Integration**: Ensure temperature, wind, humidity data is complete
- **Park Factors**: Verify park factor coverage (currently shows some missing data)
- **Pitcher Handedness**: Complete lefty/righty splits
- **Recent Form**: 7/14/21 day rolling averages for teams and pitchers

### 2. Model Ensemble Strategy
The system uses multiple approaches - you can weight them differently:

```python
# Current blend in predict_future_slate():
pred_total = 0.6*sgd_total + 0.4*rf_total  # 60% SGD, 40% Random Forest

# For better accuracy, try:
pred_total = 0.7*sgd_total + 0.3*rf_total  # More weight to SGD
# OR
pred_total = 0.5*sgd_total + 0.5*rf_total  # Equal weight
```

### 3. Market Calibration
- **Isotonic Calibration**: Maps model differences to actual probabilities
- **Recent Data**: Uses last 800-1000 games for calibration
- **Bucket-based**: Different calibration for low/mid/high total games

### 4. Conformal Prediction Intervals
- **Context-Aware**: Different intervals for different game types
- **Adaptive**: Adjusts based on recent model performance
- **Bucketed**: Low total games (≤7.5), Mid (7.5-10.0), High (≥10.0)

## Specific Improvement Areas

### Data Coverage Issues to Fix:
1. **Park Factors**: Some games missing park data
2. **Handedness Splits**: vs RHP/LHP data sometimes missing  
3. **Rolling Stats**: Ensure 7/14/21 day windows are complete
4. **Market Data**: Opening totals vs closing totals consistency

### Model Tuning Options:
1. **Warmup Period**: Currently 200 games - could reduce to 150 for faster adaptation
2. **Batch Retraining**: Every 200 games - could increase frequency to 150
3. **Learning Rate**: SGD learning rates could be adjusted for faster/slower adaptation
4. **Regularization**: Alpha parameters could be tuned for better generalization

### Performance Monitoring:
- Track daily MAE, coverage, and ROI
- Monitor conformal interval width (too wide = underconfident, too narrow = overconfident)
- Watch for drift in market calibration
- Check feature importance over time
