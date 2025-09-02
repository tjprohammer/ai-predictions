# MLB Hitting Props Prediction System

A comprehensive system for predicting MLB hitting props (hits, home runs, RBIs, total bases) using advanced statistical modeling and Empirical Bayes methodology.

## System Overview

This system provides:

- **Multi-market predictions**: HITS (0.5/1.5), HR (0.5), RBI (0.5/1.5), TB (1.5/2.5/3.5)
- **Advanced feature engineering**: Form, BvP, vs-hand splits, momentum indicators
- **Empirical Bayes blending**: Combines individual and league-wide baselines
- **Calibrated probabilities**: Platt scaling for improved probability estimates
- **Expected value calculations**: EV and Kelly criterion for betting optimization

## File Structure

```
mlb/hitting/
├── sql/
│   └── schema.sql                    # Database schema with materialized views
├── features/
│   └── comprehensive_feature_builder.py  # Advanced feature engineering
├── predict/
│   ├── enhanced_hitprops_predictor.py     # Multi-market predictor
│   └── run_hitprops.py                    # Daily prediction workflow
├── backfill/
│   └── backfill_hitting.py               # Historical data backfill
├── models/
│   └── train_hits_calibration.py         # Calibration model training
└── README.md                              # This documentation
```

## Quick Start

### 1. Database Setup

```sql
-- Create database schema and materialized views
psql -U mlbuser -d mlb -f mlb/hitting/sql/schema.sql
```

### 2. Historical Data Backfill

```bash
# Refresh materialized views after loading player game logs
cd mlb/hitting/backfill
python backfill_hitting.py --refresh-only

# Generate historical predictions for 2025 season
python backfill_hitting.py --start-date 2025-03-01 --end-date 2025-08-30
```

### 3. Train Calibration Models

```bash
# Train calibration models on 2024 data
cd mlb/hitting/models
python train_hits_calibration.py --start-date 2024-04-01 --end-date 2024-10-31
```

### 4. Daily Predictions

```bash
# Run daily hitting props predictions
cd mlb/hitting/predict
python run_hitprops.py  # Today's predictions
python run_hitprops.py 2025-08-29  # Specific date
```

## Database Schema

### Core Tables

**`player_game_logs`** - Individual player performance by game

- 30+ hitting statistics (hits, HR, RBI, total_bases, etc.)
- Plate appearances, walks, strikeouts
- Game context (home/away, opponent, pitcher)

**`hitting_props`** - Generated predictions

- Player, date, market, line value
- Probability estimates (over/under)
- Confidence scores, EV calculations
- Betting odds integration

### Materialized Views

**`mv_hitter_form`** - Rolling performance metrics

```sql
-- L5/L10/L15 rolling averages
-- Recent form indicators
-- Consistency metrics
```

**`mv_bvp_agg`** - Batter vs Pitcher history

```sql
-- Head-to-head performance
-- Vs handedness splits
-- Historical outcomes
```

**`mv_pa_distribution`** - Plate appearance patterns

```sql
-- Expected PA by lineup position
-- Usage patterns by situation
-- Rest day impacts
```

## Feature Engineering

The `ComprehensiveHittingFeatureBuilder` creates 100+ features:

### Form Features (L5/L10/L15)

- Rolling batting averages
- Power metrics (ISO, HR rate)
- Plate discipline (BB%, K%)
- Consistency measures (CV)

### Batter vs Pitcher (BvP)

- Head-to-head history
- Performance vs pitcher handedness
- Similar pitcher matchups

### Expected Plate Appearances

- Lineup position adjustments
- Rest day impacts
- Game situation modifiers

### Momentum Indicators

- Hot/cold streak detection
- Recent performance trends
- Breakout/slump identification

## Prediction Methodology

### Empirical Bayes Framework

For each market, we blend:

1. **Individual baseline**: Player's historical rate
2. **League baseline**: Position/handedness average
3. **Situational adjustments**: Matchup, form, conditions

```python
# Weighted combination based on sample size confidence
final_rate = (
    individual_weight * individual_rate +
    league_weight * league_rate +
    situation_weight * situation_rate
)
```

### Market-Specific Models

**HITS Markets** (0.5, 1.5)

- Base rate from batting average
- Plate appearance expectations
- Pitcher quality adjustments

**Home Run Market** (0.5)

- HR rate with park factors
- Power vs pitcher handedness
- Recent power form

**RBI Markets** (0.5, 1.5)

- Opportunity-based modeling
- Lineup protection effects
- Team offensive context

**Total Bases Markets** (1.5, 2.5, 3.5)

- Slugging-based approach
- Multi-hit probability
- Extra base hit rates

### Calibration

Platt scaling models adjust raw probabilities:

- Trained on historical prediction vs outcome data
- Market-specific calibration
- Confidence score integration

## Usage Examples

### Generate Today's Predictions

```python
from enhanced_hitprops_predictor import EnhancedHitPropsPredictor

predictor = EnhancedHitPropsPredictor("postgresql://...")
predictions = predictor.predict_all_props()

# Top EV picks
best_picks = predictions[predictions['ev_over'] > 0.05]
print(best_picks[['player_name', 'market', 'prob_over', 'ev_over']])
```

### Custom Feature Analysis

```python
from comprehensive_feature_builder import ComprehensiveHittingFeatureBuilder

builder = ComprehensiveHittingFeatureBuilder("postgresql://...")
features = builder.build_features('2025-08-29')

# Analyze player form
player_features = features[features['player_id'] == 'trout-mike']
print(player_features[['l5_avg', 'l10_avg', 'vs_rhp_avg', 'momentum_score']])
```

### Historical Performance Analysis

```sql
-- Market performance summary
SELECT
    market,
    COUNT(*) as predictions,
    AVG(prob_over) as avg_prob,
    AVG(CASE WHEN actual_outcome = 1 THEN 1.0 ELSE 0.0 END) as hit_rate,
    AVG(ev_over) as avg_ev
FROM hitting_props hp
JOIN player_game_logs pgl USING (player_id, date)
WHERE hp.date >= '2025-04-01'
GROUP BY market;
```

## Configuration

### Database Connection

```python
DATABASE_URL = "postgresql://mlbuser:mlbpass@localhost/mlb"
```

### Market Configuration

```python
MARKETS = {
    'HITS_0.5': {'type': 'hits', 'threshold': 1},
    'HITS_1.5': {'type': 'hits', 'threshold': 2},
    'HR_0.5': {'type': 'home_runs', 'threshold': 1},
    'RBI_0.5': {'type': 'rbi', 'threshold': 1},
    'RBI_1.5': {'type': 'rbi', 'threshold': 2},
    'TB_1.5': {'type': 'total_bases', 'threshold': 2},
    'TB_2.5': {'type': 'total_bases', 'threshold': 3},
    'TB_3.5': {'type': 'total_bases', 'threshold': 4}
}
```

## Performance Monitoring

### Key Metrics

1. **Calibration**: Brier score, log loss
2. **Discrimination**: ROC AUC
3. **Profitability**: ROI, Kelly growth
4. **Coverage**: Market availability

### Validation Queries

```sql
-- Calibration by probability bucket
WITH prob_buckets AS (
    SELECT *,
           WIDTH_BUCKET(prob_over, 0, 1, 10) as bucket
    FROM hitting_props hp
    JOIN player_game_logs pgl USING (player_id, date)
    WHERE hp.date >= '2025-07-01'
)
SELECT
    bucket,
    COUNT(*) as samples,
    AVG(prob_over) as avg_predicted,
    AVG(CASE WHEN actual_outcome = 1 THEN 1.0 ELSE 0.0 END) as actual_rate
FROM prob_buckets
GROUP BY bucket ORDER BY bucket;
```

## Troubleshooting

### Common Issues

**Import Errors**

```bash
# Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/AI_Predictions"
```

**Materialized View Refresh**

```sql
-- Manual refresh if automated fails
REFRESH MATERIALIZED VIEW mv_hitter_form;
REFRESH MATERIALIZED VIEW mv_bvp_agg;
REFRESH MATERIALIZED VIEW mv_pa_distribution;
```

**Missing Data**

```python
# Check data availability
from backfill_hitting import analyze_existing_data
analyze_existing_data("postgresql://...")
```

### Performance Optimization

1. **Index player_game_logs** on (player_id, date)
2. **Refresh materialized views** nightly
3. **Partition hitting_props** by date for large datasets
4. **Cache feature computations** for frequently accessed players

## Contributing

When extending the system:

1. **Add new markets** in `enhanced_hitprops_predictor.py`
2. **Enhance features** in `comprehensive_feature_builder.py`
3. **Update schema** in `sql/schema.sql`
4. **Retrain calibration** with new data

## License

Proprietary - MLB AI Predictions System
