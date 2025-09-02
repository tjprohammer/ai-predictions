# MLB Prediction System Improvement Action Plan

## 🎯 **MISSION: Improve from 48.8% to 55%+ Accuracy**

**Current State:**

- MAE: 3.89 runs (Target: <2.5)
- Accuracy within 1 run: 21% (Target: >60%)
- Bias: -0.66 runs (under-predicting)
- Weak feature correlations (<0.15)

---

## 📅 **PHASE 1: IMMEDIATE IMPROVEMENTS (Week 1-2)**

### ✅ **Completed:**

1. **Code Organization** - Moved all Python files to proper `mlb/` subdirectories
2. **Reality Check** - Fixed misleading 88.6% accuracy claims to actual 48.8%
3. **Performance Analysis** - Identified weak features and poor accuracy
4. **Enhanced Feature Engine** - Created `enhanced_feature_engine.py` with:
   - Pitcher recent form (last 5 starts)
   - Team recent performance (last 10 games)
   - Pitcher vs team matchup history
   - Bullpen usage and fatigue metrics
   - Environmental interaction features
   - Umpire tendency effects

### 🚀 **Next Steps (Days 1-3):**

#### Day 1: Feature Integration

```bash
# Test the enhanced feature engine
cd S:\Projects\AI_Predictions
python mlb\features\enhanced_feature_engine.py

# Integrate into prediction pipeline
# Edit: mlb\core\enhanced_bullpen_predictor.py
# Add: from mlb.features.enhanced_feature_engine import EnhancedFeatureEngine
```

#### Day 2: Model Training

```bash
# Train improved ensemble models
python mlb\models\model_improvement_pipeline.py

# Test performance improvement
# Compare old vs new model accuracy
```

#### Day 3: Performance Tracking

```bash
# Set up daily monitoring
python mlb\tracking\daily_performance_tracker.py

# Add to daily workflow
# Edit run_daily_workflow.bat to include tracking
```

---

## 📊 **PHASE 2: FEATURE OPTIMIZATION (Week 2-3)**

### High-Impact Features to Add:

#### 🎯 **Pitcher Features:**

- [ ] Last 5 starts ERA, WHIP, K/9
- [ ] Pitcher vs opposing team history
- [ ] Days rest impact on performance
- [ ] Pitcher fatigue indicators
- [ ] Home vs away pitcher splits

#### 🏟️ **Ballpark Features:**

- [ ] Temperature × Ballpark interactions
- [ ] Wind direction effects by ballpark
- [ ] Altitude adjustments
- [ ] Roof status impact
- [ ] Time of day effects

#### ⚾ **Team Features:**

- [ ] Last 10 games run scoring/allowing
- [ ] Team vs LHP/RHP splits
- [ ] Bullpen usage last 3 games
- [ ] Lineup strength vs pitcher type
- [ ] Home field advantage quantified

#### 🔗 **Interaction Features:**

- [ ] Weather × Ballpark × Team offense
- [ ] Pitcher quality × Opposing offense
- [ ] Bullpen fatigue × Game situation
- [ ] Umpire tendencies × Pitcher style

### Implementation Priority:

1. **Week 2 Focus**: Pitcher recent form and matchups
2. **Week 3 Focus**: Environmental interactions and bullpen metrics

---

## 🤖 **PHASE 3: MODEL ARCHITECTURE (Week 3-4)**

### Ensemble Method Implementation:

#### Current Approach:

- Single Random Forest model
- Basic feature engineering
- No bias correction

#### Improved Approach:

- [ ] **Random Forest** (200 trees, optimized depth)
- [ ] **Gradient Boosting** (XGBoost/LightGBM)
- [ ] **Ridge Regression** (regularized linear)
- [ ] **Neural Network** (simple 2-layer)
- [ ] **Weighted ensemble** based on performance

### Model Improvements:

```python
# File: mlb/models/ensemble_predictor.py
class EnsembleMLBPredictor:
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=200),
            'gb': GradientBoostingRegressor(n_estimators=150),
            'ridge': Ridge(alpha=1.0),
            'linear': LinearRegression()
        }
        self.weights = {}  # Based on validation performance

    def predict(self, X):
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)

        # Weighted average
        return np.average([pred for pred in predictions.values()],
                         weights=list(self.weights.values()))
```

---

## 📈 **PHASE 4: CONTINUOUS MONITORING (Ongoing)**

### Daily Tracking System:

#### Automated Monitoring:

- [ ] Daily MAE calculation
- [ ] Bias detection and alerts
- [ ] Feature importance tracking
- [ ] Performance by conditions

#### Weekly Reports:

- [ ] Model comparison (Learning vs Original vs Ultra)
- [ ] Feature effectiveness analysis
- [ ] Prediction accuracy trends
- [ ] Recommendations for retraining

#### Alert Thresholds:

- **MAE Warning**: >3.5 runs
- **MAE Critical**: >4.0 runs
- **Bias Warning**: >±0.75 runs
- **Accuracy Warning**: <35% within 1 run

### Performance Targets:

| Metric           | Current | Week 2 Target | Week 4 Target | Final Target |
| ---------------- | ------- | ------------- | ------------- | ------------ |
| MAE              | 3.89    | 3.25          | 2.75          | <2.50        |
| Within 1 Run     | 21%     | 35%           | 50%           | >60%         |
| Bias             | -0.66   | ±0.40         | ±0.25         | ±0.15        |
| Overall Accuracy | 48.8%   | 52%           | 55%           | >57%         |

---

## 🔧 **IMPLEMENTATION CHECKLIST**

### Week 1: Foundation

- [x] ✅ Clean up root directory
- [x] ✅ Fix misleading performance claims
- [x] ✅ Create enhanced feature engine
- [x] ✅ Build model improvement pipeline
- [x] ✅ Set up performance tracking
- [ ] 🔄 Integrate enhanced features into prediction system
- [ ] 🔄 Train and test ensemble models
- [ ] 🔄 Deploy daily performance monitoring

### Week 2: Feature Enhancement

- [ ] Add pitcher recent form features
- [ ] Implement team matchup history
- [ ] Add bullpen usage tracking
- [ ] Create weather interaction terms
- [ ] Test feature impact on accuracy

### Week 3: Model Optimization

- [ ] Train ensemble models
- [ ] Implement bias correction
- [ ] Add confidence scoring
- [ ] Compare model architectures
- [ ] Select best performing ensemble

### Week 4: Production Deployment

- [ ] Deploy improved models
- [ ] Set up automated retraining
- [ ] Create performance dashboards
- [ ] Document model improvements
- [ ] Plan next iteration

---

## 📊 **SUCCESS METRICS**

### Primary Goals:

1. **MAE < 2.5 runs** (currently 3.89)
2. **>60% within 1 run** (currently 21%)
3. **Bias < ±0.25 runs** (currently -0.66)
4. **Overall accuracy >55%** (currently 48.8%)

### Secondary Goals:

- Consistent daily performance
- Better than market accuracy
- Reduced false confidence
- Stable model performance

### Key Performance Indicators:

- Daily MAE tracking
- Weekly accuracy trends
- Model vs market comparison
- Feature importance stability
- Prediction confidence calibration

---

## 🚀 **IMMEDIATE ACTION ITEMS**

### Today:

1. **Test Enhanced Feature Engine**

   ```bash
   cd S:\Projects\AI_Predictions
   python mlb\features\enhanced_feature_engine.py
   ```

2. **Run Model Improvement Pipeline**

   ```bash
   python mlb\models\model_improvement_pipeline.py
   ```

3. **Set Up Daily Tracking**
   ```bash
   python mlb\tracking\daily_performance_tracker.py
   ```

### This Week:

1. Integrate enhanced features into main prediction pipeline
2. Compare ensemble vs single model performance
3. Set up automated daily performance reports
4. Begin feature impact analysis

### Success Criteria for Week 1:

- MAE reduced to <3.5 runs
- Accuracy within 1 run improved to >30%
- Enhanced features successfully integrated
- Daily tracking system operational

---

## 📞 **SUPPORT & RESOURCES**

### Files Created:

- `mlb/features/enhanced_feature_engine.py` - Advanced feature engineering
- `mlb/models/model_improvement_pipeline.py` - Ensemble modeling
- `mlb/tracking/daily_performance_tracker.py` - Performance monitoring
- `mlb/analysis/prediction_improvement_analyzer.py` - Analysis tools

### Key Dependencies:

- scikit-learn (ensemble models)
- pandas/numpy (data processing)
- psycopg2 (database connectivity)
- joblib (model persistence)

### Monitoring Commands:

```bash
# Daily performance check
python mlb\tracking\daily_performance_tracker.py

# Weekly analysis
python mlb\analysis\prediction_improvement_analyzer.py

# Model comparison
python mlb\analysis\performance_reality_check.py
```

---

**Next Steps**: Begin Phase 1 implementation by testing the enhanced feature engine and integrating it into the main prediction pipeline. Focus on immediate MAE reduction and accuracy improvement.
