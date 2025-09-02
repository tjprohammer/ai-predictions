# 🏀 MLB Prediction System

**Organized, production-ready MLB prediction system with clean modular architecture**

## 📁 **Directory Structure**

```
mlb/
├── core/                              # Core prediction engines and workflow
│   ├── daily_api_workflow.py          # Main orchestration workflow
│   ├── enhanced_bullpen_predictor.py  # Enhanced ML predictor with bullpen factors
│   └── learning_model_predictor.py    # Adaptive learning model predictor
│
├── systems/                           # Specialized prediction systems
│   ├── incremental_ultra_80_system.py # Ultra 80 incremental learning system
│   └── ultra_80_percent_system.py     # Ultra 80 percent model system
│
├── ingestion/                         # Data collection and ingestion
│   ├── working_games_ingestor.py      # Game schedule and lineup data
│   ├── real_market_ingestor.py        # Live odds and market data
│   ├── working_pitcher_ingestor.py    # Pitcher stats and ratings
│   ├── working_team_ingestor.py       # Team statistics and trends
│   └── weather_ingestor.py            # Weather conditions
│
├── validation/                        # Health checks and validation
│   ├── health_gate.py                 # System health monitoring
│   └── probabilities_and_ev.py        # Probability calculation and expected value
│
├── training/                          # Model training and maintenance
│   ├── training_bundle_audit.py       # Training bundle validation and audit
│   ├── retrain_model.py               # Model retraining pipeline
│   └── backfill_range.py              # Historical data backfill
│
├── tracking/                          # Performance tracking and monitoring
│   ├── performance/                   # Model performance analysis
│   │   ├── enhanced_prediction_tracker.py # Comprehensive performance analysis
│   │   ├── weekly_performance_tracker.py  # Weekly performance summaries
│   │   └── learning_impact_tracker.py     # Learning model impact analysis
│   ├── results/                       # Game result collection and management
│   │   ├── game_result_tracker.py     # Real-time game result monitoring
│   │   └── simple_results_checker.py  # Daily betting results checker
│   ├── validation/                    # Data validation and prediction checking
│   │   ├── check_predictions_final.py # Final prediction validation
│   │   └── check_residual_data.py     # Residual data analysis
│   └── monitoring/                    # Real-time monitoring and alerts
│       ├── todays_reality_check.py    # Daily reality check analysis
│       └── auto_prediction_tracker.py # Automated prediction tracking
│
├── models/                            # Model artifacts and state files
│   ├── adaptive_learning_model.joblib # Adaptive learning model bundle
│   ├── legitimate_model_latest.joblib # Main production model bundle
│   ├── incremental_ultra80_state.joblib # Ultra 80 system state
│   ├── ultra_80_model.joblib          # Ultra 80 model bundle
│   └── ultra_ensemble_model.joblib    # Ensemble model bundle
│
├── config/                            # Configuration and calibration data
│   ├── model_bias_corrections.json    # Model bias correction factors
│   ├── daily_market_totals.json       # Daily market total cache
│   └── daily_starting_pitchers_*.json # Daily pitcher assignments
│
└── utils/                             # Utility scripts and tools
    └── apply_prediction_override.py   # Manual prediction override tool
```

## 🚀 **Quick Start**

### **Daily Prediction Workflow**

```bash
cd mlb/core
python daily_api_workflow.py --stages markets,features,predict,ultra80,health,prob,export
```

### **Individual Components**

**Run Enhanced Predictor:**

```python
from mlb.core.enhanced_bullpen_predictor import EnhancedBullpenPredictor
predictor = EnhancedBullpenPredictor()
predictions = predictor.predict_today_games("2025-08-28")
```

**Run Ultra 80 System:**

```python
from mlb.systems.incremental_ultra_80_system import IncrementalUltra80System
system = IncrementalUltra80System()
system.load_state("../models/incremental_ultra80_state.joblib")
predictions = system.predict_for_date("2025-08-28")
```

**Run Learning Model:**

```python
from mlb.core.learning_model_predictor import predict_and_upsert_learning
predictions = predict_and_upsert_learning(engine, X, ids, "2025-08-28")
```

## 🔧 **Configuration**

### **Environment Variables**

```bash
DATABASE_URL="postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb"
MODEL_BUNDLE_PATH="../models/legitimate_model_latest.joblib"
MIN_PITCHER_COVERAGE=0.8
ALLOW_FLAT_ENV=false
DISABLE_MARKET_ANCHORING=false
```

### **Workflow Stages**

- `markets` - Pull market data and odds from APIs
- `features` - Build enhanced features for prediction
- `predict` - Generate base ML predictions
- `ultra80` - Generate Ultra 80 system predictions with intervals and EV
- `health` - Validate system calibration health before trading
- `prob` - Calculate enhanced probability predictions with EV/Kelly
- `export` - Export results to files
- `audit` - Audit and validate results

## 🏗️ **Architecture**

### **Three-Model System**

1. **Original Model** (`enhanced_bullpen_predictor.py`) - Static, well-calibrated baseline
2. **Learning Model** (`learning_model_predictor.py`) - Market-anchored adaptive predictions
3. **Ultra 80 System** (`incremental_ultra_80_system.py`) - Advanced incremental learning

### **Data Flow**

```
Ingestion → Feature Engineering → Prediction → Validation → Database → UI
```

### **Database Schema**

- `enhanced_games.predicted_total` - Learning model predictions
- `enhanced_games.predicted_total_learning` - Ultra 80 system predictions
- `enhanced_games.predicted_total_original` - Original model predictions

## 📊 **Model Performance**

### **Ultra 80 System**

- **Realistic Range:** 7.84 - 10.24 runs
- **State Persistence:** Incremental learning with joblib state files
- **Update Frequency:** After each completed game

### **Learning Model**

- **Realistic Range:** 6.86 - 8.82 runs
- **Market Anchoring:** Controlled anchoring to prevent extreme deviations
- **Calibration:** Market-anchored to maintain realistic MLB ranges

### **Original Model**

- **Stable Baseline:** Well-calibrated historical performance
- **Feature Rich:** Enhanced bullpen and weather factors
- **Production Ready:** Thoroughly tested and validated

## 🔄 **Daily Operations**

### **Morning Routine**

1. `python daily_api_workflow.py --stages markets` - Pull fresh data
2. `python daily_api_workflow.py --stages features,predict` - Generate predictions
3. `python daily_api_workflow.py --stages health` - Validate system health

### **Pre-Game Validation**

1. `python daily_api_workflow.py --stages prob` - Calculate probabilities
2. `python daily_api_workflow.py --stages export` - Export for UI
3. Review health gate status and prediction ranges

### **Post-Game Analysis**

1. `python daily_api_workflow.py --stages scores` - Collect final scores
2. `python daily_api_workflow.py --stages audit` - Performance audit
3. Ultra 80 system automatically updates with new results

## 🛠️ **Maintenance**

### **Model Retraining**

```bash
cd mlb/training
python retrain_model.py --end 2025-08-28 --window-days 150 --deploy
```

### **Historical Backfill**

```bash
cd mlb/training
python backfill_range.py --start 2025-08-01 --end 2025-08-28 --predict
```

### **Manual Overrides**

```bash
cd mlb/utils
python apply_prediction_override.py --game-id "game123" --total 8.5
```

## 🔍 **Tracking & Monitoring**

### **Performance Tracking**

```bash
# Comprehensive performance analysis
python mlb/tracking/performance/enhanced_prediction_tracker.py

# Weekly performance summaries
python mlb/tracking/performance/weekly_performance_tracker.py

# Learning model impact analysis
python mlb/tracking/performance/learning_impact_tracker.py
```

### **Results Tracking**

```bash
# Real-time game result monitoring
python mlb/tracking/results/game_result_tracker.py

# Daily betting results checker
python mlb/tracking/results/simple_results_checker.py

# Manual result updates
python mlb/tracking/results/manual_results_updater.py
```

### **Validation & Monitoring**

```bash
# Final prediction validation
python mlb/tracking/validation/check_predictions_final.py

# Daily reality check analysis
python mlb/tracking/monitoring/todays_reality_check.py

# Automated prediction tracking
python mlb/tracking/monitoring/auto_prediction_tracker.py
```

## 🧪 **Testing**

### **Import Verification**

```bash
python test_mlb_imports.py
```

### **Prediction Quality Checks**

- **Variance Check:** σ > 0.25 (prevents constant predictions)
- **Reality Check:** Mean 6.5-9.8 runs (realistic MLB range)
- **Coverage Check:** > 80% pitcher data availability
- **Health Gate:** Automated calibration monitoring

## 📈 **Monitoring**

### **Key Metrics**

- **MAE:** Mean Absolute Error vs actual totals
- **Bias:** Systematic over/under prediction
- **Calibration:** R² and slope vs actual outcomes
- **Coverage:** Prediction availability percentage

### **Alerts**

- Prediction variance < 0.25 (flat predictions)
- Mean predictions > 9.8 or < 6.5 (unrealistic)
- Health gate failures (calibration drift)
- Ingestion failures (missing data)

---

**Migration Notes:** All file paths have been updated from the old `mlb-overs/deployment/` structure to the new organized `mlb/` structure. Relative paths are maintained for cross-platform compatibility.
