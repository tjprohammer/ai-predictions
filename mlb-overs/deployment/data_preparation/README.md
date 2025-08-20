# MLB Data Preparation Scripts
## Data Collection, Backfill, and Pipeline Tools

This directory contains scripts for preparing training data and maintaining data quality for the MLB over/under prediction system.

---

## 📁 **FILE INVENTORY**

### **🔄 Backfill & Historical Data**
- **`historical_markets_backfill.py`** - Backfill historical market data and odds
- **`backfill_away_era.py`** - Backfill missing away pitcher ERA data
- **`backfill_range.py`** - Backfill data for specific date ranges
- **`complete_backfill_retrain.py`** - Complete backfill workflow with retraining
- **`complete_historical_workflow.py`** - Full historical data preparation workflow

### **🔍 Analysis & Validation**
- **`analyze_backfill.py`** - Analyze backfill completeness and quality
- **`evaluate_backfill.py`** - Evaluate effectiveness of backfill operations

### **📊 Training Data Preparation**
- **`build_training_snapshot.py`** - Create training data snapshots from raw data

---

## 🎯 **COMMON WORKFLOWS**

### **1. Historical Data Backfill**
```bash
# Backfill market data for date range
python historical_markets_backfill.py --start 2025-06-01 --end 2025-08-19

# Backfill missing pitcher data
python backfill_away_era.py --date-range 2025-06-01:2025-08-19

# Complete backfill with analysis
python complete_backfill_retrain.py --window-days 80
```

### **2. Training Data Preparation**
```bash
# Build clean training snapshot
python build_training_snapshot.py --end 2025-08-19 --window-days 80

# Analyze data completeness
python analyze_backfill.py --check-era --check-market
```

### **3. Data Quality Validation**
```bash
# Evaluate backfill effectiveness
python evaluate_backfill.py --before 2025-06-01 --after 2025-08-19

# Complete historical workflow
python complete_historical_workflow.py --full-rebuild
```

---

## 🔧 **DATA SOURCES**

### **Primary Tables**
- **`enhanced_games`** - Main production table with complete game data
- **`legitimate_game_features`** - Legacy training table (incomplete)
- **`pitcher_daily_rolling`** - Pitcher rolling statistics

### **Market Data**
- **External APIs** - Live odds and market totals
- **Historical Archives** - Backfilled market data

### **Pitcher Data**
- **MLB Stats API** - Current season statistics
- **Rolling Calculations** - Performance trends and form

---

## ⚠️ **DATA QUALITY ISSUES ADDRESSED**

### **Missing ERA Data (Fixed)**
- **Problem**: Only 31% coverage in training data
- **Solution**: `backfill_away_era.py` fills missing pitcher ERAs
- **Result**: Complete pitcher data for model training

### **Market Data Gaps (Fixed)**
- **Problem**: Incomplete market totals for historical games
- **Solution**: `historical_markets_backfill.py` retrieves missing odds
- **Result**: 100% market coverage for recent games

### **Feature Pipeline Alignment**
- **Problem**: Training vs production data mismatch
- **Solution**: `build_training_snapshot.py` uses production data format
- **Result**: Consistent feature engineering across training/serving

---

## 📈 **MONITORING & MAINTENANCE**

### **Regular Tasks**
1. **Weekly**: Run `analyze_backfill.py` to check data completeness
2. **Monthly**: Execute `complete_historical_workflow.py` for full refresh
3. **As needed**: Use specific backfill scripts for data gaps

### **Quality Metrics**
- **ERA Coverage**: Should be >95% for recent games
- **Market Coverage**: Should be 100% for prediction dates
- **Feature Completeness**: All required features present for training

---

## 🔗 **INTEGRATION WITH TRAINING**

These data preparation scripts feed into the training pipeline:

```
Data Preparation → Training Data → Model Training → Production
     (this folder)      (clean data)     (training/)     (api/)
```

### **Workflow Integration**
1. **Data Prep**: Scripts in this folder ensure complete, clean data
2. **Training**: `../training/train_model.py` uses prepared data
3. **Production**: API serves predictions using trained models

---

## 📝 **USAGE NOTES**

### **Before Training**
Always run data preparation to ensure training data quality:
```bash
python analyze_backfill.py --check-completeness
python build_training_snapshot.py --validate
```

### **After Major Changes**
When adding new features or changing data sources:
```bash
python complete_historical_workflow.py --rebuild-features
python evaluate_backfill.py --validate-pipeline
```

---

## 🚀 **CURRENT STATUS**

- ✅ **ERA Backfill**: Complete for training period
- ✅ **Market Data**: 100% coverage for active dates  
- ✅ **Training Pipeline**: Aligned with production features
- ✅ **Data Quality**: All major gaps resolved

**Last Updated**: August 20, 2025
