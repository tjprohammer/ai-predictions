# MLB Over/Under Model Training

## Centralized Training Scripts and Tools

This directory contains all active model training scripts and related tools for the MLB over/under prediction system.

---

## 📁 **FILE INVENTORY**

### **🤖 Training Scripts**

- **`train_model.py`** - **CURRENT PRODUCTION TRAINER**
  - Main training script that eliminates feature mismatch issues
  - Uses complete data from `enhanced_games` table (vs incomplete `legitimate_game_features`)
  - Removes 81 problematic zero/constant features before training
  - Produces clean 44-feature models with realistic predictions
  - **Usage**: `python train_model.py --end 2025-08-19 --window-days 80 --deploy`

### **🔍 Audit & Analysis Tools**

- **`training_bundle_audit.py`** - Model bundle auditing and validation

  - Analyzes model bundles for feature consistency, performance metrics
  - Validates training metadata and deployment readiness
  - **Usage**: `python training_bundle_audit.py`

- **`training_bundle_audit_report.json`** - Latest audit report
  - Generated output from bundle audit tool
  - Contains detailed analysis of current model bundle

---

## 🎯 **CURRENT TRAINING WORKFLOW**

### **1. Standard Model Retraining**

```bash
cd S:\Projects\AI_Predictions\mlb-overs\deployment\training
python train_model.py --end 2025-08-19 --window-days 80 --deploy
```

### **2. Model Validation**

```bash
python training_bundle_audit.py
```

### **3. Key Features of Fixed Training**

- **Data Source**: `enhanced_games` table (complete data)
- **Training Size**: ~717 games (vs 255 incomplete games previously)
- **Feature Engineering**: Same pipeline as production (`EnhancedBullpenPredictor`)
- **Feature Cleaning**: Removes constant/zero features automatically
- **Output**: 44 clean features (vs 125 corrupted features previously)
- **Performance**: Test MAE ~2.2 (good for MLB totals)

---

## ⚠️ **RESOLVED ISSUES**

### **Feature Mismatch Crisis (Aug 2025)**

- **Problem**: Training used 69% missing/zero data from `legitimate_game_features`
- **Impact**: Systematic under-prediction (impossible 3.5 run predictions)
- **Solution**: Switched to complete `enhanced_games` data source
- **Result**: Realistic 6.5-8.5 run predictions, proper market alignment

### **Problematic Features Eliminated**

- **81 out of 125 features** were corrupted (constant or high zero-rate)
- **Examples**: All ERA features (91.8% zeros), pitching quality (constant), team stats (constant)
- **Fix**: Automated detection and removal of problematic features
- **Outcome**: Clean 44-feature model with better generalization

---

## 📊 **MODEL PERFORMANCE TARGETS**

### **Training Metrics**

- **Training MAE**: < 1.0 (indicates good fit)
- **Test MAE**: 2.0-2.5 (realistic for MLB totals)
- **Training Size**: > 500 games (sufficient for robust training)
- **Feature Count**: 40-50 clean features (after removing problematic ones)

### **Production Validation**

- **Prediction Range**: 6.5-11.5 runs (realistic MLB totals)
- **Market Alignment**: Predictions within ±3 runs of market
- **Edge Detection**: Clear differentiation between games
- **Bias**: No systematic over/under prediction

---

## 🔗 **RELATED COMPONENTS**

### **Model Storage**

- **Production Model**: `../models/legitimate_model_latest.joblib`
- **Backup Models**: `../models/fixed_model_latest.joblib`

### **Feature Engineering**

- **Pipeline**: `../enhanced_bullpen_predictor.py`
- **Enhanced Features**: `../enhanced_feature_pipeline.py`

### **Data Sources**

- **Training Data**: PostgreSQL `enhanced_games` table
- **Feature Data**: Same table used by production predictions

### **Deployment Integration**

- **API**: `../api/app.py` (loads trained models)
- **Daily Workflow**: `../daily_api_workflow.py` (uses trained models)

---

## 🚀 **NEXT STEPS**

1. **Monitor Performance**: Track prediction accuracy over next few days
2. **Feature Expansion**: Consider adding new clean features if needed
3. **Automation**: Set up scheduled retraining (weekly/bi-weekly)
4. **Documentation**: Update model documentation as system evolves

---

## 📝 **TRAINING HISTORY**

- **Aug 20, 2025**: Deployed `train_model.py` - Fixed feature mismatch crisis
- **Aug 18, 2025**: Last use of problematic `train_clean_model.py` (removed)
- **Aug 13, 2025**: Deprecated old training scripts in `models/` directory (removed)

**Current Status**: ✅ **PRODUCTION READY** - All training issues resolved
