# MLB Over/Under Prediction System - Enhanced Architecture

## 🏟️ **System Overview**

A comprehensive MLB over/under prediction platform utilizing machine learning with enhanced statistical features, weather data, and ballpark factors for accurate game total predictions.

---

## 📊 **Enhanced Data Collection System**

### **Primary Dataset: `enhanced_historical_games_2025.parquet`**

**📈 Dataset Size**: ~2,230 games (March - August 2025)
**🎯 Target Variable**: `total_runs` (for over/under predictions)
**📋 Features**: 25+ comprehensive columns

### **Core Game Data**

- `game_id`: Unique MLB game identifier
- `date`: Game date (YYYY-MM-DD)
- `home_team` / `away_team`: Team names
- `home_score` / `away_score`: Final scores
- `total_runs`: Combined final score (target variable)

### **⚾ Enhanced Pitcher Statistics**

- `home_sp_id` / `away_sp_id`: Starting pitcher MLB IDs
- `home_sp_er` / `away_sp_er`: Earned runs allowed in game
- `home_sp_ip` / `away_sp_ip`: Innings pitched in game
- `home_sp_k` / `away_sp_k`: Strikeouts recorded
- `home_sp_bb` / `away_sp_bb`: Walks issued
- `home_sp_h` / `away_sp_h`: Hits allowed

### **🏏 Team Batting Performance**

- `home_team_hits` / `away_team_hits`: Team hits in game
- `home_team_runs` / `away_team_runs`: Runs scored
- `home_team_rbi` / `away_team_rbi`: RBIs recorded
- `home_team_lob` / `away_team_lob`: Runners left on base

### **🌤️ Weather & Environmental Data**

- `temperature`: Game temperature (°F)
- `wind_speed`: Wind speed (mph)
- `wind_direction`: Wind direction ("Out To LF", "In From RF", etc.)
- `weather_condition`: Weather status ("Clear", "Partly Cloudy", etc.)

### **🏟️ Ballpark Information**

- `venue_id`: MLB venue identifier
- `venue_name`: Stadium name
- `day_night`: Day ("D") or Night ("N") game
- `game_type`: Regular season ("R"), Playoff, etc.

---

## 🤖 **Machine Learning Model Architecture**

### **Model Type**: Gradient Boosting Regressor

**Target**: Predict `total_runs` for over/under betting

### **Feature Engineering Pipeline**

#### **1. Pitcher Performance Metrics**

```python
# Calculated ERA for each pitcher based on historical performance
pitcher_era = (earned_runs * 9) / innings_pitched

# Pitcher effectiveness indicators
k_bb_ratio = strikeouts / walks  # Control indicator
whip = (hits + walks) / innings_pitched  # Baserunner frequency
```

#### **2. Weather Impact Factors**

```python
# Wind impact on scoring
wind_boost = wind_speed * wind_direction_multiplier
# Temperature effect on ball flight
temp_factor = (temperature - 70) * 0.01  # Warmer = more offense

# Weather scoring adjustment
weather_impact = wind_boost + temp_factor
```

#### **3. Ballpark Factors**

```python
# Venue-specific scoring tendencies
park_factor = venue_run_environment[venue_id]
# Day vs night game adjustments
time_factor = 0.2 if day_night == "N" else 0.0  # Night games typically higher scoring
```

#### **4. Team Offensive Metrics**

```python
# Recent team performance
team_offensive_rating = (recent_runs + recent_hits + recent_rbi) / games_played
# Left-on-base efficiency (clutch hitting)
clutch_factor = 1 - (lob / (hits + walks))
```

### **Model Training Features**

**Input Features** (~20 engineered features):

1. **Pitcher Quality**: ERA, K/BB ratio, WHIP, recent form
2. **Weather Factors**: Temperature, wind speed/direction impact
3. **Ballpark Effects**: Venue factors, day/night adjustments
4. **Team Offense**: Batting averages, RBI efficiency, clutch metrics
5. **Game Context**: Season timing, series position, rest days

**Training Process**:

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Feature engineering pipeline
features = engineer_features(enhanced_dataset)

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(features, total_runs, test_size=0.2)

# Model training
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)
```

---

## 🏗️ **System Architecture**

### **Data Flow Pipeline**

1. **Collection**: `enhanced_season_collector.py` → Enhanced dataset with weather/ballpark data
2. **Processing**: Feature engineering → ML-ready format
3. **Training**: Gradient boosting model → Trained predictor
4. **Inference**: `models/infer.py` → Real-time predictions
5. **API**: `api/` → REST endpoints for predictions
6. **Frontend**: `mlb-predictions-ui/` → User interface

### **Key Components**

#### **📂 mlb-overs/**

- `data/enhanced_historical_games_2025.parquet`: Primary training dataset
- `models/`: ML model training and inference
- `api/`: REST API for predictions
- `scripts/`: Data collection and processing utilities

#### **📂 mlb-predictions-ui/**

- React-based frontend for prediction visualization
- Real-time over/under recommendations
- Performance tracking and analytics

#### **📂 scripts/**

- `enhanced_season_collector.py`: Comprehensive data collection
- `inspect_historical_data.py`: Data quality validation
- Training and validation utilities

---

## 🎯 **Prediction Capabilities**

### **Over/Under Predictions**

- **Input**: Today's games with pitcher matchups and weather
- **Output**: Predicted total runs with confidence intervals
- **Accuracy Target**: >55% accuracy (beating market)

### **Key Prediction Factors**

1. **Pitcher Matchup** (40% weight):

   - Starting pitcher ERA and recent form
   - Strikeout rates and control metrics
   - Historical performance vs opponent

2. **Weather Impact** (25% weight):

   - Wind speed and direction effects
   - Temperature impact on ball flight
   - Weather condition adjustments

3. **Ballpark Factors** (20% weight):

   - Venue-specific run environments
   - Altitude and dimension effects
   - Day vs night game tendencies

4. **Team Offense** (15% weight):
   - Recent batting performance
   - RBI efficiency and clutch hitting
   - Matchup vs pitcher handedness

### **Model Output Example**

```json
{
  "game_id": 746072,
  "predicted_total": 8.7,
  "market_total": 8.5,
  "confidence": 0.73,
  "recommendation": "OVER",
  "edge": 0.2,
  "factors": {
    "pitcher_impact": +0.3,
    "weather_boost": +0.4,
    "park_factor": -0.1,
    "team_offense": +0.1
  }
}
```

---

## 📈 **Current Status & Performance**

### **✅ Completed Features**

- Enhanced data collection with 25+ features
- Weather and ballpark data integration
- Comprehensive pitcher statistics
- Team batting performance metrics
- Data validation and quality checks

### **🚧 In Progress**

- Model training with enhanced dataset
- Feature importance analysis
- Backtesting and validation framework
- API integration with enhanced predictions

### **🎯 Next Steps**

1. Complete enhanced data collection (in progress)
2. Train ML model with new comprehensive features
3. Implement real-time prediction API
4. Frontend integration with enhanced predictions
5. Performance monitoring and model updates

---

## � **System Advantages**

- **Comprehensive Data**: Weather, ballpark, and detailed performance metrics
- **Real-world Factors**: Accounts for environmental conditions affecting gameplay
- **Advanced ML**: Gradient boosting with engineered features
- **Scalable Architecture**: Modular design for easy updates and improvements
- **Data-Driven**: 2,000+ games of training data with rich feature set

This enhanced system represents a significant upgrade from basic statistical models to a comprehensive, weather-aware, ballpark-adjusted ML prediction platform!


=== DATA TYPES ===
  game_id: int64
  date: object
  home_team: object
  away_team: object
  home_score: int64
  away_score: int64
  total_runs: int64
  weather_condition: object
  temperature: object
  wind_speed: int64
  wind_direction: object
  venue_id: int64
  venue_name: object
  home_sp_id: int64
  home_sp_er: int64
  home_sp_ip: object
  home_sp_k: int64
  home_sp_bb: int64
  home_sp_h: int64
  away_sp_id: int64
  away_sp_er: int64
  away_sp_ip: object
  away_sp_k: int64
  away_sp_bb: int64
  away_sp_h: int64
  home_team_hits: int64
  home_team_runs: int64
  home_team_rbi: int64
  home_team_lob: int64
  away_team_hits: int64
  away_team_runs: int64
  away_team_rbi: int64
  away_team_lob: int64
  game_type: object
  day_night: object


  important files:
  PS S:\Projects\AI_Predictions> python final_prediction_test.py

  python scripts\ml_validation_pipeline.py --mode validate --date 2025-08-12

  python scripts\ml_validation_pipeline.py --mode current

  python scripts\ml_validation_pipeline.py --mode train

   python scripts\daily_complete_pipeline.py 

python enhanced_predictions_api.py