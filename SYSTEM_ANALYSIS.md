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


Simple Game Cards - Each game shows:

Teams and venue
AI prediction vs market line
Edge calculation
Recommendation (OVER/UNDER/HOLD) with color coding
Confidence percentage
Pitcher Information - Just the essentials:

Names and ERAs for both pitchers
Weather Context - Basic weather that affects betting:

Temperature and conditions
Wind speed and direction
AI Reasoning - Simple explanations like:

"High winds (15 mph) may affect scoring"
"Struggling pitchers favor higher scoring"
"AI model shows significant lower scoring than market"
"Coors Field high altitude favors offense"
Summary Stats at the top:

Total games
Average confidence
Number of OVER vs UNDER picks


📈 REALISTIC PERFORMANCE EXPECTATIONS
============================================================
🎯 WITH PROPER FEATURES:
   Expected MAE: 1.5-2.2 runs
   Accuracy (±1 run): 40-55%
   Betting edge: Modest but profitable

⚠️ WHY PERFORMANCE WILL BE LOWER:
   - No more 'cheating' with game outcomes
   - Baseball has inherent randomness
   - Weather and player performance vary
   - Injuries and lineup changes

✅ WHAT'S STILL ACHIEVABLE:
   - Beat random guessing significantly
   - Identify high/low scoring game tendencies
   - Find value vs betting market lines
   - Achieve 52-55% accuracy (profitable)

🚀 IMPLEMENTATION ROADMAP
============================================================
📅 PHASE 1: Data Architecture (Week 1)
   - Set up MLB Stats API access
   - Create pitcher season stats database
   - Create team season stats database
   - Implement daily stats updates

📅 PHASE 2: Feature Engineering (Week 2)
   - Calculate pitcher performance metrics
   - Calculate team offensive metrics
   - Add ballpark and weather interactions
   - Create proper train/test splits

📅 PHASE 3: Model Development (Week 3)
   - Train model with legitimate features
   - Implement time-series validation
   - Tune hyperparameters properly
   - Test on out-of-sample data

📅 PHASE 4: Validation & Deployment (Week 4)
   - Live test on current games
   - Compare predictions vs actual outcomes
   - Monitor performance vs market lines
   - Deploy for production use



All available tables:
  api_games_today
  api_results_history
  betting_lines
  bullpens_daily
  calibrated_predictions
  catcher_framing_stats
  daily_games
  enhanced_games
  enhanced_games_features_v
  enhanced_leak_free_features
  injuries
  leak_free_features
  legit_features_staging
  legitimate_game_features
  lineups
  market_moves
  model_accuracy
  model_config
  parks
  pitcher_comprehensive_stats
  pitcher_daily_rolling
  pitchers_starts
  predictions
  probability_predictions
  team_travel_log
  team_travel_snapshot
  teams_offense_daily
  umpires
  weather_game


=== umpires schema ===
ump_id : text
name : text
called_strike_pct : numeric
edge_strike_pct : numeric
o_u_tendency : numeric
sample_size : integer

=== weather_game schema ===
game_id : text
temp_f : integer
humidity_pct : integer
wind_mph : integer
wind_dir_deg : integer
precip_prob : integer
altitude_ft : integer
air_density_idx : numeric
wind_out_mph : numeric
wind_cross_mph : numeric
is_forecast : boolean
valid_hour_utc : timestamp with time zone

weather_game columns:
  game_id
  temp_f
  humidity_pct
  wind_mph
  wind_dir_deg
  precip_prob
  altitude_ft
  air_density_idx
  wind_out_mph
  wind_cross_mph
  is_forecast
  valid_hour_utc
  
\npitchers_starts columns:
  start_id
  game_id
  pitcher_id
  team
  opp_team
  is_home
  date
  ip
  h
  bb
  k
  hr
  r
  er
  bf
  pitches
  csw_pct
  velo_fb
  velo_delta_3g
  hh_pct_allowed
  barrel_pct_allowed
  avg_ev_allowed
  xwoba_allowed
  xslg_allowed
  era_game
  fip_game
  xfip_game
  siera_game
  opp_lineup_l_pct
  opp_lineup_r_pct
  days_rest
  tto
  pitch_count_prev1
  pitch_count_prev2

=== Current enhanced_games schema ===
  id : integer : NOT NULL
  game_id : character varying : NOT NULL
  date : date : NOT NULL
  home_team : character varying : NOT NULL
  away_team : character varying : NOT NULL
  home_score : integer : NULL
  away_score : integer : NULL
  total_runs : integer : NULL
  weather_condition : character varying : NULL
  temperature : integer : NULL
  wind_speed : integer : NULL
  wind_direction : character varying : NULL
  venue_id : integer : NULL
  venue_name : character varying : NULL
  home_sp_id : integer : NULL
  home_sp_er : integer : NULL
  home_sp_ip : numeric : NULL
  home_sp_k : integer : NULL
  home_sp_bb : integer : NULL
  home_sp_h : integer : NULL
  away_sp_id : integer : NULL
  away_sp_er : integer : NULL
  away_sp_ip : numeric : NULL
  away_sp_k : integer : NULL
  away_sp_bb : integer : NULL
  away_sp_h : integer : NULL
  home_team_hits : integer : NULL
  home_team_runs : integer : NULL
  home_team_rbi : integer : NULL
  home_team_lob : integer : NULL
  away_team_hits : integer : NULL
  away_team_runs : integer : NULL
  away_team_rbi : integer : NULL
  away_team_lob : integer : NULL
  game_type : character varying : NULL
  day_night : character varying : NULL
  created_at : timestamp without time zone : NULL
  market_total : numeric : NULL
  home_sp_season_era : numeric : NULL
  away_sp_season_era : numeric : NULL
  home_sp_name : character varying : NULL
  away_sp_name : character varying : NULL
  over_odds : integer : NULL
  under_odds : integer : NULL
  predicted_total : numeric : NULL
  confidence : numeric : NULL
  recommendation : character varying : NULL
  edge : numeric : NULL
  home_team_id : integer : NULL
  away_team_id : integer : NULL

  === Current lineups schema ===
  game_id : text : NULL
  team : text : NULL
  batter_id : text : NULL
  order_spot : integer : NULL
  hand : text : NULL
  proj_pa : numeric : NULL
  xwoba_100 : numeric : NULL
  xwoba_vs_hand_100 : numeric : NULL
  iso_100 : numeric : NULL
  k_pct_100 : numeric : NULL
  bb_pct_100 : numeric : NULL