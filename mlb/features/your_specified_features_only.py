#!/usr/bin/env python3
"""
Your Specified Features Only Model
==================================

Uses ONLY the features you specifically listed, with proper weighting.
No composite features, no extras - just the baseball-smart features you identified.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class YourSpecifiedFeaturesModel:
    """Model using only your specified features"""
    
    def __init__(self):
        self.engine = create_engine('postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
        
        # YOUR EXACT FEATURE LIST (with column numbers as comments)
        self.your_features = [
            # Weather & Environment
            'weather_condition',           # 9
            'temperature',                 # 10
            'wind_speed',                 # 11
            'wind_direction',             # 12
            'wind_direction_deg',         # 58
            'wind_gust',                  # 61
            'precip_prob',                # 62
            
            # Starting Pitcher Performance
            'home_sp_er',                 # 16 - season era
            'home_sp_k',                  # 18
            'home_sp_bb',                 # 19
            'home_sp_h',                  # 20
            'away_sp_er',                 # 22 - season era
            'away_sp_k',                  # 24
            'away_sp_bb',                 # 25
            'away_sp_h',                  # 26
            'home_sp_season_era',         # 39 (possibly duplicate with home_sp_er)
            'away_sp_season_era',         # 40 (possibly duplicate with away_sp_er)
            'home_sp_hand',               # 66
            'away_sp_hand',               # 67
            'home_sp_whip',               # 76
            'away_sp_whip',               # 77
            'home_sp_season_k',           # 80
            'away_sp_season_k',           # 81
            'home_sp_season_bb',          # 82
            'away_sp_season_bb',          # 83
            'home_sp_season_ip',          # 84
            'away_sp_season_ip',          # 85
            'home_sp_era_l3starts',       # 146
            'away_sp_era_l3starts',       # 147
            
            # Team Offensive Stats
            'home_team_hits',             # 27
            'home_team_runs',             # 28
            'home_team_rbi',              # 29
            'home_team_lob',              # 30
            'away_team_hits',             # 31
            'away_team_runs',             # 32
            'away_team_rbi',              # 33
            'away_team_lob',              # 34
            'home_team_avg',              # 78
            'away_team_avg',              # 79
            'home_team_obp',              # 98
            'away_team_obp',              # 99
            'home_team_slg',              # 100
            'away_team_slg',              # 101
            'home_team_ops',              # 102
            'away_team_ops',              # 103
            'home_team_iso',              # 104
            'away_team_iso',              # 105
            'home_team_woba',             # 106
            'away_team_woba',             # 107
            'home_team_wrc_plus',         # 108
            'away_team_wrc_plus',         # 109
            'away_team_plate_appearances', # 113
            'combined_team_ops',          # 114
            'combined_team_woba',         # 115
            
            # Ballpark Effects
            'venue',                      # 53
            'ballpark',                   # 54
            'ballpark_run_factor',        # 86
            'ballpark_hr_factor',         # 87
            
            # Bullpen Performance
            'home_bp_ip',                 # 88
            'home_bp_er',                 # 89
            'home_bp_k',                  # 90
            'home_bp_bb',                 # 91
            'home_bp_h',                  # 92
            'away_bp_ip',                 # 93
            'away_bp_er',                 # 94
            'away_bp_k',                  # 95
            'away_bp_bb',                 # 96
            'away_bp_h',                  # 97
            'home_bullpen_era_l30',       # 132
            'away_bullpen_era_l30',       # 133
            'home_bullpen_whip_l30',      # 134
            'away_bullpen_whip_l30',      # 135
            'home_bullpen_usage_rate',    # 136
            'away_bullpen_usage_rate',    # 137
            'home_bullpen_rest_status',   # 138
            'away_bullpen_rest_status',   # 139
            'home_team_bullpen_usage_intensity',      # 182
            'away_team_bullpen_usage_intensity',      # 183
            'home_team_bullpen_performance_trend',    # 184
            'away_team_bullpen_performance_trend',    # 185
            'home_team_bullpen_recent_era',           # 186
            'away_team_bullpen_recent_era',           # 187
            'home_bullpen_era',           # 220
            'away_bullpen_era',           # 221
            'combined_bullpen_fip',       # 222
            'home_bullpen_fip',           # 223
            'away_bullpen_fip',           # 224
            
            # Recent Team Form
            'home_team_runs_l7',          # 140
            'away_team_runs_l7',          # 141
            'home_team_runs_allowed_l7',  # 142
            'away_team_runs_allowed_l7',  # 143
            'home_team_ops_l14',          # 144
            'away_team_ops_l14',          # 145
            'home_team_form_rating',      # 148
            'away_team_form_rating',      # 149
            'home_team_runs_l20',         # 150
            'away_team_runs_l20',         # 151
            'home_team_runs_allowed_l20', # 152
            'away_team_runs_allowed_l20', # 153
            'home_team_ops_l20',          # 154
            'away_team_ops_l20',          # 155
            'home_team_runs_l30',         # 156
            'away_team_runs_l30',         # 157
            'home_team_ops_l30',          # 158
            'away_team_ops_l30',          # 159
            'home_team_weighted_runs_scored',      # 188
            'away_team_weighted_runs_scored',      # 189
            'home_team_weighted_runs_allowed',     # 190
            'away_team_weighted_runs_allowed',     # 191
            'home_team_performance_consistency',   # 192
            'away_team_performance_consistency',   # 193
            'home_team_recent_momentum',           # 194
            'away_team_recent_momentum',           # 195
            'home_team_early_inning_strength',     # 196
            'away_team_early_inning_strength',     # 197
            'home_team_late_inning_strength',      # 198
            'away_team_late_inning_strength',      # 199
            'home_team_clutch_factor',             # 200
            'away_team_clutch_factor',             # 201
            'home_team_run_distribution_pattern',  # 202
            'away_team_run_distribution_pattern',  # 203
            'home_team_runs_pg',          # 211
            'away_team_runs_pg',          # 212
            
            # Season Team Stats
            'home_team_era',              # 235
            'away_team_era',              # 236
            'home_team_whip',             # 237
            'away_team_whip',             # 238
            'home_team_wins',             # 239
            'away_team_wins',             # 240
            'home_team_losses',           # 241
            'away_team_losses',           # 242
            'home_team_home_runs',        # 243
            'away_team_home_runs',        # 244
            'home_team_saves',            # 245
            'away_team_saves',            # 246
            'home_team_strikeouts',       # 247
            'away_team_strikeouts',       # 248
            'home_team_walks_allowed',    # 249
            'away_team_walks_allowed',    # 250
            'home_team_hits_allowed',     # 251
            'away_team_hits_allowed',     # 252
            'home_team_runs_allowed',     # 253
            'away_team_runs_allowed'      # 254
        ]
        
        # Feature importance weights based on your priorities
        self.feature_weights = {
            # Starting Pitcher - High importance
            'home_sp_season_era': 3.0,
            'away_sp_season_era': 3.0,
            'home_sp_whip': 2.5,
            'away_sp_whip': 2.5,
            'home_sp_era_l3starts': 2.8,
            'away_sp_era_l3starts': 2.8,
            
            # Recent Team Form - High importance
            'home_team_runs_l7': 2.5,
            'away_team_runs_l7': 2.5,
            'home_team_ops_l14': 2.2,
            'away_team_ops_l14': 2.2,
            'home_team_recent_momentum': 2.0,
            'away_team_recent_momentum': 2.0,
            
            # Team Offensive Power - Medium-High
            'home_team_ops': 2.0,
            'away_team_ops': 2.0,
            'home_team_wrc_plus': 1.8,
            'away_team_wrc_plus': 1.8,
            
            # Bullpen - Medium importance
            'home_bullpen_era': 1.5,
            'away_bullpen_era': 1.5,
            'home_bullpen_era_l30': 1.8,
            'away_bullpen_era_l30': 1.8,
            
            # Weather/Environment - Lower but important
            'temperature': 1.2,
            'wind_speed': 1.1,
            'ballpark_run_factor': 1.5,
            'ballpark_hr_factor': 1.3,
            
            # Default weight for other features
            'default': 1.0
        }
    
    def load_data(self, start_date='2025-07-01', end_date='2025-08-30'):
        """Load data with only your specified features"""
        
        log.info(f"Loading data from {start_date} to {end_date}")
        
        # Build feature list for SQL
        features_sql = ', '.join([f'"{feat}"' for feat in self.your_features])
        
        query = text(f"""
            SELECT 
                game_id, date, home_team, away_team, total_runs,
                {features_sql}
            FROM enhanced_games 
            WHERE date BETWEEN :start_date AND :end_date
            AND total_runs IS NOT NULL
            ORDER BY date
        """)
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={
                'start_date': start_date,
                'end_date': end_date
            })
        
        log.info(f"Loaded {len(df)} games with {len(self.your_features)} specified features")
        return df
    
    def prepare_features(self, df):
        """Prepare features with your specified weights"""
        
        # Handle categorical features
        categorical_features = ['weather_condition', 'wind_direction', 'venue', 'ballpark', 
                              'home_sp_hand', 'away_sp_hand']
        
        df_prepared = df.copy()
        
        # One-hot encode categorical features
        for feature in categorical_features:
            if feature in df_prepared.columns:
                dummies = pd.get_dummies(df_prepared[feature], prefix=feature, dummy_na=True)
                df_prepared = pd.concat([df_prepared, dummies], axis=1)
                df_prepared.drop(feature, axis=1, inplace=True)
        
        # Fill missing values with median for numeric features
        numeric_features = df_prepared.select_dtypes(include=[np.number]).columns
        numeric_features = [f for f in numeric_features if f not in ['game_id', 'total_runs']]
        
        for feature in numeric_features:
            median_val = df_prepared[feature].median()
            df_prepared[feature].fillna(median_val, inplace=True)
        
        # Apply feature weights
        for feature in numeric_features:
            weight = self.feature_weights.get(feature, self.feature_weights['default'])
            if weight != 1.0:
                df_prepared[feature] = df_prepared[feature] * weight
                log.info(f"Applied weight {weight} to {feature}")
        
        return df_prepared
    
    def train_and_evaluate(self, start_date='2025-07-01', end_date='2025-08-30'):
        """Train model using only your specified features"""
        
        # Load data
        df = self.load_data(start_date, end_date)
        
        if df.empty:
            log.error("No data loaded")
            return None
        
        # Prepare features
        df_prepared = self.prepare_features(df)
        
        # Split features and target
        feature_cols = [col for col in df_prepared.columns 
                       if col not in ['game_id', 'date', 'home_team', 'away_team', 'total_runs']]
        
        X = df_prepared[feature_cols]
        y = df_prepared['total_runs']
        
        log.info(f"Training with {len(feature_cols)} features from your specified list")
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X, y)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        log.info(f"Training Results:")
        log.info(f"  MAE: {mae:.3f}")
        log.info(f"  R²: {r2:.3f}")
        log.info(f"  Features used: {len(feature_cols)}")
        
        # Feature importance from your specified features only
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        log.info("Top 15 most important features from YOUR LIST:")
        for i, (_, row) in enumerate(feature_importance.head(15).iterrows(), 1):
            log.info(f"  {i:2d}. {row['feature']:30} ({row['importance']:.4f})")
        
        # Test predictions on recent games
        recent_df = self.load_data('2025-08-29', '2025-08-30')
        if not recent_df.empty:
            recent_prepared = self.prepare_features(recent_df)
            recent_X = recent_prepared[feature_cols]
            recent_pred = model.predict(recent_X)
            
            log.info("Sample predictions with YOUR features:")
            for i, (_, game) in enumerate(recent_df.iterrows()):
                if i < 5:  # Show first 5
                    log.info(f"  {game['away_team']} @ {game['home_team']}: {recent_pred[i]:.1f} runs")
        
        return {
            'model': model,
            'mae': mae,
            'r2': r2,
            'feature_importance': feature_importance,
            'feature_count': len(feature_cols)
        }

def main():
    """Test the model with only your specified features"""
    
    model = YourSpecifiedFeaturesModel()
    results = model.train_and_evaluate()
    
    if results:
        print(f"\n=== YOUR SPECIFIED FEATURES MODEL RESULTS ===")
        print(f"MAE: {results['mae']:.3f}")
        print(f"R²: {results['r2']:.3f}")
        print(f"Features used: {results['feature_count']} (from your specified list)")
        print(f"\nThis model uses ONLY the features you listed - no composites, no extras!")

if __name__ == "__main__":
    main()
