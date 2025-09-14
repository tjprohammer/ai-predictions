"""
Clean version of your specified features model - fixes data type issues
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import psycopg2
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CleanFocusedPredictor:
    def __init__(self):
        # Your specified features organized by category
        self.feature_categories = {
            'weather': [
                'temperature', 'humidity', 'wind_speed', 'wind_direction', 
                'pressure', 'conditions', 'dewpoint'
            ],
            'pitcher_stats': [
                'home_sp_season_era', 'away_sp_season_era', 'home_sp_whip', 'away_sp_whip',
                'home_sp_era_l3starts', 'away_sp_era_l3starts', 'home_sp_k9_season', 'away_sp_k9_season',
                'home_sp_bb9_season', 'away_sp_bb9_season', 'home_sp_hr9_season', 'away_sp_hr9_season',
                'home_sp_babip_season', 'away_sp_babip_season', 'home_sp_lob_season', 'away_sp_lob_season',
                'home_sp_era_night', 'away_sp_era_night', 'home_sp_era_day', 'away_sp_era_day',
                'home_sp_innings_season', 'away_sp_innings_season', 'home_sp_quality_starts', 'away_sp_quality_starts',
                'home_sp_era_vs_division', 'away_sp_era_vs_division'
            ],
            'team_offense': [
                'home_team_ops', 'away_team_ops', 'home_team_wrc_plus', 'away_team_wrc_plus',
                'home_team_iso', 'away_team_iso', 'home_team_babip', 'away_team_babip',
                'home_team_k_rate', 'away_team_k_rate', 'home_team_bb_rate', 'away_team_bb_rate',
                'home_team_hr_rate', 'away_team_hr_rate', 'home_team_sb_rate', 'away_team_sb_rate',
                'home_team_avg_vs_rhp', 'away_team_avg_vs_rhp', 'home_team_avg_vs_lhp', 'away_team_avg_vs_lhp',
                'home_team_ops_vs_rhp', 'away_team_ops_vs_rhp', 'home_team_ops_vs_lhp', 'away_team_ops_vs_lhp'
            ],
            'ballpark': [
                'ballpark_run_factor', 'ballpark_hr_factor', 'ballpark_doubles_factor', 'ballpark_singles_factor'
            ],
            'bullpen': [
                'home_bullpen_era', 'away_bullpen_era', 'home_bullpen_whip', 'away_bullpen_whip',
                'home_bullpen_k9', 'away_bullpen_k9', 'home_bullpen_bb9', 'away_bullpen_bb9',
                'home_bullpen_hr9', 'away_bullpen_hr9', 'home_bullpen_saves', 'away_bullpen_saves',
                'home_bullpen_blown_saves', 'away_bullpen_blown_saves', 'home_bullpen_era_l30', 'away_bullpen_era_l30',
                'home_bullpen_era_home', 'away_bullpen_era_away', 'home_bullpen_era_vs_division', 'away_bullpen_era_vs_division',
                'home_bullpen_fip', 'away_bullpen_fip', 'home_bullpen_xfip', 'away_bullpen_xfip',
                'home_bullpen_siera', 'away_bullpen_siera', 'home_closer_era', 'away_closer_era',
                'home_closer_saves', 'away_closer_saves'
            ],
            'recent_form': [
                'home_team_runs_l7', 'away_team_runs_l7', 'home_team_runs_l14', 'away_team_runs_l14',
                'home_team_ops_l7', 'away_team_ops_l7', 'home_team_ops_l14', 'away_team_ops_l14',
                'home_team_era_l7', 'away_team_era_l7', 'home_team_era_l14', 'away_team_era_l14',
                'home_team_record_l10', 'away_team_record_l10', 'home_team_runs_allowed_l7', 'away_team_runs_allowed_l7',
                'home_team_runs_allowed_l14', 'away_team_runs_allowed_l14', 'home_team_bullpen_era_l7', 'away_team_bullpen_era_l7',
                'home_team_bullpen_era_l14', 'away_team_bullpen_era_l14', 'home_team_momentum', 'away_team_momentum',
                'home_team_vs_opponent_season', 'away_team_vs_opponent_season', 'home_team_road_record', 'away_team_road_record'
            ],
            'season_stats': [
                'home_team_runs_season', 'away_team_runs_season', 'home_team_era_season', 'away_team_era_season',
                'home_team_record', 'away_team_record', 'home_team_home_record', 'away_team_away_record',
                'home_team_vs_division', 'away_team_vs_division', 'home_team_runs_per_game', 'away_team_runs_per_game',
                'home_team_runs_allowed_per_game', 'away_team_runs_allowed_per_game', 'home_team_differential', 'away_team_differential',
                'home_team_pythag_wins', 'away_team_pythag_wins', 'home_team_strength_schedule', 'away_team_strength_schedule',
                'home_team_rest_days', 'away_team_rest_days'
            ]
        }
        
        # Weights for each category (your baseball-smart priorities)
        self.category_weights = {
            'pitcher_stats': 3.0,    # Most important
            'recent_form': 2.5,      # Very important
            'team_offense': 2.0,     # Important
            'bullpen': 1.8,          # Important
            'season_stats': 1.5,     # Moderate
            'ballpark': 1.3,         # Less important
            'weather': 1.1           # Least important
        }
        
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )

    def connect_to_db(self):
        """Connect to PostgreSQL database"""
        return psycopg2.connect(
            host="localhost",
            database="mlb",
            user="mlbuser",
            password="mlbpass"
        )

    def load_data(self):
        """Load and prepare data"""
        logger.info("Loading data from PostgreSQL...")
        
        # Get all your specified features
        all_features = []
        for category, features in self.feature_categories.items():
            all_features.extend(features)
        
        query = f"""
        SELECT {', '.join(all_features)}, total_runs
        FROM enhanced_games 
        WHERE total_runs IS NOT NULL 
        AND game_date >= '2023-04-01'
        ORDER BY game_date DESC
        LIMIT 5000
        """
        
        with self.connect_to_db() as conn:
            df = pd.read_sql(query, conn)
        
        logger.info(f"Loaded {len(df)} games with {len(all_features)} features")
        return df, all_features

    def clean_data(self, df, features):
        """Clean and prepare data for modeling"""
        logger.info("Cleaning data...")
        
        # Handle categorical columns
        categorical_cols = ['wind_direction', 'conditions']
        for col in categorical_cols:
            if col in df.columns:
                # Convert categorical to numeric using factorize
                df[col] = pd.Categorical(df[col]).codes
        
        # Fill numeric nulls with median
        numeric_features = [f for f in features if f in df.columns and df[f].dtype in ['float64', 'int64', 'int32', 'float32']]
        for feature in numeric_features:
            if df[feature].isnull().sum() > 0:
                median_val = df[feature].median()
                df[feature] = df[feature].fillna(median_val)
        
        # Ensure all features are numeric
        for feature in features:
            if feature in df.columns:
                df[feature] = pd.to_numeric(df[feature], errors='coerce')
                df[feature] = df[feature].fillna(df[feature].median())
        
        return df

    def apply_weights(self, df, features):
        """Apply category-based weights to features"""
        logger.info("Applying feature weights...")
        
        for category, weight in self.category_weights.items():
            category_features = self.feature_categories[category]
            for feature in category_features:
                if feature in df.columns:
                    df[feature] = df[feature] * weight
                    logger.info(f"Applied weight {weight} to {feature}")
        
        return df

    def train_and_evaluate(self):
        """Train model and evaluate performance"""
        logger.info("Starting model training...")
        
        # Load and prepare data
        df, features = self.load_data()
        df = self.clean_data(df, features)
        df = self.apply_weights(df, features)
        
        # Get available features (some might not exist in database)
        available_features = [f for f in features if f in df.columns]
        missing_features = [f for f in features if f not in df.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        logger.info(f"Training with {len(available_features)} features from your specified list")
        
        # Prepare features and target
        X = df[available_features]
        y = df['total_runs']
        
        # Remove any rows with NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        logger.info(f"Final dataset: {len(X)} games, {X.shape[1]} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        logger.info("Training RandomForest model...")
        self.model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Accuracy within 1 run
        train_acc = np.mean(np.abs(train_pred - y_train) <= 1.0) * 100
        test_acc = np.mean(np.abs(test_pred - y_test) <= 1.0) * 100
        
        print("\n=== YOUR FOCUSED FEATURE MODEL RESULTS ===")
        print(f"Training MAE: {train_mae:.3f}")
        print(f"Test MAE: {test_mae:.3f}")
        print(f"Training R²: {train_r2:.3f}")
        print(f"Test R²: {test_r2:.3f}")
        print(f"Training Accuracy (±1 run): {train_acc:.1f}%")
        print(f"Test Accuracy (±1 run): {test_acc:.1f}%")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n=== TOP 15 MOST IMPORTANT FEATURES ===")
        print(feature_importance.head(15).to_string(index=False))
        
        # Sample predictions
        print("\n=== SAMPLE PREDICTIONS ===")
        sample_indices = np.random.choice(len(X_test), 10, replace=False)
        for i, idx in enumerate(sample_indices):
            actual = y_test.iloc[idx]
            predicted = test_pred[idx]
            print(f"Game {i+1}: Actual = {actual:.1f}, Predicted = {predicted:.1f}, Diff = {abs(actual-predicted):.1f}")
        
        return {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'feature_importance': feature_importance,
            'model': self.model
        }

def main():
    logger.info("Starting Clean Focused Feature Model...")
    predictor = CleanFocusedPredictor()
    results = predictor.train_and_evaluate()
    
    print(f"\n=== COMPARISON TO BASELINE ===")
    print(f"Current system MAE: 4.234 runs")
    print(f"Your features MAE: {results['test_mae']:.3f} runs")
    improvement = 4.234 - results['test_mae']
    print(f"Improvement: {improvement:.3f} runs ({improvement/4.234*100:.1f}%)")
    
    if results['test_mae'] < 3.0:
        print("🎯 EXCELLENT! MAE below 3.0 target!")
    elif results['test_mae'] < 3.5:
        print("✅ GOOD! MAE approaching target!")
    else:
        print("⚠️  Still work needed to reach MAE < 3.0")

if __name__ == "__main__":
    main()
