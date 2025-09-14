"""
Smart Feature Model - Using only features that exist in your database
Based on your baseball priorities but filtered to available columns
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

class SmartFeaturePredictor:
    def __init__(self):
        # Your high-priority features that actually exist in the database
        self.feature_categories = {
            'weather': [
                'temperature', 'humidity', 'wind_speed', 'wind_direction_deg', 
                'air_pressure', 'dew_point'
            ],
            'pitcher_core': [
                'home_sp_season_era', 'away_sp_season_era', 
                'home_sp_whip', 'away_sp_whip',
                'home_sp_era_l3starts', 'away_sp_era_l3starts',
                'home_sp_days_rest', 'away_sp_days_rest'
            ],
            'team_offense': [
                'home_team_ops', 'away_team_ops', 
                'home_team_wrc_plus', 'away_team_wrc_plus',
                'home_team_iso', 'away_team_iso',
                'home_team_obp', 'away_team_obp',
                'home_team_slg', 'away_team_slg',
                'home_team_woba', 'away_team_woba'
            ],
            'ballpark': [
                'ballpark_run_factor', 'ballpark_hr_factor'
            ],
            'bullpen': [
                'home_bullpen_era', 'away_bullpen_era',
                'home_bullpen_era_l30', 'away_bullpen_era_l30',
                'home_bullpen_era_l14', 'away_bullpen_era_l14',
                'home_bullpen_fip', 'away_bullpen_fip',
                'home_bullpen_fatigue', 'away_bullpen_fatigue'
            ],
            'recent_form': [
                'home_team_runs_l7', 'away_team_runs_l7',
                'home_team_runs_l20', 'away_team_runs_l20',
                'home_team_ops_l14', 'away_team_ops_l14',
                'home_team_ops_l30', 'away_team_ops_l30',
                'home_team_runs_allowed_l7', 'away_team_runs_allowed_l7',
                'home_team_recent_momentum', 'away_team_recent_momentum'
            ],
            'season_performance': [
                'home_team_era', 'away_team_era',
                'home_team_runs_pg', 'away_team_runs_pg',
                'home_team_wins', 'away_team_wins',
                'home_team_losses', 'away_team_losses'
            ],
            'advanced_metrics': [
                'home_team_defensive_efficiency', 'away_team_defensive_efficiency',
                'home_team_offensive_efficiency', 'away_team_offensive_efficiency',
                'home_pitcher_quality', 'away_pitcher_quality',
                'home_lineup_strength', 'away_lineup_strength'
            ]
        }
        
        # Your baseball-smart weights based on importance
        self.category_weights = {
            'pitcher_core': 3.0,        # Most important - starting pitching
            'recent_form': 2.5,         # Very important - current form
            'team_offense': 2.2,        # Important - run scoring ability
            'bullpen': 2.0,             # Important - late-game performance
            'advanced_metrics': 1.8,    # Moderately important
            'season_performance': 1.5,  # Background context
            'ballpark': 1.3,            # Environment factors
            'weather': 1.1              # Minor influence
        }
        
        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
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
        """Load and prepare data with existing features only"""
        logger.info("Loading data from PostgreSQL...")
        
        # Get all features from categories
        all_features = []
        for category, features in self.feature_categories.items():
            all_features.extend(features)
        
        # Test which features actually exist
        with self.connect_to_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'enhanced_games'
            """)
            existing_columns = [row[0] for row in cur.fetchall()]
        
        # Filter to only existing features
        valid_features = [f for f in all_features if f in existing_columns]
        missing_features = [f for f in all_features if f not in existing_columns]
        
        logger.info(f"Using {len(valid_features)} valid features")
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features: {missing_features[:5]}...")
        
        # Build query with valid features
        query = f"""
        SELECT {', '.join(valid_features)}, total_runs
        FROM enhanced_games 
        WHERE total_runs IS NOT NULL 
        AND game_date >= '2023-04-01'
        ORDER BY game_date DESC
        LIMIT 6000
        """
        
        with self.connect_to_db() as conn:
            df = pd.read_sql(query, conn)
        
        logger.info(f"Loaded {len(df)} games with {len(valid_features)} features")
        return df, valid_features

    def clean_data(self, df, features):
        """Clean and prepare data for modeling"""
        logger.info("Cleaning data...")
        
        # Convert all features to numeric, handling any non-numeric values
        for feature in features:
            if feature in df.columns:
                df[feature] = pd.to_numeric(df[feature], errors='coerce')
                
                # Fill nulls with median
                if df[feature].isnull().sum() > 0:
                    median_val = df[feature].median()
                    df[feature] = df[feature].fillna(median_val)
        
        return df

    def apply_smart_weights(self, df, features):
        """Apply your baseball-smart category weights"""
        logger.info("Applying smart feature weights...")
        
        for category, weight in self.category_weights.items():
            category_features = self.feature_categories[category]
            for feature in category_features:
                if feature in df.columns:
                    df[feature] = df[feature] * weight
                    logger.info(f"Applied weight {weight} to {feature} ({category})")
        
        return df

    def train_and_evaluate(self):
        """Train model and evaluate performance"""
        logger.info("Starting smart feature model training...")
        
        # Load and prepare data
        df, features = self.load_data()
        df = self.clean_data(df, features)
        df = self.apply_smart_weights(df, features)
        
        # Remove any remaining NaN rows
        df = df.dropna()
        
        logger.info(f"Final training dataset: {len(df)} games with {len(features)} features")
        
        # Prepare features and target
        X = df[features]
        y = df['total_runs']
        
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
        
        print("\n" + "="*50)
        print("SMART BASEBALL FEATURE MODEL RESULTS")
        print("="*50)
        print(f"Training Games: {len(X_train)}")
        print(f"Test Games: {len(X_test)}")
        print(f"Features Used: {len(features)}")
        print()
        print("PERFORMANCE METRICS:")
        print(f"  Training MAE: {train_mae:.3f} runs")
        print(f"  Test MAE: {test_mae:.3f} runs")
        print(f"  Training R²: {train_r2:.3f}")
        print(f"  Test R²: {test_r2:.3f}")
        print(f"  Training Accuracy (±1 run): {train_acc:.1f}%")
        print(f"  Test Accuracy (±1 run): {test_acc:.1f}%")
        
        # Compare to baseline
        print(f"\nCOMPARISON TO CURRENT SYSTEM:")
        print(f"  Current MAE: 4.234 runs")
        print(f"  New MAE: {test_mae:.3f} runs")
        improvement = 4.234 - test_mae
        print(f"  Improvement: {improvement:.3f} runs ({improvement/4.234*100:.1f}%)")
        
        # Feature importance by category
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTOP 20 MOST IMPORTANT FEATURES:")
        print("-" * 50)
        for i, (_, row) in enumerate(feature_importance.head(20).iterrows()):
            # Find which category this feature belongs to
            category = "unknown"
            for cat, cat_features in self.feature_categories.items():
                if row['feature'] in cat_features:
                    category = cat
                    break
            print(f"{i+1:2d}. {row['feature']:<35} {row['importance']:.4f} ({category})")
        
        # Show category performance
        print(f"\nCATEGORY IMPORTANCE SUMMARY:")
        print("-" * 40)
        category_importance = {}
        for category, cat_features in self.feature_categories.items():
            cat_score = feature_importance[feature_importance['feature'].isin(cat_features)]['importance'].sum()
            category_importance[category] = cat_score
        
        for category, importance in sorted(category_importance.items(), key=lambda x: x[1], reverse=True):
            print(f"{category:<20} {importance:.4f}")
        
        # Sample predictions
        print(f"\nSAMPLE PREDICTIONS:")
        print("-" * 40)
        sample_indices = np.random.choice(len(X_test), 10, replace=False)
        for i, idx in enumerate(sample_indices):
            actual = y_test.iloc[idx]
            predicted = test_pred[idx]
            diff = abs(actual - predicted)
            status = "✓" if diff <= 1.0 else "✗"
            print(f"{i+1:2d}. Actual: {actual:.1f}  Predicted: {predicted:.1f}  Diff: {diff:.1f} {status}")
        
        # Performance assessment
        print(f"\nPERFORMANCE ASSESSMENT:")
        print("-" * 30)
        if test_mae < 3.0:
            print("🎯 EXCELLENT! Target MAE < 3.0 achieved!")
        elif test_mae < 3.5:
            print("✅ GOOD! Close to target MAE < 3.0")
        else:
            print("⚠️  More optimization needed for MAE < 3.0")
            
        if test_acc > 40:
            print("🎯 EXCELLENT! Target accuracy > 40% achieved!")
        elif test_acc > 30:
            print("✅ GOOD! Approaching target accuracy > 40%")
        else:
            print("⚠️  More work needed for accuracy > 40%")
        
        return {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'feature_importance': feature_importance,
            'category_importance': category_importance,
            'model': self.model,
            'features': features
        }

def main():
    logger.info("Starting Smart Baseball Feature Model...")
    predictor = SmartFeaturePredictor()
    results = predictor.train_and_evaluate()
    
    print(f"\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    
    if results['test_mae'] < 3.5:
        print("🚀 SUCCESS! This model shows significant improvement!")
        print("   Next steps: Optimize hyperparameters and deploy")
    else:
        print("📊 PROGRESS! Good foundation, needs more tuning")
        print("   Next steps: Feature engineering and data quality")

if __name__ == "__main__":
    main()
