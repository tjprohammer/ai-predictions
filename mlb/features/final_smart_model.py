"""
FINAL Smart Feature Model - Using your baseball priorities with existing database columns
This is the working version that will show you real results vs your current 4.234 MAE
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

class FinalSmartPredictor:
    def __init__(self):
        # Your baseball-smart features that exist in the database
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
                'home_bullpen_fip', 'away_bullpen_fip'
            ],
            'recent_form': [
                'home_team_runs_l7', 'away_team_runs_l7',
                'home_team_runs_l20', 'away_team_runs_l20',
                'home_team_ops_l14', 'away_team_ops_l14',
                'home_team_ops_l30', 'away_team_ops_l30',
                'home_team_runs_allowed_l7', 'away_team_runs_allowed_l7'
            ],
            'season_performance': [
                'home_team_era', 'away_team_era',
                'home_team_runs_pg', 'away_team_runs_pg',
                'home_team_wins', 'away_team_wins',
                'home_team_losses', 'away_team_losses'
            ]
        }
        
        # Your baseball-smart weights - emphasizing what matters most
        self.category_weights = {
            'pitcher_core': 3.0,        # Most important - starting pitching
            'recent_form': 2.5,         # Very important - current form
            'team_offense': 2.2,        # Important - run scoring ability
            'bullpen': 2.0,             # Important - late-game performance
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

    def load_and_prepare_data(self):
        """Load data and prepare features"""
        logger.info("Loading data from PostgreSQL...")
        
        # Get all features
        all_features = []
        for category, features in self.feature_categories.items():
            all_features.extend(features)
        
        # Simple query with recent data only
        query = f"""
        SELECT {', '.join(all_features)}, total_runs
        FROM enhanced_games 
        WHERE total_runs IS NOT NULL 
        AND total_runs BETWEEN 2 AND 25
        ORDER BY id DESC
        LIMIT 5000
        """
        
        conn = psycopg2.connect(
            host="localhost",
            database="mlb", 
            user="mlbuser",
            password="mlbpass"
        )
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        logger.info(f"Loaded {len(df)} games with {len(all_features)} features")
        
        # Clean data
        for feature in all_features:
            if feature in df.columns:
                df[feature] = pd.to_numeric(df[feature], errors='coerce')
                df[feature] = df[feature].fillna(df[feature].median())
        
        # Apply your smart weights
        for category, weight in self.category_weights.items():
            for feature in self.feature_categories[category]:
                if feature in df.columns:
                    df[feature] = df[feature] * weight
        
        # Remove any NaN rows
        df = df.dropna()
        
        return df, all_features

    def train_and_evaluate(self):
        """Train model and show results"""
        logger.info("Training your smart baseball model...")
        
        df, features = self.load_and_prepare_data()
        
        # Prepare data
        X = df[features]
        y = df['total_runs']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Predict
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Metrics
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_acc = np.mean(np.abs(train_pred - y_train) <= 1.0) * 100
        test_acc = np.mean(np.abs(test_pred - y_test) <= 1.0) * 100
        
        # Results
        print("\n" + "="*60)
        print("YOUR SMART BASEBALL FEATURE MODEL RESULTS")
        print("="*60)
        print(f"Training Games: {len(X_train):,}")
        print(f"Test Games: {len(X_test):,}")
        print(f"Features Used: {len(features)}")
        
        print(f"\nPERFORMANCE vs CURRENT SYSTEM:")
        print(f"{'Current MAE:':<20} 4.234 runs")
        print(f"{'Your Model MAE:':<20} {test_mae:.3f} runs")
        improvement = 4.234 - test_mae
        print(f"{'Improvement:':<20} {improvement:.3f} runs ({improvement/4.234*100:.1f}%)")
        
        print(f"\nACCURACY METRICS:")
        print(f"{'R² Score:':<20} {test_r2:.3f}")
        print(f"{'±1 Run Accuracy:':<20} {test_acc:.1f}%")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTOP 15 FEATURES:")
        for i, (_, row) in enumerate(importance.head(15).iterrows()):
            print(f"{i+1:2d}. {row['feature']:<30} {row['importance']:.4f}")
        
        # Sample predictions to verify model works
        print(f"\nSAMPLE PREDICTIONS:")
        sample_idx = np.random.choice(len(X_test), 8, replace=False)
        for i, idx in enumerate(sample_idx):
            actual = y_test.iloc[idx]
            pred = test_pred[idx]
            diff = abs(actual - pred)
            status = "✓" if diff <= 1.0 else "✗"
            print(f"{i+1}. Actual: {actual:.1f}  Predicted: {pred:.1f}  Diff: {diff:.1f} {status}")
        
        # Assessment
        print(f"\nASSESSMENT:")
        if test_mae < 3.0:
            print("🎯 EXCELLENT! Target MAE < 3.0 achieved!")
        elif test_mae < 3.5:
            print("✅ VERY GOOD! Close to target")
        elif test_mae < 4.0:
            print("📈 GOOD PROGRESS! Improvement shown")
        else:
            print("⚠️  Needs more work")
            
        return {
            'mae': test_mae,
            'r2': test_r2,
            'accuracy': test_acc,
            'improvement': improvement
        }

def main():
    predictor = FinalSmartPredictor()
    results = predictor.train_and_evaluate()
    
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if results['improvement'] > 0.5:
        print("🚀 SIGNIFICANT IMPROVEMENT! Your features work much better!")
    elif results['improvement'] > 0.2:
        print("📈 GOOD IMPROVEMENT! On the right track!")
    else:
        print("📊 Some progress, need more optimization")
        
    print(f"\nNext steps:")
    print(f"- If MAE < 3.5: Deploy this model")
    print(f"- If MAE > 3.5: Tune hyperparameters or add more features")

if __name__ == "__main__":
    main()
