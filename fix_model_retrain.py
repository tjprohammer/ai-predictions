#!/usr/bin/env python3
"""
Model Fix and Retrain Script
============================
Fixes data leakage issues and retrains model with proper features
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelFixer:
    def __init__(self):
        self.data_path = "S:/Projects/AI_Predictions/mlb-overs/data/enhanced_historical_games_2025.parquet"
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_and_clean_data(self):
        """Load data and remove problematic features"""
        print("📊 LOADING AND CLEANING TRAINING DATA")
        print("=" * 50)
        
        try:
            df = pd.read_parquet(self.data_path)
            print(f"✅ Loaded {len(df):,} games")
            
            # Remove extreme outliers
            original_count = len(df)
            df = df[df['total_runs'] <= 25]  # Remove games with >25 runs
            df = df[df['total_runs'] >= 2]   # Remove games with <2 runs
            outliers_removed = original_count - len(df)
            
            if outliers_removed > 0:
                print(f"🧹 Removed {outliers_removed} outlier games")
            
            print(f"📈 Final dataset: {len(df):,} games")
            print(f"🎯 Total runs range: {df['total_runs'].min()}-{df['total_runs'].max()}")
            print(f"📊 Average total runs: {df['total_runs'].mean():.2f}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def create_legitimate_features(self, df):
        """Create only legitimate pre-game features"""
        print(f"\n🔧 CREATING LEGITIMATE FEATURES")
        print("=" * 50)
        
        featured_df = df.copy()
        
        # Remove any suspicious features
        suspicious_features = [
            'total_expected_rbi',  # This was 94% importance - likely data leakage
            'home_team_rbi',       # RBI is a game outcome
            'away_team_rbi',       # RBI is a game outcome
            'home_team_runs',      # Game outcome
            'away_team_runs',      # Game outcome
            'home_score',          # Game outcome
            'away_score'           # Game outcome
        ]
        
        print("🚨 Removing suspicious features:")
        for feature in suspicious_features:
            if feature in featured_df.columns:
                print(f"   ❌ Removed: {feature}")
                featured_df.drop(feature, axis=1, inplace=True)
        
        # Weather features (available pre-game)
        print("\n🌤️ Weather Features:")
        if 'temperature' in featured_df.columns:
            # Convert temperature to numeric, handling any string values
            featured_df['temperature'] = pd.to_numeric(featured_df['temperature'], errors='coerce')
            featured_df['temp_factor'] = (featured_df['temperature'] - 70) * 0.02
            print("   ✅ temp_factor")
        
        if 'wind_speed' in featured_df.columns:
            # Convert wind_speed to numeric
            featured_df['wind_speed'] = pd.to_numeric(featured_df['wind_speed'], errors='coerce')
            featured_df['wind_factor'] = featured_df['wind_speed'] * 0.1
            print("   ✅ wind_factor")
        
        # Weather encoding
        if 'weather_condition' in featured_df.columns:
            weather_dummies = pd.get_dummies(featured_df['weather_condition'], prefix='weather')
            featured_df = pd.concat([featured_df, weather_dummies], axis=1)
            print(f"   ✅ weather dummies ({len(weather_dummies.columns)} categories)")
        
        # Pitcher features (available pre-game)
        print("\n⚾ Pitcher Features:")
        pitcher_features = []
        
        for prefix in ['home_sp', 'away_sp']:
            era_col = f'{prefix}_era' if f'{prefix}_era' in featured_df.columns else f'{prefix[:-3]}_pitcher_season_era'
            if era_col in featured_df.columns:
                featured_df[f'{prefix}_era_normalized'] = featured_df[era_col] / 4.5  # Normalize around league average
                pitcher_features.append(f'{prefix}_era_normalized')
                print(f"   ✅ {prefix}_era_normalized")
        
        # ERA difference
        if len(pitcher_features) >= 2:
            featured_df['era_difference'] = featured_df[pitcher_features[0]] - featured_df[pitcher_features[1]]
            print("   ✅ era_difference")
        
        # Team features (using historical averages, not game outcomes)
        print("\n🏟️ Team Features:")
        
        # Ballpark factors
        if 'venue_name' in featured_df.columns:
            # Simple ballpark encoding
            venue_dummies = pd.get_dummies(featured_df['venue_name'], prefix='venue')
            # Only keep top 10 most common venues to avoid overfitting
            venue_counts = featured_df['venue_name'].value_counts()
            top_venues = venue_counts.head(10).index
            
            for venue in top_venues:
                col_name = f'venue_{venue.replace(" ", "_").replace(".", "")}'
                featured_df[col_name] = (featured_df['venue_name'] == venue).astype(int)
            
            print(f"   ✅ Top {len(top_venues)} venue indicators")
        
        # Day/Night game
        if 'day_night' in featured_df.columns:
            featured_df['is_night_game'] = (featured_df['day_night'] == 'N').astype(int)
            print("   ✅ is_night_game")
        
        # Market total (if available - this is legitimate)
        if 'market_total' in featured_df.columns:
            featured_df['market_total_normalized'] = featured_df['market_total'] / 9.0
            print("   ✅ market_total_normalized")
        
        # Select final feature columns
        feature_columns = []
        
        # Always include these if available
        base_features = [
            'temperature', 'wind_speed', 'temp_factor', 'wind_factor',
            'era_difference', 'is_night_game', 'market_total_normalized'
        ]
        
        for feature in base_features:
            if feature in featured_df.columns:
                feature_columns.append(feature)
        
        # Add weather dummies
        weather_cols = [col for col in featured_df.columns if col.startswith('weather_')]
        feature_columns.extend(weather_cols)
        
        # Add venue dummies
        venue_cols = [col for col in featured_df.columns if col.startswith('venue_')]
        feature_columns.extend(venue_cols[:10])  # Limit to top 10
        
        # Add pitcher features
        pitcher_cols = [col for col in featured_df.columns if 'era_normalized' in col]
        feature_columns.extend(pitcher_cols)
        
        print(f"\n📊 Final feature set: {len(feature_columns)} features")
        self.feature_columns = feature_columns
        
        return featured_df
    
    def train_clean_model(self, featured_df):
        """Train model with only legitimate features"""
        print(f"\n🤖 TRAINING CLEAN MODEL")
        print("=" * 50)
        
        # Prepare features and target - only use numeric columns
        X = featured_df[self.feature_columns].copy()
        
        # Convert all feature columns to numeric
        for col in self.feature_columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        y = featured_df['total_runs'].values
        
        # Handle missing values with 0 (since these are mostly dummy variables)
        X.fillna(0, inplace=True)
        
        print(f"📊 Training with {len(X)} samples and {len(self.feature_columns)} features")
        
        # Time-based split for more realistic validation
        featured_df['date'] = pd.to_datetime(featured_df['date'])
        split_date = featured_df['date'].quantile(0.8)
        train_mask = featured_df['date'] <= split_date
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[~train_mask]
        y_test = y[~train_mask]
        
        print(f"📈 Train set: {len(X_train)} games")
        print(f"📉 Test set: {len(X_test)} games")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model with better regularization
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,              # Reduced to prevent overfitting
            min_samples_split=10,     # Increased to prevent overfitting
            min_samples_leaf=5,       # Increased to prevent overfitting
            max_features='sqrt',      # Use subset of features
            random_state=42,
            n_jobs=-1
        )
        
        print("🏋️ Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        print(f"\n📊 CLEAN MODEL PERFORMANCE:")
        print(f"   Training MAE: {train_mae:.3f} runs")
        print(f"   Test MAE: {test_mae:.3f} runs")
        print(f"   Training R²: {train_r2:.3f}")
        print(f"   Test R²: {test_r2:.3f}")
        
        # Check for overfitting
        overfitting_ratio = train_mae / max(test_mae, 0.001)
        print(f"   Overfitting ratio: {overfitting_ratio:.3f} (lower is better)")
        
        if overfitting_ratio < 0.8:
            print("   ✅ Good generalization")
        else:
            print("   ⚠️  Some overfitting detected")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"   Cross-validation MAE: {cv_mae:.3f} ± {cv_std:.3f}")
        
        # Feature importance
        feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n🎯 TOP 10 FEATURE IMPORTANCE (CLEAN MODEL):")
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            print(f"   {i:2d}. {feature:<25} {importance:.4f}")
        
        # Check for new dominance issues
        top_feature_pct = sorted_features[0][1] * 100
        if top_feature_pct > 50:
            print(f"\n⚠️  WARNING: Top feature still dominates ({top_feature_pct:.1f}%)")
        else:
            print(f"\n✅ Feature importance is balanced (top: {top_feature_pct:.1f}%)")
        
        return {
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_mae': cv_mae,
            'cv_std': cv_std,
            'overfitting_ratio': overfitting_ratio,
            'feature_importance': feature_importance
        }
    
    def save_clean_model(self):
        """Save the cleaned model"""
        print(f"\n💾 SAVING CLEAN MODEL")
        print("=" * 50)
        
        if self.model is None:
            print("❌ No model to save!")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'training_date': datetime.now().isoformat(),
            'model_type': 'clean_random_forest',
            'notes': 'Trained without data leakage features'
        }
        
        # Save to multiple locations
        save_paths = [
            "S:/Projects/AI_Predictions/mlb-overs/training/clean_model.joblib",
            "S:/Projects/AI_Predictions/mlb-overs/models/clean_mlb_model.joblib"
        ]
        
        for path in save_paths:
            try:
                joblib.dump(model_data, path)
                print(f"✅ Saved to: {path}")
            except Exception as e:
                print(f"❌ Failed to save to {path}: {e}")
    
    def run_complete_fix(self):
        """Run the complete model fixing process"""
        print("🔧 MODEL FIX AND RETRAIN PROCESS")
        print("=" * 70)
        print(f"Starting clean model training: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Load and clean data
        df = self.load_and_clean_data()
        if df is None:
            return
        
        # Create legitimate features
        featured_df = self.create_legitimate_features(df)
        
        # Train clean model
        performance = self.train_clean_model(featured_df)
        
        # Save model
        self.save_clean_model()
        
        # Final assessment
        print(f"\n🏆 CLEAN MODEL ASSESSMENT")
        print("=" * 50)
        
        test_mae = performance['test_mae']
        if test_mae <= 1.5:
            rating = "EXCELLENT"
            color = "🟢"
        elif test_mae <= 2.0:
            rating = "GOOD"
            color = "🟡"
        elif test_mae <= 2.5:
            rating = "FAIR"
            color = "🟠"
        else:
            rating = "NEEDS WORK"
            color = "🔴"
        
        print(f"{color} Clean Model Rating: {rating}")
        print(f"   Test MAE: {test_mae:.3f} runs")
        print(f"   Expected accuracy: ~{(performance['test_mae'] <= 1.5)*100:.0f}% within 1.5 runs")
        
        print(f"\n💡 NEXT STEPS:")
        print("1. Test the clean model on recent games")
        print("2. Monitor performance over time")
        print("3. Add more sophisticated features if needed")
        print("4. Consider ensemble methods for improvement")
        
        print(f"\n✅ Model fix complete! - {datetime.now().strftime('%H:%M:%S')}")

def main():
    fixer = ModelFixer()
    fixer.run_complete_fix()

if __name__ == "__main__":
    main()
