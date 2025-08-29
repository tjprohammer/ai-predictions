#!/usr/bin/env python3
"""
Comprehensive Multi-Scenario A/B Testing Framework
=================================================

Advanced A/B testing framework to optimize the Ultra-80 incremental learning system across multiple dimensions:
- Learning window sizes (3d, 7d, 14d, 21d, 30d)
- Learning rates (SGD alpha parameters)
- Feature subsets and combinations
- Model architectures (SGD, Passive-Aggressive, Online Learning)
- Update frequencies (daily, every-3-days, weekly)
- Feature engineering parameters

This framework systematically tests combinations to find optimal configurations for MLB prediction accuracy.

Usage:
    python comprehensive_ab_testing.py --test-all
    python comprehensive_ab_testing.py --test-learning-windows
    python comprehensive_ab_testing.py --test-feature-combinations
    python comprehensive_ab_testing.py --test-model-architectures
"""

import argparse
import json
import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib
from itertools import combinations, product

# Add parent directories for imports
sys.path.append(str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class ComprehensiveABTestFramework:
    """
    Comprehensive A/B testing framework for incremental learning optimization
    
    Tests multiple dimensions:
    1. Learning Windows: 3, 7, 14, 21, 30 days
    2. Learning Rates: 0.001, 0.01, 0.1, adaptive
    3. Feature Combinations: Baseball intelligence subsets
    4. Model Types: SGD, Passive-Aggressive, Online Random Forest
    5. Update Frequencies: Daily, every 3 days, weekly
    6. Feature Engineering: Different recency windows, blending methods
    """
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb')
        self.engine = create_engine(self.database_url)
        
        # Test configurations
        self.learning_windows = [3, 7, 14, 21, 30]
        self.learning_rates = [0.001, 0.01, 0.1, 'adaptive']
        self.update_frequencies = ['daily', 'every_3_days', 'weekly']
        self.model_types = ['sgd', 'passive_aggressive', 'online_rf']
        
        # Feature subsets for testing
        self.feature_subsets = {
            'core_only': ['home_team_era', 'away_team_era', 'home_team_runs_pg', 'away_team_runs_pg'],
            'with_pitcher': ['home_sp_season_era', 'away_sp_season_era', 'home_sp_whip', 'away_sp_whip'],
            'with_bullpen': ['home_bullpen_era', 'away_bullpen_era', 'home_bullpen_era_l14', 'away_bullpen_era_l14'],
            'with_recency': ['home_team_runs_l7', 'away_team_runs_l7', 'home_sp_last_runs', 'away_sp_last_runs'],
            'with_advanced': ['combined_era', 'era_differential', 'combined_bullpen_era', 'bullpen_era_advantage'],
            'full_baseball_intelligence': []  # Will include all available numeric features
        }
        
        # Results storage
        self.test_results = {}
        self.performance_matrix = pd.DataFrame()
        
    def setup_test_environment(self):
        """Initialize test environment and validate data availability"""
        log.info("Setting up comprehensive A/B testing environment...")
        
        # Validate database connection
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM enhanced_games WHERE date >= '2025-04-01'"))
                game_count = result.scalar()
                log.info(f"Found {game_count} games available for testing")
                
                if game_count < 100:
                    raise ValueError("Insufficient data for comprehensive testing")
                    
        except Exception as e:
            log.error(f"Database setup failed: {e}")
            raise
            
        # Create results directory
        self.results_dir = Path("../../data/ab_test_results")
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        log.info("Test environment setup complete")
        
    def get_games_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load games data with all available features"""
        query = f"""
        SELECT 
            game_id, date, home_team_id, away_team_id, total_runs,
            -- Core team stats
            home_team_era, away_team_era, home_team_runs_pg, away_team_runs_pg,
            home_team_whip, away_team_whip,
            
            -- Enhanced pitcher features  
            home_sp_season_era, away_sp_season_era, home_sp_whip, away_sp_whip,
            home_sp_last_runs, away_sp_last_runs,
            home_sp_era_l3starts, away_sp_era_l3starts,
            
            -- Team runs and performance
            home_team_runs_l7, away_team_runs_l7,
            home_team_runs_l20, away_team_runs_l20,
            home_team_runs_l30, away_team_runs_l30,
            
            -- Bullpen quality
            home_bullpen_era, away_bullpen_era,
            home_bullpen_era_l14, away_bullpen_era_l14,
            home_bullpen_era_l30, away_bullpen_era_l30,
            home_bullpen_whip_l30, away_bullpen_whip_l30,
            
            -- Team handedness and lineup
            home_lineup_pct_r, away_lineup_pct_r,
            home_lineup_strength, away_lineup_strength,
            
            -- Environmental
            temperature, wind_speed, humidity,
            
            -- Market data
            market_total, opening_total,
            
            -- Advanced metrics
            home_pitcher_quality, away_pitcher_quality,
            combined_era, era_differential,
            combined_bullpen_era, bullpen_era_advantage
            
        FROM enhanced_games 
        WHERE date >= '{start_date}' AND date <= '{end_date}'
        AND total_runs IS NOT NULL
        ORDER BY date, game_id
        """
        
        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn)
            
        log.info(f"Loaded {len(df)} games from {start_date} to {end_date}")
        return df
        
    def prepare_features(self, df: pd.DataFrame, feature_subset: str) -> pd.DataFrame:
        """Prepare feature matrix based on specified subset"""
        
        # Define feature columns for each subset
        if feature_subset == 'core_only':
            feature_cols = ['home_team_era', 'away_team_era', 'home_team_runs_pg', 'away_team_runs_pg',
                           'home_team_whip', 'away_team_whip']
                           
        elif feature_subset == 'with_pitcher':
            feature_cols = ['home_team_era', 'away_team_era', 'home_team_runs_pg', 'away_team_runs_pg',
                           'home_sp_season_era', 'away_sp_season_era', 'home_sp_whip', 'away_sp_whip']
                           
        elif feature_subset == 'with_bullpen':
            feature_cols = ['home_team_era', 'away_team_era', 'home_team_runs_pg', 'away_team_runs_pg',
                           'home_bullpen_era', 'away_bullpen_era', 'home_bullpen_era_l14', 'away_bullpen_era_l14']
                           
        elif feature_subset == 'with_recency':
            feature_cols = ['home_team_era', 'away_team_era', 'home_team_runs_pg', 'away_team_runs_pg',
                           'home_team_runs_l7', 'away_team_runs_l7', 'home_sp_last_runs', 'away_sp_last_runs']
                           
        elif feature_subset == 'with_advanced':
            feature_cols = ['home_team_era', 'away_team_era', 'home_team_runs_pg', 'away_team_runs_pg',
                           'combined_era', 'era_differential', 'combined_bullpen_era', 'bullpen_era_advantage']
                           
        else:  # full_baseball_intelligence
            # Use all available numeric features
            feature_cols = [col for col in df.columns 
                           if col not in ['game_id', 'date', 'home_team_id', 'away_team_id', 'total_runs']
                           and df[col].dtype in ['float64', 'int64']]
        
        # Filter available columns and handle missing values
        available_cols = [col for col in feature_cols if col in df.columns]
        features_df = df[available_cols].fillna(df[available_cols].median())
        
        log.info(f"Prepared {len(available_cols)} features for subset '{feature_subset}'")
        return features_df
        
    def create_incremental_model(self, model_type: str, learning_rate: float) -> Any:
        """Create incremental learning model based on type and parameters"""
        
        if model_type == 'sgd':
            if learning_rate == 'adaptive':
                return SGDRegressor(learning_rate='adaptive', eta0=0.01, random_state=42)
            else:
                return SGDRegressor(learning_rate='constant', eta0=learning_rate, random_state=42)
                
        elif model_type == 'passive_aggressive':
            return PassiveAggressiveRegressor(C=1.0, random_state=42)
            
        elif model_type == 'online_rf':
            # For online RF, we'll use regular RF with incremental updates
            return RandomForestRegressor(n_estimators=10, random_state=42)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def test_learning_windows(self, df: pd.DataFrame) -> Dict:
        """Test different learning window sizes - extended range with detailed text output"""
        log.info("Testing extended learning window configurations...")
        
        # Extended range of learning windows - focused on most promising longer periods
        extended_windows = [14, 21, 30, 45, 60, 90]
        
        print("\n" + "="*120)
        print("🔍 EXTENDED LEARNING WINDOW A/B TESTING")
        print("="*120)
        print("Testing longer learning windows to find optimal configuration...")
        print(f"Total games in dataset: {len(df)}")
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        results = {}
        features_df = self.prepare_features(df, 'full_baseball_intelligence')
        
        print(f"\nUsing {len(features_df.columns)} features for testing")
        print("-"*140)
        print(f"{'Window':<8} {'MAE':<8} {'RMSE':<8} {'Corr':<8} {'O/U Acc':<8} {'Ultra-80':<10} {'Best ROI':<10} {'EV':<8} {'Games':<8}")
        print("-"*140)
        
        for window_size in extended_windows:
            try:
                window_results = self.run_incremental_backtest(
                    df, features_df, window_size, 'sgd', 0.01, 'daily'
                )
                
                results[f'{window_size}d'] = window_results
                
                # Display enhanced results immediately
                mae = window_results['mae']
                rmse = window_results['rmse'] 
                corr = window_results['correlation']
                ou_acc = window_results['over_under_accuracy']
                n_pred = window_results['prediction_count']
                
                # Ultra-80 metrics
                ultra_80_achieved = "❌ No"
                best_roi = "N/A"
                best_ev = "N/A"
                
                if 'best_ultra_80' in window_results:
                    ultra_80_info = window_results['best_ultra_80']
                    ultra_80_achieved = f"✅ {ultra_80_info['accuracy_percentage']:.1f}%"
                    best_roi = f"{ultra_80_info['roi_percentage']:.1f}%"
                    best_ev = f"{ultra_80_info['expected_value']:.3f}"
                elif 'summary' in window_results and window_results['summary']['ultra_80_achievable']:
                    ultra_80_achieved = "⚠️ Possible"
                
                print(f"{window_size:<8} {mae:<8.3f} {rmse:<8.3f} {corr:<8.3f} {ou_acc:<8.3f} {ultra_80_achieved:<10} {best_roi:<10} {best_ev:<8} {n_pred:<8}")
                
            except Exception as e:
                error_msg = str(e)[:10] + "..." if len(str(e)) > 10 else str(e)
                print(f"{window_size:<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'❌ ERROR':<10} {'N/A':<10} {'N/A':<8} {'0':<8}")
                results[f'{window_size}d'] = {'mae': float('inf'), 'error': str(e)}
        
        # Analysis and recommendations
        print("\n" + "="*140)
        print("📊 ULTRA-80 PERFORMANCE ANALYSIS")
        print("="*140)
        
        # Find best performing windows
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        # Find windows that achieve Ultra-80 (80%+ accuracy)
        ultra_80_windows = {}
        for window_key, result in successful_results.items():
            if 'best_ultra_80' in result:
                ultra_80_windows[window_key] = result['best_ultra_80']
        
        if ultra_80_windows:
            print("🎯 ULTRA-80 ACHIEVERS (80%+ Accuracy):")
            print("-" * 100)
            print(f"{'Window':<8} {'Accuracy':<10} {'Games':<8} {'ROI%':<8} {'EV':<8} {'Threshold':<10}")
            print("-" * 100)
            
            for window_key, ultra_info in ultra_80_windows.items():
                window_days = window_key.replace('d', '')
                print(f"{window_days:<8} {ultra_info['accuracy_percentage']:<10.1f} {ultra_info['games_count']:<8} {ultra_info['roi_percentage']:<8.1f} {ultra_info['expected_value']:<8.3f} {ultra_info['threshold']:<10}")
            
            # Best Ultra-80 performer
            best_ultra_80 = max(ultra_80_windows.items(), 
                               key=lambda x: (x[1]['accuracy_percentage'], x[1]['games_count']))
            
            print("-" * 100)
            print(f"🏆 BEST ULTRA-80: {best_ultra_80[0].replace('d', '')}-day window")
            print(f"   Accuracy: {best_ultra_80[1]['accuracy_percentage']:.1f}%")
            print(f"   Games: {best_ultra_80[1]['games_count']}")
            print(f"   ROI: {best_ultra_80[1]['roi_percentage']:.1f}%")
            print(f"   Expected Value: {best_ultra_80[1]['expected_value']:.3f}")
            print(f"   Confidence Threshold: {best_ultra_80[1]['threshold']}")
            
        else:
            print("⚠️  No windows achieved 80%+ accuracy threshold")
            
            # Show best achievable accuracy for each window
            print("\n📈 BEST ACHIEVABLE ACCURACY BY WINDOW:")
            print("-" * 80)
            print(f"{'Window':<8} {'Best Acc%':<12} {'Games':<8} {'ROI%':<8} {'EV':<8}")
            print("-" * 80)
            
            for window_key, result in successful_results.items():
                if 'conf_0.5' in result:  # Show lowest threshold results
                    best_conf = max([v for k, v in result.items() if k.startswith('conf_') and isinstance(v, dict)],
                                  key=lambda x: x['accuracy'])
                    window_days = window_key.replace('d', '')
                    print(f"{window_days:<8} {best_conf['accuracy_percentage']:<12.1f} {best_conf['games_count']:<8} {best_conf['roi_percentage']:<8.1f} {best_conf['expected_value']:<8.3f}")
        
        # Detailed confidence threshold analysis for best window
        best_mae_key = min(successful_results.keys(), key=lambda k: successful_results[k]['mae'])
        best_result = successful_results[best_mae_key]
        
        print(f"\n� DETAILED ANALYSIS - {best_mae_key.replace('d', '')}-DAY WINDOW:")
        print("-" * 90)
        print(f"{'Threshold':<10} {'Games':<8} {'Accuracy':<10} {'ROI%':<8} {'EV':<8} {'Status':<12}")
        print("-" * 90)
        
        for conf_key in sorted([k for k in best_result.keys() if k.startswith('conf_')]):
            conf_info = best_result[conf_key]
            status = "✅ Ultra-80" if conf_info['is_ultra_80'] else "❌ Below 80%"
            print(f"{conf_info['threshold']:<10} {conf_info['games_count']:<8} {conf_info['accuracy_percentage']:<10.1f} {conf_info['roi_percentage']:<8.1f} {conf_info['expected_value']:<8.3f} {status:<12}")
        
        # Find best performing windows by different criteria
        print(f"\n� PERFORMANCE RANKINGS:")
        
        # Best by MAE
        best_mae_key = min(successful_results.keys(), key=lambda k: successful_results[k]['mae'])
        best_mae = successful_results[best_mae_key]
        window_days = best_mae_key.replace('d', '')
        
        print(f"🎯 BEST PREDICTION ACCURACY (MAE): {window_days}-day window")
        print(f"   MAE: {best_mae['mae']:.4f}")
        print(f"   RMSE: {best_mae['rmse']:.4f}")
        print(f"   Correlation: {best_mae['correlation']:.4f}")
        print(f"   Over/Under Accuracy: {best_mae['over_under_accuracy']:.1%}")
        
        # Best by ROI (if any Ultra-80 exists)
        if ultra_80_windows:
            best_roi = max(ultra_80_windows.items(), key=lambda x: x[1]['roi_percentage'])
            print(f"\n� BEST ROI (Ultra-80): {best_roi[0].replace('d', '')}-day window")
            print(f"   ROI: {best_roi[1]['roi_percentage']:.1f}%")
            print(f"   Accuracy: {best_roi[1]['accuracy_percentage']:.1f}%")
            print(f"   Expected Value: {best_roi[1]['expected_value']:.3f}")
        
        # Compare to current 14-day standard
        if '14d' in successful_results:
            current_mae = successful_results['14d']['mae']
            if best_mae['mae'] < current_mae:
                improvement = ((current_mae - best_mae['mae']) / current_mae) * 100
                print(f"\n🚀 IMPROVEMENT over current 14-day window: {improvement:.1f}% better MAE")
                
                # Show Ultra-80 comparison if available
                if '14d' in ultra_80_windows and ultra_80_windows:
                    current_ultra = ultra_80_windows.get('14d')
                    best_ultra = max(ultra_80_windows.values(), key=lambda x: x['roi_percentage'])
                    if current_ultra:
                        roi_improvement = best_ultra['roi_percentage'] - current_ultra['roi_percentage']
                        print(f"🚀 ROI IMPROVEMENT: {roi_improvement:.1f}% better ROI")
            else:
                print(f"⚠️  Current 14-day window is still optimal for MAE")
        
        # Recommendations
        print(f"\n💡 ULTRA-80 RECOMMENDATIONS:")
        if ultra_80_windows:
            best_window = max(ultra_80_windows.items(), 
                            key=lambda x: (x[1]['accuracy_percentage'], x[1]['roi_percentage']))
            window_name = best_window[0].replace('d', '')
            print(f"   ✅ IMPLEMENT {window_name}-day learning window for Ultra-80 system")
            print(f"   ✅ Use confidence threshold of {best_window[1]['threshold']} for high-confidence picks")
            print(f"   ✅ Expected {best_window[1]['accuracy_percentage']:.1f}% accuracy on {best_window[1]['games_count']} games")
            print(f"   ✅ Projected ROI: {best_window[1]['roi_percentage']:.1f}%")
        else:
            best_achievable = {}
            for window_key, result in successful_results.items():
                conf_results = [v for k, v in result.items() if k.startswith('conf_') and isinstance(v, dict)]
                if conf_results:
                    best_conf = max(conf_results, key=lambda x: x['accuracy'])
                    best_achievable[window_key] = best_conf
            
            if best_achievable:
                best_overall = max(best_achievable.items(), key=lambda x: x[1]['accuracy'])
                window_name = best_overall[0].replace('d', '')
                accuracy = best_overall[1]['accuracy_percentage']
                print(f"   ⚠️  Best achievable: {accuracy:.1f}% accuracy with {window_name}-day window")
                print(f"   💡 Need to improve features or model to reach 80% threshold")
                print(f"   💡 Current best is {80 - accuracy:.1f} percentage points below Ultra-80")
        
        print("="*140)
        
        return results
        
    def test_learning_rates(self, df: pd.DataFrame) -> Dict:
        """Test different learning rates"""
        log.info("Testing learning rate configurations...")
        
        results = {}
        features_df = self.prepare_features(df, 'full_baseball_intelligence')
        
        for learning_rate in self.learning_rates:
            log.info(f"Testing learning rate: {learning_rate}")
            
            lr_results = self.run_incremental_backtest(
                df, features_df, 14, 'sgd', learning_rate, 'daily'
            )
            
            results[f'lr_{learning_rate}'] = lr_results
            
        return results
        
    def test_feature_combinations(self, df: pd.DataFrame) -> Dict:
        """Test different feature subset combinations"""
        log.info("Testing feature combination configurations...")
        
        results = {}
        
        for subset_name, _ in self.feature_subsets.items():
            log.info(f"Testing feature subset: {subset_name}")
            
            features_df = self.prepare_features(df, subset_name)
            
            subset_results = self.run_incremental_backtest(
                df, features_df, 14, 'sgd', 0.01, 'daily'
            )
            
            results[subset_name] = subset_results
            
        return results
        
    def test_model_architectures(self, df: pd.DataFrame) -> Dict:
        """Test different model architectures"""
        log.info("Testing model architecture configurations...")
        
        results = {}
        features_df = self.prepare_features(df, 'full_baseball_intelligence')
        
        for model_type in self.model_types:
            log.info(f"Testing model type: {model_type}")
            
            model_results = self.run_incremental_backtest(
                df, features_df, 14, model_type, 0.01, 'daily'
            )
            
            results[model_type] = model_results
            
        return results
        
    def test_update_frequencies(self, df: pd.DataFrame) -> Dict:
        """Test different update frequencies"""
        log.info("Testing update frequency configurations...")
        
        results = {}
        features_df = self.prepare_features(df, 'full_baseball_intelligence')
        
        for frequency in self.update_frequencies:
            log.info(f"Testing update frequency: {frequency}")
            
            freq_results = self.run_incremental_backtest(
                df, features_df, 14, 'sgd', 0.01, frequency
            )
            
            results[frequency] = freq_results
            
        return results
        
    def run_incremental_backtest(self, df: pd.DataFrame, features_df: pd.DataFrame, 
                                window_size: int, model_type: str, learning_rate: float,
                                update_frequency: str) -> Dict:
        """Run incremental learning backtest with specified configuration"""
        
        # Combine data
        full_df = pd.concat([df[['game_id', 'date', 'total_runs']], features_df], axis=1)
        full_df = full_df.dropna()
        
        if len(full_df) < 50:
            log.warning(f"Insufficient data for backtest: {len(full_df)} games")
            return {'error': 'insufficient_data'}
        
        # Sort by date
        full_df = full_df.sort_values('date').reset_index(drop=True)
        
        # Initialize model and scaler
        model = self.create_incremental_model(model_type, learning_rate)
        scaler = StandardScaler()
        
        predictions = []
        actuals = []
        prediction_details = []
        
        # Determine update frequency
        if update_frequency == 'daily':
            update_every = 1
        elif update_frequency == 'every_3_days':
            update_every = 3
        else:  # weekly
            update_every = 7
        
        # Incremental learning loop
        for i in range(window_size, len(full_df)):
            
            # Get training window
            train_start = max(0, i - window_size)
            train_data = full_df.iloc[train_start:i]
            
            if len(train_data) < 10:
                continue
                
            # Prepare training data
            X_train = train_data.drop(['game_id', 'date', 'total_runs'], axis=1)
            y_train = train_data['total_runs']
            
            # Current prediction data
            current_row = full_df.iloc[i]
            X_current = full_df.iloc[i:i+1].drop(['game_id', 'date', 'total_runs'], axis=1)
            y_current = current_row['total_runs']
            
            # Scale features
            X_train_scaled = scaler.fit_transform(X_train)
            X_current_scaled = scaler.transform(X_current)
            
            # Update model based on frequency
            if i % update_every == 0 or i == window_size:
                if model_type == 'online_rf':
                    # For RF, retrain on window
                    model.fit(X_train_scaled, y_train)
                else:
                    # For SGD/PA, partial fit
                    if hasattr(model, 'partial_fit'):
                        model.partial_fit(X_train_scaled, y_train)
                    else:
                        model.fit(X_train_scaled, y_train)
            
            # Make prediction
            try:
                pred = model.predict(X_current_scaled)[0]
                predictions.append(pred)
                actuals.append(y_current)
                
                # Store detailed prediction info for Ultra-80 analysis
                prediction_details.append({
                    'game_id': current_row['game_id'],
                    'date': current_row['date'],
                    'predicted': pred,
                    'actual': y_current,
                    'market_total': 8.5,  # Default market line
                    'prediction_confidence': abs(pred - 8.5),  # Distance from market
                    'over_prediction': pred > 8.5,
                    'actual_over': y_current > 8.5
                })
                
            except Exception as e:
                log.warning(f"Prediction failed at index {i}: {e}")
                continue
        
        # Calculate enhanced metrics including Ultra-80 specific ones
        if len(predictions) < 10:
            return {'error': 'insufficient_predictions'}
        
        # Basic metrics
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        correlation = np.corrcoef(actuals, predictions)[0, 1] if len(actuals) > 1 else 0
        
        # Over/under accuracy
        over_under_correct = sum(1 for a, p in zip(actuals, predictions) 
                                if (a > 8.5 and p > 8.5) or (a <= 8.5 and p <= 8.5))
        over_under_accuracy = over_under_correct / len(actuals) if actuals else 0
        
        # Ultra-80 Specific Metrics
        ultra_80_metrics = self.calculate_ultra_80_metrics(prediction_details)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'over_under_accuracy': over_under_accuracy,
            'prediction_count': len(predictions),
            'config': {
                'window_size': window_size,
                'model_type': model_type,
                'learning_rate': learning_rate,
                'update_frequency': update_frequency
            },
            **ultra_80_metrics  # Include all Ultra-80 specific metrics
        }
    
    def calculate_ultra_80_metrics(self, prediction_details: List[Dict]) -> Dict:
        """Calculate Ultra-80 specific performance metrics"""
        
        if not prediction_details:
            return {}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(prediction_details)
        
        # Define confidence thresholds for Ultra-80 analysis
        confidence_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        
        ultra_metrics = {}
        
        for threshold in confidence_thresholds:
            # High confidence predictions
            high_conf_mask = df['prediction_confidence'] >= threshold
            high_conf_games = df[high_conf_mask]
            
            if len(high_conf_games) == 0:
                continue
                
            # Accuracy for high confidence predictions
            correct_predictions = (
                (high_conf_games['over_prediction'] & high_conf_games['actual_over']) |
                (~high_conf_games['over_prediction'] & ~high_conf_games['actual_over'])
            ).sum()
            
            accuracy = correct_predictions / len(high_conf_games)
            
            # ROI Calculation (assuming -110 odds)
            # Win: +0.909 units, Loss: -1 unit
            wins = correct_predictions
            losses = len(high_conf_games) - correct_predictions
            total_roi = (wins * 0.909) - losses
            roi_percentage = (total_roi / len(high_conf_games)) * 100 if len(high_conf_games) > 0 else 0
            
            # Expected Value calculation
            win_rate = accuracy
            expected_value = (win_rate * 0.909) - ((1 - win_rate) * 1.0)
            
            # Store metrics for this threshold
            ultra_metrics[f'conf_{threshold}'] = {
                'threshold': threshold,
                'games_count': len(high_conf_games),
                'accuracy': accuracy,
                'accuracy_percentage': accuracy * 100,
                'roi_units': total_roi,
                'roi_percentage': roi_percentage,
                'expected_value': expected_value,
                'wins': wins,
                'losses': losses,
                'is_ultra_80': accuracy >= 0.80  # 80%+ accuracy threshold
            }
        
        # Find best Ultra-80 threshold (80%+ accuracy with most games)
        ultra_80_thresholds = {k: v for k, v in ultra_metrics.items() if v['is_ultra_80']}
        
        if ultra_80_thresholds:
            # Best threshold = highest accuracy, then most games
            best_ultra_80 = max(ultra_80_thresholds.values(), 
                               key=lambda x: (x['accuracy'], x['games_count']))
            ultra_metrics['best_ultra_80'] = best_ultra_80
        
        # Overall summary statistics
        ultra_metrics['summary'] = {
            'total_predictions': len(df),
            'overall_accuracy': df.apply(lambda row: 
                (row['over_prediction'] and row['actual_over']) or 
                (not row['over_prediction'] and not row['actual_over']), axis=1).mean(),
            'avg_confidence': df['prediction_confidence'].mean(),
            'max_confidence': df['prediction_confidence'].max(),
            'ultra_80_achievable': any(v['is_ultra_80'] for v in ultra_metrics.values() if isinstance(v, dict) and 'is_ultra_80' in v)
        }
        
        return ultra_metrics
        
    def run_comprehensive_test(self, start_date: str = '2025-04-01', end_date: str = '2025-08-27') -> Dict:
        """Run comprehensive A/B testing across all dimensions"""
        log.info("Starting comprehensive A/B testing...")
        
        # Load data
        df = self.get_games_data(start_date, end_date)
        
        comprehensive_results = {
            'test_config': {
                'start_date': start_date,
                'end_date': end_date,
                'total_games': len(df),
                'test_timestamp': datetime.now().isoformat()
            },
            'learning_windows': self.test_learning_windows(df),
            'learning_rates': self.test_learning_rates(df),
            'feature_combinations': self.test_feature_combinations(df),
            'model_architectures': self.test_model_architectures(df),
            'update_frequencies': self.test_update_frequencies(df)
        }
        
        return comprehensive_results
        
    def generate_optimization_report(self, results: Dict) -> Dict:
        """Generate optimization recommendations based on test results"""
        log.info("Generating optimization recommendations...")
        
        recommendations = {
            'best_configurations': {},
            'performance_rankings': {},
            'statistical_significance': {},
            'optimization_insights': []
        }
        
        # Find best configuration for each test dimension
        for test_type, test_results in results.items():
            if test_type == 'test_config':
                continue
                
            best_config = None
            best_mae = float('inf')
            
            for config_name, config_results in test_results.items():
                if isinstance(config_results, dict) and 'mae' in config_results:
                    if config_results['mae'] < best_mae:
                        best_mae = config_results['mae']
                        best_config = config_name
            
            if best_config:
                recommendations['best_configurations'][test_type] = {
                    'config': best_config,
                    'mae': best_mae,
                    'details': test_results[best_config]
                }
        
        # Generate insights
        insights = []
        
        if 'learning_windows' in recommendations['best_configurations']:
            best_window = recommendations['best_configurations']['learning_windows']['config']
            insights.append(f"Optimal learning window: {best_window}")
            
        if 'feature_combinations' in recommendations['best_configurations']:
            best_features = recommendations['best_configurations']['feature_combinations']['config']
            insights.append(f"Best feature combination: {best_features}")
            
        recommendations['optimization_insights'] = insights
        
        return recommendations
        
    def save_results(self, results: Dict, filename: str = None):
        """Save test results to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_ab_test_results_{timestamp}.json"
            
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        log.info(f"Results saved to {filepath}")
        
    def create_performance_visualization(self, results: Dict):
        """Create comprehensive performance visualization"""
        log.info("Creating performance visualizations...")
        
        # Setup matplotlib
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive A/B Testing Results - Ultra-80 Incremental Learning', fontsize=16)
        
        # Learning Windows comparison
        if 'learning_windows' in results:
            ax = axes[0, 0]
            windows = []
            maes = []
            
            for config, result in results['learning_windows'].items():
                if isinstance(result, dict) and 'mae' in result:
                    windows.append(config)
                    maes.append(result['mae'])
                    
            ax.bar(windows, maes, color='skyblue')
            ax.set_title('Learning Window Performance')
            ax.set_ylabel('MAE')
            ax.set_xlabel('Window Size')
            
        # Feature combinations comparison
        if 'feature_combinations' in results:
            ax = axes[0, 1]
            features = []
            maes = []
            
            for config, result in results['feature_combinations'].items():
                if isinstance(result, dict) and 'mae' in result:
                    features.append(config)
                    maes.append(result['mae'])
                    
            ax.bar(features, maes, color='lightgreen')
            ax.set_title('Feature Combination Performance')
            ax.set_ylabel('MAE')
            ax.tick_params(axis='x', rotation=45)
            
        # Model architectures comparison
        if 'model_architectures' in results:
            ax = axes[0, 2]
            models = []
            maes = []
            
            for config, result in results['model_architectures'].items():
                if isinstance(result, dict) and 'mae' in result:
                    models.append(config)
                    maes.append(result['mae'])
                    
            ax.bar(models, maes, color='salmon')
            ax.set_title('Model Architecture Performance')
            ax.set_ylabel('MAE')
            
        # Learning rates comparison
        if 'learning_rates' in results:
            ax = axes[1, 0]
            rates = []
            maes = []
            
            for config, result in results['learning_rates'].items():
                if isinstance(result, dict) and 'mae' in result:
                    rates.append(config)
                    maes.append(result['mae'])
                    
            ax.bar(rates, maes, color='orange')
            ax.set_title('Learning Rate Performance')
            ax.set_ylabel('MAE')
            
        # Update frequencies comparison
        if 'update_frequencies' in results:
            ax = axes[1, 1]
            frequencies = []
            maes = []
            
            for config, result in results['update_frequencies'].items():
                if isinstance(result, dict) and 'mae' in result:
                    frequencies.append(config)
                    maes.append(result['mae'])
                    
            ax.bar(frequencies, maes, color='purple')
            ax.set_title('Update Frequency Performance')
            ax.set_ylabel('MAE')
            
        # Overall performance heatmap
        ax = axes[1, 2]
        ax.text(0.5, 0.5, 'Comprehensive\nPerformance\nSummary', 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title('Summary Statistics')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(self.results_dir / f"comprehensive_ab_test_visualization_{timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        
        log.info("Visualization saved")
        plt.show()

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Comprehensive A/B Testing for Ultra-80 Incremental Learning')
    parser.add_argument('--test-all', action='store_true', help='Run all A/B tests')
    parser.add_argument('--test-learning-windows', action='store_true', help='Test learning window sizes')
    parser.add_argument('--test-feature-combinations', action='store_true', help='Test feature combinations')
    parser.add_argument('--test-model-architectures', action='store_true', help='Test model architectures')
    parser.add_argument('--test-learning-rates', action='store_true', help='Test learning rates')
    parser.add_argument('--test-update-frequencies', action='store_true', help='Test update frequencies')
    parser.add_argument('--start-date', default='2025-04-01', help='Start date for testing')
    parser.add_argument('--end-date', default='2025-08-27', help='End date for testing')
    parser.add_argument('--output-file', help='Output filename for results')
    
    args = parser.parse_args()
    
    # Initialize framework
    framework = ComprehensiveABTestFramework()
    framework.setup_test_environment()
    
    # Load data
    df = framework.get_games_data(args.start_date, args.end_date)
    
    results = {
        'test_config': {
            'start_date': args.start_date,
            'end_date': args.end_date,
            'total_games': len(df),
            'test_timestamp': datetime.now().isoformat()
        }
    }
    
    # Run specified tests
    if args.test_all:
        log.info("Running comprehensive A/B testing...")
        results.update(framework.run_comprehensive_test(args.start_date, args.end_date))
        
    elif args.test_learning_windows:
        results['learning_windows'] = framework.test_learning_windows(df)
        
    elif args.test_feature_combinations:
        results['feature_combinations'] = framework.test_feature_combinations(df)
        
    elif args.test_model_architectures:
        results['model_architectures'] = framework.test_model_architectures(df)
        
    elif args.test_learning_rates:
        results['learning_rates'] = framework.test_learning_rates(df)
        
    elif args.test_update_frequencies:
        results['update_frequencies'] = framework.test_update_frequencies(df)
        
    else:
        log.info("No specific test selected, running learning windows test...")
        results['learning_windows'] = framework.test_learning_windows(df)
    
    # Generate recommendations
    recommendations = framework.generate_optimization_report(results)
    results['recommendations'] = recommendations
    
    # Save results
    framework.save_results(results, args.output_file)
    
    # Only create visualizations for comprehensive tests, not individual tests
    if args.test_all:
        framework.create_performance_visualization(results)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPREHENSIVE A/B TESTING SUMMARY")
    print("="*60)
    
    for test_type, best_config in recommendations['best_configurations'].items():
        print(f"\n{test_type.upper()}:")
        print(f"  Best Configuration: {best_config['config']}")
        print(f"  MAE: {best_config['mae']:.3f}")
        
    print(f"\nOptimization Insights:")
    for insight in recommendations['optimization_insights']:
        print(f"  • {insight}")
        
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
