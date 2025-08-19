#!/usr/bin/env python3
"""
Enhanced ML Training and Validation System with Graphs
======================================================

This script:
1. Trains the model on the last 20+ games with all features
2. Tracks predictions vs actual outcomes with pitcher matchups
3. Generates visual graphs for historical performance
4. Creates comprehensive validation reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date, timedelta
import json
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class EnhancedMLTrainerWithGraphs:
    def __init__(self):
        self.engine = create_engine('postgresql://mlbuser:mlbpass@localhost:5432/mlb', echo=False)
        self.model = None
        self.feature_columns = None
        self.validation_results = []
        
    def collect_last_20_games_data(self):
        """Collect comprehensive data from the last 20+ MLB games"""
        print("📊 Collecting data from last 20+ games...")
        
        with self.engine.begin() as conn:
            # Get the last 25 completed games with all features
            query = """
            SELECT 
                date, game_id, home_team, away_team, home_score, away_score, total_runs,
                venue_name, temperature, wind_speed, wind_direction, weather_condition,
                home_sp_id, away_sp_id, home_sp_er, away_sp_er, home_sp_ip, away_sp_ip,
                home_sp_k, away_sp_k, home_sp_bb, away_sp_bb, home_sp_h, away_sp_h,
                home_team_hits, away_team_hits, home_team_runs, away_team_runs,
                home_team_rbi, away_team_rbi, home_team_lob, away_team_lob
            FROM enhanced_games 
            WHERE total_runs IS NOT NULL 
                AND home_score IS NOT NULL 
                AND away_score IS NOT NULL
                AND date >= CURRENT_DATE - INTERVAL '30 days'
            ORDER BY date DESC 
            LIMIT 25
            """
            
            games_data = pd.read_sql(query, conn)
            
            if games_data.empty:
                print("❌ No recent games found")
                return pd.DataFrame()
                
            print(f"✅ Found {len(games_data)} recent completed games")
            
            # Get team offensive stats for each game date
            team_stats_list = []
            for _, game in games_data.iterrows():
                game_date = game['date']
                home_team = game['home_team']
                away_team = game['away_team']
                
                # Map full team names to abbreviations
                team_mapping = {
                    'Arizona Diamondbacks': 'AZ', 'Atlanta Braves': 'ATL', 'Baltimore Orioles': 'BAL',
                    'Boston Red Sox': 'BOS', 'Chicago Cubs': 'CHC', 'Chicago White Sox': 'CWS',
                    'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE', 'Colorado Rockies': 'COL',
                    'Detroit Tigers': 'DET', 'Houston Astros': 'HOU', 'Kansas City Royals': 'KC',
                    'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD', 'Miami Marlins': 'MIA',
                    'Milwaukee Brewers': 'MIL', 'Minnesota Twins': 'MIN', 'New York Mets': 'NYM',
                    'New York Yankees': 'NYY', 'Oakland Athletics': 'ATH', 'Philadelphia Phillies': 'PHI',
                    'Pittsburgh Pirates': 'PIT', 'San Diego Padres': 'SD', 'San Francisco Giants': 'SF',
                    'Seattle Mariners': 'SEA', 'St. Louis Cardinals': 'STL', 'Tampa Bay Rays': 'TB',
                    'Texas Rangers': 'TEX', 'Toronto Blue Jays': 'TOR', 'Washington Nationals': 'WSH'
                }
                
                home_abbr = team_mapping.get(home_team, home_team[:3].upper())
                away_abbr = team_mapping.get(away_team, away_team[:3].upper())
                
                # Get team stats as of game date
                home_stats_query = """
                SELECT team, runs_pg, ba, woba, bb_pct, k_pct, ops, slg_pct
                FROM teams_offense_daily 
                WHERE team = %s AND date <= %s 
                ORDER BY date DESC LIMIT 1
                """
                
                away_stats_query = """
                SELECT team, runs_pg, ba, woba, bb_pct, k_pct, ops, slg_pct
                FROM teams_offense_daily 
                WHERE team = %s AND date <= %s 
                ORDER BY date DESC LIMIT 1
                """
                
                try:
                    home_stats = pd.read_sql(home_stats_query, conn, params=[home_abbr, game_date])
                    away_stats = pd.read_sql(away_stats_query, conn, params=[away_abbr, game_date])
                    
                    team_stats_list.append({
                        'game_id': game['game_id'],
                        'home_runs_pg': home_stats.iloc[0]['runs_pg'] if not home_stats.empty else 4.0,
                        'home_ba': home_stats.iloc[0]['ba'] if not home_stats.empty else 0.250,
                        'home_woba': home_stats.iloc[0]['woba'] if not home_stats.empty else 0.310,
                        'home_bb_pct': home_stats.iloc[0]['bb_pct'] if not home_stats.empty else 8.0,
                        'home_k_pct': home_stats.iloc[0]['k_pct'] if not home_stats.empty else 22.0,
                        'away_runs_pg': away_stats.iloc[0]['runs_pg'] if not away_stats.empty else 4.0,
                        'away_ba': away_stats.iloc[0]['ba'] if not away_stats.empty else 0.250,
                        'away_woba': away_stats.iloc[0]['woba'] if not away_stats.empty else 0.310,
                        'away_bb_pct': away_stats.iloc[0]['bb_pct'] if not away_stats.empty else 8.0,
                        'away_k_pct': away_stats.iloc[0]['k_pct'] if not away_stats.empty else 22.0,
                    })
                except Exception as e:
                    print(f"Warning: Error getting team stats for game {game['game_id']}: {e}")
                    team_stats_list.append({
                        'game_id': game['game_id'],
                        'home_runs_pg': 4.0, 'home_ba': 0.250, 'home_woba': 0.310,
                        'home_bb_pct': 8.0, 'home_k_pct': 22.0,
                        'away_runs_pg': 4.0, 'away_ba': 0.250, 'away_woba': 0.310,
                        'away_bb_pct': 8.0, 'away_k_pct': 22.0,
                    })
            
            # Merge team stats with games data
            team_stats_df = pd.DataFrame(team_stats_list)
            enhanced_data = games_data.merge(team_stats_df, on='game_id', how='left')
            
            print(f"✅ Enhanced data with team stats: {len(enhanced_data)} games")
            return enhanced_data
    
    def create_enhanced_features(self, data):
        """Create comprehensive feature set for ML training"""
        print("🔧 Creating enhanced feature set...")
        
        # Handle missing values
        data = data.fillna({
            'temperature': 75, 'wind_speed': 0, 
            'home_sp_er': 0, 'away_sp_er': 0, 'home_sp_ip': 1, 'away_sp_ip': 1,
            'home_sp_k': 0, 'away_sp_k': 0, 'home_sp_bb': 0, 'away_sp_bb': 0,
            'home_sp_h': 0, 'away_sp_h': 0, 'home_team_hits': 0, 'away_team_hits': 0,
            'home_team_rbi': 0, 'away_team_rbi': 0, 'home_team_lob': 0, 'away_team_lob': 0
        })
        
        features_df = pd.DataFrame()
        
        # Team offensive stats
        features_df['home_runs_pg'] = data['home_runs_pg']
        features_df['away_runs_pg'] = data['away_runs_pg']
        features_df['home_ba'] = data['home_ba']
        features_df['away_ba'] = data['away_ba']
        features_df['home_woba'] = data['home_woba']
        features_df['away_woba'] = data['away_woba']
        features_df['home_bb_pct'] = data['home_bb_pct']
        features_df['away_bb_pct'] = data['away_bb_pct']
        features_df['home_k_pct'] = data['home_k_pct']
        features_df['away_k_pct'] = data['away_k_pct']
        
        # Weather features
        features_df['temperature'] = pd.to_numeric(data['temperature'], errors='coerce').fillna(75)
        features_df['wind_speed_mph'] = pd.to_numeric(data['wind_speed'], errors='coerce').fillna(0)
        
        # Wind direction encoding
        wind_direction_mapping = {
            'Out To CF': 3, 'Out To LF': 2, 'Out To RF': 2,
            'In From CF': -2, 'In From LF': -1, 'In From RF': -1,
            'L To R': 0, 'R To L': 0, 'Varies': 0
        }
        features_df['wind_direction_effect'] = data['wind_direction'].map(wind_direction_mapping).fillna(0)
        
        # Pitcher performance features
        features_df['home_sp_era_game'] = np.where(
            data['home_sp_ip'] > 0,
            (data['home_sp_er'] * 9) / data['home_sp_ip'],
            4.50
        )
        features_df['away_sp_era_game'] = np.where(
            data['away_sp_ip'] > 0,
            (data['away_sp_er'] * 9) / data['away_sp_ip'],
            4.50
        )
        
        features_df['home_sp_whip_game'] = np.where(
            data['home_sp_ip'] > 0,
            (data['home_sp_h'] + data['home_sp_bb']) / data['home_sp_ip'],
            1.30
        )
        features_df['away_sp_whip_game'] = np.where(
            data['away_sp_ip'] > 0,
            (data['away_sp_h'] + data['away_sp_bb']) / data['away_sp_ip'],
            1.30
        )
        
        features_df['home_sp_k_per_9'] = np.where(
            data['home_sp_ip'] > 0,
            (data['home_sp_k'] * 9) / data['home_sp_ip'],
            8.0
        )
        features_df['away_sp_k_per_9'] = np.where(
            data['away_sp_ip'] > 0,
            (data['away_sp_k'] * 9) / data['away_sp_ip'],
            8.0
        )
        
        # Venue effects (encoded)
        venue_run_effects = {
            'Coors Field': 1.2, 'Fenway Park': 1.1, 'Yankee Stadium': 1.1,
            'Minute Maid Park': 1.05, 'Citizens Bank Park': 1.05,
            'Petco Park': 0.9, 'Marlins Park': 0.95, 'Oakland Coliseum': 0.95,
            'Kauffman Stadium': 0.98, 'Busch Stadium': 1.0
        }
        features_df['venue_run_factor'] = data['venue_name'].map(venue_run_effects).fillna(1.0)
        
        # Combined offensive potential
        features_df['combined_offensive_power'] = (
            features_df['home_runs_pg'] + features_df['away_runs_pg'] +
            (features_df['home_woba'] + features_df['away_woba']) * 10
        ) / 3
        
        # Pitching matchup quality
        features_df['pitching_matchup_quality'] = (
            features_df['home_sp_era_game'] + features_df['away_sp_era_game']
        ) / 2
        
        self.feature_columns = list(features_df.columns)
        print(f"✅ Created {len(self.feature_columns)} features: {self.feature_columns}")
        
        return features_df
    
    def train_enhanced_model(self, data):
        """Train Random Forest model on recent data with all features"""
        print("🤖 Training enhanced ML model...")
        
        # Create features
        X = self.create_enhanced_features(data)
        y = data['total_runs']
        
        # Remove any rows with missing target
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 10:
            print("❌ Not enough data for training")
            return None
            
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train Random Forest
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"✅ Model trained successfully!")
        print(f"   Training MAE: {train_mae:.2f}")
        print(f"   Test MAE: {test_mae:.2f}")
        print(f"   Training R²: {train_r2:.3f}")
        print(f"   Test R²: {test_r2:.3f}")
        print(f"   CV MAE: {cv_mae:.2f} ± {cv_std:.2f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\\n📊 Top 10 Most Important Features:")
        for _, row in importance.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        return {
            'train_mae': train_mae, 'test_mae': test_mae,
            'train_r2': train_r2, 'test_r2': test_r2,
            'cv_mae': cv_mae, 'cv_std': cv_std,
            'feature_importance': importance,
            'training_data_size': len(X)
        }
    
    def validate_on_historical_games(self, data):
        """Validate predictions against actual outcomes with detailed tracking"""
        print("🎯 Validating predictions against actual outcomes...")
        
        validation_results = []
        
        for _, game in data.iterrows():
            # Create features for this game
            game_df = pd.DataFrame([game])
            X_game = self.create_enhanced_features(game_df)
            
            # Make prediction
            if self.model and len(X_game) > 0:
                predicted_total = self.model.predict(X_game)[0]
                actual_total = game['total_runs']
                prediction_error = abs(predicted_total - actual_total)
                
                # Get pitcher names
                home_pitcher = f"ID_{game['home_sp_id']}" if pd.notna(game['home_sp_id']) else "Unknown"
                away_pitcher = f"ID_{game['away_sp_id']}" if pd.notna(game['away_sp_id']) else "Unknown"
                
                result = {
                    'date': game['date'],
                    'game_id': game['game_id'],
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'home_pitcher': home_pitcher,
                    'away_pitcher': away_pitcher,
                    'actual_home_score': game['home_score'],
                    'actual_away_score': game['away_score'],
                    'actual_total': actual_total,
                    'predicted_total': round(predicted_total, 1),
                    'prediction_error': round(prediction_error, 1),
                    'percentage_error': round((prediction_error / actual_total) * 100, 1),
                    'venue': game['venue_name'],
                    'temperature': game['temperature'],
                    'weather': game['weather_condition']
                }
                
                validation_results.append(result)
        
        self.validation_results = validation_results
        
        if validation_results:
            errors = [r['prediction_error'] for r in validation_results]
            mae = np.mean(errors)
            max_error = np.max(errors)
            min_error = np.min(errors)
            
            print(f"✅ Validation complete on {len(validation_results)} games")
            print(f"   Average Error: {mae:.2f} runs")
            print(f"   Max Error: {max_error:.1f} runs")
            print(f"   Min Error: {min_error:.1f} runs")
            
            # Show worst and best predictions
            worst = max(validation_results, key=lambda x: x['prediction_error'])
            best = min(validation_results, key=lambda x: x['prediction_error'])
            
            print(f"\\n🔴 Worst Prediction:")
            print(f"   {worst['away_team']} @ {worst['home_team']} ({worst['date']})")
            print(f"   Predicted: {worst['predicted_total']} | Actual: {worst['actual_total']} | Error: {worst['prediction_error']}")
            
            print(f"\\n🟢 Best Prediction:")
            print(f"   {best['away_team']} @ {best['home_team']} ({best['date']})")
            print(f"   Predicted: {best['predicted_total']} | Actual: {best['actual_total']} | Error: {best['prediction_error']}")
        
        return validation_results
    
    def create_visualization_graphs(self):
        """Create comprehensive visualization graphs"""
        print("📈 Creating visualization graphs...")
        
        if not self.validation_results:
            print("❌ No validation results to visualize")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Enhanced MLB Prediction Model - Validation Results', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        predicted = [r['predicted_total'] for r in self.validation_results]
        actual = [r['actual_total'] for r in self.validation_results]
        errors = [r['prediction_error'] for r in self.validation_results]
        dates = [r['date'] for r in self.validation_results]
        
        # 1. Predicted vs Actual Scatter Plot
        axes[0,0].scatter(actual, predicted, alpha=0.7, s=60)
        axes[0,0].plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--', lw=2)
        axes[0,0].set_xlabel('Actual Total Runs')
        axes[0,0].set_ylabel('Predicted Total Runs')
        axes[0,0].set_title('Predicted vs Actual Total Runs')
        axes[0,0].grid(True, alpha=0.3)
        
        # Add R² score
        r2 = r2_score(actual, predicted)
        axes[0,0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0,0].transAxes, 
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 2. Prediction Error Distribution
        axes[0,1].hist(errors, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,1].set_xlabel('Prediction Error (runs)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Distribution of Prediction Errors')
        axes[0,1].axvline(np.mean(errors), color='red', linestyle='--', 
                         label=f'Mean Error: {np.mean(errors):.2f}')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Error Over Time
        sorted_results = sorted(self.validation_results, key=lambda x: x['date'])
        sorted_dates = [r['date'] for r in sorted_results]
        sorted_errors = [r['prediction_error'] for r in sorted_results]
        
        axes[1,0].plot(range(len(sorted_errors)), sorted_errors, 'o-', alpha=0.7)
        axes[1,0].set_xlabel('Game Sequence (by date)')
        axes[1,0].set_ylabel('Prediction Error (runs)')
        axes[1,0].set_title('Prediction Error Over Time')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Actual vs Predicted by Venue (top venues)
        venue_data = {}
        for r in self.validation_results:
            venue = r['venue']
            if venue not in venue_data:
                venue_data[venue] = {'actual': [], 'predicted': []}
            venue_data[venue]['actual'].append(r['actual_total'])
            venue_data[venue]['predicted'].append(r['predicted_total'])
        
        # Get top 5 venues by game count
        top_venues = sorted(venue_data.items(), key=lambda x: len(x[1]['actual']), reverse=True)[:5]
        
        venue_names = [v[0][:15] for v in top_venues]  # Truncate long names
        venue_avg_actual = [np.mean(v[1]['actual']) for v in top_venues]
        venue_avg_predicted = [np.mean(v[1]['predicted']) for v in top_venues]
        
        x = np.arange(len(venue_names))
        width = 0.35
        
        axes[1,1].bar(x - width/2, venue_avg_actual, width, label='Actual', alpha=0.8)
        axes[1,1].bar(x + width/2, venue_avg_predicted, width, label='Predicted', alpha=0.8)
        axes[1,1].set_xlabel('Venue')
        axes[1,1].set_ylabel('Average Total Runs')
        axes[1,1].set_title('Actual vs Predicted by Venue (Top 5)')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(venue_names, rotation=45, ha='right')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('enhanced_ml_validation_graphs.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualization graphs saved as 'enhanced_ml_validation_graphs.png'")
    
    def save_detailed_results(self):
        """Save detailed validation results to files"""
        print("💾 Saving detailed results...")
        
        if self.validation_results:
            # Save validation results as CSV
            results_df = pd.DataFrame(self.validation_results)
            results_df.to_csv('enhanced_validation_results.csv', index=False)
            print("✅ Validation results saved to 'enhanced_validation_results.csv'")
            
            # Save summary statistics
            summary = {
                'total_games_validated': len(self.validation_results),
                'average_prediction_error': float(np.mean([r['prediction_error'] for r in self.validation_results])),
                'max_prediction_error': float(np.max([r['prediction_error'] for r in self.validation_results])),
                'min_prediction_error': float(np.min([r['prediction_error'] for r in self.validation_results])),
                'r_squared': float(r2_score([r['actual_total'] for r in self.validation_results], 
                                          [r['predicted_total'] for r in self.validation_results])),
                'validation_date': datetime.now().isoformat()
            }
            
            with open('enhanced_validation_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            print("✅ Summary statistics saved to 'enhanced_validation_summary.json'")
        
        # Save model if trained
        if self.model:
            joblib.dump(self.model, 'enhanced_mlb_model_v2.joblib')
            print("✅ Enhanced model saved to 'enhanced_mlb_model_v2.joblib'")
    
    def run_complete_enhanced_training(self):
        """Run the complete enhanced training and validation process"""
        print("🚀 Starting Enhanced ML Training and Validation")
        print("=" * 60)
        
        # Step 1: Collect last 20+ games data
        recent_data = self.collect_last_20_games_data()
        if recent_data.empty:
            print("❌ No data available for training")
            return
        
        # Step 2: Train enhanced model
        training_results = self.train_enhanced_model(recent_data)
        if not training_results:
            print("❌ Model training failed")
            return
        
        # Step 3: Validate on historical games
        validation_results = self.validate_on_historical_games(recent_data)
        
        # Step 4: Create visualization graphs
        self.create_visualization_graphs()
        
        # Step 5: Save detailed results
        self.save_detailed_results()
        
        print("\\n🎉 Enhanced training and validation complete!")
        print(f"📊 Model trained on {training_results['training_data_size']} games")
        print(f"🎯 Validated on {len(validation_results)} games")
        print(f"📈 Average prediction error: {np.mean([r['prediction_error'] for r in validation_results]):.2f} runs")
        print("📁 Results saved to files and graphs generated")

def main():
    """Run the enhanced ML training system"""
    trainer = EnhancedMLTrainerWithGraphs()
    trainer.run_complete_enhanced_training()

if __name__ == "__main__":
    main()
