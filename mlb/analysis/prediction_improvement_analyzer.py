#!/usr/bin/env python3
"""
MLB Prediction System - Comprehensive Improvement Plan
Based on current 48.8% accuracy, targeting 55%+ performance
"""

import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionImprovementAnalyzer:
    """Analyze current system performance and suggest improvements"""
    
    def __init__(self):
        self.engine = create_engine("postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
        
    def analyze_current_features(self):
        """Analyze which features are actually contributing to predictions"""
        print("🔍 CURRENT FEATURE ANALYSIS")
        print("=" * 50)
        
        # Get recent predictions with features
        query = text("""
        SELECT 
            date,
            game_id,
            home_team,
            away_team,
            predicted_total,
            predicted_total_learning,
            total_runs,
            market_total,
            confidence,
            temperature,
            wind_speed,
            home_sp_season_era,
            away_sp_season_era,
            ballpark_run_factor,
            ballpark_hr_factor
        FROM enhanced_games 
        WHERE date >= :thirty_days_ago 
            AND total_runs IS NOT NULL 
            AND (predicted_total IS NOT NULL OR predicted_total_learning IS NOT NULL)
        ORDER BY date DESC 
        LIMIT 200
        """)
        
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        df = pd.read_sql(query, self.engine, params={'thirty_days_ago': thirty_days_ago})
        
        if df.empty:
            print("❌ No recent games found for analysis")
            return
            
        print(f"📊 Analyzing {len(df)} recent games")
        
        # Use learning model predictions if available, otherwise regular
        df['prediction'] = df['predicted_total_learning'].fillna(df['predicted_total'])
        df['error'] = abs(df['prediction'] - df['total_runs'])
        df['bias'] = df['prediction'] - df['total_runs']
        
        # Basic accuracy stats
        mae = df['error'].mean()
        bias = df['bias'].mean()
        accuracy_1 = (df['error'] <= 1.0).mean()
        accuracy_2 = (df['error'] <= 2.0).mean()
        
        print(f"   Current MAE: {mae:.2f}")
        print(f"   Current Bias: {bias:+.2f}")
        print(f"   Within 1 run: {accuracy_1:.1%}")
        print(f"   Within 2 runs: {accuracy_2:.1%}")
        
        # Feature correlation analysis
        numeric_cols = ['temperature', 'wind_speed', 'home_sp_season_era', 'away_sp_season_era', 
                       'ballpark_run_factor', 'ballpark_hr_factor', 'market_total']
        
        feature_data = df[numeric_cols + ['total_runs']].dropna()
        if not feature_data.empty:
            correlations = feature_data.corr()['total_runs'].drop('total_runs').abs().sort_values(ascending=False)
            
            print(f"\n🔗 Feature Correlations with Actual Runs:")
            for feature, corr in correlations.items():
                status = "✅ Strong" if corr > 0.3 else "⚠️ Weak" if corr > 0.1 else "❌ Minimal"
                print(f"   {feature:<25}: {corr:.3f} {status}")
        
        # Prediction quality by conditions
        self._analyze_by_conditions(df)
        
        return df
        
    def _analyze_by_conditions(self, df):
        """Analyze prediction accuracy under different conditions"""
        print(f"\n🌡️ PREDICTION ACCURACY BY CONDITIONS")
        print("-" * 40)
        
        # Temperature bins
        df['temp_bin'] = pd.cut(df['temperature'], bins=[0, 60, 75, 90, 120], 
                               labels=['Cold (<60°)', 'Cool (60-75°)', 'Warm (75-90°)', 'Hot (>90°)'])
        temp_analysis = df.groupby('temp_bin')['error'].agg(['count', 'mean']).round(2)
        print("Temperature Analysis:")
        for temp, row in temp_analysis.iterrows():
            print(f"   {temp:<15}: {row['count']} games, {row['mean']:.2f} MAE")
            
        # Market total bins (game expectations)
        df['total_bin'] = pd.cut(df['market_total'], bins=[0, 8, 9.5, 11, 15], 
                                labels=['Low (<8)', 'Medium (8-9.5)', 'High (9.5-11)', 'Very High (>11)'])
        total_analysis = df.groupby('total_bin')['error'].agg(['count', 'mean']).round(2)
        print("\nMarket Total Analysis:")
        for total, row in total_analysis.iterrows():
            print(f"   {total:<20}: {row['count']} games, {row['mean']:.2f} MAE")
    
    def identify_improvement_opportunities(self):
        """Identify specific areas for improvement"""
        print(f"\n🎯 IMPROVEMENT OPPORTUNITIES")
        print("=" * 50)
        
        improvements = {
            "Feature Engineering": [
                "Add pitcher vs team historical matchups",
                "Include bullpen usage/fatigue metrics",
                "Add ballpark wind direction effects",
                "Include team recent form (L10 games)",
                "Add umpire strike zone tendencies"
            ],
            "Data Quality": [
                "Verify weather data accuracy",
                "Add missing pitcher advanced stats", 
                "Include real-time lineup changes",
                "Add park-specific factors",
                "Improve starting pitcher projections"
            ],
            "Model Architecture": [
                "Try ensemble methods (XGBoost + RandomForest)",
                "Add feature interactions",
                "Implement position-specific models",
                "Use recency weighting",
                "Add confidence intervals"
            ],
            "Prediction Logic": [
                "Separate models for different game types",
                "Add market adjustment factors",
                "Include Vegas line movement",
                "Weight predictions by confidence",
                "Add bias correction mechanisms"
            ]
        }
        
        for category, items in improvements.items():
            print(f"\n📋 {category}:")
            for i, item in enumerate(items, 1):
                print(f"   {i}. {item}")
                
    def create_feature_improvement_plan(self):
        """Create specific feature improvements to implement"""
        print(f"\n🚀 IMMEDIATE FEATURE IMPROVEMENTS")
        print("=" * 50)
        
        plan = {
            "Week 1 - Basic Enhancements": [
                "Add pitcher L5 game ERA and WHIP",
                "Include team offensive rating last 15 games",
                "Add ballpark temperature interaction terms",
                "Include day vs night game effects"
            ],
            "Week 2 - Matchup Features": [
                "Add pitcher vs opposing team history",
                "Include bullpen usage last 3 games",
                "Add team vs LHP/RHP splits",
                "Include home/away performance gaps"
            ],
            "Week 3 - Advanced Features": [
                "Add umpire run environment history",
                "Include weather trend effects",
                "Add lineup protection metrics",
                "Include park-specific wind effects"
            ],
            "Week 4 - Model Optimization": [
                "Feature selection via importance",
                "Add feature interactions",
                "Implement ensemble methods",
                "Add prediction confidence scoring"
            ]
        }
        
        for week, tasks in plan.items():
            print(f"\n📅 {week}:")
            for i, task in enumerate(tasks, 1):
                print(f"   {i}. {task}")
                
        return plan
        
    def track_prediction_performance(self):
        """Enhanced prediction tracking system"""
        print(f"\n📊 ENHANCED PREDICTION TRACKING")
        print("=" * 50)
        
        # Get predictions for last 7 days
        query = text("""
        SELECT 
            date,
            COUNT(*) as total_games,
            COUNT(CASE WHEN total_runs IS NOT NULL THEN 1 END) as completed_games,
            AVG(CASE WHEN total_runs IS NOT NULL 
                THEN ABS(COALESCE(predicted_total_learning, predicted_total) - total_runs) END) as mae,
            AVG(CASE WHEN total_runs IS NOT NULL 
                THEN COALESCE(predicted_total_learning, predicted_total) - total_runs END) as bias,
            COUNT(CASE WHEN total_runs IS NOT NULL 
                AND ABS(COALESCE(predicted_total_learning, predicted_total) - total_runs) <= 1 
                THEN 1 END) as within_1_run
        FROM enhanced_games 
        WHERE date >= :seven_days_ago 
        GROUP BY date 
        ORDER BY date DESC
        """)
        
        seven_days_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        df = pd.read_sql(query, self.engine, params={'seven_days_ago': seven_days_ago})
        
        if df.empty:
            print("❌ No recent tracking data available")
            return
            
        print("📈 Daily Performance Tracking:")
        print(f"{'Date':<12} {'Games':<6} {'Completed':<10} {'MAE':<6} {'Bias':<7} {'Within 1':<9}")
        print("-" * 60)
        
        for _, row in df.iterrows():
            completed = int(row['completed_games']) if row['completed_games'] else 0
            mae = f"{row['mae']:.2f}" if row['mae'] else "N/A"
            bias = f"{row['bias']:+.2f}" if row['bias'] else "N/A"
            within_1 = f"{row['within_1_run'] or 0}/{completed}" if completed > 0 else "N/A"
            
            print(f"{row['date']:<12} {int(row['total_games']):<6} {completed:<10} {mae:<6} {bias:<7} {within_1:<9}")
            
    def generate_improvement_report(self):
        """Generate comprehensive improvement report"""
        print(f"\n📋 MLB PREDICTION IMPROVEMENT REPORT")
        print("=" * 60)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current Status: 48.8% accuracy (needs significant improvement)")
        print("=" * 60)
        
        # Run all analyses
        df = self.analyze_current_features()
        self.identify_improvement_opportunities()
        plan = self.create_feature_improvement_plan()
        self.track_prediction_performance()
        
        print(f"\n🎯 PRIORITY RECOMMENDATIONS:")
        print("1. Focus on pitcher recent form (L5 games)")
        print("2. Add team matchup history features")
        print("3. Improve ballpark environmental factors")
        print("4. Implement ensemble modeling approach")
        print("5. Add real-time confidence scoring")
        
        print(f"\n📊 SUCCESS METRICS TO TRACK:")
        print("- Target: 55%+ accuracy within 30 days")
        print("- MAE improvement: Current ~3.2 → Target <2.5")
        print("- Within 1 run: Current ~40% → Target >60%")
        print("- Betting accuracy: Target 52-55% on O/U")
        
        return plan

def main():
    """Run comprehensive improvement analysis"""
    analyzer = PredictionImprovementAnalyzer()
    
    try:
        plan = analyzer.generate_improvement_report()
        
        # Save plan to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f'improvement_plan_{timestamp}.txt', 'w') as f:
            f.write(f"MLB Prediction Improvement Plan - {datetime.now()}\n")
            f.write("="*60 + "\n\n")
            
            for week, tasks in plan.items():
                f.write(f"{week}:\n")
                for i, task in enumerate(tasks, 1):
                    f.write(f"  {i}. {task}\n")
                f.write("\n")
                
        print(f"\n💾 Improvement plan saved to: improvement_plan_{timestamp}.txt")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
