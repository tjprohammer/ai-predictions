#!/usr/bin/env python3
"""
📊 Learning Impact Monitor
Tracks whether 20-session learnings are improving daily predictions.
"""

import pandas as pd
import numpy as np
import json
import sqlite3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class LearningImpactMonitor:
    """Monitors the impact of enhanced learning on prediction accuracy"""
    
    def __init__(self, db_path='S:/Projects/AI_Predictions/mlb-overs/data/mlb_data.db'):
        self.db_path = db_path
        self.baseline_metrics = self._load_baseline_metrics()
        self.performance_history = []
        
    def _load_baseline_metrics(self):
        """Load baseline performance metrics from before enhanced learning"""
        return {
            'baseline_mae': 1.2,    # Typical baseline before enhancements
            'baseline_r2': 0.75,    # Typical baseline R²
            'target_mae': 0.898,    # Best session target
            'target_r2': 0.911      # Best session target
        }
    
    def load_recent_predictions(self, days_back=30):
        """Load recent predictions to analyze performance"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
            SELECT 
                game_date,
                home_team,
                away_team,
                total_runs as actual_total,
                predicted_total,
                ABS(total_runs - predicted_total) as absolute_error
            FROM enhanced_game_data
            WHERE game_date >= ? AND game_date <= ?
            AND total_runs IS NOT NULL 
            AND predicted_total IS NOT NULL
            ORDER BY game_date DESC
            """
            
            df = pd.read_sql_query(
                query, conn, 
                params=[start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')]
            )
            conn.close()
            
            print(f"📊 Loaded {len(df)} recent predictions for analysis")
            return df
            
        except Exception as e:
            print(f"❌ Error loading predictions: {e}")
            return pd.DataFrame()
    
    def analyze_daily_performance(self, df):
        """Analyze daily prediction performance"""
        if df.empty:
            return {}
        
        # Calculate daily metrics
        daily_metrics = df.groupby('game_date').agg({
            'absolute_error': ['mean', 'count'],
            'actual_total': ['mean'],
            'predicted_total': ['mean']
        }).round(3)
        
        daily_metrics.columns = ['daily_mae', 'games_count', 'avg_actual', 'avg_predicted']
        daily_metrics = daily_metrics.reset_index()
        
        # Calculate R² for recent period
        overall_r2 = r2_score(df['actual_total'], df['predicted_total'])
        overall_mae = mean_absolute_error(df['actual_total'], df['predicted_total'])
        
        # Compare to baselines
        baseline_mae = self.baseline_metrics['baseline_mae']
        target_mae = self.baseline_metrics['target_mae']
        
        improvement_from_baseline = ((baseline_mae - overall_mae) / baseline_mae) * 100
        distance_to_target = overall_mae - target_mae
        
        return {
            'daily_metrics': daily_metrics,
            'overall_mae': overall_mae,
            'overall_r2': overall_r2,
            'improvement_from_baseline': improvement_from_baseline,
            'distance_to_target': distance_to_target,
            'games_analyzed': len(df)
        }
    
    def track_learning_trend(self, df, window_days=7):
        """Track if learning is showing improvement trends"""
        if len(df) < window_days:
            return {}
        
        df = df.sort_values('game_date')
        df['rolling_mae'] = df['absolute_error'].rolling(window=window_days).mean()
        df['game_number'] = range(len(df))
        
        # Calculate trend slope
        valid_data = df.dropna(subset=['rolling_mae'])
        if len(valid_data) > 10:
            # Simple linear trend
            x = valid_data['game_number'].values
            y = valid_data['rolling_mae'].values
            trend_slope = np.polyfit(x, y, 1)[0]
            
            # Trend interpretation
            if trend_slope < -0.01:
                trend_direction = "📈 IMPROVING"
            elif trend_slope > 0.01:
                trend_direction = "📉 DECLINING"
            else:
                trend_direction = "➡️  STABLE"
        else:
            trend_slope = 0
            trend_direction = "❓ INSUFFICIENT_DATA"
        
        return {
            'trend_slope': trend_slope,
            'trend_direction': trend_direction,
            'rolling_mae_data': df[['game_date', 'rolling_mae']].dropna()
        }
    
    def generate_performance_report(self, days_back=30):
        """Generate comprehensive performance report"""
        print(f"📊 LEARNING IMPACT ANALYSIS - Last {days_back} Days")
        print("="*60)
        
        # Load recent data
        df = self.load_recent_predictions(days_back)
        if df.empty:
            print("❌ No recent prediction data available")
            return {}
        
        # Analyze performance
        performance = self.analyze_daily_performance(df)
        trend_analysis = self.track_learning_trend(df)
        
        # Display results
        print(f"\n📈 OVERALL PERFORMANCE:")
        print(f"   Games Analyzed: {performance['games_analyzed']}")
        print(f"   Current MAE: {performance['overall_mae']:.3f} runs")
        print(f"   Current R²: {performance['overall_r2']:.3f}")
        
        print(f"\n🎯 COMPARISON TO TARGETS:")
        print(f"   Baseline MAE: {self.baseline_metrics['baseline_mae']:.3f}")
        print(f"   Target MAE: {self.baseline_metrics['target_mae']:.3f}")
        print(f"   Improvement from Baseline: {performance['improvement_from_baseline']:+.1f}%")
        print(f"   Distance to Target: {performance['distance_to_target']:+.3f} runs")
        
        print(f"\n📊 LEARNING TREND:")
        print(f"   Direction: {trend_analysis['trend_direction']}")
        print(f"   Slope: {trend_analysis['trend_slope']:.4f} runs/game")
        
        # Performance assessment
        if performance['improvement_from_baseline'] > 10:
            status = "🎉 EXCELLENT - Learnings are significantly improving predictions!"
        elif performance['improvement_from_baseline'] > 5:
            status = "✅ GOOD - Learnings show positive impact"
        elif performance['improvement_from_baseline'] > 0:
            status = "⚠️  MARGINAL - Small improvement, monitor closely"
        else:
            status = "❌ POOR - Performance below baseline, need investigation"
        
        print(f"\n🏆 ASSESSMENT: {status}")
        
        # Save performance record
        self.performance_history.append({
            'date': datetime.now().isoformat(),
            'days_analyzed': days_back,
            'mae': performance['overall_mae'],
            'r2': performance['overall_r2'],
            'improvement_pct': performance['improvement_from_baseline'],
            'trend_slope': trend_analysis['trend_slope']
        })
        
        return {
            'performance': performance,
            'trend': trend_analysis,
            'status': status
        }
    
    def create_performance_dashboard(self, df, save_path='learning_performance_dashboard.png'):
        """Create visual dashboard of learning performance"""
        if df.empty:
            print("❌ No data for dashboard")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('🎯 Enhanced Learning Performance Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Daily MAE trend
        daily_metrics = df.groupby('game_date')['absolute_error'].mean().reset_index()
        axes[0,0].plot(pd.to_datetime(daily_metrics['game_date']), 
                      daily_metrics['absolute_error'], 'b-', linewidth=2)
        axes[0,0].axhline(y=self.baseline_metrics['target_mae'], color='g', 
                         linestyle='--', label=f"Target MAE ({self.baseline_metrics['target_mae']:.3f})")
        axes[0,0].axhline(y=self.baseline_metrics['baseline_mae'], color='r', 
                         linestyle='--', label=f"Baseline MAE ({self.baseline_metrics['baseline_mae']:.3f})")
        axes[0,0].set_title('Daily Mean Absolute Error')
        axes[0,0].set_ylabel('MAE (runs)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Prediction vs Actual scatter
        axes[0,1].scatter(df['actual_total'], df['predicted_total'], alpha=0.6)
        axes[0,1].plot([df['actual_total'].min(), df['actual_total'].max()], 
                      [df['actual_total'].min(), df['actual_total'].max()], 'r--', linewidth=2)
        axes[0,1].set_title('Predicted vs Actual Total Runs')
        axes[0,1].set_xlabel('Actual Total Runs')
        axes[0,1].set_ylabel('Predicted Total Runs')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Error distribution
        axes[1,0].hist(df['absolute_error'], bins=20, alpha=0.7, edgecolor='black')
        axes[1,0].axvline(x=df['absolute_error'].mean(), color='r', linestyle='--', 
                         label=f"Mean Error: {df['absolute_error'].mean():.3f}")
        axes[1,0].set_title('Error Distribution')
        axes[1,0].set_xlabel('Absolute Error (runs)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Rolling performance
        df_sorted = df.sort_values('game_date')
        df_sorted['rolling_mae'] = df_sorted['absolute_error'].rolling(window=7).mean()
        axes[1,1].plot(pd.to_datetime(df_sorted['game_date']), 
                      df_sorted['rolling_mae'], 'g-', linewidth=2)
        axes[1,1].set_title('7-Day Rolling MAE')
        axes[1,1].set_ylabel('Rolling MAE (runs)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Dashboard saved: {save_path}")
        
        return save_path
    
    def identify_improvement_opportunities(self, df):
        """Identify specific areas where learning can be improved"""
        if df.empty:
            return []
        
        opportunities = []
        
        # High error games analysis
        high_error_threshold = df['absolute_error'].quantile(0.8)
        high_error_games = df[df['absolute_error'] > high_error_threshold]
        
        if len(high_error_games) > 0:
            avg_high_error = high_error_games['absolute_error'].mean()
            opportunities.append({
                'type': 'High Error Games',
                'description': f"{len(high_error_games)} games with MAE > {high_error_threshold:.2f}",
                'impact': f"Average error: {avg_high_error:.3f} runs",
                'recommendation': "Analyze these games for pattern recognition improvements"
            })
        
        # Prediction bias analysis
        prediction_bias = df['predicted_total'].mean() - df['actual_total'].mean()
        if abs(prediction_bias) > 0.1:
            bias_direction = "over-predicting" if prediction_bias > 0 else "under-predicting"
            opportunities.append({
                'type': 'Prediction Bias',
                'description': f"Model is {bias_direction} by {abs(prediction_bias):.3f} runs",
                'impact': f"Systematic bias affecting all predictions",
                'recommendation': "Adjust model calibration or feature weights"
            })
        
        # Team-specific analysis
        team_performance = df.groupby(['home_team', 'away_team'])['absolute_error'].mean()
        worst_matchups = team_performance.nlargest(5)
        
        if len(worst_matchups) > 0:
            opportunities.append({
                'type': 'Team Matchup Issues',
                'description': f"Worst prediction accuracy for specific team matchups",
                'impact': f"Top error: {worst_matchups.iloc[0]:.3f} runs",
                'recommendation': "Enhance team-specific or matchup-specific features"
            })
        
        return opportunities
    
    def export_performance_log(self, filename='learning_performance_log.json'):
        """Export performance history for tracking"""
        log_data = {
            'baseline_metrics': self.baseline_metrics,
            'performance_history': self.performance_history,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"📁 Performance log exported: {filename}")
        return filename

def main():
    """Run learning impact monitoring"""
    print("📊 LEARNING IMPACT MONITORING SYSTEM")
    print("="*50)
    
    monitor = LearningImpactMonitor()
    
    # Generate performance report
    report = monitor.generate_performance_report(days_back=30)
    
    if report:
        # Load data for visualization
        df = monitor.load_recent_predictions(30)
        
        if not df.empty:
            # Create dashboard
            monitor.create_performance_dashboard(df)
            
            # Identify opportunities
            opportunities = monitor.identify_improvement_opportunities(df)
            
            if opportunities:
                print(f"\n🎯 IMPROVEMENT OPPORTUNITIES:")
                for i, opp in enumerate(opportunities, 1):
                    print(f"   {i}. {opp['type']}: {opp['description']}")
                    print(f"      Impact: {opp['impact']}")
                    print(f"      Recommendation: {opp['recommendation']}")
                    print()
            
            # Export log
            monitor.export_performance_log()
        
        print(f"\n✅ Monitoring complete! Check dashboard and logs for details.")

if __name__ == "__main__":
    main()
