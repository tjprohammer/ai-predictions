#!/usr/bin/env python3
"""
Daily Performance Tracking System
Monitor model performance and trigger retraining when needed
"""

import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DailyPerformanceTracker:
    """Track and monitor daily prediction performance"""
    
    def __init__(self):
        self.engine = create_engine("postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
        self.tracking_dir = Path("mlb/tracking/daily_performance")
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance thresholds for alerts
        self.thresholds = {
            'mae_warning': 3.5,
            'mae_critical': 4.0,
            'accuracy_warning': 0.35,  # 35% within 1 run
            'accuracy_critical': 0.25,  # 25% within 1 run
            'bias_warning': 0.75,
            'bias_critical': 1.0
        }
        
    def calculate_daily_metrics(self, date_str):
        """Calculate performance metrics for a specific date"""
        logger.info(f"📊 Calculating metrics for {date_str}")
        
        query = text("""
        SELECT 
            game_id,
            home_team,
            away_team,
            predicted_total,
            predicted_total_learning,
            total_runs,
            market_total,
            confidence,
            temperature,
            venue_name,
            COALESCE(predicted_total_learning, predicted_total) as primary_prediction
        FROM enhanced_games 
        WHERE date = :date_str
            AND total_runs IS NOT NULL 
            AND (predicted_total IS NOT NULL OR predicted_total_learning IS NOT NULL)
        """)
        
        df = pd.read_sql(query, self.engine, params={'date_str': date_str})
        
        if df.empty:
            logger.warning(f"No completed games found for {date_str}")
            return None
            
        # Calculate metrics
        predictions = df['primary_prediction'].values
        actual = df['total_runs'].values
        
        metrics = {
            'date': date_str,
            'games_analyzed': len(df),
            'mae': float(np.mean(np.abs(predictions - actual))),
            'rmse': float(np.sqrt(np.mean((predictions - actual) ** 2))),
            'bias': float(np.mean(predictions - actual)),
            'accuracy_within_1': float(np.mean(np.abs(predictions - actual) <= 1.0)),
            'accuracy_within_2': float(np.mean(np.abs(predictions - actual) <= 2.0)),
            'accuracy_within_3': float(np.mean(np.abs(predictions - actual) <= 3.0)),
            'r_squared': float(np.corrcoef(predictions, actual)[0, 1] ** 2) if len(predictions) > 1 else 0.0
        }
        
        # Model comparison if both predictions available
        learning_available = df['predicted_total_learning'].notna().sum()
        original_available = df['predicted_total'].notna().sum()
        
        if learning_available > 0 and original_available > 0:
            learning_mae = np.mean(np.abs(df['predicted_total_learning'].dropna() - 
                                        df.loc[df['predicted_total_learning'].notna(), 'total_runs']))
            original_mae = np.mean(np.abs(df['predicted_total'].dropna() - 
                                        df.loc[df['predicted_total'].notna(), 'total_runs']))
            
            metrics['learning_model_mae'] = float(learning_mae)
            metrics['original_model_mae'] = float(original_mae)
            metrics['learning_advantage'] = float(original_mae - learning_mae)
            
        # Market comparison
        if 'market_total' in df.columns and df['market_total'].notna().sum() > 0:
            market_data = df[df['market_total'].notna()]
            market_mae = np.mean(np.abs(market_data['market_total'] - market_data['total_runs']))
            metrics['market_mae'] = float(market_mae)
            metrics['model_vs_market'] = float(market_mae - metrics['mae'])
            
        # Performance by conditions
        metrics['performance_by_temp'] = self._analyze_by_temperature(df)
        metrics['performance_by_total'] = self._analyze_by_total_range(df)
        
        return metrics
        
    def _analyze_by_temperature(self, df):
        """Analyze performance by temperature ranges"""
        df['temp_bin'] = pd.cut(df['temperature'], 
                               bins=[0, 60, 75, 90, 120], 
                               labels=['Cold', 'Cool', 'Warm', 'Hot'])
        
        temp_analysis = {}
        for temp_range in df['temp_bin'].unique():
            if pd.notna(temp_range):
                temp_games = df[df['temp_bin'] == temp_range]
                if len(temp_games) > 0:
                    predictions = temp_games['primary_prediction'].values
                    actual = temp_games['total_runs'].values
                    temp_analysis[str(temp_range)] = {
                        'games': len(temp_games),
                        'mae': float(np.mean(np.abs(predictions - actual))),
                        'bias': float(np.mean(predictions - actual))
                    }
                    
        return temp_analysis
        
    def _analyze_by_total_range(self, df):
        """Analyze performance by expected total ranges"""
        df['total_bin'] = pd.cut(df['market_total'], 
                                bins=[0, 8, 9.5, 11, 15], 
                                labels=['Low', 'Medium', 'High', 'Very High'])
        
        total_analysis = {}
        for total_range in df['total_bin'].unique():
            if pd.notna(total_range):
                total_games = df[df['total_bin'] == total_range]
                if len(total_games) > 0:
                    predictions = total_games['primary_prediction'].values
                    actual = total_games['total_runs'].values
                    total_analysis[str(total_range)] = {
                        'games': len(total_games),
                        'mae': float(np.mean(np.abs(predictions - actual))),
                        'bias': float(np.mean(predictions - actual))
                    }
                    
        return total_analysis
        
    def check_performance_alerts(self, metrics):
        """Check if performance metrics trigger alerts"""
        alerts = []
        
        # MAE alerts
        if metrics['mae'] > self.thresholds['mae_critical']:
            alerts.append({
                'type': 'CRITICAL',
                'metric': 'MAE',
                'value': metrics['mae'],
                'threshold': self.thresholds['mae_critical'],
                'message': f"MAE ({metrics['mae']:.3f}) exceeds critical threshold ({self.thresholds['mae_critical']})"
            })
        elif metrics['mae'] > self.thresholds['mae_warning']:
            alerts.append({
                'type': 'WARNING',
                'metric': 'MAE',
                'value': metrics['mae'],
                'threshold': self.thresholds['mae_warning'],
                'message': f"MAE ({metrics['mae']:.3f}) exceeds warning threshold ({self.thresholds['mae_warning']})"
            })
            
        # Accuracy alerts
        if metrics['accuracy_within_1'] < self.thresholds['accuracy_critical']:
            alerts.append({
                'type': 'CRITICAL',
                'metric': 'Accuracy',
                'value': metrics['accuracy_within_1'],
                'threshold': self.thresholds['accuracy_critical'],
                'message': f"Accuracy within 1 run ({metrics['accuracy_within_1']:.1%}) below critical threshold ({self.thresholds['accuracy_critical']:.1%})"
            })
        elif metrics['accuracy_within_1'] < self.thresholds['accuracy_warning']:
            alerts.append({
                'type': 'WARNING',
                'metric': 'Accuracy',
                'value': metrics['accuracy_within_1'],
                'threshold': self.thresholds['accuracy_warning'],
                'message': f"Accuracy within 1 run ({metrics['accuracy_within_1']:.1%}) below warning threshold ({self.thresholds['accuracy_warning']:.1%})"
            })
            
        # Bias alerts
        abs_bias = abs(metrics['bias'])
        if abs_bias > self.thresholds['bias_critical']:
            direction = "over-predicting" if metrics['bias'] > 0 else "under-predicting"
            alerts.append({
                'type': 'CRITICAL',
                'metric': 'Bias',
                'value': metrics['bias'],
                'threshold': self.thresholds['bias_critical'],
                'message': f"Model is {direction} by {abs_bias:.3f} runs (critical threshold: {self.thresholds['bias_critical']})"
            })
        elif abs_bias > self.thresholds['bias_warning']:
            direction = "over-predicting" if metrics['bias'] > 0 else "under-predicting"
            alerts.append({
                'type': 'WARNING',
                'metric': 'Bias',
                'value': metrics['bias'],
                'threshold': self.thresholds['bias_warning'],
                'message': f"Model is {direction} by {abs_bias:.3f} runs (warning threshold: {self.thresholds['bias_warning']})"
            })
            
        return alerts
        
    def save_daily_metrics(self, metrics):
        """Save daily metrics to file"""
        if not metrics:
            return
            
        date_str = metrics['date']
        filepath = self.tracking_dir / f"metrics_{date_str.replace('-', '')}.json"
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"💾 Metrics saved to {filepath}")
        
    def generate_weekly_report(self):
        """Generate weekly performance report"""
        logger.info("📋 Generating weekly performance report...")
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=7)
        
        # Collect metrics for each day
        daily_metrics = []
        for i in range(7):
            date = start_date + timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            
            metrics = self.calculate_daily_metrics(date_str)
            if metrics:
                daily_metrics.append(metrics)
                self.save_daily_metrics(metrics)
                
                # Check for alerts
                alerts = self.check_performance_alerts(metrics)
                if alerts:
                    logger.warning(f"⚠️ Performance alerts for {date_str}:")
                    for alert in alerts:
                        logger.warning(f"  {alert['type']}: {alert['message']}")
                        
        if not daily_metrics:
            logger.warning("No metrics available for weekly report")
            return
            
        # Calculate weekly aggregates
        weekly_metrics = self._calculate_weekly_aggregates(daily_metrics)
        
        # Generate report
        report = self._generate_report_text(weekly_metrics, daily_metrics)
        
        # Save report
        report_path = self.tracking_dir / f"weekly_report_{end_date.strftime('%Y%m%d')}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
            
        logger.info(f"📊 Weekly report saved to {report_path}")
        print(report)
        
        return weekly_metrics
        
    def _calculate_weekly_aggregates(self, daily_metrics):
        """Calculate weekly aggregate metrics"""
        total_games = sum(m['games_analyzed'] for m in daily_metrics)
        
        # Weight metrics by number of games
        weighted_mae = sum(m['mae'] * m['games_analyzed'] for m in daily_metrics) / total_games
        weighted_bias = sum(m['bias'] * m['games_analyzed'] for m in daily_metrics) / total_games
        weighted_accuracy = sum(m['accuracy_within_1'] * m['games_analyzed'] for m in daily_metrics) / total_games
        
        return {
            'period': f"{daily_metrics[0]['date']} to {daily_metrics[-1]['date']}",
            'total_games': total_games,
            'avg_mae': weighted_mae,
            'avg_bias': weighted_bias,
            'avg_accuracy_within_1': weighted_accuracy,
            'days_analyzed': len(daily_metrics)
        }
        
    def _generate_report_text(self, weekly_metrics, daily_metrics):
        """Generate formatted report text"""
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          MLB PREDICTION WEEKLY REPORT                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Period: {weekly_metrics['period']:<62} ║
║ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<58} ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 WEEKLY PERFORMANCE SUMMARY
════════════════════════════════════════════════════════════════════════════════
Total Games Analyzed: {weekly_metrics['total_games']}
Days with Data: {weekly_metrics['days_analyzed']}/7

Key Metrics:
├── Mean Absolute Error: {weekly_metrics['avg_mae']:.3f} runs
├── Average Bias: {weekly_metrics['avg_bias']:+.3f} runs  
└── Accuracy (within 1 run): {weekly_metrics['avg_accuracy_within_1']:.1%}

Performance Assessment:
"""
        
        # Performance assessment
        if weekly_metrics['avg_mae'] < 2.5:
            report += "✅ EXCELLENT - MAE under target threshold\n"
        elif weekly_metrics['avg_mae'] < 3.0:
            report += "✅ GOOD - MAE approaching target\n" 
        elif weekly_metrics['avg_mae'] < 3.5:
            report += "⚠️ FAIR - MAE needs improvement\n"
        else:
            report += "❌ POOR - MAE significantly above target\n"
            
        if weekly_metrics['avg_accuracy_within_1'] > 0.5:
            report += "✅ HIGH PRECISION - Good accuracy within 1 run\n"
        elif weekly_metrics['avg_accuracy_within_1'] > 0.35:
            report += "⚠️ MODERATE PRECISION - Accuracy could improve\n"
        else:
            report += "❌ LOW PRECISION - Accuracy needs significant work\n"
            
        if abs(weekly_metrics['avg_bias']) < 0.5:
            report += "✅ WELL CALIBRATED - Low bias\n"
        elif abs(weekly_metrics['avg_bias']) < 1.0:
            report += "⚠️ SLIGHT BIAS - Monitor for systematic error\n"
        else:
            direction = "over-predicting" if weekly_metrics['avg_bias'] > 0 else "under-predicting"
            report += f"❌ SIGNIFICANT BIAS - Systematically {direction}\n"
            
        report += f"""
📈 DAILY BREAKDOWN
════════════════════════════════════════════════════════════════════════════════
{'Date':<12} {'Games':<6} {'MAE':<6} {'Bias':<7} {'Acc1':<6}
"""
        report += "─" * 50 + "\n"
        
        for metrics in daily_metrics:
            report += f"{metrics['date']:<12} {metrics['games_analyzed']:<6} "
            report += f"{metrics['mae']:<6.3f} {metrics['bias']:<+7.3f} "
            report += f"{metrics['accuracy_within_1']:<6.1%}\n"
            
        report += f"""
🎯 RECOMMENDATIONS
════════════════════════════════════════════════════════════════════════════════
"""
        
        if weekly_metrics['avg_mae'] > 3.5:
            report += "• URGENT: Retrain models - MAE significantly above target\n"
        elif weekly_metrics['avg_mae'] > 3.0:
            report += "• Consider model retraining - MAE trending high\n"
            
        if abs(weekly_metrics['avg_bias']) > 0.75:
            report += f"• Apply bias correction - Systematic {'over' if weekly_metrics['avg_bias'] > 0 else 'under'}-prediction\n"
            
        if weekly_metrics['avg_accuracy_within_1'] < 0.35:
            report += "• Focus on feature engineering - Low precision suggests weak features\n"
            
        report += "• Continue daily monitoring and data collection\n"
        report += "• Review feature importance and model performance\n"
        
        return report
        
    def track_single_date(self, date_str):
        """Track performance for a single date"""
        metrics = self.calculate_daily_metrics(date_str)
        if metrics:
            self.save_daily_metrics(metrics)
            alerts = self.check_performance_alerts(metrics)
            
            print(f"\n📊 Performance Report for {date_str}")
            print("=" * 50)
            print(f"Games Analyzed: {metrics['games_analyzed']}")
            print(f"MAE: {metrics['mae']:.3f}")
            print(f"Bias: {metrics['bias']:+.3f}")
            print(f"Accuracy (1 run): {metrics['accuracy_within_1']:.1%}")
            print(f"Accuracy (2 runs): {metrics['accuracy_within_2']:.1%}")
            
            if alerts:
                print(f"\n⚠️ ALERTS:")
                for alert in alerts:
                    print(f"  {alert['type']}: {alert['message']}")
            else:
                print(f"\n✅ No performance alerts")
                
        return metrics

def main():
    """Run daily performance tracking"""
    tracker = DailyPerformanceTracker()
    
    # Track yesterday's performance
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"Tracking performance for {yesterday}...")
    
    metrics = tracker.track_single_date(yesterday)
    
    # Generate weekly report
    print(f"\nGenerating weekly report...")
    tracker.generate_weekly_report()

if __name__ == "__main__":
    main()
