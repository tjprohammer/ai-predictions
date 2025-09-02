#!/usr/bin/env python3
"""
Enhanced Prediction Tracker with Real-time Model Performance Monitoring
========================================================================
Provides comprehensive analysis of model performance with automated insights
and bias correction recommendations.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os
import warnings
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

class EnhancedPredictionTracker:
    def __init__(self):
        # Use PostgreSQL connection
        DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
        self.engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        self.predictions_file = "S:/Projects/AI_Predictions/mlb/core/exports/daily_predictions.json"
        
    def get_comprehensive_performance_analysis(self, days=14) -> Dict:
        """Get comprehensive model performance analysis"""
        print(f"[ANALYSIS] ENHANCED PREDICTION PERFORMANCE ANALYSIS ({days} days)")
        print("=" * 60)
        
        try:
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            
            query = text("""
            SELECT 
                date,
                game_id,
                home_team,
                away_team,
                home_score,
                away_score,
                total_runs,
                venue_name,
                temperature,
                wind_speed,
                weather_condition,
                home_sp_id,
                away_sp_id,
                market_total,
                over_odds,
                under_odds,
                predicted_total,
                predicted_total_learning,
                COALESCE(predicted_total_learning, predicted_total) as primary_prediction,
                confidence,
                edge,
                recommendation
            FROM enhanced_games 
            WHERE date >= :start_date 
                AND date <= :end_date
                AND home_score IS NOT NULL 
                AND away_score IS NOT NULL
                AND (predicted_total_learning IS NOT NULL OR predicted_total IS NOT NULL)
            ORDER BY date DESC, game_id
            """)
            
            df = pd.read_sql(query, self.engine, params={
                'start_date': start_date,
                'end_date': end_date
            })
            
            if df.empty:
                print("[ERROR] No games with outcomes found in database")
                return {}
                
            print(f"[SUCCESS] Found {len(df)} games with outcomes")
            
            # Calculate comprehensive metrics
            analysis = self._calculate_comprehensive_metrics(df)
            
            # Generate insights and recommendations
            insights = self._generate_performance_insights(df, analysis)
            
            # Create bias correction recommendations
            corrections = self._generate_bias_corrections(df, analysis)
            
            return {
                'period': f"{start_date} to {end_date}",
                'games_analyzed': len(df),
                'metrics': analysis,
                'insights': insights,
                'bias_corrections': corrections,
                'raw_data': df.to_dict('records')
            }
            
        except Exception as e:
            print(f"[ERROR] Error in performance analysis: {e}")
            return {}
    
    def _calculate_comprehensive_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive performance metrics using primary prediction (Ultra 80 when available)"""
        
        # Basic accuracy metrics using primary prediction (Ultra 80 when available)
        df['prediction_error'] = abs(df['primary_prediction'] - df['total_runs'])
        df['prediction_bias'] = df['primary_prediction'] - df['total_runs']
        
        # Track which model was used
        df['model_used'] = df['predicted_total_learning'].notna().map({True: 'Ultra 80 Incremental', False: 'Learning Model'})
        
        # Scoring ranges for analysis
        df['scoring_range'] = pd.cut(df['total_runs'], 
                                   bins=[0, 7, 9, 11, float('inf')], 
                                   labels=['Low (≤7)', 'Medium (8-9)', 'High (10-11)', 'Very High (12+)'])
        
        # Market comparison
        df['market_error'] = abs(df['market_total'] - df['total_runs']) if 'market_total' in df.columns else None
        df['vs_market'] = df['primary_prediction'] - df['market_total'] if 'market_total' in df.columns else None
        
        # Confidence-based analysis (simplified since we don't have is_high_confidence flags)
        high_conf = df[df['confidence'] >= 75] if 'confidence' in df.columns else pd.DataFrame()
        premium_picks = df[df['confidence'] >= 85] if 'confidence' in df.columns else pd.DataFrame()
        
        metrics = {
            'overall': {
                'mean_absolute_error': float(df['prediction_error'].mean()),
                'median_absolute_error': float(df['prediction_error'].median()),
                'mean_bias': float(df['prediction_bias'].mean()),
                'rmse': float(np.sqrt((df['prediction_bias'] ** 2).mean())),
                'accuracy_within_1': float((df['prediction_error'] <= 1.0).mean()),
                'accuracy_within_2': float((df['prediction_error'] <= 2.0).mean()),
                'r_squared': float(np.corrcoef(df['predicted_total'], df['total_runs'])[0,1] ** 2),
            },
            'by_scoring_range': {},
            'confidence_analysis': {},
            'weather_impact': {},
            'venue_analysis': {},
            'market_comparison': {},
            'day_patterns': {},      # NEW: Day-of-week analysis
            'pitcher_quality': {},   # NEW: ERA-based analysis
            'market_analysis': {}    # NEW: Market deviation analysis
        }
        
        # Performance by scoring range
        for range_name in df['scoring_range'].cat.categories:
            range_data = df[df['scoring_range'] == range_name]
            if len(range_data) > 0:
                metrics['by_scoring_range'][range_name] = {
                    'games': len(range_data),
                    'mean_error': float(range_data['prediction_error'].mean()),
                    'mean_bias': float(range_data['prediction_bias'].mean()),
                    'accuracy_within_1': float((range_data['prediction_error'] <= 1.0).mean())
                }
        
        # High confidence analysis
        if len(high_conf) > 0:
            metrics['confidence_analysis']['high_confidence'] = {
                'games': len(high_conf),
                'mean_error': float(high_conf['prediction_error'].mean()),
                'accuracy_within_1': float((high_conf['prediction_error'] <= 1.0).mean()),
                'mean_bias': float(high_conf['prediction_bias'].mean())
            }
        
        # Premium picks analysis
        if len(premium_picks) > 0:
            metrics['confidence_analysis']['premium_picks'] = {
                'games': len(premium_picks),
                'mean_error': float(premium_picks['prediction_error'].mean()),
                'accuracy_within_1': float((premium_picks['prediction_error'] <= 1.0).mean()),
                'mean_bias': float(premium_picks['prediction_bias'].mean())
            }
        
        # Weather impact
        if 'temperature' in df.columns:
            df['temp_range'] = pd.cut(df['temperature'], 
                                    bins=[0, 60, 75, 85, float('inf')], 
                                    labels=['Cold', 'Cool', 'Warm', 'Hot'])
            
            for temp_range in df['temp_range'].cat.categories:
                temp_data = df[df['temp_range'] == temp_range]
                if len(temp_data) > 5:  # Minimum sample size
                    metrics['weather_impact'][temp_range] = {
                        'games': len(temp_data),
                        'mean_error': float(temp_data['prediction_error'].mean()),
                        'mean_bias': float(temp_data['prediction_bias'].mean())
                    }
        
        # Market comparison
        if 'market_total' in df.columns and df['market_total'].notna().any():
            valid_market = df.dropna(subset=['market_total'])
            if len(valid_market) > 0:
                metrics['market_comparison'] = {
                    'games_with_market': len(valid_market),
                    'model_vs_market_mae': float(valid_market['prediction_error'].mean()),
                    'market_mae': float(valid_market['market_error'].mean()),
                    'model_advantage': float(valid_market['market_error'].mean() - valid_market['prediction_error'].mean()),
                    'mean_difference_from_market': float(valid_market['vs_market'].mean())
                }
        
        # NEW: Day-of-week patterns (from 14-day analysis insights)
        if 'date' in df.columns:
            df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
            day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                        4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
            
            for day_num, day_name in day_names.items():
                day_data = df[df['day_of_week'] == day_num]
                if len(day_data) > 3:  # Minimum sample size
                    metrics['day_patterns'][day_name] = {
                        'games': len(day_data),
                        'mae': float(day_data['prediction_error'].mean()),
                        'mean_bias': float(day_data['prediction_bias'].mean()),
                        'accuracy_within_1': float((day_data['prediction_error'] <= 1.0).mean())
                    }
        
        # NEW: Pitcher quality analysis (ERA-based from 14-day insights)
        if 'home_sp_season_era' in df.columns and 'away_sp_season_era' in df.columns:
            # Calculate combined ERA
            df['combined_era'] = (df['home_sp_season_era'] + df['away_sp_season_era']) / 2
            df['era_range'] = pd.cut(df['combined_era'], 
                                   bins=[0, 3.5, 4.5, 5.5, float('inf')], 
                                   labels=['Elite (<3.5)', 'Good (3.5-4.5)', 'Average (4.5-5.5)', 'Poor (5.5+)'])
            
            for era_range in df['era_range'].cat.categories:
                era_data = df[df['era_range'] == era_range]
                if len(era_data) > 3:  # Minimum sample size
                    metrics['pitcher_quality'][era_range] = {
                        'games': len(era_data),
                        'mae': float(era_data['prediction_error'].mean()),
                        'mean_bias': float(era_data['prediction_bias'].mean()),
                        'accuracy_within_1': float((era_data['prediction_error'] <= 1.0).mean())
                    }
        
        # NEW: Market deviation analysis (from 14-day insights)
        if 'market_total' in df.columns and df['market_total'].notna().any():
            valid_market = df.dropna(subset=['market_total'])
            if len(valid_market) > 0:
                # Calculate market deviation
                valid_market['market_deviation'] = abs(valid_market['predicted_total'] - valid_market['market_total'])
                valid_market['deviation_range'] = pd.cut(valid_market['market_deviation'], 
                                                       bins=[0, 0.5, 1.0, 2.0, float('inf')], 
                                                       labels=['Close (≤0.5)', 'Small (0.5-1.0)', 'Medium (1.0-2.0)', 'Large (2.0+)'])
                
                for dev_range in valid_market['deviation_range'].cat.categories:
                    dev_data = valid_market[valid_market['deviation_range'] == dev_range]
                    if len(dev_data) > 3:  # Minimum sample size
                        metrics['market_analysis'][dev_range] = {
                            'games': len(dev_data),
                            'mae': float(dev_data['prediction_error'].mean()),
                            'mean_bias': float(dev_data['prediction_bias'].mean()),
                            'accuracy_within_1': float((dev_data['prediction_error'] <= 1.0).mean())
                        }
        
        return metrics
    
    def _generate_performance_insights(self, df: pd.DataFrame, metrics: Dict) -> List[str]:
        """Generate actionable performance insights"""
        insights = []
        
        # Overall performance assessment
        mae = metrics['overall']['mean_absolute_error']
        bias = metrics['overall']['mean_bias']
        accuracy_1 = metrics['overall']['accuracy_within_1']
        
        if mae < 2.5:
            insights.append(f"[EXCELLENT] Model accuracy is strong with {mae:.2f} runs average error")
        elif mae < 3.5:
            insights.append(f"[GOOD] Model performance is solid with {mae:.2f} runs average error")
        else:
            insights.append(f"[WARNING] NEEDS IMPROVEMENT: Model accuracy needs work with {mae:.2f} runs average error")
        
        # Bias analysis
        if abs(bias) > 0.75:
            direction = "over-predicting" if bias > 0 else "under-predicting"
            insights.append(f"🎯 BIAS DETECTED: Model is systematically {direction} by {abs(bias):.2f} runs")
        
        # Accuracy rate assessment
        if accuracy_1 > 0.6:
            insights.append(f"🎯 HIGH PRECISION: {accuracy_1:.1%} of predictions within 1 run of actual")
        elif accuracy_1 > 0.4:
            insights.append(f"🎯 MODERATE PRECISION: {accuracy_1:.1%} of predictions within 1 run of actual")
        else:
            insights.append(f"[WARNING] LOW PRECISION: Only {accuracy_1:.1%} of predictions within 1 run of actual")
        
        # Scoring range analysis
        if 'by_scoring_range' in metrics:
            best_range = min(metrics['by_scoring_range'].items(), 
                           key=lambda x: x[1]['mean_error'])[0]
            worst_range = max(metrics['by_scoring_range'].items(), 
                            key=lambda x: x[1]['mean_error'])[0]
            
            insights.append(f"📊 RANGE PERFORMANCE: Best accuracy on {best_range} games, struggles with {worst_range}")
        
        # Confidence analysis
        if 'confidence_analysis' in metrics:
            if 'high_confidence' in metrics['confidence_analysis']:
                hc_metrics = metrics['confidence_analysis']['high_confidence']
                insights.append(f"🔥 HIGH CONFIDENCE: {hc_metrics['games']} games with {hc_metrics['mean_error']:.2f} runs error")
            
            if 'premium_picks' in metrics['confidence_analysis']:
                pp_metrics = metrics['confidence_analysis']['premium_picks']
                insights.append(f"⭐ PREMIUM PICKS: {pp_metrics['games']} games with {pp_metrics['mean_error']:.2f} runs error")
        
        # Market comparison
        if 'market_comparison' in metrics and metrics['market_comparison']:
            mc = metrics['market_comparison']
            if mc.get('model_advantage', 0) > 0:
                insights.append(f"💰 MARKET EDGE: Model outperforms Vegas by {mc['model_advantage']:.2f} runs")
            else:
                insights.append(f"📈 MARKET CHALLENGE: Vegas currently outperforms model by {abs(mc['model_advantage']):.2f} runs")
        
        return insights
    
    def _generate_bias_corrections(self, df: pd.DataFrame, metrics: Dict) -> Dict:
        """Generate bias correction recommendations"""
        corrections = {
            'overall_adjustment': 0.0,
            'scoring_range_adjustments': {},
            'confidence_adjustments': {},
            'weather_adjustments': {},
            'recommendations': []
        }
        
        # Overall bias correction
        overall_bias = metrics['overall']['mean_bias']
        if abs(overall_bias) > 0.5:
            corrections['overall_adjustment'] = -overall_bias
            direction = "reduce" if overall_bias > 0 else "increase"
            corrections['recommendations'].append(
                f"Apply global adjustment of {-overall_bias:.2f} runs to {direction} systematic bias"
            )
        
        # Scoring range corrections
        if 'by_scoring_range' in metrics:
            for range_name, range_metrics in metrics['by_scoring_range'].items():
                bias = range_metrics['mean_bias']
                if abs(bias) > 0.75 and range_metrics['games'] >= 5:
                    corrections['scoring_range_adjustments'][range_name] = -bias
                    corrections['recommendations'].append(
                        f"Adjust {range_name} scoring games by {-bias:.2f} runs"
                    )
        
        # Model retraining recommendations
        if metrics['overall']['mean_absolute_error'] > 3.0:
            corrections['recommendations'].append(
                "CRITICAL: Consider model retraining with recent data to improve accuracy"
            )
        
        if metrics['overall']['accuracy_within_1'] < 0.4:
            corrections['recommendations'].append(
                "Consider feature engineering improvements to boost precision"
            )
        
        return corrections
    
    def generate_performance_report(self, days=14) -> str:
        """Generate comprehensive performance report"""
        analysis = self.get_comprehensive_performance_analysis(days)
        
        if not analysis:
            return "[ERROR] Unable to generate performance report"
        
        report = []
        report.append("🎯 ENHANCED MODEL PERFORMANCE REPORT")
        report.append("=" * 50)
        report.append(f"Period: {analysis['period']}")
        report.append(f"Games Analyzed: {analysis['games_analyzed']}")
        report.append("")
        
        # Overall metrics
        metrics = analysis['metrics']['overall']
        report.append("📊 OVERALL PERFORMANCE:")
        report.append(f"  • Mean Absolute Error: {metrics['mean_absolute_error']:.2f} runs")
        report.append(f"  • Median Absolute Error: {metrics['median_absolute_error']:.2f} runs")
        report.append(f"  • Systematic Bias: {metrics['mean_bias']:.2f} runs")
        report.append(f"  • RMSE: {metrics['rmse']:.2f}")
        report.append(f"  • Accuracy within 1 run: {metrics['accuracy_within_1']:.1%}")
        report.append(f"  • Accuracy within 2 runs: {metrics['accuracy_within_2']:.1%}")
        report.append(f"  • R-Squared: {metrics['r_squared']:.3f}")
        report.append("")
        
        # Insights
        report.append("💡 KEY INSIGHTS:")
        for insight in analysis['insights']:
            report.append(f"  • {insight}")
        report.append("")
        
        # Bias corrections
        if analysis['bias_corrections']['recommendations']:
            report.append("🔧 RECOMMENDED CORRECTIONS:")
            for rec in analysis['bias_corrections']['recommendations']:
                report.append(f"  • {rec}")
            report.append("")
        
        # Scoring range performance
        if analysis['metrics']['by_scoring_range']:
            report.append("📊 PERFORMANCE BY SCORING RANGE:")
            for range_name, range_metrics in analysis['metrics']['by_scoring_range'].items():
                report.append(f"  • {range_name}: {range_metrics['mean_error']:.2f} runs error ({range_metrics['games']} games)")
            report.append("")
        
        # Market comparison
        if analysis['metrics']['market_comparison']:
            mc = analysis['metrics']['market_comparison']
            report.append("💰 MARKET COMPARISON:")
            report.append(f"  • Model MAE: {mc['model_vs_market_mae']:.2f} runs")
            report.append(f"  • Market MAE: {mc['market_mae']:.2f} runs")
            report.append(f"  • Model Advantage: {mc['model_advantage']:.2f} runs")
            report.append("")
        
        return "\n".join(report)
    
    def save_performance_analysis(self, days=14):
        """Save comprehensive performance analysis to JSON"""
        analysis = self.get_comprehensive_performance_analysis(days)
        
        if analysis:
            # Remove raw_data for clean JSON
            clean_analysis = {k: v for k, v in analysis.items() if k != 'raw_data'}
            
            output_file = f"performance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(clean_analysis, f, indent=2, default=str)
            
            print(f"💾 Performance analysis saved to: {output_file}")
            return output_file
        return None

def main():
    """Run enhanced prediction tracking analysis"""
    tracker = EnhancedPredictionTracker()
    
    # Generate and display performance report
    report = tracker.generate_performance_report(days=14)
    print(report)
    
    # Save analysis to file
    tracker.save_performance_analysis(days=14)

if __name__ == "__main__":
    main()
