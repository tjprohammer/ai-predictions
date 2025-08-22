#!/usr/bin/env python3
"""
Comprehensive Quality Analysis for Enhanced Pitcher Data
========================================================

This script performs comprehensive quality testing and analysis of:
1. Enhanced pitcher statistics coverage and completeness
2. Learning model performance before/after pitcher data enhancement
3. Prediction accuracy validation against historical games
4. Data integrity and consistency checks
5. Feature engineering validation

Usage:
    python comprehensive_quality_analysis.py [--date-range START END] [--verbose]
"""

import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import sys
import os
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from feature_engineering.enhanced_feature_builder import EnhancedFeatureBuilder
    from api.models.ml_models import MLBOversPredictor
except ImportError as e:
    print(f"⚠️  Warning: Could not import ML components: {e}")
    print("   Some analysis features may be limited")

class ComprehensiveQualityAnalyzer:
    """Comprehensive quality analysis for enhanced pitcher data and ML models"""
    
    def __init__(self, db_config: Optional[Dict] = None, verbose: bool = False):
        self.verbose = verbose
        self.db_config = db_config or {
            'host': 'localhost',
            'database': 'mlb',
            'user': 'mlbuser',
            'password': 'mlbpass'
        }
        self.conn = None
        self.results = {}
        
    def connect_db(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            if self.verbose:
                print("✅ Database connection established")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            sys.exit(1)
    
    def close_db(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if self.verbose or level in ["ERROR", "WARNING"]:
            print(f"[{timestamp}] {level}: {message}")
    
    def analyze_data_completeness(self, start_date: str = "2025-03-20", end_date: str = "2025-08-20") -> Dict:
        """Analyze completeness of enhanced pitcher data"""
        self.log("📊 Analyzing data completeness...")
        
        cursor = self.conn.cursor()
        
        # Overall completeness metrics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_games,
                COUNT(CASE WHEN game_status = 'Final' THEN 1 END) as final_games,
                COUNT(home_sp_ip) as with_starting_pitcher,
                COUNT(home_bp_ip) as with_bullpen_stats,
                COUNT(CASE WHEN home_sp_ip IS NOT NULL AND home_bp_ip IS NOT NULL 
                           AND away_sp_ip IS NOT NULL AND away_bp_ip IS NOT NULL 
                      THEN 1 END) as with_complete_pitcher_stats,
                COUNT(home_final_score) as with_scores,
                COUNT(predicted_total) as with_predictions,
                COUNT(market_total) as with_market_data
            FROM enhanced_games 
            WHERE date >= %s AND date <= %s
        """, (start_date, end_date))
        
        total, final, sp, bp, complete, scores, predictions, market = cursor.fetchone()
        
        completeness = {
            'total_games': total,
            'final_games': final,
            'starting_pitcher_coverage': sp / total * 100 if total > 0 else 0,
            'bullpen_coverage': bp / total * 100 if total > 0 else 0,
            'complete_pitcher_coverage': complete / total * 100 if total > 0 else 0,
            'scores_coverage': scores / total * 100 if total > 0 else 0,
            'predictions_coverage': predictions / total * 100 if total > 0 else 0,
            'market_data_coverage': market / total * 100 if total > 0 else 0
        }
        
        # Monthly breakdown
        cursor.execute("""
            SELECT 
                EXTRACT(MONTH FROM date::date) as month,
                COUNT(*) as total_games,
                COUNT(CASE WHEN home_sp_ip IS NOT NULL AND home_bp_ip IS NOT NULL 
                           AND away_sp_ip IS NOT NULL AND away_bp_ip IS NOT NULL 
                      THEN 1 END) as complete_pitcher_stats,
                ROUND(COUNT(CASE WHEN home_sp_ip IS NOT NULL AND home_bp_ip IS NOT NULL 
                                 AND away_sp_ip IS NOT NULL AND away_bp_ip IS NOT NULL 
                            THEN 1 END) * 100.0 / COUNT(*), 1) as coverage_pct
            FROM enhanced_games 
            WHERE date >= %s AND date <= %s
            GROUP BY EXTRACT(MONTH FROM date::date)
            ORDER BY month
        """, (start_date, end_date))
        
        monthly_coverage = []
        month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for month, games, complete_stats, coverage in cursor.fetchall():
            monthly_coverage.append({
                'month': month_names[int(month)],
                'total_games': games,
                'complete_pitcher_stats': complete_stats,
                'coverage_percentage': coverage
            })
        
        completeness['monthly_breakdown'] = monthly_coverage
        
        self.log(f"✅ Data completeness analysis complete - {complete}/{total} games ({complete/total*100:.1f}%) with complete pitcher stats")
        return completeness
    
    def analyze_data_quality(self, sample_size: int = 100) -> Dict:
        """Analyze quality and consistency of pitcher data"""
        self.log("🔍 Analyzing data quality...")
        
        cursor = self.conn.cursor()
        
        # Sample recent games for quality checks
        cursor.execute("""
            SELECT game_id, home_team, away_team, date,
                   home_final_score, away_final_score,
                   home_sp_ip, home_sp_er, home_sp_k, home_sp_bb, home_sp_h,
                   home_bp_ip, home_bp_er, home_bp_k, home_bp_bb, home_bp_h,
                   away_sp_ip, away_sp_er, away_sp_k, away_sp_bb, away_sp_h,
                   away_bp_ip, away_bp_er, away_bp_k, away_bp_bb, away_bp_h
            FROM enhanced_games 
            WHERE date >= '2025-08-10' 
                AND home_sp_ip IS NOT NULL AND home_bp_ip IS NOT NULL
                AND away_sp_ip IS NOT NULL AND away_bp_ip IS NOT NULL
                AND game_status = 'Final'
            ORDER BY date DESC
            LIMIT %s
        """, (sample_size,))
        
        quality_issues = []
        valid_games = 0
        total_sampled = 0
        
        for row in cursor.fetchall():
            total_sampled += 1
            game_id, home, away, date, h_score, a_score = row[:6]
            h_sp_ip, h_sp_er, h_sp_k, h_sp_bb, h_sp_h = row[6:11]
            h_bp_ip, h_bp_er, h_bp_k, h_bp_bb, h_bp_h = row[11:16]
            a_sp_ip, a_sp_er, a_sp_k, a_sp_bb, a_sp_h = row[16:21]
            a_bp_ip, a_bp_er, a_bp_k, a_bp_bb, a_bp_h = row[21:26]
            
            issues = []
            
            # Check for logical consistency
            total_innings_home = (h_sp_ip or 0) + (h_bp_ip or 0)
            total_innings_away = (a_sp_ip or 0) + (a_bp_ip or 0)
            
            # Most games should be around 9 innings total
            if total_innings_home < 8 or total_innings_home > 12:
                issues.append(f"Home innings suspicious: {total_innings_home}")
            if total_innings_away < 8 or total_innings_away > 12:
                issues.append(f"Away innings suspicious: {total_innings_away}")
            
            # ERA consistency checks (ER should be reasonable for IP)
            if h_sp_ip and h_sp_ip > 0:
                h_sp_era = (h_sp_er or 0) * 9 / h_sp_ip
                if h_sp_era > 20:  # ERA over 20 is very suspicious
                    issues.append(f"Home SP ERA suspicious: {h_sp_era:.1f}")
            
            if a_sp_ip and a_sp_ip > 0:
                a_sp_era = (a_sp_er or 0) * 9 / a_sp_ip
                if a_sp_era > 20:
                    issues.append(f"Away SP ERA suspicious: {a_sp_era:.1f}")
            
            # Check for missing critical data
            if not all([h_sp_ip, h_bp_ip, a_sp_ip, a_bp_ip]):
                issues.append("Missing innings pitched data")
            
            if issues:
                quality_issues.append({
                    'game_id': game_id,
                    'matchup': f"{away} @ {home}",
                    'date': str(date),
                    'issues': issues
                })
            else:
                valid_games += 1
        
        quality_score = (valid_games / total_sampled * 100) if total_sampled > 0 else 0
        
        quality_analysis = {
            'total_sampled': total_sampled,
            'valid_games': valid_games,
            'quality_score': quality_score,
            'issues_found': len(quality_issues),
            'sample_issues': quality_issues[:10]  # Show first 10 issues
        }
        
        self.log(f"✅ Data quality analysis complete - {quality_score:.1f}% quality score")
        return quality_analysis
    
    def analyze_prediction_accuracy(self, days_back: int = 30) -> Dict:
        """Analyze prediction accuracy against actual outcomes"""
        self.log("🎯 Analyzing prediction accuracy...")
        
        cursor = self.conn.cursor()
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        # Get games with predictions and actual outcomes
        cursor.execute("""
            SELECT game_id, home_team, away_team, date,
                   home_final_score, away_final_score,
                   predicted_total, market_total, edge, confidence
            FROM enhanced_games 
            WHERE date >= %s AND date <= %s
                AND game_status = 'Final'
                AND predicted_total IS NOT NULL
                AND home_final_score IS NOT NULL
                AND away_final_score IS NOT NULL
            ORDER BY date DESC
        """, (start_date, end_date))
        
        predictions = []
        for row in cursor.fetchall():
            game_id, home, away, date, h_score, a_score, pred_total, market_total, edge, confidence = row
            actual_total = (h_score or 0) + (a_score or 0)
            
            predictions.append({
                'game_id': game_id,
                'matchup': f"{away} @ {home}",
                'date': str(date),
                'actual_total': actual_total,
                'predicted_total': float(pred_total) if pred_total else None,
                'market_total': float(market_total) if market_total else None,
                'edge': float(edge) if edge else None,
                'confidence': confidence,
                'prediction_error': abs(actual_total - float(pred_total)) if pred_total else None,
                'market_error': abs(actual_total - float(market_total)) if market_total else None
            })
        
        if not predictions:
            return {'error': 'No predictions found for analysis period'}
        
        df = pd.DataFrame(predictions)
        
        # Calculate accuracy metrics
        mean_prediction_error = df['prediction_error'].mean()
        mean_market_error = df['market_error'].mean()
        median_prediction_error = df['prediction_error'].median()
        median_market_error = df['market_error'].median()
        
        # Accuracy by confidence level
        confidence_analysis = []
        if 'confidence' in df.columns and df['confidence'].notna().any():
            for conf_level in ['High', 'Medium', 'Low']:
                conf_games = df[df['confidence'] == conf_level]
                if len(conf_games) > 0:
                    confidence_analysis.append({
                        'confidence_level': conf_level,
                        'game_count': len(conf_games),
                        'mean_error': conf_games['prediction_error'].mean(),
                        'median_error': conf_games['prediction_error'].median()
                    })
        
        # Over/Under accuracy
        over_under_analysis = {
            'correct_over_calls': 0,
            'total_over_calls': 0,
            'correct_under_calls': 0,
            'total_under_calls': 0
        }
        
        for _, game in df.iterrows():
            if game['predicted_total'] and game['market_total']:
                if game['predicted_total'] > game['market_total']:  # Over prediction
                    over_under_analysis['total_over_calls'] += 1
                    if game['actual_total'] > game['market_total']:
                        over_under_analysis['correct_over_calls'] += 1
                elif game['predicted_total'] < game['market_total']:  # Under prediction
                    over_under_analysis['total_under_calls'] += 1
                    if game['actual_total'] < game['market_total']:
                        over_under_analysis['correct_under_calls'] += 1
        
        accuracy_analysis = {
            'total_games_analyzed': len(predictions),
            'mean_prediction_error': mean_prediction_error,
            'mean_market_error': mean_market_error,
            'median_prediction_error': median_prediction_error,
            'median_market_error': median_market_error,
            'improvement_over_market': mean_market_error - mean_prediction_error,
            'confidence_level_analysis': confidence_analysis,
            'over_under_accuracy': over_under_analysis,
            'sample_predictions': predictions[:10]  # Show first 10 predictions
        }
        
        self.log(f"✅ Prediction accuracy analysis complete - MAE: {mean_prediction_error:.2f} (vs market: {mean_market_error:.2f})")
        return accuracy_analysis
    
    def analyze_feature_coverage(self) -> Dict:
        """Analyze coverage of features used in ML models"""
        self.log("🔧 Analyzing feature coverage...")
        
        cursor = self.conn.cursor()
        
        # Check availability of key features
        feature_checks = {
            'pitcher_stats': 'home_sp_ip IS NOT NULL AND away_sp_ip IS NOT NULL',
            'bullpen_stats': 'home_bp_ip IS NOT NULL AND away_bp_ip IS NOT NULL',
            'team_stats': 'home_team IS NOT NULL AND away_team IS NOT NULL',
            'weather_data': 'weather_temp IS NOT NULL',
            'ballpark_factors': 'ballpark_factor IS NOT NULL',
            'recent_form': 'home_runs_last_10 IS NOT NULL',
            'pitcher_matchups': 'home_starter IS NOT NULL AND away_starter IS NOT NULL'
        }
        
        feature_coverage = {}
        total_recent_games = 0
        
        # Check recent games (last 7 days)
        cursor.execute("""
            SELECT COUNT(*) FROM enhanced_games 
            WHERE date >= CURRENT_DATE - INTERVAL '7 days'
                AND game_status = 'Final'
        """)
        total_recent_games = cursor.fetchone()[0]
        
        for feature_name, condition in feature_checks.items():
            cursor.execute(f"""
                SELECT COUNT(*) FROM enhanced_games 
                WHERE date >= CURRENT_DATE - INTERVAL '7 days'
                    AND game_status = 'Final'
                    AND {condition}
            """)
            count = cursor.fetchone()[0]
            coverage_pct = (count / total_recent_games * 100) if total_recent_games > 0 else 0
            
            feature_coverage[feature_name] = {
                'available_games': count,
                'total_games': total_recent_games,
                'coverage_percentage': coverage_pct
            }
        
        # Critical feature availability
        cursor.execute("""
            SELECT COUNT(*) FROM enhanced_games 
            WHERE date >= CURRENT_DATE - INTERVAL '7 days'
                AND game_status = 'Final'
                AND home_sp_ip IS NOT NULL 
                AND away_sp_ip IS NOT NULL
                AND home_bp_ip IS NOT NULL 
                AND away_bp_ip IS NOT NULL
                AND home_final_score IS NOT NULL
        """)
        
        critical_features_count = cursor.fetchone()[0]
        critical_coverage = (critical_features_count / total_recent_games * 100) if total_recent_games > 0 else 0
        
        feature_analysis = {
            'total_recent_games': total_recent_games,
            'individual_features': feature_coverage,
            'critical_features_coverage': critical_coverage,
            'ml_readiness_score': critical_coverage
        }
        
        self.log(f"✅ Feature coverage analysis complete - {critical_coverage:.1f}% ML readiness")
        return feature_analysis
    
    def compare_model_performance(self) -> Dict:
        """Compare model performance before and after pitcher data enhancement"""
        self.log("📈 Comparing model performance...")
        
        cursor = self.conn.cursor()
        
        # Define periods for comparison
        # Before enhancement: games without complete pitcher stats
        # After enhancement: games with complete pitcher stats
        
        cursor.execute("""
            -- Performance without pitcher stats
            SELECT 
                COUNT(*) as games,
                AVG(ABS(COALESCE(predicted_total, market_total) - (home_final_score + away_final_score))) as mae,
                STDDEV(ABS(COALESCE(predicted_total, market_total) - (home_final_score + away_final_score))) as std_error
            FROM enhanced_games 
            WHERE date >= '2025-03-20' AND date <= '2025-08-20'
                AND game_status = 'Final'
                AND home_final_score IS NOT NULL
                AND away_final_score IS NOT NULL
                AND (home_sp_ip IS NULL OR away_sp_ip IS NULL)
                AND predicted_total IS NOT NULL
        """)
        
        without_pitcher_stats = cursor.fetchone()
        
        cursor.execute("""
            -- Performance with complete pitcher stats
            SELECT 
                COUNT(*) as games,
                AVG(ABS(predicted_total - (home_final_score + away_final_score))) as mae,
                STDDEV(ABS(predicted_total - (home_final_score + away_final_score))) as std_error
            FROM enhanced_games 
            WHERE date >= '2025-03-20' AND date <= '2025-08-20'
                AND game_status = 'Final'
                AND home_final_score IS NOT NULL
                AND away_final_score IS NOT NULL
                AND home_sp_ip IS NOT NULL AND away_sp_ip IS NOT NULL
                AND home_bp_ip IS NOT NULL AND away_bp_ip IS NOT NULL
                AND predicted_total IS NOT NULL
        """)
        
        with_pitcher_stats = cursor.fetchone()
        
        # Calculate improvement metrics
        performance_comparison = {
            'without_pitcher_stats': {
                'games': without_pitcher_stats[0] if without_pitcher_stats[0] else 0,
                'mae': float(without_pitcher_stats[1]) if without_pitcher_stats[1] else None,
                'std_error': float(without_pitcher_stats[2]) if without_pitcher_stats[2] else None
            },
            'with_pitcher_stats': {
                'games': with_pitcher_stats[0] if with_pitcher_stats[0] else 0,
                'mae': float(with_pitcher_stats[1]) if with_pitcher_stats[1] else None,
                'std_error': float(with_pitcher_stats[2]) if with_pitcher_stats[2] else None
            }
        }
        
        # Calculate improvement
        if (performance_comparison['without_pitcher_stats']['mae'] and 
            performance_comparison['with_pitcher_stats']['mae']):
            
            mae_improvement = (performance_comparison['without_pitcher_stats']['mae'] - 
                             performance_comparison['with_pitcher_stats']['mae'])
            improvement_pct = (mae_improvement / performance_comparison['without_pitcher_stats']['mae']) * 100
            
            performance_comparison['improvement'] = {
                'mae_reduction': mae_improvement,
                'improvement_percentage': improvement_pct
            }
        
        self.log(f"✅ Model performance comparison complete")
        return performance_comparison
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        # Data completeness recommendations
        if 'data_completeness' in self.results:
            completeness = self.results['data_completeness']
            if completeness['complete_pitcher_coverage'] < 95:
                recommendations.append(
                    f"🔧 Improve pitcher data coverage from {completeness['complete_pitcher_coverage']:.1f}% to >95%"
                )
            
            if completeness['market_data_coverage'] < 90:
                recommendations.append(
                    f"📊 Enhance market data collection from {completeness['market_data_coverage']:.1f}% to >90%"
                )
        
        # Data quality recommendations
        if 'data_quality' in self.results:
            quality = self.results['data_quality']
            if quality['quality_score'] < 95:
                recommendations.append(
                    f"🔍 Address data quality issues - current score: {quality['quality_score']:.1f}%"
                )
        
        # Prediction accuracy recommendations
        if 'prediction_accuracy' in self.results:
            accuracy = self.results['prediction_accuracy']
            if accuracy.get('improvement_over_market', 0) < 0.1:
                recommendations.append(
                    "🎯 Focus on model tuning to improve accuracy over market predictions"
                )
        
        # Feature coverage recommendations
        if 'feature_coverage' in self.results:
            features = self.results['feature_coverage']
            if features['ml_readiness_score'] < 90:
                recommendations.append(
                    f"🔧 Improve feature availability from {features['ml_readiness_score']:.1f}% to >90%"
                )
        
        if not recommendations:
            recommendations.append("✅ All quality metrics are within acceptable ranges")
        
        return recommendations
    
    def run_comprehensive_analysis(self, start_date: str = "2025-03-20", 
                                 end_date: str = "2025-08-20") -> Dict:
        """Run complete quality analysis"""
        print("\n" + "="*80)
        print("🚀 COMPREHENSIVE QUALITY ANALYSIS")
        print("="*80)
        
        self.connect_db()
        
        try:
            # Run all analyses
            self.results['data_completeness'] = self.analyze_data_completeness(start_date, end_date)
            self.results['data_quality'] = self.analyze_data_quality()
            self.results['prediction_accuracy'] = self.analyze_prediction_accuracy()
            self.results['feature_coverage'] = self.analyze_feature_coverage()
            self.results['model_performance'] = self.compare_model_performance()
            
            # Generate recommendations
            self.results['recommendations'] = self.generate_recommendations()
            
            # Add metadata
            self.results['analysis_metadata'] = {
                'analysis_date': datetime.now().isoformat(),
                'date_range': f"{start_date} to {end_date}",
                'version': "1.0"
            }
            
        finally:
            self.close_db()
        
        return self.results
    
    def print_summary_report(self):
        """Print a formatted summary report"""
        print("\n" + "="*80)
        print("📋 QUALITY ANALYSIS SUMMARY REPORT")
        print("="*80)
        
        if 'data_completeness' in self.results:
            completeness = self.results['data_completeness']
            print(f"\n📊 DATA COMPLETENESS:")
            print(f"   Total Games: {completeness['total_games']:,}")
            print(f"   Complete Pitcher Stats: {completeness['complete_pitcher_coverage']:.1f}%")
            print(f"   Bullpen Coverage: {completeness['bullpen_coverage']:.1f}%")
            print(f"   Market Data Coverage: {completeness['market_data_coverage']:.1f}%")
        
        if 'data_quality' in self.results:
            quality = self.results['data_quality']
            print(f"\n🔍 DATA QUALITY:")
            print(f"   Quality Score: {quality['quality_score']:.1f}%")
            print(f"   Issues Found: {quality['issues_found']}")
            print(f"   Valid Games: {quality['valid_games']}/{quality['total_sampled']}")
        
        if 'prediction_accuracy' in self.results:
            accuracy = self.results['prediction_accuracy']
            print(f"\n🎯 PREDICTION ACCURACY:")
            print(f"   Mean Prediction Error: {accuracy['mean_prediction_error']:.2f}")
            print(f"   Mean Market Error: {accuracy['mean_market_error']:.2f}")
            print(f"   Improvement: {accuracy.get('improvement_over_market', 0):.2f}")
        
        if 'model_performance' in self.results:
            perf = self.results['model_performance']
            print(f"\n📈 MODEL PERFORMANCE:")
            if perf['with_pitcher_stats']['mae']:
                print(f"   With Pitcher Stats MAE: {perf['with_pitcher_stats']['mae']:.2f}")
            if perf.get('improvement'):
                print(f"   Improvement: {perf['improvement']['improvement_percentage']:.1f}%")
        
        if 'recommendations' in self.results:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in self.results['recommendations']:
                print(f"   {rec}")
        
        print("\n" + "="*80)

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Comprehensive Quality Analysis')
    parser.add_argument('--start-date', default='2025-03-20', help='Start date for analysis')
    parser.add_argument('--end-date', default='2025-08-20', help='End date for analysis')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--output', '-o', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = ComprehensiveQualityAnalyzer(verbose=args.verbose)
    results = analyzer.run_comprehensive_analysis(args.start_date, args.end_date)
    
    # Print summary
    analyzer.print_summary_report()
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to {args.output}")
    
    return results

if __name__ == "__main__":
    main()
