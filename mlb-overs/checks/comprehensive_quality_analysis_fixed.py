#!/usr/bin/env python3
"""
Comprehensive Quality Analysis for Enhanced MLB Prediction System
Validates data quality, model performance, and prediction accuracy.
Updated for correct database schema.
"""

import sys
import os
import argparse
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Database and data handling
import psycopg2
import pandas as pd
import numpy as np

# ML imports (optional - skip if not available)
try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: sklearn not available. ML analysis will be skipped.")
    SKLEARN_AVAILABLE = False

class ComprehensiveQualityAnalyzer:
    def __init__(self, connection_params: Dict[str, str], verbose: bool = False):
        """Initialize the quality analyzer with database connection."""
        self.connection_params = connection_params
        self.verbose = verbose
        self.conn = None
        self.cursor = None
        
        # Analysis results storage
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'data_completeness': {},
            'data_quality': {},
            'prediction_accuracy': {},
            'feature_coverage': {},
            'recommendations': []
        }
        
    def connect_database(self) -> bool:
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(**self.connection_params)
            self.cursor = self.conn.cursor()
            if self.verbose:
                print("✓ Database connection established")
            return True
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            return False
            
    def disconnect_database(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            
    def analyze_data_completeness(self, start_date: str, end_date: str) -> Dict:
        """Analyze completeness of enhanced pitcher data."""
        if self.verbose:
            print(f"\n📊 Analyzing data completeness from {start_date} to {end_date}")
            
        # Data completeness analysis - using correct column names
        completeness_query = """
        SELECT 
            COUNT(*) as total_games,
            COUNT(CASE WHEN home_score IS NOT NULL AND away_score IS NOT NULL THEN 1 END) as completed_games,
            COUNT(CASE WHEN home_score IS NOT NULL AND away_score IS NOT NULL THEN 1 END) as games_with_scores,
            COUNT(CASE WHEN home_sp_name IS NOT NULL THEN 1 END) as games_with_home_sp,
            COUNT(CASE WHEN away_sp_name IS NOT NULL THEN 1 END) as games_with_away_sp,
            COUNT(CASE WHEN home_bp_ip IS NOT NULL THEN 1 END) as games_with_home_bp,
            COUNT(CASE WHEN away_bp_ip IS NOT NULL THEN 1 END) as games_with_away_bp,
            COUNT(CASE WHEN market_total IS NOT NULL THEN 1 END) as games_with_ou_line
        FROM enhanced_games 
        WHERE date >= %s AND date <= %s
        """
        
        self.cursor.execute(completeness_query, (start_date, end_date))
        completeness_data = self.cursor.fetchone()
        
        # Calculate completeness percentages
        total = completeness_data[0]
        if total == 0:
            print("⚠️  No games found in specified date range")
            return {}
            
        completeness_metrics = {
            'total_games': total,
            'completed_games': completeness_data[1],
            'scores_completeness': round(completeness_data[2] / total * 100, 2),
            'home_sp_completeness': round(completeness_data[3] / total * 100, 2),
            'away_sp_completeness': round(completeness_data[4] / total * 100, 2),
            'home_bp_completeness': round(completeness_data[5] / total * 100, 2),
            'away_bp_completeness': round(completeness_data[6] / total * 100, 2),
            'ou_line_completeness': round(completeness_data[7] / total * 100, 2)
        }
        
        if self.verbose:
            print(f"  Total games: {total}")
            print(f"  Completed games: {completeness_data[1]} ({completeness_metrics['scores_completeness']}%)")
            print(f"  Home SP data: {completeness_metrics['home_sp_completeness']}%")
            print(f"  Away SP data: {completeness_metrics['away_sp_completeness']}%")
            print(f"  Home BP data: {completeness_metrics['home_bp_completeness']}%")
            print(f"  Away BP data: {completeness_metrics['away_bp_completeness']}%")
            print(f"  O/U lines: {completeness_metrics['ou_line_completeness']}%")
            
        return completeness_metrics
        
    def analyze_data_quality(self, start_date: str, end_date: str) -> Dict:
        """Analyze quality metrics for pitcher and game data."""
        if self.verbose:
            print(f"\n🔍 Analyzing data quality metrics")
            
        quality_metrics = {}
        
        # Check for reasonable pitcher statistics
        pitcher_quality_query = """
        SELECT 
            AVG(CASE WHEN home_sp_season_era BETWEEN 0.5 AND 15.0 THEN 1 ELSE 0 END) as home_sp_era_valid,
            AVG(CASE WHEN away_sp_season_era BETWEEN 0.5 AND 15.0 THEN 1 ELSE 0 END) as away_sp_era_valid,
            AVG(CASE WHEN home_sp_whip BETWEEN 0.5 AND 3.0 THEN 1 ELSE 0 END) as home_sp_whip_valid,
            AVG(CASE WHEN away_sp_whip BETWEEN 0.5 AND 3.0 THEN 1 ELSE 0 END) as away_sp_whip_valid,
            AVG(CASE WHEN home_bp_ip >= 0 AND home_bp_ip <= 18 THEN 1 ELSE 0 END) as home_bp_ip_valid,
            AVG(CASE WHEN away_bp_ip >= 0 AND away_bp_ip <= 18 THEN 1 ELSE 0 END) as away_bp_ip_valid
        FROM enhanced_games
        WHERE date >= %s AND date <= %s
        AND home_score IS NOT NULL AND away_score IS NOT NULL
        """
        
        self.cursor.execute(pitcher_quality_query, (start_date, end_date))
        quality_data = self.cursor.fetchone()
        
        if quality_data:
            quality_metrics = {
                'home_sp_era_validity': round(quality_data[0] * 100, 2) if quality_data[0] else 0,
                'away_sp_era_validity': round(quality_data[1] * 100, 2) if quality_data[1] else 0,
                'home_sp_whip_validity': round(quality_data[2] * 100, 2) if quality_data[2] else 0,
                'away_sp_whip_validity': round(quality_data[3] * 100, 2) if quality_data[3] else 0,
                'home_bp_ip_validity': round(quality_data[4] * 100, 2) if quality_data[4] else 0,
                'away_bp_ip_validity': round(quality_data[5] * 100, 2) if quality_data[5] else 0,
            }
            
            # Calculate overall quality score
            valid_scores = [v for v in quality_metrics.values() if v > 0]
            quality_metrics['overall_quality_score'] = round(np.mean(valid_scores), 2) if valid_scores else 0
            
            if self.verbose:
                print(f"  Home SP ERA validity: {quality_metrics['home_sp_era_validity']}%")
                print(f"  Away SP ERA validity: {quality_metrics['away_sp_era_validity']}%")
                print(f"  Home SP WHIP validity: {quality_metrics['home_sp_whip_validity']}%")
                print(f"  Away SP WHIP validity: {quality_metrics['away_sp_whip_validity']}%")
                print(f"  Home BP IP validity: {quality_metrics['home_bp_ip_validity']}%")
                print(f"  Away BP IP validity: {quality_metrics['away_bp_ip_validity']}%")
                print(f"  Overall quality score: {quality_metrics['overall_quality_score']}%")
        
        return quality_metrics
        
    def analyze_prediction_accuracy(self, start_date: str, end_date: str) -> Dict:
        """Analyze prediction accuracy against actual game outcomes."""
        if self.verbose:
            print(f"\n🎯 Analyzing prediction accuracy")
            
        accuracy_metrics = {}
        
        # Check if we have prediction data
        prediction_check_query = """
        SELECT 
            COUNT(*) as total_predictions,
            COUNT(CASE WHEN predicted_total IS NOT NULL THEN 1 END) as with_predicted_total,
            COUNT(CASE WHEN market_total IS NOT NULL THEN 1 END) as with_market_line
        FROM enhanced_games
        WHERE date >= %s AND date <= %s
        AND home_score IS NOT NULL AND away_score IS NOT NULL
        """
        
        self.cursor.execute(prediction_check_query, (start_date, end_date))
        pred_data = self.cursor.fetchone()
        
        if pred_data and pred_data[0] > 0:
            accuracy_metrics['total_completed_games'] = pred_data[0]
            accuracy_metrics['games_with_predictions'] = pred_data[1]
            accuracy_metrics['games_with_market_lines'] = pred_data[2]
            
            # Analyze market line accuracy
            if pred_data[2] > 0:
                market_accuracy_query = """
                SELECT 
                    AVG(ABS(market_total - (home_score + away_score))) as market_mae,
                    STDDEV(ABS(market_total - (home_score + away_score))) as market_std_error
                FROM enhanced_games
                WHERE date >= %s AND date <= %s
                AND home_score IS NOT NULL AND away_score IS NOT NULL
                AND market_total IS NOT NULL
                """
                
                self.cursor.execute(market_accuracy_query, (start_date, end_date))
                market_acc = self.cursor.fetchone()
                
                if market_acc:
                    accuracy_metrics['market_mae'] = float(market_acc[0]) if market_acc[0] else 0
                    accuracy_metrics['market_std_error'] = float(market_acc[1]) if market_acc[1] else 0
            
            # Analyze model prediction accuracy (if available)
            if pred_data[1] > 0:
                model_accuracy_query = """
                SELECT 
                    AVG(ABS(predicted_total - (home_score + away_score))) as model_mae,
                    STDDEV(ABS(predicted_total - (home_score + away_score))) as model_std_error
                FROM enhanced_games
                WHERE date >= %s AND date <= %s
                AND home_score IS NOT NULL AND away_score IS NOT NULL
                AND predicted_total IS NOT NULL
                """
                
                self.cursor.execute(model_accuracy_query, (start_date, end_date))
                model_acc = self.cursor.fetchone()
                
                if model_acc:
                    accuracy_metrics['model_mae'] = float(model_acc[0]) if model_acc[0] else 0
                    accuracy_metrics['model_std_error'] = float(model_acc[1]) if model_acc[1] else 0
            
            if self.verbose:
                print(f"  Completed games analyzed: {accuracy_metrics['total_completed_games']}")
                print(f"  Games with model predictions: {accuracy_metrics['games_with_predictions']}")
                print(f"  Games with market lines: {accuracy_metrics['games_with_market_lines']}")
                if 'market_mae' in accuracy_metrics:
                    print(f"  Market line MAE: {accuracy_metrics['market_mae']}")
                if 'model_mae' in accuracy_metrics:
                    print(f"  Model prediction MAE: {accuracy_metrics['model_mae']}")
        
        return accuracy_metrics
        
    def analyze_feature_coverage(self, start_date: str, end_date: str) -> Dict:
        """Analyze coverage of enhanced pitcher features."""
        if self.verbose:
            print(f"\n📈 Analyzing feature coverage")
            
        # Enhanced pitcher feature coverage
        feature_query = """
        SELECT 
            AVG(CASE WHEN home_sp_season_era IS NOT NULL THEN 1 ELSE 0 END) as home_sp_era_coverage,
            AVG(CASE WHEN away_sp_season_era IS NOT NULL THEN 1 ELSE 0 END) as away_sp_era_coverage,
            AVG(CASE WHEN home_sp_whip IS NOT NULL THEN 1 ELSE 0 END) as home_sp_whip_coverage,
            AVG(CASE WHEN away_sp_whip IS NOT NULL THEN 1 ELSE 0 END) as away_sp_whip_coverage,
            AVG(CASE WHEN home_sp_days_rest IS NOT NULL THEN 1 ELSE 0 END) as home_sp_rest_coverage,
            AVG(CASE WHEN away_sp_days_rest IS NOT NULL THEN 1 ELSE 0 END) as away_sp_rest_coverage,
            AVG(CASE WHEN home_bp_ip IS NOT NULL THEN 1 ELSE 0 END) as home_bp_coverage,
            AVG(CASE WHEN away_bp_ip IS NOT NULL THEN 1 ELSE 0 END) as away_bp_coverage,
            AVG(CASE WHEN home_catcher IS NOT NULL THEN 1 ELSE 0 END) as home_catcher_coverage,
            AVG(CASE WHEN away_catcher IS NOT NULL THEN 1 ELSE 0 END) as away_catcher_coverage
        FROM enhanced_games
        WHERE date >= %s AND date <= %s
        """
        
        self.cursor.execute(feature_query, (start_date, end_date))
        feature_data = self.cursor.fetchone()
        
        coverage_metrics = {}
        if feature_data:
            coverage_metrics = {
                'home_sp_era_coverage': round(feature_data[0] * 100, 2) if feature_data[0] else 0,
                'away_sp_era_coverage': round(feature_data[1] * 100, 2) if feature_data[1] else 0,
                'home_sp_whip_coverage': round(feature_data[2] * 100, 2) if feature_data[2] else 0,
                'away_sp_whip_coverage': round(feature_data[3] * 100, 2) if feature_data[3] else 0,
                'home_sp_rest_coverage': round(feature_data[4] * 100, 2) if feature_data[4] else 0,
                'away_sp_rest_coverage': round(feature_data[5] * 100, 2) if feature_data[5] else 0,
                'home_bp_coverage': round(feature_data[6] * 100, 2) if feature_data[6] else 0,
                'away_bp_coverage': round(feature_data[7] * 100, 2) if feature_data[7] else 0,
                'home_catcher_coverage': round(feature_data[8] * 100, 2) if feature_data[8] else 0,
                'away_catcher_coverage': round(feature_data[9] * 100, 2) if feature_data[9] else 0,
            }
            
            # Calculate overall feature coverage
            coverage_values = [v for v in coverage_metrics.values() if v > 0]
            coverage_metrics['overall_feature_coverage'] = round(np.mean(coverage_values), 2) if coverage_values else 0
            
            if self.verbose:
                print(f"  Home SP ERA coverage: {coverage_metrics['home_sp_era_coverage']}%")
                print(f"  Away SP ERA coverage: {coverage_metrics['away_sp_era_coverage']}%")
                print(f"  Home BP coverage: {coverage_metrics['home_bp_coverage']}%")
                print(f"  Away BP coverage: {coverage_metrics['away_bp_coverage']}%")
                print(f"  Overall feature coverage: {coverage_metrics['overall_feature_coverage']}%")
        
        return coverage_metrics
        
    def generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on analysis results."""
        recommendations = []
        
        # Data completeness recommendations
        if 'data_completeness' in self.results:
            completeness = self.results['data_completeness']
            
            if completeness.get('home_sp_completeness', 0) < 95:
                recommendations.append(f"CRITICAL: Home starting pitcher data only {completeness.get('home_sp_completeness', 0)}% complete")
                
            if completeness.get('away_sp_completeness', 0) < 95:
                recommendations.append(f"CRITICAL: Away starting pitcher data only {completeness.get('away_sp_completeness', 0)}% complete")
                
            if completeness.get('home_bp_completeness', 0) < 80:
                recommendations.append(f"WARNING: Home bullpen data only {completeness.get('home_bp_completeness', 0)}% complete")
                
            if completeness.get('away_bp_completeness', 0) < 80:
                recommendations.append(f"WARNING: Away bullpen data only {completeness.get('away_bp_completeness', 0)}% complete")
        
        # Data quality recommendations
        if 'data_quality' in self.results:
            quality = self.results['data_quality']
            
            if quality.get('overall_quality_score', 0) < 90:
                recommendations.append(f"WARNING: Overall data quality score is {quality.get('overall_quality_score', 0)}% - investigate data validation")
                
            for metric, value in quality.items():
                if 'validity' in metric and value < 85:
                    recommendations.append(f"DATA ISSUE: {metric} is only {value}% valid - check data ranges")
        
        # Prediction accuracy recommendations
        if 'prediction_accuracy' in self.results:
            accuracy = self.results['prediction_accuracy']
            
            if accuracy.get('model_mae', 0) > 0 and accuracy.get('market_mae', 0) > 0:
                if accuracy['model_mae'] > accuracy['market_mae'] * 1.1:
                    recommendations.append("MODEL PERFORMANCE: Model predictions are less accurate than market lines - retrain recommended")
                elif accuracy['model_mae'] < accuracy['market_mae'] * 0.9:
                    recommendations.append("MODEL PERFORMANCE: Model outperforming market lines - good performance")
            
            if accuracy.get('games_with_predictions', 0) < accuracy.get('total_completed_games', 1) * 0.8:
                recommendations.append("PREDICTION COVERAGE: Less than 80% of games have model predictions - increase prediction coverage")
        
        # Feature coverage recommendations  
        if 'feature_coverage' in self.results:
            coverage = self.results['feature_coverage']
            
            if coverage.get('overall_feature_coverage', 0) < 85:
                recommendations.append(f"FEATURE ENGINEERING: Overall feature coverage is {coverage.get('overall_feature_coverage', 0)}% - enhance data collection")
                
            if coverage.get('home_bp_coverage', 0) < 75 or coverage.get('away_bp_coverage', 0) < 75:
                recommendations.append("BULLPEN DATA: Bullpen statistics coverage is low - prioritize bullpen data collection")
        
        # General recommendations
        if not recommendations:
            recommendations.append("EXCELLENT: All quality metrics are within acceptable ranges - system ready for production")
        else:
            recommendations.insert(0, f"SUMMARY: Found {len(recommendations)} areas for improvement")
            
        return recommendations
        
    def run_comprehensive_analysis(self, start_date: str = None, end_date: str = None) -> Dict:
        """Run complete quality analysis."""
        if not self.connect_database():
            return {}
            
        # Default date range: last 30 days
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"🔍 Running Comprehensive Quality Analysis")
        print(f"📅 Analysis Period: {start_date} to {end_date}")
        print("=" * 60)
        
        try:
            # Run all analysis components
            self.results['data_completeness'] = self.analyze_data_completeness(start_date, end_date)
            self.results['data_quality'] = self.analyze_data_quality(start_date, end_date)
            self.results['prediction_accuracy'] = self.analyze_prediction_accuracy(start_date, end_date)
            self.results['feature_coverage'] = self.analyze_feature_coverage(start_date, end_date)
            
            # Generate recommendations
            self.results['recommendations'] = self.generate_recommendations()
            
            # Summary
            print(f"\n📋 ANALYSIS SUMMARY")
            print("=" * 60)
            
            if self.results['data_completeness']:
                print(f"📊 Data Completeness: {self.results['data_completeness'].get('total_games', 0)} games analyzed")
                
            if self.results['data_quality']:
                print(f"🔍 Data Quality Score: {self.results['data_quality'].get('overall_quality_score', 0)}%")
                
            if self.results['feature_coverage']:
                print(f"📈 Feature Coverage: {self.results['feature_coverage'].get('overall_feature_coverage', 0)}%")
                
            if self.results['prediction_accuracy']:
                print(f"🎯 Prediction Coverage: {self.results['prediction_accuracy'].get('games_with_predictions', 0)} games")
            
            print(f"\n💡 RECOMMENDATIONS ({len(self.results['recommendations'])})")
            print("=" * 60)
            for i, rec in enumerate(self.results['recommendations'], 1):
                print(f"{i:2d}. {rec}")
                
        except Exception as e:
            print(f"✗ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.disconnect_database()
            
        return self.results
        
    def save_results(self, output_file: str = None):
        """Save analysis results to JSON file."""
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"quality_analysis_{timestamp}.json"
        
        # Convert any decimal values to float for JSON serialization
        import json
        def convert_decimals(obj):
            if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
                if isinstance(obj, dict):
                    return {k: convert_decimals(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_decimals(v) for v in obj]
                else:
                    return obj
            else:
                if hasattr(obj, 'quantize'):  # Decimal object
                    return float(obj)
                return obj
        
        converted_results = convert_decimals(self.results)
        
        with open(output_file, 'w') as f:
            json.dump(converted_results, f, indent=2)
            
        print(f"📄 Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Quality Analysis for Enhanced MLB Prediction System')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Database connection parameters
    connection_params = {
        'host': 'localhost',
        'database': 'mlb', 
        'user': 'mlbuser',
        'password': 'mlbpass'
    }
    
    # Create analyzer and run analysis
    analyzer = ComprehensiveQualityAnalyzer(connection_params, verbose=args.verbose)
    results = analyzer.run_comprehensive_analysis(args.start_date, args.end_date)
    
    # Save results
    if results:
        analyzer.save_results(args.output)
        
        # Print final status
        total_recommendations = len(results.get('recommendations', []))
        critical_issues = len([r for r in results.get('recommendations', []) if 'CRITICAL' in r])
        
        if critical_issues > 0:
            print(f"\n🚨 CRITICAL: {critical_issues} critical issues found - immediate attention required")
        elif total_recommendations > 3:
            print(f"\n⚠️  WARNING: {total_recommendations} recommendations - improvements suggested")
        else:
            print(f"\n✅ EXCELLENT: System quality is within acceptable parameters")
            
        return 0
    else:
        print("\n❌ Analysis failed - check database connection and configuration")
        return 1


if __name__ == "__main__":
    sys.exit(main())
