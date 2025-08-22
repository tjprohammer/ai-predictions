#!/usr/bin/env python3
"""
CORRECTED Data Quality Analysis - Fixed Logic for NULL Handling
Properly calculates validity percentages by excluding NULL values from calculations.
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Database and data handling
import psycopg2
import pandas as pd
import numpy as np

class CorrectedQualityAnalyzer:
    def __init__(self, connection_params: Dict[str, str], verbose: bool = False):
        """Initialize the corrected quality analyzer."""
        self.connection_params = connection_params
        self.verbose = verbose
        self.conn = None
        self.cursor = None
        
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
            
    def analyze_corrected_data_quality(self, start_date: str, end_date: str) -> Dict:
        """Analyze quality metrics with CORRECTED logic for NULL handling."""
        if self.verbose:
            print(f"\n🔍 Analyzing CORRECTED data quality metrics")
            
        # CORRECTED quality analysis - only count non-NULL values in denominator
        corrected_quality_query = """
        SELECT 
            -- ERA validity: Only calculate percentage for non-NULL values
            CASE 
                WHEN COUNT(home_sp_season_era) > 0 
                THEN (COUNT(CASE WHEN home_sp_season_era BETWEEN 0.5 AND 15.0 THEN 1 END) * 100.0 / COUNT(home_sp_season_era))
                ELSE 0 
            END as home_sp_era_validity,
            
            CASE 
                WHEN COUNT(away_sp_season_era) > 0 
                THEN (COUNT(CASE WHEN away_sp_season_era BETWEEN 0.5 AND 15.0 THEN 1 END) * 100.0 / COUNT(away_sp_season_era))
                ELSE 0 
            END as away_sp_era_validity,
            
            -- WHIP validity: Only calculate percentage for non-NULL values
            CASE 
                WHEN COUNT(home_sp_whip) > 0 
                THEN (COUNT(CASE WHEN home_sp_whip BETWEEN 0.5 AND 3.0 THEN 1 END) * 100.0 / COUNT(home_sp_whip))
                ELSE 0 
            END as home_sp_whip_validity,
            
            CASE 
                WHEN COUNT(away_sp_whip) > 0 
                THEN (COUNT(CASE WHEN away_sp_whip BETWEEN 0.5 AND 3.0 THEN 1 END) * 100.0 / COUNT(away_sp_whip))
                ELSE 0 
            END as away_sp_whip_validity,
            
            -- Coverage statistics
            COUNT(*) as total_games,
            COUNT(home_sp_season_era) as home_era_count,
            COUNT(away_sp_season_era) as away_era_count,
            COUNT(home_sp_whip) as home_whip_count,
            COUNT(away_sp_whip) as away_whip_count
            
        FROM enhanced_games
        WHERE date >= %s AND date <= %s
        AND home_score IS NOT NULL AND away_score IS NOT NULL
        """
        
        self.cursor.execute(corrected_quality_query, (start_date, end_date))
        quality_data = self.cursor.fetchone()
        
        quality_metrics = {}
        if quality_data:
            quality_metrics = {
                'home_sp_era_validity': round(float(quality_data[0]), 2),
                'away_sp_era_validity': round(float(quality_data[1]), 2),
                'home_sp_whip_validity': round(float(quality_data[2]), 2),
                'away_sp_whip_validity': round(float(quality_data[3]), 2),
                'total_games': quality_data[4],
                'home_era_coverage': round(quality_data[5] / quality_data[4] * 100, 2),
                'away_era_coverage': round(quality_data[6] / quality_data[4] * 100, 2),
                'home_whip_coverage': round(quality_data[7] / quality_data[4] * 100, 2),
                'away_whip_coverage': round(quality_data[8] / quality_data[4] * 100, 2)
            }
            
            # Calculate overall quality score (only for non-zero validity scores)
            valid_scores = [v for k, v in quality_metrics.items() if 'validity' in k and v > 0]
            quality_metrics['overall_quality_score'] = round(np.mean(valid_scores), 2) if valid_scores else 0
            
            if self.verbose:
                print(f"  📊 CORRECTED QUALITY ANALYSIS:")
                print(f"  Total completed games: {quality_metrics['total_games']}")
                print(f"  ")
                print(f"  📈 COVERAGE (% of games with data):")
                print(f"    Home ERA coverage: {quality_metrics['home_era_coverage']}%")
                print(f"    Away ERA coverage: {quality_metrics['away_era_coverage']}%")
                print(f"    Home WHIP coverage: {quality_metrics['home_whip_coverage']}%")
                print(f"    Away WHIP coverage: {quality_metrics['away_whip_coverage']}%")
                print(f"  ")
                print(f"  ✅ VALIDITY (% of non-NULL values that are valid):")
                print(f"    Home SP ERA validity: {quality_metrics['home_sp_era_validity']}%")
                print(f"    Away SP ERA validity: {quality_metrics['away_sp_era_validity']}%")
                print(f"    Home SP WHIP validity: {quality_metrics['home_sp_whip_validity']}%")
                print(f"    Away SP WHIP validity: {quality_metrics['away_sp_whip_validity']}%")
                print(f"  ")
                print(f"  🎯 Overall quality score: {quality_metrics['overall_quality_score']}%")
        
        return quality_metrics
        
    def comprehensive_data_summary(self, start_date: str, end_date: str) -> Dict:
        """Generate comprehensive data summary with corrected metrics."""
        if not self.connect_database():
            return {}
            
        print(f"🔍 CORRECTED Data Quality Analysis")
        print(f"📅 Analysis Period: {start_date} to {end_date}")
        print("=" * 60)
        
        try:
            # Get corrected quality metrics
            quality_results = self.analyze_corrected_data_quality(start_date, end_date)
            
            # Generate assessment
            print(f"\n📋 CORRECTED ASSESSMENT")
            print("=" * 60)
            
            coverage_issues = []
            validity_issues = []
            
            # Check coverage issues
            if quality_results.get('home_era_coverage', 0) < 50:
                coverage_issues.append(f"Home ERA coverage low: {quality_results['home_era_coverage']}%")
            if quality_results.get('away_era_coverage', 0) < 50:
                coverage_issues.append(f"Away ERA coverage low: {quality_results['away_era_coverage']}%")
            if quality_results.get('home_whip_coverage', 0) < 50:
                coverage_issues.append(f"Home WHIP coverage low: {quality_results['home_whip_coverage']}%")
            if quality_results.get('away_whip_coverage', 0) < 50:
                coverage_issues.append(f"Away WHIP coverage low: {quality_results['away_whip_coverage']}%")
                
            # Check validity issues (should be high for non-NULL values)
            if quality_results.get('home_sp_era_validity', 0) < 90:
                validity_issues.append(f"Home ERA validity: {quality_results['home_sp_era_validity']}%")
            if quality_results.get('away_sp_era_validity', 0) < 90:
                validity_issues.append(f"Away ERA validity: {quality_results['away_sp_era_validity']}%")
            if quality_results.get('home_sp_whip_validity', 0) < 90:
                validity_issues.append(f"Home WHIP validity: {quality_results['home_sp_whip_validity']}%")
            if quality_results.get('away_sp_whip_validity', 0) < 90:
                validity_issues.append(f"Away WHIP validity: {quality_results['away_sp_whip_validity']}%")
            
            # Print results
            if coverage_issues:
                print("⚠️  COVERAGE ISSUES (missing data):")
                for issue in coverage_issues:
                    print(f"   - {issue}")
            else:
                print("✅ COVERAGE: All metrics have adequate coverage")
                
            if validity_issues:
                print("⚠️  VALIDITY ISSUES (invalid ranges in existing data):")
                for issue in validity_issues:
                    print(f"   - {issue}")
            else:
                print("✅ VALIDITY: All existing data is within valid ranges")
                
            # Overall assessment
            overall_score = quality_results.get('overall_quality_score', 0)
            if overall_score >= 95:
                print(f"\n🏆 EXCELLENT: Overall quality score {overall_score}% - Ready for model training!")
            elif overall_score >= 85:
                print(f"\n✅ GOOD: Overall quality score {overall_score}% - Minor improvements possible")
            elif overall_score >= 70:
                print(f"\n⚠️  FAIR: Overall quality score {overall_score}% - Some improvements needed")
            else:
                print(f"\n❌ POOR: Overall quality score {overall_score}% - Significant improvements required")
                
            return quality_results
            
        except Exception as e:
            print(f"✗ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            self.disconnect_database()
            
        return {}


def main():
    parser = argparse.ArgumentParser(description='Corrected Data Quality Analysis')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Default date range: last 30 days
    if not args.start_date:
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    else:
        start_date = args.start_date
        
    if not args.end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    else:
        end_date = args.end_date
    
    # Database connection parameters
    connection_params = {
        'host': 'localhost',
        'database': 'mlb', 
        'user': 'mlbuser',
        'password': 'mlbpass'
    }
    
    # Create analyzer and run analysis
    analyzer = CorrectedQualityAnalyzer(connection_params, verbose=args.verbose)
    results = analyzer.comprehensive_data_summary(start_date, end_date)
    
    if results and results.get('overall_quality_score', 0) >= 85:
        print(f"\n🚀 READY: Data quality is sufficient for model retraining")
        return 0
    elif results:
        print(f"\n⏸️  REVIEW: Data quality issues should be addressed before model retraining")
        return 1
    else:
        print(f"\n❌ FAILED: Could not complete quality analysis")
        return 2


if __name__ == "__main__":
    sys.exit(main())
