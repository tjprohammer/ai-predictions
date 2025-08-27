#!/usr/bin/env python3
"""
COMPREHENSIVE FEATURE QUALITY ASSESSMENT - LAST 60 DAYS
Real data validation for all features with detailed statistics
"""

import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class FeatureQualityValidator:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'mlb',
            'user': 'mlbuser',
            'password': 'mlbpass'
        }
        
    def connect_db(self):
        return psycopg2.connect(**self.db_config)
    
    def load_last_60_days_data(self):
        """Load ALL data from the last 60 days for comprehensive analysis"""
        
        # Calculate 60 days ago
        sixty_days_ago = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        
        print(f"🔍 LOADING ALL DATA FROM LAST 60 DAYS")
        print(f"📅 Date range: {sixty_days_ago} to present")
        print("=" * 70)
        
        conn = self.connect_db()
        
        # Get ALL columns from enhanced_games for the last 60 days
        query = f"""
        SELECT *
        FROM enhanced_games
        WHERE date >= '{sixty_days_ago}'
        ORDER BY date DESC;
        """
        
        df = pd.read_sql(query, conn)
        conn.close()
        
        print(f"📊 Loaded {len(df):,} games")
        print(f"📋 Total columns: {len(df.columns)}")
        print(f"📅 Actual date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def analyze_feature_quality(self, df):
        """Comprehensive analysis of every feature's quality"""
        
        print(f"\n🔬 COMPREHENSIVE FEATURE QUALITY ANALYSIS")
        print("=" * 70)
        
        # Convert date for analysis
        df['date'] = pd.to_datetime(df['date'])
        
        feature_analysis = {}
        
        # Analyze each column
        for col in df.columns:
            if col in ['date', 'home_team', 'away_team']:  # Skip metadata
                continue
                
            analysis = self.analyze_single_feature(df, col)
            feature_analysis[col] = analysis
        
        return feature_analysis
    
    def analyze_single_feature(self, df, column):
        """Detailed analysis of a single feature"""
        
        data = df[column]
        
        # Basic stats
        total_records = len(data)
        null_count = data.isna().sum()
        null_pct = (null_count / total_records) * 100
        non_null_count = total_records - null_count
        
        analysis = {
            'total_records': total_records,
            'null_count': null_count,
            'null_percentage': null_pct,
            'non_null_count': non_null_count,
            'data_type': str(data.dtype)
        }
        
        if non_null_count > 0:
            # For numeric data (but not boolean)
            if pd.api.types.is_numeric_dtype(data) and not pd.api.types.is_bool_dtype(data):
                clean_data = data.dropna()
                try:
                    analysis.update({
                        'min_value': float(clean_data.min()),
                        'max_value': float(clean_data.max()),
                        'mean': float(clean_data.mean()),
                        'median': float(clean_data.median()),
                        'std': float(clean_data.std()) if len(clean_data) > 1 else 0,
                        'q25': float(clean_data.quantile(0.25)),
                        'q75': float(clean_data.quantile(0.75)),
                        'zeros_count': int((clean_data == 0).sum()),
                        'negative_count': int((clean_data < 0).sum()),
                        'unique_values': int(clean_data.nunique())
                    })
                except (TypeError, ValueError) as e:
                    # Handle edge cases with non-standard numeric data
                    analysis.update({
                        'min_value': 'ERROR',
                        'max_value': 'ERROR',
                        'mean': 'ERROR',
                        'unique_values': int(clean_data.nunique()),
                        'error': str(e)
                    })
                
                # Quality assessment for specific feature types
                if any(pattern in column.lower() for pattern in ['_l7', '_l14', '_l20', '_l30']):
                    if 'error' not in analysis:
                        analysis['quality_assessment'] = self.assess_rolling_stats_quality(clean_data, column)
                    else:
                        analysis['quality_assessment'] = 'ERROR_IN_DATA'
                elif 'era' in column.lower():
                    if 'error' not in analysis:
                        analysis['quality_assessment'] = self.assess_era_quality(clean_data)
                    else:
                        analysis['quality_assessment'] = 'ERROR_IN_DATA'
                elif any(pattern in column.lower() for pattern in ['avg', 'ops', 'obp', 'woba']):
                    if 'error' not in analysis:
                        analysis['quality_assessment'] = self.assess_batting_stats_quality(clean_data, column)
                    else:
                        analysis['quality_assessment'] = 'ERROR_IN_DATA'
                else:
                    analysis['quality_assessment'] = 'NUMERIC_DATA'
            
            elif pd.api.types.is_bool_dtype(data):
                # Handle boolean data
                clean_data = data.dropna()
                analysis.update({
                    'true_count': int(clean_data.sum()),
                    'false_count': int((~clean_data).sum()),
                    'unique_values': int(clean_data.nunique()),
                    'quality_assessment': 'BOOLEAN_DATA'
                })
            
            else:
                # For non-numeric data
                analysis.update({
                    'unique_values': int(data.nunique()),
                    'most_common': str(data.mode().iloc[0]) if len(data.mode()) > 0 else 'N/A',
                    'quality_assessment': 'NON_NUMERIC'
                })
        
        # Time-based analysis
        analysis['recent_quality'] = self.analyze_recent_data_quality(df, column)
        
        return analysis
    
    def assess_rolling_stats_quality(self, data, column):
        """Assess quality of rolling statistics"""
        
        if '_l7' in column:
            # 7-game rolling should be roughly 14-70 runs
            realistic_min, realistic_max = 7, 80
        elif '_l14' in column:
            # 14-game rolling should be roughly 28-140 runs
            realistic_min, realistic_max = 20, 160
        elif '_l20' in column:
            # 20-game rolling should be roughly 40-200 runs
            realistic_min, realistic_max = 30, 220
        elif '_l30' in column:
            # 30-game rolling should be roughly 60-300 runs
            realistic_min, realistic_max = 45, 330
        else:
            return 'UNKNOWN_ROLLING_PERIOD'
        
        realistic_count = ((data >= realistic_min) & (data <= realistic_max)).sum()
        realistic_pct = (realistic_count / len(data)) * 100
        
        if realistic_pct >= 95:
            return f'EXCELLENT_{realistic_pct:.1f}%_REALISTIC'
        elif realistic_pct >= 85:
            return f'GOOD_{realistic_pct:.1f}%_REALISTIC'
        elif realistic_pct >= 70:
            return f'FAIR_{realistic_pct:.1f}%_REALISTIC'
        else:
            return f'POOR_{realistic_pct:.1f}%_REALISTIC'
    
    def assess_era_quality(self, data):
        """Assess ERA quality (typical range 2.00-7.00)"""
        realistic_count = ((data >= 1.5) & (data <= 8.0)).sum()
        realistic_pct = (realistic_count / len(data)) * 100
        
        if realistic_pct >= 95:
            return f'EXCELLENT_ERA_{realistic_pct:.1f}%'
        elif realistic_pct >= 85:
            return f'GOOD_ERA_{realistic_pct:.1f}%'
        else:
            return f'QUESTIONABLE_ERA_{realistic_pct:.1f}%'
    
    def assess_batting_stats_quality(self, data, column):
        """Assess batting statistics quality"""
        if 'avg' in column.lower():
            # Batting average typically 0.150-0.350
            realistic_count = ((data >= 0.100) & (data <= 0.400)).sum()
        elif 'ops' in column.lower():
            # OPS typically 0.500-1.200
            realistic_count = ((data >= 0.400) & (data <= 1.400)).sum()
        elif 'obp' in column.lower():
            # OBP typically 0.250-0.450
            realistic_count = ((data >= 0.200) & (data <= 0.500)).sum()
        elif 'woba' in column.lower():
            # wOBA typically 0.250-0.450
            realistic_count = ((data >= 0.200) & (data <= 0.500)).sum()
        else:
            return 'UNKNOWN_BATTING_STAT'
        
        realistic_pct = (realistic_count / len(data)) * 100
        
        if realistic_pct >= 95:
            return f'EXCELLENT_{realistic_pct:.1f}%'
        else:
            return f'CHECK_{realistic_pct:.1f}%'
    
    def analyze_recent_data_quality(self, df, column):
        """Analyze data quality trends over the last 60 days"""
        
        # Split into weekly periods
        df_sorted = df.sort_values('date')
        
        recent_weeks = {}
        for week in range(0, 9):  # Last 8+ weeks
            start_date = datetime.now() - timedelta(days=(week+1)*7)
            end_date = datetime.now() - timedelta(days=week*7)
            
            week_data = df_sorted[
                (df_sorted['date'] >= start_date) & 
                (df_sorted['date'] < end_date)
            ][column]
            
            if len(week_data) > 0:
                null_pct = (week_data.isna().sum() / len(week_data)) * 100
                recent_weeks[f'week_{week}_ago'] = {
                    'games': len(week_data),
                    'null_pct': null_pct,
                    'date_range': f"{start_date.strftime('%m-%d')} to {end_date.strftime('%m-%d')}"
                }
        
        return recent_weeks
    
    def generate_feature_categories(self, feature_analysis):
        """Categorize features by type and quality"""
        
        categories = {
            'rolling_stats': [],
            'era_stats': [],
            'batting_stats': [],
            'core_stats': [],
            'other_stats': [],
            'problematic': []
        }
        
        for feature, analysis in feature_analysis.items():
            # Categorize by feature type
            if any(pattern in feature.lower() for pattern in ['_l7', '_l14', '_l20', '_l30']):
                categories['rolling_stats'].append((feature, analysis))
            elif 'era' in feature.lower():
                categories['era_stats'].append((feature, analysis))
            elif any(pattern in feature.lower() for pattern in ['avg', 'ops', 'obp', 'woba', 'hits', 'rbi']):
                categories['batting_stats'].append((feature, analysis))
            elif feature in ['total_runs', 'home_score', 'away_score', 'market_total']:
                categories['core_stats'].append((feature, analysis))
            else:
                categories['other_stats'].append((feature, analysis))
            
            # Flag problematic features
            if analysis['null_percentage'] > 20:
                categories['problematic'].append((feature, analysis))
        
        return categories
    
    def print_comprehensive_report(self, feature_analysis, categories):
        """Print detailed feature quality report"""
        
        print(f"\n📊 FEATURE QUALITY REPORT - LAST 60 DAYS")
        print("=" * 70)
        
        total_features = len(feature_analysis)
        print(f"🔢 Total Features Analyzed: {total_features}")
        
        # Summary by category
        print(f"\n📋 FEATURE CATEGORIES:")
        for category, features in categories.items():
            if category != 'problematic':
                print(f"   {category.upper()}: {len(features)} features")
        
        # Rolling Stats Analysis (PRIORITY)
        print(f"\n🏃 ROLLING STATISTICS (RECENTLY FIXED!)")
        print("-" * 50)
        if categories['rolling_stats']:
            for feature, analysis in categories['rolling_stats']:
                quality = analysis.get('quality_assessment', 'UNKNOWN')
                null_pct = analysis['null_percentage']
                mean_val = analysis.get('mean', 0)
                print(f"   {feature:<35} | {quality:<20} | {null_pct:5.1f}% null | avg: {mean_val:6.1f}")
        
        # ERA Stats Analysis
        print(f"\n⚾ ERA STATISTICS")
        print("-" * 30)
        if categories['era_stats']:
            for feature, analysis in categories['era_stats']:
                quality = analysis.get('quality_assessment', 'UNKNOWN')
                null_pct = analysis['null_percentage']
                mean_val = analysis.get('mean', 0)
                print(f"   {feature:<35} | {quality:<15} | {null_pct:5.1f}% null | avg: {mean_val:5.2f}")
        
        # Batting Stats Analysis
        print(f"\n🏏 BATTING STATISTICS")
        print("-" * 35)
        if categories['batting_stats']:
            for feature, analysis in categories['batting_stats']:
                quality = analysis.get('quality_assessment', 'UNKNOWN')
                null_pct = analysis['null_percentage']
                mean_val = analysis.get('mean', 0)
                print(f"   {feature:<35} | {quality:<15} | {null_pct:5.1f}% null | avg: {mean_val:5.3f}")
        
        # Core Stats
        print(f"\n🎯 CORE STATISTICS")
        print("-" * 25)
        if categories['core_stats']:
            for feature, analysis in categories['core_stats']:
                null_pct = analysis['null_percentage']
                mean_val = analysis.get('mean', 0)
                print(f"   {feature:<20} | {null_pct:5.1f}% null | avg: {mean_val:6.1f}")
        
        # Problematic Features
        print(f"\n❌ PROBLEMATIC FEATURES (>20% NULL)")
        print("-" * 40)
        if categories['problematic']:
            for feature, analysis in categories['problematic']:
                null_pct = analysis['null_percentage']
                print(f"   {feature:<35} | {null_pct:5.1f}% null")
        else:
            print("   ✅ No significantly problematic features!")
        
        # Best Features for Modeling
        print(f"\n🏆 RECOMMENDED FEATURES FOR MODELING")
        print("-" * 45)
        reliable_features = []
        for feature, analysis in feature_analysis.items():
            if (analysis['null_percentage'] < 10 and 
                analysis['non_null_count'] > 50 and
                'EXCELLENT' in analysis.get('quality_assessment', '') or
                'GOOD' in analysis.get('quality_assessment', '')):
                reliable_features.append((feature, analysis))
        
        reliable_features.sort(key=lambda x: x[1]['null_percentage'])
        
        for i, (feature, analysis) in enumerate(reliable_features[:20]):  # Top 20
            quality = analysis.get('quality_assessment', 'RELIABLE')
            null_pct = analysis['null_percentage']
            print(f"   {i+1:2d}. {feature:<35} | {quality:<20} | {null_pct:4.1f}% null")
        
        return reliable_features

def main():
    print("🔬 COMPREHENSIVE FEATURE QUALITY ASSESSMENT")
    print("   Real data validation for all features - Last 60 days")
    print("=" * 70)
    
    validator = FeatureQualityValidator()
    
    # Load last 60 days of data
    df = validator.load_last_60_days_data()
    
    # Analyze all features
    feature_analysis = validator.analyze_feature_quality(df)
    
    # Categorize features
    categories = validator.generate_feature_categories(feature_analysis)
    
    # Print comprehensive report
    reliable_features = validator.print_comprehensive_report(feature_analysis, categories)
    
    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save full analysis as JSON
    results_file = f'S:/Projects/AI_Predictions/feature_quality_analysis_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(feature_analysis, f, indent=2, default=str)
    
    # Save reliable features list
    reliable_file = f'S:/Projects/AI_Predictions/reliable_features_{timestamp}.json'
    reliable_data = {
        'analysis_date': timestamp,
        'total_features_analyzed': len(feature_analysis),
        'reliable_features_count': len(reliable_features),
        'reliable_features': [
            {
                'name': feature,
                'quality': analysis.get('quality_assessment', 'RELIABLE'),
                'null_percentage': analysis['null_percentage'],
                'mean': analysis.get('mean', None)
            }
            for feature, analysis in reliable_features
        ]
    }
    
    with open(reliable_file, 'w') as f:
        json.dump(reliable_data, f, indent=2, default=str)
    
    print(f"\n💾 RESULTS SAVED:")
    print(f"   Full analysis: {results_file}")
    print(f"   Reliable features: {reliable_file}")
    print(f"\n🎯 SUMMARY:")
    print(f"   Total features: {len(feature_analysis)}")
    print(f"   Reliable features: {len(reliable_features)}")
    print(f"   Rolling stats quality: {'FIXED!' if any('EXCELLENT' in str(analysis.get('quality_assessment', '')) for analysis in feature_analysis.values()) else 'CHECK RESULTS'}")

if __name__ == "__main__":
    main()
