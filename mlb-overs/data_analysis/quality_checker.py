#!/usr/bin/env python3
"""
BRUTAL HONEST DATA QUALITY ASSESSMENT
No sugarcoating - find REAL problems preventing quality predictions
"""

import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class BrutalDataQualityChecker:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'mlb',
            'user': 'mlbuser',
            'password': 'mlbpass'
        }
        
    def connect_db(self):
        return psycopg2.connect(**self.db_config)
    
    def check_prediction_target_quality(self):
        """Check if we can even predict properly with our target variable"""
        
        print("🎯 PREDICTION TARGET (total_runs) QUALITY CHECK")
        print("=" * 60)
        
        conn = self.connect_db()
        
        # Check target variable completeness over time
        query = """
        SELECT 
            DATE_TRUNC('week', date) as week,
            COUNT(*) as total_games,
            SUM(CASE WHEN total_runs IS NOT NULL THEN 1 ELSE 0 END) as games_with_runs,
            SUM(CASE WHEN home_score IS NOT NULL AND away_score IS NOT NULL THEN 1 ELSE 0 END) as games_with_scores,
            AVG(total_runs) as avg_runs,
            MIN(total_runs) as min_runs,
            MAX(total_runs) as max_runs
        FROM enhanced_games
        WHERE date >= '2025-07-01'
        GROUP BY DATE_TRUNC('week', date)
        ORDER BY week DESC
        LIMIT 8;
        """
        
        df = pd.read_sql(query, conn)
        
        print("📊 WEEKLY TARGET COMPLETENESS:")
        for _, row in df.iterrows():
            completeness = (row['games_with_runs'] / row['total_games']) * 100
            scores_completeness = (row['games_with_scores'] / row['total_games']) * 100
            avg_runs = row['avg_runs'] if pd.notna(row['avg_runs']) else 0
            
            print(f"   {row['week'].strftime('%Y-%m-%d')}: {completeness:5.1f}% complete | {scores_completeness:5.1f}% scores | avg: {avg_runs:5.1f} runs")
            
            if completeness < 80:
                print(f"   ❌ PROBLEM: Week has <80% complete target data")
        
        conn.close()
    
    def check_feature_consistency(self):
        """Check if features are consistent and make sense"""
        
        print(f"\n🔍 FEATURE CONSISTENCY CHECK")
        print("=" * 40)
        
        conn = self.connect_db()
        
        # Check for logical consistency
        query = """
        SELECT 
            date,
            home_team,
            away_team,
            home_score,
            away_score,
            total_runs,
            home_score + away_score as calculated_total,
            ABS((home_score + away_score) - total_runs) as total_difference,
            
            home_team_runs_l7,
            away_team_runs_l7,
            home_team_avg,
            away_team_avg,
            home_sp_season_era,
            away_sp_season_era
        FROM enhanced_games
        WHERE date >= '2025-08-15'
          AND home_score IS NOT NULL 
          AND away_score IS NOT NULL
          AND total_runs IS NOT NULL
        ORDER BY date DESC
        LIMIT 20;
        """
        
        df = pd.read_sql(query, conn)
        
        print("🧮 LOGICAL CONSISTENCY CHECK:")
        print("   (home_score + away_score should = total_runs)")
        
        bad_totals = 0
        for _, row in df.iterrows():
            diff = row['total_difference'] if pd.notna(row['total_difference']) else 999
            if diff > 0.1:  # Allow for small floating point errors
                bad_totals += 1
                print(f"   ❌ {row['date']} {row['home_team']} vs {row['away_team']}: {row['home_score']}+{row['away_score']}={row['calculated_total']} but total_runs={row['total_runs']}")
        
        if bad_totals == 0:
            print("   ✅ All totals are mathematically correct")
        else:
            print(f"   ❌ {bad_totals} games have incorrect total_runs calculations")
        
        # Check for impossible values
        print(f"\n🚨 IMPOSSIBLE VALUES CHECK:")
        
        impossible_count = 0
        for _, row in df.iterrows():
            issues = []
            
            # Batting averages should be 0.150-0.400
            if pd.notna(row['home_team_avg']) and (row['home_team_avg'] < 0.100 or row['home_team_avg'] > 0.450):
                issues.append(f"Home BA: {row['home_team_avg']:.3f}")
            if pd.notna(row['away_team_avg']) and (row['away_team_avg'] < 0.100 or row['away_team_avg'] > 0.450):
                issues.append(f"Away BA: {row['away_team_avg']:.3f}")
            
            # ERAs should be 1.00-8.00
            if pd.notna(row['home_sp_season_era']) and (row['home_sp_season_era'] < 0.50 or row['home_sp_season_era'] > 10.0):
                issues.append(f"Home ERA: {row['home_sp_season_era']:.2f}")
            if pd.notna(row['away_sp_season_era']) and (row['away_sp_season_era'] < 0.50 or row['away_sp_season_era'] > 10.0):
                issues.append(f"Away ERA: {row['away_sp_season_era']:.2f}")
            
            # L7 runs should be 14-70
            if pd.notna(row['home_team_runs_l7']) and (row['home_team_runs_l7'] < 10 or row['home_team_runs_l7'] > 80):
                issues.append(f"Home L7: {row['home_team_runs_l7']:.1f}")
            if pd.notna(row['away_team_runs_l7']) and (row['away_team_runs_l7'] < 10 or row['away_team_runs_l7'] > 80):
                issues.append(f"Away L7: {row['away_team_runs_l7']:.1f}")
            
            if issues:
                impossible_count += 1
                print(f"   ❌ {row['date']} {row['home_team']} vs {row['away_team']}: {', '.join(issues)}")
        
        if impossible_count == 0:
            print("   ✅ No impossible values found in recent games")
        
        conn.close()
        return bad_totals, impossible_count
    
    def check_data_freshness(self):
        """Check if data is being updated regularly"""
        
        print(f"\n📅 DATA FRESHNESS CHECK")
        print("=" * 30)
        
        conn = self.connect_db()
        
        # Check when data was last updated
        query = """
        SELECT 
            MAX(date) as latest_date,
            COUNT(*) as total_games,
            COUNT(CASE WHEN date >= CURRENT_DATE - INTERVAL '7 days' THEN 1 END) as games_last_7_days,
            COUNT(CASE WHEN date >= CURRENT_DATE - INTERVAL '3 days' THEN 1 END) as games_last_3_days
        FROM enhanced_games;
        """
        
        freshness_df = pd.read_sql(query, conn)
        
        latest_date = freshness_df['latest_date'].iloc[0]
        days_behind = (datetime.now().date() - latest_date).days
        
        print(f"   Latest game date: {latest_date}")
        print(f"   Days behind: {days_behind}")
        print(f"   Games last 7 days: {freshness_df['games_last_7_days'].iloc[0]}")
        print(f"   Games last 3 days: {freshness_df['games_last_3_days'].iloc[0]}")
        
        if days_behind > 2:
            print(f"   ❌ PROBLEM: Data is {days_behind} days behind - predictions will be stale")
        elif days_behind > 1:
            print(f"   ⚠️  WARNING: Data is {days_behind} day behind")
        else:
            print(f"   ✅ Data is current")
        
        conn.close()
        return days_behind
    
    def check_feature_coverage(self):
        """Check what percentage of games have each important feature"""
        
        print(f"\n📊 FEATURE COVERAGE CHECK")
        print("=" * 35)
        
        conn = self.connect_db()
        
        # Get feature coverage for last 30 days
        query = """
        SELECT 
            COUNT(*) as total_games,
            
            -- Core prediction features
            COUNT(home_score) as has_home_score,
            COUNT(away_score) as has_away_score,
            COUNT(total_runs) as has_total_runs,
            COUNT(market_total) as has_market_total,
            
            -- Rolling stats (should be fixed)
            COUNT(home_team_runs_l7) as has_home_l7,
            COUNT(away_team_runs_l7) as has_away_l7,
            
            -- Batting stats
            COUNT(home_team_avg) as has_home_avg,
            COUNT(away_team_avg) as has_away_avg,
            COUNT(home_team_ops) as has_home_ops,
            COUNT(away_team_ops) as has_away_ops,
            
            -- Pitching stats
            COUNT(home_sp_season_era) as has_home_era,
            COUNT(away_sp_season_era) as has_away_era,
            COUNT(home_sp_era_l3starts) as has_home_l3era,
            COUNT(away_sp_era_l3starts) as has_away_l3era
            
        FROM enhanced_games
        WHERE date >= CURRENT_DATE - INTERVAL '30 days';
        """
        
        coverage_df = pd.read_sql(query, conn)
        total_games = coverage_df['total_games'].iloc[0]
        
        print(f"   Last 30 days: {total_games} games")
        print(f"\n   CORE FEATURES:")
        
        critical_features = [
            ('home_score', 'has_home_score'),
            ('away_score', 'has_away_score'), 
            ('total_runs', 'has_total_runs'),
            ('market_total', 'has_market_total')
        ]
        
        critical_problems = 0
        for feature_name, col_name in critical_features:
            count = coverage_df[col_name].iloc[0]
            coverage = (count / total_games) * 100
            print(f"     {feature_name:<15}: {coverage:5.1f}% ({count}/{total_games})")
            if coverage < 90 and feature_name in ['home_score', 'away_score', 'total_runs']:
                critical_problems += 1
        
        print(f"\n   ROLLING STATS (RECENTLY FIXED):")
        rolling_features = [
            ('home_l7_runs', 'has_home_l7'),
            ('away_l7_runs', 'has_away_l7')
        ]
        
        for feature_name, col_name in rolling_features:
            count = coverage_df[col_name].iloc[0]
            coverage = (count / total_games) * 100
            print(f"     {feature_name:<15}: {coverage:5.1f}% ({count}/{total_games})")
        
        print(f"\n   BATTING STATS:")
        batting_features = [
            ('home_avg', 'has_home_avg'),
            ('away_avg', 'has_away_avg'),
            ('home_ops', 'has_home_ops'),
            ('away_ops', 'has_away_ops')
        ]
        
        for feature_name, col_name in batting_features:
            count = coverage_df[col_name].iloc[0]
            coverage = (count / total_games) * 100
            print(f"     {feature_name:<15}: {coverage:5.1f}% ({count}/{total_games})")
        
        print(f"\n   PITCHING STATS:")
        pitching_features = [
            ('home_season_era', 'has_home_era'),
            ('away_season_era', 'has_away_era'),
            ('home_l3_era', 'has_home_l3era'),
            ('away_l3_era', 'has_away_l3era')
        ]
        
        for feature_name, col_name in pitching_features:
            count = coverage_df[col_name].iloc[0]
            coverage = (count / total_games) * 100
            print(f"     {feature_name:<15}: {coverage:5.1f}% ({count}/{total_games})")
        
        conn.close()
        return critical_problems
    
    def final_training_readiness_assessment(self, bad_totals, impossible_count, days_behind, critical_problems):
        """Give final verdict on training readiness"""
        
        print(f"\n🏁 FINAL TRAINING READINESS ASSESSMENT")
        print("=" * 50)
        
        total_issues = bad_totals + impossible_count + critical_problems
        if days_behind > 2:
            total_issues += 1
        
        print(f"📋 ISSUE SUMMARY:")
        print(f"   Mathematical errors: {bad_totals}")
        print(f"   Impossible values: {impossible_count}")
        print(f"   Critical feature gaps: {critical_problems}")
        print(f"   Data freshness issues: {1 if days_behind > 2 else 0}")
        print(f"   TOTAL ISSUES: {total_issues}")
        
        if total_issues == 0:
            print(f"\n🎉 VERDICT: READY FOR TRAINING!")
            print(f"   Your data quality is excellent for ML training")
            return True
        elif total_issues <= 2:
            print(f"\n⚠️  VERDICT: MARGINALLY READY")
            print(f"   Training possible but expect mediocre results")
            print(f"   Fix these issues for better performance")
            return False
        else:
            print(f"\n❌ VERDICT: NOT READY FOR TRAINING")
            print(f"   Too many data quality issues for reliable ML")
            print(f"   Fix fundamental problems before training")
            return False

def main():
    print("🔥 BRUTAL HONEST DATA QUALITY ASSESSMENT")
    print("   No sugarcoating - finding REAL problems")
    print("=" * 60)
    
    checker = BrutalDataQualityChecker()
    
    # Run all checks
    checker.check_prediction_target_quality()
    bad_totals, impossible_count = checker.check_feature_consistency()
    days_behind = checker.check_data_freshness()
    critical_problems = checker.check_feature_coverage()
    
    # Final assessment
    ready = checker.final_training_readiness_assessment(bad_totals, impossible_count, days_behind, critical_problems)
    
    if not ready:
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   1. Fix data collection pipeline for consistent updates")
        print(f"   2. Validate all mathematical calculations")
        print(f"   3. Clean impossible/outlier values")
        print(f"   4. Ensure all critical features have >90% coverage")

if __name__ == "__main__":
    main()
