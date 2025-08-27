#!/usr/bin/env python3
"""
SAMPLE TRAINING DATA PREVIEW
Show actual feature data from 5 games per month for training validation
"""

import psycopg2
import pandas as pd
import numpy as np

class TrainingDataPreviewer:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'mlb',
            'user': 'mlbuser',
            'password': 'mlbpass'
        }
        
    def connect_db(self):
        return psycopg2.connect(**self.db_config)
    
    def get_sample_games_by_month(self):
        """Get 5 sample games from each month with all training features"""
        
        print("📊 SAMPLE TRAINING DATA PREVIEW")
        print("   5 games per month with all features")
        print("=" * 60)
        
        conn = self.connect_db()
        
        # Get 5 games from each month with complete data
        query = """
        WITH monthly_samples AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY DATE_TRUNC('month', date) 
                       ORDER BY RANDOM()
                   ) as rn
            FROM enhanced_games
            WHERE date >= '2025-03-01'
              AND total_runs IS NOT NULL
              AND home_team_runs_l7 IS NOT NULL
              AND away_team_runs_l7 IS NOT NULL
              AND home_sp_season_era IS NOT NULL
              AND away_sp_season_era IS NOT NULL
              AND home_team_avg IS NOT NULL
              AND away_team_avg IS NOT NULL
        )
        SELECT 
            -- Game identification
            date,
            home_team,
            away_team,
            
            -- Target variable
            total_runs,
            home_score,
            away_score,
            
            -- Market data
            market_total,
            
            -- Rolling statistics (recently fixed!)
            home_team_runs_l7,
            away_team_runs_l7,
            home_team_runs_allowed_l7,
            away_team_runs_allowed_l7,
            home_team_runs_l20,
            away_team_runs_l20,
            home_team_runs_l30,
            away_team_runs_l30,
            
            -- Batting statistics
            home_team_avg,
            away_team_avg,
            home_team_obp,
            away_team_obp,
            home_team_ops,
            away_team_ops,
            home_team_woba,
            away_team_woba,
            
            -- Pitching statistics
            home_sp_season_era,
            away_sp_season_era,
            home_sp_era_l3starts,
            away_sp_era_l3starts,
            
            -- Additional features
            home_team_hits,
            away_team_hits,
            home_team_rbi,
            away_team_rbi,
            
            -- Environmental factors
            temperature,
            wind_speed,
            
            -- Advanced metrics
            home_bullpen_era,
            away_bullpen_era,
            home_lineup_strength,
            away_lineup_strength,
            offensive_environment_score
            
        FROM monthly_samples
        WHERE rn <= 5
        ORDER BY date, home_team;
        """
        
        sample_df = pd.read_sql(query, conn)
        conn.close()
        
        return sample_df
    
    def display_sample_data(self, sample_df):
        """Display the sample data in an organized way"""
        
        # Group by month
        sample_df['month'] = pd.to_datetime(sample_df['date']).dt.to_period('M')
        
        for month in sample_df['month'].unique():
            month_data = sample_df[sample_df['month'] == month]
            
            print(f"\n📅 {month} SAMPLE GAMES ({len(month_data)} games)")
            print("=" * 70)
            
            for i, (_, game) in enumerate(month_data.iterrows(), 1):
                self.display_single_game(game, i)
    
    def display_single_game(self, game, game_num):
        """Display detailed data for a single game"""
        
        print(f"\n🏟️  GAME {game_num}: {game['date']} | {game['home_team']} vs {game['away_team']}")
        print("-" * 60)
        
        # Game outcome
        print(f"📊 OUTCOME:")
        print(f"   Final Score: {game['home_team']} {game['home_score']:.0f} - {game['away_score']:.0f} {game['away_team']}")
        market_total_str = f"{game['market_total']:.1f}" if pd.notna(game['market_total']) else 'N/A'
        print(f"   Total Runs: {game['total_runs']:.0f} | Market Total: {market_total_str}")
        
        # Rolling statistics (our recently fixed key features!)
        print(f"\n🏃 ROLLING STATISTICS (Recently Fixed!):")
        
        home_l7_str = f"{game['home_team_runs_l7']:.0f}" if pd.notna(game['home_team_runs_l7']) else 'N/A'
        away_l7_str = f"{game['away_team_runs_l7']:.0f}" if pd.notna(game['away_team_runs_l7']) else 'N/A'
        home_l7_allowed_str = f"{game['home_team_runs_allowed_l7']:.0f}" if pd.notna(game['home_team_runs_allowed_l7']) else 'N/A'
        away_l7_allowed_str = f"{game['away_team_runs_allowed_l7']:.0f}" if pd.notna(game['away_team_runs_allowed_l7']) else 'N/A'
        home_l20_str = f"{game['home_team_runs_l20']:.0f}" if pd.notna(game['home_team_runs_l20']) else 'N/A'
        away_l20_str = f"{game['away_team_runs_l20']:.0f}" if pd.notna(game['away_team_runs_l20']) else 'N/A'
        home_l30_str = f"{game['home_team_runs_l30']:.0f}" if pd.notna(game['home_team_runs_l30']) else 'N/A'
        away_l30_str = f"{game['away_team_runs_l30']:.0f}" if pd.notna(game['away_team_runs_l30']) else 'N/A'
        
        print(f"   L7 Runs Scored:    Home: {home_l7_str:>3} | Away: {away_l7_str:>3}")
        print(f"   L7 Runs Allowed:   Home: {home_l7_allowed_str:>3} | Away: {away_l7_allowed_str:>3}")
        print(f"   L20 Runs Scored:   Home: {home_l20_str:>3} | Away: {away_l20_str:>3}")
        print(f"   L30 Runs Scored:   Home: {home_l30_str:>3} | Away: {away_l30_str:>3}")
        
        # Batting statistics
        print(f"\n🏏 BATTING STATISTICS:")
        
        home_avg_str = f"{game['home_team_avg']:.3f}" if pd.notna(game['home_team_avg']) else 'N/A'
        away_avg_str = f"{game['away_team_avg']:.3f}" if pd.notna(game['away_team_avg']) else 'N/A'
        home_obp_str = f"{game['home_team_obp']:.3f}" if pd.notna(game['home_team_obp']) else 'N/A'
        away_obp_str = f"{game['away_team_obp']:.3f}" if pd.notna(game['away_team_obp']) else 'N/A'
        home_ops_str = f"{game['home_team_ops']:.3f}" if pd.notna(game['home_team_ops']) else 'N/A'
        away_ops_str = f"{game['away_team_ops']:.3f}" if pd.notna(game['away_team_ops']) else 'N/A'
        home_woba_str = f"{game['home_team_woba']:.3f}" if pd.notna(game['home_team_woba']) else 'N/A'
        away_woba_str = f"{game['away_team_woba']:.3f}" if pd.notna(game['away_team_woba']) else 'N/A'
        
        print(f"   Batting Average:   Home: {home_avg_str:>5} | Away: {away_avg_str:>5}")
        print(f"   On-Base Pct:       Home: {home_obp_str:>5} | Away: {away_obp_str:>5}")
        print(f"   OPS:               Home: {home_ops_str:>5} | Away: {away_ops_str:>5}")
        print(f"   wOBA:              Home: {home_woba_str:>5} | Away: {away_woba_str:>5}")
        
        # Pitching statistics
        print(f"\n⚾ PITCHING STATISTICS:")
        
        home_era_str = f"{game['home_sp_season_era']:.2f}" if pd.notna(game['home_sp_season_era']) else 'N/A'
        away_era_str = f"{game['away_sp_season_era']:.2f}" if pd.notna(game['away_sp_season_era']) else 'N/A'
        home_l3_str = f"{game['home_sp_era_l3starts']:.2f}" if pd.notna(game['home_sp_era_l3starts']) else 'N/A'
        away_l3_str = f"{game['away_sp_era_l3starts']:.2f}" if pd.notna(game['away_sp_era_l3starts']) else 'N/A'
        home_bp_str = f"{game['home_bullpen_era']:.2f}" if pd.notna(game['home_bullpen_era']) else 'N/A'
        away_bp_str = f"{game['away_bullpen_era']:.2f}" if pd.notna(game['away_bullpen_era']) else 'N/A'
        
        print(f"   Season ERA:        Home: {home_era_str:>5} | Away: {away_era_str:>5}")
        print(f"   L3 Starts ERA:     Home: {home_l3_str:>5} | Away: {away_l3_str:>5}")
        print(f"   Bullpen ERA:       Home: {home_bp_str:>5} | Away: {away_bp_str:>5}")
        
        # Game context
        print(f"\n🌤️  GAME CONDITIONS:")
        temp_str = f"{game['temperature']:.0f}" if pd.notna(game['temperature']) else 'N/A'
        wind_str = f"{game['wind_speed']:.1f}" if pd.notna(game['wind_speed']) else 'N/A'
        print(f"   Temperature: {temp_str}°F")
        print(f"   Wind Speed: {wind_str} mph")
        
        # Advanced metrics
        print(f"\n📈 ADVANCED METRICS:")
        home_lineup_str = f"{game['home_lineup_strength']:.3f}" if pd.notna(game['home_lineup_strength']) else 'N/A'
        away_lineup_str = f"{game['away_lineup_strength']:.3f}" if pd.notna(game['away_lineup_strength']) else 'N/A'
        off_env_str = f"{game['offensive_environment_score']:.3f}" if pd.notna(game['offensive_environment_score']) else 'N/A'
        
        print(f"   Lineup Strength:   Home: {home_lineup_str:>5} | Away: {away_lineup_str:>5}")
        print(f"   Offensive Environment: {off_env_str}")
        
        # Reality check
        self.validate_game_data(game)
    
    def validate_game_data(self, game):
        """Quick validation of game data quality"""
        
        issues = []
        
        # Check if total_runs matches home_score + away_score
        if pd.notna(game['total_runs']) and pd.notna(game['home_score']) and pd.notna(game['away_score']):
            calculated_total = game['home_score'] + game['away_score']
            if abs(game['total_runs'] - calculated_total) > 0.1:
                issues.append(f"Total runs mismatch: {game['total_runs']} vs {calculated_total}")
        
        # Check for realistic values
        if pd.notna(game['home_team_runs_l7']) and (game['home_team_runs_l7'] < 10 or game['home_team_runs_l7'] > 80):
            issues.append(f"Unrealistic L7 runs: {game['home_team_runs_l7']}")
        
        if pd.notna(game['home_sp_season_era']) and (game['home_sp_season_era'] < 1.0 or game['home_sp_season_era'] > 8.0):
            issues.append(f"Questionable ERA: {game['home_sp_season_era']}")
        
        if pd.notna(game['home_team_avg']) and (game['home_team_avg'] < 0.150 or game['home_team_avg'] > 0.350):
            issues.append(f"Unrealistic BA: {game['home_team_avg']}")
        
        if issues:
            print(f"\n   ⚠️  DATA ISSUES: {'; '.join(issues)}")
        else:
            print(f"\n   ✅ Data validation: All values look realistic")
    
    def summarize_feature_completeness(self, sample_df):
        """Summarize feature completeness across all samples"""
        
        print(f"\n📊 FEATURE COMPLETENESS SUMMARY")
        print("=" * 50)
        
        total_games = len(sample_df)
        
        key_features = [
            ('total_runs', 'Target Variable'),
            ('market_total', 'Market Total'),
            ('home_team_runs_l7', 'L7 Runs (Fixed)'),
            ('away_team_runs_l7', 'L7 Runs Away'),
            ('home_team_avg', 'Batting Average'),
            ('home_team_ops', 'OPS'),
            ('home_sp_season_era', 'Season ERA'),
            ('home_sp_era_l3starts', 'L3 ERA'),
            ('temperature', 'Temperature'),
            ('home_lineup_strength', 'Lineup Strength')
        ]
        
        print(f"Sample size: {total_games} games across 6 months")
        print()
        print("Feature                | Completeness | Avg Value")
        print("-" * 50)
        
        for feature, description in key_features:
            if feature in sample_df.columns:
                non_null_count = sample_df[feature].notna().sum()
                completeness = (non_null_count / total_games) * 100
                
                if non_null_count > 0:
                    avg_value = sample_df[feature].mean()
                    print(f"{description:<22} | {completeness:6.1f}%     | {avg_value:8.2f}")
                else:
                    print(f"{description:<22} | {completeness:6.1f}%     | N/A")
        
        print()
        
        # Overall assessment
        core_features = ['total_runs', 'home_team_runs_l7', 'away_team_runs_l7', 'home_team_avg', 'away_team_avg', 'home_sp_season_era', 'away_sp_season_era']
        
        core_completeness = []
        for feature in core_features:
            if feature in sample_df.columns:
                completeness = (sample_df[feature].notna().sum() / total_games) * 100
                core_completeness.append(completeness)
        
        avg_core_completeness = np.mean(core_completeness) if core_completeness else 0
        
        print(f"📈 TRAINING READINESS ASSESSMENT:")
        print(f"   Core features avg completeness: {avg_core_completeness:.1f}%")
        
        if avg_core_completeness >= 95:
            print(f"   ✅ EXCELLENT - Ready for high-quality training")
        elif avg_core_completeness >= 90:
            print(f"   ✅ GOOD - Ready for training")
        elif avg_core_completeness >= 80:
            print(f"   ⚠️ FAIR - Workable but could be better")
        else:
            print(f"   ❌ POOR - Need more complete data")

def main():
    print("🔍 TRAINING DATA PREVIEW - ACTUAL GAME SAMPLES")
    print("   Showing real feature data from 5 games per month")
    print("=" * 65)
    
    previewer = TrainingDataPreviewer()
    
    # Get sample games
    print("📋 Loading sample games with complete feature sets...")
    sample_df = previewer.get_sample_games_by_month()
    
    if len(sample_df) == 0:
        print("❌ No complete games found! Check data quality.")
        return
    
    print(f"✅ Found {len(sample_df)} sample games with complete data")
    
    # Display detailed game data
    previewer.display_sample_data(sample_df)
    
    # Summary
    previewer.summarize_feature_completeness(sample_df)
    
    print(f"\n🎯 This is exactly what your ML model will train on!")
    print(f"   Review these samples to ensure data quality meets your expectations")

if __name__ == "__main__":
    main()
