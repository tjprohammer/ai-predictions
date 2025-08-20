#!/usr/bin/env python3
"""
Advanced Performance Analyzer - Deep dive into prediction factors for MAE improvement
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

class AdvancedPerformanceAnalyzer:
    def __init__(self):
        # Use same database connection as daily workflow
        DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
        self.engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    
    def get_detailed_prediction_data(self, days=14):
        """Get comprehensive prediction data with all available factors"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        query = text("""
            SELECT 
                g.*,
                -- Prediction error metrics
                ABS(g.predicted_total - g.total_runs) as abs_error,
                (g.predicted_total - g.total_runs) as error,
                
                -- Game context factors
                EXTRACT(dow FROM g.date) as day_of_week,
                EXTRACT(hour FROM g.game_start_time) as game_hour,
                g.venue_name,
                g.venue_surface,
                
                -- Weather factors (if available)
                g.temperature,
                g.humidity,
                g.wind_speed,
                g.wind_direction,
                g.precipitation,
                g.cloud_cover,
                
                -- Team performance factors
                g.home_team_runs_last_5,
                g.away_team_runs_last_5,
                g.home_team_wins_last_10,
                g.away_team_wins_last_10,
                
                -- Pitching factors
                g.home_sp_era,
                g.away_sp_era,
                g.home_sp_k,
                g.away_sp_k,
                g.home_sp_bb,
                g.away_sp_bb,
                g.home_sp_ip,
                g.away_sp_ip,
                
                -- Bullpen factors
                g.home_bullpen_era,
                g.away_bullpen_era,
                g.home_bullpen_k9,
                g.away_bullpen_k9,
                g.home_bullpen_bb9,
                g.away_bullpen_bb9,
                
                -- Market factors
                g.total_line,
                g.home_ml,
                g.away_ml,
                
                -- Advanced metrics (if available)
                g.home_team_wrc_plus,
                g.away_team_wrc_plus,
                g.home_team_fip,
                g.away_team_fip
                
            FROM enhanced_games g
            WHERE g.date BETWEEN :start_date AND :end_date
              AND g.total_runs IS NOT NULL
              AND g.predicted_total IS NOT NULL
            ORDER BY g.date DESC, g.game_id
        """)
        
        return pd.read_sql(query, self.engine, params={
            'start_date': start_date,
            'end_date': end_date
        })
    
    def analyze_error_patterns(self, df):
        """Analyze patterns in prediction errors across various factors"""
        
        print("\n[ADVANCED] ERROR PATTERN ANALYSIS")
        print("=" * 50)
        
        # Overall statistics
        print(f"📊 Dataset: {len(df)} games")
        print(f"📊 MAE: {df['abs_error'].mean():.3f} runs")
        print(f"📊 Median Error: {df['abs_error'].median():.3f} runs")
        print(f"📊 Std Dev: {df['abs_error'].std():.3f} runs")
        print(f"📊 Bias: {df['error'].mean():.3f} runs")
        
        analysis_results = {
            'overall_mae': df['abs_error'].mean(),
            'bias': df['error'].mean(),
            'patterns': {}
        }
        
        # Day of week analysis
        print("\n🗓️ DAY OF WEEK ANALYSIS:")
        dow_analysis = df.groupby('day_of_week').agg({
            'abs_error': ['mean', 'count'],
            'error': 'mean'
        }).round(3)
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for i, day in enumerate(days):
            if i in dow_analysis.index:
                mae = dow_analysis.loc[i, ('abs_error', 'mean')]
                bias = dow_analysis.loc[i, ('error', 'mean')]
                count = dow_analysis.loc[i, ('abs_error', 'count')]
                print(f"  {day}: {mae:.3f} MAE, {bias:+.3f} bias ({count} games)")
        
        analysis_results['patterns']['day_of_week'] = dow_analysis.to_dict()
        
        # Game time analysis
        print("\n🕐 GAME TIME ANALYSIS:")
        if 'game_hour' in df.columns and df['game_hour'].notna().any():
            time_analysis = df.groupby('game_hour').agg({
                'abs_error': ['mean', 'count'],
                'error': 'mean'
            }).round(3)
            
            for hour in sorted(time_analysis.index):
                mae = time_analysis.loc[hour, ('abs_error', 'mean')]
                bias = time_analysis.loc[hour, ('error', 'mean')]
                count = time_analysis.loc[hour, ('abs_error', 'count')]
                time_str = f"{hour:02d}:00"
                print(f"  {time_str}: {mae:.3f} MAE, {bias:+.3f} bias ({count} games)")
            
            analysis_results['patterns']['game_time'] = time_analysis.to_dict()
        
        # Temperature analysis
        print("\n🌡️ TEMPERATURE ANALYSIS:")
        if 'temperature' in df.columns and df['temperature'].notna().any():
            # Create temperature bins
            df['temp_bin'] = pd.cut(df['temperature'], 
                                   bins=[0, 60, 70, 80, 90, 100], 
                                   labels=['Cold (<60)', 'Cool (60-70)', 'Warm (70-80)', 'Hot (80-90)', 'Very Hot (90+)'])
            
            temp_analysis = df.groupby('temp_bin').agg({
                'abs_error': ['mean', 'count'],
                'error': 'mean'
            }).round(3)
            
            for temp_range in temp_analysis.index:
                if pd.notna(temp_range):
                    mae = temp_analysis.loc[temp_range, ('abs_error', 'mean')]
                    bias = temp_analysis.loc[temp_range, ('error', 'mean')]
                    count = temp_analysis.loc[temp_range, ('abs_error', 'count')]
                    print(f"  {temp_range}: {mae:.3f} MAE, {bias:+.3f} bias ({count} games)")
            
            analysis_results['patterns']['temperature'] = temp_analysis.to_dict()
        
        # Venue analysis
        print("\n🏟️ TOP VENUES BY ERROR:")
        venue_analysis = df.groupby('venue_name').agg({
            'abs_error': ['mean', 'count'],
            'error': 'mean'
        }).round(3)
        
        # Show venues with at least 3 games
        venue_filtered = venue_analysis[venue_analysis[('abs_error', 'count')] >= 3]
        venue_sorted = venue_filtered.sort_values(('abs_error', 'mean'), ascending=False)
        
        for venue in venue_sorted.head(10).index:
            mae = venue_sorted.loc[venue, ('abs_error', 'mean')]
            bias = venue_sorted.loc[venue, ('error', 'mean')]
            count = venue_sorted.loc[venue, ('abs_error', 'count')]
            print(f"  {venue}: {mae:.3f} MAE, {bias:+.3f} bias ({count} games)")
        
        # Pitching matchup analysis
        print("\n⚾ PITCHING FACTORS:")
        if 'home_sp_era' in df.columns and df['home_sp_era'].notna().any():
            # Combined starter ERA
            df['combined_sp_era'] = (df['home_sp_era'] + df['away_sp_era']) / 2
            df['era_bin'] = pd.cut(df['combined_sp_era'], 
                                  bins=[0, 3.5, 4.5, 5.5, 10], 
                                  labels=['Elite (<3.5)', 'Good (3.5-4.5)', 'Average (4.5-5.5)', 'Poor (5.5+)'])
            
            era_analysis = df.groupby('era_bin').agg({
                'abs_error': ['mean', 'count'],
                'error': 'mean'
            }).round(3)
            
            for era_range in era_analysis.index:
                if pd.notna(era_range):
                    mae = era_analysis.loc[era_range, ('abs_error', 'mean')]
                    bias = era_analysis.loc[era_range, ('error', 'mean')]
                    count = era_analysis.loc[era_range, ('abs_error', 'count')]
                    print(f"  {era_range}: {mae:.3f} MAE, {bias:+.3f} bias ({count} games)")
        
        # Market line analysis
        print("\n💰 MARKET LINE ANALYSIS:")
        if 'total_line' in df.columns and df['total_line'].notna().any():
            df['line_bin'] = pd.cut(df['total_line'], 
                                   bins=[0, 8, 9, 10, 11, 20], 
                                   labels=['Low (<8)', 'Med-Low (8-9)', 'Medium (9-10)', 'Med-High (10-11)', 'High (11+)'])
            
            line_analysis = df.groupby('line_bin').agg({
                'abs_error': ['mean', 'count'],
                'error': 'mean'
            }).round(3)
            
            for line_range in line_analysis.index:
                if pd.notna(line_range):
                    mae = line_analysis.loc[line_range, ('abs_error', 'mean')]
                    bias = line_analysis.loc[line_range, ('error', 'mean')]
                    count = line_analysis.loc[line_range, ('abs_error', 'count')]
                    print(f"  {line_range}: {mae:.3f} MAE, {bias:+.3f} bias ({count} games)")
        
        return analysis_results
    
    def identify_improvement_opportunities(self, df):
        """Identify specific areas where the model can be improved"""
        
        print("\n🔍 IMPROVEMENT OPPORTUNITIES")
        print("=" * 50)
        
        opportunities = []
        
        # High error games analysis
        high_error_games = df[df['abs_error'] > 4.0]
        if len(high_error_games) > 0:
            print(f"\n❗ HIGH ERROR GAMES ({len(high_error_games)} games with >4 runs error):")
            
            # Analyze common factors in high-error games
            if 'venue_name' in high_error_games.columns:
                venue_counts = high_error_games['venue_name'].value_counts()
                if len(venue_counts) > 0:
                    print(f"  Most problematic venues: {', '.join(venue_counts.head(3).index.tolist())}")
                    opportunities.append(f"Venue-specific adjustments needed for {', '.join(venue_counts.head(3).index.tolist())}")
            
            if 'day_of_week' in high_error_games.columns:
                dow_counts = high_error_games['day_of_week'].value_counts()
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                if len(dow_counts) > 0:
                    problematic_days = [days[i] for i in dow_counts.head(2).index if i < len(days)]
                    print(f"  Most problematic days: {', '.join(problematic_days)}")
                    opportunities.append(f"Day-of-week adjustments needed for {', '.join(problematic_days)}")
        
        # Temperature correlation
        if 'temperature' in df.columns and df['temperature'].notna().any():
            temp_corr = df['temperature'].corr(df['error'])
            if abs(temp_corr) > 0.1:
                print(f"\n🌡️ TEMPERATURE CORRELATION: {temp_corr:.3f}")
                if temp_corr > 0.1:
                    opportunities.append("Add temperature adjustment: higher temps = more runs")
                elif temp_corr < -0.1:
                    opportunities.append("Add temperature adjustment: higher temps = fewer runs")
        
        # Wind analysis
        if 'wind_speed' in df.columns and df['wind_speed'].notna().any():
            wind_corr = df['wind_speed'].corr(df['error'])
            if abs(wind_corr) > 0.1:
                print(f"\n💨 WIND CORRELATION: {wind_corr:.3f}")
                if wind_corr > 0.1:
                    opportunities.append("Add wind adjustment: stronger winds = more runs")
                elif wind_corr < -0.1:
                    opportunities.append("Add wind adjustment: stronger winds = fewer runs")
        
        # Recent team performance
        if 'home_team_runs_last_5' in df.columns and df['home_team_runs_last_5'].notna().any():
            home_runs_corr = df['home_team_runs_last_5'].corr(df['error'])
            away_runs_corr = df['away_team_runs_last_5'].corr(df['error'])
            
            if abs(home_runs_corr) > 0.1 or abs(away_runs_corr) > 0.1:
                print(f"\n🏃 RECENT FORM CORRELATION:")
                print(f"  Home team L5 runs: {home_runs_corr:.3f}")
                print(f"  Away team L5 runs: {away_runs_corr:.3f}")
                opportunities.append("Enhance recent form weighting in model")
        
        print(f"\n💡 RECOMMENDED IMPROVEMENTS:")
        for i, opp in enumerate(opportunities, 1):
            print(f"  {i}. {opp}")
        
        return opportunities
    
    def generate_enhanced_corrections(self, df):
        """Generate more sophisticated bias corrections based on detailed analysis"""
        
        print("\n🔧 ENHANCED BIAS CORRECTIONS")
        print("=" * 50)
        
        corrections = {
            'global_adjustment': df['error'].mean(),
            'enhanced_adjustments': {},
            'new_factors': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Day of week corrections
        dow_corrections = {}
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for i in range(7):
            day_games = df[df['day_of_week'] == i]
            if len(day_games) >= 5:  # Minimum sample size
                bias = day_games['error'].mean()
                if abs(bias) > 0.3:  # Significant bias
                    dow_corrections[days[i]] = -bias
                    print(f"  {days[i]}: {-bias:+.2f} runs correction")
        
        if dow_corrections:
            corrections['enhanced_adjustments']['day_of_week'] = dow_corrections
        
        # Temperature corrections
        if 'temperature' in df.columns and df['temperature'].notna().any():
            temp_corrections = {}
            df['temp_bin'] = pd.cut(df['temperature'], 
                                   bins=[0, 60, 70, 80, 90, 100], 
                                   labels=['cold', 'cool', 'warm', 'hot', 'very_hot'])
            
            for temp_bin in df['temp_bin'].cat.categories:
                temp_games = df[df['temp_bin'] == temp_bin]
                if len(temp_games) >= 5:
                    bias = temp_games['error'].mean()
                    if abs(bias) > 0.3:
                        temp_corrections[temp_bin] = -bias
                        print(f"  {temp_bin} temp: {-bias:+.2f} runs correction")
            
            if temp_corrections:
                corrections['enhanced_adjustments']['temperature'] = temp_corrections
        
        # Venue-specific corrections (for venues with enough games)
        venue_corrections = {}
        venue_analysis = df.groupby('venue_name').agg({
            'error': ['mean', 'count']
        })
        
        for venue in venue_analysis.index:
            games_count = venue_analysis.loc[venue, ('error', 'count')]
            bias = venue_analysis.loc[venue, ('error', 'mean')]
            
            if games_count >= 3 and abs(bias) > 0.5:  # Significant venue bias
                venue_corrections[venue] = -bias
                print(f"  {venue}: {-bias:+.2f} runs correction")
        
        if venue_corrections:
            corrections['enhanced_adjustments']['venue'] = venue_corrections
        
        # Save enhanced corrections
        corrections_file = 'enhanced_model_bias_corrections.json'
        with open(corrections_file, 'w') as f:
            json.dump(corrections, f, indent=2)
        
        print(f"\n💾 Enhanced corrections saved to: {corrections_file}")
        
        return corrections

def main():
    analyzer = AdvancedPerformanceAnalyzer()
    
    print("[ADVANCED] PERFORMANCE ANALYZER")
    print("=" * 50)
    
    # Get 14 days of detailed data
    print("\n📥 Loading 14 days of prediction data...")
    df = analyzer.get_detailed_prediction_data(days=14)
    
    if len(df) == 0:
        print("❌ No prediction data found")
        return
    
    # Analyze error patterns
    analysis = analyzer.analyze_error_patterns(df)
    
    # Identify improvement opportunities
    opportunities = analyzer.identify_improvement_opportunities(df)
    
    # Generate enhanced corrections
    corrections = analyzer.generate_enhanced_corrections(df)
    
    print("\n✅ Advanced analysis complete!")
    print(f"📊 Current MAE: {analysis['overall_mae']:.3f} runs")
    print(f"🎯 Found {len(opportunities)} improvement opportunities")
    print(f"🔧 Generated {sum(len(adj) for adj in corrections['enhanced_adjustments'].values())} enhanced corrections")

if __name__ == '__main__':
    main()
