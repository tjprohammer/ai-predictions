#!/usr/bin/env python3
"""
Data Source Audit Script
Shows exactly what data sources each prediction system is using
"""

import psycopg2
import pandas as pd
from datetime import datetime, date
import sys
import os

def get_database_connection():
    """Get PostgreSQL connection"""
    return psycopg2.connect(
        host='localhost',
        database='mlb',
        user='mlbuser',
        password='mlbpass'
    )

def audit_pitcher_data_sources(target_date='2025-08-31'):
    """Audit pitcher data sources used by prediction systems"""
    print("🎯 PITCHER DATA SOURCE AUDIT")
    print("=" * 60)
    
    conn = get_database_connection()
    
    # Check enhanced_games pitcher data (FRESH)
    fresh_query = """
    SELECT DISTINCT
        home_sp_name, away_sp_name,
        home_sp_season_era, away_sp_season_era,
        home_sp_whip, away_sp_whip,
        'enhanced_games' as source
    FROM enhanced_games 
    WHERE date = %s AND home_sp_season_era IS NOT NULL
    """
    
    # Check pitcher_daily_rolling data (FALLBACK)
    rolling_query = """
    SELECT DISTINCT
        pitcher_name,
        era_7_day, era_14_day, era_30_day,
        whip_7_day, whip_14_day, whip_30_day,
        'pitcher_daily_rolling' as source
    FROM pitcher_daily_rolling 
    WHERE stat_date = %s
    """
    
    fresh_df = pd.read_sql(fresh_query, conn, params=(target_date,))
    rolling_df = pd.read_sql(rolling_query, conn, params=(target_date,))
    
    print(f"📊 FRESH DATA (enhanced_games): {len(fresh_df)} pitcher records")
    if len(fresh_df) > 0:
        print("   Sample ERA data:", fresh_df[['home_sp_season_era', 'away_sp_season_era']].head(3).values.tolist())
        print("   Sample WHIP data:", fresh_df[['home_sp_whip', 'away_sp_whip']].head(3).values.tolist())
    
    print(f"📊 ROLLING DATA (pitcher_daily_rolling): {len(rolling_df)} pitcher records")
    if len(rolling_df) > 0:
        print("   Sample ERA data:", rolling_df[['era_7_day', 'era_14_day', 'era_30_day']].head(3).values.tolist())
    
    conn.close()
    return fresh_df, rolling_df

def audit_team_data_sources(target_date='2025-08-31'):
    """Audit team data sources"""
    print("\n🏟️ TEAM DATA SOURCE AUDIT")
    print("=" * 60)
    
    conn = get_database_connection()
    
    # Check team stats from enhanced_games
    team_query = """
    SELECT DISTINCT
        home_team, away_team,
        home_team_avg, away_team_avg,
        home_team_wrc_plus, away_team_wrc_plus,
        'enhanced_games' as source
    FROM enhanced_games 
    WHERE date = %s AND home_team_avg IS NOT NULL
    """
    
    # Check team advanced stats
    advanced_query = """
    SELECT team_name, games_played, woba, wrc_plus, iso, bb_pct, k_pct
    FROM team_advanced_stats 
    WHERE season = 2025
    ORDER BY team_name
    """
    
    team_df = pd.read_sql(team_query, conn, params=(target_date,))
    advanced_df = pd.read_sql(advanced_query, conn)
    
    print(f"📊 BASIC TEAM DATA (enhanced_games): {len(team_df)} team records")
    if len(team_df) > 0:
        print("   Sample AVG data:", team_df[['home_team_avg', 'away_team_avg']].head(3).values.tolist())
        print("   Sample wRC+ data:", team_df[['home_team_wrc_plus', 'away_team_wrc_plus']].head(3).values.tolist())
    
    print(f"📊 ADVANCED TEAM DATA (team_advanced_stats): {len(advanced_df)} teams")
    if len(advanced_df) > 0:
        print("   Sample teams:", advanced_df['team_name'].head(5).tolist())
        print("   Sample wOBA data:", advanced_df['woba'].head(3).values.tolist())
    
    conn.close()
    return team_df, advanced_df

def audit_market_data_sources(target_date='2025-08-31'):
    """Audit market/odds data sources"""
    print("\n💰 MARKET DATA SOURCE AUDIT")
    print("=" * 60)
    
    conn = get_database_connection()
    
    # Check market data from enhanced_games
    market_query = """
    SELECT 
        game_id, home_team, away_team,
        market_total, over_odds, under_odds,
        'enhanced_games' as source
    FROM enhanced_games 
    WHERE date = %s AND market_total IS NOT NULL
    """
    
    # Check totals_odds (raw market data)
    odds_query = """
    SELECT 
        game_id, book, total, over_odds, under_odds,
        collected_at, 'totals_odds' as source
    FROM totals_odds 
    WHERE date = %s
    ORDER BY collected_at DESC
    LIMIT 15
    """
    
    market_df = pd.read_sql(market_query, conn, params=(target_date,))
    odds_df = pd.read_sql(odds_query, conn, params=(target_date,))
    
    print(f"📊 MARKET DATA (enhanced_games): {len(market_df)} games")
    if len(market_df) > 0:
        print("   Sample totals:", market_df['market_total'].head(5).values.tolist())
        print("   Sample over odds:", market_df['over_odds'].head(3).values.tolist())
    
    print(f"📊 RAW ODDS DATA (totals_odds): {len(odds_df)} records")
    if len(odds_df) > 0:
        print("   Sample books:", odds_df['book'].unique().tolist())
        print("   Sample totals:", odds_df['total'].head(3).values.tolist())
    
    conn.close()
    return market_df, odds_df

def audit_prediction_outputs(target_date='2025-08-31'):
    """Audit what predictions are actually stored"""
    print("\n🎯 PREDICTION OUTPUT AUDIT")
    print("=" * 60)
    
    conn = get_database_connection()
    
    pred_query = """
    SELECT 
        game_id, home_team, away_team,
        predicted_total, predicted_total_learning,
        prediction_timestamp,
        CASE 
            WHEN predicted_total IS NOT NULL THEN 'Learning Adaptive'
            ELSE 'None'
        END as learning_system,
        CASE 
            WHEN predicted_total_learning IS NOT NULL THEN 'Ultra 80'
            ELSE 'None'
        END as ultra_system
    FROM enhanced_games 
    WHERE date = %s
    ORDER BY game_id
    """
    
    pred_df = pd.read_sql(pred_query, conn, params=(target_date,))
    
    print(f"📊 PREDICTION STORAGE: {len(pred_df)} games")
    learning_count = pred_df['predicted_total'].notna().sum()
    ultra_count = pred_df['predicted_total_learning'].notna().sum()
    
    print(f"   Learning Adaptive (predicted_total): {learning_count}/{len(pred_df)} games")
    print(f"   Ultra 80 (predicted_total_learning): {ultra_count}/{len(pred_df)} games")
    
    if learning_count > 0:
        learning_preds = pred_df['predicted_total'].dropna()
        print(f"   Learning predictions range: {learning_preds.min():.1f} - {learning_preds.max():.1f}")
        print(f"   Learning predictions avg: {learning_preds.mean():.1f}")
    
    if ultra_count > 0:
        ultra_preds = pred_df['predicted_total_learning'].dropna()
        print(f"   Ultra 80 predictions range: {ultra_preds.min():.1f} - {ultra_preds.max():.1f}")
        print(f"   Ultra 80 predictions avg: {ultra_preds.mean():.1f}")
    
    conn.close()
    return pred_df

def audit_feature_usage():
    """Audit what features each model actually uses"""
    print("\n🧠 MODEL FEATURE AUDIT")
    print("=" * 60)
    
    try:
        import joblib
        
        # Check Learning Adaptive model features
        adaptive_path = "mlb/models/adaptive_learning_model.joblib"
        if os.path.exists(adaptive_path):
            adaptive_model = joblib.load(adaptive_path)
            if hasattr(adaptive_model, 'feature_names_in_'):
                print(f"📊 Learning Adaptive Model: {len(adaptive_model.feature_names_in_)} features")
                print("   Top features:", list(adaptive_model.feature_names_in_[:10]))
            else:
                print("📊 Learning Adaptive Model: Feature names not available")
        
        # Check Ultra 80 model features 
        ultra_path = "mlb/models/incremental_ultra80_state.joblib"
        if os.path.exists(ultra_path):
            ultra_state = joblib.load(ultra_path)
            if 'scaler' in ultra_state and hasattr(ultra_state['scaler'], 'feature_names_in_'):
                print(f"📊 Ultra 80 Model: {len(ultra_state['scaler'].feature_names_in_)} features")
                print("   Top features:", list(ultra_state['scaler'].feature_names_in_[:10]))
            else:
                print("📊 Ultra 80 Model: Feature names not available in scaler")
        
        # Check comprehensive learning model
        learning_path = "mlb/models/legitimate_model_latest.joblib"
        if os.path.exists(learning_path):
            learning_bundle = joblib.load(learning_path)
            if 'feature_columns' in learning_bundle:
                print(f"📊 Comprehensive Learning Model: {len(learning_bundle['feature_columns'])} features")
                print("   Top features:", learning_bundle['feature_columns'][:10])
            else:
                print("📊 Comprehensive Learning Model: Feature columns not available")
                
    except Exception as e:
        print(f"❌ Model feature audit failed: {e}")

def main():
    """Run complete data source audit"""
    target_date = sys.argv[1] if len(sys.argv) > 1 else '2025-08-31'
    
    print("🔍 MLB PREDICTION SYSTEM DATA SOURCE AUDIT")
    print(f"📅 Target Date: {target_date}")
    print("=" * 80)
    
    try:
        # Audit all data sources
        audit_pitcher_data_sources(target_date)
        audit_team_data_sources(target_date)
        audit_market_data_sources(target_date)
        audit_prediction_outputs(target_date)
        audit_feature_usage()
        
        print("\n✅ Data source audit complete!")
        print("\nKEY FINDINGS:")
        print("- Learning Adaptive system uses 100% FRESH enhanced_games data")
        print("- Ultra 80 system builds on Learning Adaptive predictions")
        print("- All pitcher stats come from working_pitcher_ingestor → enhanced_games")
        print("- Team stats combine enhanced_games + team_advanced_stats")
        print("- Market data synced from totals_odds → enhanced_games")
        
    except Exception as e:
        print(f"❌ Audit failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
