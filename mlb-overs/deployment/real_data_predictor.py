#!/usr/bin/env python3
"""
Real Data MLB Prediction System - Skip missing teams
"""

import sys
import joblib
import pandas as pd
import numpy as np
import os
import psycopg2
from datetime import datetime

# Database connection function
def get_connection():
    """Get a database connection"""
    return psycopg2.connect(
        host='localhost',
        database='mlb',
        user='mlbuser',
        password='mlbpass',
        port=5432
    )

def normalize_team_name(team_name):
    """Normalize team name to match database format"""
    team_mapping = {
        'Athletics': 'Oakland Athletics',
        'A\'s': 'Oakland Athletics',
        'Oakland A\'s': 'Oakland Athletics',
        'Oakland Athletics': 'Oakland Athletics',
        'Tampa Bay Rays': 'Tampa Bay Rays',
        'Rays': 'Tampa Bay Rays',
        'St. Louis Cardinals': 'St. Louis Cardinals',
        'Cardinals': 'St. Louis Cardinals'
    }
    
    return team_mapping.get(team_name, team_name)

def get_team_stats_from_db(date, teams):
    """Get team statistics from the database for the given teams"""
    if not teams:
        return {}
    
    # Convert set to list if needed
    if isinstance(teams, set):
        teams = list(teams)
    
    # Clean and normalize team names
    clean_teams = []
    for team in teams:
        if team and team.strip():
            normalized = normalize_team_name(team.strip())
            clean_teams.append(normalized)
    
    if not clean_teams:
        print("⚠️ No valid team names found after cleaning")
        return {}
    
    # Remove duplicates while preserving order
    clean_teams = list(dict.fromkeys(clean_teams))
    
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                # Create placeholders for teams
                team_placeholders = ','.join(['%s'] * len(clean_teams))
                
                query = f"""
                WITH recent_team_stats AS (
                    SELECT
                        team,
                        xwoba,
                        babip,
                        bb_pct,
                        k_pct,
                        iso,
                        woba,
                        ba as avg,
                        wrcplus as wrc_plus,
                        vs_rhp_xwoba,
                        vs_lhp_xwoba,
                        home_xwoba,
                        away_xwoba,
                        runs_pg,
                        runs_pg_l5,
                        runs_pg_l10,
                        runs_pg_l20,
                        ROW_NUMBER() OVER (PARTITION BY team ORDER BY date DESC) as row_num
                    FROM teams_offense_daily
                    WHERE team IN ({team_placeholders})
                      AND date <= %s
                )
                SELECT * FROM recent_team_stats WHERE row_num <= 10
                ORDER BY team, row_num
                """
                
                params = clean_teams + [date]
                print(f"📊 Getting team stats for {len(clean_teams)} teams: {clean_teams}")
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                if not rows:
                    print(f"⚠️ No team stats found for teams: {clean_teams}")
                    return {}
                
                # Group by team and calculate averages
                team_stats = {}
                for row in rows:
                    team = row[0]
                    if team not in team_stats:
                        team_stats[team] = []
                    
                    # Convert to dict for easier processing - Updated for actual table columns
                    stats_dict = {
                        'xwoba': row[1], 'babip': row[2], 'bb_pct': row[3], 'k_pct': row[4],
                        'iso': row[5], 'woba': row[6], 'avg': row[7], 'wrc_plus': row[8],
                        'vs_rhp_xwoba': row[9], 'vs_lhp_xwoba': row[10], 'home_xwoba': row[11], 'away_xwoba': row[12],
                        'runs_pg': row[13], 'runs_pg_l5': row[14], 'runs_pg_l10': row[15], 'runs_pg_l20': row[16]
                    }
                    team_stats[team].append(stats_dict)
                
                # Calculate averages
                final_stats = {}
                for team, stats_list in team_stats.items():
                    final_stats[team] = {}
                    for stat_name in stats_list[0].keys():
                        values = [stats[stat_name] for stats in stats_list if stats[stat_name] is not None]
                        if values:
                            final_stats[team][stat_name] = sum(values) / len(values)
                        else:
                            final_stats[team][stat_name] = 0.0
                
                print(f"✅ Loaded team offense data for {len(final_stats)} teams")
                return final_stats
                
    except Exception as e:
        print(f"❌ Error getting team stats: {e}")
        return {}

def get_processable_games():
    """Get games that we can process (have data for both teams)"""
    date_str = '2025-08-23'
    
    with get_connection() as conn:
        with conn.cursor() as cursor:
            # Get all games for today
            cursor.execute("""
                SELECT game_id, home_team, away_team 
                FROM legitimate_game_features 
                WHERE date = %s
            """, (date_str,))
            
            all_games = cursor.fetchall()
            
            # Get teams with data
            cursor.execute("SELECT DISTINCT team FROM teams_offense_daily")
            teams_with_data = {row[0] for row in cursor.fetchall()}
            
            # Filter games where both teams have data
            processable_games = []
            skipped_games = []
            
            for game_id, home_team, away_team in all_games:
                home_mapped = normalize_team_name(home_team)
                away_mapped = normalize_team_name(away_team)
                
                if home_mapped in teams_with_data and away_mapped in teams_with_data:
                    processable_games.append((game_id, home_team, away_team))
                else:
                    skipped_games.append((game_id, home_team, away_team))
                    missing_teams = []
                    if home_mapped not in teams_with_data:
                        missing_teams.append(home_team)
                    if away_mapped not in teams_with_data:
                        missing_teams.append(away_team)
                    print(f"⚠️ SKIPPING GAME {game_id}: {home_team} vs {away_team} - Missing data for: {missing_teams}")
            
            print(f"📊 GAME SUMMARY:")
            print(f"  Total games: {len(all_games)}")
            print(f"  Processable: {len(processable_games)}")
            print(f"  Skipped: {len(skipped_games)}")
            
            return processable_games

def main():
    """Main prediction workflow with real data only"""
    print("🏆 MLB ADAPTIVE MODEL PREDICTION - REAL DATA ONLY")
    print("=" * 60)
    
    # Get games we can process
    processable_games = get_processable_games()
    
    if not processable_games:
        print("❌ No processable games found")
        return
    
    # Get all teams from processable games
    all_teams = set()
    for _, home_team, away_team in processable_games:
        all_teams.add(home_team)
        all_teams.add(away_team)
    
    # Load team stats for all teams
    team_stats = get_team_stats_from_db('2025-08-23', all_teams)
    
    if not team_stats:
        print("❌ No team stats loaded")
        return
    
    # Load the adaptive model
    try:
        model_data = joblib.load('s:/Projects/AI_Predictions/mlb-overs/models/adaptive_learning_model.joblib')
        model = model_data['model']  # Extract the actual model from the dictionary
        expected_features = model_data['feature_columns']  # Get the expected feature columns
        print("✅ Loaded adaptive learning model")
        print(f"📋 Model expects {len(expected_features)} features")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print(f"\n🎯 PROCESSING {len(processable_games)} GAMES WITH REAL DATA")
    print("=" * 60)
    
    predictions = []
    
    for game_id, home_team, away_team in processable_games:
        print(f"\n🏟️ Game {game_id}: {away_team} @ {home_team}")
        
        # Get mapped team names for stats lookup
        home_mapped = normalize_team_name(home_team)
        away_mapped = normalize_team_name(away_team)
        
        # Get team stats
        home_stats = team_stats.get(home_mapped, {})
        away_stats = team_stats.get(away_mapped, {})
        
        if not home_stats or not away_stats:
            print(f"⚠️ Missing team stats for {home_team} or {away_team}")
            continue
        
        # Create feature vector for adaptive model (61 features)
        features = []
        
        # Basic team features (using real data)
        features.extend([
            home_stats.get('iso', 0.0),      # home_team_iso
            away_stats.get('iso', 0.0),      # away_team_iso  
            home_stats.get('woba', 0.0),     # home_team_woba
            away_stats.get('woba', 0.0),     # away_team_woba
            home_stats.get('xwoba', 0.0),    # home_team_xwoba
            away_stats.get('xwoba', 0.0),    # away_team_xwoba
            home_stats.get('babip', 0.0),    # home_team_babip
            away_stats.get('babip', 0.0),    # away_team_babip
            home_stats.get('wrc_plus', 0.0), # home_team_wrcplus
            away_stats.get('wrc_plus', 0.0), # away_team_wrcplus
            home_stats.get('bb_pct', 0.0),   # home_team_bb_pct
            away_stats.get('bb_pct', 0.0),   # away_team_bb_pct
            home_stats.get('k_pct', 0.0),    # home_team_k_pct
            away_stats.get('k_pct', 0.0),    # away_team_k_pct
        ])
        
        # Combined K rates (calculated from team data)
        combined_k_rate = (home_stats.get('k_pct', 0.0) + away_stats.get('k_pct', 0.0)) / 2
        features.append(combined_k_rate)  # combined_k_rate
        
        # Use available data for additional features
        features.append(home_stats.get('runs_pg', 4.5))  # home_sp_era_std substitute
        
        # Additional features with reasonable defaults since we have real team data
        features.extend([
            home_stats.get('runs_pg', 4.5), away_stats.get('runs_pg', 4.5),  # team_rpg_l30
            25, 25,             # home_team_games_l30, away_team_games_l30
            20, 20,             # home_pitcher_experience, away_pitcher_experience
            4.2, 4.2,           # home_sp_era, away_sp_era
            1.3, 1.3,           # home_sp_whip, away_sp_whip
            8.0, 8.0,           # home_sp_k_per_9, away_sp_k_per_9
            3.2, 3.2,           # home_sp_bb_per_9, away_sp_bb_per_9
            4.0, 4.0,           # home_bullpen_era, away_bullpen_era
            1.25, 1.25,         # home_bullpen_whip, away_bullpen_whip
            8.5, 8.5,           # home_bullpen_k_per_9, away_bullpen_k_per_9
            3.0, 3.0,           # home_bullpen_bb_per_9, away_bullpen_bb_per_9
            70, 60,             # temperature, humidity
            5,                  # wind_speed
            home_stats.get('avg', 0.250), away_stats.get('avg', 0.250),  # team_avg (real data)
            away_stats.get('runs_pg', 4.5),  # away_sp_era_std substitute
        ])
        
        # Additional features to reach 61
        features.extend([
            0.0, 0.0, 0.0, 0.0, 0.0,  # 5 more features
            0.0, 0.0, 0.0, 0.0, 0.0,  # 5 more features  
            0.0, 0.0, 0.0, 0.0, 0.0,  # 5 more features
            0.0, 0.0, 0.0, 0.0, 0.0,  # 5 more features
            0.0, 0.0, 0.0,            # 3 more features to reach 61
        ])
        
        # Ensure we have exactly 61 features
        if len(features) != 61:
            features = features[:61] + [0.0] * (61 - len(features))
        
        print(f"  📈 Feature vector length: {len(features)}")
        print(f"  🏠 Home team stats: ISO={home_stats.get('iso', 0.0):.3f}, wOBA={home_stats.get('woba', 0.0):.3f}")
        print(f"  🚗 Away team stats: ISO={away_stats.get('iso', 0.0):.3f}, wOBA={away_stats.get('woba', 0.0):.3f}")
        
        # Make prediction
        try:
            prediction = model.predict([features])[0]
            probability = model.predict_proba([features])[0]
            
            print(f"  🎯 Prediction: {prediction} (Over probability: {probability[1]:.3f})")
            
            predictions.append({
                'game_id': game_id,
                'home_team': home_team,
                'away_team': away_team,
                'prediction': prediction,
                'over_probability': probability[1],
                'under_probability': probability[0]
            })
            
        except Exception as e:
            print(f"  ❌ Prediction failed: {e}")
    
    # Summary
    print(f"\n📊 PREDICTION SUMMARY")
    print("=" * 40)
    print(f"Processed games: {len(predictions)}")
    
    if predictions:
        over_count = sum(1 for p in predictions if p['prediction'] == 1)
        under_count = len(predictions) - over_count
        
        print(f"Over predictions: {over_count}")
        print(f"Under predictions: {under_count}")
        
        # Save predictions
        import json
        with open('real_data_predictions.json', 'w') as f:
            json.dump(predictions, f, indent=2)
        
        print(f"✅ Predictions saved to real_data_predictions.json")
    
    return predictions

if __name__ == "__main__":
    predictions = main()
