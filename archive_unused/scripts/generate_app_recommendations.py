#!/usr/bin/env python3
"""
Enhanced ML Betting Recommendations for Web App
Uses the enhanced ML model for better predictions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from psycopg2.extras import RealDictCursor
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, date
from pathlib import Path
from team_utils import normalize_team_name, get_ballpark_factor, get_team_ops, estimate_pitcher_last_10_performance, get_weather_impact

# Import enhanced predictor
try:
    from enhanced_ml_predictions import EnhancedMLPredictor
    ENHANCED_MODEL_AVAILABLE = True
except ImportError:
    ENHANCED_MODEL_AVAILABLE = False
    print("⚠️  Enhanced ML model not available, falling back to basic model")

def get_db_connection():
    """Get PostgreSQL connection"""
    return psycopg2.connect(
        "postgresql://mlb:mlbpass@localhost:5432/mlb",
        cursor_factory=RealDictCursor
    )

def estimate_team_ops(team_name):
    """Estimate team OPS using enhanced team data"""
    return get_team_ops(team_name)

def create_prediction_features(game, cursor=None):
    """Create feature vector for AI prediction (maintaining 8 features for compatibility)"""
    try:
        home_pitcher = {}
        away_pitcher = {}
        
        if game['home_starter_metrics']:
            home_pitcher = json.loads(game['home_starter_metrics']) if isinstance(game['home_starter_metrics'], str) else game['home_starter_metrics']
        
        if game['away_starter_metrics']:
            away_pitcher = json.loads(game['away_starter_metrics']) if isinstance(game['away_starter_metrics'], str) else game['away_starter_metrics']
        
        # Basic pitcher stats with ballpark adjustment
        home_era_base = float(home_pitcher.get('era', 4.0))
        away_era_base = float(away_pitcher.get('era', 4.0))
        
        # Enhanced team offensive stats
        home_team_name = normalize_team_name(game.get('home_team_name', ''))
        away_team_name = normalize_team_name(game.get('away_team_name', ''))
        home_ops = estimate_team_ops(home_team_name)
        away_ops = estimate_team_ops(away_team_name)
        
        # Apply ballpark and recent form adjustments to the base stats
        venue_name = game.get('venue_name', '')
        ballpark_factor = get_ballpark_factor(venue_name)
        
        # Incorporate ballpark factor into ERA (lower ERA in pitcher-friendly parks)
        home_era = home_era_base / ballpark_factor if ballpark_factor > 1 else home_era_base * (2 - ballpark_factor)
        away_era = away_era_base / ballpark_factor if ballpark_factor > 1 else away_era_base * (2 - ballpark_factor)
        
        # Apply recent form estimates with real last 10 games data when available
        home_pitcher_id = None
        away_pitcher_id = None
        
        if game['home_starter_metrics']:
            home_pitcher = json.loads(game['home_starter_metrics']) if isinstance(game['home_starter_metrics'], str) else game['home_starter_metrics']
            home_pitcher_id = home_pitcher.get('id')
            
        if game['away_starter_metrics']:
            away_pitcher = json.loads(game['away_starter_metrics']) if isinstance(game['away_starter_metrics'], str) else game['away_starter_metrics']
            away_pitcher_id = away_pitcher.get('id')
        
        home_era_adjusted = estimate_pitcher_last_10_performance(home_era, away_ops, home_pitcher_id)
        away_era_adjusted = estimate_pitcher_last_10_performance(away_era, home_ops, away_pitcher_id)
        
        # Adjust team OPS for ballpark
        home_ops_adjusted = home_ops * ballpark_factor
        away_ops_adjusted = away_ops * ballpark_factor
        
        # Add weather impact to the prediction (fallback to no weather if not available)
        weather_impact = 0.0
        weather_data = None
        
        try:
            # Try to get weather data for this venue
            venue_id = game.get('venue_id')
            if venue_id and cursor:
                cursor.execute("""
                    SELECT temperature, humidity, wind_speed, wind_direction, 
                           weather_condition, precipitation
                    FROM game_weather 
                    WHERE venue_id = %s
                """, (venue_id,))
                weather_row = cursor.fetchone()
                
                if weather_row:
                    weather_data = {
                        'temperature': weather_row.get('temperature'),
                        'humidity': weather_row.get('humidity'),
                        'wind_speed': weather_row.get('wind_speed'),
                        'wind_direction': weather_row.get('wind_direction'),
                        'weather_condition': weather_row.get('weather_condition'),
                        'precipitation': weather_row.get('precipitation')
                    }
                    
                    weather_impact = get_weather_impact(
                        venue_name,
                        temperature=float(weather_data['temperature'] or 75),
                        wind_speed=float(weather_data['wind_speed'] or 5),
                        wind_direction=weather_data['wind_direction'] or 'N',
                        humidity=float(weather_data['humidity'] or 50)
                    )
                    
        except Exception as e:
            print(f"[DEBUG] Weather calculation fallback for {venue_name}: {e}")
            weather_impact = 0.0
        
        # Create 8-feature vector (same as original model) + apply weather impact
        feature_vector = [
            home_era_adjusted, away_era_adjusted, home_ops_adjusted, away_ops_adjusted,
            (home_era_adjusted + away_era_adjusted) / 2, 
            abs(home_era_adjusted - away_era_adjusted),
            (home_ops_adjusted + away_ops_adjusted) / 2, 
            abs(home_ops_adjusted - away_ops_adjusted)
        ]
        
        return np.array(feature_vector).reshape(1, -1), weather_impact, weather_data
    except Exception as e:
        print(f"[ERROR] Error creating features: {e}")
        return None

def generate_app_recommendations():
    """Generate betting recommendations for the app - Enhanced Version using working predictor logic"""
    
    print("🚀 [APP] Using Enhanced Predictor Logic for recommendations...")
    
    try:
        # Use the same approach as the working daily_betting_predictor.py
        import sys
        sys.path.append('..')
        from daily_betting_predictor import DailyBettingPredictor
        
        # Create the predictor (this loads the enhanced model)
        predictor = DailyBettingPredictor()
        
        # Get today's predictions
        predictions = predictor.predict_todays_games()
        
        if not predictions:
            print("❌ No predictions from enhanced predictor")
            return generate_basic_app_recommendations_fallback()
        
        # Transform to app format
        app_recommendations = {
            "generated_at": datetime.now().isoformat(),
            "model_version": "enhanced_v2.0", 
            "date": datetime.now().strftime('%Y-%m-%d'),
            "games": [],
            "best_bets": []
        }
        
        all_bets = []
        
        # Convert predictions to app format
        for pred in predictions:
            try:
                # Extract prediction data
                ai_prediction = pred.get('predicted_total', 8.5)
                market_total = pred.get('betting_line', 8.5)
                difference = ai_prediction - market_total
                
                # Determine recommendation
                recommendation = "NO_BET"
                confidence = "LOW"
                bet_type = None
                
                if abs(difference) >= 0.5:
                    if difference > 0:
                        recommendation = "OVER"
                        bet_type = f"OVER {market_total}"
                    else:
                        recommendation = "UNDER" 
                        bet_type = f"UNDER {market_total}"
                    
                    # Set confidence based on difference
                    if abs(difference) >= 2.0:
                        confidence = "HIGH"
                    elif abs(difference) >= 1.0:
                        confidence = "MEDIUM"
                    else:
                        confidence = "LOW"
                
                game_data = {
                    "game_id": pred.get('game_id', ''),
                    "matchup": pred.get('matchup', ''),
                    "away_team": pred.get('away_team', ''),
                    "home_team": pred.get('home_team', ''),
                    "away_pitcher": {
                        "name": pred.get('away_pitcher', 'TBD'),
                        "era": pred.get('away_era', 'N/A')
                    },
                    "home_pitcher": {
                        "name": pred.get('home_pitcher', 'TBD'),
                        "era": pred.get('home_era', 'N/A')
                    },
                    "ai_prediction": round(ai_prediction, 1),
                    "market_total": market_total,
                    "difference": round(difference, 1),
                    "recommendation": recommendation,
                    "bet_type": bet_type,
                    "confidence": confidence,
                    "weather": {
                        "temperature": pred.get('temperature'),
                        "condition": pred.get('weather_condition', 'Clear'),
                        "impact": pred.get('weather_impact', 0)
                    },
                    "enhanced_features": {
                        "home_pitcher_era": pred.get('home_era'),
                        "away_pitcher_era": pred.get('away_era'),
                        "weather_impact": pred.get('weather_impact', 0),
                        "confidence_level": pred.get('confidence', 0)
                    }
                }
                
                app_recommendations["games"].append(game_data)
                
                # Add to bets list if it's a recommendation
                if recommendation != "NO_BET":
                    all_bets.append({
                        "matchup": game_data["matchup"],
                        "bet_type": bet_type,
                        "ai_prediction": game_data["ai_prediction"],
                        "market_total": market_total,
                        "difference": game_data["difference"],
                        "confidence": confidence,
                        "confidence_score": abs(difference)
                    })
                    
            except Exception as e:
                print(f"❌ Error processing prediction: {e}")
                continue
        
        # Sort bets by confidence (highest difference first)
        all_bets.sort(key=lambda x: x['confidence_score'], reverse=True)
        app_recommendations["best_bets"] = all_bets[:5]  # Top 5 recommendations
        
        # Save to web app format
        output_file = "../web_app/daily_recommendations.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(app_recommendations, f, indent=2)
        
        print(f"✅ [APP] Enhanced recommendations saved to {output_file}")
        print(f"🎮 Generated {len(app_recommendations['games'])} game predictions")
        print(f"🔥 Found {len(app_recommendations['best_bets'])} strong recommendations")
        
        # Also save a backup with timestamp
        backup_file = f"../data/enhanced_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            os.makedirs(os.path.dirname(backup_file), exist_ok=True)
            with open(backup_file, 'w') as f:
                json.dump(app_recommendations, f, indent=2)
            print(f"✅ Backup saved to {backup_file}")
        except Exception as e:
            print(f"⚠️  Could not save backup: {e}")
        
        return app_recommendations
        
    except ImportError:
        print("❌ Could not import DailyBettingPredictor - falling back to basic model")
        return generate_basic_app_recommendations_fallback()
    except Exception as e:
        print(f"❌ Enhanced predictor failed: {e}")
        return generate_basic_app_recommendations_fallback()


def generate_basic_app_recommendations_fallback():
    """Generate recommendations using the basic model"""
    
    print("[APP] Generating basic recommendations...")
    
    recommendations = {
        "generated_at": datetime.now().isoformat(),
        "model_version": "basic_v1.0",
        "date": datetime.now().strftime('%Y-%m-%d'),
        "games": [],
        "best_bets": []
    }
    
    # Get today's games
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT 
                g.game_id, g.date_utc,
                ht.name as home_team_name, ht.abbrev as home_team_abbrev,
                at.name as away_team_name, at.abbrev as away_team_abbrev,
                v.name as venue_name, g.venue_id,
                s.home_starter_metrics, s.away_starter_metrics,
                g.market_total
            FROM games g
            LEFT JOIN teams ht ON g.home_team_id = ht.team_id
            LEFT JOIN teams at ON g.away_team_id = at.team_id  
            LEFT JOIN venues v ON g.venue_id = v.venue_id
            LEFT JOIN starters s ON g.game_id = s.game_id
            WHERE DATE(g.date_utc) IN (CURRENT_DATE, CURRENT_DATE + INTERVAL '1 day')
            ORDER BY g.date_utc
        """)
        
        games = cursor.fetchall()
        
    except Exception as e:
        print(f"[ERROR] Database error: {e}")
        return None
    finally:
        conn.close()
    
    # Generate recommendations
    recommendations = {
        "generated_at": datetime.now().isoformat(),
        "date": datetime.now().strftime('%Y-%m-%d'),
        "model_info": {
            "accuracy": "3.90 MAE",
            "training_games": 100,
            "confidence_threshold": 0.5
        },
        "games": [],
        "best_bets": []
    }
    
    all_bets = []
    
    for game in games:
        try:
            result = create_prediction_features(game, cursor)
            if result is None:
                continue
            
            features, weather_impact, weather_data = result
            features_scaled = scaler.transform(features)
            base_prediction = model.predict(features_scaled)[0]
            
            # Apply weather impact to the base prediction
            prediction = base_prediction + weather_impact
            
            market_total = game.get('market_total')
            if market_total:
                market_total = float(market_total)
            
            # Get pitcher info
            home_pitcher_name = "TBD"
            away_pitcher_name = "TBD" 
            home_era = "N/A"
            away_era = "N/A"
            
            if game['home_starter_metrics']:
                home_pitcher = json.loads(game['home_starter_metrics']) if isinstance(game['home_starter_metrics'], str) else game['home_starter_metrics']
                home_pitcher_name = home_pitcher.get('name', 'TBD')
                home_era = home_pitcher.get('era', 'N/A')
            
            if game['away_starter_metrics']:
                away_pitcher = json.loads(game['away_starter_metrics']) if isinstance(game['away_starter_metrics'], str) else game['away_starter_metrics']
                away_pitcher_name = away_pitcher.get('name', 'TBD')
                away_era = away_pitcher.get('era', 'N/A')
            
            # Determine recommendation
            recommendation = "NO_BET"
            confidence = "LOW"
            bet_type = None
            difference = 0
            
            if market_total:
                difference = prediction - market_total
                
                if abs(difference) >= 0.5:
                    if difference > 0:
                        recommendation = "OVER"
                        bet_type = f"OVER {market_total}"
                    else:
                        recommendation = "UNDER" 
                        bet_type = f"UNDER {market_total}"
                    
                    # Set confidence based on difference
                    if abs(difference) >= 2.0:
                        confidence = "HIGH"
                    elif abs(difference) >= 1.0:
                        confidence = "MEDIUM"
                    else:
                        confidence = "LOW"
            
            # Normalize team names for better matching
            home_team_normalized = normalize_team_name(game['home_team_name'])
            away_team_normalized = normalize_team_name(game['away_team_name'])
            
            game_data = {
                "game_id": game['game_id'],
                "matchup": f"{away_team_normalized} @ {home_team_normalized}",
                "away_team": away_team_normalized,
                "home_team": home_team_normalized,
                "away_pitcher": {
                    "name": away_pitcher_name,
                    "era": away_era
                },
                "home_pitcher": {
                    "name": home_pitcher_name,
                    "era": home_era
                },
                "ai_prediction": round(prediction, 1),
                "market_total": market_total,
                "difference": round(difference, 1),
                "recommendation": recommendation,
                "bet_type": bet_type,
                "confidence": confidence,
                "weather": {
                    "temperature": weather_data.get('temperature') if weather_data else None,
                    "condition": weather_data.get('weather_condition') if weather_data else None,
                    "wind_speed": weather_data.get('wind_speed') if weather_data else None,
                    "wind_direction": weather_data.get('wind_direction') if weather_data else None,
                    "humidity": weather_data.get('humidity') if weather_data else None,
                    "impact": round(weather_impact, 2) if weather_impact else 0
                },
                "ballpark_factor": round(get_ballpark_factor(game.get('venue_name', '')), 2)
            }
            
            recommendations["games"].append(game_data)
            
            # Add to bets list if it's a recommendation
            if recommendation != "NO_BET":
                all_bets.append({
                    "matchup": game_data["matchup"],
                    "bet_type": bet_type,
                    "ai_prediction": game_data["ai_prediction"],
                    "market_total": market_total,
                    "difference": game_data["difference"],
                    "confidence": confidence,
                    "confidence_score": abs(difference)
                })
        
        except Exception as e:
            print(f"[ERROR] Error processing game: {e}")
            continue
    
    # Sort bets by confidence (highest difference first)
    all_bets.sort(key=lambda x: x['confidence_score'], reverse=True)
    recommendations["best_bets"] = all_bets[:5]  # Top 5 recommendations
    
    # Save to JSON file for the app
    output_file = "../web_app/daily_recommendations.json"
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(recommendations, f, indent=2)
        print(f"[SUCCESS] Recommendations saved to {output_file}")
    except Exception as e:
        print(f"[ERROR] Could not save recommendations: {e}")
    
    # Also save a backup with timestamp
    backup_file = f"../data/recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        os.makedirs(os.path.dirname(backup_file), exist_ok=True)
        with open(backup_file, 'w') as f:
            json.dump(recommendations, f, indent=2)
        print(f"[SUCCESS] Backup saved to {backup_file}")
    except Exception as e:
        print(f"[WARNING] Could not save backup: {e}")
    
    print(f"[CHART] Generated {len(recommendations['games'])} game predictions")
    print(f"[CHART] Found {len(recommendations['best_bets'])} strong recommendations")
    
    return recommendations

if __name__ == "__main__":
    generate_app_recommendations()
